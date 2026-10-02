"""Model factory: dispatch on ``cfg.model_type`` to build the right classifier.

Usage::

    from aerial_openvla.models.factory import create_model

    processor, model = create_model(cfg, device=device)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from aerial_openvla.core.config import TrainConfig


def create_model(cfg: TrainConfig, device: torch.device) -> tuple[Any, nn.Module]:
    """Build a (processor, model) pair for the configured model_type.

    Parameters
    ----------
    cfg : TrainConfig
        Full training configuration.  ``cfg.model_type`` selects the variant.
    device : torch.device
        Target CUDA device.

    Returns
    -------
    processor : Any
        Tokenizer / image processor (may be ``None`` for text-only models).
    model : nn.Module
        The action classifier, moved to ``device``.
    """
    model_type = getattr(cfg, "model_type", "openvla")

    if model_type == "openvla":
        return _create_openvla(cfg, device)

    if model_type == "cma":
        return _create_cma(cfg, device)

    if model_type == "airminivla":
        return _create_airminivla(cfg, device)

    if model_type == "qwen35_native":
        # The legacy Native1T Stage1/Stage2 lineage remains an explicitly
        # authorized experiment.  NativeMEM is a separate model family; do
        # not silently redirect a matched Native1T checkpoint/cache to it.
        return _create_qwen35_native(cfg, device)

    if model_type == "qwen35_nativemem":
        return _create_qwen35_nativemem(cfg, device)

    raise ValueError(
        f"Unknown model_type={model_type!r}. "
        f"Supported types: openvla, cma, airminivla, qwen35_native, qwen35_nativemem. "
        f"Add a new branch in {__file__}."
    )


# ---------------------------------------------------------------------------
# Per-model builders
# ---------------------------------------------------------------------------

def _create_qwen35_nativemem(cfg: TrainConfig, device: torch.device):
    from aerial_openvla.models.qwen35_nativemem_v2_classifier import (
        CONTRACT_VERSION,
        SINGLE_STAGE_V2_CONTRACT_VERSION,
        Qwen35NativeMEMV2BuildConfig,
        build_qwen35_nativemem_v2_classifier,
    )
    if cfg.nativemem_stage not in (1, 2):
        raise ValueError("NativeMEM stage must be 1 or 2")
    if cfg.nativemem_history_length != 16 or cfg.nativemem_stage1_frames != 9:
        raise ValueError("NativeMEM requires H16 and nine Stage1 frames")
    single_stage = bool(cfg.nativemem_single_stage)
    if single_stage and cfg.nativemem_stage != 2:
        raise ValueError("NativeMEM single-stage V2 requires strict causal H16 (stage=2)")
    if single_stage and cfg.init_checkpoint:
        raise ValueError("NativeMEM single-stage V2 starts from official Qwen, not a Stage1 checkpoint")
    from aerial_openvla.data.qwen35_relative_state import (
        QWEN35_RELATIVE_NONE,
        load_qwen35_relative_state_cache,
        validate_qwen35_relative_state_mode,
    )
    relative_state_mode = validate_qwen35_relative_state_mode(
        getattr(cfg, "qwen35_relative_state_mode", QWEN35_RELATIVE_NONE)
    )
    cache = cfg.nativemem_premerger_train_cache
    if not cache:
        raise ValueError("NativeMEM requires nativemem_premerger_train_cache")
    cache_meta = json.loads((Path(cache) / "cache_meta.json").read_text())
    if not cache_meta.get("complete") or cache_meta.get("shape", [None])[1:] != [196, 768]:
        raise ValueError("NativeMEM requires a complete [N,196,768] pre-merger cache")
    relative_state_metadata = None
    if relative_state_mode != QWEN35_RELATIVE_NONE:
        relative_state_cache = getattr(cfg, "qwen35_relative_state_cache", None)
        if not relative_state_cache:
            raise ValueError(
                "Qwen3.5 append-planar Relative requires qwen35_relative_state_cache"
            )
        _, relative_state_metadata = load_qwen35_relative_state_cache(
            relative_state_cache,
            cfg.train_manifest,
            int(json.loads((Path(cache) / "index_meta.json").read_text())["manifest_rows"]),
        )
    memory_cache = getattr(cfg, "nativemem_1t_train_cache", None)
    memory_meta = None
    if cfg.nativemem_stage == 2 and not single_stage:
        if not memory_cache:
            raise ValueError(
                "NativeMEM Stage2 requires nativemem_1t_train_cache"
            )
        memory_meta = json.loads(
            (Path(memory_cache) / "metadata.json").read_text()
        )
        if (
            not memory_meta.get("complete")
            or memory_meta.get("cache_type") != "qwen35_nativemem_v2_1t"
            or memory_meta.get("shape", [None])[1:] != [768]
        ):
            raise ValueError(
                "NativeMEM Stage2 requires a complete checkpoint-matched "
                "[N,768] 1T cache"
            )
        if memory_meta.get("manifest_sha256") != cache_meta.get("manifest_sha256"):
            raise ValueError("NativeMEM pre-merger/1T cache manifest mismatch")
    contract = {
        "version": (
            SINGLE_STAGE_V2_CONTRACT_VERSION if single_stage else CONTRACT_VERSION
        ),
        "model_type": "qwen35_nativemem",
        "stage": cfg.nativemem_stage,
        "single_stage": single_stage,
        "input_resolution": [224, 224],
        "premerger_shape": [196, 768],
        "current_tokens": [49, 1024],
        "history_tokens": [16, 1024],
        "history_order": "oldest_to_newest_left_padded",
        "history_excludes_current": True,
        "no_memory_projector": single_stage,
        "memory_projector": (
            None if single_stage else "LayerNorm768-Linear2048-GELU-Linear1024"
        ),
        "native1t_tokenizer": (
            "LayerNorm1024-single-query-8head-MHA-49to1-1024"
            if single_stage else "LayerNorm768-single-query-12head-MHA-768"
        ),
        "no_temporal_embedding": True,
        "sequence_order": "BOS,current49,instruction,im_end,MEM_BOS,history,ACT",
        "native_image_placeholders": True,
        "preserve_mm_token_type_ids": True,
        "train_action_query": bool(getattr(cfg, "nativemem_train_action_query", False)),
        "training_mode": (
            "single_stage_v2" if single_stage else
            ("stage1" if cfg.nativemem_stage == 1 else "stage2")
        ),
        "cache": str(Path(cache).resolve()),
        "cache_manifest_sha256": cache_meta.get("manifest_sha256"),
        "memory_cache": (
            str(Path(memory_cache).resolve()) if memory_cache else None
        ),
        "memory_cache_source_checkpoint": (
            memory_meta.get("source_checkpoint") if memory_meta else None
        ),
        "relative_state": {
            "mode": relative_state_mode,
            "prompt_placement": (
                "append_after_instruction"
                if relative_state_mode != QWEN35_RELATIVE_NONE else None
            ),
            "state_timing_train": (
                "gt_actions_before_current_action"
                if relative_state_mode != QWEN35_RELATIVE_NONE else None
            ),
            "state_timing_rollout": (
                "predicted_actions_before_current_action"
                if relative_state_mode != QWEN35_RELATIVE_NONE else None
            ),
            "cache": (
                str(Path(cfg.qwen35_relative_state_cache).resolve())
                if relative_state_mode != QWEN35_RELATIVE_NONE else None
            ),
            "cache_manifest_sha256": (
                relative_state_metadata["manifest_sha256"]
                if relative_state_metadata else None
            ),
        },
    }
    processor, model = build_qwen35_nativemem_v2_classifier(Qwen35NativeMEMV2BuildConfig(
        model_name_or_path=cfg.model_name_or_path,
        stage=cfg.nativemem_stage,
        history_length=cfg.nativemem_history_length,
        stage1_frames=cfg.nativemem_stage1_frames,
        num_actions=cfg.num_actions,
        use_lora=cfg.use_lora,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        use_flash_attention_2=cfg.use_flash_attention_2,
        bf16=cfg.bf16,
        learning_rate_tokenizer=cfg.learning_rate_nativemem,
        learning_rate_memory_projector=cfg.learning_rate_memory_projector,
        train_action_query=bool(contract.get("train_action_query", False)),
        single_stage=single_stage,
        load_vision_backbone=cfg.nativemem_load_vision_backbone,
        contract=contract,
    ), device=device)
    return processor, model

def _create_qwen35_native(cfg: TrainConfig, device: torch.device) -> tuple[Any, nn.Module]:
    from aerial_openvla.models.qwen35_native_classifier import (
        Qwen35NativeBuildConfig,
        build_qwen35_native_classifier,
    )

    native1t_contract = _qwen35_native1t_contract(cfg)
    build_cfg = Qwen35NativeBuildConfig(
        model_name_or_path=cfg.model_name_or_path,
        num_actions=cfg.num_actions,
        use_lora=cfg.use_lora,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        freeze_vision=cfg.freeze_vision,
        train_projector=cfg.train_projector,
        gradient_checkpointing=cfg.gradient_checkpointing,
        use_flash_attention_2=cfg.use_flash_attention_2,
        bf16=cfg.bf16,
        use_learned_action_query=cfg.use_learned_action_query,
        use_native1t_postfusion=cfg.use_native1t_postfusion,
        native1t_stage=cfg.native1t_stage,
        native1t_history_length=cfg.native1t_history_length,
        native1t_stage1_memory_frames=cfg.native1t_stage1_memory_frames,
        native1t_history_mode=getattr(cfg, "native1t_history_mode", "fifo"),
        native1t_event_tau_local=getattr(
            cfg, "native1t_event_tau_local", 0.757488
        ),
        native1t_event_tau_anchor=getattr(
            cfg, "native1t_event_tau_anchor", 0.765
        ),
        native1t_event_max_length=getattr(
            cfg, "native1t_event_max_length", 8
        ),
        native1t_event_max_nodes=getattr(
            cfg, "native1t_event_max_nodes", 16
        ),
        native1t_event_recent_protected=getattr(
            cfg, "native1t_event_recent_protected", 4
        ),
        native1t_memory_tokenizer_lr=cfg.learning_rate_memory_tokenizer,
        native1t_memory_projector_lr=cfg.learning_rate_memory_projector,
        task_state_mode=getattr(cfg, "task_state_mode", "none"),
        task_state_max_segments=getattr(cfg, "task_state_max_segments", 10),
        task_state_attention_dim=getattr(cfg, "task_state_attention_dim", 256),
        task_state_dropout=getattr(cfg, "task_state_dropout", 0.0),
        task_state_position_key=getattr(cfg, "task_state_position_key", False),
        task_state_pooling_mode=getattr(cfg, "task_state_pooling_mode", "mean"),
        task_state_use_gru=getattr(cfg, "task_state_use_gru", False),
        task_gru_hidden_dim=getattr(cfg, "task_gru_hidden_dim", 256),
        task_gru_burn_in=getattr(cfg, "task_gru_burn_in", 0),
        task_gru_loss_window=getattr(cfg, "task_gru_loss_window", 0),
        task_state_hierarchical_ia=getattr(
            cfg, "task_state_hierarchical_ia", False
        ),
        task_state_sticky_belief=getattr(
            cfg, "task_state_sticky_belief", False
        ),
        task_state_detach_previous_belief=getattr(
            cfg, "task_state_detach_previous_belief", True
        ),
        qwen35_relative_state_mode=getattr(
            cfg, "qwen35_relative_state_mode", "none"
        ),
        native1t_contract=native1t_contract,
    )
    return build_qwen35_native_classifier(build_cfg, device=device)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _qwen35_native1t_contract(cfg: TrainConfig) -> dict[str, Any] | None:
    """Build and validate the checkpoint's complete Native1T protocol.

    Qwen Native1T has its own current-RGB + cached-H16 pipeline.  Legacy
    AirMiniVLA frame/action defaults must never leak into its configuration.
    """
    if not getattr(cfg, "use_native1t_postfusion", False):
        return None

    stage = int(cfg.native1t_stage)
    if stage not in (1, 2):
        raise ValueError("Qwen3.5-Native1T stage must be 1 or 2")
    if stage == 2:
        expected = {
            "num_frames": 1,
            "use_prev_action": False,
            "frame_history_stride": 1,
        }
        mismatches = {
            key: (getattr(cfg, key), value)
            for key, value in expected.items()
            if getattr(cfg, key) != value
        }
        if mismatches:
            raise ValueError(
                "Qwen3.5-Native1T Stage2 legacy input fields disagree with "
                f"the cached-H16 protocol: {mismatches}"
            )

    contract: dict[str, Any] = {
        "version": "qwen35-native1t-v2",
        "model_type": cfg.model_type,
        "stage": stage,
        "prompt_template": cfg.prompt_template,
        "max_text_tokens": int(cfg.max_text_tokens),
        "image_size": 224,
        "use_learned_action_query": bool(cfg.use_learned_action_query),
        "use_relative_state_text": bool(cfg.use_relative_state_text),
        "use_relative_planar_state_text": bool(cfg.use_relative_planar_state_text),
        "history_length": int(cfg.native1t_history_length),
        "history_mode": str(cfg.native1t_history_mode),
        "history_order": "oldest_to_newest_left_padded",
        "history_excludes_current": True,
        "predict_before_append": True,
        "episode_reset_clears_history": True,
        "stage1_memory_frames": int(cfg.native1t_stage1_memory_frames),
        "use_native1t_action_history": bool(cfg.use_native1t_action_history),
        "train_manifest": str(Path(cfg.train_manifest).resolve()),
        "train_manifest_sha256": _sha256(Path(cfg.train_manifest).resolve()),
        "stage1_checkpoint": str(Path(
            cfg.native1t_stage1_checkpoint or getattr(cfg, "init_checkpoint", "")
        ).resolve()),
    }

    if stage == 2:
        contract["legacy_field_guard"] = {
            "num_frames": int(cfg.num_frames),
            "use_prev_action": bool(cfg.use_prev_action),
            "frame_history_stride": int(cfg.frame_history_stride),
        }
        if not cfg.native1t_train_cache:
            raise ValueError("Qwen3.5-Native1T Stage2 requires native1t_train_cache")
        cache_dir = Path(cfg.native1t_train_cache).resolve()
        metadata_path = cache_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not metadata.get("complete", False):
            raise ValueError(f"Native1T cache is incomplete: {cache_dir}")
        contract["cache"] = {
            "directory": str(cache_dir),
            "metadata_sha256": _sha256(metadata_path),
            "shape": metadata.get("shape"),
            "dtype": metadata.get("dtype"),
            "feature_space": metadata.get("feature_space"),
            "manifest_sha256": metadata.get("manifest_sha256"),
            "tokenizer_checkpoint": metadata.get("tokenizer_checkpoint"),
            "tokenizer_sha256": metadata.get("tokenizer_sha256"),
        }
        if contract["cache"]["manifest_sha256"] != contract["train_manifest_sha256"]:
            raise ValueError("Native1T cache and training manifest hashes disagree")
        cache_source = Path(str(contract["cache"]["tokenizer_checkpoint"])).resolve()
        stage1_source = Path(contract["stage1_checkpoint"]).resolve()
        if cache_source != stage1_source:
            raise ValueError(
                "Native1T cache tokenizer checkpoint does not match Stage2 init: "
                f"cache={cache_source}, stage1={stage1_source}"
            )

        if cfg.native1t_history_mode == "event_h16_v1":
            if not cfg.native1t_event_train_indices:
                raise ValueError("Event-H16 Stage2 requires event index metadata")
            from aerial_openvla.data.event_h16 import require_event_h16_cache_calibration

            calibration = require_event_h16_cache_calibration(
                cache_dir,
                tau_local=float(cfg.native1t_event_tau_local),
                tau_anchor=float(cfg.native1t_event_tau_anchor),
                max_event_length=int(cfg.native1t_event_max_length),
                max_nodes=int(cfg.native1t_event_max_nodes),
                recent_protected=int(cfg.native1t_event_recent_protected),
                verify_cache_content=False,
            )
            event_metadata = Path(cfg.native1t_event_train_indices).resolve() / "metadata.json"
            contract["cache_calibration"] = {
                "path": calibration["_path"],
                "sha256": calibration["_sha256"],
                "format": calibration["format"],
                "thresholds": calibration["thresholds"],
            }
            contract["event_history"] = {
                "directory": str(event_metadata.parent),
                "metadata_sha256": _sha256(event_metadata),
                "tau_local": float(cfg.native1t_event_tau_local),
                "tau_anchor": float(cfg.native1t_event_tau_anchor),
                "max_event_length": int(cfg.native1t_event_max_length),
                "max_nodes": int(cfg.native1t_event_max_nodes),
                "recent_protected": int(cfg.native1t_event_recent_protected),
            }
    return contract

def _create_openvla(cfg: TrainConfig, device: torch.device) -> tuple[Any, nn.Module]:
    from aerial_openvla.models.openvla_classifier import (
        ClassifierBuildConfig,
        build_classifier,
    )

    build_cfg = ClassifierBuildConfig(
        model_name_or_path=cfg.model_name_or_path,
        num_actions=cfg.num_actions,
        pooling=getattr(cfg, "pooling", "last_token"),
        use_lora=getattr(cfg, "use_lora", True),
        lora_rank=getattr(cfg, "lora_rank", 16),
        lora_alpha=getattr(cfg, "lora_alpha", 16),
        lora_dropout=getattr(cfg, "lora_dropout", 0.05),
        train_projector=getattr(cfg, "train_projector", False),
        freeze_vision=getattr(cfg, "freeze_vision", True),
        gradient_checkpointing=getattr(cfg, "gradient_checkpointing", True),
        use_flash_attention_2=getattr(cfg, "use_flash_attention_2", True),
        bf16=getattr(cfg, "bf16", True),
    )
    return build_classifier(build_cfg, device=device)


def _create_airminivla(cfg: TrainConfig, device: torch.device) -> tuple[Any, nn.Module]:
    from aerial_openvla.models.airminivla_classifier import (
        AirMiniVLAConfig,
        build_airminivla_classifier,
    )

    event_calibration_sha256 = None
    if (
        getattr(cfg, "use_native1t_postfusion", False)
        and int(getattr(cfg, "native1t_stage", 2)) == 2
        and getattr(cfg, "native1t_history_mode", "fifo") == "event_h16_v1"
    ):
        from aerial_openvla.data.event_h16 import require_event_h16_cache_calibration

        cache_dir = getattr(cfg, "native1t_train_cache", None)
        if not cache_dir:
            raise ValueError("Event-H16 Stage2 requires native1t_train_cache")
        calibration = require_event_h16_cache_calibration(
            cache_dir,
            tau_local=float(cfg.native1t_event_tau_local),
            tau_anchor=float(cfg.native1t_event_tau_anchor),
            max_event_length=int(cfg.native1t_event_max_length),
            max_nodes=int(cfg.native1t_event_max_nodes),
            recent_protected=int(cfg.native1t_event_recent_protected),
            verify_cache_content=False,
        )
        event_calibration_sha256 = calibration["_sha256"]

    airminivla_cfg = AirMiniVLAConfig(
        model_name_or_path=cfg.model_name_or_path,
        variant=getattr(cfg, "airminivla_variant", "m1"),
        num_actions=cfg.num_actions,
        num_frames=getattr(cfg, "num_frames", 4),
        history_pool_size=getattr(cfg, "history_pool_size", 2),
        history_pool_stride=getattr(cfg, "history_pool_stride", 2),
        visual_tokens_per_frame=getattr(cfg, "visual_tokens_per_frame", 256),
        use_task_aware_pooler=getattr(cfg, "use_task_aware_pooler", False),
        task_aware_pooler_tokens=getattr(cfg, "task_aware_pooler_tokens", 32),
        task_aware_pooler_heads=getattr(cfg, "task_aware_pooler_heads", 8),
        task_aware_pooler_only=getattr(cfg, "task_aware_pooler_only", False),
        use_instruction_local_pooler=getattr(
            cfg, "use_instruction_local_pooler", False
        ),
        history_conditioned_local_pooler=getattr(cfg, "history_conditioned_local_pooler", False),
        local_history_variant=getattr(cfg, "local_history_variant", "query"),
        instruction_local_pooler_tokens=getattr(
            cfg, "instruction_local_pooler_tokens", 32
        ),
        use_spatial_anchored_residual_pooler=getattr(
            cfg, "use_spatial_anchored_residual_pooler", False
        ),
        spatial_anchored_pooler_tokens=getattr(
            cfg, "spatial_anchored_pooler_tokens", 32
        ),
        spatial_anchored_pooler_heads=getattr(
            cfg, "spatial_anchored_pooler_heads", 8
        ),
        visual_history_mode=getattr(cfg, "visual_history_mode", "none"),
        history_tokens_per_frame=getattr(cfg, "history_tokens_per_frame", 8),
        use_frame_slot_embedding=getattr(cfg, "use_frame_slot_embedding", False),
        use_temporal_attention=getattr(cfg, "use_temporal_attention", False),
        temporal_attention_dim=getattr(cfg, "temporal_attention_dim", 256),
        temporal_attention_heads=getattr(cfg, "temporal_attention_heads", 4),
        temporal_attention_ffn_dim=getattr(cfg, "temporal_attention_ffn_dim", 1024),
        temporal_attention_dropout=getattr(cfg, "temporal_attention_dropout", 0.1),
        temporal_attention_initial_scale=getattr(
            cfg, "temporal_attention_initial_scale", 0.1
        ),
        use_temporal_memory=getattr(cfg, "use_temporal_memory", False),
        temporal_memory_num_history=getattr(cfg, "temporal_memory_num_history", 7),
        temporal_memory_attention_heads=getattr(
            cfg, "temporal_memory_attention_heads", 8
        ),
        use_native1t_postfusion=getattr(cfg, "use_native1t_postfusion", False),
        native1t_stage=getattr(cfg, "native1t_stage", 2),
        native1t_history_length=getattr(cfg, "native1t_history_length", 16),
        native1t_stage1_memory_frames=getattr(
            cfg, "native1t_stage1_memory_frames", 9
        ),
        native1t_history_mode=getattr(cfg, "native1t_history_mode", "fifo"),
        native1t_event_tau_local=getattr(
            cfg, "native1t_event_tau_local", 0.757488
        ),
        native1t_event_tau_anchor=getattr(
            cfg, "native1t_event_tau_anchor", 0.765
        ),
        native1t_event_max_length=getattr(
            cfg, "native1t_event_max_length", 8
        ),
        native1t_event_max_nodes=getattr(
            cfg, "native1t_event_max_nodes", 16
        ),
        native1t_event_recent_protected=getattr(
            cfg, "native1t_event_recent_protected", 4
        ),
        native1t_event_calibration_sha256=event_calibration_sha256,
        native1t_segts_mode=getattr(cfg, "native1t_segts_mode", "none"),
        use_native1t_action_history=getattr(
            cfg, "use_native1t_action_history", False
        ),
        native1t_action_history_length=getattr(
            cfg, "native1t_action_history_length", 4
        ),
        native1t_action_embedding_dim=getattr(
            cfg, "native1t_action_embedding_dim", 16
        ),
        use_h16_frozen_pool=getattr(cfg, "use_h16_frozen_pool", False),
        h16_frozen_pool_history_length=getattr(
            cfg, "h16_frozen_pool_history_length", 16
        ),
        use_omtrack=getattr(cfg, "use_omtrack", False),
        omtrack_history_length=getattr(cfg, "omtrack_history_length", 31),
        omtrack_history_tokens_per_frame=getattr(
            cfg, "omtrack_history_tokens_per_frame", 4
        ),
        omtrack_current_tokens=getattr(cfg, "omtrack_current_tokens", 64),
        omtrack_use_relative_time_embedding=getattr(
            cfg, "omtrack_use_relative_time_embedding", True
        ),
        dino_teacher_init_checkpoint=getattr(
            cfg, "dino_teacher_init_checkpoint", None
        ),
        dino_teacher_init_sha256=getattr(cfg, "dino_teacher_init_sha256", None),
        vision_train_last_n_layers=getattr(cfg, "vision_train_last_n_layers", 0),
        vision_train_siglip_last_n_layers=getattr(
            cfg, "vision_train_siglip_last_n_layers", 0
        ),
        use_prev_action=getattr(cfg, "use_prev_action", True),
        use_previous_actions_text=getattr(cfg, "use_previous_actions_text", False),
        use_relative_state_text=getattr(cfg, "use_relative_state_text", False),
        use_relative_planar_state_text=getattr(
            cfg, "use_relative_planar_state_text", False
        ),
        instruction_only_text_budget=getattr(
            cfg, "instruction_only_text_budget", False
        ),
        # Reserve the final embedding ID as the episode-start marker.  This is
        # 9 for legacy 8-way policies and 11 for native OpenFly 10-way models.
        prev_action_vocab=getattr(cfg, "prev_action_vocab", cfg.num_actions + 1),
        prev_action_dim=getattr(cfg, "prev_action_embedding_dim", 64),
        use_memory=getattr(cfg, "use_memory", False),
        memory_hidden_dim=getattr(cfg, "memory_hidden_dim", 512),
        action_head_hidden=getattr(cfg, "action_head_hidden", 512),
        openfly_action_policy=getattr(cfg, "openfly_action_policy", None),
        openfly_action_head_type=getattr(
            cfg, "openfly_action_head_type", "categorical"
        ),
        openfly_action_token_dim=getattr(cfg, "openfly_action_token_dim", 8),
        openfly_action_token_bins=getattr(cfg, "openfly_action_token_bins", 256),
        openfly_action_codec_path=getattr(cfg, "openfly_action_codec_path", None),
        openfly_action_codec_sha256=getattr(
            cfg, "openfly_action_codec_sha256", None
        ),
        action_chunk_horizon=getattr(cfg, "action_chunk_horizon", 1),
        use_motion_instruction_attention=getattr(
            cfg, "use_motion_instruction_attention", False
        ),
        motion_instruction_attention_heads=getattr(
            cfg, "motion_instruction_attention_heads", 4
        ),
        motion_instruction_attention_dropout=getattr(
            cfg, "motion_instruction_attention_dropout", 0.0
        ),
        task_state_mode=getattr(cfg, "task_state_mode", "none"),
        task_state_max_segments=getattr(cfg, "task_state_max_segments", 10),
        task_state_attention_dim=getattr(cfg, "task_state_attention_dim", 256),
        task_state_dropout=getattr(cfg, "task_state_dropout", 0.0),
        task_state_position_key=getattr(cfg, "task_state_position_key", False),
        task_state_pooling_mode=getattr(cfg, "task_state_pooling_mode", "mean"),
        task_state_use_gru=getattr(cfg, "task_state_use_gru", False),
        task_gru_hidden_dim=getattr(cfg, "task_gru_hidden_dim", 256),
        task_gru_burn_in=getattr(cfg, "task_gru_burn_in", 0),
        task_gru_loss_window=getattr(cfg, "task_gru_loss_window", 0),
        task_state_hierarchical_ia=getattr(
            cfg, "task_state_hierarchical_ia", False
        ),
        task_state_sticky_belief=getattr(cfg, "task_state_sticky_belief", False),
        task_state_belief_momentum=getattr(
            cfg, "task_state_belief_momentum", 0.8
        ),
        task_state_detach_previous_belief=getattr(
            cfg, "task_state_detach_previous_belief", True
        ),
        task_state_segment_prior_beta=getattr(
            cfg, "task_state_segment_prior_beta", 1.0
        ),
        task_state_eps=getattr(cfg, "task_state_eps", 1.0e-6),
        task_state_entropy_weight=getattr(cfg, "task_state_entropy_weight", 0.0),
        use_transition_aux_head=getattr(cfg, "use_transition_aux_head", False),
        use_stop_aux_head=getattr(cfg, "use_stop_aux_head", False),
        stop_aux_logit_weight=getattr(cfg, "stop_aux_logit_weight", 1.0),
        use_hierarchical_stop=getattr(cfg, "use_hierarchical_stop", False),
        hierarchical_stop_mode=getattr(cfg, "hierarchical_stop_mode", "navigation"),
        hierarchical_stop_threshold=getattr(cfg, "hierarchical_stop_threshold", 0.5),
        hierarchical_stop_initial_probability=getattr(
            cfg, "hierarchical_stop_initial_probability", 0.01
        ),
        stop_attention_dim=getattr(cfg, "stop_attention_dim", 256),
        stop_attention_heads=getattr(cfg, "stop_attention_heads", 4),
        stop_hidden_dim=getattr(cfg, "stop_hidden_dim", 256),
        stop_dropout=getattr(cfg, "stop_dropout", 0.1),
        use_progress_aux_head=getattr(cfg, "use_progress_aux_head", False),
        progress_mode=getattr(cfg, "progress_mode", "direct"),
        progress_hidden_dim=getattr(cfg, "progress_hidden_dim", 256),
        progress_dropout=getattr(cfg, "progress_dropout", 0.1),
        progress_attention_dim=getattr(cfg, "progress_attention_dim", 256),
        progress_attention_heads=getattr(cfg, "progress_attention_heads", 4),
        freeze_vision=getattr(cfg, "freeze_vision", True),
        train_projector=getattr(cfg, "train_projector", True),
        use_lora=getattr(cfg, "use_lora", True),
        train_language_model=getattr(cfg, "train_language_model", False),
        lora_rank=getattr(cfg, "lora_rank", 16),
        lora_alpha=getattr(cfg, "lora_alpha", 16),
        lora_dropout=getattr(cfg, "lora_dropout", 0.05),
        gradient_checkpointing=getattr(cfg, "gradient_checkpointing", True),
        use_flash_attention_2=getattr(cfg, "use_flash_attention_2", True),
        bf16=getattr(cfg, "bf16", True),
    )
    return build_airminivla_classifier(airminivla_cfg, device=device)


def _create_cma(cfg: TrainConfig, device: torch.device) -> tuple[Any, nn.Module]:
    """Build a CMA classifier wrapped for step-level classification.

    This wraps AirVLN"s existing CMAPolicy, adapting it to the
    :class:`ActionClassifier` interface so it can be trained through
    the same training loop as the OpenVLA classifier.

    .. note::
        CMA is a recurrent model.  Its collator must provide
        ``features``, ``instruction_ids``, ``prev_actions``, and
        ``masks`` in addition to ``action_id``.
    """
    raise NotImplementedError(
        "CMA classifier adapter is not yet implemented. "
        "Use `train_cma.py` with `configs/training/cma/cma_train.yaml` for now."
    )
