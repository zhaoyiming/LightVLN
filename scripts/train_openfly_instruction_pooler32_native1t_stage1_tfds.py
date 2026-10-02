#!/usr/bin/env python3
"""Run an instruction-conditioned 32T -> Native1T Stage1 on retained OpenFly TFDS.

This adapter is intentionally separate from the direct S0 and pre-merger
cache entry points. It keeps the observed8 categorical contract, loads one of
the two new instruction-conditioned 32T parent policies, and supplies nine
RGB frames from the immutable episode-level TFDS export.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from native1t_reducer_contract import (
    build_native1t_visual_contract,
    verify_native1t_source,
)


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--openfly-tfds-dir", required=True)
    parser.add_argument("--openfly-tfds-cache-episodes", type=int, default=32)
    parser.add_argument(
        "--allow-parent-epoch",
        type=int,
        default=None,
        help="Explicitly permit a parent epoch other than the selected E3 parent.",
    )
    return parser.parse_known_args(argv)


def _validate_config(cfg: Any) -> None:
    expected = {
        "model_type": "airminivla",
        "input_mode": "multiframe_image",
        "airminivla_variant": "s0",
        "training_stage": 2,
        "num_actions": 8,
        "openfly_action_policy": "openfly_observed_8",
        "openfly_action_head_type": "categorical",
        "openfly_action_token_dim": 0,
        "openfly_action_token_bins": 0,
        "num_frames": 1,
        "frame_history_stride": 1,
        "sequence_length": 1,
        "action_chunk_horizon": 1,
        "use_native1t_postfusion": True,
        "native1t_stage": 1,
        "native1t_history_length": 16,
        "native1t_stage1_memory_frames": 9,
        "native1t_history_mode": "fifo",
        "openfly_native1t_stage1_context_window": 16,
        "visual_tokens_per_frame": 32,
        "use_prev_action": False,
        "use_previous_actions_text": False,
        "use_relative_state_text": False,
        "use_relative_planar_state_text": False,
        "use_memory": False,
        "use_temporal_memory": False,
        "use_h16_frozen_pool": False,
        "use_omtrack": False,
        "use_motion_instruction_attention": False,
        "task_state_mode": "none",
        "use_transition_aux_head": False,
        "use_stop_aux_head": False,
        "use_hierarchical_stop": False,
        "use_progress_aux_head": False,
        "val_manifest": None,
        "freeze_vision": True,
        "vision_train_last_n_layers": 0,
        "vision_train_siglip_last_n_layers": 0,
        "train_projector": False,
        "use_lora": True,
        "train_language_model": False,
        "gradient_checkpointing": False,
        "use_flash_attention_2": True,
        "bf16": True,
        "epochs": 1,
        "batch_size": 32,
        "num_workers": 12,
        "gradient_accumulation_steps": 4,
        "save_every_steps": 500,
        "keep_last_checkpoints": 0,
    }
    mismatches = [
        f"{key}={getattr(cfg, key, None)!r}, expected {value!r}"
        for key, value in expected.items()
        if getattr(cfg, key, None) != value
    ]
    poolers = {
        "instruction_local": bool(getattr(cfg, "use_instruction_local_pooler", False)),
        "spatial_anchored_residual": bool(
            getattr(cfg, "use_spatial_anchored_residual_pooler", False)
        ),
    }
    if sum(poolers.values()) != 1 or bool(getattr(cfg, "use_task_aware_pooler", False)):
        mismatches.append(
            "exactly one of use_instruction_local_pooler/use_spatial_anchored_residual_pooler "
            "must be enabled and legacy use_task_aware_pooler must be false"
        )
    if poolers["instruction_local"] and int(getattr(cfg, "instruction_local_pooler_tokens", -1)) != 32:
        mismatches.append("instruction_local_pooler_tokens must be 32")
    if poolers["spatial_anchored_residual"] and (
        int(getattr(cfg, "spatial_anchored_pooler_tokens", -1)) != 32
        or int(getattr(cfg, "spatial_anchored_pooler_heads", -1)) != 8
    ):
        mismatches.append("spatial anchored pooler must be 32 tokens with 8 heads")
    if getattr(cfg, "save_at_steps", None) not in (None, []):
        mismatches.append("Stage1 save_at_steps must be empty")
    if cfg.openfly_action_codec_path or cfg.openfly_action_codec_sha256:
        mismatches.append("observed8 Stage1 must not use an action codec")
    if cfg.native1t_stage1_checkpoint or cfg.native1t_train_cache:
        mismatches.append("Stage1 cannot use a Stage1 source or a 1T feature cache")
    if getattr(cfg, "openfly_premerger_train_cache", None):
        mismatches.append(
            "TFDS Stage1 must not use openfly_premerger_train_cache; the pre-cache was deleted"
        )
    if cfg.max_train_steps is not None and int(cfg.max_train_steps) <= 0:
        mismatches.append("max_train_steps must be positive when provided")
    if mismatches:
        raise ValueError(
            "Instruction-conditioned 32T Native1T Stage1 TFDS contract failed: "
            + "; ".join(mismatches)
        )


def _validate_parent(cfg: Any, *, allow_parent_epoch: int | None) -> dict[str, Any]:
    import torch

    checkpoint = Path(str(cfg.init_checkpoint)).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"instruction-conditioned parent checkpoint is missing: {checkpoint}")
    metadata_path = checkpoint / "classifier_meta.pt"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
    expected = {
        "num_actions": 8,
        "variant": "s0",
        "num_frames": 1,
        "openfly_action_policy": "openfly_observed_8",
        "openfly_action_head_type": "categorical",
        "openfly_action_token_dim": 0,
        "openfly_action_token_bins": 0,
    }
    if bool(getattr(cfg, "use_instruction_local_pooler", False)):
        pooler = {
            "type": "instruction_local_weighted32",
            "checkpoint_file": "instruction_local_pooler.pt",
            "metadata": {
                "use_instruction_local_pooler": True,
                "use_spatial_anchored_residual_pooler": False,
                "use_task_aware_pooler": False,
                "instruction_local_pooler_tokens": 32,
            },
        }
    elif bool(getattr(cfg, "use_spatial_anchored_residual_pooler", False)):
        pooler = {
            "type": "spatial_anchored_residual32",
            "checkpoint_file": "spatial_anchored_residual_pooler.pt",
            "metadata": {
                "use_instruction_local_pooler": False,
                "use_spatial_anchored_residual_pooler": True,
                "use_task_aware_pooler": False,
                "spatial_anchored_pooler_tokens": 32,
                "spatial_anchored_pooler_heads": 8,
            },
        }
    else:
        raise ValueError("parent pooler cannot be inferred from config")
    expected.update(pooler["metadata"])
    mismatches = {}
    for key, value in expected.items():
        actual = bool(metadata.get(key, False)) if isinstance(value, bool) else metadata.get(key)
        if actual != value:
            mismatches[key] = (metadata.get(key), value)
    if mismatches:
        raise ValueError(f"parent is not the selected instruction-conditioned 32T model: {mismatches}")
    required = (
        "classifier_meta.pt",
        "action_head.pt",
        "action_query.pt",
        "fuse_proj.pt",
        "projector.pt",
        pooler["checkpoint_file"],
    )
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"instruction-conditioned parent is incomplete: {missing} under {checkpoint}"
        )
    state_path = checkpoint / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False) if state_path.is_file() else {}
    actual_epoch = int(state.get("epoch", -1))
    if actual_epoch != 3 and allow_parent_epoch != actual_epoch:
        raise ValueError(
            "formal Stage1 requires the selected instruction-conditioned 32T E3 parent, but the selected parent is "
            f"E{actual_epoch}; pass --allow-parent-epoch only for an explicitly selected alternate parent"
        )
    return {
        "path": str(checkpoint),
        "epoch": actual_epoch,
        "global_step": int(state.get("global_step", -1)),
        "benchmark_proxy": actual_epoch != 3,
        "classifier_meta": expected,
        "pooler": pooler,
    }


def _write_profile_contract(
    cfg: Any,
    *,
    tfds_dir: Path,
    tfds_cache_episodes: int,
    parent: dict[str, Any],
    pooler: dict[str, Any],
    source_evidence: dict[str, Any],
    visual_contract: dict[str, Any],
) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    output = Path(cfg.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "openfly_observed8_instruction_pooler32_native1t_stage1_tfds_v1",
        "complete": True,
        "stage": 1,
        "model_family": "legacy_airminivla_s0_instruction_conditioned_pooler32",
        "pooler": pooler,
        "parent": parent,
        "tfds_dataset_dir": str(tfds_dir),
        "train_manifest": str(Path(cfg.train_manifest).expanduser().resolve()),
        "history": "I0 + seven sampled chronological past slots + It",
        "source": source_evidence,
        "native1t_current_visual_contract": visual_contract,
        "memory_slots": 9,
        "current_visual_tokens": 32,
        "native1t_memory_tokens": 1,
        "premerger_cache": None,
        "feature_files": None,
        "training": {
            "epochs": int(cfg.epochs),
            "max_train_steps": cfg.max_train_steps,
            "batch_size_per_device": int(cfg.batch_size),
            "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps),
            "world_size": 2,
            "num_workers_per_rank": int(cfg.num_workers),
            "tfds_cache_episodes_per_worker": int(tfds_cache_episodes),
            "save_every_steps": int(cfg.save_every_steps),
            "save_at_steps": list(getattr(cfg, "save_at_steps", []) or []),
            "keep_last_checkpoints": 0,
            "checkpoint_retention": "unlimited",
        },
    }
    (output / "instruction_pooler32_native1t_tfds_contract.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    adapter_args, trainer_argv = _parse_args(sys.argv[1:])
    if adapter_args.openfly_tfds_cache_episodes <= 0:
        raise ValueError("--openfly-tfds-cache-episodes must be positive")
    tfds_dir = Path(adapter_args.openfly_tfds_dir).expanduser().resolve()
    if not tfds_dir.is_dir():
        raise NotADirectoryError(tfds_dir)

    trainer_argv = ["--config", adapter_args.config, *trainer_argv]
    from aerial_openvla.core.config import TrainConfig, load_yaml_dataclass
    from aerial_openvla.data.native1t_postfusion_dataset import Native1TCollator
    from aerial_openvla.data.pipeline import DataPipeline, _cached_processor
    from aerial_openvla.openfly.observed8_native1t import (
        OpenFlyObserved8Native1TStage1TFDSDataset,
    )
    import aerial_openvla.training.train as trainer

    cfg = load_yaml_dataclass(adapter_args.config, TrainConfig)
    _validate_config(cfg)
    source_evidence = verify_native1t_source(Path(__file__).resolve().parents[1])
    visual_contract = build_native1t_visual_contract(
        cfg, source_sha256=source_evidence["classifier_sha256"]
    )
    parent = _validate_parent(cfg, allow_parent_epoch=adapter_args.allow_parent_epoch)
    dataset_probe = OpenFlyObserved8Native1TStage1TFDSDataset(
        cfg.train_manifest,
        tfds_dir,
        seed=cfg.seed,
        history_context_window=cfg.openfly_native1t_stage1_context_window,
        tfds_cache_episodes=adapter_args.openfly_tfds_cache_episodes,
    )
    if len(dataset_probe) != 1_608_400:
        raise RuntimeError(f"unexpected TFDS Stage1 cardinality: {len(dataset_probe)}")

    def create_dataset(runtime_cfg: Any, manifest_path: str) -> Any:
        if Path(manifest_path).expanduser().resolve() != Path(
            runtime_cfg.train_manifest
        ).expanduser().resolve():
            raise ValueError("Stage1 received an unexpected manifest path")
        return dataset_probe

    def create_pipeline(runtime_cfg: Any) -> DataPipeline:
        processor = _cached_processor(runtime_cfg)
        return DataPipeline(
            dataset=create_dataset(runtime_cfg, runtime_cfg.train_manifest),
            collator=Native1TCollator(
                processor,
                runtime_cfg.prompt_template,
                runtime_cfg.max_text_tokens,
                stage=1,
                instruction_only_text_budget=False,
                include_segment_token_ids=False,
                include_instruction_token_mask=True,
                use_relative_state_text=False,
                use_relative_planar_state_text=False,
                action_chunk_horizon=1,
            ),
        )

    trainer.create_dataset = create_dataset
    trainer.create_pipeline = create_pipeline
    sys.argv = [sys.argv[0], *trainer_argv]
    _write_profile_contract(
        cfg,
        tfds_dir=tfds_dir,
        tfds_cache_episodes=adapter_args.openfly_tfds_cache_episodes,
        parent=parent,
        pooler=parent["pooler"],
        source_evidence=source_evidence,
        visual_contract=visual_contract,
    )
    trainer.main()


if __name__ == "__main__":
    main()
