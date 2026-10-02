#!/usr/bin/env python3
"""Train instruction-conditioned 32T Native1T Stage2 using TFDS RGB and H16 cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from selected_stage1_contract import require_selected_stage1

from native1t_reducer_contract import (
    build_native1t_visual_contract,
    require_visual_contract_compatible,
    validate_pooler32_config,
    verify_native1t_source,
)

from cache_openfly_pooler32_native1t_tfds_parallel import (
    COMPACT_TO_OPENFLY,
    EXPECTED_EPISODES,
    EXPECTED_FEATURE_DIM,
    EXPECTED_PARENT_STEP,
    EXPECTED_ROWS,
    EXPECTED_STAGE1_STEP,
    STAGE1_CONTRACT_FILENAME,
    STAGE1_CONTRACT_FORMAT,
    _pooler_spec,
    _canonical_tfds_episode_id,
    _json,
    _load_manifest,
    _sha256_file,
)


def _require_files(root: Path, names: tuple[str, ...], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in names:
        path = root / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"{label} file is missing or empty: {path}")
        result[name] = _sha256_file(path)
    return result


def _checkpoint_info(
    checkpoint: Path,
    label: str,
    expected_step: int,
    expected_epoch: int,
    *,
    require_native1t: bool,
    pooler: dict[str, Any],
) -> dict[str, Any]:
    required = (
        "classifier_meta.pt",
        "action_head.pt",
        "action_query.pt",
        "fuse_proj.pt",
        "projector.pt",
        "training_state.pt",
        "metrics.json",
        "openfly_observed8_action_map.json",
    ) + (pooler["checkpoint_file"],) + (("native1t_postfusion.pt",) if require_native1t else ())
    files = _require_files(checkpoint, required, label)
    metadata = torch.load(checkpoint / "classifier_meta.pt", map_location="cpu", weights_only=True)
    if not isinstance(metadata, dict):
        raise ValueError(f"{label} classifier metadata is not a dictionary")
    expected = {
        "num_actions": 8,
        "variant": "s0",
        "num_frames": 1,
        "openfly_action_policy": "openfly_observed_8",
        "openfly_action_head_type": "categorical",
        "openfly_action_token_dim": 0,
        "openfly_action_token_bins": 0,
    }
    expected.update(pooler["metadata"])
    mismatches = {}
    for key, value in expected.items():
        actual = bool(metadata.get(key, False)) if isinstance(value, bool) else metadata.get(key)
        if actual != value:
            mismatches[key] = (metadata.get(key), value)
    if mismatches:
        raise ValueError(f"{label} metadata mismatch: {mismatches}")
    action_map = _json(checkpoint / "openfly_observed8_action_map.json")
    compact = action_map.get("compact_to_openfly")
    if compact is None:
        compact = [action_map["head_class_to_openfly_action_id"][str(i)] for i in range(8)]
    if [int(value) for value in compact] != COMPACT_TO_OPENFLY:
        raise ValueError(f"{label} action map mismatch")
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    metrics = _json(checkpoint / "metrics.json")
    actual_epoch = int(state.get("epoch", metrics.get("epoch", -1)))
    actual_step = int(state.get("global_step", metrics.get("global_step", -1)))
    if actual_epoch != expected_epoch or actual_step != expected_step:
        raise ValueError(
            f"{label} state mismatch: epoch={actual_epoch}, step={actual_step}; "
            f"expected epoch={expected_epoch}, step={expected_step}"
        )
    if int(metrics.get("epoch", -1)) != expected_epoch or int(metrics.get("global_step", -1)) != expected_step:
        raise ValueError(f"{label} metrics/state mismatch")
    if metrics.get("early_stopped") is True:
        raise ValueError(f"{label} was early-stopped")
    return {
        "path": str(checkpoint),
        "files": files,
        "metadata": {key: metadata.get(key) for key in expected},
        "classifier_meta": metadata,
        "epoch": actual_epoch,
        "global_step": actual_step,
        "metrics": metrics,
    }


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
        "native1t_stage": 2,
        "native1t_history_length": 16,
        "native1t_stage1_memory_frames": 9,
        "native1t_history_mode": "fifo",
        "openfly_native1t_stage1_context_window": 16,
        "use_prev_action": False,
        "use_previous_actions_text": False,
        "use_relative_state_text": False,
        "use_relative_planar_state_text": False,
        "use_memory": False,
        "use_temporal_memory": False,
        "use_h16_frozen_pool": False,
        "use_omtrack": False,
        "use_transition_aux_head": False,
        "use_stop_aux_head": False,
        "use_hierarchical_stop": False,
        "use_progress_aux_head": False,
        "val_manifest": None,
        "freeze_vision": True,
        "vision_train_last_n_layers": 0,
        "vision_train_siglip_last_n_layers": 0,
        "train_projector": True,
        "use_lora": True,
        "train_language_model": False,
        "gradient_checkpointing": False,
        "use_flash_attention_2": True,
        "bf16": True,
        "epochs": 4,
        "batch_size": 64,
        "num_workers": 12,
        "gradient_accumulation_steps": 2,
        "save_every_steps": 0,
        "keep_last_checkpoints": 0,
        "scheduler_horizon_epochs": 4,
    }
    mismatches = {
        key: (getattr(cfg, key, None), value)
        for key, value in expected.items()
        if getattr(cfg, key, None) != value
    }
    if mismatches:
        raise ValueError(f"instruction-conditioned 32T Native1T Stage2 config mismatch: {mismatches}")
    if cfg.openfly_action_codec_path or cfg.openfly_action_codec_sha256:
        raise ValueError("observed8 categorical Stage2 must not use an action codec")
    if not cfg.init_checkpoint or not cfg.native1t_stage1_checkpoint or not cfg.native1t_train_cache:
        raise ValueError("Stage2 requires parent, Stage1 checkpoint, and 1T cache")
    _pooler_spec(cfg)
    if getattr(cfg, "save_at_steps", None) not in (None, []):
        raise ValueError("Stage2 save_at_steps must be empty")


class TFDSStage2Dataset(torch.utils.data.Dataset):
    """Causal FIFO H16 cache rows with current image read from TFDS."""

    def __init__(
        self,
        manifest_path: Path,
        cache_dir: Path,
        tfds_dir: Path,
        *,
        tfds_cache_episodes: int = 32,
        feature_dim: int = EXPECTED_FEATURE_DIM,
    ) -> None:
        from aerial_openvla.openfly.tfds_current_frame import OpenFlyCurrentFrameTFDSDataset

        self.manifest_path = manifest_path.expanduser().resolve()
        self.records, self.metadata, self.manifest_sha256, self.ordered_identity_sha256 = _load_manifest(
            self.manifest_path
        )
        self.tfds_dir = tfds_dir.expanduser().resolve()
        self.cache_dir = cache_dir.expanduser().resolve()
        self.feature_dim = int(feature_dim)
        if len(self.records) != EXPECTED_ROWS or int(self.metadata.get("episodes", -1)) != EXPECTED_EPISODES:
            raise ValueError("Stage2 manifest cardinality mismatch")
        self.episode_rows: dict[str, tuple[int, ...]] = {}
        by_episode: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            by_episode[record["episode_id"]].append(index)
        self.episode_rows = {key: tuple(value) for key, value in by_episode.items()}
        self._tfds = OpenFlyCurrentFrameTFDSDataset(
            self.tfds_dir,
            action_policy="openfly_10",
            num_frames=1,
            frame_history_stride=1,
            action_chunk_horizon=1,
            cache_episodes=int(tfds_cache_episodes),
        )
        if len(self._tfds) != len(self.records):
            raise ValueError("TFDS and manifest row counts differ")
        self.tfds_alignment = self._verify_tfds_alignment()
        self._validate_cache_contract()
        self._features: np.ndarray | None = None
        self._row_identity_hashes: np.ndarray | None = None
        self._verify_row_identity_array()
        _ = self.features

    def _verify_tfds_alignment(self) -> dict[str, Any]:
        retained = self._tfds.index_dir / "retained_episodes.jsonl"
        if not retained.is_file():
            raise FileNotFoundError(retained)
        compact = {openfly: index for index, openfly in enumerate(COMPACT_TO_OPENFLY)}
        rows = 0
        episodes = 0
        with retained.open("r", encoding="utf-8") as handle:
            for episode_index, line in enumerate(handle):
                episode = json.loads(line)
                if int(episode.get("episode_index", -1)) != episode_index or int(episode.get("row_start", -1)) != rows:
                    raise ValueError("TFDS retained index drift")
                episode_id = f"openfly-s0-{int(episode['episode_id']):08d}"
                actions = [int(value) for value in episode["action_ids"]]
                image_ids = [str(value) for value in episode["source_image_ids"]]
                if len(actions) != int(episode["num_steps"]) or len(actions) != len(image_ids):
                    raise ValueError("TFDS retained episode length drift")
                instructions: list[bytes] = []
                for position, (action, image_id) in enumerate(zip(actions, image_ids)):
                    record = self.records[rows]
                    actual = {
                        "episode_id": record["episode_id"],
                        "episode_position": record["episode_position"],
                        "step_id": record["step_id"],
                        "image_id": record["image_id"],
                        "openfly_action_id": record["openfly_action_id"],
                        "action_id": record["action_id"],
                    }
                    expected = {
                        "episode_id": episode_id,
                        "episode_position": position,
                        "step_id": position,
                        "image_id": image_id,
                        "openfly_action_id": action,
                        "action_id": compact[action],
                    }
                    if actual != expected:
                        raise ValueError(f"TFDS/manifest drift at row {rows}: {actual} != {expected}")
                    instructions.append(record["instruction"].encode("utf-8"))
                    rows += 1
                if hashlib.sha256(b"\0".join(instructions)).hexdigest() != episode.get("instruction_sha256"):
                    raise ValueError(f"TFDS instruction drift at episode {episode_index}")
                episodes += 1
        if rows != EXPECTED_ROWS or episodes != EXPECTED_EPISODES:
            raise ValueError(f"TFDS alignment cardinality mismatch: {rows}/{episodes}")
        return {
            "status": "TFDS_FULL_ROW_ALIGNMENT_PASS",
            "rows": rows,
            "episodes": episodes,
            "retained_episodes": str(retained),
            "retained_episodes_sha256": _sha256_file(retained),
        }

    def _validate_cache_contract(self) -> None:
        metadata = _json(self.cache_dir / "metadata.json")
        validation = _json(self.cache_dir / "validation.json")
        provenance = _json(self.cache_dir / "build_provenance.json")
        expected = {
            "format": "openfly_observed8_instruction_pooler32_native1t_tfds_cache_v1",
            "complete": True,
            "shape": [EXPECTED_ROWS, EXPECTED_FEATURE_DIM],
            "feature_dim": EXPECTED_FEATURE_DIM,
            "dtype": "float16",
            "feature_space": "frozen_memory_tokenizer_pre_projector",
            "query_type": "learned",
            "history_mode": "fifo",
            "history_length": 16,
            "history_context_window": 16,
            "memory_slots": 9,
            "manifest_sha256": self.manifest_sha256,
            "ordered_row_identity_sha256": self.ordered_identity_sha256,
            "rows": EXPECTED_ROWS,
            "episodes": EXPECTED_EPISODES,
            "action_policy": "openfly_observed_8",
            "num_actions": 8,
            "compact_to_openfly": COMPACT_TO_OPENFLY,
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if Path(str(metadata.get("tfds_dataset_dir", ""))).expanduser().resolve() != self.tfds_dir:
            mismatches["tfds_dataset_dir"] = metadata.get("tfds_dataset_dir")
        if metadata.get("tfds_alignment", {}).get("status") != "TFDS_FULL_ROW_ALIGNMENT_PASS":
            mismatches["tfds_alignment"] = metadata.get("tfds_alignment")
        if mismatches:
            raise ValueError(f"Stage2 cache metadata mismatch: {mismatches}")
        if validation.get("complete") is not True or validation.get("manifest_sha256") != self.manifest_sha256:
            raise ValueError("Stage2 cache validation is incomplete or misaligned")
        if validation.get("features_shape") != [EXPECTED_ROWS, EXPECTED_FEATURE_DIM] or validation.get("dtype") != "float16":
            raise ValueError("Stage2 cache validation geometry mismatch")
        parity = validation.get("parity", [])
        if not parity or any(
            item.get("exact_float16") is not True
            or float(item.get("max_abs", -1.0)) != 0.0
            or float(item.get("mean_abs", -1.0)) != 0.0
            for item in parity
        ):
            raise ValueError("Stage2 cache does not have exact TFDS parity")
        if provenance.get("manifest_sha256") != self.manifest_sha256:
            raise ValueError("Stage2 cache provenance manifest mismatch")
        self.cache_metadata = metadata

    @property
    def features(self) -> np.ndarray:
        if self._features is None:
            path = self.cache_dir / str(self.cache_metadata.get("features_file", "features.npy"))
            features = np.load(path, mmap_mode="r", allow_pickle=False)
            if tuple(features.shape) != (EXPECTED_ROWS, EXPECTED_FEATURE_DIM) or features.dtype != np.float16:
                raise ValueError("Stage2 feature array shape/dtype mismatch")
            self._features = features
        return self._features

    @property
    def row_identity_hashes(self) -> np.ndarray:
        if self._row_identity_hashes is None:
            path = self.cache_dir / str(self.cache_metadata.get("row_identity_file", "row_identity_sha256.npy"))
            hashes = np.load(path, mmap_mode="r", allow_pickle=False)
            if tuple(hashes.shape) != (EXPECTED_ROWS, 32) or hashes.dtype != np.uint8:
                raise ValueError("Stage2 row identity array shape/dtype mismatch")
            self._row_identity_hashes = hashes
        return self._row_identity_hashes

    def _verify_row_identity_array(self) -> None:
        hashes = self.row_identity_hashes
        for start in range(0, EXPECTED_ROWS, 8192):
            end = min(start + 8192, EXPECTED_ROWS)
            expected = np.frombuffer(
                b"".join(bytes.fromhex(r["row_identity_sha256"]) for r in self.records[start:end]),
                dtype=np.uint8,
            ).reshape(end - start, 32)
            if not np.array_equal(hashes[start:end], expected):
                raise ValueError(f"Stage2 cache row identity mismatch at {start}:{end}")

    def __len__(self) -> int:
        return len(self.records)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_features"] = self._features
        state["_row_identity_hashes"] = None
        return state

    def __getitem__(self, item: int) -> dict[str, Any]:
        record = self.records[item]
        episode = self.episode_rows[record["episode_id"]]
        if episode[record["episode_position"]] != item:
            raise RuntimeError("Stage2 episode row index drift")
        history = np.zeros((16, self.feature_dim), dtype=np.float16)
        valid = np.zeros(16, dtype=np.bool_)
        source_rows = [-1] * 16
        for slot, position in enumerate(range(record["episode_position"] - 16, record["episode_position"])):
            if position < 0:
                continue
            source = episode[position]
            if source >= item:
                raise RuntimeError("Stage2 history includes current/future row")
            history[slot] = self.features[source]
            valid[slot] = True
            source_rows[slot] = source
        sample = self._tfds[item]
        if (
            _canonical_tfds_episode_id(sample["episode_id"]) != record["episode_id"]
            or int(sample["step_id"]) != int(record["step_id"])
            or str(sample["instruction"]) != record["instruction"]
            or int(sample["action_id"]) != int(record["public_builder_action_id"])
            or not str(sample["image_path"]).endswith(f"#{record['image_id']}")
        ):
            raise RuntimeError(f"TFDS runtime row drift at {item}")
        return {
            "image": sample["image"],
            "memory_features": torch.from_numpy(history),
            "memory_valid_mask": torch.from_numpy(valid),
            "memory_source_manifest_rows": source_rows,
            "instruction": record["instruction"],
            "action_id": int(record["action_id"]),
            "source_action_id": int(record["source_action_id"]),
            "episode_id": record["episode_id"],
            "step_id": int(record["step_id"]),
        }


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--openfly-tfds-dir", required=True)
    parser.add_argument("--openfly-tfds-cache-episodes", type=int, default=32)
    return parser.parse_known_args(argv)


def _write_contract(
    cfg: Any,
    parent: dict[str, Any],
    stage1: dict[str, Any],
    dataset: TFDSStage2Dataset,
    config_path: Path,
    pooler: dict[str, Any],
    source_evidence: dict[str, Any],
    visual_contract: dict[str, Any],
) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    output = Path(cfg.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = Path(cfg.train_manifest).expanduser().resolve()
    manifest_meta = Path(f"{manifest}.meta.json")
    payload = {
        "format": "openfly_observed8_instruction_pooler32_native1t_tfds_stage2_run_contract_v1",
        "complete": True,
        "stage": 2,
        "model_family": "legacy_airminivla_s0_instruction_conditioned_pooler32",
        "pooler": pooler,
        "source": source_evidence,
        "native1t_current_visual_contract": visual_contract,
        "input": {
            "mode": "current_tfds_rgb_plus_cached_past_features",
            "current_frame_tfds_dir": str(dataset.tfds_dir),
            "tfds_alignment": dataset.tfds_alignment,
            "manifest": str(manifest),
            "manifest_sha256": _sha256_file(manifest),
            "manifest_metadata": str(manifest_meta),
            "manifest_metadata_sha256": _sha256_file(manifest_meta),
            "history": "H16 FIFO, oldest-to-newest, current excluded",
            "history_length": 16,
            "strict_predict_before_append": True,
            "cache": str(Path(cfg.native1t_train_cache).expanduser().resolve()),
            "cache_metadata_sha256": _sha256_file(Path(cfg.native1t_train_cache).expanduser().resolve() / "metadata.json"),
            "current_preprocessing": "TFDS PNG decode plus exact Prismatic processor transform",
        },
        "action": {
            "policy": "openfly_observed_8",
            "head_type": "categorical",
            "num_actions": 8,
            "compact_to_openfly": COMPACT_TO_OPENFLY,
        },
        "training": {
            "epochs": int(cfg.epochs),
            "scheduler_horizon_epochs": int(cfg.scheduler_horizon_epochs),
            "batch_size_per_device": int(cfg.batch_size),
            "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps),
            "world_size": int(os.environ.get("WORLD_SIZE", "2")),
            "effective_global_batch": int(cfg.batch_size) * int(cfg.gradient_accumulation_steps) * int(os.environ.get("WORLD_SIZE", "2")),
            "num_workers_per_rank": int(cfg.num_workers),
            "save_every_steps": 0,
            "save_at_steps": list(getattr(cfg, "save_at_steps", []) or []),
            "keep_last_checkpoints": 0,
            "checkpoint_retention": "unlimited",
            "vision_frozen": True,
            "trainable_contract": "Instruction-conditioned 32T Stage2: Native1T memory projector/MEM_BOS plus loaded policy projector/fuse/action surface/LoRA",
        },
        "parent_checkpoint": parent,
        "stage1_checkpoint": stage1,
        "cache_metadata": dataset.cache_metadata,
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
    }
    path = output / "instruction_pooler32_native1t_tfds_stage2_contract.json"
    temporary = path.with_name(f".{path.name}.inprogress")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    adapter_args, trainer_argv = _parse_args(sys.argv[1:])
    if adapter_args.openfly_tfds_cache_episodes <= 0:
        raise ValueError("--openfly-tfds-cache-episodes must be positive")
    trainer_argv = ["--config", adapter_args.config, *trainer_argv]
    from aerial_openvla.core.config import TrainConfig, load_yaml_dataclass
    from aerial_openvla.data.native1t_postfusion_dataset import Native1TCollator
    from aerial_openvla.data.pipeline import DataPipeline, _cached_processor
    import aerial_openvla.training.train as trainer

    config_path = Path(adapter_args.config).expanduser().resolve()
    cfg = load_yaml_dataclass(config_path, TrainConfig)
    _validate_config(cfg)
    validate_pooler32_config(cfg, expected_stage=2)
    source_evidence = verify_native1t_source(Path(__file__).resolve().parents[1])
    visual_contract = build_native1t_visual_contract(
        cfg, source_sha256=source_evidence["classifier_sha256"]
    )
    pooler = _pooler_spec(cfg)
    parent_path = Path(cfg.init_checkpoint).expanduser().resolve()
    stage1_path = Path(cfg.native1t_stage1_checkpoint).expanduser().resolve()
    parent = _checkpoint_info(parent_path, "instruction-conditioned parent E3", EXPECTED_PARENT_STEP, 3, require_native1t=False, pooler=pooler)
    selected = require_selected_stage1(stage1_path)
    stage1 = _checkpoint_info(stage1_path, "selected Native1T Stage1", selected["global_step"], selected["epoch"], require_native1t=True, pooler=pooler)
    stage1_contract_path = stage1_path.parent / STAGE1_CONTRACT_FILENAME
    stage1_contract = _json(stage1_contract_path)
    if (
        stage1_contract.get("format") != STAGE1_CONTRACT_FORMAT
        or stage1_contract.get("complete") is not True
        or int(stage1_contract.get("stage", -1)) != 1
    ):
        raise ValueError("Stage1 source contract is not the completed TFDS Stage1 contract")
    if stage1_contract.get("pooler", {}).get("type") != pooler["type"]:
        raise ValueError("Stage1 source contract and Stage2 pooler disagree")
    expected_stage1_contract = build_native1t_visual_contract(
        cfg,
        source_sha256=source_evidence["classifier_sha256"],
        stage_override=1,
    )
    require_visual_contract_compatible(
        stage1_contract.get("native1t_current_visual_contract"),
        expected_stage1_contract,
        label=f"Stage1 contract {stage1_contract_path}",
    )
    if stage1_contract.get("native1t_current_visual_contract", {}).get(
        "source_classifier_sha256"
    ) != ("5f355d1198dd49b8b57a2a8c2f6d2bbc991d300a8171f615f65a97214b04cd62" if cfg.history_conditioned_local_pooler else source_evidence["classifier_sha256"]):
        raise ValueError("Stage1 contract was created by a different classifier source")
    require_visual_contract_compatible(
        stage1["classifier_meta"].get("native1t_visual_reducer_contract"),
        expected_stage1_contract,
        label=f"Stage1 checkpoint {stage1_path}",
    )
    if Path(stage1_contract.get("tfds_dataset_dir", "")).expanduser().resolve() != Path(
        adapter_args.openfly_tfds_dir
    ).expanduser().resolve():
        raise ValueError("Stage1 contract and Stage2 TFDS directory disagree")
    if Path(stage1_contract.get("train_manifest", "")).expanduser().resolve() != Path(
        cfg.train_manifest
    ).expanduser().resolve():
        raise ValueError("Stage1 contract and Stage2 manifest disagree")
    dataset = TFDSStage2Dataset(
        Path(cfg.train_manifest),
        Path(cfg.native1t_train_cache),
        Path(adapter_args.openfly_tfds_dir),
        tfds_cache_episodes=adapter_args.openfly_tfds_cache_episodes,
    )
    cache_source = Path(str(dataset.cache_metadata.get("source_stage1_checkpoint", ""))).expanduser().resolve()
    if cache_source != stage1_path:
        raise ValueError(f"cache source Stage1 mismatch: {cache_source} != {stage1_path}")
    if Path(str(dataset.cache_metadata.get("tokenizer_file", ""))).expanduser().resolve() != stage1_path / "native1t_postfusion.pt":
        raise ValueError("cache tokenizer does not match Stage1 checkpoint")
    if Path(str(dataset.cache_metadata.get("parent_checkpoint", ""))).expanduser().resolve() != parent_path:
        raise ValueError("cache parent does not match Stage2 parent")
    if dataset.cache_metadata.get("pooler_type") != pooler["type"]:
        raise ValueError("cache and Stage2 parent pooler disagree")
    require_visual_contract_compatible(
        dataset.cache_metadata.get("native1t_current_visual_contract"),
        expected_stage1_contract,
        label=f"cache {cfg.native1t_train_cache}",
    )
    if dataset.cache_metadata.get("source_classifier_sha256") != ("5f355d1198dd49b8b57a2a8c2f6d2bbc991d300a8171f615f65a97214b04cd62" if cfg.history_conditioned_local_pooler else source_evidence["classifier_sha256"]):
        raise ValueError("cache was built by a different classifier source")
    _ = dataset[0]
    for handle in dataset._tfds._shard_handles.values():
        handle.close()
    dataset._tfds._shard_handles.clear()
    dataset._tfds._episode_cache.clear()
    processor = _cached_processor(cfg)
    _write_contract(
        cfg,
        parent,
        stage1,
        dataset,
        config_path,
        pooler,
        source_evidence,
        visual_contract,
    )

    def create_dataset(runtime_cfg: Any, manifest_path: str) -> Any:
        if Path(manifest_path).expanduser().resolve() != Path(cfg.train_manifest).expanduser().resolve():
            raise ValueError("Stage2 received an unexpected manifest path")
        return dataset

    def create_pipeline(runtime_cfg: Any) -> DataPipeline:
        return DataPipeline(
            dataset=create_dataset(runtime_cfg, runtime_cfg.train_manifest),
            collator=Native1TCollator(
                processor,
                runtime_cfg.prompt_template,
                runtime_cfg.max_text_tokens,
                stage=2,
                instruction_only_text_budget=False,
                include_segment_token_ids=False,
                include_instruction_token_mask=True,
                use_relative_state_text=False,
                use_relative_planar_state_text=False,
                action_chunk_horizon=runtime_cfg.action_chunk_horizon,
            ),
        )

    trainer.create_dataset = create_dataset
    trainer.create_pipeline = create_pipeline
    sys.argv = [sys.argv[0], *trainer_argv]
    trainer.main()


if __name__ == "__main__":
    main()
