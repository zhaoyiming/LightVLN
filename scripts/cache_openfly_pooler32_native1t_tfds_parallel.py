#!/usr/bin/env python3
"""Extract the instruction-conditioned 32T Native1T H16 cache from verified OpenFly TFDS.

The cache is one frozen Stage1 Native1T tokenizer feature per immutable
manifest row. Both ranks read the indexed TFDS PNG examples directly; no
Parquet image read, pre-merger cache, or deleted features.npy/H5 source is
allowed. Rank-local NPY shards are merged only after every rank completes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from selected_stage1_contract import require_selected_stage1
from torch.utils.data import DataLoader, Dataset

from native1t_reducer_contract import (
    build_native1t_visual_contract,
    require_visual_contract_compatible,
    validate_pooler32_config,
    verify_native1t_source,
)


CACHE_FORMAT = "openfly_observed8_instruction_pooler32_native1t_tfds_cache_v1"
VALIDATION_FORMAT = "openfly_observed8_instruction_pooler32_native1t_tfds_cache_validation_v1"
STAGE1_CONTRACT_FORMAT = "openfly_observed8_instruction_pooler32_native1t_stage1_tfds_v1"
STAGE1_CONTRACT_FILENAME = "instruction_pooler32_native1t_tfds_contract.json"
MANIFEST_FORMAT = "openfly_observed8_native1t_h16_manifest_v1"
COMPACT_TO_OPENFLY = [0, 1, 2, 3, 4, 5, 8, 9]
EXPECTED_ROWS = 1_608_400
EXPECTED_EPISODES = 100_221
EXPECTED_PARENT_EPOCH = 3
EXPECTED_PARENT_STEP = 18_849
EXPECTED_STAGE1_EPOCH = 1
EXPECTED_STAGE1_STEP = 6_283
EXPECTED_FEATURE_DIM = 2_176


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"missing or empty JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.inprogress")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _stable_file(path: Path, delay: float = 0.25) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"required file is missing or empty: {path}")
    first = (path.stat().st_size, path.stat().st_mtime_ns)
    if delay:
        time.sleep(delay)
    second = (path.stat().st_size, path.stat().st_mtime_ns)
    if first != second:
        raise RuntimeError(f"source file is still changing: {path}: {first} -> {second}")


def _sample_rows(size: int, requested: int) -> list[int]:
    if size <= 0 or requested <= 0:
        raise ValueError("dataset and sample count must be positive")
    points = {0, size // 2, size - 1}
    if requested > 3:
        points.update(int(round(i * (size - 1) / (requested - 1))) for i in range(requested))
    return sorted(points)[:requested]


def _canonical_tfds_episode_id(value: Any) -> str:
    text = str(value)
    try:
        number = int(text.rsplit("-", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"invalid TFDS episode id: {value!r}") from exc
    return f"openfly-s0-{number:08d}"


def _canonical_identity_bytes(raw: dict[str, Any]) -> bytes:
    payload = {
        "action_id": int(raw["action_id"]),
        "boundary_mapped_action_id": int(raw["boundary_mapped_action_id"]),
        "episode_id": str(raw["episode_id"]),
        "episode_position": int(raw["episode_position"]),
        "image_id": str(raw["image_id"]),
        "instruction": str(raw["instruction"]),
        "openfly_action_id": int(raw["openfly_action_id"]),
        "parquet_path": str(Path(str(raw["parquet_path"])).expanduser().resolve()),
        "public_builder_action_id": int(raw["public_builder_action_id"]),
        "source_action_id": int(raw["source_action_id"]),
        "step_id": int(raw["step_id"]),
        "trajectory_id": str(raw.get("trajectory_id", raw["episode_id"])),
        "virtual_image_path": str(raw["virtual_image_path"]),
    }
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _load_manifest(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], str, str]:
    path = path.expanduser().resolve()
    sidecar = Path(f"{path}.meta.json")
    metadata = _json(sidecar)
    if metadata.get("format") != MANIFEST_FORMAT or metadata.get("complete") is not True:
        raise ValueError(f"observed8 manifest metadata is incomplete: {sidecar}")
    if metadata.get("manifest_path") != str(path):
        raise ValueError("manifest metadata path mismatch")
    manifest_sha = _sha256_file(path)
    if metadata.get("manifest_sha256") != manifest_sha:
        raise ValueError("manifest SHA-256 mismatch")
    if metadata.get("action_policy") != "openfly_observed_8" or int(metadata.get("num_actions", -1)) != 8:
        raise ValueError("manifest action policy mismatch")
    compact = metadata.get("compact_to_openfly")
    if isinstance(compact, dict):
        compact = [compact[str(i)] for i in range(8)]
    if [int(value) for value in compact] != COMPACT_TO_OPENFLY:
        raise ValueError("manifest compact action map mismatch")

    records: list[dict[str, Any]] = []
    episode_rows: dict[str, list[int]] = defaultdict(list)
    closed: set[str] = set()
    current_episode: str | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank manifest row at line {line_number}")
            raw = json.loads(line)
            row = int(raw.get("manifest_row", -1))
            if row != len(records):
                raise ValueError(f"manifest row numbering drift at line {line_number}")
            if raw.get("format") != MANIFEST_FORMAT:
                raise ValueError(f"manifest row format drift at line {line_number}")
            if not 0 <= int(raw.get("action_id", -1)) < 8:
                raise ValueError(f"compact action out of range at line {line_number}")
            expected_identity = hashlib.sha256(_canonical_identity_bytes(raw)).hexdigest()
            if str(raw.get("row_identity_sha256")) != expected_identity:
                raise ValueError(f"row identity mismatch at line {line_number}")
            episode_id = str(raw["episode_id"])
            step_id = int(raw["step_id"])
            position = int(raw["episode_position"])
            if step_id != position:
                raise ValueError(f"episode position mismatch at line {line_number}")
            if episode_id != current_episode:
                if current_episode is not None:
                    closed.add(current_episode)
                if episode_id in closed:
                    raise ValueError("manifest episode rows are not contiguous")
                current_episode = episode_id
            expected_position = len(episode_rows[episode_id])
            if position != expected_position:
                raise ValueError(f"episode step is not consecutive at line {line_number}")
            episode_rows[episode_id].append(row)
            records.append({
                "episode_id": episode_id,
                "step_id": step_id,
                "episode_position": position,
                "image_id": str(raw["image_id"]),
                "instruction": str(raw["instruction"]),
                "public_builder_action_id": int(raw["public_builder_action_id"]),
                "openfly_action_id": int(raw["openfly_action_id"]),
                "action_id": int(raw["action_id"]),
                "source_action_id": int(raw["source_action_id"]),
                "row_identity_sha256": str(raw["row_identity_sha256"]),
            })
    if len(records) != int(metadata.get("rows", -1)):
        raise ValueError("manifest row count mismatch")
    if len(episode_rows) != int(metadata.get("episodes", -1)):
        raise ValueError("manifest episode count mismatch")
    ordered = hashlib.sha256()
    for record in records:
        ordered.update(bytes.fromhex(record["row_identity_sha256"]))
    ordered_sha = ordered.hexdigest()
    if metadata.get("ordered_row_identity_sha256") != ordered_sha:
        raise ValueError("ordered row-identity digest mismatch")
    return records, metadata, manifest_sha, ordered_sha


class TFDSCurrentRows(Dataset[dict[str, Any]]):
    """Current RGB rows backed only by the indexed OpenFly TFDS export."""

    def __init__(self, manifest_path: Path, tfds_dir: Path, cache_episodes: int) -> None:
        from aerial_openvla.openfly.observed8_native1t import (
            OpenFlyObserved8Native1TStage1TFDSDataset,
        )

        self.manifest_path = manifest_path.expanduser().resolve()
        self.tfds_dir = tfds_dir.expanduser().resolve()
        self.records, self.metadata, self.manifest_sha256, self.ordered_identity_sha256 = _load_manifest(
            self.manifest_path
        )
        self._validated_source = OpenFlyObserved8Native1TStage1TFDSDataset(
            self.manifest_path,
            self.tfds_dir,
            seed=7,
            history_context_window=16,
            tfds_cache_episodes=int(cache_episodes),
        )
        self._tfds = self._validated_source._tfds
        if len(self.records) != EXPECTED_ROWS or len(self._tfds) != EXPECTED_ROWS:
            raise ValueError(
                f"unexpected TFDS cache cardinality: manifest={len(self.records)}, tfds={len(self._tfds)}"
            )
        if len(self._validated_source._episode_ranges) != EXPECTED_EPISODES:
            raise ValueError("unexpected TFDS episode cardinality")
        self.tfds_alignment = self._verify_full_alignment()

    def _verify_full_alignment(self) -> dict[str, Any]:
        retained = self._tfds.index_dir / "retained_episodes.jsonl"
        if not retained.is_file():
            raise FileNotFoundError(retained)
        compact = {openfly: index for index, openfly in enumerate(COMPACT_TO_OPENFLY)}
        rows = 0
        episodes = 0
        with retained.open("r", encoding="utf-8") as handle:
            for episode_index, line in enumerate(handle):
                episode = json.loads(line)
                if int(episode.get("episode_index", -1)) != episode_index:
                    raise ValueError("TFDS retained episode index drift")
                if int(episode.get("row_start", -1)) != rows:
                    raise ValueError("TFDS retained row_start drift")
                episode_id = f"openfly-s0-{int(episode['episode_id']):08d}"
                actions = [int(value) for value in episode["action_ids"]]
                image_ids = [str(value) for value in episode["source_image_ids"]]
                if len(actions) != int(episode["num_steps"]) or len(actions) != len(image_ids):
                    raise ValueError("TFDS retained episode length drift")
                instructions = []
                for position, (action, image_id) in enumerate(zip(actions, image_ids)):
                    record = self.records[rows]
                    expected = {
                        "episode_id": episode_id,
                        "episode_position": position,
                        "step_id": position,
                        "image_id": image_id,
                        "openfly_action_id": action,
                        "action_id": compact[action],
                    }
                    actual = {
                        "episode_id": record["episode_id"],
                        "episode_position": record["episode_position"],
                        "step_id": record["step_id"],
                        "image_id": record["image_id"],
                        "openfly_action_id": record["openfly_action_id"],
                        "action_id": record["action_id"],
                    }
                    if actual != expected:
                        raise ValueError(
                            f"TFDS/manifest alignment drift at row {rows}: actual={actual}, expected={expected}"
                        )
                    instructions.append(record["instruction"].encode("utf-8"))
                    rows += 1
                if hashlib.sha256(b"\0".join(instructions)).hexdigest() != episode.get(
                    "instruction_sha256"
                ):
                    raise ValueError(f"TFDS instruction drift at episode {episode_index}")
                episodes += 1
        if rows != len(self.records) or episodes != EXPECTED_EPISODES:
            raise ValueError(f"TFDS alignment cardinality drift: rows={rows}, episodes={episodes}")
        return {
            "status": "TFDS_FULL_ROW_ALIGNMENT_PASS",
            "rows": rows,
            "episodes": episodes,
            "retained_episodes": str(retained),
            "retained_episodes_sha256": _sha256_file(retained),
            "fields": ["row", "episode", "step", "image_id", "instruction_sha256", "openfly_action", "compact_action"],
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item: int) -> dict[str, Any]:
        if item < 0:
            item += len(self)
        if item < 0 or item >= len(self):
            raise IndexError(item)
        record = self.records[item]
        sample = self._tfds[item]
        if (
            _canonical_tfds_episode_id(sample["episode_id"]) != record["episode_id"]
            or int(sample["step_id"]) != record["step_id"]
            or str(sample["instruction"]) != record["instruction"]
            or int(sample["action_id"]) != record["public_builder_action_id"]
            or not str(sample["image_path"]).endswith(f"#{record['image_id']}")
        ):
            raise RuntimeError(f"TFDS runtime row alignment failed at row {item}")
        return {
            "manifest_row": item,
            "image": sample["image"],
            "instruction": record["instruction"],
            "row_identity_sha256": record["row_identity_sha256"],
        }


class ShardDataset(Dataset[dict[str, Any]]):
    def __init__(self, source: TFDSCurrentRows, start: int, end: int) -> None:
        self.source = source
        self.start = int(start)
        self.end = int(end)

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, item: int) -> dict[str, Any]:
        if item < 0:
            item += len(self)
        if item < 0 or item >= len(self):
            raise IndexError(item)
        return self.source[self.start + item]


def _collate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "manifest_rows": torch.tensor([int(row["manifest_row"]) for row in rows], dtype=torch.long),
        "images": [row["image"] for row in rows],
        "instructions": [row["instruction"] for row in rows],
        "row_identity_sha256": [row["row_identity_sha256"] for row in rows],
    }


def _pooler_spec(cfg: Any) -> dict[str, Any]:
    """Return the exact 32T parent-pooler identity used by this chain."""
    local = bool(getattr(cfg, "use_instruction_local_pooler", False))
    spatial = bool(getattr(cfg, "use_spatial_anchored_residual_pooler", False))
    if bool(getattr(cfg, "use_task_aware_pooler", False)) or int(local) + int(spatial) != 1:
        raise ValueError(
            "cache requires exactly one new parent pooler: instruction-local or spatial-anchored-residual"
        )
    if local:
        if int(getattr(cfg, "instruction_local_pooler_tokens", -1)) != 32:
            raise ValueError("instruction_local_pooler_tokens must be 32")
        return {
            "type": "instruction_local_weighted32",
            "checkpoint_file": "instruction_local_pooler.pt",
            "metadata": {
                "use_instruction_local_pooler": True,
                "use_spatial_anchored_residual_pooler": False,
                "use_task_aware_pooler": False,
                "instruction_local_pooler_tokens": 32,
            },
        }
    if int(getattr(cfg, "spatial_anchored_pooler_tokens", -1)) != 32 or int(
        getattr(cfg, "spatial_anchored_pooler_heads", -1)
    ) != 8:
        raise ValueError("spatial anchored pooler must be 32 tokens with 8 heads")
    return {
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


def _validate_source_contract(
    config_path: Path,
    checkpoint: Path,
    parent_checkpoint: Path,
    tfds_dir: Path,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from aerial_openvla.core.config import TrainConfig, load_yaml_dataclass

    cfg = load_yaml_dataclass(config_path, TrainConfig)
    validate_pooler32_config(cfg, expected_stage=1)
    source_evidence = verify_native1t_source(Path(__file__).resolve().parents[1])
    visual_contract = build_native1t_visual_contract(
        cfg, source_sha256=source_evidence["classifier_sha256"]
    )
    expected = {
        "model_type": "airminivla",
        "airminivla_variant": "s0",
        "training_stage": 2,
        "openfly_action_policy": "openfly_observed_8",
        "openfly_action_head_type": "categorical",
        "num_actions": 8,
        "num_frames": 1,
        "visual_tokens_per_frame": 32,
        "use_native1t_postfusion": True,
        "native1t_stage": 1,
        "native1t_history_length": 16,
        "native1t_stage1_memory_frames": 9,
        "native1t_history_mode": "fifo",
        "use_prev_action": False,
        "freeze_vision": True,
        "train_projector": False,
        "train_language_model": False,
        "val_manifest": None,
    }
    mismatches = {
        key: (getattr(cfg, key, None), value)
        for key, value in expected.items()
        if getattr(cfg, key, None) != value
    }
    if cfg.openfly_action_token_dim != 0 or cfg.openfly_action_token_bins != 0:
        mismatches["action_token_shape"] = (
            cfg.openfly_action_token_dim,
            cfg.openfly_action_token_bins,
        )
    if cfg.openfly_action_codec_path or cfg.openfly_action_codec_sha256:
        mismatches["action_codec"] = "must be null"
    if getattr(cfg, "openfly_premerger_train_cache", None):
        mismatches["openfly_premerger_train_cache"] = "must be null"
    pooler = _pooler_spec(cfg)
    for key, value in pooler["metadata"].items():
        if getattr(cfg, key, None) != value:
            mismatches[key] = (getattr(cfg, key, None), value)
    if getattr(cfg, "save_every_steps", 0) != 500:
        mismatches["save_every_steps"] = (getattr(cfg, "save_every_steps", None), 500)
    if getattr(cfg, "keep_last_checkpoints", 0) != 0:
        mismatches["keep_last_checkpoints"] = (getattr(cfg, "keep_last_checkpoints", None), 0)
    if mismatches:
        raise ValueError(f"Stage1 TFDS config contract mismatch: {mismatches}")

    contract_path = checkpoint.parent / STAGE1_CONTRACT_FILENAME
    contract = _json(contract_path)
    if (
        contract.get("format") != STAGE1_CONTRACT_FORMAT
        or contract.get("complete") is not True
        or int(contract.get("stage", -1)) != 1
    ):
        raise ValueError(f"invalid current Stage1 TFDS contract: {contract_path}")
    manifest_path = Path(cfg.train_manifest).expanduser().resolve()
    if Path(contract.get("train_manifest", "")).expanduser().resolve() != manifest_path:
        raise ValueError("Stage1 contract/config manifest mismatch")
    if Path(contract.get("tfds_dataset_dir", "")).expanduser().resolve() != tfds_dir:
        raise ValueError("Stage1 contract/TFDS directory mismatch")
    if contract.get("premerger_cache") is not None or contract.get("feature_files") is not None:
        raise ValueError("Stage1 contract declares a non-TFDS cache source")
    if int(contract.get("memory_slots", -1)) != 9 or int(contract.get("native1t_memory_tokens", -1)) != 1:
        raise ValueError("Stage1 Native1T geometry drifted")
    if contract.get("pooler", {}).get("type") != pooler["type"]:
        raise ValueError("Stage1 contract/config parent pooler mismatch")
    require_visual_contract_compatible(
        contract.get("native1t_current_visual_contract"),
        visual_contract,
        label=f"Stage1 contract {contract_path}",
    )
    if contract.get("native1t_current_visual_contract", {}).get(
        "source_classifier_sha256"
    ) != source_evidence["classifier_sha256"]:
        raise ValueError("Stage1 contract was created by a different classifier source")
    parent = contract.get("parent", {})
    if Path(parent.get("path", "")).expanduser().resolve() != parent_checkpoint:
        raise ValueError("Stage1 contract/parent checkpoint mismatch")
    if int(parent.get("epoch", -1)) != EXPECTED_PARENT_EPOCH or int(parent.get("global_step", -1)) != EXPECTED_PARENT_STEP:
        raise ValueError("Stage1 parent is not the selected instruction-conditioned 32T E3")
    training = contract.get("training", {})
    if {
        "epochs": int(training.get("epochs", -1)),
        "world_size": int(training.get("world_size", -1)),
        "batch_size_per_device": int(training.get("batch_size_per_device", -1)),
        "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", -1)),
    } != {"epochs": 1, "world_size": 2, "batch_size_per_device": 32, "gradient_accumulation_steps": 4}:
        raise ValueError("Stage1 training contract drifted")
    if int(training.get("save_every_steps", -1)) != 500 or training.get("checkpoint_retention") != "unlimited":
        raise ValueError("Stage1 checkpoint cadence/retention contract drifted")

    required = (
        "classifier_meta.pt",
        "action_head.pt",
        "action_query.pt",
        "fuse_proj.pt",
        "projector.pt",
        "native1t_postfusion.pt",
        "training_state.pt",
        "metrics.json",
        pooler["checkpoint_file"],
    )
    for filename in required:
        _stable_file(checkpoint / filename)
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    metrics = _json(checkpoint / "metrics.json")
    selected = require_selected_stage1(checkpoint)
    if int(state.get("epoch", -1)) != selected["epoch"] or int(state.get("global_step", -1)) != selected["global_step"]:
        raise ValueError("Stage1 checkpoint state does not match its completed selection gate")
    if int(metrics.get("epoch", -1)) != selected["epoch"] or int(metrics.get("global_step", -1)) != selected["global_step"]:
        raise ValueError(f"Stage1 checkpoint metrics mismatch: {metrics}")
    metadata = torch.load(checkpoint / "classifier_meta.pt", map_location="cpu", weights_only=True)
    for key, value in pooler["metadata"].items():
        actual = bool(metadata.get(key, False)) if isinstance(value, bool) else metadata.get(key)
        if actual != value:
            raise ValueError(f"Stage1 checkpoint pooler metadata mismatch for {key}")
    return cfg, contract, {
        "config_sha256": _sha256_file(config_path),
        "contract_sha256": _sha256_file(contract_path),
        "stage1_state": {"epoch": selected["epoch"], "global_step": selected["global_step"]},
        "parent_state": {"epoch": EXPECTED_PARENT_EPOCH, "global_step": EXPECTED_PARENT_STEP},
        "checkpoint_metrics": metrics,
        "pooler_type": pooler["type"],
        "source_classifier_sha256": source_evidence["classifier_sha256"],
        "source_evidence": source_evidence,
        "native1t_current_visual_contract": visual_contract,
    }


def _build_model(
    cfg: Any, parent_checkpoint: Path, checkpoint: Path, device: torch.device,
    expected_visual_contract: dict[str, Any],
):
    from aerial_openvla.models.factory import create_model

    processor, model = create_model(cfg, device=device)
    model.load_classifier_checkpoint(parent_checkpoint)
    model.load_classifier_checkpoint(checkpoint)
    if model.native1t_memory_tokenizer is None:
        raise RuntimeError("Stage1 model has no Native1T memory tokenizer")
    model.eval()
    model.native1t_memory_tokenizer.eval()
    model.native1t_memory_tokenizer.requires_grad_(False)
    live_visual_contract = model.native1t_visual_reducer_contract()
    require_visual_contract_compatible(
        live_visual_contract,
        expected_visual_contract,
        label="live cache-extraction model",
    )
    feature_dim = int(model._vision_feature_size)
    if feature_dim != EXPECTED_FEATURE_DIM:
        raise ValueError(f"unexpected Native1T feature dimension: {feature_dim}")
    return processor, model, feature_dim


def _extract_features(images: list[Any], processor: Any, model: Any, device: torch.device) -> np.ndarray:
    image_processor = getattr(processor, "image_processor", processor)
    encoded = image_processor(images, return_tensors="pt")
    pixels = encoded["pixel_values"].to(device=device, dtype=torch.bfloat16, non_blocking=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        raw = model.encode_frozen_vision(pixels)
        memory = model.native1t_memory_tokenizer(raw.unsqueeze(1)).squeeze(1)
    if memory.ndim != 2:
        raise RuntimeError(f"Native1T feature output must be [B,D], got {tuple(memory.shape)}")
    result = memory.to(torch.float16).cpu().numpy()
    if not np.isfinite(result).all():
        raise RuntimeError("Native1T cache feature contains non-finite values")
    return result


def _parity(
    dataset: TFDSCurrentRows,
    processor: Any,
    model: Any,
    device: torch.device,
    cached: np.ndarray,
    batch_size: int,
    rows: list[int],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    checked: set[tuple[int, int]] = set()
    for row in rows:
        start = (int(row) // batch_size) * batch_size
        end = min(start + batch_size, len(dataset))
        if (start, end) in checked:
            continue
        checked.add((start, end))
        source = [dataset[index] for index in range(start, end)]
        direct = _extract_features([item["image"] for item in source], processor, model, device)
        actual = np.asarray(cached[start:end])
        exact = bool(np.array_equal(direct, actual))
        reports.append({
            "batch_start": start,
            "batch_end": end,
            "batch_size": end - start,
            "sample_row": int(row),
            "max_abs": float(np.max(np.abs(direct.astype(np.float32) - actual.astype(np.float32)))),
            "mean_abs": float(np.mean(np.abs(direct.astype(np.float32) - actual.astype(np.float32)))),
            "exact_float16": exact,
        })
        if not exact:
            raise RuntimeError(f"TFDS direct/cache parity failed: {reports[-1]}")
    return reports


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--tfds-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--tfds-cache-episodes", type=int, default=32)
    parser.add_argument("--parity-samples", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.tfds_cache_episodes <= 0 or args.parity_samples <= 0:
        raise ValueError("batch size/cache episodes/parity samples must be positive and workers non-negative")
    output = Path(args.output).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    parent_checkpoint = Path(args.parent_checkpoint).expanduser().resolve()
    tfds_dir = Path(args.tfds_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite cache output: {output}")
    if not tfds_dir.is_dir() or not checkpoint.is_dir() or not parent_checkpoint.is_dir():
        raise FileNotFoundError("TFDS, Stage1 checkpoint, or parent checkpoint is missing")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 2:
        raise RuntimeError(f"this cache launcher requires exactly two ranks, got {world_size}")
    if torch.cuda.device_count() < world_size:
        raise RuntimeError(f"allocated GPUs={torch.cuda.device_count()}, expected at least {world_size}")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl")

    try:
        cfg, stage1_contract, evidence = _validate_source_contract(
            config_path, checkpoint, parent_checkpoint, tfds_dir
        )
        manifest_path = Path(cfg.train_manifest).expanduser().resolve()
        dataset = TFDSCurrentRows(manifest_path, tfds_dir, args.tfds_cache_episodes)
        processor, model, feature_dim = _build_model(
            cfg,
            parent_checkpoint,
            checkpoint,
            device,
            evidence["native1t_current_visual_contract"],
        )
        start = (len(dataset) * local_rank) // world_size
        end = (len(dataset) * (local_rank + 1)) // world_size
        shard_length = end - start

        if local_rank == 0:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.mkdir(exist_ok=False)
        dist.barrier()

        shard_features_path = output / f"features.rank{local_rank:02d}.inprogress.npy"
        shard_ids_path = output / f"row_identity_sha256.rank{local_rank:02d}.inprogress.npy"
        shard_features = np.lib.format.open_memmap(
            shard_features_path, mode="w+", dtype=np.float16, shape=(shard_length, feature_dim)
        )
        shard_ids = np.lib.format.open_memmap(
            shard_ids_path, mode="w+", dtype=np.uint8, shape=(shard_length, 32)
        )
        loader_kwargs: dict[str, Any] = {
            "dataset": ShardDataset(dataset, start, end),
            "batch_size": args.batch_size,
            "shuffle": False,
            "num_workers": args.num_workers,
            "pin_memory": True,
            "collate_fn": _collate_rows,
        }
        if args.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        loader = DataLoader(**loader_kwargs)
        started = time.perf_counter()
        rows_written = 0
        for batch_index, batch in enumerate(loader, start=1):
            manifest_rows = batch["manifest_rows"]
            expected_rows = torch.arange(
                start + rows_written,
                start + rows_written + len(manifest_rows),
                dtype=torch.long,
            )
            if not torch.equal(manifest_rows.cpu(), expected_rows):
                raise RuntimeError("cache loader changed immutable manifest-row order")
            expected_ids = [
                dataset.records[index]["row_identity_sha256"]
                for index in range(start + rows_written, start + rows_written + len(manifest_rows))
            ]
            if batch["row_identity_sha256"] != expected_ids:
                raise RuntimeError("cache loader changed immutable row-identity order")
            result = _extract_features(batch["images"], processor, model, device)
            if tuple(result.shape) != (len(manifest_rows), feature_dim):
                raise RuntimeError(f"feature shape mismatch: {result.shape}")
            local_end = rows_written + len(manifest_rows)
            shard_features[rows_written:local_end] = result
            shard_ids[rows_written:local_end] = np.frombuffer(
                b"".join(bytes.fromhex(identity) for identity in expected_ids), dtype=np.uint8
            ).reshape(len(manifest_rows), 32)
            rows_written = local_end
            if local_rank == 0 and (batch_index % 20 == 0 or rows_written == shard_length):
                elapsed = max(time.perf_counter() - started, 1.0e-6)
                print(
                    json.dumps({
                        "rank": 0,
                        "rank_rows": rows_written,
                        "shard_rows": shard_length,
                        "total_rows": len(dataset),
                        "global_percent_estimate": round(100.0 * rows_written * world_size / len(dataset), 3),
                        "aggregate_rows_per_second_estimate": round(world_size * rows_written / elapsed, 2),
                        "batches": batch_index,
                    }, sort_keys=True),
                    flush=True,
                )
        del loader
        if rows_written != shard_length:
            raise RuntimeError(f"rank {local_rank} wrote {rows_written}/{shard_length} rows")
        shard_features.flush()
        shard_ids.flush()
        del shard_features, shard_ids
        rank_elapsed = max(time.perf_counter() - started, 1.0e-6)
        _atomic_json(output / f"rank_{local_rank:02d}.json", {
            "rank": local_rank,
            "world_size": world_size,
            "rows": shard_length,
            "shard_start": start,
            "shard_end": end,
            "batch_size_per_rank": args.batch_size,
            "workers_per_rank": args.num_workers,
            "elapsed_seconds": rank_elapsed,
            "rows_per_second": shard_length / rank_elapsed,
            "feature_dim": feature_dim,
        })
        dist.barrier()

        if local_rank == 0:
            merge_started = time.perf_counter()
            global_features_path = output / "features.inprogress.npy"
            global_ids_path = output / "row_identity_sha256.inprogress.npy"
            global_features = np.lib.format.open_memmap(
                global_features_path, mode="w+", dtype=np.float16, shape=(len(dataset), feature_dim)
            )
            global_ids = np.lib.format.open_memmap(
                global_ids_path, mode="w+", dtype=np.uint8, shape=(len(dataset), 32)
            )
            rank_records = []
            for rank in range(world_size):
                rank_features = np.load(
                    output / f"features.rank{rank:02d}.inprogress.npy", mmap_mode="r", allow_pickle=False
                )
                rank_ids = np.load(
                    output / f"row_identity_sha256.rank{rank:02d}.inprogress.npy", mmap_mode="r", allow_pickle=False
                )
                rank_record = _json(output / f"rank_{rank:02d}.json")
                shard_start = int(rank_record["shard_start"])
                shard_end = int(rank_record["shard_end"])
                if tuple(rank_features.shape) != (shard_end - shard_start, feature_dim):
                    raise RuntimeError(f"rank {rank} feature shard shape mismatch")
                if tuple(rank_ids.shape) != (shard_end - shard_start, 32):
                    raise RuntimeError(f"rank {rank} identity shard shape mismatch")
                global_features[shard_start:shard_end] = rank_features
                global_ids[shard_start:shard_end] = rank_ids
                rank_records.append(rank_record)
            global_features.flush()
            global_ids.flush()
            del global_features, global_ids, rank_features, rank_ids
            final_features = output / "features.npy"
            final_ids = output / "row_identity_sha256.npy"
            os.replace(global_features_path, final_features)
            os.replace(global_ids_path, final_ids)
            cached = np.load(final_features, mmap_mode="r", allow_pickle=False)
            cached_ids = np.load(final_ids, mmap_mode="r", allow_pickle=False)
            if tuple(cached.shape) != (EXPECTED_ROWS, EXPECTED_FEATURE_DIM) or cached.dtype != np.float16:
                raise RuntimeError(f"merged feature array mismatch: shape={cached.shape}, dtype={cached.dtype}")
            if tuple(cached_ids.shape) != (EXPECTED_ROWS, 32) or cached_ids.dtype != np.uint8:
                raise RuntimeError(f"merged identity array mismatch: shape={cached_ids.shape}, dtype={cached_ids.dtype}")
            for row in _sample_rows(len(dataset), args.parity_samples):
                expected = np.frombuffer(bytes.fromhex(dataset.records[row]["row_identity_sha256"]), dtype=np.uint8)
                if not np.array_equal(cached_ids[row], expected):
                    raise RuntimeError(f"merged identity mismatch at row {row}")
            parity = _parity(
                dataset, processor, model, device, cached, args.batch_size,
                _sample_rows(len(dataset), args.parity_samples),
            )
            extraction_wall = max(float(record["elapsed_seconds"]) for record in rank_records)
            merge_seconds = max(time.perf_counter() - merge_started, 1.0e-6)
            total_seconds = max(time.perf_counter() - started, 1.0e-6)
            manifest_meta = Path(f"{manifest_path}.meta.json")
            metadata = {
                "format": CACHE_FORMAT,
                "complete": True,
                "cache_dir": str(output),
                "shape": [EXPECTED_ROWS, EXPECTED_FEATURE_DIM],
                "feature_dim": EXPECTED_FEATURE_DIM,
                "dtype": "float16",
                "feature_space": "frozen_memory_tokenizer_pre_projector",
                "query_type": "learned",
                "pooler_type": evidence["pooler_type"],
                "history_cache_semantics": "one Stage1 Native1T tokenizer feature per immutable manifest row",
                "history_mode": "fifo",
                "history_length": 16,
                "history_context_window": 16,
                "memory_slots": 9,
                "manifest_path": str(manifest_path),
                "manifest_sha256": dataset.manifest_sha256,
                "manifest_metadata_sha256": _sha256_file(manifest_meta),
                "ordered_row_identity_sha256": dataset.ordered_identity_sha256,
                "row_identity_file": final_ids.name,
                "row_identity_file_sha256": _sha256_file(final_ids),
                "features_file": final_features.name,
                "features_file_sha256": _sha256_file(final_features),
                "rows": EXPECTED_ROWS,
                "episodes": EXPECTED_EPISODES,
                "source_action_counts": dataset.metadata.get("source_action_counts"),
                "mapped_action_counts": dataset.metadata.get("compact_action_counts"),
                "action_policy": "openfly_observed_8",
                "num_actions": 8,
                "openfly_action_head_type": "categorical",
                "compact_to_openfly": COMPACT_TO_OPENFLY,
                "boundary_source_map": {"-1": 4, "-2": 5},
                "final_two_stop": True,
                "tfds_dataset_dir": str(tfds_dir),
                "tfds_index_metadata": str(dataset._tfds.index_dir / "metadata.json"),
                "tfds_index_metadata_sha256": _sha256_file(dataset._tfds.index_dir / "metadata.json"),
                "tfds_alignment": dataset.tfds_alignment,
                "image_decode_provenance": "indexed TFDS TFRecord PNG -> PIL RGB -> exact Prismatic transform",
                "source_stage1_checkpoint": str(checkpoint),
                "source_stage1_contract": str(checkpoint.parent / STAGE1_CONTRACT_FILENAME),
                "source_stage1_contract_sha256": evidence["contract_sha256"],
                "source_stage1_config": str(config_path),
                "source_stage1_config_sha256": evidence["config_sha256"],
                "source_classifier_sha256": evidence["source_classifier_sha256"],
                "native1t_current_visual_contract": evidence["native1t_current_visual_contract"],
                "parent_checkpoint": str(parent_checkpoint),
                "parent_training_state": evidence["parent_state"],
                "stage1_training_state": evidence["stage1_state"],
                "tokenizer_file": str(checkpoint / "native1t_postfusion.pt"),
                "tokenizer_sha256": _sha256_file(checkpoint / "native1t_postfusion.pt"),
                "stage1_contract": stage1_contract,
                "stage1_checkpoint_metrics": evidence["checkpoint_metrics"],
                "parity_samples": parity,
                "parallel_world_size": world_size,
                "parallel_batch_size_per_rank": args.batch_size,
                "parallel_workers_per_rank": args.num_workers,
                "parallel_rank_records": rank_records,
                "parallel_extraction_wall_seconds": round(extraction_wall, 3),
                "elapsed_seconds": round(total_seconds, 3),
                "merge_seconds": round(merge_seconds, 3),
            }
            metadata_path = output / "metadata.json"
            _atomic_json(metadata_path, metadata)
            validation = {
                "format": VALIDATION_FORMAT,
                "complete": True,
                "cache_dir": str(output),
                "cache_metadata_sha256": _sha256_file(metadata_path),
                "manifest_sha256": dataset.manifest_sha256,
                "ordered_row_identity_sha256": dataset.ordered_identity_sha256,
                "features_shape": [EXPECTED_ROWS, EXPECTED_FEATURE_DIM],
                "dtype": "float16",
                "parity": parity,
                "identity_mapping": "one manifest row -> one cache row; ordered row identities verified",
                "tfds_alignment": dataset.tfds_alignment,
            }
            _atomic_json(output / "validation.json", validation)
            _atomic_json(output / "build_provenance.json", {
                "format": "openfly_observed8_instruction_pooler32_native1t_tfds_cache_provenance_v1",
                "cache_format": CACHE_FORMAT,
                "source_stage1_checkpoint": str(checkpoint),
                "parent_checkpoint": str(parent_checkpoint),
                "manifest": str(manifest_path),
                "manifest_sha256": dataset.manifest_sha256,
                "tfds_dataset_dir": str(tfds_dir),
                "tfds_alignment": dataset.tfds_alignment,
                "source_stage1_config_sha256": evidence["config_sha256"],
                "source_stage1_contract_sha256": evidence["contract_sha256"],
                "source_classifier_sha256": evidence["source_classifier_sha256"],
                "native1t_current_visual_contract": evidence["native1t_current_visual_contract"],
                "cache_metadata": str(metadata_path),
                "cache_validation": str(output / "validation.json"),
            })
            for rank in range(world_size):
                (output / f"features.rank{rank:02d}.inprogress.npy").unlink()
                (output / f"row_identity_sha256.rank{rank:02d}.inprogress.npy").unlink()
            print(json.dumps({
                "status": "OPENFLY_INSTRUCTION_POOLER32_NATIVE1T_TFDS_CACHE_PASS",
                "cache_dir": str(output),
                "rows": EXPECTED_ROWS,
                "feature_dim": EXPECTED_FEATURE_DIM,
                "parallel_extraction_wall_seconds": round(extraction_wall, 3),
                "rows_per_second": round(EXPECTED_ROWS / extraction_wall, 2),
                "validation": str(output / "validation.json"),
            }, indent=2, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
