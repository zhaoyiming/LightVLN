"""OpenFly-specific Native1T data and cache contracts.

This module deliberately does not reuse the AirVLN ``Native1TStage*Dataset``
classes: OpenFly frames are immutable image bytes in Parquet files rather than
filesystem PNG paths.  It provides a separate, manifest-row-addressed H8 FIFO
pipeline while retaining the shared AirMiniVLA Native1T model and collator.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import time
from collections import Counter, OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from aerial_openvla.openfly.parquet_dataset import (
    OPENFLY_ACTION_NAMES,
    OpenFlyParquetClassificationDataset,
    action_policy_num_actions,
)


MANIFEST_FORMAT = "openfly_native1t_manifest_v1"
CACHE_FORMAT = "openfly_native1t_1t_cache_v1"
CACHE_VALIDATION_FORMAT = "openfly_native1t_1t_cache_validation_v1"
NATIVE_ACTION_POLICY = "openfly_10"


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest for a stable on-disk artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    """Write one metadata document atomically without accepting partial state."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.inprogress")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite incomplete metadata: {temporary}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def require_complete_openfly_native1t_stage1_checkpoint(
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Fail closed on a partial or non-OpenFly Native1T Stage1 checkpoint.

    The legacy saver writes files individually rather than publishing a single
    archive.  This gate therefore requires the full Stage1 inference/training
    surface, stable file stats, loadable metadata and a completed epoch before
    a cache or Stage2 can consume the checkpoint.
    """
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"OpenFly Native1T Stage1 checkpoint is missing: {checkpoint}")
    required_names = (
        "classifier_meta.pt",
        "action_head.pt",
        "action_query.pt",
        "fuse_proj.pt",
        "projector.pt",
        "native1t_postfusion.pt",
        "training_state.pt",
        "openfly_action_codec.json",
    )
    required = [checkpoint / name for name in required_names]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "OpenFly Native1T Stage1 checkpoint is incomplete; missing: "
            + ", ".join(missing)
        )
    processor_dir = checkpoint / "processor"
    if not processor_dir.is_dir() or not any(processor_dir.iterdir()):
        raise FileNotFoundError(
            f"OpenFly Native1T Stage1 checkpoint has no saved processor: {processor_dir}"
        )
    lora_dir = checkpoint / "lora"
    if not lora_dir.is_dir() or not (lora_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"OpenFly Native1T Stage1 checkpoint has no canonical LoRA adapter: {lora_dir}"
        )
    if not any(
        (lora_dir / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise FileNotFoundError(
            f"OpenFly Native1T Stage1 LoRA has no adapter weights: {lora_dir}"
        )
    watched = [*required, lora_dir / "adapter_config.json"]
    adapter_files = [
        path for path in (lora_dir / "adapter_model.safetensors", lora_dir / "adapter_model.bin")
        if path.is_file()
    ]
    watched.extend(adapter_files)
    before = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in watched}
    if any(size <= 0 for size, _ in before.values()):
        raise ValueError("OpenFly Native1T Stage1 checkpoint contains an empty required file")
    time.sleep(0.2)
    after = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in watched}
    if before != after:
        raise RuntimeError(
            "OpenFly Native1T Stage1 checkpoint is still changing; retry after it is stable"
        )
    try:
        metadata = torch.load(
            checkpoint / "classifier_meta.pt", map_location="cpu", weights_only=True
        )
        state = torch.load(
            checkpoint / "training_state.pt", map_location="cpu", weights_only=False
        )
    except Exception as exc:
        raise RuntimeError(
            f"OpenFly Native1T Stage1 checkpoint metadata/state is unreadable: {checkpoint}"
        ) from exc
    if not isinstance(metadata, dict) or not isinstance(state, dict):
        raise ValueError("OpenFly Native1T Stage1 checkpoint metadata/state must be dictionaries")
    expected_meta = {
        "num_actions": 10,
        "variant": "s0",
        "num_frames": 1,
        "use_native1t_postfusion": True,
        "native1t_stage": 1,
        "native1t_history_length": 8,
        "native1t_history_mode": "fifo",
        "openfly_action_token_dim": 8,
        "openfly_action_token_bins": 256,
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in expected_meta.items()
        if metadata.get(key) != expected
    }
    if metadata.get("openfly_action_head_type") not in {
        "action_tokens_v1", "action_tokens_ar_v2"
    }:
        mismatches["openfly_action_head_type"] = (
            metadata.get("openfly_action_head_type"),
            "action_tokens_v1|action_tokens_ar_v2",
        )
    if mismatches:
        raise ValueError(
            f"Stage1 checkpoint does not match the minimal OpenFly Native1T contract: {mismatches}"
        )
    codec_digest = metadata.get("openfly_action_codec_sha256")
    if not isinstance(codec_digest, str) or not codec_digest:
        raise ValueError("Stage1 checkpoint is missing its OpenFly action codec digest")
    from aerial_openvla.openfly.action_codec import load_openfly_action_codec

    codec = load_openfly_action_codec(
        checkpoint / "openfly_action_codec.json", expected_sha256=codec_digest
    )
    try:
        completed_epoch = int(state["epoch"])
        global_step = int(state["global_step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("OpenFly Native1T Stage1 training_state has no epoch/global_step") from exc
    if completed_epoch < 1 or global_step <= 0:
        raise ValueError(
            "OpenFly Native1T Stage1 checkpoint is not a completed epoch boundary"
        )
    return {
        "path": str(checkpoint),
        "classifier_meta_sha256": sha256_file(checkpoint / "classifier_meta.pt"),
        "native1t_postfusion_sha256": sha256_file(checkpoint / "native1t_postfusion.pt"),
        "training_state_sha256": sha256_file(checkpoint / "training_state.pt"),
        "openfly_action_codec_file_sha256": sha256_file(
            checkpoint / "openfly_action_codec.json"
        ),
        "openfly_action_codec_sha256": codec.sha256,
        "completed_epoch": completed_epoch,
        "global_step": global_step,
    }


def manifest_metadata_path(manifest_path: str | Path) -> Path:
    path = Path(manifest_path).resolve()
    return Path(f"{path}.meta.json")


def require_native_action_policy(action_policy: str) -> None:
    """Native1T OpenFly keeps all three forward-distance commands distinct."""
    if action_policy != NATIVE_ACTION_POLICY:
        raise ValueError(
            "OpenFly Native1T requires action_policy='openfly_10'; legacy "
            "8-way collapse/reject policies would lose the 6m/9m commands."
        )
    if action_policy_num_actions(action_policy) != 10:
        raise RuntimeError("OpenFly Native1T action-policy metadata is inconsistent")


def _count_dict(counter: Counter[int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _normalise_count_dict(value: Any, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    result: dict[str, int] = {}
    for key, count in value.items():
        parsed_key = int(key)
        parsed_count = int(count)
        if parsed_count < 0:
            raise ValueError(f"{label}[{key!r}] must be non-negative")
        result[str(parsed_key)] = parsed_count
    return dict(sorted(result.items(), key=lambda item: int(item[0])))


@dataclass(frozen=True, slots=True)
class OpenFlyNative1TRecord:
    """One immutable supervised OpenFly row, indexed by manifest row."""

    manifest_row: int
    episode_id: str
    trajectory_id: str
    step_id: int
    episode_position: int
    parquet_path: Path
    image_id: str
    virtual_image_path: str
    instruction: str
    source_action_id: int
    action_id: int
    row_identity_sha256: str

    def identity_payload(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "trajectory_id": self.trajectory_id,
            "step_id": self.step_id,
            "episode_position": self.episode_position,
            "parquet_path": str(self.parquet_path),
            "image_id": self.image_id,
            "virtual_image_path": self.virtual_image_path,
            "instruction": self.instruction,
            "source_action_id": self.source_action_id,
            "action_id": self.action_id,
        }

    @property
    def identity_bytes(self) -> bytes:
        return hashlib.sha256(canonical_json_bytes(self.identity_payload())).digest()


@dataclass(frozen=True, slots=True)
class OpenFlyNative1TManifest:
    path: Path
    metadata_path: Path
    records: tuple[OpenFlyNative1TRecord, ...]
    metadata: dict[str, Any]
    episode_rows: dict[str, tuple[int, ...]]

    @property
    def action_policy(self) -> str:
        return str(self.metadata["action_policy"])

    @property
    def num_actions(self) -> int:
        return int(self.metadata["num_actions"])

    @property
    def manifest_sha256(self) -> str:
        return str(self.metadata["manifest_sha256"])

    @property
    def row_identity_sha256(self) -> str:
        return str(self.metadata["ordered_row_identity_sha256"])


def _ordered_identity_digest(records: list[OpenFlyNative1TRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.identity_bytes)
    return digest.hexdigest()


def _require_nonempty_string(value: Any, field: str, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest row {line_number} has no non-empty {field}")
    return value.strip()


def _record_from_json(raw: Mapping[str, Any], line_number: int) -> OpenFlyNative1TRecord:
    if raw.get("format") != MANIFEST_FORMAT:
        raise ValueError(
            f"Manifest row {line_number} has format={raw.get('format')!r}; "
            f"expected {MANIFEST_FORMAT!r}"
        )
    try:
        manifest_row = int(raw["manifest_row"])
        step_id = int(raw["step_id"])
        episode_position = int(raw["episode_position"])
        source_action_id = int(raw["source_action_id"])
        action_id = int(raw["action_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Malformed numeric field at manifest row {line_number}") from exc
    if manifest_row != line_number - 1:
        raise ValueError(
            f"Manifest row field drift at line {line_number}: {manifest_row} != {line_number - 1}"
        )
    if step_id < 0 or episode_position < 0:
        raise ValueError(f"Manifest row {line_number} has a negative step/position")
    if source_action_id not in OPENFLY_ACTION_NAMES or action_id not in OPENFLY_ACTION_NAMES:
        raise ValueError(f"Manifest row {line_number} contains an out-of-range OpenFly action")
    # Native1T deliberately implements the full OpenFly command space only.
    if source_action_id != action_id:
        raise ValueError(
            f"Manifest row {line_number} maps source action {source_action_id} to {action_id}; "
            "Native1T requires native identity labels."
        )
    parquet_path = Path(_require_nonempty_string(raw.get("parquet_path"), "parquet_path", line_number)).expanduser().resolve()
    if parquet_path.suffix != ".parquet" or not parquet_path.is_file():
        raise FileNotFoundError(f"Manifest row {line_number} references missing Parquet: {parquet_path}")
    image_id = _require_nonempty_string(raw.get("image_id"), "image_id", line_number)
    expected_virtual_path = f"{parquet_path}#{image_id}"
    virtual_image_path = _require_nonempty_string(
        raw.get("virtual_image_path"), "virtual_image_path", line_number
    )
    if virtual_image_path != expected_virtual_path:
        raise ValueError(
            f"Manifest row {line_number} virtual image path does not match its Parquet/image ID"
        )
    record = OpenFlyNative1TRecord(
        manifest_row=manifest_row,
        episode_id=_require_nonempty_string(raw.get("episode_id"), "episode_id", line_number),
        trajectory_id=_require_nonempty_string(raw.get("trajectory_id"), "trajectory_id", line_number),
        step_id=step_id,
        episode_position=episode_position,
        parquet_path=parquet_path,
        image_id=image_id,
        virtual_image_path=virtual_image_path,
        instruction=_require_nonempty_string(raw.get("instruction"), "instruction", line_number),
        source_action_id=source_action_id,
        action_id=action_id,
        row_identity_sha256=_require_nonempty_string(
            raw.get("row_identity_sha256"), "row_identity_sha256", line_number
        ),
    )
    expected_identity = record.identity_bytes.hex()
    if record.row_identity_sha256 != expected_identity:
        raise ValueError(f"Manifest row {line_number} identity hash mismatch")
    return record


def _episode_rows(records: list[OpenFlyNative1TRecord]) -> dict[str, tuple[int, ...]]:
    """Require contiguous, strictly causal OpenFly episodes in manifest order."""
    result: dict[str, list[int]] = {}
    closed: set[str] = set()
    current_episode: str | None = None
    for record in records:
        if record.episode_id != current_episode:
            if current_episode is not None:
                closed.add(current_episode)
            if record.episode_id in closed:
                raise ValueError(
                    f"Episode {record.episode_id!r} is not contiguous in the Native1T manifest"
                )
            current_episode = record.episode_id
        rows = result.setdefault(record.episode_id, [])
        expected_position = len(rows)
        if record.episode_position != expected_position or record.step_id != expected_position:
            raise ValueError(
                f"Episode {record.episode_id!r} requires consecutive step/position 0..; "
                f"got step={record.step_id}, position={record.episode_position}, expected={expected_position}"
            )
        rows.append(record.manifest_row)
    if not result:
        raise ValueError("Native1T manifest contains no rows")
    return {episode_id: tuple(rows) for episode_id, rows in result.items()}


def load_openfly_native1t_manifest(
    manifest_path: str | Path,
    *,
    action_policy: str | None = NATIVE_ACTION_POLICY,
    num_actions: int | None = 10,
) -> OpenFlyNative1TManifest:
    """Load and fail-closed validate the manifest plus its immutable sidecar."""
    path = Path(manifest_path).expanduser().resolve()
    metadata_path = manifest_metadata_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"OpenFly Native1T manifest not found: {path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"OpenFly Native1T manifest metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Manifest metadata must be an object: {metadata_path}")
    if metadata.get("format") != MANIFEST_FORMAT or not metadata.get("complete"):
        raise ValueError(f"Manifest metadata is incomplete or has the wrong format: {metadata_path}")
    if metadata.get("manifest_path") != str(path):
        raise ValueError("Manifest metadata path mismatch")
    if metadata.get("manifest_sha256") != sha256_file(path):
        raise ValueError("Manifest metadata SHA-256 mismatch")
    recorded_policy = str(metadata.get("action_policy"))
    require_native_action_policy(recorded_policy)
    if action_policy is not None and recorded_policy != action_policy:
        raise ValueError(
            f"Manifest action policy mismatch: {recorded_policy!r} != {action_policy!r}"
        )
    recorded_num_actions = int(metadata.get("num_actions", -1))
    if recorded_num_actions != 10:
        raise ValueError("Native1T manifest must advertise exactly 10 OpenFly action IDs")
    if num_actions is not None and recorded_num_actions != int(num_actions):
        raise ValueError(
            f"Manifest action dimension mismatch: {recorded_num_actions} != {num_actions}"
        )

    records: list[OpenFlyNative1TRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank row in Native1T manifest at line {line_number}")
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at manifest line {line_number}") from exc
            if not isinstance(raw, Mapping):
                raise ValueError(f"Manifest line {line_number} is not an object")
            records.append(_record_from_json(raw, line_number))
    if int(metadata.get("row_count", -1)) != len(records):
        raise ValueError("Manifest row count disagrees with metadata")
    identities = _ordered_identity_digest(records)
    if metadata.get("ordered_row_identity_sha256") != identities:
        raise ValueError("Manifest ordered row-identity digest mismatch")
    source_counts = _count_dict(Counter(record.source_action_id for record in records))
    mapped_counts = _count_dict(Counter(record.action_id for record in records))
    if _normalise_count_dict(metadata.get("source_action_counts"), "source_action_counts") != source_counts:
        raise ValueError("Manifest source action counts mismatch")
    if _normalise_count_dict(metadata.get("mapped_action_counts"), "mapped_action_counts") != mapped_counts:
        raise ValueError("Manifest mapped action counts mismatch")
    episodes = _episode_rows(records)
    return OpenFlyNative1TManifest(
        path=path,
        metadata_path=metadata_path,
        records=tuple(records),
        metadata=metadata,
        episode_rows=episodes,
    )


def build_openfly_native1t_manifest(
    *,
    annotation_json: str | Path,
    trajectory_root: str | Path,
    output_path: str | Path,
    action_policy: str = NATIVE_ACTION_POLICY,
    max_episodes: int = 0,
) -> OpenFlyNative1TManifest:
    """Build one non-overwritable, row-addressed Native1T training manifest."""
    require_native_action_policy(action_policy)
    if max_episodes < 0:
        raise ValueError("max_episodes must be non-negative")
    output = Path(output_path).expanduser().resolve()
    sidecar = manifest_metadata_path(output)
    if output.exists() or sidecar.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing OpenFly Native1T manifest or metadata: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    source = OpenFlyParquetClassificationDataset(
        annotation_json,
        trajectory_root,
        action_policy=action_policy,
        num_frames=1,
        max_episodes=max_episodes,
    )
    records: list[OpenFlyNative1TRecord] = []
    temporary = output.with_name(f".{output.name}.inprogress")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite incomplete manifest: {temporary}")
    with temporary.open("x", encoding="utf-8") as handle:
        for manifest_row, source_record in enumerate(source.records):
            identity_payload = {
                "episode_id": source_record.episode_id,
                "trajectory_id": source_record.trajectory_id,
                "step_id": int(source_record.step_id),
                "episode_position": int(source_record.episode_position),
                "parquet_path": str(source_record.parquet_path.resolve()),
                "image_id": str(source_record.image_id),
                "virtual_image_path": source_record.virtual_image_path,
                "instruction": source_record.instruction,
                "source_action_id": int(source_record.source_action_id),
                "action_id": int(source_record.action_id),
            }
            identity = hashlib.sha256(canonical_json_bytes(identity_payload)).hexdigest()
            row = {
                "format": MANIFEST_FORMAT,
                "manifest_row": manifest_row,
                **identity_payload,
                "row_identity_sha256": identity,
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            records.append(
                OpenFlyNative1TRecord(
                    manifest_row=manifest_row,
                    episode_id=source_record.episode_id,
                    trajectory_id=source_record.trajectory_id,
                    step_id=int(source_record.step_id),
                    episode_position=int(source_record.episode_position),
                    parquet_path=source_record.parquet_path.resolve(),
                    image_id=str(source_record.image_id),
                    virtual_image_path=source_record.virtual_image_path,
                    instruction=source_record.instruction,
                    source_action_id=int(source_record.source_action_id),
                    action_id=int(source_record.action_id),
                    row_identity_sha256=identity,
                )
            )
    os.replace(temporary, output)
    metadata = {
        "format": MANIFEST_FORMAT,
        "complete": True,
        "manifest_path": str(output),
        "manifest_sha256": sha256_file(output),
        "row_count": len(records),
        "ordered_row_identity_sha256": _ordered_identity_digest(records),
        "action_policy": action_policy,
        "num_actions": 10,
        "source_annotation": str(Path(annotation_json).expanduser().resolve()),
        "source_annotation_sha256": sha256_file(source.annotation_json),
        "trajectory_root": str(Path(trajectory_root).expanduser().resolve()),
        "source_action_counts": _count_dict(source.source_action_counts),
        "mapped_action_counts": _count_dict(source.mapped_action_counts),
    }
    atomic_write_json(sidecar, metadata)
    return load_openfly_native1t_manifest(output)


class _ParquetImageStore:
    """Per-worker LRU cache of decoded OpenFly Parquet image bytes."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("trajectory cache capacity must be positive")
        self.capacity = int(capacity)
        self._trajectories: OrderedDict[Path, dict[str, bytes]] = OrderedDict()

    def image(self, record: OpenFlyNative1TRecord) -> Image.Image:
        images = self._load_trajectory(record.parquet_path)
        try:
            payload = images[record.image_id]
        except KeyError as exc:
            raise KeyError(
                f"OpenFly Parquet {record.parquet_path} does not contain image_id={record.image_id!r}"
            ) from exc
        with Image.open(io.BytesIO(payload)) as image:
            return image.convert("RGB")

    def _load_trajectory(self, parquet_path: Path) -> dict[str, bytes]:
        cached = self._trajectories.pop(parquet_path, None)
        if cached is not None:
            self._trajectories[parquet_path] = cached
            return cached
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - environment contract
            raise RuntimeError("OpenFly Native1T Parquet decoding requires pyarrow") from exc
        table = pq.read_table(parquet_path, columns=["image_id", "image"])
        images: dict[str, bytes] = {}
        for row in table.to_pylist():
            image = row.get("image")
            payload = image.get("bytes") if isinstance(image, Mapping) else None
            if payload:
                images[str(row["image_id"])] = bytes(payload)
        if not images:
            raise RuntimeError(f"OpenFly Parquet has no usable RGB bytes: {parquet_path}")
        self._trajectories[parquet_path] = images
        while len(self._trajectories) > self.capacity:
            self._trajectories.popitem(last=False)
        return images

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_trajectories"] = OrderedDict()
        return state


class OpenFlyNative1TFrameDataset(Dataset[dict[str, Any]]):
    """One RGB frame per immutable OpenFly Native1T manifest row."""

    def __init__(
        self, manifest_path: str | Path, *, trajectory_cache_size: int = 4
    ) -> None:
        self.manifest = load_openfly_native1t_manifest(manifest_path)
        self._images = _ParquetImageStore(trajectory_cache_size)

    def __len__(self) -> int:
        return len(self.manifest.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.manifest.records[index]
        return {
            "manifest_row": int(index),
            "image": self._images.image(record),
            "record": record,
        }


class OpenFlyNative1TStage1Dataset(Dataset[dict[str, Any]]):
    """OpenFly Stage1: I0 + sampled prior frames + It, with nine fixed slots."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        stage1_context_window: int = 16,
        seed: int = 7,
        trajectory_cache_size: int = 4,
    ) -> None:
        if stage1_context_window < 7:
            raise ValueError("Stage1 context window must be at least seven frames")
        self.manifest = load_openfly_native1t_manifest(manifest_path)
        self.context_window = int(stage1_context_window)
        self.seed = int(seed)
        self._images = _ParquetImageStore(trajectory_cache_size)

    def __len__(self) -> int:
        return len(self.manifest.records)

    def __getitem__(self, item: int) -> dict[str, Any]:
        record = self.manifest.records[item]
        episode = self.manifest.episode_rows[record.episode_id]
        if episode[record.episode_position] != item:
            raise RuntimeError("OpenFly manifest episode index drift")
        candidate_positions = list(
            range(max(0, record.episode_position - self.context_window), record.episode_position)
        )
        rng = random.Random(self.seed + 1_000_003 * (item + 1))
        selected_positions = sorted(
            rng.sample(candidate_positions, min(7, len(candidate_positions)))
        )
        selected_rows = [episode[position] for position in selected_positions]
        first_row = episode[0]
        padding = 7 - len(selected_rows)
        memory_rows = [first_row] + [first_row] * padding + selected_rows + [item]
        valid = [True] + [False] * padding + [True] * len(selected_rows) + [True]
        if len(memory_rows) != 9 or len(valid) != 9:
            raise RuntimeError("OpenFly Native1T Stage1 did not produce nine frame slots")
        if any(source > item for source in memory_rows):
            raise RuntimeError("OpenFly Native1T Stage1 leaked a future row")
        return {
            "memory_images": [
                self._images.image(self.manifest.records[row]) for row in memory_rows
            ],
            "memory_valid_mask": torch.tensor(valid, dtype=torch.bool),
            "memory_source_manifest_rows": memory_rows,
            "instruction": record.instruction,
            "action_id": int(record.action_id),
            "source_action_id": int(record.source_action_id),
            "episode_id": record.episode_id,
            "step_id": int(record.step_id),
        }


def _load_cache_metadata(cache_dir: str | Path) -> tuple[Path, dict[str, Any], Path]:
    directory = Path(cache_dir).expanduser().resolve()
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"OpenFly Native1T cache metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("OpenFly Native1T cache metadata must be an object")
    if metadata.get("format") != CACHE_FORMAT or not metadata.get("complete"):
        raise ValueError("OpenFly Native1T cache is incomplete or has the wrong format")
    return directory, metadata, metadata_path


class OpenFlyNative1TStage2Dataset(Dataset[dict[str, Any]]):
    """OpenFly Stage2: current RGB plus strictly past H8 cache rows."""

    def __init__(
        self,
        manifest_path: str | Path,
        cache_dir: str | Path,
        stage1_checkpoint: str | Path,
        *,
        history_length: int = 8,
        trajectory_cache_size: int = 4,
    ) -> None:
        if history_length != 8:
            raise ValueError("The minimal OpenFly Native1T pipeline currently implements FIFO H8 only")
        self.manifest = load_openfly_native1t_manifest(manifest_path)
        self.cache_dir, self.metadata, metadata_path = _load_cache_metadata(cache_dir)
        self.history_length = int(history_length)
        self.feature_dim = int(self.metadata.get("feature_dim", -1))
        expected_shape = [len(self.manifest.records), self.feature_dim]
        if self.feature_dim <= 0 or self.metadata.get("shape") != expected_shape:
            raise ValueError(
                f"OpenFly Native1T cache shape mismatch: {self.metadata.get('shape')}, "
                f"expected {expected_shape}"
            )
        if self.metadata.get("manifest_sha256") != self.manifest.manifest_sha256:
            raise ValueError("OpenFly Native1T cache manifest SHA-256 mismatch")
        if self.metadata.get("ordered_row_identity_sha256") != self.manifest.row_identity_sha256:
            raise ValueError("OpenFly Native1T cache row-identity digest mismatch")
        if self.metadata.get("action_policy") != self.manifest.action_policy:
            raise ValueError("OpenFly Native1T cache action-policy mismatch")
        if int(self.metadata.get("num_actions", -1)) != self.manifest.num_actions:
            raise ValueError("OpenFly Native1T cache action-dimension mismatch")
        expected_action_head = {
            "openfly_action_token_dim": 8,
            "openfly_action_token_bins": 256,
        }
        action_head_mismatches = {
            key: (self.metadata.get(key), expected)
            for key, expected in expected_action_head.items()
            if self.metadata.get(key) != expected
        }
        if self.metadata.get("openfly_action_head_type") not in {
            "action_tokens_v1", "action_tokens_ar_v2"
        }:
            action_head_mismatches["openfly_action_head_type"] = (
                self.metadata.get("openfly_action_head_type"),
                "action_tokens_v1|action_tokens_ar_v2",
            )
        if action_head_mismatches or not self.metadata.get("openfly_action_codec_sha256"):
            raise ValueError(
                "OpenFly Native1T cache action-token contract mismatch: "
                f"{action_head_mismatches}"
            )
        if self.metadata.get("feature_space") != "frozen_memory_tokenizer_pre_projector":
            raise ValueError("OpenFly Native1T cache feature-space mismatch")
        if self.metadata.get("dtype") != "float16":
            raise ValueError("OpenFly Native1T cache dtype must be float16")
        stage1_file = Path(stage1_checkpoint).expanduser().resolve() / "native1t_postfusion.pt"
        if not stage1_file.is_file():
            raise FileNotFoundError(f"OpenFly Native1T Stage1 tokenizer is missing: {stage1_file}")
        if self.metadata.get("tokenizer_sha256") != sha256_file(stage1_file):
            raise ValueError("OpenFly Native1T cache Stage1 tokenizer SHA-256 mismatch")
        validation_path = self.cache_dir / "validation.json"
        if not validation_path.is_file():
            raise FileNotFoundError(
                "OpenFly Native1T cache is not validation-gated; missing validation.json"
            )
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if (
            not isinstance(validation, dict)
            or validation.get("format") != CACHE_VALIDATION_FORMAT
            or not validation.get("complete")
            or validation.get("manifest_sha256") != self.manifest.manifest_sha256
            or validation.get("ordered_row_identity_sha256") != self.manifest.row_identity_sha256
            or validation.get("cache_metadata_sha256") != sha256_file(metadata_path)
            or validation.get("tokenizer_sha256") != sha256_file(stage1_file)
            or validation.get("openfly_action_codec_sha256")
            != self.metadata.get("openfly_action_codec_sha256")
            or int(validation.get("feature_dim", -1)) != self.feature_dim
            or int(validation.get("parity_samples", 0)) <= 0
        ):
            raise ValueError("OpenFly Native1T cache validation report is incompatible")
        self._features: np.ndarray | None = None
        self._row_identity_hashes: np.ndarray | None = None
        if self.metadata.get("features_file") != "features.npy":
            raise ValueError("OpenFly Native1T cache features file name mismatch")
        if self.metadata.get("row_identity_file") != "row_identity_sha256.npy":
            raise ValueError("OpenFly Native1T cache row-identity file name mismatch")
        if self.metadata.get("row_identity_file_sha256") != sha256_file(
            self.cache_dir / "row_identity_sha256.npy"
        ):
            raise ValueError("OpenFly Native1T cache row-identity file SHA-256 mismatch")
        self._verify_row_identity_array()
        _ = self.features
        self._images = _ParquetImageStore(trajectory_cache_size)

    @property
    def features(self) -> np.ndarray:
        if self._features is None:
            features = np.load(self.cache_dir / "features.npy", allow_pickle=False)
            if list(features.shape) != [len(self.manifest.records), self.feature_dim]:
                raise ValueError("OpenFly Native1T feature file shape mismatch")
            if features.dtype != np.float16:
                raise ValueError("OpenFly Native1T feature file dtype mismatch")
            features.setflags(write=False)
            self._features = features
        return self._features

    @property
    def row_identity_hashes(self) -> np.ndarray:
        if self._row_identity_hashes is None:
            hashes = np.load(self.cache_dir / "row_identity_sha256.npy", mmap_mode="r")
            if hashes.dtype != np.uint8 or list(hashes.shape) != [len(self.manifest.records), 32]:
                raise ValueError("OpenFly Native1T row identity array shape/dtype mismatch")
            self._row_identity_hashes = hashes
        return self._row_identity_hashes

    def _verify_row_identity_array(self) -> None:
        hashes = self.row_identity_hashes
        for start in range(0, len(self.manifest.records), 8192):
            stop = min(start + 8192, len(self.manifest.records))
            expected = np.frombuffer(
                b"".join(record.identity_bytes for record in self.manifest.records[start:stop]),
                dtype=np.uint8,
            ).reshape(stop - start, 32)
            if not np.array_equal(hashes[start:stop], expected):
                raise ValueError(
                    f"OpenFly Native1T cache identity mapping is not one-to-one at rows {start}:{stop}"
                )

    def __len__(self) -> int:
        return len(self.manifest.records)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Keep the read-only RAM array so forked workers inherit/share its pages.
        state["_features"] = self._features
        state["_row_identity_hashes"] = None
        return state

    def __getitem__(self, item: int) -> dict[str, Any]:
        record = self.manifest.records[item]
        episode = self.manifest.episode_rows[record.episode_id]
        if episode[record.episode_position] != item:
            raise RuntimeError("OpenFly manifest episode index drift")
        history = np.zeros((self.history_length, self.feature_dim), dtype=np.float16)
        valid = np.zeros(self.history_length, dtype=np.bool_)
        source_rows = [-1] * self.history_length
        first_position = record.episode_position - self.history_length
        for slot, position in enumerate(range(first_position, record.episode_position)):
            if position < 0:
                continue
            source = episode[position]
            if source >= item:
                raise RuntimeError("OpenFly Native1T Stage2 leaked current/future cache row")
            history[slot] = self.features[source]
            valid[slot] = True
            source_rows[slot] = source
        return {
            "image": self._images.image(record),
            "memory_features": torch.from_numpy(history),
            "memory_valid_mask": torch.from_numpy(valid),
            "memory_source_manifest_rows": source_rows,
            "instruction": record.instruction,
            "action_id": int(record.action_id),
            "source_action_id": int(record.source_action_id),
            "episode_id": record.episode_id,
            "step_id": int(record.step_id),
        }


def cache_metadata(cache_dir: str | Path) -> dict[str, Any]:
    """Read basic cache metadata for preflight before a model is allocated."""
    _, metadata, _ = _load_cache_metadata(cache_dir)
    return metadata
