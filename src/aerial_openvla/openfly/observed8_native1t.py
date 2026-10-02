#!/usr/bin/env python3
"""OpenFly observed8 Native1T Stage1 data adapter.

This module is intentionally separate from the action-token Native1T wrapper.
It feeds the existing legacy AirMiniVLA Native1T model a compact eight-way
categorical target while preserving the original OpenFly ID in each manifest
row for auditability.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image
from torch.utils.data import Dataset


FORMAT = "openfly_observed8_native1t_h16_manifest_v1"
COMPACT_TO_OPENFLY = (0, 1, 2, 3, 4, 5, 8, 9)


def _canonical_tfds_episode_id(value: Any) -> str:
    """Normalize zero-padded manifest IDs to the TFDS episode spelling."""
    text = str(value)
    prefix = "openfly-s0-"
    if text.startswith(prefix):
        suffix = text[len(prefix):]
        if suffix.isdigit():
            return f"{prefix}{int(suffix)}"
    return text


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_identity(row: Mapping[str, Any]) -> bytes:
    payload = {
        key: row[key]
        for key in (
            "episode_id",
            "trajectory_id",
            "step_id",
            "episode_position",
            "parquet_path",
            "image_id",
            "virtual_image_path",
            "instruction",
            "source_action_id",
            "boundary_mapped_action_id",
            "public_builder_action_id",
            "openfly_action_id",
            "action_id",
        )
    }
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class Record:
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
    boundary_mapped_action_id: int
    public_builder_action_id: int
    openfly_action_id: int
    action_id: int
    row_identity_sha256: str


class OpenFlyObserved8Native1TStage1Dataset(Dataset[dict[str, Any]]):
    """Return I0 + seven sampled past frames + It, with RGB loaded from Parquet."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        seed: int = 7,
        history_context_window: int = 16,
        trajectory_cache_size: int = 4,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.metadata_path = Path(f"{self.manifest_path}.meta.json")
        self.seed = int(seed)
        self.history_context_window = int(history_context_window)
        self.trajectory_cache_size = int(trajectory_cache_size)
        if self.history_context_window < 7:
            raise ValueError("Native1T Stage1 context window must be at least seven")
        if self.trajectory_cache_size <= 0:
            raise ValueError("trajectory_cache_size must be positive")
        if not self.manifest_path.is_file() or not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"manifest and metadata are required: {self.manifest_path}"
            )

        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format") != FORMAT or not metadata.get("complete"):
            raise ValueError("observed8 Native1T manifest metadata is incomplete")
        if metadata.get("manifest_path") != str(self.manifest_path):
            raise ValueError("manifest metadata path mismatch")
        if metadata.get("manifest_sha256") != sha256_file(self.manifest_path):
            raise ValueError("manifest SHA-256 mismatch")
        if metadata.get("action_policy") != "openfly_observed_8" or int(
            metadata.get("num_actions", -1)
        ) != 8:
            raise ValueError("manifest is not the observed8 categorical contract")
        compact_map = metadata.get("compact_to_openfly")
        if [compact_map[str(index)] for index in range(8)] != list(COMPACT_TO_OPENFLY):
            raise ValueError("manifest compact action map drifted")
        self.metadata = metadata
        self.records: list[Record] = []
        self._episode_ranges: dict[str, tuple[int, int]] = {}
        identity_digest = hashlib.sha256()
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if raw.get("format") != FORMAT:
                    raise ValueError(f"wrong manifest row format at line {line_number}")
                row = int(raw.get("manifest_row", -1))
                if row != len(self.records):
                    raise ValueError(f"manifest row numbering drift at line {line_number}")
                if int(raw.get("action_id", -1)) not in range(8):
                    raise ValueError(f"observed8 action out of range at row {row}")
                parquet_path = Path(str(raw.get("parquet_path", ""))).expanduser().resolve()
                if parquet_path.suffix != ".parquet" or not parquet_path.is_file():
                    raise FileNotFoundError(f"manifest row {row} Parquet is missing: {parquet_path}")
                expected_virtual = f"{parquet_path}#{raw.get('image_id')}"
                if raw.get("virtual_image_path") != expected_virtual:
                    raise ValueError(f"virtual image path mismatch at row {row}")
                expected_identity = hashlib.sha256(_canonical_identity(raw)).hexdigest()
                if raw.get("row_identity_sha256") != expected_identity:
                    raise ValueError(f"row identity mismatch at row {row}")
                record = Record(
                    manifest_row=row,
                    episode_id=str(raw["episode_id"]),
                    trajectory_id=str(raw["trajectory_id"]),
                    step_id=int(raw["step_id"]),
                    episode_position=int(raw["episode_position"]),
                    parquet_path=parquet_path,
                    image_id=str(raw["image_id"]),
                    virtual_image_path=str(raw["virtual_image_path"]),
                    instruction=str(raw["instruction"]),
                    source_action_id=int(raw["source_action_id"]),
                    boundary_mapped_action_id=int(raw["boundary_mapped_action_id"]),
                    public_builder_action_id=int(raw["public_builder_action_id"]),
                    openfly_action_id=int(raw["openfly_action_id"]),
                    action_id=int(raw["action_id"]),
                    row_identity_sha256=str(raw["row_identity_sha256"]),
                )
                if record.episode_id not in self._episode_ranges:
                    self._episode_ranges[record.episode_id] = (row, row + 1)
                else:
                    start, end = self._episode_ranges[record.episode_id]
                    if end != row:
                        raise ValueError(f"episode {record.episode_id} is not contiguous")
                    self._episode_ranges[record.episode_id] = (start, row + 1)
                expected_position = self._episode_ranges[record.episode_id][1] - self._episode_ranges[record.episode_id][0] - 1
                if record.step_id != record.episode_position or record.step_id != expected_position:
                    raise ValueError(f"episode position is not consecutive at row {row}")
                identity_digest.update(bytes.fromhex(record.row_identity_sha256))
                self.records.append(record)
        if len(self.records) != int(metadata.get("rows", -1)):
            raise ValueError("manifest row count does not match metadata")
        if len(self._episode_ranges) != int(metadata.get("episodes", -1)):
            raise ValueError("manifest episode count does not match metadata")
        if identity_digest.hexdigest() != metadata.get("ordered_row_identity_sha256"):
            raise ValueError("ordered row identity digest mismatch")
        self.num_actions = 8
        self._trajectory_cache: OrderedDict[Path, dict[str, bytes]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item: int) -> dict[str, Any]:
        if item < 0:
            item += len(self.records)
        if item < 0 or item >= len(self.records):
            raise IndexError(item)
        record = self.records[item]
        start, _ = self._episode_ranges[record.episode_id]
        candidates = list(range(max(start, item - self.history_context_window), item))
        rng = random.Random(self.seed + item)
        selected = sorted(rng.sample(candidates, min(7, len(candidates))))
        padding = 7 - len(selected)
        frame_indices = [start] + [start] * padding + selected + [item]
        valid = [True] + [False] * padding + [True] * len(selected) + [True]
        if len(frame_indices) != 9 or len(valid) != 9:
            raise RuntimeError("Stage1 must return exactly nine memory slots")
        return {
            "memory_images": [self._load_image(self.records[index]) for index in frame_indices],
            "memory_valid_mask": torch.tensor(valid, dtype=torch.bool),
            "memory_frame_indices": frame_indices,
            "instruction": record.instruction,
            "action_id": record.action_id,
            "episode_id": record.episode_id,
            "step_id": record.step_id,
            "source_action_id": record.source_action_id,
            "openfly_action_id": record.openfly_action_id,
        }

    def _load_image(self, record: Record) -> Image.Image:
        images = self._load_trajectory(record.parquet_path)
        try:
            payload = images[record.image_id]
        except KeyError as exc:
            raise KeyError(
                f"{record.parquet_path} has no image_id={record.image_id!r}"
            ) from exc
        with Image.open(io.BytesIO(payload)) as source:
            return source.convert("RGB")

    def _load_trajectory(self, parquet_path: Path) -> dict[str, bytes]:
        cached = self._trajectory_cache.pop(parquet_path, None)
        if cached is not None:
            self._trajectory_cache[parquet_path] = cached
            return cached
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Stage1 online Parquet input requires pyarrow") from exc
        table = pq.read_table(parquet_path, columns=["image_id", "image"], use_threads=False)
        images: dict[str, bytes] = {}
        for row in table.to_pylist():
            image = row.get("image")
            payload = image.get("bytes") if isinstance(image, Mapping) else None
            if payload:
                images[str(row["image_id"])] = bytes(payload)
        if not images:
            raise RuntimeError(f"Parquet contains no image bytes: {parquet_path}")
        self._trajectory_cache[parquet_path] = images
        while len(self._trajectory_cache) > self.trajectory_cache_size:
            self._trajectory_cache.popitem(last=False)
        return images

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_trajectory_cache"] = OrderedDict()
        return state


class OpenFlyObserved8Native1TStage1TFDSDataset(Dataset[dict[str, Any]]):
    """Return the Stage1 nine-frame contract from the retained OpenFly TFDS.

    The TFDS export stores one complete episode per TFRecord entry.  This
    adapter keeps the immutable Native1T manifest as the row/action contract,
    while resolving all nine RGB slots from the matching TFDS episode.  No
    Parquet image bytes or feature cache are touched.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        tfds_dataset_dir: str | Path,
        *,
        seed: int = 7,
        history_context_window: int = 16,
        tfds_cache_episodes: int = 32,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.metadata_path = Path(f"{self.manifest_path}.meta.json")
        self.seed = int(seed)
        self.history_context_window = int(history_context_window)
        if self.history_context_window < 7:
            raise ValueError("Native1T Stage1 context window must be at least seven")
        if tfds_cache_episodes <= 0:
            raise ValueError("tfds_cache_episodes must be positive")
        if not self.manifest_path.is_file() or not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"manifest and metadata are required: {self.manifest_path}"
            )

        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format") != FORMAT or not metadata.get("complete"):
            raise ValueError("observed8 Native1T manifest metadata is incomplete")
        if metadata.get("manifest_path") != str(self.manifest_path):
            raise ValueError("manifest metadata path mismatch")
        if metadata.get("manifest_sha256") != sha256_file(self.manifest_path):
            raise ValueError("manifest SHA-256 mismatch")
        if metadata.get("action_policy") != "openfly_observed_8" or int(
            metadata.get("num_actions", -1)
        ) != 8:
            raise ValueError("manifest is not the observed8 categorical contract")
        compact_map = metadata.get("compact_to_openfly")
        if [compact_map[str(index)] for index in range(8)] != list(COMPACT_TO_OPENFLY):
            raise ValueError("manifest compact action map drifted")

        self.metadata = metadata
        self.records: list[dict[str, Any]] = []
        self._episode_ranges: dict[str, tuple[int, int]] = {}
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                row = int(raw.get("manifest_row", -1))
                if row != len(self.records):
                    raise ValueError(f"manifest row numbering drift at line {line_number}")
                if raw.get("format") != FORMAT:
                    raise ValueError(f"wrong manifest row format at line {line_number}")
                if int(raw.get("action_id", -1)) not in range(8):
                    raise ValueError(f"observed8 action out of range at row {row}")
                episode_id = str(raw["episode_id"])
                step_id = int(raw["step_id"])
                episode_position = int(raw["episode_position"])
                if step_id != episode_position:
                    raise ValueError(f"episode position mismatch at row {row}")
                if episode_id not in self._episode_ranges:
                    self._episode_ranges[episode_id] = (row, row + 1)
                else:
                    start, end = self._episode_ranges[episode_id]
                    if end != row:
                        raise ValueError(f"episode {episode_id} is not contiguous")
                    self._episode_ranges[episode_id] = (start, row + 1)
                start, end = self._episode_ranges[episode_id]
                if step_id != end - start - 1:
                    raise ValueError(f"episode step is not consecutive at row {row}")
                self.records.append({
                    "episode_id": episode_id,
                    "trajectory_id": str(raw["trajectory_id"]),
                    "step_id": step_id,
                    "episode_position": episode_position,
                    "image_id": str(raw["image_id"]),
                    "instruction": str(raw["instruction"]),
                    "source_action_id": int(raw["source_action_id"]),
                    "public_builder_action_id": int(raw["public_builder_action_id"]),
                    "openfly_action_id": int(raw["openfly_action_id"]),
                    "action_id": int(raw["action_id"]),
                })
        if len(self.records) != int(metadata.get("rows", -1)):
            raise ValueError("manifest row count does not match metadata")
        if len(self._episode_ranges) != int(metadata.get("episodes", -1)):
            raise ValueError("manifest episode count does not match metadata")

        source_tfds = metadata.get("source_tfds")
        if not isinstance(source_tfds, Mapping):
            raise ValueError("Native1T manifest has no source TFDS provenance")
        expected_tfds_dir = Path(str(source_tfds.get("tfds_dataset_dir", ""))).expanduser().resolve()
        self.tfds_dataset_dir = Path(tfds_dataset_dir).expanduser().resolve()
        if expected_tfds_dir != self.tfds_dataset_dir:
            raise ValueError(
                "TFDS directory disagrees with manifest provenance: "
                f"expected={expected_tfds_dir}, got={self.tfds_dataset_dir}"
            )
        if int(source_tfds.get("rows", -1)) != len(self.records):
            raise ValueError("source TFDS row count does not match Native1T manifest")
        if int(source_tfds.get("episodes", -1)) != len(self._episode_ranges):
            raise ValueError("source TFDS episode count does not match Native1T manifest")

        from aerial_openvla.openfly.tfds_current_frame import OpenFlyCurrentFrameTFDSDataset

        self._tfds = OpenFlyCurrentFrameTFDSDataset(
            self.tfds_dataset_dir,
            action_policy="openfly_10",
            num_frames=1,
            frame_history_stride=1,
            action_chunk_horizon=1,
            cache_episodes=int(tfds_cache_episodes),
        )
        if len(self._tfds) != len(self.records):
            raise ValueError(
                "TFDS row count does not match Native1T manifest: "
                f"tfds={len(self._tfds)}, manifest={len(self.records)}"
            )
        self._validate_alignment_probe()
        # Do not let the DataLoader workers inherit the probe's open handles or
        # its small decoded-episode cache when the platform uses fork workers.
        for handle in self._tfds._shard_handles.values():
            handle.close()
        self._tfds._shard_handles.clear()
        self._tfds._episode_cache.clear()
        self.num_actions = 8

    def __len__(self) -> int:
        return len(self.records)

    def _validate_alignment_probe(self) -> None:
        indices = sorted({0, len(self.records) // 2, len(self.records) - 1})
        for index in indices:
            record = self.records[index]
            sample = self._tfds[index]
            expected_action = int(record["public_builder_action_id"])
            if (
                _canonical_tfds_episode_id(sample["episode_id"])
                != _canonical_tfds_episode_id(record["episode_id"])
                or int(sample["step_id"]) != int(record["step_id"])
                or sample["instruction"] != record["instruction"]
                or int(sample["action_id"]) != expected_action
                or not str(sample["image_path"]).endswith(f"#{record['image_id']}")
            ):
                raise ValueError(
                    "TFDS/native1t row alignment probe failed at row "
                    f"{index}: tfds={sample['episode_id']}:{sample['step_id']}"
                )

    def __getitem__(self, item: int) -> dict[str, Any]:
        if item < 0:
            item += len(self.records)
        if item < 0 or item >= len(self.records):
            raise IndexError(item)
        record = self.records[item]
        start, _ = self._episode_ranges[record["episode_id"]]
        candidates = list(range(max(start, item - self.history_context_window), item))
        rng = random.Random(self.seed + item)
        selected = sorted(rng.sample(candidates, min(7, len(candidates))))
        padding = 7 - len(selected)
        frame_indices = [start] + [start] * padding + selected + [item]
        valid = [True] + [False] * padding + [True] * len(selected) + [True]
        if len(frame_indices) != 9 or len(valid) != 9:
            raise RuntimeError("Stage1 must return exactly nine memory slots")

        image_cache: dict[int, Image.Image] = {}
        for index in frame_indices:
            if index not in image_cache:
                sample = self._tfds[index]
                image_cache[index] = sample["image"]
        current = self._tfds[item]
        if (
            _canonical_tfds_episode_id(current["episode_id"])
            != _canonical_tfds_episode_id(record["episode_id"])
            or int(current["step_id"]) != int(record["step_id"])
            or current["instruction"] != record["instruction"]
            or int(current["action_id"]) != int(record["public_builder_action_id"])
        ):
            raise ValueError(f"TFDS/native1t row alignment failed at row {item}")
        return {
            "memory_images": [image_cache[index] for index in frame_indices],
            "memory_valid_mask": torch.tensor(valid, dtype=torch.bool),
            "memory_frame_indices": frame_indices,
            "instruction": record["instruction"],
            "action_id": record["action_id"],
            "episode_id": record["episode_id"],
            "step_id": record["step_id"],
            "source_action_id": record["source_action_id"],
            "openfly_action_id": record["openfly_action_id"],
        }
