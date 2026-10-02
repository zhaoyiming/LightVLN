"""Manifest-to-cache datasets for Qwen3.5 NativeMEM v2."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from aerial_openvla.data.manifest import load_step_records
from aerial_openvla.data.qwen35_relative_state import (
    QWEN35_RELATIVE_NONE,
    load_qwen35_relative_state_cache,
    validate_qwen35_relative_state_mode,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _EpisodeIndex:
    def __init__(self, records) -> None:
        self.by_episode: dict[str, dict[int, int]] = {}
        self.first_by_episode: dict[str, int] = {}
        for index, record in enumerate(records):
            steps = self.by_episode.setdefault(record.episode_id, {})
            if record.step_id in steps:
                raise ValueError(
                    f"Duplicate episode/step in NativeMEM manifest: "
                    f"{record.episode_id}/{record.step_id}"
                )
            steps[int(record.step_id)] = index
        for episode_id, steps in self.by_episode.items():
            self.first_by_episode[episode_id] = steps[min(steps)]


class Qwen35NativeMEMV2Dataset(Dataset[dict[str, Any]]):
    """Read pre-merger F directly from the deduplicated cache.

    Stage 1 returns exactly ``I0 + seven sampled previous frames + It`` (nine
    F slots, with an explicit validity mask).  Stage 2 returns the strict
    causal ``[t-16,...,t-1] + It`` layout (17 F slots; the mask has 16 slots).
    No RGB file is opened by either training stage.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        cache_dir: str | Path,
        stage: int,
        history_length: int = 16,
        stage1_frames: int = 9,
        seed: int = 7,
        memory_cache_dir: str | Path | None = None,
        single_stage: bool = False,
        relative_state_mode: str = QWEN35_RELATIVE_NONE,
        relative_state_cache_dir: str | Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.cache_dir = Path(cache_dir).resolve()
        self.memory_cache_dir = (
            Path(memory_cache_dir).resolve() if memory_cache_dir else None
        )
        self.stage = int(stage)
        self.history_length = int(history_length)
        self.stage1_frames = int(stage1_frames)
        self.seed = int(seed)
        self.single_stage = bool(single_stage)
        self.relative_state_mode = validate_qwen35_relative_state_mode(
            relative_state_mode
        )
        if self.stage not in (1, 2):
            raise ValueError("NativeMEM v2 stage must be 1 or 2")
        if self.history_length != 16 or self.stage1_frames != 9:
            raise ValueError("NativeMEM v2 fixes H16 and nine Stage1 slots")
        if self.single_stage and self.stage != 2:
            raise ValueError("NativeMEM single-stage V2 requires strict causal H16 (stage=2)")

        self.records = load_step_records(self.manifest_path)
        self.relative_states = None
        self.relative_state_metadata = None
        if self.relative_state_mode != QWEN35_RELATIVE_NONE:
            if relative_state_cache_dir is None:
                raise ValueError(
                    "Qwen3.5 append-planar Relative requires "
                    "qwen35_relative_state_cache"
                )
            self.relative_states, self.relative_state_metadata = (
                load_qwen35_relative_state_cache(
                    relative_state_cache_dir,
                    self.manifest_path,
                    len(self.records),
                )
            )
        self.index = _EpisodeIndex(self.records)
        expected_manifest_sha = _sha256(self.manifest_path)
        cache_meta_path = self.cache_dir / "cache_meta.json"
        index_meta_path = self.cache_dir / "index_meta.json"
        if not cache_meta_path.is_file() or not index_meta_path.is_file():
            raise FileNotFoundError("NativeMEM cache must contain cache_meta.json and index_meta.json")
        self.cache_meta = json.loads(cache_meta_path.read_text())
        self.index_meta = json.loads(index_meta_path.read_text())
        if not self.cache_meta.get("complete") or not self.index_meta.get("complete"):
            raise ValueError("NativeMEM pre-merger cache/index is incomplete")
        if self.cache_meta.get("cache_type") != "qwen35_vision_pre_merger":
            raise ValueError("NativeMEM v2 requires qwen35_vision_pre_merger cache")
        if self.cache_meta.get("manifest_sha256") != expected_manifest_sha:
            raise ValueError("NativeMEM cache manifest hash mismatch")
        if self.index_meta.get("manifest_sha256") != expected_manifest_sha:
            raise ValueError("NativeMEM cache index manifest hash mismatch")
        shape = self.cache_meta.get("shape")
        expected_shape = [int(self.index_meta.get("cache_rows", -1)), 196, 768]
        if shape != expected_shape:
            raise ValueError(f"NativeMEM cache shape must be {expected_shape}, got {shape}")
        if self.index_meta.get("manifest_rows") != len(self.records):
            raise ValueError("NativeMEM mapping row count does not match manifest")
        if self.stage == 2 and not self.single_stage and self.memory_cache_dir is None:
            raise ValueError(
                "NativeMEM Stage2 requires a checkpoint-matched [N,1,768] "
                "Native1T cache"
            )
        self.memory_meta = None
        self._memory_features = None
        if self.memory_cache_dir is not None:
            memory_meta_path = self.memory_cache_dir / "metadata.json"
            if not memory_meta_path.is_file():
                raise FileNotFoundError(
                    f"NativeMEM 1T cache metadata missing: {memory_meta_path}"
                )
            self.memory_meta = json.loads(memory_meta_path.read_text())
            if not self.memory_meta.get("complete"):
                raise ValueError("NativeMEM 1T cache is incomplete")
            if self.memory_meta.get("cache_type") != "qwen35_nativemem_v2_1t":
                raise ValueError("NativeMEM 1T cache has the wrong cache_type")
            if self.memory_meta.get("manifest_sha256") != expected_manifest_sha:
                raise ValueError("NativeMEM 1T cache manifest hash mismatch")
            expected_memory_shape = [
                int(self.index_meta.get("cache_rows", -1)), 768
            ]
            if self.memory_meta.get("shape") != expected_memory_shape:
                raise ValueError(
                    "NativeMEM 1T cache shape must be "
                    f"{expected_memory_shape}, got {self.memory_meta.get('shape')}"
                )
            if self.memory_meta.get("source_checkpoint") is None:
                raise ValueError("NativeMEM 1T cache lacks source_checkpoint identity")
        self._features = None
        self._mapping = None

    @property
    def features(self) -> np.ndarray:
        if self._features is None:
            self._features = np.load(self.cache_dir / "features.npy", mmap_mode="r")
            if self._features.dtype != np.float16 or tuple(self._features.shape[1:]) != (196, 768):
                raise ValueError(f"NativeMEM feature array has wrong dtype/shape: {self._features.dtype} {self._features.shape}")
        return self._features

    @property
    def mapping(self) -> np.ndarray:
        if self._mapping is None:
            self._mapping = np.load(self.cache_dir / "manifest_to_cache.npy", mmap_mode="r")
            if self._mapping.ndim != 1 or self._mapping.shape[0] != len(self.records):
                raise ValueError("NativeMEM manifest_to_cache mapping shape mismatch")
            if int(self._mapping.min()) < 0 or int(self._mapping.max()) >= self.features.shape[0]:
                raise ValueError("NativeMEM manifest_to_cache contains out-of-range rows")
        return self._mapping

    @property
    def memory_features(self) -> np.ndarray:
        if self.memory_cache_dir is None:
            raise RuntimeError("NativeMEM 1T cache is not configured")
        if self._memory_features is None:
            self._memory_features = np.load(
                self.memory_cache_dir / "features.npy", mmap_mode="r"
            )
            if (
                self._memory_features.dtype != np.float16
                or tuple(self._memory_features.shape[1:]) != (768,)
            ):
                raise ValueError(
                    "NativeMEM 1T feature array has wrong dtype/shape: "
                    f"{self._memory_features.dtype} {self._memory_features.shape}"
                )
        return self._memory_features

    def __len__(self) -> int:
        return len(self.records)

    def _gather(self, rows: list[int | None]) -> torch.Tensor:
        result = np.zeros((len(rows), 196, 768), dtype=np.float16)
        valid_positions = [slot for slot, row in enumerate(rows) if row is not None]
        if valid_positions:
            cache_rows = np.asarray([self.mapping[int(rows[slot])] for slot in valid_positions], dtype=np.int64)
            result[np.asarray(valid_positions)] = self.features[cache_rows]
        return torch.from_numpy(result)

    def _gather_memory(self, rows: list[int | None]) -> torch.Tensor:
        result = np.zeros((len(rows), 768), dtype=np.float16)
        valid_positions = [slot for slot, row in enumerate(rows) if row is not None]
        if valid_positions:
            cache_rows = np.asarray(
                [self.mapping[int(rows[slot])] for slot in valid_positions],
                dtype=np.int64,
            )
            result[np.asarray(valid_positions)] = self.memory_features[cache_rows]
        return torch.from_numpy(result)

    def __getitem__(self, item: int) -> dict[str, Any]:
        record = self.records[item]
        steps = self.index.by_episode[record.episode_id]
        if self.stage == 1:
            previous = [
                steps[step]
                for step in range(int(record.step_id) - 16, int(record.step_id))
                if step in steps
            ]
            rng = random.Random(self.seed + 1000003 * int(item))
            selected = sorted(rng.sample(previous, min(7, len(previous))))
            padding = 7 - len(selected)
            first = self.index.first_by_episode[record.episode_id]
            rows: list[int | None] = [first] + [None] * padding + selected + [item]
            valid = [True] + [False] * padding + [True] * len(selected) + [True]
            if len(rows) != 9 or len(valid) != 9:
                raise RuntimeError("NativeMEM Stage1 must produce nine F slots")
            memory_mask = torch.tensor(valid, dtype=torch.bool)
        else:
            history_rows = [
                steps.get(step)
                for step in range(int(record.step_id) - self.history_length, int(record.step_id))
            ]
            for source in history_rows:
                if source is not None:
                    source_record = self.records[source]
                    if source_record.episode_id != record.episode_id or source_record.step_id >= record.step_id:
                        raise RuntimeError("NativeMEM Stage2 history violates episode-local causality")
            rows = history_rows + [item]
            memory_mask = torch.tensor([source is not None for source in history_rows], dtype=torch.bool)
        output = {
            "nativemem_premerger_features": self._gather(rows),
            "nativemem_history_valid_mask": memory_mask,
            "instruction": record.instruction,
            "action_id": int(record.action_id),
            "episode_id": record.episode_id,
            "trajectory_id": record.trajectory_id,
            "step_id": int(record.step_id),
            "current_visual_cache_row": int(self.mapping[item]),
        }
        if self.relative_states is not None:
            heading, altitude, forward, left = self.relative_states[item]
            output.update(
                relative_heading_degrees=float(heading),
                relative_altitude_meters=float(altitude),
                relative_forward_meters=float(forward),
                relative_left_meters=float(left),
            )
        if self.stage == 2 and not self.single_stage:
            # The current frame remains in the pre-merger cache for the
            # trainable Native Merger.  History uses only the matched,
            # frozen-tokenizer 1T latents, strictly t-16..t-1.
            output["nativemem_history_latents"] = self._gather_memory(history_rows)
        return output
