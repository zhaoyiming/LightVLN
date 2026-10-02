"""OmTrack datasets: cached sparse history frames plus one current RGB."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from aerial_openvla.data.manifest import load_step_records
from aerial_openvla.data.schema import ClassificationRecord


class OmTrackDataset(Dataset[dict[str, Any]]):
    """Read episode-local history from a manifest-aligned FP16 memmap cache."""

    def __init__(
        self,
        manifest_path: str | Path,
        sparse_cache_dir: str | Path,
        history_length: int = 31,
        history_tokens_per_frame: int = 4,
        feature_dim: int = 2176,
        action_chunk_horizon: int = 8,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.cache_dir = Path(sparse_cache_dir).resolve()
        self.history_length = history_length
        self.history_tokens_per_frame = history_tokens_per_frame
        self.feature_dim = feature_dim
        self.action_chunk_horizon = action_chunk_horizon
        if history_length not in {16, 31}:
            raise ValueError("OmTrack history_length must be 16 or 31")
        if history_tokens_per_frame != 4 or action_chunk_horizon != 8:
            raise ValueError("OmTrack fixes sparse tokens=4 and chunk horizon=8")
        self.records: list[ClassificationRecord] = load_step_records(self.manifest_path)
        metadata_path = self.cache_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing OmTrack cache metadata: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        expected_shape = [len(self.records), history_tokens_per_frame, feature_dim]
        if metadata.get("shape") != expected_shape or not metadata.get("complete", False):
            raise ValueError(
                f"Incomplete or incompatible OmTrack cache: expected {expected_shape}, "
                f"metadata={metadata}"
            )
        cached_manifest = Path(metadata["manifest_path"]).resolve()
        if cached_manifest != self.manifest_path:
            raise ValueError(
                f"Cache belongs to {cached_manifest}, not {self.manifest_path}"
            )
        self._features: np.ndarray | None = None

        self._episode_by_step: dict[str, dict[int, int]] = {}
        self._episode_order: dict[str, list[int]] = {}
        self._position: dict[int, int] = {}
        for index, record in enumerate(self.records):
            by_step = self._episode_by_step.setdefault(record.episode_id, {})
            if record.step_id in by_step:
                raise ValueError(
                    f"Duplicate step {record.step_id} in episode {record.episode_id}"
                )
            by_step[record.step_id] = index
        for episode_id, by_step in self._episode_by_step.items():
            ordered = [by_step[step] for step in sorted(by_step)]
            self._episode_order[episode_id] = ordered
            for position, index in enumerate(ordered):
                self._position[index] = position

    def __len__(self) -> int:
        return len(self.records)

    @property
    def features(self) -> np.ndarray:
        if self._features is None:
            self._features = np.load(self.cache_dir / "features.npy", mmap_mode="r")
        return self._features

    def _chunk_targets(self, index: int, record: ClassificationRecord) -> list[int]:
        ordered = self._episode_order[record.episode_id]
        position = self._position[index]
        targets = [-100] * self.action_chunk_horizon
        for offset in range(self.action_chunk_horizon):
            future_position = position + offset
            if future_position >= len(ordered):
                break
            future = self.records[ordered[future_position]]
            if future.step_id != record.step_id + offset:
                break
            targets[offset] = int(future.action_id)
            if future.action_id == 0:
                break
        if targets[0] != record.action_id:
            raise RuntimeError("OmTrack target slot 0 disagrees with current action")
        return targets

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        by_step = self._episode_by_step[record.episode_id]
        history = np.zeros(
            (self.history_length, self.history_tokens_per_frame, self.feature_dim),
            dtype=np.float16,
        )
        valid = np.zeros(self.history_length, dtype=np.bool_)
        for slot, target_step in enumerate(
            range(record.step_id - self.history_length, record.step_id)
        ):
            cache_index = by_step.get(target_step)
            if cache_index is not None:
                history[slot] = self.features[cache_index]
                valid[slot] = True
        image_path = record.resolve_image(self.manifest_path)
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to load current OmTrack image {image_path}") from exc
        return {
            "image": image,
            "omtrack_history_features": torch.from_numpy(history),
            "omtrack_history_valid_mask": torch.from_numpy(valid),
            "instruction": record.instruction,
            "action_id": int(record.action_id),
            "action_chunk_targets": self._chunk_targets(index, record),
            "episode_id": record.episode_id,
            "step_id": record.step_id,
            "image_path": str(image_path),
        }


class OmTrackCollator:
    def __init__(self, processor: Any, prompt_template: str, max_text_tokens: int = 256) -> None:
        self.processor = processor
        self.prompt_template = prompt_template
        self.max_text_tokens = max_text_tokens

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        image_processor = getattr(self.processor, "image_processor", self.processor)
        pixels = image_processor([item["image"] for item in examples], return_tensors="pt")
        prompts = [
            self.prompt_template.format(instruction=item["instruction"].strip())
            for item in examples
        ]
        text = self.processor.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_text_tokens,
            return_tensors="pt",
        )
        return {
            "input_ids": text["input_ids"],
            "attention_mask": text.get(
                "attention_mask", torch.ones_like(text["input_ids"])
            ),
            "pixel_values": pixels["pixel_values"],
            "omtrack_history_features": torch.stack(
                [item["omtrack_history_features"] for item in examples]
            ),
            "omtrack_history_valid_mask": torch.stack(
                [item["omtrack_history_valid_mask"] for item in examples]
            ),
            "action_id": torch.tensor([item["action_id"] for item in examples]),
            "action_chunk_targets": torch.tensor(
                [item["action_chunk_targets"] for item in examples], dtype=torch.long
            ),
            "episode_id": [item["episode_id"] for item in examples],
            "step_id": [item["step_id"] for item in examples],
            "image_path": [item["image_path"] for item in examples],
        }
