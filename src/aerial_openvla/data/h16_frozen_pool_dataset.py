"""H16-1T-FrozenPool: cached frozen-space history plus current RGB."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from aerial_openvla.data.manifest import load_step_records


class H16FrozenPoolDataset(Dataset[dict[str, Any]]):
    def __init__(self, manifest_path, cache_dir, history_length: int = 16) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.cache_dir = Path(cache_dir).resolve()
        self.history_length = int(history_length)
        if self.history_length != 16:
            raise ValueError("H16-1T-FrozenPool fixes history_length=16")
        self.records = load_step_records(self.manifest_path)
        metadata = json.loads((self.cache_dir / "metadata.json").read_text())
        expected = [len(self.records), 2176]
        if metadata.get("shape") != expected or not metadata.get("complete", False):
            raise ValueError(f"Incomplete/incompatible H16 cache: {metadata}")
        if Path(metadata["manifest_path"]).resolve() != self.manifest_path:
            raise ValueError("H16 cache manifest mismatch")
        self._features = None
        self._by_episode: dict[str, dict[int, int]] = {}
        for index, record in enumerate(self.records):
            by_step = self._by_episode.setdefault(record.episode_id, {})
            if record.step_id in by_step:
                raise ValueError("Duplicate episode step")
            by_step[record.step_id] = index

    @property
    def features(self):
        if self._features is None:
            self._features = np.load(self.cache_dir / "features.npy", mmap_mode="r")
        return self._features

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        history = np.zeros((16, 2176), dtype=np.float16)
        valid = np.zeros(16, dtype=np.bool_)
        by_step = self._by_episode[record.episode_id]
        for slot, step in enumerate(range(record.step_id - 16, record.step_id)):
            cache_index = by_step.get(step)
            if cache_index is not None:
                history[slot] = self.features[cache_index]
                valid[slot] = True
        image_path = record.resolve_image(self.manifest_path)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        return {
            "image": image,
            "history_features": torch.from_numpy(history),
            "history_valid_mask": torch.from_numpy(valid),
            "instruction": record.instruction,
            "action_id": int(record.action_id),
            "episode_id": record.episode_id,
            "step_id": record.step_id,
            "image_path": str(image_path),
        }


class H16FrozenPoolCollator:
    def __init__(self, processor, prompt_template: str, max_text_tokens: int = 256):
        self.processor = processor
        self.prompt_template = prompt_template
        self.max_text_tokens = max_text_tokens

    def __call__(self, examples):
        image_processor = getattr(self.processor, "image_processor", self.processor)
        pixels = image_processor([e["image"] for e in examples], return_tensors="pt")
        prompts = [
            self.prompt_template.format(instruction=e["instruction"].strip())
            for e in examples
        ]
        text = self.processor.tokenizer(
            prompts, padding=True, truncation=True,
            max_length=self.max_text_tokens, return_tensors="pt",
        )
        return {
            "input_ids": text["input_ids"],
            "attention_mask": text.get("attention_mask", torch.ones_like(text["input_ids"])),
            "pixel_values": pixels["pixel_values"],
            "h16_frozen_pool_history_features": torch.stack([e["history_features"] for e in examples]),
            "h16_frozen_pool_history_valid_mask": torch.stack([e["history_valid_mask"] for e in examples]),
            "action_id": torch.tensor([e["action_id"] for e in examples], dtype=torch.long),
            "episode_id": [e["episode_id"] for e in examples],
            "step_id": [e["step_id"] for e in examples],
            "image_path": [e["image_path"] for e in examples],
        }
