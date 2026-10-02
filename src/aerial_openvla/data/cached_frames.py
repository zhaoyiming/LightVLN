"""Dataset adapters for RGB frames captured from AirVLN rollouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


class CachedFrameDataset(Dataset[dict[str, Any]]):
    """Load cached RGB frames and their ground-truth action metadata."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        metadata_path = self.cache_dir / "metadata.jsonl"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"metadata.jsonl not found: {metadata_path}")

        self.samples: list[dict[str, Any]] = []
        with metadata_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                self.samples.append(json.loads(line))
        if not self.samples:
            raise ValueError(f"No cached samples in {metadata_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image_path = self.cache_dir / sample["rgb"]
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to load cached image {image_path}") from exc
        return {
            "image": image,
            "instruction": sample.get("instruction", ""),
            "action_id": sample["gt_action"],
            "episode_id": sample["episode_id"],
            "step_id": sample["step_id"],
            "image_path": str(image_path),
            "scene_id": sample.get("scene_id", ""),
            "trajectory_id": sample.get("trajectory_id", ""),
        }


class CachedFrameCollator:
    """Delegate cached-frame batching to the selected model policy."""

    def __init__(self, policy: Any) -> None:
        self._policy = policy

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        collator = getattr(self._policy, "collator", None)
        if collator is None:
            raise RuntimeError("The selected policy does not expose a collator")
        return collator(examples)
