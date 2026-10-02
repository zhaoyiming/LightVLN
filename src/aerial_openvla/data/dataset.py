from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from aerial_openvla.data.manifest import load_step_records
from aerial_openvla.data.schema import ClassificationRecord


class AerialVLNClassificationDataset(Dataset[dict]):
    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.records: list[ClassificationRecord] = load_step_records(self.manifest_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image_path = record.resolve_image(self.manifest_path)
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to load image {image_path}") from exc
        return {
            "image": image,
            "instruction": record.instruction,
            "action_id": record.action_id,
            "episode_id": record.episode_id,
            "step_id": record.step_id,
            "image_path": str(image_path),
        }
