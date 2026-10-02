"""Single-frame Qwen3.5 dataset with episode-relative state text."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from aerial_openvla.data.manifest import load_step_records
from aerial_openvla.data.multiframe_dataset import update_relative_state
from aerial_openvla.data.qwen35_relative_state import load_qwen35_relative_state_cache


class Qwen35RelativeClassificationDataset(Dataset[dict]):
    """Attach the state immediately before each action to a single RGB frame."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.records = load_step_records(self.manifest_path)
        self.relative_states = self._build_relative_states()

    def _build_relative_states(self) -> list[tuple[int, int]]:
        states: list[tuple[int, int]] = []
        episode_state: dict[str, tuple[int, int]] = {}
        previous_step: dict[str, int] = {}
        for record in self.records:
            expected_step = previous_step.get(record.episode_id, record.step_id - 1) + 1
            if record.step_id != expected_step:
                raise ValueError(
                    "Relative Qwen3.5 requires contiguous, episode-ordered records; "
                    f"got {record.episode_id} step {record.step_id} after "
                    f"{previous_step.get(record.episode_id)}"
                )
            heading, altitude = episode_state.get(record.episode_id, (0, 0))
            states.append((heading, altitude))
            episode_state[record.episode_id] = update_relative_state(
                heading, altitude, int(record.action_id)
            )
            previous_step[record.episode_id] = record.step_id
        return states

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
        heading, altitude = self.relative_states[index]
        return {
            "image": image,
            "instruction": record.instruction,
            "relative_heading_degrees": heading,
            "relative_altitude_meters": altitude,
            "action_id": int(record.action_id),
            "episode_id": record.episode_id,
            "step_id": record.step_id,
            "image_path": str(image_path),
        }


class Qwen35RelativeAppendPlanarClassificationDataset(Dataset[dict]):
    """Current-frame Qwen input plus immutable pre-action planar state.

    The cache is generated from ground-truth actions *before* the labeled
    action.  Loading it rather than replaying actions in the Dataset keeps the
    training timing and manifest identity explicit and auditable.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        relative_state_cache_dir: str | Path,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.records = load_step_records(self.manifest_path)
        self.relative_states, self.relative_state_metadata = (
            load_qwen35_relative_state_cache(
                relative_state_cache_dir,
                self.manifest_path,
                len(self.records),
            )
        )

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
        heading, altitude, forward, left = self.relative_states[index].tolist()
        return {
            "image": image,
            "instruction": record.instruction,
            "relative_heading_degrees": float(heading),
            "relative_altitude_meters": float(altitude),
            "relative_forward_meters": float(forward),
            "relative_left_meters": float(left),
            "action_id": int(record.action_id),
            "episode_id": record.episode_id,
            "step_id": record.step_id,
            "image_path": str(image_path),
        }
