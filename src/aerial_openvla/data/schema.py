from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aerial_openvla.core.actions import validate_action_id


@dataclass(frozen=True, slots=True)
class ClassificationRecord:
    episode_id: str
    trajectory_id: str
    step_id: int
    image: str
    instruction: str
    action_id: int
    split: str
    scene_id: int | str | None = None
    prev_action_id: int | None = None
    progress_label: float | None = None

    def __post_init__(self) -> None:
        if self.step_id < 0:
            raise ValueError("step_id must be non-negative")
        validate_action_id(self.action_id)
        if self.prev_action_id is not None and not 0 <= self.prev_action_id <= 8:
            raise ValueError("prev_action_id must be an action ID or the start token (8)")
        if self.progress_label is not None and not 0.0 <= self.progress_label <= 1.0:
            raise ValueError("progress_label must be in [0, 1]")
        if not self.instruction.strip():
            raise ValueError("instruction must not be empty")
        if not self.image.strip():
            raise ValueError("image path must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ClassificationRecord":
        return cls(
            episode_id=str(raw["episode_id"]),
            trajectory_id=str(raw.get("trajectory_id", raw["episode_id"])),
            step_id=int(raw["step_id"]),
            image=str(raw["image"]),
            instruction=str(raw["instruction"]),
            action_id=int(raw["action_id"]),
            split=str(raw.get("split", "unknown")),
            scene_id=raw.get("scene_id"),
            prev_action_id=(
                int(raw["prev_action_id"]) if raw.get("prev_action_id") is not None else None
            ),
            progress_label=(
                float(raw["progress_label"])
                if raw.get("progress_label") is not None
                else (float(raw["progress"]) if raw.get("progress") is not None else None)
            ),
        )

    def resolve_image(self, manifest_path: Path) -> Path:
        image_path = Path(self.image)
        if image_path.is_absolute():
            return image_path
        return (manifest_path.parent / image_path).resolve()
