"""Direct OpenFly Parquet input for the isolated legacy MiniVLA adapter.

OpenFly annotations contain one record per trajectory while frame bytes live in
``${OPENFLY_TRAJECTORY_ROOT}/${image_path}.parquet``.  This dataset keeps that
storage immutable: it reads RGB bytes on demand and never materializes a
second image dataset on disk.

OpenFly represents controls as an 8-D action vector.  Its execution codebook
has IDs 0..9, with three distinct forward distances: 3m (ID 1), 6m (ID 8),
and 9m (ID 9).  The OpenFly action-token head consumes this native codebook;
it is not a ten-way softmax.  The default ``openfly_10`` policy preserves all
codebook IDs:

``openfly_10``
    Identity mapping for action IDs 0..9.  Use this with the OpenFly 8x256
    action-token head, whose logical command codebook remains 0..9.

``strict_8``
    Legacy compatibility only: reject an annotation that contains actions 8
    or 9.
``collapse_forward``
    Legacy compatibility only: map 8 and 9 to action 1.  This discards
    forward-distance granularity and is deliberately opt-in.
"""

from __future__ import annotations

import io
import json
from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from aerial_openvla.openfly.action_codec import (
    OPENFLY_ACTION_NAMES,
    OPENFLY_SENTINEL_ACTION_IDS,
    OPENFLY_VALID_ACTION_IDS,
)


_ACTION_POLICIES: dict[str, dict[int, int]] = {
    "openfly_10": {action_id: action_id for action_id in range(10)},
    "strict_8": {action_id: action_id for action_id in range(8)},
    "collapse_forward": {
        **{action_id: action_id for action_id in range(8)},
        8: 1,
        9: 1,
    },
}

OPENFLY_ACTION_POLICY_NUM_ACTIONS: dict[str, int] = {
    "openfly_10": 10,
    "strict_8": 8,
    "collapse_forward": 8,
}
OPENFLY_ACTION_POLICIES: tuple[str, ...] = tuple(_ACTION_POLICIES)


def action_policy_num_actions(action_policy: str) -> int:
    """Return the logical action-ID space required by an action policy."""
    try:
        return OPENFLY_ACTION_POLICY_NUM_ACTIONS[action_policy]
    except KeyError as exc:
        raise ValueError(
            f"Unknown action_policy={action_policy!r}; "
            f"choose one of {list(OPENFLY_ACTION_POLICIES)}"
        ) from exc


@dataclass(frozen=True, slots=True)
class OpenFlyStepRecord:
    """One OpenFly action supervision row after an explicit action mapping."""

    episode_id: str
    trajectory_id: str
    step_id: int
    episode_position: int
    parquet_path: Path
    image_id: str
    instruction: str
    action_id: int
    source_action_id: int

    @property
    def virtual_image_path(self) -> str:
        return f"{self.parquet_path}#{self.image_id}"


class OpenFlyParquetClassificationDataset(Dataset[dict[str, Any]]):
    """Causal, direct-RGB OpenFly dataset for legacy MiniVLA variants.

    It returns the same sample fields used by ``MultiFrameCollator``.  For
    ``num_frames=1`` it additionally returns ``image`` for lightweight tools.
    History is selected by in-episode position, oldest first.  At the start of
    an episode, unavailable slots repeat the earliest available RGB image and
    are marked invalid, matching the legacy ``nearest_previous`` contract.
    """

    def __init__(
        self,
        annotation_json: str | Path,
        trajectory_root: str | Path,
        *,
        action_policy: str = "openfly_10",
        num_frames: int = 1,
        frame_history_stride: int = 1,
        action_chunk_horizon: int = 1,
        max_episodes: int = 0,
        cache_trajectories: int = 4,
    ) -> None:
        action_policy_num_actions(action_policy)
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if frame_history_stride <= 0:
            raise ValueError("frame_history_stride must be positive")
        if action_chunk_horizon not in {1, 4}:
            raise ValueError("action_chunk_horizon must be 1 or 4")
        if max_episodes < 0:
            raise ValueError("max_episodes must be non-negative")
        if cache_trajectories <= 0:
            raise ValueError("cache_trajectories must be positive")

        self.annotation_json = Path(annotation_json).expanduser().resolve()
        self.trajectory_root = Path(trajectory_root).expanduser().resolve()
        if not self.annotation_json.is_file():
            raise FileNotFoundError(f"OpenFly annotation JSON not found: {self.annotation_json}")
        if not self.trajectory_root.is_dir():
            raise NotADirectoryError(
                f"OPENFLY_TRAJECTORY_ROOT is not a directory: {self.trajectory_root}"
            )

        self.action_policy = action_policy
        self._action_map = _ACTION_POLICIES[action_policy]
        self.num_actions = action_policy_num_actions(action_policy)
        self.num_frames = int(num_frames)
        self.frame_history_stride = int(frame_history_stride)
        self.action_chunk_horizon = int(action_chunk_horizon)
        self.cache_trajectories = int(cache_trajectories)
        self._trajectory_cache: OrderedDict[Path, dict[str, bytes]] = OrderedDict()
        self._episodes: dict[str, list[OpenFlyStepRecord]] = {}
        self.records: list[OpenFlyStepRecord] = []
        self.source_action_counts: Counter[int] = Counter()
        self.mapped_action_counts: Counter[int] = Counter()
        self.ignored_boundary_sentinel_counts: Counter[int] = Counter()

        annotations = self._load_annotations()
        selected = 0
        for annotation_index, annotation in enumerate(annotations):
            if max_episodes and selected >= max_episodes:
                break
            records = self._build_episode(annotation_index, annotation)
            episode_id = records[0].episode_id
            if episode_id in self._episodes:
                raise ValueError(
                    f"Duplicate OpenFly episode_id={episode_id!r}; annotation IDs must be unique"
                )
            self._episodes[episode_id] = records
            self.records.extend(records)
            selected += 1

        if not self.records:
            raise RuntimeError(
                "OpenFly dataset has no valid training rows after applying the requested "
                f"action policy ({action_policy!r})."
            )

    def _load_annotations(self) -> list[Mapping[str, Any]]:
        with self.annotation_json.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if isinstance(raw, Mapping):
            raw = raw.get("episodes")
        if not isinstance(raw, list):
            raise ValueError(
                "OpenFly annotation must be a JSON list or an object with an 'episodes' list"
            )
        if not all(isinstance(item, Mapping) for item in raw):
            raise ValueError("Every OpenFly annotation item must be an object")
        return list(raw)

    def _build_episode(
        self, annotation_index: int, annotation: Mapping[str, Any]
    ) -> list[OpenFlyStepRecord]:
        raw_path = annotation.get("image_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"Annotation {annotation_index} has no non-empty image_path")
        parquet_path = self._resolve_parquet_path(raw_path)
        if not parquet_path.is_file():
            raise FileNotFoundError(
                f"Annotation {annotation_index} references missing trajectory parquet: {parquet_path}"
            )

        index_list = annotation.get("index_list")
        actions = annotation.get("action", annotation.get("actions"))
        if not isinstance(index_list, Sequence) or isinstance(index_list, (str, bytes)):
            raise ValueError(f"Annotation {annotation_index} index_list must be an array")
        if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
            raise ValueError(f"Annotation {annotation_index} action must be an array")
        if not index_list or not actions:
            raise ValueError(f"Annotation {annotation_index} has an empty index_list or action list")
        if len(index_list) != len(actions):
            raise ValueError(
                f"Annotation {annotation_index} has {len(index_list)} image IDs but {len(actions)} actions"
            )

        instruction = annotation.get("gpt_instruction", annotation.get("instruction", ""))
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"Annotation {annotation_index} has no non-empty instruction")

        trajectory_id = str(annotation.get("trajectory_id", raw_path))
        episode_id = str(annotation.get("episode_id", f"openfly-{annotation_index:08d}"))
        image_ids = [str(value) for value in index_list]
        records: list[OpenFlyStepRecord] = []
        saw_valid = False
        saw_post_boundary = False
        for raw_step_id, (image_id, source_action) in enumerate(zip(image_ids, actions)):
            try:
                source_action_id = int(source_action)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Annotation {annotation_index} step {raw_step_id} has non-integer action {source_action!r}"
                ) from exc
            if source_action_id in OPENFLY_SENTINEL_ACTION_IDS:
                self.ignored_boundary_sentinel_counts[source_action_id] += 1
                if saw_valid:
                    saw_post_boundary = True
                continue
            if source_action_id not in OPENFLY_VALID_ACTION_IDS:
                raise ValueError(
                    f"Annotation {annotation_index} step {raw_step_id} has unsupported OpenFly action "
                    f"{source_action_id}; expected 0..9 or boundary sentinel -1/-2"
                )
            if saw_post_boundary:
                raise ValueError(
                    f"Annotation {annotation_index} has executable action {source_action_id} after "
                    "the -2 post-episode boundary sentinel"
                )
            saw_valid = True
            if source_action_id not in self._action_map:
                raise ValueError(
                    f"Annotation {annotation_index} step {raw_step_id} uses OpenFly action "
                    f"{source_action_id} ({OPENFLY_ACTION_NAMES[source_action_id]}), which cannot "
                    "be represented by action_policy='strict_8'. Use action_policy='openfly_10' "
                    "to preserve the distance class, or use action_policy='collapse_forward' "
                    "only when collapsing forward distance is intended."
                )
            action_id = self._action_map[source_action_id]
            self.source_action_counts[source_action_id] += 1
            self.mapped_action_counts[action_id] += 1
            # Step IDs are the causal *supervised* action positions.  The
            # raw JSON can contain leading -1 and trailing -2 camera-only
            # sentinels, which never belong to the policy history.
            step_id = len(records)
            records.append(
                OpenFlyStepRecord(
                    episode_id=episode_id,
                    trajectory_id=trajectory_id,
                    step_id=step_id,
                    episode_position=step_id,
                    parquet_path=parquet_path,
                    image_id=image_id,
                    instruction=instruction.strip(),
                    action_id=action_id,
                    source_action_id=source_action_id,
                )
            )
        if not records:
            raise ValueError(
                f"Annotation {annotation_index} has no executable OpenFly action after boundary filtering"
            )
        return records

    def _resolve_parquet_path(self, image_path: str) -> Path:
        candidate = Path(image_path)
        if not candidate.is_absolute():
            candidate = self.trajectory_root / candidate
        # The released builder uses ``f"{image_path}.parquet"``.  Preserve that
        # behavior while accepting callers that already include the suffix.
        if candidate.suffix != ".parquet":
            candidate = Path(f"{candidate}.parquet")
        return candidate.resolve()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        episode = self._episodes[record.episode_id]
        images: list[Image.Image] = []
        valid_mask: list[bool] = []
        for frame_slot in range(self.num_frames):
            offset = self.frame_history_stride * (self.num_frames - 1 - frame_slot)
            requested_position = record.episode_position - offset
            is_valid = requested_position >= 0
            source_position = max(0, requested_position)
            source = episode[source_position]
            images.append(self._load_image(source.parquet_path, source.image_id))
            valid_mask.append(is_valid)

        if not valid_mask[-1]:
            raise RuntimeError("The current OpenFly frame must always be valid")
        previous_action = self.num_actions if record.episode_position == 0 else episode[
            record.episode_position - 1
        ].action_id
        previous_actions_text: list[int | None] = []
        for position in range(record.episode_position - 3, record.episode_position):
            previous_actions_text.append(
                episode[position].action_id if position >= 0 else None
            )
        item: dict[str, Any] = {
            "images": images,
            "image": images[-1],
            "frame_valid_mask": valid_mask,
            "instruction": record.instruction,
            "action_id": record.action_id,
            "episode_id": record.episode_id,
            "step_id": record.step_id,
            "image_path": record.virtual_image_path,
            "prev_action": previous_action,
            "previous_actions_text": previous_actions_text,
            "source_action_id": record.source_action_id,
        }
        if self.action_chunk_horizon > 1:
            item["action_chunk_targets"] = self._action_chunk_targets(record, episode)
        return item

    def _action_chunk_targets(
        self, record: OpenFlyStepRecord, episode: list[OpenFlyStepRecord]
    ) -> list[int]:
        targets = [-100] * self.action_chunk_horizon
        for slot in range(self.action_chunk_horizon):
            position = record.episode_position + slot
            if position >= len(episode):
                break
            future = episode[position]
            targets[slot] = future.action_id
            if future.action_id == 0:
                break
        if targets[0] != record.action_id:
            raise RuntimeError("OpenFly action-chunk slot 0 disagrees with the current action")
        return targets

    def _load_image(self, parquet_path: Path, image_id: str) -> Image.Image:
        images = self._load_trajectory(parquet_path)
        try:
            payload = images[image_id]
        except KeyError as exc:
            raise KeyError(
                f"Trajectory {parquet_path} does not contain image_id={image_id!r}"
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
        except ImportError as exc:  # pragma: no cover - environment error
            raise RuntimeError(
                "OpenFly Parquet loading requires pyarrow. Install it in the active "
                "training environment before using this adapter."
            ) from exc

        table = pq.read_table(parquet_path, columns=["image_id", "image"])
        images: dict[str, bytes] = {}
        for row in table.to_pylist():
            image = row.get("image")
            payload = image.get("bytes") if isinstance(image, Mapping) else None
            if payload:
                images[str(row["image_id"])] = bytes(payload)
        if not images:
            raise RuntimeError(f"OpenFly trajectory has no usable image bytes: {parquet_path}")
        self._trajectory_cache[parquet_path] = images
        while len(self._trajectory_cache) > self.cache_trajectories:
            self._trajectory_cache.popitem(last=False)
        return images

    def __getstate__(self) -> dict[str, Any]:
        """Do not fork a potentially large Parquet byte cache into DataLoader workers."""
        state = self.__dict__.copy()
        state["_trajectory_cache"] = OrderedDict()
        return state
