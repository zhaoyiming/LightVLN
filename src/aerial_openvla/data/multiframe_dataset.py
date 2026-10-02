"""Multi-frame dataset and collator for AirMiniVLA.

Each sample loads F frames per step (default 4 at t-6, t-4, t-2, t) from
the same episode, enabling the model to see visual history.
"""

from __future__ import annotations

from bisect import bisect_right
import json
import math
import os
import random
import warnings
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from aerial_openvla.data.manifest import load_step_records
from aerial_openvla.data.schema import ClassificationRecord
from aerial_openvla.data.instruction_segments import InstructionSegment, segment_instruction


ACTION_TEXT_NAMES: tuple[str, ...] = (
    "stop",
    "forward",
    "left",
    "right",
    "up",
    "down",
    "move left",
    "move right",
)
PREVIOUS_ACTION_SLOTS = 3
PREVIOUS_ACTIONS_PREFIX = "Previous actions:"

# AirVLN's discrete control increments.  Relative altitude is expressed in a
# user-facing up-positive coordinate; the simulator internally uses NED z.
TURN_ANGLE_DEGREES = 15
ALTITUDE_STEP_METERS = 2
PLANAR_STEP_METERS = 5


def truncate_instruction_to_token_budget(
    tokenizer: Any, instruction: str, max_tokens: int
) -> str:
    """Truncate only the original instruction, before prompt construction."""
    source = str(instruction).strip()
    if max_tokens <= 0:
        raise ValueError("Instruction token budget must be positive")
    try:
        encoded = tokenizer(
            source,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError) as exc:
        raise RuntimeError(
            "Instruction-only token budgeting requires a fast tokenizer with offsets"
        ) from exc
    token_ids = encoded.get("input_ids")
    offsets = encoded.get("offset_mapping")
    if token_ids is None or offsets is None:
        raise RuntimeError("Tokenizer did not return instruction token offsets")
    if len(token_ids) <= max_tokens:
        return source
    end = int(offsets[max_tokens - 1][1])
    trimmed = source[:end].rstrip()
    if not trimmed:
        raise RuntimeError("Instruction-only truncation removed all instruction text")
    return trimmed

def append_previous_actions_text(
    instruction: str,
    action_ids: list[int | None] | tuple[int | None, ...],
) -> str:
    """Append a fixed-width, oldest-to-newest action history to an instruction."""
    if len(action_ids) > PREVIOUS_ACTION_SLOTS:
        action_ids = action_ids[-PREVIOUS_ACTION_SLOTS:]
    padded = [None] * (PREVIOUS_ACTION_SLOTS - len(action_ids)) + list(action_ids)
    names: list[str] = []
    for action_id in padded:
        if action_id is None:
            names.append("none")
        elif 0 <= action_id < len(ACTION_TEXT_NAMES):
            names.append(ACTION_TEXT_NAMES[action_id])
        else:
            raise ValueError(f"Invalid previous action ID: {action_id}")
    return f"{instruction.strip()} {PREVIOUS_ACTIONS_PREFIX} {', '.join(names)}"


def previous_action_ids_for_step(
    records_by_step: Mapping[int, ClassificationRecord], current_step: int
) -> list[int | None]:
    """Return exact GT actions at t-3, t-2 and t-1, with missing slots empty."""
    return [
        records_by_step[step].action_id if step in records_by_step else None
        for step in range(current_step - PREVIOUS_ACTION_SLOTS, current_step)
    ]


def normalize_relative_heading(heading_degrees: int) -> int:
    """Normalize heading to [-180, 180], preserving the sign at 180 degrees."""
    normalized = (heading_degrees + 180) % 360 - 180
    if normalized == -180 and heading_degrees > 0:
        return 180
    return normalized


def update_relative_state(
    heading_degrees: int, altitude_meters: int, action_id: int
) -> tuple[int, int]:
    """Apply one AirVLN action to a start-relative heading/altitude state."""
    if action_id == 2:  # TURN_LEFT
        heading_degrees -= TURN_ANGLE_DEGREES
    elif action_id == 3:  # TURN_RIGHT
        heading_degrees += TURN_ANGLE_DEGREES
    elif action_id == 4:  # GO_UP
        altitude_meters += ALTITUDE_STEP_METERS
    elif action_id == 5:  # GO_DOWN
        altitude_meters -= ALTITUDE_STEP_METERS
    return normalize_relative_heading(heading_degrees), altitude_meters


def update_relative_planar_state(
    heading_degrees: int,
    altitude_meters: int,
    forward_meters: float,
    left_meters: float,
    action_id: int,
) -> tuple[int, int, float, float]:
    """Apply one action in an episode-start-local horizontal frame.

    ``forward_meters`` and ``left_meters`` are fixed to the starting heading,
    while each movement action is expressed in the current body frame.
    """
    theta = math.radians(heading_degrees)
    if action_id == 1:  # MOVE_FORWARD
        forward_meters += PLANAR_STEP_METERS * math.cos(theta)
        left_meters -= PLANAR_STEP_METERS * math.sin(theta)
    elif action_id == 6:  # MOVE_LEFT
        forward_meters += PLANAR_STEP_METERS * math.sin(theta)
        left_meters += PLANAR_STEP_METERS * math.cos(theta)
    elif action_id == 7:  # MOVE_RIGHT
        forward_meters -= PLANAR_STEP_METERS * math.sin(theta)
        left_meters -= PLANAR_STEP_METERS * math.cos(theta)
    heading_degrees, altitude_meters = update_relative_state(
        heading_degrees, altitude_meters, action_id
    )
    return heading_degrees, altitude_meters, forward_meters, left_meters


def relative_state_for_step(
    records_by_step: Mapping[int, ClassificationRecord], current_step: int
) -> tuple[int, int]:
    """Return GT-relative state immediately before predicting ``current_step``."""
    heading_degrees = 0
    altitude_meters = 0
    for step_id in sorted(records_by_step):
        if step_id >= current_step:
            break
        heading_degrees, altitude_meters = update_relative_state(
            heading_degrees, altitude_meters, records_by_step[step_id].action_id
        )
    return heading_degrees, altitude_meters


def relative_planar_state_for_step(
    records_by_step: Mapping[int, ClassificationRecord], current_step: int
) -> tuple[int, int, float, float]:
    """Return GT relative yaw, altitude and horizontal position before a step."""
    heading_degrees = 0
    altitude_meters = 0
    forward_meters = 0.0
    left_meters = 0.0
    for step_id in sorted(records_by_step):
        if step_id >= current_step:
            break
        heading_degrees, altitude_meters, forward_meters, left_meters = (
            update_relative_planar_state(
                heading_degrees,
                altitude_meters,
                forward_meters,
                left_meters,
                records_by_step[step_id].action_id,
            )
        )
    return heading_degrees, altitude_meters, forward_meters, left_meters


def prepend_relative_state_text(
    instruction: str,
    heading_degrees: int,
    altitude_meters: int,
    forward_meters: float | None = None,
    left_meters: float | None = None,
) -> str:
    """Render the fixed S2-Relative prefix before the original instruction."""
    heading_direction = "clockwise" if heading_degrees >= 0 else "counterclockwise"
    altitude_direction = "above" if altitude_meters >= 0 else "below"
    prefix = (
        f"Relative heading from start: {abs(heading_degrees)} degrees "
        f"{heading_direction}.\n"
        f"Relative altitude from start: {abs(altitude_meters)} meters "
        f"{altitude_direction}.\n"
    )
    if forward_meters is not None and left_meters is not None:
        # Keep the text vocabulary compact while retaining full-precision state
        # internally. Python's round-to-even is undesirable at .5 boundaries.
        rounded_forward = int(math.copysign(math.floor(abs(forward_meters) + 0.5), forward_meters))
        rounded_left = int(math.copysign(math.floor(abs(left_meters) + 0.5), left_meters))
        forward_label = "Forward" if rounded_forward >= 0 else "Backward"
        left_label = "Left" if rounded_left >= 0 else "Right"
        prefix += (
            f"{forward_label} from start: {abs(rounded_forward)} meters.\n"
            f"{left_label} from start: {abs(rounded_left)} meters.\n"
        )
    return f"{prefix}{instruction.strip()}"


def tokenize_s2_prompts(
    tokenizer: Any,
    prompts: list[str],
    *,
    max_length: int,
    return_offsets_mapping: bool = False,
) -> dict[str, torch.Tensor]:
    """Tokenize S2 prompts while guaranteeing that action history survives.

    Normal right truncation removes ``Previous actions`` from long AirVLN
    instructions.  We instead keep the beginning of the prompt and its entire
    action-history tail, dropping only the middle when the token budget is
    exceeded.  Offset mappings make the history boundary tokenizer-exact.
    """
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    try:
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
    except (NotImplementedError, TypeError) as exc:
        raise RuntimeError("S2 requires a fast tokenizer with offset mappings") from exc

    offsets = encoded.pop("offset_mapping", None)
    if offsets is None:
        raise RuntimeError("S2 tokenizer did not return offset_mapping")
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    kept_ids: list[torch.Tensor] = []
    kept_offsets: list[torch.Tensor] = []
    for row, prompt in enumerate(prompts):
        marker_start = prompt.rfind(PREVIOUS_ACTIONS_PREFIX)
        if marker_start < 0:
            raise ValueError("S2 prompt is missing the Previous actions suffix")
        valid_indices = attention_mask[row].bool().nonzero(as_tuple=False).flatten()
        row_offsets = offsets[row, valid_indices]
        tail_candidates = (row_offsets[:, 1] > marker_start).nonzero(as_tuple=False).flatten()
        if not len(tail_candidates):
            raise ValueError("Tokenizer offsets do not cover the Previous actions suffix")
        tail_position = int(tail_candidates[0].item())
        tail_indices = valid_indices[tail_position:]
        if len(tail_indices) >= max_length:
            raise ValueError(
                "Previous actions suffix exceeds max_text_tokens; increase max_text_tokens"
            )
        prefix_count = min(tail_position, max_length - len(tail_indices))
        selected = torch.cat((valid_indices[:prefix_count], tail_indices))
        kept_ids.append(input_ids[row, selected])
        kept_offsets.append(offsets[row, selected])

    # ``max_length`` is a ceiling, matching the original classifier path:
    # short prompts stay short while overlong prompts are cropped above.
    fixed_padding = os.environ.get("AIRVLN_S2_FIXED_PADDING", "0").strip().lower()
    batch_length = (
        max_length
        if fixed_padding in {"1", "true", "yes", "on"}
        else max(len(row) for row in kept_ids)
    )
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        raise ValueError("S2 tokenizer must define pad_token_id")
    output_ids = torch.full(
        (len(prompts), batch_length), int(pad_token_id), dtype=input_ids.dtype
    )
    output_mask = torch.zeros(
        (len(prompts), batch_length), dtype=attention_mask.dtype
    )
    output_offsets = torch.zeros(
        (len(prompts), batch_length, 2), dtype=offsets.dtype
    )
    padding_side = getattr(tokenizer, "padding_side", "right")
    for row, (ids, row_offsets) in enumerate(zip(kept_ids, kept_offsets)):
        start = batch_length - len(ids) if padding_side == "left" else 0
        end = start + len(ids)
        output_ids[row, start:end] = ids
        output_mask[row, start:end] = 1
        output_offsets[row, start:end] = row_offsets

    result = {"input_ids": output_ids, "attention_mask": output_mask}
    if return_offsets_mapping:
        result["offset_mapping"] = output_offsets
    return result


def history_step_indices(step_id: int, num_frames: int, history_stride: int) -> list[int]:
    """Return fixed-slot causal indices, oldest first, without clamping.

    Negative indices denote history slots that do not exist yet.  Callers may
    repeat an image to keep the tensor shape fixed, but must mark those slots
    invalid instead of presenting the repeated image as real history.
    """
    if step_id < 0:
        raise ValueError("step_id must be non-negative")
    if num_frames <= 0 or history_stride <= 0:
        raise ValueError("num_frames and history_stride must be positive")
    return [step_id - history_stride * (num_frames - 1 - index) for index in range(num_frames)]


class MultiFrameClassificationDataset(Dataset[dict]):
    """Dataset that loads multiple frames per action sample.

    Frames are selected at offsets [-(F-1)*2, ..., -2, 0] relative to the
    current step, which provides evenly-spaced visual history.
    For F=4: t-6, t-4, t-2, t (current).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        num_frames: int = 4,
        history_stride: int = 2,
        missing_history: str = "error",
        visual_history_mode: str = "none",
        progress_annotations_dir: str | Path | None = None,
        action_chunk_horizon: int = 1,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.num_frames = num_frames
        self.history_stride = history_stride
        if visual_history_mode not in {
            "none", "recent3", "keyframe3", "motion_recent3"
        }:
            raise ValueError(
                "visual_history_mode must be none, recent3, keyframe3, "
                "or motion_recent3"
            )
        self.visual_history_mode = visual_history_mode
        if action_chunk_horizon not in {1, 4}:
            raise ValueError("action_chunk_horizon must be 1 or 4")
        self.action_chunk_horizon = action_chunk_horizon
        if num_frames <= 0 or history_stride <= 0:
            raise ValueError("num_frames and history_stride must be positive")
        if missing_history not in {"error", "nearest_previous"}:
            raise ValueError("missing_history must be 'error' or 'nearest_previous'")
        self.missing_history = missing_history

        self.records: list[ClassificationRecord] = load_step_records(self.manifest_path)
        self._progress_action_lengths: dict[str, int] | None = None
        if progress_annotations_dir is not None:
            self._progress_action_lengths = self._load_progress_action_lengths(
                Path(progress_annotations_dir)
            )

        # Build episode index: episode_id → list of (global_record_index, record)
        self._episodes: dict[str, list[tuple[int, ClassificationRecord]]] = {}
        for idx, rec in enumerate(self.records):
            self._episodes.setdefault(rec.episode_id, []).append((idx, rec))

        # Sort each episode"s steps
        for ep_id in self._episodes:
            self._episodes[ep_id].sort(key=lambda x: x[1].step_id)

        self._episode_records_by_step: dict[str, dict[int, ClassificationRecord]] = {}
        self._episode_step_ids: dict[str, list[int]] = {}
        for episode_id, episode in self._episodes.items():
            by_step = {record.step_id: record for _, record in episode}
            if len(by_step) != len(episode):
                raise ValueError(f"Duplicate step_id values in episode {episode_id}")
            self._episode_records_by_step[episode_id] = by_step
            self._episode_step_ids[episode_id] = sorted(by_step)

        # ``__getitem__`` is called for every training example.  Keeping the
        # in-episode position avoids a repeated linear scan of each episode.
        self._record_positions: dict[int, int] = {}
        for episode in self._episodes.values():
            for position, (record_index, _record) in enumerate(episode):
                self._record_positions[record_index] = position

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        ep_records = self._episodes.get(record.episode_id, [(index, record)])

        if self.visual_history_mode != "none":
            return self._variable_history_item(index, record, ep_records)

        # For four frames and stride two the slots are always [t-6, t-4, t-2, t].
        target_steps = history_step_indices(record.step_id, self.num_frames, self.history_stride)

        images: list[Image.Image] = []
        frame_valid_mask: list[bool] = []
        prev_action: int = 8  # start token (8 = pad/start)
        previous_actions_text: list[int | None] = []
        last_successful: Image.Image | None = None

        for slot_index, target_step in enumerate(target_steps):
            target_rec, is_exact = self._history_record(record, target_step)
            img_path = target_rec.resolve_image(self.manifest_path)
            try:
                with Image.open(img_path) as source:
                    image = source.convert("RGB")
            except Exception as exc:
                # A missing current observation cannot be treated as valid.
                if slot_index == self.num_frames - 1 or last_successful is None:
                    raise RuntimeError(
                        f"Failed to load required image {img_path} for "
                        f"episode={record.episode_id}, step={record.step_id}"
                    ) from exc
                image = last_successful.copy()
                is_exact = False
            images.append(image)
            frame_valid_mask.append(is_exact)
            last_successful = image

        if not frame_valid_mask[-1]:
            raise RuntimeError("The current frame slot must always be valid")

        # Previous action: if we have a step before current, use its action
        current_pos = self._record_positions[index]
        if record.prev_action_id is not None:
            prev_action = record.prev_action_id
        elif current_pos > 0:
            _, prev_rec = ep_records[current_pos - 1]
            prev_action = prev_rec.action_id

        # S2 uses the exact three preceding steps as prompt text.  Missing
        # episode-start or sparse-manifest slots remain explicit ``none``.
        by_step = self._episode_records_by_step[record.episode_id]
        previous_actions_text = previous_action_ids_for_step(by_step, record.step_id)
        (
            relative_heading_degrees,
            relative_altitude_meters,
            relative_forward_meters,
            relative_left_meters,
        ) = relative_planar_state_for_step(by_step, record.step_id)

        return {
            "images": images,           # [PIL.Image × F]
            "frame_valid_mask": frame_valid_mask,
            "instruction": record.instruction,
            "action_id": record.action_id,
            **(
                {"action_chunk_targets": self._action_chunk_targets(index, record, ep_records)}
                if self.action_chunk_horizon > 1
                else {}
            ),
            "episode_id": record.episode_id,
            "step_id": record.step_id,
            "image_path": str(record.resolve_image(self.manifest_path)),
            "prev_action": prev_action,
            "previous_actions_text": previous_actions_text,
            "relative_heading_degrees": relative_heading_degrees,
            "relative_altitude_meters": relative_altitude_meters,
            "relative_forward_meters": relative_forward_meters,
            "relative_left_meters": relative_left_meters,
            **self._progress_fields(record),
        }

    def _variable_history_item(
        self,
        index: int,
        record: ClassificationRecord,
        ep_records: list[tuple[int, ClassificationRecord]],
    ) -> dict[str, Any]:
        """Return only real causal history frames plus the current frame.

        Recent-3 selects the three immediately preceding observations.  Keyframe-3
        selects observations whose GT action differs from the preceding action.
        No image is repeated and no empty visual item is synthesized.
        """
        current_pos = self._record_positions[index]
        prior_records = [item[1] for item in ep_records[:current_pos]]
        if self.visual_history_mode in {"recent3", "motion_recent3"}:
            history_records = prior_records[-3:]
        else:
            history_records = [
                ep_records[position][1]
                for position in range(1, current_pos)
                if ep_records[position][1].action_id
                != ep_records[position - 1][1].action_id
            ][-3:]

        selected_records = [*history_records, record]
        images: list[Image.Image] = []
        for selected in selected_records:
            image_path = selected.resolve_image(self.manifest_path)
            try:
                with Image.open(image_path) as source:
                    images.append(source.convert("RGB"))
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load real history image {image_path} for "
                    f"episode={record.episode_id}, step={record.step_id}"
                ) from exc

        # Slot 3 is always current.  With fewer than three history frames the
        # existing history is right-aligned to recent/middle/old semantics.
        first_slot = self.num_frames - len(selected_records)
        frame_slot_ids = list(range(first_slot, self.num_frames))
        prev_action = 8
        if record.prev_action_id is not None:
            prev_action = record.prev_action_id
        elif current_pos > 0:
            prev_action = ep_records[current_pos - 1][1].action_id

        by_step = self._episode_records_by_step[record.episode_id]
        (
            relative_heading_degrees,
            relative_altitude_meters,
            relative_forward_meters,
            relative_left_meters,
        ) = relative_planar_state_for_step(by_step, record.step_id)
        return {
            "images": images,
            "frame_valid_mask": [True] * len(images),
            "frame_slot_ids": frame_slot_ids,
            "instruction": record.instruction,
            "action_id": record.action_id,
            **(
                {"action_chunk_targets": self._action_chunk_targets(index, record, ep_records)}
                if self.action_chunk_horizon > 1
                else {}
            ),
            "episode_id": record.episode_id,
            "step_id": record.step_id,
            "image_path": str(record.resolve_image(self.manifest_path)),
            "prev_action": prev_action,
            "previous_actions_text": previous_action_ids_for_step(by_step, record.step_id),
            "relative_heading_degrees": relative_heading_degrees,
            "relative_altitude_meters": relative_altitude_meters,
            "relative_forward_meters": relative_forward_meters,
            "relative_left_meters": relative_left_meters,
            **self._progress_fields(record),
        }

    def _action_chunk_targets(
        self,
        index: int,
        record: ClassificationRecord,
        ep_records: list[tuple[int, ClassificationRecord]],
    ) -> list[int]:
        """Return [a_t..a_t+K-1] without crossing gaps, STOP, or episodes."""
        targets = [-100] * self.action_chunk_horizon
        current_pos = self._record_positions[index]
        for horizon_index in range(self.action_chunk_horizon):
            future_pos = current_pos + horizon_index
            if future_pos >= len(ep_records):
                break
            future_record = ep_records[future_pos][1]
            if future_record.step_id != record.step_id + horizon_index:
                break
            targets[horizon_index] = int(future_record.action_id)
            if future_record.action_id == 0:  # include first STOP, mask everything after it
                break
        if targets[0] != record.action_id:
            raise RuntimeError("AC target slot 0 must equal the current action")
        return targets

    def _load_progress_action_lengths(self, annotations_dir: Path) -> dict[str, int]:
        """Load the denominator used by AirVLN's official TF progress sensor."""
        if not self.records:
            return {}
        split = self.records[0].split
        if split.endswith("_cache"):
            split = split.removesuffix("_cache")
        annotation_path = annotations_dir / f"{split}.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(
                f"Progress supervision requires AirVLN annotations: {annotation_path}"
            )
        with annotation_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        episodes = raw.get("episodes", raw) if isinstance(raw, dict) else raw
        lengths = {
            str(episode["episode_id"]): len(episode["actions"])
            for episode in episodes
        }
        if any(length <= 0 for length in lengths.values()):
            raise ValueError(f"Empty action sequence in {annotation_path}")
        return lengths

    def _progress_fields(self, record: ClassificationRecord) -> dict[str, Any]:
        if record.progress_label is not None:
            return {"progress_target": record.progress_label, "progress_valid": True}
        if self._progress_action_lengths is None:
            return {}
        action_count = self._progress_action_lengths.get(record.episode_id)
        if action_count is None:
            raise KeyError(
                f"Episode {record.episode_id} is absent from progress annotations"
            )
        if record.step_id >= action_count:
            raise ValueError(
                f"step_id={record.step_id} exceeds official action count={action_count} "
                f"for episode={record.episode_id}"
            )
        # Match AirVLN ENV.get_obs_at TF exactly.  In particular, the final
        # recorded STOP step is (N-1)/N rather than the common step/(N-1).
        return {
            "progress_target": record.step_id / action_count,
            "progress_valid": True,
        }

    def _history_record(
        self, record: ClassificationRecord, target_step: int
    ) -> tuple[ClassificationRecord, bool]:
        by_step = self._episode_records_by_step[record.episode_id]
        exact = by_step.get(target_step)
        if exact is not None:
            return exact, True
        step_ids = self._episode_step_ids[record.episode_id]
        # Episode-start padding is expected and is never real history.
        if target_step < step_ids[0]:
            return by_step[step_ids[0]], False
        if self.missing_history == "error":
            raise ValueError(
                f"Missing exact history frame for episode={record.episode_id}, "
                f"step={record.step_id}, required_step={target_step}. "
                "Use a complete cache or explicitly enable missing_history='nearest_previous'."
            )
        position = bisect_right(step_ids, target_step) - 1
        if position < 0:
            return by_step[step_ids[0]], False
        return by_step[step_ids[position]], False


class MultiFrameCollator:
    """Collate multi-frame samples into a single batch.

    Processes each frame independently through the Prismatic image processor,
    then stacks into shape [B, F, C, H, W] (or [B, F*C, H, W] for fused
    backbones that channel-stack).
    """

    def __init__(
        self,
        processor: Any,
        prompt_template: str,
        max_text_tokens: int = 256,
        instruction_only_text_budget: bool = False,
        num_frames: int = 4,
        include_instruction_token_mask: bool = False,
        include_segment_token_ids: bool = False,
        task_state_max_segments: int = 10,
        task_state_metadata_path: str | None = None,
        use_previous_actions_text: bool = False,
        use_relative_state_text: bool = False,
        use_relative_planar_state_text: bool = False,
        action_chunk_horizon: int = 1,
        visual_history_mode: str = "none",
    ) -> None:
        self.processor = processor
        self.prompt_template = prompt_template
        self.max_text_tokens = max_text_tokens
        self.instruction_only_text_budget = bool(instruction_only_text_budget)
        self.num_frames = num_frames
        self.include_instruction_token_mask = include_instruction_token_mask
        self.include_segment_token_ids = include_segment_token_ids
        self.task_state_max_segments = task_state_max_segments
        self.segment_metadata: dict[str, dict[str, object]] | None = None
        self._segment_metadata_fallbacks: set[str] = set()
        if task_state_metadata_path is not None:
            with Path(task_state_metadata_path).open("r", encoding="utf-8") as handle:
                self.segment_metadata = json.load(handle)
        if include_segment_token_ids and not include_instruction_token_mask:
            raise ValueError("Segment token IDs require the instruction token mask")
        self.use_previous_actions_text = use_previous_actions_text
        self.use_relative_state_text = use_relative_state_text
        self.use_relative_planar_state_text = use_relative_planar_state_text
        if action_chunk_horizon not in {1, 4}:
            raise ValueError("action_chunk_horizon must be 1 or 4")
        self.action_chunk_horizon = action_chunk_horizon
        self.visual_history_mode = visual_history_mode
        self.variable_visual_history = visual_history_mode in {
            "recent3", "keyframe3", "motion_recent3"
        }
        if use_relative_planar_state_text and not use_relative_state_text:
            raise ValueError("Planar relative text requires relative-state text")
        if use_previous_actions_text and use_relative_state_text:
            raise ValueError("Choose either action-history text or relative-state text")

        if self.instruction_only_text_budget and (
            self.use_previous_actions_text or self.use_relative_state_text
        ):
            raise ValueError("Instruction-only text budget requires raw non-Relative input")
    def __call__(self, examples: list[dict]) -> dict[str, Any]:
        F = self.num_frames

        image_processor = getattr(self.processor, "image_processor", self.processor)
        frame_batch_indices = None
        frame_slot_indices = None
        if self.variable_visual_history:
            flat_images = [image for ex in examples for image in ex["images"]]
            encoded = image_processor(flat_images, return_tensors="pt")
            pixel_values = encoded["pixel_values"]
            if pixel_values.dim() != 4:
                raise ValueError("Variable visual history requires [N,C,H,W] pixels")
            frame_batch_indices = torch.tensor(
                [row for row, ex in enumerate(examples) for _ in ex["images"]],
                dtype=torch.long,
            )
            frame_slot_indices = torch.tensor(
                [slot for ex in examples for slot in ex["frame_slot_ids"]],
                dtype=torch.long,
            )
            frame_valid_mask = torch.zeros(len(examples), F, dtype=torch.bool)
            for row, ex in enumerate(examples):
                frame_valid_mask[row, ex["frame_slot_ids"]] = True
        else:
            all_pixel_values: list[torch.Tensor] = []
            for frame_idx in range(F):
                frame_images = [ex["images"][frame_idx] for ex in examples]
                encoded = image_processor(frame_images, return_tensors="pt")
                pv = encoded["pixel_values"]
                if pv.dim() == 4:
                    pv = pv.unsqueeze(1)
                all_pixel_values.append(pv)
            pixel_values = torch.cat(all_pixel_values, dim=1)
            frame_valid_mask = torch.tensor(
                [ex.get("frame_valid_mask", [True] * F) for ex in examples],
                dtype=torch.bool,
            )
        source_instructions: list[str] = []
        for ex in examples:
            source_instruction = str(ex["instruction"]).strip()
            if self.instruction_only_text_budget:
                source_instruction = truncate_instruction_to_token_budget(
                    self.processor.tokenizer, source_instruction, self.max_text_tokens
                )
            source_instructions.append(source_instruction)
        # Text processing (shared instruction across frames)
        rendered_instructions = []
        for ex, source_instruction in zip(examples, source_instructions):
            if self.use_previous_actions_text:
                rendered = append_previous_actions_text(
                    ex["instruction"], ex["previous_actions_text"]
                )
            elif self.use_relative_state_text:
                rendered = prepend_relative_state_text(
                    ex["instruction"],
                    int(ex["relative_heading_degrees"]),
                    int(ex["relative_altitude_meters"]),
                    (
                        float(ex["relative_forward_meters"])
                        if self.use_relative_planar_state_text
                        else None
                    ),
                    (
                        float(ex["relative_left_meters"])
                        if self.use_relative_planar_state_text
                        else None
                    ),
                )
            else:
                rendered = source_instruction
            rendered_instructions.append(rendered)
        prompts = [
            self.prompt_template.format(instruction=instruction)
            for instruction in rendered_instructions
        ]
        if self.use_previous_actions_text:
            text_encoded = tokenize_s2_prompts(
                self.processor.tokenizer,
                prompts,
                max_length=self.max_text_tokens,
                return_offsets_mapping=self.include_instruction_token_mask,
            )
        else:
            tokenizer_kwargs = dict(
                padding=True,
                truncation=not self.instruction_only_text_budget,
                return_tensors="pt",
            )
            if not self.instruction_only_text_budget:
                tokenizer_kwargs["max_length"] = self.max_text_tokens
            if self.include_instruction_token_mask:
                tokenizer_kwargs["return_offsets_mapping"] = True
            try:
                text_encoded = self.processor.tokenizer(prompts, **tokenizer_kwargs)
            except (NotImplementedError, TypeError) as exc:
                raise RuntimeError(
                    "text_query progress requires a fast tokenizer with offset mappings"
                ) from exc
        offsets = text_encoded.pop("offset_mapping", None)

        if self.variable_visual_history:
            # The model trims masked text and left-pads the final multimodal
            # sequence.  Keep tokenizer tensors left padded as an auditable
            # collator invariant as well.
            input_ids = text_encoded["input_ids"]
            text_mask = text_encoded.get("attention_mask", torch.ones_like(input_ids))
            left_ids = torch.full_like(input_ids, int(self.processor.tokenizer.pad_token_id))
            left_mask = torch.zeros_like(text_mask)
            left_offsets = torch.zeros_like(offsets) if offsets is not None else None
            for row in range(input_ids.shape[0]):
                valid = text_mask[row].bool()
                count = int(valid.sum().item())
                left_ids[row, -count:] = input_ids[row, valid]
                left_mask[row, -count:] = 1
                if left_offsets is not None:
                    left_offsets[row, -count:] = offsets[row, valid]
            text_encoded["input_ids"] = left_ids
            text_encoded["attention_mask"] = left_mask
            offsets = left_offsets

        batch = {
            "input_ids": text_encoded["input_ids"],
            "attention_mask": (
                text_encoded.get("attention_mask")
                if text_encoded.get("attention_mask") is not None
                else torch.ones_like(text_encoded["input_ids"])
            ),
            "pixel_values": pixel_values,
            "frame_valid_mask": frame_valid_mask,
            "action_id": torch.tensor(
                [ex["action_id"] for ex in examples], dtype=torch.long
            ),
            "episode_id": [ex["episode_id"] for ex in examples],
            "step_id": [ex["step_id"] for ex in examples],
            "image_path": [ex["image_path"] for ex in examples],
            "prev_actions": torch.tensor(
                [ex["prev_action"] for ex in examples], dtype=torch.long
            ),
        }
        if self.action_chunk_horizon > 1:
            targets = [ex["action_chunk_targets"] for ex in examples]
            if any(len(target) != self.action_chunk_horizon for target in targets):
                raise ValueError("Action-chunk target length disagrees with collator horizon")
            batch["action_chunk_targets"] = torch.tensor(targets, dtype=torch.long)
        if frame_batch_indices is not None:
            batch["frame_batch_indices"] = frame_batch_indices
            batch["frame_slot_indices"] = frame_slot_indices
            text_lengths = batch["attention_mask"].sum(dim=1).long()
            if self.visual_history_mode == "motion_recent3":
                # Three earliest-padded delta slots always collapse into the
                # same 8 motion tokens, including at episode start.
                sequence_lengths = text_lengths + 1 + 256 + 8
            else:
                frame_lengths = torch.tensor(
                    [len(ex["images"]) for ex in examples], dtype=torch.long
                )
                sequence_lengths = (
                    text_lengths + 1 + 256 + (frame_lengths - 1) * 8
                )
            batch["multimodal_left_padding"] = sequence_lengths.max() - sequence_lengths
        if "progress_target" in examples[0]:
            batch["progress_targets"] = torch.tensor(
                [ex["progress_target"] for ex in examples], dtype=torch.float32
            )
            batch["progress_valid_mask"] = torch.tensor(
                [ex.get("progress_valid", True) for ex in examples], dtype=torch.bool
            )
        if self.include_instruction_token_mask:
            if offsets is None:
                raise RuntimeError("Tokenizer did not return offset_mapping")
            instruction_mask = torch.zeros_like(text_encoded["attention_mask"], dtype=torch.bool)
            # Segment supervision applies to the original instruction only.
            # State prefixes are useful model context, but they are not part of
            # the offline instruction-segmentation metadata.
            segment_sources = source_instructions
            for row, (prompt, instruction, segment_source) in enumerate(
                zip(prompts, rendered_instructions, segment_sources)
            ):
                supervised_instruction = (
                    segment_source if self.include_segment_token_ids else instruction
                )
                char_start = prompt.find(supervised_instruction)
                if char_start < 0:
                    raise ValueError("Rendered prompt does not contain its instruction")
                char_end = char_start + len(supervised_instruction)
                for column, (token_start, token_end) in enumerate(offsets[row].tolist()):
                    if (
                        token_end > char_start
                        and token_start < char_end
                        and prompt[token_start:token_end].strip()
                    ):
                        instruction_mask[row, column] = True
                instruction_mask[row] &= batch["attention_mask"][row].bool()
                if not instruction_mask[row].any():
                    raise ValueError(
                        "Instruction was fully truncated; increase max_text_tokens"
                    )
            batch["instruction_token_mask"] = instruction_mask
            if self.include_segment_token_ids:
                segment_token_ids = torch.full_like(
                    text_encoded["input_ids"], -1, dtype=torch.long
                )
                num_segments = torch.zeros(len(examples), dtype=torch.long)
                for row, (prompt, instruction) in enumerate(
                    zip(prompts, segment_sources)
                ):
                    instruction_start = prompt.find(instruction)
                    cached = (
                        None if self.segment_metadata is None
                        else self.segment_metadata.get(instruction)
                    )
                    if cached is None:
                        # Metadata is an offline acceleration/cache, not the
                        # source of truth.  A mixed temporal batch can contain
                        # padding rows or a real instruction omitted by the
                        # cache; both must use the exact deterministic online
                        # segmenter instead of aborting an otherwise valid run.
                        if (
                            self.segment_metadata is not None
                            and instruction not in self._segment_metadata_fallbacks
                        ):
                            warnings.warn(
                                "Instruction is absent from offline segment metadata; "
                                "falling back to deterministic segment_instruction",
                                RuntimeWarning,
                                stacklevel=2,
                            )
                            self._segment_metadata_fallbacks.add(instruction)
                        segments = segment_instruction(
                            instruction, max_segments=self.task_state_max_segments
                        )
                    else:
                        segments = [
                            InstructionSegment(
                                text=str(item["text"]),
                                char_start=int(item["char_start"]),
                                char_end=int(item["char_end"]),
                            )
                            for item in cached["segments"]
                        ]
                    num_segments[row] = len(segments)
                    for segment_index, segment in enumerate(segments):
                        span_start = instruction_start + segment.char_start
                        span_end = instruction_start + segment.char_end
                        for column, (token_start, token_end) in enumerate(
                            offsets[row].tolist()
                        ):
                            if (
                                instruction_mask[row, column]
                                and token_end > span_start
                                and token_start < span_end
                            ):
                                if segment_token_ids[row, column] >= 0:
                                    raise ValueError("Segment token spans overlap")
                                segment_token_ids[row, column] = segment_index
                    if torch.any(
                        instruction_mask[row] & (segment_token_ids[row] < 0)
                    ):
                        raise ValueError("Segment spans leave uncovered instruction tokens")
                batch["segment_token_ids"] = segment_token_ids
                batch["num_segments"] = num_segments
        return batch


class EpisodeSequenceDataset(Dataset[dict]):
    """Fixed-length, in-order episode windows for M2 truncated BPTT."""

    def __init__(
        self,
        manifest_path: str | Path,
        num_frames: int = 4,
        sequence_length: int = 8,
        history_stride: int = 2,
        missing_history: str = "error",
        progress_annotations_dir: str | Path | None = None,
        burn_in: int = 0,
        loss_window: int | None = None,
    ) -> None:
        if sequence_length < 2:
            raise ValueError("EpisodeSequenceDataset requires sequence_length >= 2")
        self.frames = MultiFrameClassificationDataset(
            manifest_path,
            num_frames=num_frames,
            history_stride=history_stride,
            missing_history=missing_history,
            progress_annotations_dir=progress_annotations_dir,
        )
        self.sequence_length = sequence_length
        self.burn_in = int(burn_in)
        self.loss_window = int(loss_window or (sequence_length - self.burn_in))
        if self.burn_in < 0 or self.loss_window <= 0:
            raise ValueError("Sequence burn-in/loss-window values are invalid")
        if self.burn_in + self.loss_window != self.sequence_length:
            raise ValueError("burn_in + loss_window must equal sequence_length")
        self._windows: list[
            tuple[list[int], list[bool], list[bool], str, bool, bool]
        ] = []
        self.episode_windows: list[list[int]] = []
        for episode_id, episode in self.frames._episodes.items():
            indices = [index for index, _ in episode]
            episode_window_indices: list[int] = []
            for target_start in range(0, len(indices), self.loss_window):
                context_start = max(0, target_start - self.burn_in)
                context = indices[context_start:target_start]
                targets = indices[target_start:target_start + self.loss_window]
                left_pad = self.burn_in - len(context)
                right_pad = self.loss_window - len(targets)
                first_item = context[0] if context else targets[0]
                last_item = targets[-1] if targets else context[-1]
                window = (
                    [first_item] * left_pad + context + targets + [last_item] * right_pad
                )
                valid = (
                    [False] * left_pad
                    + [True] * len(context)
                    + [True] * len(targets)
                    + [False] * right_pad
                )
                loss_mask = (
                    [False] * self.burn_in
                    + [True] * len(targets)
                    + [False] * right_pad
                )
                episode_window_indices.append(len(self._windows))
                self._windows.append(
                    (
                        window,
                        valid,
                        loss_mask,
                        episode_id,
                        target_start == 0,
                        target_start + self.loss_window >= len(indices),
                    )
                )
            self.episode_windows.append(episode_window_indices)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> dict:
        if index == -1:
            padding_sample = {
                "images": [Image.new("RGB", (224, 224), color=(0, 0, 0)) for _ in range(self.frames.num_frames)],
                "frame_valid_mask": [False] * self.frames.num_frames,
                "instruction": "padding",
                "action_id": 0,
                **(
                    {
                        "action_chunk_targets": [0]
                        + [-100] * (self.frames.action_chunk_horizon - 1)
                    }
                    if self.frames.action_chunk_horizon > 1
                    else {}
                ),
                "episode_id": "__tm_stream_padding__",
                "step_id": -1,
                "image_path": "<stream-padding>",
                "prev_action": 8,
                "previous_actions_text": [None, None, None],
                "relative_heading_degrees": 0,
                "relative_altitude_meters": 0,
                "relative_forward_meters": 0.0,
                "relative_left_meters": 0.0,
            }
            return {
                "samples": [dict(padding_sample) for _ in range(self.sequence_length)],
                "sequence_mask": [False] * self.sequence_length,
                "loss_mask": [False] * self.sequence_length,
                "episode_id": "__tm_stream_padding__",
                "stream_reset": True,
                "stream_end": True,
            }
        window, valid, loss_mask, episode_id, stream_reset, stream_end = self._windows[index]
        samples: list[dict[str, Any]] = []
        last_real: dict[str, Any] | None = None
        first_real = self.frames[next(item for item, is_valid in zip(window, valid) if is_valid)]
        for item, is_valid in zip(window, valid):
            if is_valid:
                last_real = self.frames[item]
                samples.append(last_real)
                continue
            if last_real is None:
                padding = dict(first_real)
            else:
                padding = dict(last_real)
            padding["images"] = [
                Image.new("RGB", image.size, color=(0, 0, 0))
                for image in padding["images"]
            ]
            padding["frame_valid_mask"] = [False] * len(padding["images"])
            padding["image_path"] = "<stream-padding>"
            samples.append(padding)
        return {
            "samples": samples,
            "sequence_mask": valid,
            "loss_mask": loss_mask,
            "episode_id": episode_id,
            "stream_reset": stream_reset,
            "stream_end": stream_end,
        }


class TaskGRUEpisodeStreamDataset(Dataset[dict[str, Any]]):
    """Load sampler-provided TBPTT chunks without breaking episode streams.

    The matching sampler owns episode/lane scheduling.  A dataset key contains
    the exact frame indices and per-timestep reset/end flags for one batch lane,
    allowing an episode to end and a new episode to begin inside one chunk.
    """

    _KEY = "__task_gru_episode_stream__"

    def __init__(
        self,
        manifest_path: str | Path,
        num_frames: int = 1,
        tbptt_steps: int = 4,
        history_stride: int = 2,
        missing_history: str = "error",
        progress_annotations_dir: str | Path | None = None,
    ) -> None:
        if tbptt_steps <= 0:
            raise ValueError("TaskGRU tbptt_steps must be positive")
        self.frames = MultiFrameClassificationDataset(
            manifest_path,
            num_frames=num_frames,
            history_stride=history_stride,
            missing_history=missing_history,
            progress_annotations_dir=progress_annotations_dir,
        )
        self.tbptt_steps = int(tbptt_steps)
        self.episode_steps: list[list[int]] = [
            [index for index, _ in episode]
            for episode in self.frames._episodes.values()
        ]
        if any(not steps for steps in self.episode_steps):
            raise ValueError("TaskGRU stream dataset contains an empty episode")

    def __len__(self) -> int:
        # A batch_sampler supplies composite chunk keys; this length remains the
        # underlying epoch definition: every action frame is consumed once.
        return len(self.frames)

    def _padding_sample(self) -> dict[str, Any]:
        return {
            "images": [
                Image.new("RGB", (224, 224), color=(0, 0, 0))
                for _ in range(self.frames.num_frames)
            ],
            "frame_valid_mask": [False] * self.frames.num_frames,
            "instruction": "padding",
            "action_id": 0,
            "episode_id": "__task_gru_stream_padding__",
            "step_id": -1,
            "image_path": "<stream-padding>",
            "prev_action": 8,
            "previous_actions_text": [None, None, None],
            "relative_heading_degrees": 0,
            "relative_altitude_meters": 0,
            "relative_forward_meters": 0.0,
            "relative_left_meters": 0.0,
        }

    def __getitem__(self, key: Any) -> dict[str, Any]:
        if not isinstance(key, tuple) or len(key) != 6 or key[0] != self._KEY:
            raise TypeError("TaskGRU stream dataset requires a composite chunk key")
        _, indices, valid, reset, end, episode_ids = key
        if not all(
            len(values) == self.tbptt_steps
            for values in (indices, valid, reset, end, episode_ids)
        ):
            raise ValueError("TaskGRU stream chunk length mismatch")
        samples = [
            self.frames[item] if is_valid else self._padding_sample()
            for item, is_valid in zip(indices, valid)
        ]
        return {
            "samples": samples,
            "sequence_mask": list(valid),
            "loss_mask": list(valid),
            "episode_id": list(episode_ids),
            "stream_reset": list(reset),
            "stream_end": list(end),
        }


class TaskGRUEpisodeStreamBatchSampler(Sampler[list[tuple[Any, ...]]]):
    """Maintain independent episode streams in fixed batch lanes.

    A finished episode is immediately replaced inside the same TBPTT chunk.
    Reset/end masks prevent recurrent state leakage while avoiding artificial
    padding at ordinary episode boundaries.  Only the final exhausted lanes of
    an epoch are masked.
    """

    def __init__(
        self,
        dataset: TaskGRUEpisodeStreamDataset,
        batch_size: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 7,
        shuffle: bool = True,
    ) -> None:
        if batch_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("Invalid TaskGRU stream sampler configuration")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _episode_order(self, rank: int) -> list[int]:
        order = list(range(len(self.dataset.episode_steps)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        return order[rank::self.world_size]

    def _empty_key(self) -> tuple[Any, ...]:
        length = self.dataset.tbptt_steps
        return (
            self.dataset._KEY,
            tuple([-1] * length),
            tuple([False] * length),
            tuple([False] * length),
            tuple([False] * length),
            tuple(["__task_gru_stream_padding__"] * length),
        )

    def _rank_batches(self, rank: int) -> list[list[tuple[Any, ...]]]:
        episode_order = iter(self._episode_order(rank))
        lanes: list[list[int] | None] = [None] * self.batch_size
        lane_positions = [0] * self.batch_size

        def refill(lane: int) -> bool:
            try:
                episode_index = next(episode_order)
            except StopIteration:
                lanes[lane] = None
                lane_positions[lane] = 0
                return False
            lanes[lane] = self.dataset.episode_steps[episode_index]
            lane_positions[lane] = 0
            return True

        for lane in range(self.batch_size):
            refill(lane)

        batches: list[list[tuple[Any, ...]]] = []
        while any(steps is not None for steps in lanes):
            batch: list[tuple[Any, ...]] = []
            for lane in range(self.batch_size):
                indices: list[int] = []
                valid: list[bool] = []
                reset: list[bool] = []
                end: list[bool] = []
                episode_ids: list[str] = []
                for _ in range(self.dataset.tbptt_steps):
                    if lanes[lane] is None and not refill(lane):
                        indices.append(-1)
                        valid.append(False)
                        reset.append(False)
                        end.append(False)
                        episode_ids.append("__task_gru_stream_padding__")
                        continue
                    steps = lanes[lane]
                    assert steps is not None
                    position = lane_positions[lane]
                    frame_index = steps[position]
                    record = self.dataset.frames.records[frame_index]
                    is_first = position == 0
                    is_last = position + 1 == len(steps)
                    indices.append(frame_index)
                    valid.append(True)
                    reset.append(is_first)
                    end.append(is_last)
                    episode_ids.append(str(record.episode_id))
                    lane_positions[lane] += 1
                    if is_last:
                        lanes[lane] = None
                        lane_positions[lane] = 0
                batch.append(
                    (
                        self.dataset._KEY,
                        tuple(indices),
                        tuple(valid),
                        tuple(reset),
                        tuple(end),
                        tuple(episode_ids),
                    )
                )
            batches.append(batch)
        return batches

    def __iter__(self):
        local_batches = self._rank_batches(self.rank)
        yield from local_batches
        empty_batch = [self._empty_key() for _ in range(self.batch_size)]
        for _ in range(len(self) - len(local_batches)):
            yield empty_batch

    def __len__(self) -> int:
        return max(
            len(self._rank_batches(rank)) for rank in range(self.world_size)
        )


class TemporalStreamBatchSampler(Sampler[list[int]]):
    """Keep one ordered episode stream in each batch lane.

    Episodes, rather than individual windows, are sharded across DDP ranks.
    A lane is refilled only after its current episode ends, so a detached
    compact memory-token state can be carried safely between adjacent windows.
    """

    def __init__(
        self,
        dataset: EpisodeSequenceDataset,
        batch_size: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 7,
        shuffle: bool = True,
    ) -> None:
        if batch_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("Invalid stream sampler batch/rank configuration")
        self.dataset = dataset
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _episode_order(self, rank: int | None = None) -> list[int]:
        rank = self.rank if rank is None else rank
        order = list(range(len(self.dataset.episode_windows)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        return order[rank::self.world_size]

    def _rank_batches(self, rank: int) -> list[list[int]]:
        episode_order = iter(self._episode_order(rank))
        lanes: list[tuple[int, int]] = []
        for _ in range(self.batch_size):
            try:
                lanes.append((next(episode_order), 0))
            except StopIteration:
                break
        batches: list[list[int]] = []
        while lanes:
            batch: list[int] = []
            next_lanes: list[tuple[int, int]] = []
            for episode_index, window_position in lanes:
                windows = self.dataset.episode_windows[episode_index]
                batch.append(windows[window_position])
                window_position += 1
                if window_position < len(windows):
                    next_lanes.append((episode_index, window_position))
                else:
                    try:
                        next_lanes.append((next(episode_order), 0))
                    except StopIteration:
                        pass
            batch.extend([-1] * (self.batch_size - len(batch)))
            batches.append(batch)
            lanes = next_lanes
        return batches

    def __iter__(self):
        local_batches = self._rank_batches(self.rank)
        target_length = len(self)
        yield from local_batches
        for _ in range(target_length - len(local_batches)):
            yield [-1] * self.batch_size

    def __len__(self) -> int:
        return max(len(self._rank_batches(rank)) for rank in range(self.world_size))


class EpisodeSequenceCollator:
    """Collate episode windows into [B, T, ...] tensors while retaining time order."""

    def __init__(self, collator: MultiFrameCollator, sequence_length: int) -> None:
        self.collator = collator
        self.sequence_length = sequence_length

    def __call__(self, examples: list[dict]) -> dict[str, Any]:
        per_step = [self.collator([example["samples"][step] for example in examples]) for step in range(self.sequence_length)]
        batch: dict[str, Any] = {}
        tensor_keys = [
            "input_ids", "attention_mask", "pixel_values", "frame_valid_mask",
            "action_id", "prev_actions",
        ]
        tensor_keys += [
            key for key in (
                "progress_targets", "progress_valid_mask", "instruction_token_mask",
                "segment_token_ids", "num_segments",
            )
            if key in per_step[0]
        ]
        for key in tensor_keys:
            values = [step[key] for step in per_step]
            shapes = {tuple(value.shape) for value in values}
            if len(shapes) > 1:
                # Prompt prefixes can make the tokenizer's per-step padding
                # differ inside a temporal window.  Pad text-like tensors to
                # the window maximum before adding the time dimension.
                if any(value.ndim < 1 for value in values):
                    raise RuntimeError(f"Cannot temporally pad scalar key {key}")
                max_length = max(value.shape[-1] for value in values)
                pad_value = -1 if key == "segment_token_ids" else 0
                values = [
                    torch.nn.functional.pad(
                        value, (max_length - value.shape[-1], 0), value=pad_value
                    )
                    for value in values
                ]
            batch[key] = torch.stack(values, dim=1)
        batch["sequence_mask"] = torch.tensor([example["sequence_mask"] for example in examples], dtype=torch.bool)
        batch["loss_mask"] = torch.tensor(
            [example["loss_mask"] for example in examples], dtype=torch.bool
        )
        batch["episode_id"] = [example["episode_id"] for example in examples]
        batch["stream_reset"] = torch.tensor(
            [example["stream_reset"] for example in examples], dtype=torch.bool
        )
        batch["stream_end"] = torch.tensor(
            [example["stream_end"] for example in examples], dtype=torch.bool
        )
        if batch["stream_reset"].ndim not in (1, 2):
            raise ValueError("stream_reset must be [B] or [B,T]")
        if batch["stream_end"].shape != batch["stream_reset"].shape:
            raise ValueError("stream reset/end masks must have identical shape")
        return batch
