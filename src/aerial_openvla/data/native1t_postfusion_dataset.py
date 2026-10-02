"""Datasets for Native1T-PostFusion-H16 Stage 1 and Stage 2."""

from __future__ import annotations

import hashlib
import json
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from aerial_openvla.data.manifest import load_step_records
from aerial_openvla.data.instruction_segments import InstructionSegment, segment_instruction
from aerial_openvla.data.multiframe_dataset import (
    prepend_relative_state_text,
    truncate_instruction_to_token_budget,
    update_relative_planar_state,
    update_relative_state,
)
from aerial_openvla.data.event_h16 import require_event_h16_cache_calibration


def manifest_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
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
                raise ValueError(f"Duplicate step {record.episode_id}/{record.step_id}")
            steps[record.step_id] = index
        for episode_id, steps in self.by_episode.items():
            self.first_by_episode[episode_id] = steps[min(steps)]


class Native1TStage1Dataset(Dataset[dict[str, Any]]):
    """I_0 + seven sampled recent-history frames + I_t."""

    def __init__(
        self,
        manifest_path: str | Path,
        seed: int = 7,
        use_relative_state_text: bool = False,
        use_relative_planar_state_text: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.records = load_step_records(self.manifest_path)
        self.index = _EpisodeIndex(self.records)
        self.seed = int(seed)
        self.use_relative_state_text = bool(use_relative_state_text)
        self.use_relative_planar_state_text = bool(use_relative_planar_state_text)
        if self.use_relative_planar_state_text and not self.use_relative_state_text:
            raise ValueError("Planar relative text requires relative-state text")

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, record_index: int) -> Image.Image:
        path = self.records[record_index].resolve_image(self.manifest_path)
        with Image.open(path) as source:
            return source.convert("RGB")

    def __getitem__(self, item: int) -> dict[str, Any]:
        record = self.records[item]
        episode_steps = self.index.by_episode[record.episode_id]
        candidates = [
            episode_steps[step]
            for step in range(record.step_id - 16, record.step_id)
            if step in episode_steps
        ]
        selected = sorted(random.sample(candidates, min(7, len(candidates))))
        # Left padding preserves chronological alignment inside the seven slots.
        padding = 7 - len(selected)
        first_index = self.index.first_by_episode[record.episode_id]
        frame_indices = [first_index] + [first_index] * padding + selected + [item]
        valid = [True] + [False] * padding + [True] * len(selected) + [True]
        if len(frame_indices) != 9:
            raise RuntimeError("Native1T Stage 1 must produce exactly nine frames")
        result = {
            "memory_images": [self._load_image(index) for index in frame_indices],
            "memory_valid_mask": torch.tensor(valid, dtype=torch.bool),
            # Audit-only metadata; the collator never feeds these IDs to the model.
            "memory_frame_indices": frame_indices,
            "instruction": record.instruction,
            "action_id": int(record.action_id),
            "episode_id": record.episode_id,
            "step_id": record.step_id,
        }
        if self.use_relative_state_text:
            (
                result["relative_heading_degrees"],
                result["relative_altitude_meters"],
                result["relative_forward_meters"],
                result["relative_left_meters"],
            ) = _relative_state_before_step(
                self.records,
                episode_steps,
                record.step_id,
                planar=self.use_relative_planar_state_text,
            )
        return result


class Native1TStage2Dataset(Dataset[dict[str, Any]]):
    def __init__(self, manifest_path: str | Path, cache_dir: str | Path,
                 tokenizer_checkpoint: str | Path,
                 feature_dim: int = 2176,
                 tokenizer_filename: str = "native1t_postfusion.pt",
                 history_length: int = 16,
                 history_mode: str = "fifo",
                 event_history_dir: str | Path | None = None,
                 current_vision_cache_dir: str | Path | None = None,
                 current_vision_cache_contract: str | None = None,
                 event_tau_local: float = 0.757488,
                 event_tau_anchor: float = 0.765,
                 event_max_length: int = 8,
                 event_recent_protected: int = 4,
                 action_chunk_horizon: int = 1,
                 use_action_history: bool = False,
                 use_relative_state_text: bool = False,
                 use_relative_planar_state_text: bool = False) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.cache_dir = Path(cache_dir).resolve()
        self.records = load_step_records(self.manifest_path)
        self.index = _EpisodeIndex(self.records)
        self.feature_dim = int(feature_dim)
        self.history_length = int(history_length)
        self.history_mode = str(history_mode)
        if self.history_length not in (8, 16, 31):
            raise ValueError(
                "Native1T Stage 2 supports fixed history lengths 8, 16 or 31"
            )
        if self.history_mode not in {"fifo", "event_h16_v1"}:
            raise ValueError(
                f"Unknown Native1T history mode: {self.history_mode!r}"
            )
        if action_chunk_horizon not in (1, 4):
            raise ValueError("Native1T Stage 2 action_chunk_horizon must be 1 or 4")
        self.action_chunk_horizon = int(action_chunk_horizon)
        self.use_action_history = bool(use_action_history)
        self.use_relative_state_text = bool(use_relative_state_text)
        self.use_relative_planar_state_text = bool(use_relative_planar_state_text)
        if self.use_relative_planar_state_text and not self.use_relative_state_text:
            raise ValueError("Planar relative text requires relative-state text")
        metadata = json.loads((self.cache_dir / "metadata.json").read_text())
        if not metadata.get("complete", False):
            raise ValueError("Native1T cache is incomplete")
        if metadata.get("shape") != [len(self.records), self.feature_dim]:
            raise ValueError(f"Native1T cache shape mismatch: {metadata.get('shape')}")
        if metadata.get("manifest_sha256") != manifest_sha256(self.manifest_path):
            raise ValueError("Native1T cache manifest hash mismatch")
        if metadata.get("feature_space") != "frozen_memory_tokenizer_pre_projector":
            raise ValueError("Native1T cache feature-space mismatch")
        tokenizer_file = Path(tokenizer_checkpoint) / tokenizer_filename
        if not tokenizer_file.is_file():
            raise FileNotFoundError(f"Missing Native1T tokenizer checkpoint: {tokenizer_file}")
        if metadata.get("tokenizer_sha256") != manifest_sha256(tokenizer_file):
            raise ValueError("Native1T cache tokenizer hash mismatch")
        self.metadata = metadata
        features_path = self.cache_dir / "features.npy"
        if not features_path.is_file():
            raise FileNotFoundError(f"Native1T feature file is missing: {features_path}")
        features = np.load(features_path, allow_pickle=False)
        if list(features.shape) != [len(self.records), self.feature_dim]:
            raise ValueError("Native1T feature file shape mismatch")
        if features.dtype != np.float16:
            raise ValueError("Native1T feature file dtype mismatch")
        features.setflags(write=False)
        self._features = features
        self.current_vision_cache_dir = (
            None if current_vision_cache_dir is None
            else Path(current_vision_cache_dir).resolve()
        )
        self.current_vision_cache_contract = (
            None
            if current_vision_cache_contract is None
            else str(current_vision_cache_contract)
        )
        self._current_premerger_features = None
        self._manifest_to_current_cache = None
        self._current_premerger_path_hashes = None
        if self.current_vision_cache_dir is not None:
            index_meta = json.loads(
                (self.current_vision_cache_dir / "index_meta.json").read_text()
            )
            cache_meta = json.loads(
                (self.current_vision_cache_dir / "cache_meta.json").read_text()
            )
            expected_hash = manifest_sha256(self.manifest_path)
            if not index_meta.get("complete") or not cache_meta.get("complete"):
                raise ValueError("Qwen pre-merger cache/index is incomplete")
            if index_meta.get("manifest_sha256") != expected_hash:
                raise ValueError("Qwen pre-merger index manifest hash mismatch")
            if cache_meta.get("manifest_sha256") != expected_hash:
                raise ValueError("Qwen pre-merger cache manifest hash mismatch")
            if index_meta.get("manifest_rows") != len(self.records):
                raise ValueError("Qwen pre-merger mapping row count mismatch")
            if cache_meta.get("shape") != [index_meta.get("cache_rows"), 196, 768]:
                raise ValueError("Qwen pre-merger feature shape contract mismatch")
            actual_contract = index_meta.get("format")
            if (
                self.current_vision_cache_contract is not None
                and actual_contract != self.current_vision_cache_contract
            ):
                raise ValueError(
                    "Qwen pre-merger cache contract mismatch: "
                    f"expected {self.current_vision_cache_contract!r}, "
                    f"got {actual_contract!r}"
                )
            if actual_contract == "manifest_row_image_path_v2":
                if index_meta.get("cache_rows") != len(self.records):
                    raise ValueError(
                        "Manifest-row Qwen pre-merger cache must have one row "
                        "per manifest row"
                    )
                if not index_meta.get("mapping_identity", False):
                    raise ValueError(
                        "Manifest-row Qwen pre-merger cache requires identity mapping"
                    )
                if index_meta.get("path_hash_algorithm") != "sha256":
                    raise ValueError(
                        "Manifest-row Qwen pre-merger cache requires SHA256 paths"
                    )
                if (
                    cache_meta.get("index_contract")
                    != "manifest_row_image_path_v2"
                ):
                    raise ValueError(
                        "Qwen pre-merger cache/index contract disagreement"
                    )
                validation_path = (
                    self.current_vision_cache_dir
                    / "validation_manifestrow_v2.json"
                )
                if not validation_path.is_file():
                    raise ValueError(
                        "Manifest-row Qwen pre-merger cache lacks validation report"
                    )
                validation = json.loads(validation_path.read_text())
                if (
                    not validation.get("complete")
                    or validation.get("manifest_sha256") != expected_hash
                    or validation.get("contract")
                    != "manifest_row_image_path_v2"
                ):
                    raise ValueError(
                        "Manifest-row Qwen pre-merger validation report mismatch"
                    )
            self.current_vision_cache_metadata = cache_meta
        self._event_history_indices = None
        self.event_metadata = None
        if self.history_mode == "event_h16_v1":
            calibration = require_event_h16_cache_calibration(
                self.cache_dir,
                tau_local=event_tau_local,
                tau_anchor=event_tau_anchor,
                max_event_length=event_max_length,
                max_nodes=self.history_length,
                recent_protected=event_recent_protected,
                verify_cache_content=True,
            )
            calibrated = calibration["thresholds"]
            if event_history_dir is None:
                raise ValueError(
                    "Event-H16 Stage 2 requires native1t_event_*_indices"
                )
            event_dir = Path(event_history_dir).resolve()
            event_metadata = json.loads((event_dir / "metadata.json").read_text())
            expected = {
                "complete": True,
                "format": "event_h16_indices_v1",
                "shape": [len(self.records), self.history_length],
                "manifest_sha256": metadata["manifest_sha256"],
                "tau_local": float(calibrated["tau_local"]),
                "tau_anchor": float(calibrated["tau_anchor"]),
                "max_event_length": int(calibrated["max_event_length"]),
                "max_nodes": self.history_length,
                "recent_protected": int(calibrated["recent_protected"]),
                "strict_predict_before_append": True,
                "history_order": "oldest_to_newest_left_padded",
                "representative": "real_stage1_memory_token_no_averaging",
                "cache_calibration_sha256": calibration["_sha256"],
                "cache_calibration_format": calibration["format"],
            }
            mismatches = {
                key: (event_metadata.get(key), value)
                for key, value in expected.items()
                if event_metadata.get(key) != value
            }
            source_metadata = event_metadata.get("source_cache_metadata", {})
            if source_metadata.get("tokenizer_sha256") != metadata.get(
                "tokenizer_sha256"
            ):
                mismatches["source_cache_tokenizer_sha256"] = (
                    source_metadata.get("tokenizer_sha256"),
                    metadata.get("tokenizer_sha256"),
                )
            if mismatches:
                raise ValueError(
                    f"Event-H16 index contract mismatch: {mismatches}"
                )
            indices_path = event_dir / "history_indices.npy"
            if event_metadata.get("history_indices_sha256") != manifest_sha256(
                indices_path
            ):
                raise ValueError("Event-H16 history index hash mismatch")
            self.event_history_dir = event_dir
            self.event_metadata = event_metadata
            self.event_calibration = calibration

    @property
    def features(self):
        return self._features

    @property
    def current_premerger_features(self):
        if self.current_vision_cache_dir is None:
            raise RuntimeError("Current Qwen pre-merger cache is not configured")
        if self._current_premerger_features is None:
            self._current_premerger_features = np.load(
                self.current_vision_cache_dir / "features.npy", mmap_mode="r"
            )
        return self._current_premerger_features

    @property
    def manifest_to_current_cache(self):
        if self.current_vision_cache_dir is None:
            raise RuntimeError("Current Qwen pre-merger cache is not configured")
        if self._manifest_to_current_cache is None:
            self._manifest_to_current_cache = np.load(
                self.current_vision_cache_dir / "manifest_to_cache.npy",
                mmap_mode="r",
            )
            if self.current_vision_cache_contract == "manifest_row_image_path_v2":
                mapping = self._manifest_to_current_cache
                if mapping.shape != (len(self.records),):
                    raise ValueError(
                        "Manifest-row Qwen pre-merger mapping shape mismatch"
                    )
                for start in range(0, len(mapping), 1_000_000):
                    stop = min(start + 1_000_000, len(mapping))
                    expected = np.arange(start, stop, dtype=mapping.dtype)
                    if not np.array_equal(mapping[start:stop], expected):
                        raise ValueError(
                            "Manifest-row Qwen pre-merger mapping is not identity"
                        )
        return self._manifest_to_current_cache

    @property
    def current_premerger_path_hashes(self):
        if self.current_vision_cache_dir is None:
            raise RuntimeError("Current Qwen pre-merger cache is not configured")
        if self._current_premerger_path_hashes is None:
            self._current_premerger_path_hashes = np.load(
                self.current_vision_cache_dir / "cache_image_path_sha256.npy",
                mmap_mode="r",
            )
            if self._current_premerger_path_hashes.shape != (len(self.records), 32):
                raise ValueError(
                    "Manifest-row Qwen pre-merger path-hash shape mismatch"
                )
        return self._current_premerger_path_hashes

    def _validate_current_premerger_record(
        self,
        item: int,
        record,
        cache_row: int,
    ) -> None:
        if self.current_vision_cache_contract != "manifest_row_image_path_v2":
            return
        if cache_row != item:
            raise RuntimeError(
                "Manifest-row Qwen pre-merger cache returned a non-identity row"
            )
        image_path = str(record.resolve_image(self.manifest_path))
        digest = hashlib.sha256(image_path.encode("utf-8")).digest()
        cached = self.current_premerger_path_hashes[cache_row]
        if cached.tobytes() != digest:
            raise RuntimeError(
                "Manifest-row Qwen pre-merger cache image-path mismatch "
                f"at manifest row {item}"
            )

    @property
    def event_history_indices(self):
        if self.history_mode != "event_h16_v1":
            raise RuntimeError("FIFO-H16 has no event history index artifact")
        if self._event_history_indices is None:
            self._event_history_indices = np.load(
                self.event_history_dir / "history_indices.npy", mmap_mode="r"
            )
            if list(self._event_history_indices.shape) != [
                len(self.records), self.history_length
            ]:
                raise ValueError("Event-H16 history index array has the wrong shape")
        return self._event_history_indices

    def __len__(self) -> int:
        return len(self.records)

    def _action_chunk_targets(
        self,
        record,
        episode_steps: dict[int, int],
    ) -> list[int]:
        """Return strictly in-episode AC targets starting at the current step.

        AC4 predicts ``[a_t, a_t+1, a_t+2, a_t+3]``.  A missing consecutive
        step, the first STOP, and the episode tail all terminate supervision;
        they never borrow labels from a later episode or another trajectory.
        """
        targets = [-100] * self.action_chunk_horizon
        for horizon_index in range(self.action_chunk_horizon):
            future_index = episode_steps.get(record.step_id + horizon_index)
            if future_index is None:
                break
            future_record = self.records[future_index]
            if future_record.episode_id != record.episode_id:
                raise RuntimeError("Native1T AC target crossed an episode boundary")
            if future_record.step_id != record.step_id + horizon_index:
                raise RuntimeError("Native1T AC target has a non-consecutive step")
            targets[horizon_index] = int(future_record.action_id)
            if future_record.action_id == 0:  # include the first STOP, then mask
                break
        if targets[0] != int(record.action_id):
            raise RuntimeError("Native1T AC target slot 0 must equal action_id")
        return targets

    def __getitem__(self, item: int) -> dict[str, Any]:
        record = self.records[item]
        history = np.zeros(
            (self.history_length, self.feature_dim), dtype=np.float16
        )
        valid = np.zeros(self.history_length, dtype=np.bool_)
        episode_steps = self.index.by_episode[record.episode_id]
        if self.history_mode == "event_h16_v1":
            sources = np.asarray(self.event_history_indices[item], dtype=np.int64)
            for slot, source in enumerate(sources):
                if source < 0:
                    continue
                if source >= item:
                    raise RuntimeError(
                        f"Event-H16 leaked current/future row {source} into {item}"
                    )
                source_record = self.records[int(source)]
                if source_record.episode_id != record.episode_id:
                    raise RuntimeError("Event-H16 crossed an episode boundary")
                if source_record.step_id >= record.step_id:
                    raise RuntimeError("Event-H16 history is not strictly causal")
                history[slot] = self.features[source]
                valid[slot] = True
        else:
            for slot, step in enumerate(range(
                record.step_id - self.history_length, record.step_id
            )):
                source = episode_steps.get(step)
                if source is not None:
                    history[slot] = self.features[source]
                    valid[slot] = True
        result = {
            "memory_features": torch.from_numpy(history),
            "memory_valid_mask": torch.from_numpy(valid),
            "instruction": record.instruction,
            "action_id": int(record.action_id),
            **(
                {
                    "action_chunk_targets": self._action_chunk_targets(
                        record, episode_steps
                    )
                }
                if self.action_chunk_horizon > 1
                else {}
            ),
            "episode_id": record.episode_id,
            "step_id": record.step_id,
        }
        if self.current_vision_cache_dir is None:
            image_path = record.resolve_image(self.manifest_path)
            with Image.open(image_path) as source:
                result["image"] = source.convert("RGB")
        else:
            cache_row = int(self.manifest_to_current_cache[item])
            self._validate_current_premerger_record(item, record, cache_row)
            current = np.array(
                self.current_premerger_features[cache_row], copy=True
            )
            if current.shape != (196, 768):
                raise RuntimeError("Qwen pre-merger cache row has the wrong shape")
            result["current_premerger_features"] = torch.from_numpy(current)
            result["current_visual_cache_row"] = cache_row
        if self.use_action_history:
            action_history = torch.zeros(4, dtype=torch.long)
            action_history_valid_mask = torch.zeros(4, dtype=torch.bool)
            for slot, step in enumerate(range(record.step_id - 4, record.step_id)):
                source = episode_steps.get(step)
                if source is not None:
                    action_history[slot] = int(self.records[source].action_id)
                    action_history_valid_mask[slot] = True
            result["native1t_action_history"] = action_history
            result["native1t_action_history_valid_mask"] = action_history_valid_mask
        if self.use_relative_state_text:
            (
                result["relative_heading_degrees"],
                result["relative_altitude_meters"],
                result["relative_forward_meters"],
                result["relative_left_meters"],
            ) = _relative_state_before_step(
                self.records,
                episode_steps,
                record.step_id,
                planar=self.use_relative_planar_state_text,
            )
        return result


def _relative_state_before_step(
    records,
    episode_steps: dict[int, int],
    current_step: int,
    *,
    planar: bool,
) -> tuple[int, int, float, float]:
    heading_degrees = 0
    altitude_meters = 0
    forward_meters = 0.0
    left_meters = 0.0
    for step in sorted(episode_steps):
        if step >= current_step:
            break
        action_id = int(records[episode_steps[step]].action_id)
        if planar:
            (
                heading_degrees,
                altitude_meters,
                forward_meters,
                left_meters,
            ) = update_relative_planar_state(
                heading_degrees,
                altitude_meters,
                forward_meters,
                left_meters,
                action_id,
            )
        else:
            heading_degrees, altitude_meters = update_relative_state(
                heading_degrees, altitude_meters, action_id
            )
    return heading_degrees, altitude_meters, forward_meters, left_meters


class Native1TCollator:
    def __init__(
        self,
        processor,
        prompt_template: str,
        max_text_tokens: int,
        stage: int,
        instruction_only_text_budget: bool = False,
        include_segment_token_ids: bool = False,
        include_instruction_token_mask: bool = False,
        task_state_max_segments: int = 10,
        task_state_metadata_path: str | None = None,
        use_relative_state_text: bool = False,
        use_relative_planar_state_text: bool = False,
        action_chunk_horizon: int = 1,
    ):
        self.processor = processor
        self.prompt_template = prompt_template
        self.max_text_tokens = int(max_text_tokens)
        self.stage = int(stage)
        self.instruction_only_text_budget = bool(instruction_only_text_budget)
        self.include_segment_token_ids = bool(include_segment_token_ids)
        self.include_instruction_token_mask = bool(include_instruction_token_mask)
        self.task_state_max_segments = int(task_state_max_segments)
        self.use_relative_state_text = bool(use_relative_state_text)
        self.use_relative_planar_state_text = bool(use_relative_planar_state_text)
        if action_chunk_horizon not in (1, 4):
            raise ValueError("Native1T action_chunk_horizon must be 1 or 4")
        if action_chunk_horizon > 1 and self.stage != 2:
            raise ValueError("Native1T AC4 is supported only for Stage 2")
        self.action_chunk_horizon = int(action_chunk_horizon)
        if self.use_relative_planar_state_text and not self.use_relative_state_text:
            raise ValueError("Planar relative text requires relative-state text")
        if self.instruction_only_text_budget and self.use_relative_state_text:
            raise ValueError("Instruction-only text budget requires raw non-Relative input")
        self.segment_metadata: dict[str, dict[str, object]] | None = None
        self._segment_metadata_fallbacks: set[str] = set()
        if task_state_metadata_path is not None:
            with Path(task_state_metadata_path).open("r", encoding="utf-8") as handle:
                self.segment_metadata = json.load(handle)

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        image_processor = getattr(self.processor, "image_processor", self.processor)
        source_instructions = []
        rendered_instructions = []
        for example in examples:
            source_instruction = str(example["instruction"]).strip()
            if self.instruction_only_text_budget:
                source_instruction = truncate_instruction_to_token_budget(
                    self.processor.tokenizer, source_instruction, self.max_text_tokens
                )
            instruction = source_instruction
            if self.use_relative_state_text:
                instruction = prepend_relative_state_text(
                    instruction,
                    int(example["relative_heading_degrees"]),
                    int(example["relative_altitude_meters"]),
                    (
                        float(example["relative_forward_meters"])
                        if self.use_relative_planar_state_text
                        else None
                    ),
                    (
                        float(example["relative_left_meters"])
                        if self.use_relative_planar_state_text
                        else None
                    ),
                )
            rendered_instructions.append(instruction)
            source_instructions.append(source_instruction)
        prompts = [
            self.prompt_template.format(instruction=instruction)
            for instruction in rendered_instructions
        ]
        tokenizer_kwargs = dict(
            padding=True,
            truncation=not self.instruction_only_text_budget,
            return_tensors="pt",
        )
        if not self.instruction_only_text_budget:
            tokenizer_kwargs["max_length"] = self.max_text_tokens
        if self.include_instruction_token_mask or self.include_segment_token_ids:
            tokenizer_kwargs["return_offsets_mapping"] = True
        try:
            text = self.processor.tokenizer(prompts, **tokenizer_kwargs)
        except (NotImplementedError, TypeError) as exc:
            raise RuntimeError(
                "Native1T SegTS requires a fast tokenizer with offset mappings"
            ) from exc
        offsets = text.pop("offset_mapping", None)
        batch = {
            "input_ids": text["input_ids"],
            "attention_mask": text.get("attention_mask", torch.ones_like(text["input_ids"])),
            "action_id": torch.tensor([e["action_id"] for e in examples], dtype=torch.long),
            "episode_id": [e["episode_id"] for e in examples],
            "step_id": [e["step_id"] for e in examples],
            "native1t_memory_valid_mask": torch.stack([
                e["memory_valid_mask"] for e in examples
            ]),
        }
        if self.action_chunk_horizon > 1:
            targets = [example["action_chunk_targets"] for example in examples]
            if any(len(target) != self.action_chunk_horizon for target in targets):
                raise RuntimeError("Native1T AC target horizon mismatch in collator")
            if any(
                int(target[0]) != int(example["action_id"])
                for target, example in zip(targets, examples)
            ):
                raise RuntimeError("Native1T AC target slot 0 mismatch in collator")
            batch["action_chunk_targets"] = torch.tensor(targets, dtype=torch.long)
        if self.include_instruction_token_mask and not self.include_segment_token_ids:
            if offsets is None:
                raise RuntimeError("Tokenizer did not return offset_mapping")
            instruction_mask = torch.zeros_like(
                batch["attention_mask"], dtype=torch.bool
            )
            for row, (prompt, instruction) in enumerate(
                zip(prompts, source_instructions)
            ):
                instruction_start = prompt.find(instruction)
                if instruction_start < 0:
                    raise ValueError("Rendered prompt does not contain its instruction")
                instruction_end = instruction_start + len(instruction)
                for column, (token_start, token_end) in enumerate(
                    offsets[row].tolist()
                ):
                    if (
                        token_end > instruction_start
                        and token_start < instruction_end
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
            if offsets is None:
                raise RuntimeError("Tokenizer did not return offset_mapping")
            instruction_mask = torch.zeros_like(
                batch["attention_mask"], dtype=torch.bool
            )
            segment_token_ids = torch.full_like(
                batch["input_ids"], -1, dtype=torch.long
            )
            num_segments = torch.zeros(len(examples), dtype=torch.long)
            for row, (prompt, instruction) in enumerate(zip(prompts, source_instructions)):
                instruction_start = prompt.find(instruction)
                if instruction_start < 0:
                    raise ValueError("Rendered prompt does not contain its instruction")
                instruction_end = instruction_start + len(instruction)
                for column, (token_start, token_end) in enumerate(offsets[row].tolist()):
                    if (
                        token_end > instruction_start
                        and token_start < instruction_end
                        and prompt[token_start:token_end].strip()
                    ):
                        instruction_mask[row, column] = True
                instruction_mask[row] &= batch["attention_mask"][row].bool()
                if not instruction_mask[row].any():
                    raise ValueError(
                        "Instruction was fully truncated; increase max_text_tokens"
                    )

                cached = (
                    None if self.segment_metadata is None
                    else self.segment_metadata.get(instruction)
                )
                if cached is None:
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
                if torch.any(instruction_mask[row] & (segment_token_ids[row] < 0)):
                    raise ValueError("Segment spans leave uncovered instruction tokens")
            batch["instruction_token_mask"] = instruction_mask
            batch["segment_token_ids"] = segment_token_ids
            batch["num_segments"] = num_segments
        if self.stage == 1:
            flat = [image for example in examples for image in example["memory_images"]]
            pixels = image_processor(flat, return_tensors="pt")["pixel_values"]
            batch["native1t_memory_pixel_values"] = pixels.reshape(
                len(examples), 9, *pixels.shape[1:]
            )
        else:
            batch["pixel_values"] = image_processor(
                [e["image"] for e in examples], return_tensors="pt"
            )["pixel_values"]
            batch["native1t_memory_features"] = torch.stack([
                e["memory_features"] for e in examples
            ])
            if "native1t_action_history" in examples[0]:
                batch["native1t_action_history"] = torch.stack([
                    e["native1t_action_history"] for e in examples
                ])
                batch["native1t_action_history_valid_mask"] = torch.stack([
                    e["native1t_action_history_valid_mask"] for e in examples
                ])
        return batch
