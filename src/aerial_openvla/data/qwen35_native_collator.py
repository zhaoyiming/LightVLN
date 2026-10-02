from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from aerial_openvla.data.qwen35_relative_state import (
    QWEN35_RELATIVE_NONE,
    append_qwen35_relative_state_text,
    validate_qwen35_relative_state_mode,
)


@dataclass(slots=True)
class Qwen35NativeClassificationCollator:
    """Build native Qwen3.5 multimodal inputs, including M-RoPE metadata."""

    processor: Any
    prompt_template: str
    max_text_tokens: int = 256
    image_size: int = 224
    use_relative_state_text: bool = False
    qwen35_relative_state_mode: str = QWEN35_RELATIVE_NONE
    include_segment_token_ids: bool = False
    truncate_text_keep_native_chat_end: bool = False
    task_state_max_segments: int = 10
    task_state_metadata_path: str | None = None
    allow_online_segment_fallback: bool = False
    _segment_metadata: dict[str, Any] | None = field(
        init=False, default=None, repr=False
    )
    _segment_metadata_fallbacks: set[str] = field(
        init=False, default_factory=set, repr=False
    )

    def __post_init__(self) -> None:
        self.qwen35_relative_state_mode = validate_qwen35_relative_state_mode(
            self.qwen35_relative_state_mode
        )
        if (
            self.use_relative_state_text
            and self.qwen35_relative_state_mode != QWEN35_RELATIVE_NONE
        ):
            raise ValueError(
                "Legacy prepended Relative and append-planar Relative are mutually exclusive"
            )
        if self.max_text_tokens <= 0:
            raise ValueError("max_text_tokens must be positive")
        native_pixels = self.image_size * self.image_size
        self.processor.image_processor.size["shortest_edge"] = native_pixels
        self.processor.image_processor.size["longest_edge"] = native_pixels
        if self.include_segment_token_ids:
            if self.task_state_max_segments != 10:
                raise ValueError("Qwen3.5 SegTS-v6 fixes task_state_max_segments=10")
            if not self.task_state_metadata_path:
                if not self.allow_online_segment_fallback:
                    raise ValueError("Qwen3.5 SegTS requires task_state_metadata_path")
            else:
                with Path(self.task_state_metadata_path).open("r", encoding="utf-8") as handle:
                    self._segment_metadata = json.load(handle)

    def _truncate_original_instruction(self, instruction: str) -> str:
        """Cap the *original instruction* alone, before prompt/state construction.

        The fixed Qwen prompt and any appended Relative block are deliberately
        outside this budget.  Offset slicing retains exactly the source text
        covered by the first ``max_text_tokens`` tokenizer tokens, rather than
        decoding token IDs and changing punctuation/whitespace.
        """
        instruction = instruction.strip()
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        tokenized = tokenizer(
            instruction,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
        input_ids = tokenized["input_ids"]
        offsets = tokenized["offset_mapping"]
        if len(input_ids) <= self.max_text_tokens:
            return instruction
        end = int(offsets[self.max_text_tokens - 1][1])
        if end <= 0:
            raise RuntimeError("Tokenizer returned an invalid instruction offset")
        return instruction[:end].rstrip()

    @staticmethod
    def _image_from_example(example: dict[str, Any]) -> Image.Image:
        if "image" in example:
            return example["image"]
        if "images" in example and len(example["images"]) == 1:
            return example["images"][0]
        raise ValueError("Qwen3.5 native input requires exactly one current RGB image")

    def _instruction_segments(self, instruction: str) -> list[tuple[int, int]]:
        cached = None if self._segment_metadata is None else self._segment_metadata.get(instruction)
        if cached is None:
            from aerial_openvla.data.instruction_segments import segment_instruction

            if instruction not in self._segment_metadata_fallbacks:
                warnings.warn(
                    "Instruction is absent from SegTS metadata; using deterministic "
                    "online segmentation",
                    RuntimeWarning,
                    stacklevel=3,
                )
                self._segment_metadata_fallbacks.add(instruction)
            return [
                (int(segment.char_start), int(segment.char_end))
                for segment in segment_instruction(
                    instruction, max_segments=self.task_state_max_segments
                )
            ]
        segments = cached.get("segments", [])
        result = [
            (int(segment["char_start"]), int(segment["char_end"]))
            for segment in segments
        ]
        if not result or len(result) > self.task_state_max_segments:
            raise ValueError("Invalid cached Qwen3.5 SegTS instruction segmentation")
        return result

    def _append_segment_token_ids(
        self,
        result: dict[str, Any],
        prompts: list[str],
        instructions: list[str],
    ) -> None:
        """Map original-instruction token spans onto processor-native IDs.

        The image chat template adds role/vision tokens around ``prompt``.  We
        tokenize the exact text prompt with offsets, locate its retained prefix
        in those native IDs, and fail if that correspondence is ever changed.
        Thus no image, role, punctuation-only, or padding token becomes a
        segment member.
        """
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        text = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        prompt_ids = text["input_ids"]
        prompt_mask = text.get("attention_mask", torch.ones_like(prompt_ids)).bool()
        offsets = text["offset_mapping"]
        input_ids = result["input_ids"]
        attention_mask = result["attention_mask"].bool()
        instruction_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        segment_token_ids = torch.full_like(input_ids, -1, dtype=torch.long)
        num_segments = torch.zeros(input_ids.shape[0], dtype=torch.long)

        for row, (prompt, instruction) in enumerate(zip(prompts, instructions)):
            actual = input_ids[row, attention_mask[row]].tolist()
            ids = prompt_ids[row, prompt_mask[row]].tolist()
            row_offsets = offsets[row, prompt_mask[row]].tolist()
            # With native-end-preserving truncation, the template sequence is
            # ``prompt-prefix + <|im_end|> + newline``.  Locate the longest
            # retained prefix rather than requiring the entire untruncated
            # prompt to remain contiguous.
            start = None
            retained = 0
            for candidate in range(len(actual)):
                maximum = min(len(ids), len(actual) - candidate)
                count = 0
                while count < maximum and actual[candidate + count] == ids[count]:
                    count += 1
                if count > retained:
                    start, retained = candidate, count
            if start is None or retained == 0:
                raise RuntimeError(
                    "Qwen3.5 chat template no longer preserves the prompt token "
                    "subsequence required for SegTS"
                )
            instruction_start = prompt.find(instruction)
            if instruction_start < 0:
                raise ValueError("Qwen3.5 prompt does not contain its original instruction")
            instruction_end = instruction_start + len(instruction)
            for index, (token_start, token_end) in enumerate(row_offsets[:retained]):
                if (
                    token_end > instruction_start
                    and token_start < instruction_end
                    and prompt[token_start:token_end].strip()
                ):
                    instruction_mask[row, start + index] = True
            if not instruction_mask[row].any():
                raise ValueError("Qwen3.5 instruction was fully truncated")
            segments = self._instruction_segments(instruction)
            num_segments[row] = len(segments)
            for segment_index, (segment_start, segment_end) in enumerate(segments):
                absolute_start = instruction_start + segment_start
                absolute_end = instruction_start + segment_end
                for index, (token_start, token_end) in enumerate(row_offsets[:retained]):
                    column = start + index
                    if (
                        instruction_mask[row, column]
                        and token_end > absolute_start
                        and token_start < absolute_end
                    ):
                        if segment_token_ids[row, column] >= 0:
                            raise ValueError("Qwen3.5 SegTS segment spans overlap")
                        segment_token_ids[row, column] = segment_index
            if torch.any(instruction_mask[row] & (segment_token_ids[row] < 0)):
                raise ValueError("Qwen3.5 SegTS leaves an instruction token uncovered")
        result["instruction_token_mask"] = instruction_mask
        result["segment_token_ids"] = segment_token_ids
        result["num_segments"] = num_segments

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        conversations = []
        prompts = []
        instructions = []
        for example in examples:
            instruction = self._truncate_original_instruction(example["instruction"])
            if self.use_relative_state_text:
                from aerial_openvla.data.multiframe_dataset import (
                    prepend_relative_state_text,
                )

                instruction = prepend_relative_state_text(
                    instruction,
                    int(example["relative_heading_degrees"]),
                    int(example["relative_altitude_meters"]),
                )
            prompt = self.prompt_template.format(instruction=instruction)
            if self.qwen35_relative_state_mode != QWEN35_RELATIVE_NONE:
                required_state = (
                    "relative_heading_degrees",
                    "relative_altitude_meters",
                    "relative_forward_meters",
                    "relative_left_meters",
                )
                missing_state = [key for key in required_state if key not in example]
                if missing_state:
                    raise ValueError(
                        "Append-planar Qwen3.5 Relative sample is missing: "
                        + ", ".join(missing_state)
                    )
                prompt = append_qwen35_relative_state_text(
                    prompt,
                    example["relative_heading_degrees"],
                    example["relative_altitude_meters"],
                    example["relative_forward_meters"],
                    example["relative_left_meters"],
                )
            instructions.append(instruction)
            prompts.append(prompt)
            conversations.append([{"role": "user", "content": [
                {"type": "image", "image": self._image_from_example(example)},
                {"type": "text", "text": prompt},
            ]}])
        encoded = self.processor.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pt",
            processor_kwargs={
                "padding": True,
                # max_text_tokens was already applied to the original
                # instruction.  Do not truncate the native wrapper, fixed
                # prompt, or append-planar state block.
                "truncation": False,
            },
        )
        required = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")
        missing = [key for key in required if key not in encoded]
        if missing:
            raise RuntimeError(f"Native Qwen3.5 processor omitted fields: {missing}")
        result = {
            **{key: encoded[key] for key in required},
            "action_id": torch.tensor([example["action_id"] for example in examples], dtype=torch.long),
            "episode_id": [example["episode_id"] for example in examples],
            "step_id": [example["step_id"] for example in examples],
            "image_path": [example["image_path"] for example in examples],
            "prev_actions": torch.tensor(
                [example.get("prev_action", 8) for example in examples],
                dtype=torch.long,
            ),
        }
        if self.include_segment_token_ids:
            self._append_segment_token_ids(result, prompts, instructions)
        return result


@dataclass(slots=True)
class Qwen35NativeTaskGRUEpisodeCollator:
    """Stack native-Qwen batches into chronological TaskGRU [B,T,...] input."""

    collator: Qwen35NativeClassificationCollator
    sequence_length: int

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        per_step = [
            self.collator([example["samples"][step] for example in examples])
            for step in range(self.sequence_length)
        ]
        sequence_keys = {
            "input_ids", "attention_mask", "mm_token_type_ids",
            "instruction_token_mask", "segment_token_ids",
        }
        pad_values = {
            "input_ids": int(self.collator.processor.tokenizer.pad_token_id),
            "attention_mask": 0,
            "mm_token_type_ids": 0,
            "instruction_token_mask": False,
            "segment_token_ids": -1,
        }
        batch: dict[str, Any] = {}
        tensor_keys = [key for key, value in per_step[0].items() if isinstance(value, torch.Tensor)]
        for key in tensor_keys:
            values = [step[key] for step in per_step]
            if key in sequence_keys:
                max_length = max(value.shape[-1] for value in values)
                values = [
                    torch.nn.functional.pad(
                        value,
                        (0, max_length - value.shape[-1]),
                        value=pad_values[key],
                    )
                    for value in values
                ]
            elif len({tuple(value.shape) for value in values}) != 1:
                raise RuntimeError(f"Qwen3.5 TaskGRU cannot stack varying tensor {key}")
            batch[key] = torch.stack(values, dim=1)
        batch["sequence_mask"] = torch.tensor(
            [example["sequence_mask"] for example in examples], dtype=torch.bool
        )
        batch["loss_mask"] = torch.tensor(
            [example["loss_mask"] for example in examples], dtype=torch.bool
        )
        batch["stream_reset"] = torch.tensor(
            [example["stream_reset"] for example in examples], dtype=torch.bool
        )
        batch["stream_end"] = torch.tensor(
            [example["stream_end"] for example in examples], dtype=torch.bool
        )
        batch["episode_id"] = [example["episode_id"] for example in examples]
        return batch


@dataclass(slots=True)
class Qwen35Native1TStage1Collator:
    """Build native current-image chat input plus nine Stage-1 RGB frames."""

    processor: Any
    prompt_template: str
    max_text_tokens: int = 256
    image_size: int = 224
    use_relative_state_text: bool = False

    def __post_init__(self) -> None:
        native_pixels = self.image_size * self.image_size
        self.processor.image_processor.size["shortest_edge"] = native_pixels
        self.processor.image_processor.size["longest_edge"] = native_pixels

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        conversations = []
        for example in examples:
            instruction = example["instruction"].strip()
            if self.use_relative_state_text:
                from aerial_openvla.data.multiframe_dataset import (
                    prepend_relative_state_text,
                )

                instruction = prepend_relative_state_text(
                    instruction,
                    int(example["relative_heading_degrees"]),
                    int(example["relative_altitude_meters"]),
                )
            prompt = self.prompt_template.format(
                instruction=instruction
            )
            conversations.append([{"role": "user", "content": [
                {"type": "image", "image": example["memory_images"][-1]},
                {"type": "text", "text": prompt},
            ]}])
        current = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "padding": True,
                "truncation": True,
                "max_length": self.max_text_tokens,
            },
        )
        required_current = (
            "input_ids", "attention_mask", "image_grid_thw",
            "mm_token_type_ids",
        )
        missing_current = [
            key for key in required_current if key not in current
        ]
        if missing_current:
            raise RuntimeError(
                f"Qwen3.5 Stage 1 processor omitted fields: {missing_current}"
            )
        flat_images = [
            image for example in examples for image in example["memory_images"]
        ]
        images = self.processor.image_processor(flat_images, return_tensors="pt")
        if "pixel_values" not in images or "image_grid_thw" not in images:
            raise RuntimeError(
                "Qwen3.5 image processor omitted pixel_values or image_grid_thw"
            )
        expected_images = len(examples) * 9
        if images["image_grid_thw"].shape[0] != expected_images:
            raise RuntimeError("Qwen3.5-Native1T processor did not emit B*9 grids")
        return {
            **{key: current[key] for key in required_current},
            "native1t_memory_pixel_values": images["pixel_values"],
            "native1t_memory_grid_thw": images["image_grid_thw"].reshape(
                len(examples), 9, 3
            ),
            "native1t_memory_valid_mask": torch.stack([
                example["memory_valid_mask"] for example in examples
            ]),
            "action_id": torch.tensor(
                [example["action_id"] for example in examples], dtype=torch.long
            ),
            "episode_id": [example["episode_id"] for example in examples],
            "step_id": [example["step_id"] for example in examples],
        }


@dataclass(slots=True)
class Qwen35Native1TStage2Collator:
    """Current RGB plus cached frozen-tokenizer H16 for Stage 2.

    Current RGB uses the exact native Qwen image chat template.  The model
    inserts MEM_BOS, valid history tokens and learned ACT after ``im_end``.
    """

    processor: Any
    prompt_template: str
    max_text_tokens: int = 256
    image_size: int = 224
    use_relative_state_text: bool = False
    _dummy_image: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        native_pixels = self.image_size * self.image_size
        self.processor.image_processor.size["shortest_edge"] = native_pixels
        self.processor.image_processor.size["longest_edge"] = native_pixels
        self._dummy_image = Image.new("RGB", (self.image_size, self.image_size))

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        cached_flags = ["current_premerger_features" in e for e in examples]
        if any(cached_flags) and not all(cached_flags):
            raise RuntimeError("A batch cannot mix RGB and pre-merger current inputs")
        use_premerger_cache = all(cached_flags)
        conversations = []
        for example in examples:
            instruction = example["instruction"].strip()
            if self.use_relative_state_text:
                from aerial_openvla.data.multiframe_dataset import (
                    prepend_relative_state_text,
                )

                instruction = prepend_relative_state_text(
                    instruction,
                    int(example["relative_heading_degrees"]),
                    int(example["relative_altitude_meters"]),
                )
            prompt = self.prompt_template.format(
                instruction=instruction
            )
            conversations.append([{"role": "user", "content": [
                {"type": "image", "image": (
                    self._dummy_image if use_premerger_cache else example["image"]
                )},
                {"type": "text", "text": prompt},
            ]}])
        encoded = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "padding": True,
                "truncation": True,
                "max_length": self.max_text_tokens,
            },
        )
        required = (
            "input_ids", "attention_mask", "pixel_values",
            "image_grid_thw", "mm_token_type_ids",
        )
        missing = [key for key in required if key not in encoded]
        if missing:
            raise RuntimeError(
                f"Qwen3.5 Stage 2 processor omitted fields: {missing}"
            )
        if tuple(encoded["image_grid_thw"].shape) != (len(examples), 3):
            raise RuntimeError("Qwen3.5-Native1T current grids must have shape [B,3]")
        result = {
            **{
                key: encoded[key] for key in required
                if key != "pixel_values" or not use_premerger_cache
            },
            "native1t_memory_features": torch.stack([
                example["memory_features"] for example in examples
            ]),
            "native1t_memory_valid_mask": torch.stack([
                example["memory_valid_mask"] for example in examples
            ]),
            "action_id": torch.tensor(
                [example["action_id"] for example in examples], dtype=torch.long
            ),
            "episode_id": [example["episode_id"] for example in examples],
            "step_id": [example["step_id"] for example in examples],
        }
        if use_premerger_cache:
            result["native1t_current_premerger_features"] = torch.stack([
                example["current_premerger_features"] for example in examples
            ])
            result["current_visual_cache_row"] = [
                example["current_visual_cache_row"] for example in examples
            ]
        return result
