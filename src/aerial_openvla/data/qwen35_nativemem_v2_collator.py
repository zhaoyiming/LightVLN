"""Native Qwen processor collator for cached NativeMEM v2 training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image

from aerial_openvla.data.qwen35_relative_state import (
    QWEN35_RELATIVE_NONE,
    append_qwen35_relative_state_text,
    validate_qwen35_relative_state_mode,
)


@dataclass(slots=True)
class Qwen35NativeMEMV2Collator:
    processor: Any
    prompt_template: str
    stage: int
    max_text_tokens: int = 256
    single_stage: bool = False
    relative_state_mode: str = QWEN35_RELATIVE_NONE
    image_size: int = 224
    _dummy_image: Image.Image = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.stage not in (1, 2):
            raise ValueError("NativeMEM v2 collator stage must be 1 or 2")
        if self.single_stage and self.stage != 2:
            raise ValueError("NativeMEM single-stage V2 collator requires strict causal H16")
        self.relative_state_mode = validate_qwen35_relative_state_mode(
            self.relative_state_mode
        )
        native_pixels = self.image_size * self.image_size
        self.processor.image_processor.size["shortest_edge"] = native_pixels
        self.processor.image_processor.size["longest_edge"] = native_pixels
        self._dummy_image = Image.new("RGB", (self.image_size, self.image_size))

    def _encode(self, conversations: list[list[dict[str, Any]]]) -> Any:
        """Encode without truncating image/special tokens.

        ``max_text_tokens`` is a text-only budget.  The 49 image placeholders
        and native chat suffix are deliberately outside that budget.
        """
        return self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "padding": True,
                "truncation": False,
            },
        )

    def _truncate_text_to_fit(self, prompt: str) -> str:
        """Truncate only textual prompt tokens to the configured budget."""
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) <= self.max_text_tokens:
            return prompt
        return tokenizer.decode(
            token_ids[: self.max_text_tokens],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        conversations = []
        for example in examples:
            instruction = example["instruction"].strip()
            state_suffix = None
            if self.relative_state_mode != QWEN35_RELATIVE_NONE:
                required = (
                    "relative_heading_degrees",
                    "relative_altitude_meters",
                    "relative_forward_meters",
                    "relative_left_meters",
                )
                missing = [key for key in required if key not in example]
                if missing:
                    raise RuntimeError(
                        "Qwen3.5 append-planar Relative example is missing "
                        f"{missing}"
                    )
                state_suffix = append_qwen35_relative_state_text(
                    "",
                    example["relative_heading_degrees"],
                    example["relative_altitude_meters"],
                    example["relative_forward_meters"],
                    example["relative_left_meters"],
                )
            prompt = self.prompt_template.format(instruction=instruction)
            # The 256-token budget belongs only to the original Qwen prompt
            # and Instruction.  Relative State is appended afterwards and
            # deliberately does not consume that Instruction budget.
            prompt = self._truncate_text_to_fit(prompt)
            if state_suffix is not None:
                prompt = f"{prompt}{state_suffix}"
            if (
                self.relative_state_mode != QWEN35_RELATIVE_NONE
                and "Current state:" not in prompt
            ):
                raise RuntimeError(
                    "Qwen3.5 append-planar Relative State was not appended "
                    "after the Instruction text budget was applied"
                )
            conversations.append([
                {"role": "user", "content": [
                    {"type": "image", "image": self._dummy_image},
                    {"type": "text", "text": prompt},
                ]}
            ])
        encoded = self._encode(conversations)
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        for row in range(len(examples)):
            valid = encoded["attention_mask"][row].bool()
            ids = encoded["input_ids"][row][valid]
            types = encoded["mm_token_type_ids"][row][valid]
            positions = (
                torch.where(ids == int(im_end_id))[0]
                if im_end_id is not None else torch.empty(0, dtype=torch.long)
            )
            suffix_ok = (
                positions.numel() > 0
                and int(positions[-1]) == ids.numel() - 2
                and int(types[-1]) == 0
            )
            if not suffix_ok:
                raise RuntimeError(
                    "NativeMEM processor did not preserve the complete native "
                    "<|im_end|> + action-slot suffix"
                )
        required = ("input_ids", "attention_mask", "image_grid_thw", "mm_token_type_ids")
        missing = [key for key in required if key not in encoded]
        if missing:
            raise RuntimeError(f"NativeMEM v2 processor omitted {missing}")
        if tuple(encoded["image_grid_thw"].shape) != (len(examples), 3):
            raise RuntimeError("NativeMEM v2 requires one current grid per sample")
        grids = encoded["image_grid_thw"]
        if not torch.all(grids[:, 0] == 1) or not torch.all(grids[:, 1] * grids[:, 2] == 196):
            raise RuntimeError(f"NativeMEM v2 requires a 224px 14x14 grid, got {grids.tolist()}")
        image_counts = (encoded["mm_token_type_ids"] == 1).sum(dim=1)
        if not torch.all(image_counts == 49):
            raise RuntimeError(f"NativeMEM v2 processor emitted non-49 image placeholders: {image_counts.tolist()}")
        expected_mask = 9 if self.stage == 1 else 16
        for example in examples:
            if tuple(example["nativemem_history_valid_mask"].shape) != (expected_mask,):
                raise RuntimeError(f"NativeMEM v2 Stage {self.stage} expects mask length {expected_mask}")
        batch = {
            **{key: encoded[key] for key in required},
            "nativemem_premerger_features": torch.stack([
                example["nativemem_premerger_features"] for example in examples
            ]),
            "nativemem_history_valid_mask": torch.stack([
                example["nativemem_history_valid_mask"] for example in examples
            ]),
            "action_id": torch.tensor([
                example["action_id"] for example in examples
            ], dtype=torch.long),
            "episode_id": [example["episode_id"] for example in examples],
            "trajectory_id": [example["trajectory_id"] for example in examples],
            "step_id": [example["step_id"] for example in examples],
            "current_visual_cache_row": [
                example["current_visual_cache_row"] for example in examples
            ],
        }
        if self.stage == 2 and not self.single_stage:
            if any("nativemem_history_latents" not in example for example in examples):
                raise RuntimeError(
                    "NativeMEM Stage2 examples are missing checkpoint-matched "
                    "1T history latents"
                )
            latent_shapes = {
                tuple(example["nativemem_history_latents"].shape)
                for example in examples
            }
            if latent_shapes != {(16, 768)}:
                raise RuntimeError(
                    f"NativeMEM Stage2 history latent shapes are {latent_shapes}, "
                    "expected {(16, 768)}"
                )
            batch["nativemem_history_latents"] = torch.stack([
                example["nativemem_history_latents"] for example in examples
            ])
        return batch
