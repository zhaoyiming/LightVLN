from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class OpenVLAClassificationCollator:
    processor: Any
    prompt_template: str
    max_text_tokens: int = 256

    def __call__(self, examples: list[dict]) -> dict[str, Any]:
        prompts = [
            self.prompt_template.format(instruction=example["instruction"].strip()) for example in examples
        ]
        images = [example["image"] for example in examples]
        encoded = self.processor(
            text=prompts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_text_tokens,
            return_tensors="pt",
        )
        pixel_values = encoded["pixel_values"]
        if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 4:
            raise ValueError(
                f"Expected batched pixel_values [B,C,H,W], got {type(pixel_values)!r} "
                f"with shape {getattr(pixel_values, 'shape', None)}"
            )
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            warnings.warn(
                "Processor did not return an attention_mask. "
                "Falling back to all-ones mask, which may degrade classification accuracy "
                "by pooling from padding tokens instead of the last real token.",
                stacklevel=2,
            )
            attention_mask = torch.ones_like(encoded["input_ids"], dtype=torch.long)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "action_id": torch.tensor([example["action_id"] for example in examples], dtype=torch.long),
            "episode_id": [example["episode_id"] for example in examples],
            "step_id": [example["step_id"] for example in examples],
            "image_path": [example["image_path"] for example in examples],
        }
