"""Protocol (duck-typing contract) for action classifier models.

This module does NOT define a strict ABC — it follows the project"s
existing duck-typing convention.  Any model that satisfies this
implicit interface can be swapped into train.py and rollout.py.

To add a new model:
1. Implement a class matching the signatures below.
2. Add a branch to `models.factory.create_model`.
3. (Optionally) implement a corresponding Policy class for evaluation.
"""

from __future__ import annotations

from typing import Any



# ---------------------------------------------------------------------------
# The implicit interface (documentation only, no runtime enforcement)
# ---------------------------------------------------------------------------
#
# class ActionClassifier(nn.Module):
#     """Any model that takes a collator batch and returns action logits."""
#
#     def forward(self, **batch_kwargs: Any) -> torch.Tensor:
#         """Return [B, num_actions] logits tensor.
#
#         The exact keyword arguments depend on the collator used:
#         - image mode: input_ids, attention_mask, pixel_values
#         - multiframe_image mode: input_ids, attention_mask, pixel_values, prev_actions
#         - feature mode: instruction_ids, instruction_mask, features, prev_actions, masks
#
#         For compatibility, use **batch_kwargs and extract what you need.
#         """
#         ...
#
#     def trainable_parameter_groups(self, **lr_kwargs: float) -> list[dict[str, Any]]:
#         """Return parameter groups for torch.optim.AdamW.
#
#         Each group is a dict with keys: "params", "lr", "weight_decay".
#         Called as: model.trainable_parameter_groups(head_lr=..., **extra)
#         where extra carries model-type-specific learning rates.
#         """
#         ...
#
#     def save_checkpoint(
#         self,
#         output_dir: str | Path,
#         processor: Any,
#         *,
#         optimizer: Any | None = None,
#         scheduler: Any | None = None,
#         extra: dict[str, Any] | None = None,
#     ) -> None:
#         """Save a complete, reloadable checkpoint to output_dir."""
#         ...
#
#     def load_checkpoint(
#         self,
#         checkpoint_dir: str | Path,
#         *,
#         optimizer: Any | None = None,
#         scheduler: Any | None = None,
#     ) -> dict[str, Any] | None:
#         """Load checkpoint, return training_state dict or None."""
#         ...


# ---------------------------------------------------------------------------
# Shared helper: extract extra learning-rate kwargs from TrainConfig
# ---------------------------------------------------------------------------

def collect_lr_kwargs(cfg: Any, model_type: str) -> dict[str, float]:
    """Extract model-specific LR kwargs from a TrainConfig-like object.

    Returns a dict suitable for passing to
    ``model.trainable_parameter_groups(head_lr=..., **extra)``.
    """
    extra: dict[str, float] = {}
    if model_type == "openvla":
        extra["lora_lr"] = getattr(cfg, "learning_rate_lora", 5e-5)
        extra["projector_lr"] = getattr(cfg, "learning_rate_projector", 1e-5)
    elif model_type in {"qwen35_native", "qwen35_nativemem"}:
        extra["lora_lr"] = getattr(cfg, "learning_rate_lora", 5e-5)
        extra["projector_lr"] = getattr(cfg, "learning_rate_projector", 1e-5)
    elif model_type == "cma":
        # CMA uses a single LR for all parameters
        extra["cma_lr"] = getattr(cfg, "learning_rate_lora", 0.00025)
    elif model_type == "airminivla":
        extra["lora_lr"] = getattr(cfg, "learning_rate_lora", 5e-5)
        extra["projector_lr"] = getattr(cfg, "learning_rate_projector", 1e-5)
        extra["vision_lr"] = getattr(cfg, "learning_rate_vision", 1e-6)
        extra["memory_tokenizer_lr"] = getattr(
            cfg, "learning_rate_memory_tokenizer", 1e-4
        )
        extra["memory_projector_lr"] = getattr(
            cfg, "learning_rate_memory_projector", 5e-5
        )
        extra["action_history_lr"] = getattr(
            cfg, "learning_rate_action_history", 2e-4
        )
    return extra
