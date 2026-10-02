"""Use a legacy AirMiniVLA checkpoint from OpenFly's camera/evaluator code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from aerial_openvla.evaluation.policy import AirMiniVLAClassifierPolicy
from aerial_openvla.openfly.action_codec import (
    OPENFLY_OBSERVED8_ACTION_ID_TO_CLASS,
    OPENFLY_OBSERVED8_ACTION_IDS,
    OPENFLY_OBSERVED8_ACTION_POLICY,
    OPENFLY_OBSERVED8_CLASS_TO_ACTION_ID,
)


DEFAULT_PROMPT = (
    "In: What primitive action should the aerial robot take to follow the instruction: "
    "{instruction}?\nOut:"
)


class OpenFlyMiniVLAPolicy:
    """Small adapter around the copied MiniVLA rollout policy.

    ``act`` consumes OpenFly RGB or BGR camera arrays (or a PIL image), returns
    the checkpoint's action ID plus diagnostic codebook probabilities, and
    keeps the exact MiniVLA recurrent/history state.  A checkpoint with
    ``action_head_type='categorical'`` plus ``action_policy='openfly_10'`` is
    a direct ten-way MiniVLA policy: its argmax is already the executable
    OpenFly ID 0..9.  ``action_policy='openfly_observed_8'`` has eight compact
    classes for original IDs ``0,1,2,3,4,5,8,9`` and restores the original ID
    before returning it to the evaluator.  Historical parallel 8x256
    action-token checkpoints and unproven legacy categorical checkpoints
    remain separately identifiable.
    """

    def __init__(
        self,
        *,
        checkpoint_dir: str | Path,
        model_name_or_path: str | Path,
        device: str | torch.device | None = None,
        prompt_template: str = DEFAULT_PROMPT,
        max_text_tokens: int = 256,
        use_flash_attention_2: bool = True,
        bf16: bool = True,
        **policy_overrides: Any,
    ) -> None:
        resolved_device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if resolved_device.type != "cuda" and use_flash_attention_2:
            raise ValueError("FlashAttention2 requires CUDA; pass use_flash_attention_2=False on CPU")
        if resolved_device.type != "cuda" and bf16:
            raise ValueError("bf16 rollout is configured for CUDA; pass bf16=False on CPU")
        self._policy = AirMiniVLAClassifierPolicy(
            model_name_or_path=str(Path(model_name_or_path).expanduser().resolve()),
            checkpoint_dir=Path(checkpoint_dir).expanduser().resolve(),
            prompt_template=prompt_template,
            max_text_tokens=int(max_text_tokens),
            device=resolved_device,
            use_flash_attention_2=bool(use_flash_attention_2),
            bf16=bool(bf16),
            **policy_overrides,
        )
        self.head_num_actions = int(self._policy.model.config.num_actions)
        self._observed8 = (
            self._policy.openfly_action_head_type == "categorical"
            and self._policy.openfly_action_policy == OPENFLY_OBSERVED8_ACTION_POLICY
        )
        if self._observed8:
            if self.head_num_actions != len(OPENFLY_OBSERVED8_ACTION_IDS):
                raise ValueError(
                    "Observed OpenFly policy must expose an 8-logit head, got "
                    f"{self.head_num_actions}"
                )
            self.num_actions = self.head_num_actions
            self.executable_action_ids = OPENFLY_OBSERVED8_ACTION_IDS
        elif self.head_num_actions in {8, 10}:
            self.num_actions = self.head_num_actions
            self.executable_action_ids = tuple(range(self.num_actions))
        else:
            raise ValueError(
                "OpenFlyMiniVLAPolicy supports only 8-way legacy or 10-way "
                f"OpenFly checkpoints, got num_actions={self.head_num_actions}"
            )
        self._last_openfly_action: dict[str, object] | None = None

    def reset(self) -> None:
        """Reset MiniVLA temporal state at an OpenFly episode boundary."""
        self._policy.reset()
        self._last_openfly_action = None

    def record_executed_action(self, action_id: int) -> None:
        """Commit an executed action for checkpoints that use action history."""
        executed_action_id = int(action_id)
        if executed_action_id not in self.executable_action_ids:
            raise ValueError(
                "Checkpoint cannot execute OpenFly action "
                f"{executed_action_id}; expected one of {self.executable_action_ids}"
            )
        head_action_id = (
            OPENFLY_OBSERVED8_ACTION_ID_TO_CLASS[executed_action_id]
            if self._observed8
            else executed_action_id
        )
        self._policy.record_executed_action(head_action_id)

    def act(
        self,
        image: Image.Image | np.ndarray,
        instruction: str,
        *,
        bgr: bool = True,
    ) -> tuple[int, list[float]]:
        """Predict one legacy action from an OpenFly observation.

        OpenFly's AirSim camera arrays are normally BGR, hence ``bgr=True`` by
        default.  Set it to ``False`` for RGB NumPy arrays.
        """
        pil_image = self._as_pil(image, bgr=bgr)
        head_action_id, probabilities = self._policy.act(pil_image, str(instruction))
        if not 0 <= head_action_id < self.head_num_actions:
            raise RuntimeError(
                f"MiniVLA returned invalid head action {head_action_id}; expected "
                f"0..{self.head_num_actions - 1}"
            )
        if self._observed8:
            action_id = OPENFLY_OBSERVED8_CLASS_TO_ACTION_ID[head_action_id]
            self._last_openfly_action = {
                "action_policy": OPENFLY_OBSERVED8_ACTION_POLICY,
                "head_class_id": int(head_action_id),
                "openfly_action_id": int(action_id),
                "head_class_to_openfly_action_id": {
                    str(class_id): original_action_id
                    for class_id, original_action_id in OPENFLY_OBSERVED8_CLASS_TO_ACTION_ID.items()
                },
                "probability_semantics": "softmax_over_observed_openfly_actions",
            }
            return action_id, probabilities
        self._last_openfly_action = self._policy.last_openfly_action
        return head_action_id, probabilities

    @property
    def action_head_type(self) -> str:
        """The reconstructed checkpoint's action-head contract."""
        return str(self._policy.openfly_action_head_type)

    @property
    def action_policy(self) -> str | None:
        """The checkpoint-pinned action-ID semantics, if declared."""
        return self._policy.openfly_action_policy

    @property
    def last_openfly_action(self) -> dict[str, object] | None:
        """Decoded vector/match diagnostics for an action-token checkpoint."""
        return self._last_openfly_action

    @staticmethod
    def _as_pil(image: Image.Image | np.ndarray, *, bgr: bool) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] not in {3, 4}:
            raise ValueError(
                f"Expected HxWx3/HxWx4 image array, got shape {tuple(array.shape)}"
            )
        if array.dtype != np.uint8:
            upper = float(np.nanmax(array)) if array.size else 0.0
            if upper <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.shape[-1] == 4:
            array = array[..., :3]
        if bgr:
            array = array[..., ::-1]
        return Image.fromarray(np.ascontiguousarray(array), mode="RGB")
