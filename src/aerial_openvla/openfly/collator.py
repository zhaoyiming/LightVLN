"""OpenFly action-token targets layered on the shared MiniVLA collators."""

from __future__ import annotations

from typing import Any

from aerial_openvla.data.multiframe_dataset import MultiFrameCollator
from aerial_openvla.data.native1t_postfusion_dataset import Native1TCollator
from aerial_openvla.openfly.action_codec import OpenFlyActionCodec


class OpenFlyActionTokenMultiFrameCollator(MultiFrameCollator):
    """Attach one OpenFly [8] codec sequence to each direct RGB row.

    ``action_tokens_v1`` consumes these as parallel class labels; the v2
    causal-LM head converts the same classes to OpenFly's reserved tail-token
    IDs and appends EOS internally for teacher forcing.
    """

    def __init__(self, *args: Any, action_codec: OpenFlyActionCodec, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.action_codec = action_codec

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = super().__call__(examples)
        batch["openfly_action_token_targets"] = self.action_codec.targets_tensor(
            batch["action_id"]
        )
        return batch


class OpenFlyActionTokenNative1TCollator(Native1TCollator):
    """Attach OpenFly token targets while preserving the shared H8 batch ABI."""

    def __init__(self, *args: Any, action_codec: OpenFlyActionCodec, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.action_codec = action_codec

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = super().__call__(examples)
        batch["openfly_action_token_targets"] = self.action_codec.targets_tensor(
            batch["action_id"]
        )
        return batch
