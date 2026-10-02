"""OpenFly adapters for the isolated legacy AirMiniVLA snapshot.

Keep dataset modules lazy: rollout evaluation only needs the action codec and
policy adapter, while dataset modules deliberately depend on the heavier
Parquet/TFDS stack used on the training host.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .action_codec import OpenFlyActionCodec

if TYPE_CHECKING:
    from .parquet_dataset import OpenFlyParquetClassificationDataset
    from .policy_adapter import OpenFlyMiniVLAPolicy
    from .tfds_current_frame import OpenFlyCurrentFrameTFDSDataset


__all__ = [
    "OpenFlyActionCodec",
    "OpenFlyCurrentFrameTFDSDataset",
    "OpenFlyMiniVLAPolicy",
    "OpenFlyParquetClassificationDataset",
]


def __getattr__(name: str) -> Any:
    """Load data or policy adapters only when a caller actually requests one."""
    if name == "OpenFlyMiniVLAPolicy":
        from .policy_adapter import OpenFlyMiniVLAPolicy

        return OpenFlyMiniVLAPolicy
    if name == "OpenFlyParquetClassificationDataset":
        from .parquet_dataset import OpenFlyParquetClassificationDataset

        return OpenFlyParquetClassificationDataset
    if name == "OpenFlyCurrentFrameTFDSDataset":
        from .tfds_current_frame import OpenFlyCurrentFrameTFDSDataset

        return OpenFlyCurrentFrameTFDSDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
