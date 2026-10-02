"""Shared Qwen3.5 NativeMEM append-after-instruction Relative State contract."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from aerial_openvla.data.multiframe_dataset import update_relative_planar_state


QWEN35_RELATIVE_NONE = "none"
QWEN35_RELATIVE_APPEND_PLANAR_V1 = "append_planar_v1"
QWEN35_RELATIVE_STATE_CACHE_VERSION = 1
QWEN35_RELATIVE_STATE_COLUMNS = (
    "heading_degrees",
    "altitude_meters",
    "forward_meters",
    "left_meters",
)


def validate_qwen35_relative_state_mode(mode: str) -> str:
    if mode not in {QWEN35_RELATIVE_NONE, QWEN35_RELATIVE_APPEND_PLANAR_V1}:
        raise ValueError(
            "qwen35_relative_state_mode must be one of "
            f"{QWEN35_RELATIVE_NONE!r}, {QWEN35_RELATIVE_APPEND_PLANAR_V1!r}; "
            f"got {mode!r}"
        )
    return mode


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _round_half_away_from_zero(value: float) -> int:
    return int(math.copysign(math.floor(abs(float(value)) + 0.5), value))


def append_qwen35_relative_state_text(
    instruction: str,
    heading_degrees: float,
    altitude_meters: float,
    forward_meters: float,
    left_meters: float,
) -> str:
    """Append the user-approved, signed planar state block after Instruction."""
    heading = _round_half_away_from_zero(heading_degrees)
    altitude = _round_half_away_from_zero(altitude_meters)
    forward = _round_half_away_from_zero(forward_meters)
    left = _round_half_away_from_zero(left_meters)
    return (
        f"{instruction.strip()}\n"
        "Current state:\n"
        f"Heading from start: {heading:+d} deg.\n"
        f"Altitude from start: {altitude:+d} m.\n"
        f"Forward displacement from start: {forward:+d} m.\n"
        f"Left displacement from start: {left:+d} m."
    )


def update_qwen35_relative_state(
    heading_degrees: float,
    altitude_meters: float,
    forward_meters: float,
    left_meters: float,
    action_id: int,
) -> tuple[int, int, float, float]:
    """Use the shared AirVLN discrete-motion rule for train and rollout."""
    return update_relative_planar_state(
        int(round(heading_degrees)),
        int(round(altitude_meters)),
        float(forward_meters),
        float(left_meters),
        int(action_id),
    )


def load_qwen35_relative_state_cache(
    cache_dir: str | Path,
    manifest_path: str | Path,
    expected_rows: int,
) -> tuple[np.ndarray, dict]:
    """Open an immutable, manifest-bound per-step Relative State cache."""
    root = Path(cache_dir).resolve()
    manifest = Path(manifest_path).resolve()
    metadata_path = root / "metadata.json"
    states_path = root / "states.npy"
    if not metadata_path.is_file() or not states_path.is_file():
        raise FileNotFoundError(
            "Qwen3.5 Relative State cache requires metadata.json and states.npy: "
            f"{root}"
        )
    metadata = json.loads(metadata_path.read_text())
    required = {
        "complete": True,
        "version": QWEN35_RELATIVE_STATE_CACHE_VERSION,
        "mode": QWEN35_RELATIVE_APPEND_PLANAR_V1,
        "manifest_sha256": sha256_file(manifest),
        "rows": int(expected_rows),
        "columns": list(QWEN35_RELATIVE_STATE_COLUMNS),
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Qwen3.5 Relative State cache metadata mismatch for {key}: "
                f"expected {expected!r}, got {metadata.get(key)!r}"
            )
    states = np.load(states_path, mmap_mode="r")
    if states.dtype != np.float32 or tuple(states.shape) != (expected_rows, 4):
        raise ValueError(
            "Qwen3.5 Relative State cache states.npy must be "
            f"float32 [{expected_rows},4], got {states.dtype} {states.shape}"
        )
    return states, metadata
