from __future__ import annotations

from enum import IntEnum


class AirVLNAction(IntEnum):
    """Official AirVLN primitive action IDs."""

    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    GO_UP = 4
    GO_DOWN = 5
    MOVE_LEFT = 6
    MOVE_RIGHT = 7


NUM_ACTIONS = len(AirVLNAction)

# Convenience constants so callers can use ``STOP`` / ``MOVE_FORWARD``
# instead of ``AirVLNAction.STOP``.
STOP = AirVLNAction.STOP.value
MOVE_FORWARD = AirVLNAction.MOVE_FORWARD.value
TURN_LEFT = AirVLNAction.TURN_LEFT.value
TURN_RIGHT = AirVLNAction.TURN_RIGHT.value
GO_UP = AirVLNAction.GO_UP.value
GO_DOWN = AirVLNAction.GO_DOWN.value
MOVE_LEFT = AirVLNAction.MOVE_LEFT.value
MOVE_RIGHT = AirVLNAction.MOVE_RIGHT.value

ACTION_NAMES: dict[int, str] = {action.value: action.name for action in AirVLNAction}


def validate_action_id(action_id: int) -> int:
    try:
        return AirVLNAction(int(action_id)).value
    except ValueError as exc:
        raise ValueError(f"Invalid AirVLN action ID {action_id}; expected 0..{NUM_ACTIONS - 1}") from exc


def assert_matches_airvln(airsim_actions: object) -> None:
    """Fail fast if a checked-out AirVLN repo changes the canonical mapping."""

    expected = {
        "STOP": 0,
        "MOVE_FORWARD": 1,
        "TURN_LEFT": 2,
        "TURN_RIGHT": 3,
        "GO_UP": 4,
        "GO_DOWN": 5,
        "MOVE_LEFT": 6,
        "MOVE_RIGHT": 7,
    }
    observed = {name: int(getattr(airsim_actions, name)) for name in expected}
    if observed != expected:
        raise RuntimeError(f"AirVLN action mapping mismatch: expected {expected}, observed {observed}")
