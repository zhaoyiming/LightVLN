from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


METRIC_KEYS = (
    "distance_to_goal",
    "success",
    "ndtw",
    "sdtw",
    "path_length",
    "oracle_success",
    "steps_taken",
)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_completed_episode_ids(path: str | Path, include_errors: bool = False) -> set[str]:
    """Episode IDs already recorded in a results file.

    Error stubs are skipped by default so a failed episode is retried on the
    next resumed run instead of being permanently written off.
    """
    source = Path(path)
    if not source.is_file():
        return set()
    completed: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not include_errors and "error" in row.get("final_info", {}):
                continue
            completed.add(str(row["episode_id"]))
    return completed


def aggregate_results(path: str | Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    # Sentinel values in error stubs (distance_to_goal=-1.0, zeroed metrics)
    # must not enter the means; report them as a separate count.
    errors = sum(1 for row in rows if "error" in row.get("final_info", {}))
    rows = [row for row in rows if "error" not in row.get("final_info", {})]
    sums: dict[str, float] = defaultdict(float)
    for row in rows:
        final = row.get("final_info", {})
        for key in METRIC_KEYS:
            if key in final:
                sums[key] += float(final[key])
    count = len(rows)
    means = {key: value / max(count, 1) for key, value in sums.items()}
    raw_switches = 0
    sticky_switches = 0
    segment_transitions = 0
    for row in rows:
        states = [
            action.get("task_state")
            for action in row.get("actions", [])
            if action.get("task_state") is not None
        ]
        for previous, current in zip(states, states[1:]):
            if (
                "raw_argmax_segment" not in previous
                or "raw_argmax_segment" not in current
                or "sticky_argmax_segment" not in previous
                or "sticky_argmax_segment" not in current
            ):
                continue
            segment_transitions += 1
            raw_switches += int(
                previous["raw_argmax_segment"] != current["raw_argmax_segment"]
            )
            sticky_switches += int(
                previous["sticky_argmax_segment"]
                != current["sticky_argmax_segment"]
            )
    result = {"episodes": count, "errors": errors, "mean": means}
    action_history_counts = {
        "left_to_right": 0,
        "right_to_left": 0,
        "up_to_down": 0,
        "down_to_up": 0,
        "lrlr": 0,
        "rlrl": 0,
        "udud": 0,
        "dudu": 0,
        "timeout_500": 0,
    }
    for row in rows:
        names = [action.get("action_name") for action in row.get("actions", [])]
        for previous, current in zip(names, names[1:]):
            action_history_counts["left_to_right"] += int(
                previous == "TURN_LEFT" and current == "TURN_RIGHT"
            )
            action_history_counts["right_to_left"] += int(
                previous == "TURN_RIGHT" and current == "TURN_LEFT"
            )
            action_history_counts["up_to_down"] += int(
                previous == "GO_UP" and current == "GO_DOWN"
            )
            action_history_counts["down_to_up"] += int(
                previous == "GO_DOWN" and current == "GO_UP"
            )
        patterns = {
            "lrlr": ["TURN_LEFT", "TURN_RIGHT", "TURN_LEFT", "TURN_RIGHT"],
            "rlrl": ["TURN_RIGHT", "TURN_LEFT", "TURN_RIGHT", "TURN_LEFT"],
            "udud": ["GO_UP", "GO_DOWN", "GO_UP", "GO_DOWN"],
            "dudu": ["GO_DOWN", "GO_UP", "GO_DOWN", "GO_UP"],
        }
        for key, pattern in patterns.items():
            action_history_counts[key] += sum(
                names[index:index + 4] == pattern
                for index in range(max(0, len(names) - 3))
            )
        action_history_counts["timeout_500"] += int(len(names) >= 500)
    result["action_history_diagnostics"] = action_history_counts
    if segment_transitions:
        result["segment_switch"] = {
            "transitions": segment_transitions,
            "raw_segment_switch_rate": raw_switches / segment_transitions,
            "sticky_segment_switch_rate": sticky_switches / segment_transitions,
        }
    return result
