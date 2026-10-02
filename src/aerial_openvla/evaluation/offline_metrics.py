"""Offline metric computations for fast validation.

All functions operate on tensors or plain Python types — no model or dataloader
dependency.  This makes them easy to test and compose independently.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aerial_openvla.core.actions import ACTION_NAMES, NUM_ACTIONS


def per_action_metrics(
    confusion: NDArray[np.float64],
) -> dict[str, Any]:
    """Precision, Recall, F1 per action class from a confusion matrix.

    Parameters
    ----------
    confusion : (NUM_ACTIONS, NUM_ACTIONS) float array
        Rows = true label, columns = prediction.

    Returns
    -------
    dict with per-class metrics, macro-F1, and weighted-F1.
    """
    epsilon = 1e-12
    per_class: dict[str, float] = {}
    f1s: list[float] = []
    weights: list[float] = []
    support = confusion.sum(axis=1)  # true count per class

    for action_id in range(NUM_ACTIONS):
        tp = confusion[action_id, action_id]
        fp = confusion[:, action_id].sum() - tp
        fn = support[action_id] - tp
        precision = tp / max(tp + fp, epsilon)
        recall = tp / max(tp + fn, epsilon)
        f1 = 2 * precision * recall / max(precision + recall, epsilon)
        name = ACTION_NAMES.get(action_id, str(action_id))
        per_class[name] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": int(support[action_id]),
        }
        f1s.append(f1)
        # Always append so f1s and weights stay the same length; a class with
        # zero support gets zero weight, matching sklearn's average="weighted".
        weights.append(float(support[action_id]))

    macro_f1 = float(np.mean(f1s))
    total_support = float(np.sum(weights))
    weighted_f1 = (
        float(np.average(f1s, weights=weights)) if total_support > 0 else 0.0
    )
    return {
        "classes": per_class,
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
    }


def trajectory_metrics(
    per_step_predictions: dict[str, list[dict[str, int]]],
) -> dict[str, Any]:
    """Episode-level (trajectory-level) metrics from per-step predictions.

    Parameters
    ----------
    per_step_predictions : dict
        Maps episode_id → list of {"true": action_id, "pred": action_id} per step.

    Returns
    -------
    dict with exact_match_rate, mean_first_error_step, mean_norm_edit_distance.
    """
    num_episodes = len(per_step_predictions)
    if num_episodes == 0:
        return {
            "num_episodes": 0,
            "exact_match_rate": 0.0,
            "mean_first_error_step": None,
            "mean_norm_edit_distance": None,
        }

    exact_matches = 0
    first_error_steps: list[int] = []
    edit_distances_normalized: list[float] = []

    for _ep_id, steps in per_step_predictions.items():
        all_correct = True
        first_error = -1
        n = len(steps)
        preds = [s["pred"] for s in steps]
        trues = [s["true"] for s in steps]

        for i in range(n):
            if preds[i] != trues[i]:
                all_correct = False
                if first_error == -1:
                    first_error = i
        if all_correct:
            exact_matches += 1
        first_error_steps.append(first_error if first_error >= 0 else n)
        edit_distances_normalized.append(
            _normalized_levenshtein(preds, trues)
        )

    return {
        "num_episodes": num_episodes,
        "exact_match_rate": round(exact_matches / num_episodes, 4),
        "mean_first_error_step": round(float(np.mean(first_error_steps)), 2),
        "mean_norm_edit_distance": round(
            float(np.mean(edit_distances_normalized)), 4
        ),
    }


def scene_breakdown(
    per_step_predictions: dict[str, list[dict[str, int]]],
    episode_scenes: dict[str, int | str],
) -> dict[str, Any]:
    """Accuracy stratified by scene_id.

    Parameters
    ----------
    per_step_predictions : dict
        Maps episode_id → list of per-step {"true", "pred"} dicts.
    episode_scenes : dict
        Maps episode_id → scene_id.

    Returns
    -------
    dict mapping scene_id → {"accuracy", "samples", "episodes"}.
    """
    scene_correct: dict[int | str, int] = defaultdict(int)
    scene_samples: dict[int | str, int] = defaultdict(int)
    scene_episodes: dict[int | str, set[str]] = defaultdict(set)

    for episode_id, steps in per_step_predictions.items():
        scene = episode_scenes.get(episode_id, "unknown")
        for s in steps:
            scene_samples[scene] += 1
            if s["pred"] == s["true"]:
                scene_correct[scene] += 1
        scene_episodes[scene].add(episode_id)

    result: dict[str, Any] = {}
    for scene in sorted(scene_samples, key=lambda k: str(k)):
        total = scene_samples[scene]
        result[str(scene)] = {
            "accuracy": round(scene_correct[scene] / max(total, 1), 4),
            "samples": total,
            "episodes": len(scene_episodes[scene]),
        }
    return result


def calibration_metrics(
    logits: NDArray[np.float32],
    labels: NDArray[np.int64],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Expected/Maximum Calibration Error with uniform-mass binning.

    Parameters
    ----------
    logits : (N, NUM_ACTIONS) float array
    labels : (N,) int array
    n_bins : int

    Returns
    -------
    dict with ece, mce, reliability curve data.
    """
    probabilities = _softmax(logits)
    confidences = probabilities.max(axis=-1)
    predictions = probabilities.argmax(axis=-1)
    correct = (predictions == labels).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    total = len(labels)
    reliability_bins: list[dict] = []

    for i in range(n_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        count = int(mask.sum())
        if count > 0:
            bin_acc = float(correct[mask].mean())
            bin_conf = float(confidences[mask].mean())
            bin_error = abs(bin_acc - bin_conf)
        else:
            bin_acc = 0.0
            bin_conf = 0.0
            bin_error = 0.0
        ece += (count / max(total, 1)) * bin_error
        mce = max(mce, bin_error)
        reliability_bins.append(
            {
                "bin": i,
                "range": [round(float(lo), 2), round(float(hi), 2)],
                "count": count,
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
                "gap": round(bin_error, 4),
            }
        )

    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "n_bins": n_bins,
        "reliability_bins": reliability_bins,
    }


def step_proximity_accuracy(
    per_step_predictions: dict[str, list[dict[str, int]]],
    n_segments: int = 3,
) -> dict[str, Any]:
    """Accuracy by trajectory progress — start, middle, end.

    Parameters
    ----------
    per_step_predictions : dict
    n_segments : int — number of equal-width segments to split each episode into.

    Returns
    -------
    dict mapping segment name → {"accuracy", "samples"}.
    """
    segment_correct: dict[int, int] = defaultdict(int)
    segment_total: dict[int, int] = defaultdict(int)

    for _ep_id, steps in per_step_predictions.items():
        n = len(steps)
        if n == 0:
            continue
        for idx, s in enumerate(steps):
            segment = min(int(idx * n_segments / n), n_segments - 1)
            segment_total[segment] += 1
            if s["pred"] == s["true"]:
                segment_correct[segment] += 1

    result: dict[str, Any] = {}
    names = {0: "start", n_segments - 1: "end"}
    for seg in range(n_segments):
        label = names.get(seg, f"segment_{seg}")
        total = segment_total[seg]
        result[label] = {
            "accuracy": round(segment_correct[seg] / max(total, 1), 4),
            "samples": total,
        }
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _softmax(logits: NDArray[np.float32]) -> NDArray[np.float32]:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _normalized_levenshtein(a: list[int], b: list[int]) -> float:
    """Levenshtein distance normalised by the longer sequence length."""
    n, m = len(a), len(b)
    if max(n, m) == 0:
        return 0.0
    # One-row DP.
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev
    return float(prev[m]) / max(n, m)
