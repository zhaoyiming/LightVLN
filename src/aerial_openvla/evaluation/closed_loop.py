"""Closed-loop fast validation with AirSim simulator.

Load a closed-loop manifest (subset of episode IDs per scene), run the model
through the real AirVLN simulator, and produce per-episode results + aggregated
metrics.

Key features:
- Subset of full dataset: 2-3 trajectories per scene (~16-24 episodes, ~20-45 min)
- Robust retry: each failed episode retried up to 3 times
- Per-episode JSON output + scene-macro aggregated summary
- Resume support via existing episodes.jsonl

Usage:
    python -m aerial_openvla.evaluation.closed_loop \\
        --config configs/eval.example.yaml \\
        --manifest manifests/airvlns_closed_seen_v1.json \\
        --output-dir results/closed_loop/val_seen \\
        --max-retries 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

torch.set_num_threads(1)

from tqdm import tqdm

from aerial_openvla.core.actions import ACTION_NAMES
from aerial_openvla.core.config import EvalConfig, load_yaml_dataclass
from aerial_openvla.evaluation.airvln_adapter import AirVLNEvaluationAdapter
from aerial_openvla.evaluation.metrics import (
    METRIC_KEYS,
    append_jsonl,
    load_completed_episode_ids,
)
from aerial_openvla.evaluation.policy import create_policy
from aerial_openvla.utils.io import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Closed-loop fast validation with AirSim"
    )
    parser.add_argument("--config", required=True, help="Eval YAML config")
    parser.add_argument(
        "--manifest",
        required=True,
        help="Closed-loop manifest JSON (from tools/build_closed_manifest.py)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: outputs/closed_loop/<split>)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per failed episode (default: 3)",
    )
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load the closed-loop manifest and return flat episode_id -> scene_id map."""
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    ep_to_scene: dict[str, str] = {}
    all_episode_ids: list[str] = []
    for scene_id, scene_info in data.get("scenes", {}).items():
        for ep_id in scene_info.get("episode_ids", []):
            ep_to_scene[str(ep_id)] = scene_id
            all_episode_ids.append(str(ep_id))
    return {
        "split": data.get("split", "unknown"),
        "ep_to_scene": ep_to_scene,
        "episode_ids": all_episode_ids,
    }


def run_episode(
    environment: AirVLNEvaluationAdapter,
    policy: Any,
    episode_index: int,
    max_actions: int,
) -> dict[str, Any]:
    """Run a single episode and return the result row."""
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()
    image, info = environment.start_episode(episode_index)
    instruction = environment.instruction
    trace: list[dict] = []
    act_times: list[float] = []
    step_times: list[float] = []

    while not info.get("done", False):
        t0 = time.perf_counter()
        action_id, probabilities = policy.act(image, instruction)
        t1 = time.perf_counter()
        image, info = environment.step(action_id)
        t2 = time.perf_counter()
        act_times.append(t1 - t0)
        step_times.append(t2 - t1)
        action_row = {
            "step": len(trace),
            "action_id": action_id,
            "action_name": ACTION_NAMES[action_id],
            "probabilities": probabilities,
        }
        task_state = getattr(policy, "last_task_state", None)
        if task_state is not None:
            action_row["task_state"] = {
                "episode_id": str(environment.current_episode.get("episode_id", "")),
                "step": len(trace),
                "action": action_id,
                **task_state,
            }
        trace.append(action_row)
        if len(trace) > max_actions + 2:
            raise RuntimeError(
                f"Episode exceeded max actions limit ({max_actions}); "
                "check environment termination."
            )

    episode_id = str(environment.current_episode.get("episode_id", ""))
    return {
        "episode_id": episode_id,
        "trajectory_id": str(environment.current_episode.get("trajectory_id", "")),
        "scene_id": environment.current_episode.get("scene_id"),
        "instruction": instruction,
        "actions": trace,
        "final_info": info,
        "timing": {
            "act_mean": float(np.mean(act_times)) if act_times else 0.0,
            "step_mean": float(np.mean(step_times)) if step_times else 0.0,
            "total_steps": len(trace),
        },
    }


def _find_episode_index(
    environment: AirVLNEvaluationAdapter, target_id: str
) -> int | None:
    """Find the index of an episode ID in the environment's episode list."""
    for i, ep in enumerate(environment.episodes):
        if str(ep.get("episode_id", "")) == target_id:
            return i
    return None


def aggregate_scene_macro(
    rows: list[dict[str, Any]], ep_to_scene: dict[str, str]
) -> dict[str, Any]:
    """Scene-macro aggregation: average per-scene metrics, then average across scenes.

    Error stubs (rows whose ``final_info`` carries an ``error`` key) are
    excluded: their sentinel values (``distance_to_goal=-1.0`` and zeroed
    metrics) would otherwise bias every mean.  Failures are reported separately
    as ``excluded_errors``.
    """
    excluded_errors = sum(1 for row in rows if "error" in row.get("final_info", {}))
    rows = [row for row in rows if "error" not in row.get("final_info", {})]
    scene_values: dict[str, list[dict[str, float]]] = defaultdict(list)

    for row in rows:
        ep_id = str(row.get("episode_id", ""))
        scene = ep_to_scene.get(ep_id, "unknown")
        final = row.get("final_info", {})
        ep_metrics = {}
        for key in METRIC_KEYS:
            if key in final:
                ep_metrics[key] = float(final[key])
        scene_values[scene].append(ep_metrics)

    # Per-scene means
    scene_means: dict[str, dict[str, float]] = {}
    for scene, metrics_list in scene_values.items():
        scene_means[scene] = {}
        for key in METRIC_KEYS:
            vals = [m[key] for m in metrics_list if key in m]
            scene_means[scene][key] = float(np.mean(vals)) if vals else 0.0
        scene_means[scene]["episodes"] = len(metrics_list)

    # Macro average across scenes
    macro: dict[str, float] = {}
    for key in METRIC_KEYS:
        vals = [sm[key] for sm in scene_means.values() if key in sm]
        macro[key] = float(np.mean(vals)) if vals else 0.0
    macro["num_scenes"] = len(scene_means)

    # Episode-average (standard aggregation)
    ep_average: dict[str, float] = {}
    for key in METRIC_KEYS:
        vals = [
            float(row["final_info"][key])
            for row in rows
            if key in row.get("final_info", {})
        ]
        ep_average[key] = float(np.mean(vals)) if vals else 0.0
    ep_average["num_episodes"] = len(rows)

    return {
        "scene_macro": macro,
        "scene_breakdown": scene_means,
        "episode_average": ep_average,
        "excluded_errors": excluded_errors,
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml_dataclass(args.config, EvalConfig)
    manifest_info = load_manifest(Path(args.manifest))

    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires a CUDA GPU for OpenVLA")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda", 0)

    openvla_root = Path(cfg.openvla_root).resolve()
    if openvla_root.is_dir() and str(openvla_root) not in sys.path:
        sys.path.insert(0, str(openvla_root))

    output_dir = Path(
        args.output_dir
        or f"outputs/closed_loop/{manifest_info['split']}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "episodes.jsonl"
    completed = load_completed_episode_ids(results_path) if cfg.resume else set()
    if not cfg.resume and results_path.exists():
        results_path.unlink()

    policy = create_policy(cfg, device)

    target_episodes = manifest_info["episode_ids"]
    ep_to_scene = manifest_info["ep_to_scene"]
    max_retries = args.max_retries

    print(f"Closed-loop evaluation: {len(target_episodes)} target episodes")
    print(f"  Already completed: {len(completed)}")
    print(f"  Max retries: {max_retries}")

    with AirVLNEvaluationAdapter(
        airvln_root=cfg.airvln_root,
        project_root=cfg.airvln_project_root,
        dataset_dir=cfg.aerialvln_s_data_dir,
        split=cfg.split,
        max_actions=cfg.max_actions,
        simulator_port=cfg.simulator_port,
        eval_num=cfg.eval_num,
        image_height_rgb=cfg.image_height_rgb,
        image_width_rgb=cfg.image_width_rgb,
        reset_warmup_seconds=cfg.reset_warmup_seconds,
        seed=cfg.seed,
    ) as environment:
        # Filter to manifest episodes
        manifest_episodes: list[tuple[int, str]] = []
        skipped: list[str] = []
        for target_id in target_episodes:
            idx = _find_episode_index(environment, target_id)
            if idx is not None:
                manifest_episodes.append((idx, target_id))
            elif target_id not in completed:
                skipped.append(target_id)

        if skipped:
            print(f"  WARNING: {len(skipped)} episodes not found in environment: {skipped}")

        pending = [
            (idx, eid)
            for idx, eid in manifest_episodes
            if eid not in completed
        ]
        print(f"  Pending: {len(pending)}")

        progress = tqdm(pending, desc="Closed-loop")
        all_rows: list[dict[str, Any]] = []
        if cfg.resume and results_path.is_file():
            with results_path.open("r", encoding="utf-8") as handle:
                all_rows = [json.loads(line) for line in handle if line.strip()]

        for episode_index, episode_id in progress:
            for attempt in range(1, max_retries + 1):
                try:
                    row = run_episode(
                        environment, policy, episode_index, cfg.max_actions
                    )
                    append_jsonl(results_path, row)
                    completed.add(episode_id)
                    all_rows.append(row)
                    final = row.get("final_info", {})
                    progress.set_postfix(
                        sr=float(final.get("success", 0.0)),
                        ne=float(final.get("distance_to_goal", 0.0)),
                        att=attempt,
                    )
                    break
                except Exception as exc:
                    print(
                        f"\n  Episode {episode_id} attempt {attempt}/{max_retries} "
                        f"failed: {exc}"
                    )
                    if attempt < max_retries:
                        time.sleep(2.0)
                    else:
                        # Final attempt failed — write stub
                        print(
                            f"  Episode {episode_id} FAILED after {max_retries} attempts"
                        )
                        stub = {
                            "episode_id": episode_id,
                            "trajectory_id": "",
                            "scene_id": ep_to_scene.get(episode_id, "unknown"),
                            "instruction": "",
                            "actions": [],
                            "final_info": {
                                "success": 0,
                                "distance_to_goal": -1.0,
                                "ndtw": 0.0,
                                "sdtw": 0.0,
                                "path_length": 0.0,
                                "oracle_success": 0,
                                "steps_taken": 0,
                                "error": str(exc),
                            },
                            "timing": {"act_mean": 0.0, "step_mean": 0.0, "total_steps": 0},
                        }
                        append_jsonl(results_path, stub)
                        all_rows.append(stub)

    # --- Aggregate & Report -----------------------------------------------
    # Deduplicate by episode_id, keeping the most recent row: a retried episode
    # supersedes the error stub an earlier run appended for it.
    deduped: dict[str, dict[str, Any]] = {}
    for row in all_rows:
        if isinstance(row, dict):
            deduped[str(row.get("episode_id", ""))] = row
    evaluated = list(deduped.values())
    error_rows = [r for r in evaluated if "error" in r.get("final_info", {})]
    completed_rows = [r for r in evaluated if "error" not in r.get("final_info", {})]
    navigation_successes = [
        r for r in completed_rows if float(r.get("final_info", {}).get("success", 0.0)) > 0.0
    ]
    print(
        f"\nCompleted: {len(completed_rows)} episodes; navigation success: "
        f"{len(navigation_successes)}; errors: {len(error_rows)}"
    )

    aggregation = aggregate_scene_macro(evaluated, ep_to_scene)

    summary = {
        "config": {
            "config_file": str(args.config),
            "manifest_file": str(args.manifest),
            "checkpoint": str(cfg.checkpoint_dir),
            "split": cfg.split,
            "max_retries": max_retries,
        },
        "counts": {
            "total_target": len(target_episodes),
            "completed": len(completed_rows),
            "navigation_successes": len(navigation_successes),
            "errors": len(error_rows),
            "not_evaluated": len(
                set(map(str, target_episodes)) - set(deduped)
            ),
        },
        "aggregation": aggregation,
    }

    atomic_write_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, indent=2))

    # Also print a compact summary
    print("\n=== Scene-Macro Averages ===")
    for key in METRIC_KEYS:
        print(f"  {key}: {aggregation['scene_macro'].get(key, 0):.4f}")
    print(f"  scenes: {aggregation['scene_macro'].get('num_scenes', 0)}")


if __name__ == "__main__":
    main()
