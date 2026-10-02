from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Prevent PyTorch from spawning 96+ threads for tiny 224x224 tensor ops.
torch.set_num_threads(1)

from tqdm import tqdm

from aerial_openvla.core.actions import ACTION_NAMES
from aerial_openvla.core.config import EvalConfig, load_yaml_dataclass
from aerial_openvla.evaluation.airvln_adapter import AirVLNEvaluationAdapter
from aerial_openvla.evaluation.metrics import (
    aggregate_results,
    append_jsonl,
    load_completed_episode_ids,
)
from aerial_openvla.evaluation.policy import create_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    cfg = load_yaml_dataclass(parse_args().config, EvalConfig)
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires a CUDA GPU for OpenVLA")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda", 0)

    openvla_root = Path(cfg.openvla_root).resolve()
    if openvla_root.is_dir() and str(openvla_root) not in sys.path:
        sys.path.insert(0, str(openvla_root))

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "episodes.jsonl"
    completed = load_completed_episode_ids(results_path) if cfg.resume else set()
    if not cfg.resume and results_path.exists():
        results_path.unlink()

    policy = create_policy(cfg, device)

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
        # The verified four-worker runner supplies an explicit partition.  Use
        # its episode IDs rather than arithmetic slicing so non-divisible
        # splits (notably val_unseen 530 = 133/133/132/132) remain exact.
        worker_id = int(os.environ.get("AIRVLN_WORKER_ID", "0"))
        num_workers = int(os.environ.get("AIRVLN_NUM_WORKERS", "1"))
        assignment_path = os.environ.get("AIRVLN_EPISODE_ASSIGNMENT")
        if assignment_path:
            assignment = json.loads(Path(assignment_path).read_text(encoding="utf-8"))
            if int(assignment.get("num_workers", -1)) != num_workers:
                raise RuntimeError(
                    "AIRVLN_EPISODE_ASSIGNMENT worker count does not match rollout"
                )
            workers = assignment.get("workers", [])
            if worker_id >= len(workers):
                raise RuntimeError(f"assignment has no worker {worker_id}")
            requested_ids = [str(value) for value in workers[worker_id]["episode_ids"]]
            all_ids = [str(episode["episode_id"]) for episode in environment.episodes]
            if len(all_ids) != len(set(all_ids)):
                raise RuntimeError("environment returned duplicate episode IDs")
            available = {str(episode["episode_id"]): episode for episode in environment.episodes}
            missing = [episode_id for episode_id in requested_ids if episode_id not in available]
            if missing:
                raise RuntimeError(
                    f"assignment episodes unavailable in split {cfg.split}: {missing[:4]}"
                )
            environment.episodes = [available[episode_id] for episode_id in requested_ids]
            print(
                f"Worker {worker_id}/{num_workers}: assignment="
                f"{len(requested_ids)} episodes (explicit IDs)"
            )
        else:
            total_eps = min(
                len(environment.episodes),
                cfg.eval_num if cfg.eval_num > 0 else len(environment.episodes),
            )
            per_worker = total_eps // num_workers
            start_idx = worker_id * per_worker
            end_idx = total_eps if worker_id == num_workers - 1 else start_idx + per_worker
            environment.episodes = environment.episodes[start_idx:end_idx]
            print(
                f"Worker {worker_id}/{num_workers}: "
                f"episodes [{start_idx}, {end_idx}) = {len(environment.episodes)}"
            )

        act_times_all: list[float] = []
        step_times_all: list[float] = []

        progress = tqdm(range(len(environment.episodes)), desc=f"AirVLN-S w{worker_id}")
        for episode_index in progress:
            episode_id = str(environment.episodes[episode_index]["episode_id"])
            if episode_id in completed:
                continue
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
                record_executed_action = getattr(
                    policy, "record_executed_action", None
                )
                if callable(record_executed_action):
                    record_executed_action(action_id)
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
                        "episode_id": episode_id,
                        "step": len(trace),
                        "action": action_id,
                        **task_state,
                    }
                chunk_probabilities = getattr(
                    policy, "last_action_chunk_probabilities", None
                )
                if chunk_probabilities is not None:
                    action_row["action_chunk_probabilities"] = chunk_probabilities
                    action_row["action_chunk_slot0_probabilities"] = getattr(
                        policy, "last_action_chunk_slot0_probabilities", None
                    )
                    action_row["action_chunk_temporal_ensemble"] = bool(
                        getattr(policy, "action_chunk_temporal_ensemble", False)
                    )
                    navigation_probabilities = getattr(
                        policy, "last_action_chunk_navigation_probabilities", None
                    )
                    if navigation_probabilities is not None:
                        action_row["action_chunk_navigation_probabilities"] = (
                            navigation_probabilities
                        )
                        action_row["action_chunk_stop_probabilities"] = getattr(
                            policy, "last_action_chunk_stop_probabilities", None
                        )
                        action_row["hierarchical_stop_probability"] = getattr(
                            policy, "last_stop_probability", None
                        )
                        action_row["hierarchical_stop_threshold"] = getattr(
                            policy, "hierarchical_stop_threshold", None
                        )
                trace.append(action_row)
                if len(trace) > cfg.max_actions + 2:
                    raise RuntimeError(
                        f"Episode {episode_id} exceeded the configured AirVLN max action limit; "
                        "check environment termination."
                    )

            act_times_all.extend(act_times)
            step_times_all.extend(step_times)
            if len(trace) > 0:
                gap = 100 if len(trace) >= 500 else max(1, len(trace) // 5)
                for n in range(gap, len(trace) + 1, gap):
                    sub_a = act_times[:n]
                    sub_s = step_times[:n]
                    print(
                        f"  [{n} steps] avg_act={np.mean(sub_a):.3f}s "
                        f"avg_step={np.mean(sub_s):.3f}s "
                        f"total_per_step={np.mean(sub_a) + np.mean(sub_s):.3f}s "
                        f"elapsed={sum(sub_a) + sum(sub_s):.0f}s"
                    )

            row = {
                "episode_id": episode_id,
                "trajectory_id": str(environment.current_episode.get("trajectory_id", "")),
                "scene_id": environment.current_episode.get("scene_id"),
                "instruction": instruction,
                "actions": trace,
                "final_info": info,
            }
            append_jsonl(results_path, row)
            completed.add(episode_id)
            progress.set_postfix(
                sr=float(info.get("success", 0.0)),
                ne=float(info.get("distance_to_goal", 0.0)),
                steps=int(info.get("steps_taken", len(trace))),
            )

    summary = aggregate_results(results_path)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
