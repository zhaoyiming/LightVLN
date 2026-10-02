"""Offline evaluation against a pre-cached frame dataset.

Loads RGB frames and ground-truth actions from an offline cache, runs a
single batched inference pass through the model, and computes the full
offline-metrics report.

Unlike ``fast_validate.py``, this tool does *not* require access to the
original RGB directories — it works entirely from the offline cache
built by ``tools/build_offline_cache.py``.

Usage::

    python -m aerial_openvla.evaluation.eval_offline \\
        --config configs/eval-bench.yaml \\
        --manifest manifests/airvlns_offline_seen_v1.json \\
        --cache offline_cache/val_seen_v1 \\
        --batch-size 16 \\
        --output results/offline/step-0033000
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from aerial_openvla.core.actions import NUM_ACTIONS
from aerial_openvla.core.config import EvalConfig, load_yaml_dataclass
from aerial_openvla.data.cached_frames import CachedFrameCollator, CachedFrameDataset
from aerial_openvla.evaluation.offline_metrics import (
    calibration_metrics,
    per_action_metrics,
    scene_breakdown,
    step_proximity_accuracy,
    trajectory_metrics,
)
from aerial_openvla.evaluation.policy import create_policy
from aerial_openvla.utils.io import atomic_write_json


# ---------------------------------------------------------------------------
# Report dataclass — mirrors FastValidationReport so the output schema is
# identical across fast_validate and eval_offline.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class OfflineReport:
    manifest: str
    cache: str
    checkpoint: str
    total_samples: int
    per_step: dict[str, Any]
    per_action_class: dict[str, Any]
    per_trajectory: dict[str, Any]
    per_scene: dict[str, Any]
    calibration: dict[str, Any]
    step_proximity: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _forward_batch(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    bf16: bool,
) -> tuple[np.ndarray, list[int], list[str], list[str]]:
    """Return ``(logits_np, labels, episode_ids, scene_ids)`` for one batch.

    Builds keyword arguments from whichever tensor keys are present in the
    batch — the model's ``forward`` signature determines which keys it accepts.
    """
    tensor_kwargs: dict[str, torch.Tensor] = {}
    for key in ("input_ids", "attention_mask", "pixel_values", "frame_valid_mask"):
        if key in batch:
            val = batch[key]
            if key == "pixel_values":
                val = val.to(device, dtype=torch.bfloat16 if bf16 else torch.float32)
            else:
                val = val.to(device)
            tensor_kwargs[key] = val

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        logits = model(**tensor_kwargs)

    return (
        logits.float().detach().cpu().numpy(),
        batch.get("action_id", torch.tensor([-1])).tolist(),
        batch.get("episode_id", []),
        batch.get("scene_id", []),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_eval(
    policy: Any,
    cache_dir: Path,
    batch_size: int,
    device: torch.device,
    bf16: bool,
) -> OfflineReport:
    dataset = CachedFrameDataset(cache_dir)
    collator = CachedFrameCollator(policy)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collator,
        drop_last=False,
    )

    model = getattr(policy, "model", None)
    if model is None:
        raise RuntimeError(
            "Policy does not expose a 'model' attribute for batched inference"
        )
    model.eval()

    # Accumulators
    all_logits: list[np.ndarray] = []
    all_labels: list[int] = []
    episode_predictions: dict[str, list[dict[str, int]]] = defaultdict(list)
    episode_scenes: dict[str, int | str] = {}
    confusion = torch.zeros((NUM_ACTIONS, NUM_ACTIONS), dtype=torch.float64)
    total_correct = 0
    total = 0

    progress = tqdm(loader, desc="Offline eval")
    for batch in progress:
        logits_np, labels, ep_ids, scene_ids = _forward_batch(
            model, batch, device, bf16
        )
        predictions = np.argmax(logits_np, axis=-1)

        all_logits.append(logits_np)
        all_labels.extend(labels)

        for true_id, pred_id in zip(labels, predictions):
            confusion[true_id, pred_id] += 1

        total_correct += int((predictions == np.array(labels)).sum())
        total += len(labels)

        for i, ep_id in enumerate(ep_ids):
            episode_predictions[ep_id].append(
                {"true": int(labels[i]), "pred": int(predictions[i])}
            )
            if i < len(scene_ids) and scene_ids[i]:
                episode_scenes[ep_id] = scene_ids[i]

        progress.set_postfix(acc=round(total_correct / max(total, 1), 3))

    # --- Compute metrics ---
    confusion_np = confusion.numpy()
    all_logits_cat = np.concatenate(all_logits, axis=0)
    all_labels_arr = np.array(all_labels, dtype=np.int64)

    per_step = {
        "accuracy": round(total_correct / max(total, 1), 4),
        "total_samples": total,
        "confusion": confusion_np.tolist(),
    }
    per_action = per_action_metrics(confusion_np)
    traj_metrics = trajectory_metrics(dict(episode_predictions))
    scene = scene_breakdown(dict(episode_predictions), episode_scenes)
    calibration = calibration_metrics(all_logits_cat, all_labels_arr)
    proximity = step_proximity_accuracy(dict(episode_predictions))

    return OfflineReport(
        manifest="(via cache)",
        cache=str(cache_dir),
        checkpoint=str(getattr(policy, "processor", "unknown")),
        total_samples=total,
        per_step=per_step,
        per_action_class=per_action,
        per_trajectory=traj_metrics,
        per_scene=scene,
        calibration=calibration,
        step_proximity=proximity,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline evaluation against a cached frame dataset"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Eval YAML config (provides model / checkpoint settings)",
    )
    parser.add_argument(
        "--manifest",
        help="Fastval manifest JSON (optional; for metadata / validation). "
        "When omitted the manifest field in the report is set to '(via cache)'.",
    )
    parser.add_argument(
        "--cache",
        required=True,
        help="Offline cache directory containing metadata.jsonl and rgb/",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size (default: 16)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for the offline_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_dataclass(args.config, EvalConfig)

    if not torch.cuda.is_available():
        raise RuntimeError("Offline eval requires CUDA")
    device = torch.device("cuda", 0)

    # Validate input paths
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_file():
            print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
            sys.exit(1)

    cache_dir = Path(args.cache)
    if not cache_dir.is_dir():
        print(f"ERROR: cache dir not found: {cache_dir}", file=sys.stderr)
        sys.exit(1)
    if not (cache_dir / "metadata.jsonl").is_file():
        print(
            f"ERROR: metadata.jsonl not found in cache: {cache_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_type = getattr(cfg, "model_type", "openvla")
    if model_type == "airminivla":
        raise NotImplementedError(
            "Cached-frame offline evaluation is single-frame only. "
            "Use fast_validate.py for stateless AirMiniVLA variants or closed_loop.py "
            "for stateful rollout evaluation."
        )
    print(f"Loading policy via create_policy (model_type={model_type})...")
    policy = create_policy(cfg, device)

    start = time.perf_counter()
    report = run_eval(policy, cache_dir, args.batch_size, device, cfg.bf16)
    elapsed = time.perf_counter() - start

    # Save report
    report_dict = report.to_dict()
    report_path = output_dir / "offline_report.json"
    atomic_write_json(report_dict, report_path)

    # Print summary
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Report: {report_path}")
    print()
    print(f"  Samples:      {report.total_samples}")
    print(f"  Accuracy:     {report.per_step['accuracy']:.4f}")
    print(f"  Macro F1:     {report.per_action_class['macro_f1']:.4f}")
    print(f"  Weighted F1:  {report.per_action_class['weighted_f1']:.4f}")
    print(f"  Exact Match:  {report.per_trajectory['exact_match_rate']:.4f}")
    print(f"  1st Err Step: {report.per_trajectory['mean_first_error_step']:.1f}")
    print(f"  ECE:          {report.calibration['ece']:.4f}")

    print("\n  Per-class F1:")
    for cls_name, metrics in report.per_action_class["classes"].items():
        print(
            f"    {cls_name:>15}: F1={metrics['f1']:.4f} "
            f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
            f"({metrics['support']} samples)"
        )


if __name__ == "__main__":
    main()
