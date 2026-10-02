"""Offline fast validation — no simulator required.

Load a checkpoint + manifest JSONL, run a single inference pass, and produce a
comprehensive metrics report: per-step accuracy, per-class F1, per-trajectory
exact-match / first-error-step, per-scene breakdown, calibration errors, and
step-proximity accuracy.

Usage:
    python -m aerial_openvla.evaluation.fast_validate \\
        --manifest data/manifests/val_seen.jsonl \\
        --checkpoint runs/openvla-airvln-classifier/latest \\
        --output results/fastval/val_seen_report.json

Config-file usage (uses the same EvalConfig fields):
    python -m aerial_openvla.evaluation.fast_validate --config configs/fastval.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from aerial_openvla.core.actions import NUM_ACTIONS
from aerial_openvla.core.config import EvalConfig, load_yaml_dataclass
from aerial_openvla.data.dataset import AerialVLNClassificationDataset
from aerial_openvla.data.multiframe_dataset import MultiFrameClassificationDataset
from aerial_openvla.evaluation.offline_metrics import (
    calibration_metrics,
    per_action_metrics,
    scene_breakdown,
    step_proximity_accuracy,
    trajectory_metrics,
)
from aerial_openvla.evaluation.policy import create_policy
from aerial_openvla.utils.io import atomic_write_json


_STEP_MANIFEST_FIELDS = {"episode_id", "step_id", "image", "instruction", "action_id"}


def validate_step_manifest(manifest_path: str | Path) -> Path:
    """Validate that ``manifest_path`` is the step-level JSONL format this tool needs."""
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Step-level validation manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path} is not step-level JSONL (line {line_number} is not JSON). "
                    "Offline episode manifests from tools/data/build_fastval_manifest.py "
                    "are not accepted here."
                ) from exc
            if not isinstance(record, dict) or not _STEP_MANIFEST_FIELDS <= record.keys():
                missing = sorted(_STEP_MANIFEST_FIELDS - set(record) if isinstance(record, dict) else _STEP_MANIFEST_FIELDS)
                raise ValueError(
                    f"{path} is not a step-level validation JSONL manifest; missing {missing}. "
                    "Use data/manifests/val_seen.jsonl, or build an offline episode manifest "
                    "only for the offline cache evaluator."
                )
            return path
    raise ValueError(f"Step-level validation manifest is empty: {path}")


@dataclasses.dataclass(slots=True)
class FastValidationReport:
    manifest: str
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


class _FastValConfig:
    """Minimal duck-typed config for :func:`create_policy`."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            object.__setattr__(self, k, v)


class FastValidator:
    """Single-pass predictor + metric accumulator for manifest-based validation.

    Uses the policy factory so any ``model_type`` with a checkpoint can be
    evaluated without a simulator.  For policies that expose a raw ``.model``
    and ``.collator`` (e.g. OpenVLA) the validator runs batched inference;
    otherwise it falls back to the policy's ``.act()`` method.
    """

    def __init__(
        self,
        *,
        model_type: str,
        model_name_or_path: str,
        checkpoint_dir: str | Path,
        prompt_template: str,
        max_text_tokens: int,
        device: torch.device,
        use_flash_attention_2: bool,
        bf16: bool,
        batch_size: int = 8,
        airminivla_variant: str = "m1",
        num_frames: int = 4,
        use_prev_action: bool = True,
        use_previous_actions_text: bool = False,
        use_memory: bool = False,
        frame_history_stride: int = 2,
        missing_history: str = "error",
    ) -> None:
        cfg = _FastValConfig(
            model_type=model_type,
            model_name_or_path=model_name_or_path,
            checkpoint_dir=str(checkpoint_dir),
            prompt_template=prompt_template,
            max_text_tokens=max_text_tokens,
            use_flash_attention_2=use_flash_attention_2,
            bf16=bf16,
            airminivla_variant=airminivla_variant,
            num_frames=num_frames,
            use_prev_action=use_prev_action,
            use_previous_actions_text=use_previous_actions_text,
            use_memory=use_memory,
            frame_history_stride=frame_history_stride,
        )
        self.policy = create_policy(cfg, device)
        self.device = device
        self.bf16 = bf16
        self.batch_size = batch_size
        self.model_type = model_type
        self.use_memory = use_memory
        self.checkpoint_dir = str(checkpoint_dir)
        self.missing_history = missing_history
        self.frame_history_stride = frame_history_stride
        self._model: torch.nn.Module | None = getattr(self.policy, "model", None)
        self._collator = getattr(self.policy, "collator", None)

    def run(self, manifest_path: str | Path) -> FastValidationReport:
        """Run inference over the manifest and compute all metrics.

        Returns a FastValidationReport dataclass.
        """
        manifest = validate_step_manifest(manifest_path)
        if self.model_type == "airminivla":
            if self.use_memory:
                raise ValueError(
                    "Fast validation does not support M2 memory state. "
                    "Use the trainer's sequence-aware validation path."
                )
            dataset = MultiFrameClassificationDataset(
                manifest,
                num_frames=getattr(self.policy, "num_frames", 4),
                history_stride=self.frame_history_stride,
                missing_history=self.missing_history,
            )
        else:
            dataset = AerialVLNClassificationDataset(manifest)

        if self._model is None or self._collator is None:
            raise ValueError(
                "Fast validation requires a policy exposing .model and .collator "
                "for batched inference; the removed stepwise fallback produced "
                "incorrect calibration metrics and never reset stateful policies."
            )
        return self._run_batched(dataset, manifest)

    # ------------------------------------------------------------------
    # Batched path (OpenVLA-like policies with .model + .collator)
    # ------------------------------------------------------------------

    def _run_batched(
        self, dataset: AerialVLNClassificationDataset, manifest: Path
    ) -> FastValidationReport:
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=self._collator,
            drop_last=False,
        )

        all_logits: list[np.ndarray] = []
        all_labels: list[int] = []
        episode_predictions: dict[str, list[dict[str, int]]] = defaultdict(list)
        episode_scenes: dict[str, int | str] = {}
        confusion = torch.zeros((NUM_ACTIONS, NUM_ACTIONS), dtype=torch.float64)
        total_correct = 0
        total = 0

        model = self._model
        model.eval()
        device = self.device
        bf16 = self.bf16

        progress = tqdm(loader, desc="Fast validate")
        for batch in progress:
            logits_fp32, labels, ep_ids = _forward_batch(model, batch, device, bf16)
            predictions = logits_fp32.argmax(dim=-1)
            labels_np = labels.numpy()

            all_logits.append(logits_fp32.numpy())
            all_labels.extend(labels_np.tolist())

            total_correct += int((predictions == labels).sum().item())
            total += len(labels_np)

            for true_id, pred_id in zip(labels_np, predictions.tolist()):
                confusion[true_id, pred_id] += 1

            for i, ep_id in enumerate(ep_ids):
                episode_predictions[ep_id].append(
                    {"true": int(labels_np[i]), "pred": int(predictions[i].item())}
                )

            batch_start = total - len(labels_np)
            for i, ep_id in enumerate(ep_ids):
                record_idx = batch_start + i
                if record_idx < len(dataset):
                    scene = dataset.records[record_idx].scene_id
                    if scene is not None:
                        episode_scenes[ep_id] = scene

            progress.set_postfix(acc=round(total_correct / max(total, 1), 3))

        return _build_report(
            manifest=manifest,
            checkpoint_hint=self.checkpoint_dir,
            total=total,
            total_correct=total_correct,
            confusion=confusion,
            all_logits_cat=np.concatenate(all_logits, axis=0),
            all_labels_arr=np.array(all_labels, dtype=np.int64),
            episode_predictions=dict(episode_predictions),
            episode_scenes=episode_scenes,
        )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _forward_batch(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    bf16: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (logits, labels, episode_ids) for one batch.

    Builds keyword arguments from whichever tensor keys are present in the
    batch — the model's ``forward`` signature determines which keys it accepts.
    """
    tensor_kwargs: dict[str, torch.Tensor] = {}
    for key in (
        "input_ids", "attention_mask", "pixel_values", "frame_valid_mask",
        "prev_actions", "memory_state",
    ):
        if key in batch:
            val = batch[key]
            if key == "pixel_values":
                val = val.to(device, dtype=torch.bfloat16 if bf16 else torch.float32)
            else:
                val = val.to(device)
            tensor_kwargs[key] = val

    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            output = model(**tensor_kwargs)
        logits = output[0] if isinstance(output, tuple) else output
    return logits.float().detach().cpu(), batch["action_id"], batch["episode_id"]


def _build_report(
    *,
    manifest: Path,
    checkpoint_hint: str,
    total: int,
    total_correct: int,
    confusion: torch.Tensor,
    all_logits_cat: np.ndarray,
    all_labels_arr: np.ndarray,
    episode_predictions: dict[str, list[dict[str, int]]],
    episode_scenes: dict[str, int | str],
) -> FastValidationReport:
    confusion_np = confusion.numpy()

    per_step = {
        "accuracy": round(total_correct / max(total, 1), 4),
        "total_samples": total,
        "confusion": confusion_np.tolist(),
    }

    per_action = per_action_metrics(confusion_np)
    traj_metrics = trajectory_metrics(episode_predictions)
    scene = scene_breakdown(episode_predictions, episode_scenes)
    calibration = calibration_metrics(all_logits_cat, all_labels_arr)
    proximity = step_proximity_accuracy(episode_predictions)

    return FastValidationReport(
        manifest=str(manifest),
        checkpoint=checkpoint_hint,
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
        description="Offline fast validation (no simulator)"
    )
    parser.add_argument(
        "--config",
        help="Path to YAML config (same format as eval). "
        "When provided, model/checkpoint settings come from the config; "
        "--manifest and --output override the config defaults.",
    )
    parser.add_argument(
        "--manifest",
        help="JSONL manifest path (overrides config if provided)",
    )
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint directory (overrides config if provided)",
    )
    parser.add_argument(
        "--output",
        help="Output JSON path for the report (default: next to manifest, "
        "or fast_validation_report.json in cwd)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size (default: 32)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.config:
        cfg = load_yaml_dataclass(args.config, EvalConfig)
        model_type = getattr(cfg, "model_type", "openvla")
        model_name_or_path = cfg.model_name_or_path
        checkpoint = args.checkpoint or cfg.checkpoint_dir
        prompt_template = cfg.prompt_template
        max_text_tokens = cfg.max_text_tokens
        use_flash_attention_2 = cfg.use_flash_attention_2
        bf16 = cfg.bf16
        airminivla_variant = cfg.airminivla_variant
        num_frames = cfg.num_frames
        use_prev_action = cfg.use_prev_action
        use_previous_actions_text = cfg.use_previous_actions_text
        use_memory = cfg.use_memory
        frame_history_stride = cfg.frame_history_stride
        missing_history = cfg.missing_history
        manifest = args.manifest or getattr(cfg, "fast_manifest", None)
        output = args.output
    else:
        if not args.manifest or not args.checkpoint:
            print(
                "ERROR: --manifest and --checkpoint are required when --config is not provided.",
                file=sys.stderr,
            )
            sys.exit(1)
        model_type = "openvla"
        model_name_or_path = "openvla/openvla-7b"
        checkpoint = args.checkpoint
        prompt_template = (
            "In: What primitive action should the aerial robot take to follow "
            "the instruction: {instruction}?\nOut:"
        )
        max_text_tokens = 256
        use_flash_attention_2 = True
        bf16 = True
        airminivla_variant = "m1"
        num_frames = 4
        use_prev_action = True
        use_previous_actions_text = False
        use_memory = False
        frame_history_stride = 2
        missing_history = "error"
        manifest = args.manifest
        output = args.output

    if not manifest:
        print(
            "ERROR: --manifest is required (or set fast_manifest in config).",
            file=sys.stderr,
        )
        sys.exit(1)

    if model_type == "airminivla" and use_memory:
        raise ValueError(
            "Fast validation does not support M2 memory state. "
            "Use the trainer's sequence-aware validation path."
        )

    if output:
        out_path = Path(output)
    else:
        manifest_path = Path(manifest)
        out_path = manifest_path.parent / (
            manifest_path.stem + "_fast_validation_report.json"
        )

    device = torch.device("cuda", 0)
    if not torch.cuda.is_available():
        raise RuntimeError("Fast validation requires CUDA")

    validator = FastValidator(
        model_type=model_type,
        model_name_or_path=model_name_or_path,
        checkpoint_dir=checkpoint,
        prompt_template=prompt_template,
        max_text_tokens=max_text_tokens,
        device=device,
        use_flash_attention_2=use_flash_attention_2,
        bf16=bf16,
        batch_size=args.batch_size,
        airminivla_variant=airminivla_variant,
        num_frames=num_frames,
        use_prev_action=use_prev_action,
        use_previous_actions_text=use_previous_actions_text,
        use_memory=use_memory,
        frame_history_stride=frame_history_stride,
        missing_history=missing_history,
    )

    start = time.perf_counter()
    report = validator.run(manifest)
    elapsed = time.perf_counter() - start
    print(f"\nValidation completed in {elapsed:.1f}s")

    report_dict = report.to_dict()
    atomic_write_json(report_dict, out_path)
    print(f"Report written to {out_path}")

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
