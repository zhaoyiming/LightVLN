from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import time
from contextlib import nullcontext
from dataclasses import MISSING, fields as dataclass_fields
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from tqdm import tqdm

from aerial_openvla.core.config import TrainConfig, load_yaml_dataclass
from aerial_openvla.data.pipeline import create_dataset, create_pipeline
from aerial_openvla.models.base import collect_lr_kwargs
from aerial_openvla.models.factory import create_model
from aerial_openvla.utils.distributed import (
    barrier,
    cleanup_distributed,
    init_distributed,
    reduce_sum,
    seed_everything,
)
from aerial_openvla.utils.io import update_latest_symlink


class Stage1TimingProfiler:
    """Low-overhead wall/CUDA timing for a bounded Native1T Stage1 run."""

    def __init__(self, *, rank: int, world_size: int, device: torch.device) -> None:
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.warmup_updates = int(os.environ.get("OPENFLY_STAGE1_PROFILE_WARMUP_UPDATES", "10"))
        self.measure_updates = int(os.environ.get("OPENFLY_STAGE1_PROFILE_MEASURE_UPDATES", "40"))
        self.phase_seconds: dict[str, float] = defaultdict(float)
        self.phase_calls: dict[str, int] = defaultdict(int)
        self.module_seconds: dict[str, float] = defaultdict(float)
        self.module_calls: dict[str, int] = defaultdict(int)
        self._module_stacks: dict[str, list[torch.cuda.Event]] = defaultdict(list)
        self._module_pending: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
        self.active = False
        self.loop_end = time.perf_counter()
        self.window_start: float | None = None
        self.window_end: float | None = None
        self.measured_microbatches = 0
        self.measured_updates = 0
        self.peak_allocated_bytes = 0
        self.peak_reserved_bytes = 0
        self._handles: list[Any] = []

    def should_measure(self, global_step: int) -> bool:
        return self.warmup_updates <= global_step < self.warmup_updates + self.measure_updates

    def begin_microbatch(self, global_step: int) -> None:
        now = time.perf_counter()
        self.active = self.should_measure(global_step)
        if self.active:
            if self.window_start is None:
                self.window_start = now
                torch.cuda.reset_peak_memory_stats(self.device)
            self.add("data_next_wait", now - self.loop_end)
            self.measured_microbatches += 1

    def end_microbatch(self, *, did_update: bool) -> None:
        now = time.perf_counter()
        if self.active:
            self.window_end = now
            if did_update:
                self.measured_updates += 1
            self.peak_allocated_bytes = max(
                self.peak_allocated_bytes, torch.cuda.max_memory_allocated(self.device)
            )
            self.peak_reserved_bytes = max(
                self.peak_reserved_bytes, torch.cuda.max_memory_reserved(self.device)
            )
        self.loop_end = now

    def add(self, name: str, seconds: float) -> None:
        self.phase_seconds[name] += float(seconds)
        self.phase_calls[name] += 1

    def timed_cuda_start(self) -> float:
        torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def timed_cuda_end(self, name: str, started: float) -> None:
        torch.cuda.synchronize(self.device)
        self.add(name, time.perf_counter() - started)

    def install_module_hook(self, name: str, module: torch.nn.Module | None) -> None:
        if module is None:
            return

        def before(_module: torch.nn.Module, _inputs: tuple[Any, ...]) -> None:
            if not self.active:
                return
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._module_stacks[name].append(event)

        def after(
            _module: torch.nn.Module,
            _inputs: tuple[Any, ...],
            _output: Any,
        ) -> None:
            if not self.active:
                return
            if not self._module_stacks[name]:
                raise RuntimeError(f"Stage1 profiler hook stack underflow for {name}")
            started = self._module_stacks[name].pop()
            ended = torch.cuda.Event(enable_timing=True)
            ended.record()
            self._module_pending[name].append((started, ended))

        self._handles.append(module.register_forward_pre_hook(before))
        self._handles.append(module.register_forward_hook(after))

    def flush_module_events(self) -> None:
        for name, pairs in self._module_pending.items():
            for started, ended in pairs:
                self.module_seconds[name] += started.elapsed_time(ended) / 1000.0
                self.module_calls[name] += 1
            pairs.clear()

    def write(self, output_dir: Path, cfg: TrainConfig, global_step: int) -> None:
        for handle in self._handles:
            handle.remove()
        window_seconds = (
            self.window_end - self.window_start
            if self.window_start is not None and self.window_end is not None
            else 0.0
        )
        rows = self.measured_microbatches * int(cfg.batch_size)
        loop_phase_total = sum(
            value
            for key, value in self.phase_seconds.items()
            if key != "final_checkpoint_save"
        )
        payload = {
            "format": "openfly_native1t_stage1_phase_profile_v1",
            "rank": self.rank,
            "world_size": self.world_size,
            "warmup_updates": self.warmup_updates,
            "requested_measure_updates": self.measure_updates,
            "measured_updates": self.measured_updates,
            "measured_microbatches": self.measured_microbatches,
            "measured_rows_per_rank": rows,
            "batch_size_per_rank": int(cfg.batch_size),
            "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps),
            "num_workers_per_rank": int(cfg.num_workers),
            "global_step_at_exit": int(global_step),
            "window_seconds": window_seconds,
            "rows_per_second_per_rank": rows / window_seconds if window_seconds else 0.0,
            "phase_seconds": dict(sorted(self.phase_seconds.items())),
            "phase_calls": dict(sorted(self.phase_calls.items())),
            "loop_phase_total_seconds": loop_phase_total,
            "phase_percent_of_accounted_loop": {
                key: value / loop_phase_total * 100.0 if loop_phase_total else 0.0
                for key, value in sorted(self.phase_seconds.items())
                if key != "final_checkpoint_save"
            },
            "module_cuda_seconds_nested_in_forward": dict(sorted(self.module_seconds.items())),
            "module_calls": dict(sorted(self.module_calls.items())),
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"stage1_timing_rank_{self.rank:02d}.json"
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)

class OffsetSampler(Sampler[int]):
    """Skip an already-consumed prefix without loading those samples again."""

    def __init__(self, sampler: Sampler[int]) -> None:
        self.sampler = sampler
        self.offset = 0

    def __iter__(self):
        return itertools.islice(iter(self.sampler), self.offset, None)

    def __len__(self) -> int:
        return max(0, len(self.sampler) - self.offset)


class EpochRandomSampler(Sampler[int]):
    """Deterministic per-epoch shuffling for single-process resume."""

    def __init__(self, dataset: Dataset[Any], seed: int) -> None:
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        yield from torch.randperm(len(self.dataset), generator=generator).tolist()

    def __len__(self) -> int:
        return len(self.dataset)


def capture_rng_state() -> dict[str, Any]:
    """Capture all local RNGs that influence model and worker initialization."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def capture_ranked_rng_states(rank: int, world_size: int, distributed: bool) -> dict[str, dict[str, Any]]:
    """Collect rank-local states so DDP resume does not reuse rank zero's RNG."""
    local_state = capture_rng_state()
    if not distributed:
        return {str(rank): local_state}
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_state)
    return {str(index): state for index, state in enumerate(gathered) if state is not None}


def file_sha256(path: str | Path) -> str:
    """Return a content digest for a manifest without retaining it in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_contract(cfg: TrainConfig) -> dict[str, Any]:
    """Record the exact configuration and manifests required for safe resume."""
    contract: dict[str, Any] = {"config": config_to_dict(cfg)}
    for field in ("train_manifest", "val_manifest"):
        manifest = getattr(cfg, field)
        if manifest:
            resolved = Path(manifest).resolve()
            contract[field] = {"path": str(resolved), "sha256": file_sha256(resolved)}
    if getattr(cfg, "openfly_action_head_type", "categorical") in {
        "action_tokens_v1", "action_tokens_ar_v2"
    }:
        codec_path = Path(str(cfg.openfly_action_codec_path or "")).resolve()
        if not codec_path.is_file():
            raise FileNotFoundError(
                "OpenFly action-token training contract is missing its codec: "
                f"{codec_path}"
            )
        contract["openfly_action_codec"] = {
            "path": str(codec_path),
            "file_sha256": file_sha256(codec_path),
            # This is the canonical payload digest embedded in the codec, not
            # the JSON file's byte digest above.
            "payload_sha256": str(cfg.openfly_action_codec_sha256),
        }
    native1t_source = getattr(cfg, "native1t_stage1_checkpoint", None)
    if (
        cfg.use_native1t_postfusion
        and int(getattr(cfg, "native1t_stage", 2)) == 2
        and (native1t_source or cfg.init_checkpoint)
    ):
        tokenizer_filename = (
            "qwen35_native1t.pt"
            if getattr(cfg, "model_type", None) == "qwen35_native"
            else "native1t_postfusion.pt"
        )
        tokenizer_file = (
            Path(native1t_source or cfg.init_checkpoint).resolve()
            / tokenizer_filename
        )
        if not tokenizer_file.is_file():
            raise FileNotFoundError(
                f"Native1T Stage-1 tokenizer checkpoint is missing: {tokenizer_file}"
            )
        contract["native1t_stage1_tokenizer"] = {
            "path": str(tokenizer_file),
            "sha256": file_sha256(tokenizer_file),
        }
    return contract


def native1t_trainable_contract_names(
    model: torch.nn.Module, cfg: TrainConfig
) -> list[str]:
    """Fail closed on the legacy Prism Native1T stage freeze contracts."""
    if (
        not cfg.use_native1t_postfusion
        or cfg.model_type != "airminivla"
    ):
        return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    stage = int(cfg.native1t_stage)
    siglip_tail_names: set[str] = set()
    if getattr(cfg, "vision_train_siglip_last_n_layers", 0) > 0:
        if stage != 2:
            raise RuntimeError("SigLIP-tail Native1T tuning is Stage-2-only")
        fused_tower = getattr(model.backbone.vision_backbone, "fused_featurizer", None)
        blocks = getattr(fused_tower, "blocks", None)
        active_stop = 0 if blocks is None else len(blocks) - 1
        tail_count = int(cfg.vision_train_siglip_last_n_layers)
        if blocks is None or tail_count > active_stop:
            raise RuntimeError(
                "Fused SigLIP tower lacks the requested number of "
                "output-contributing tail blocks"
            )
        tail_parameter_ids = {
            id(parameter)
            for block in blocks[active_stop - tail_count:active_stop]
            for parameter in block.parameters()
        }
        siglip_tail_names = {
            name for name, parameter in model.named_parameters()
            if id(parameter) in tail_parameter_ids
        }
        if not siglip_tail_names:
            raise RuntimeError("Resolved SigLIP tail has no named parameters")
    if stage == 1:
        allowed = lambda name: (
            name.startswith("native1t_memory_tokenizer.")
            or name.startswith("native1t_memory_projector.")
            or name == "native1t_mem_bos"
        )
        required = {
            "native1t_memory_tokenizer": lambda name: name.startswith(
                "native1t_memory_tokenizer."
            ),
            "native1t_memory_projector": lambda name: name.startswith(
                "native1t_memory_projector."
            ),
            "native1t_mem_bos": lambda name: name == "native1t_mem_bos",
        }
    elif stage == 2:
        def allowed(name: str) -> bool:
            if name in {"native1t_mem_bos", "action_query"}:
                return True
            if name.startswith((
                "native1t_memory_projector.",
                "backbone.projector.",
                "fuse_proj.",
                "action_head.",
                "action_chunk_horizon_embeddings.",
                "action_chunk_norm.",
                "segment_task_state.",
            )):
                return True
            if "lora_" in name:
                return True
            if cfg.use_native1t_action_history and name.startswith((
                "native1t_action_embedding.",
                "native1t_action_history_projector.",
            )):
                return True
            if name in siglip_tail_names:
                return True
            return cfg.use_native1t_action_history and name == "native1t_action_bos"

        required = {
            "native1t_memory_projector": lambda name: name.startswith(
                "native1t_memory_projector."
            ),
            "native1t_mem_bos": lambda name: name == "native1t_mem_bos",
            "backbone_projector": lambda name: name.startswith("backbone.projector."),
            "fuse_proj": lambda name: name.startswith("fuse_proj."),
            "action_head": lambda name: name.startswith("action_head."),
            "action_query": lambda name: name == "action_query",
        }
        if cfg.task_state_mode != "none":
            required["segment_task_state"] = lambda name: name.startswith(
                "segment_task_state."
            )
        if cfg.action_chunk_horizon > 1:
            required["action_chunk_horizon_embeddings"] = lambda name: name.startswith(
                "action_chunk_horizon_embeddings."
            )
            required["action_chunk_norm"] = lambda name: name.startswith(
                "action_chunk_norm."
            )
        if cfg.use_lora:
            required["lora"] = lambda name: "lora_" in name
        if siglip_tail_names:
            required["siglip_tail"] = lambda name: name in siglip_tail_names
    else:
        raise ValueError(f"Unsupported Native1T stage {stage}")
    unexpected = [name for name in names if not allowed(name)]
    missing = [label for label, predicate in required.items() if not any(
        predicate(name) for name in names
    )]
    missing_siglip_tail = sorted(siglip_tail_names.difference(names))
    if unexpected or missing or missing_siglip_tail:
        raise RuntimeError(
            "Native1T trainable contract violation: "
            + json.dumps({
                "stage": stage,
                "unexpected": unexpected,
                "missing": missing,
                "missing_siglip_tail": missing_siglip_tail,
            })
        )
    return names


def contracts_compatible(saved: dict[str, Any], current: dict[str, Any]) -> bool:
    """Strict contract equality, except config fields added after the checkpoint.

    A checkpoint written before a TrainConfig field existed must stay resumable
    as long as the new field is still at its dataclass default — the training
    semantics it records are unchanged.  Any other difference is rejected.
    """
    if saved == current:
        return True
    saved_cfg = dict(saved.get("config", {}))
    current_cfg = dict(current.get("config", {}))
    # Extending a completed run by increasing only the epoch ceiling is a safe
    # resume: the optimizer/scheduler state and global step still come from the
    # checkpoint.  Reducing the ceiling remains incompatible.
    saved_epochs = saved_cfg.pop("epochs", None)
    current_epochs = current_cfg.pop("epochs", None)
    if (
        saved_epochs is not None
        and current_epochs is not None
        and current_epochs < saved_epochs
    ):
        return False
    defaults = {
        f.name: f.default for f in dataclass_fields(TrainConfig) if f.default is not MISSING
    }
    for key in set(current_cfg) - set(saved_cfg):
        if key in defaults and current_cfg[key] == defaults[key]:
            current_cfg.pop(key)
    saved_rest = {k: v for k, v in saved.items() if k != "config"}
    current_rest = {k: v for k, v in current.items() if k != "config"}
    return saved_cfg == current_cfg and saved_rest == current_rest


def load_class_weights(
    path: str | None,
    device: torch.device,
    num_actions: int,
    max_weight: float | None = None,
) -> torch.Tensor | None:
    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    values = raw.get("weights", raw) if isinstance(raw, dict) else raw
    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    if tensor.numel() != num_actions:
        raise ValueError(f"Expected {num_actions} class weights, got {tensor.numel()}")
    if max_weight is not None:
        if max_weight <= 0:
            raise ValueError("class_weight_max must be positive")
        # Raw inverse-frequency weights reach 14x for STOP; cap them so rare
        # classes are boosted without destabilising the common ones.
        tensor = tensor.clamp(max=max_weight)
    return tensor


def weighted_step_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    prev_actions: torch.Tensor | None,
    class_weights: torch.Tensor | None,
    inflection_coef: float,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross-entropy with optional per-class and inflection weighting.

    With ``inflection_coef == 1.0`` this is exactly ``F.cross_entropy`` with
    ``weight=class_weights`` — the historical behaviour.  Otherwise steps whose
    action differs from the previous action (AirVLN's "inflection" steps, with
    the START token making every episode head an inflection) are up-weighted by
    ``inflection_coef``, and the mean is taken over the combined weights as in
    the official AirVLN/VLN-CE trainers.
    """
    if inflection_coef == 1.0:
        return F.cross_entropy(logits, labels, weight=class_weights, reduction=reduction)
    if prev_actions is None:
        raise ValueError(
            "inflection_weight_coef != 1.0 requires batches that carry prev_actions; "
            "the AirMiniVLA multiframe pipeline provides them"
        )
    losses = F.cross_entropy(logits, labels, reduction="none")
    weights = torch.where(
        labels != prev_actions,
        losses.new_tensor(inflection_coef),
        losses.new_tensor(1.0),
    )
    if class_weights is not None:
        weights = weights * class_weights.to(losses.dtype)[labels]
    if reduction == "none":
        return losses * weights
    weighted = (losses * weights).sum()
    if reduction == "sum":
        return weighted
    return weighted / weights.sum().clamp_min(1e-12)


def action_chunk_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Equal-weight CE over valid AC slots; -100 masks tail/after-STOP slots."""
    if logits.ndim != 3 or targets.ndim != 2:
        raise ValueError("AC logits/targets must be [B,K,A] and [B,K]")
    if logits.shape[:2] != targets.shape:
        raise ValueError("AC logits and targets disagree on [B,K]")
    valid = targets != -100
    if not valid.any():
        raise ValueError("AC batch has no valid targets")
    losses = F.cross_entropy(logits.float()[valid], targets[valid], reduction="none")
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


def hierarchical_action_chunk_losses(
    move_logits: torch.Tensor,
    stop_logits: torch.Tensor,
    targets: torch.Tensor,
    positive_weight: float,
    *,
    reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent seven-way navigation CE and binary STOP BCE for AC slots."""
    if move_logits.ndim != 3 or targets.ndim != 2:
        raise ValueError("Hierarchical AC MOVE logits/targets must be [B,K,7] and [B,K]")
    if move_logits.shape[:2] != targets.shape or stop_logits.shape != targets.shape:
        raise ValueError("Hierarchical AC MOVE, STOP, and targets disagree on [B,K]")
    if positive_weight <= 0:
        raise ValueError("stop_aux_positive_weight must be positive")
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"Unsupported reduction: {reduction}")
    valid = targets != -100
    if not valid.any():
        raise ValueError("Hierarchical AC batch has no valid targets")
    moving = valid & (targets != 0)
    if moving.any():
        move_losses = F.cross_entropy(
            move_logits.float()[moving], targets[moving] - 1, reduction="none"
        )
        move_loss = move_losses.sum() if reduction == "sum" else move_losses.mean()
    else:
        move_loss = move_logits.float().sum() * 0.0
    stop_targets = (targets[valid] == 0).to(dtype=torch.float32)
    stop_losses = F.binary_cross_entropy_with_logits(
        stop_logits.float()[valid],
        stop_targets,
        pos_weight=stop_logits.new_tensor(float(positive_weight), dtype=torch.float32),
        reduction="none",
    )
    stop_loss = stop_losses.sum() if reduction == "sum" else stop_losses.mean()
    return move_loss, stop_loss


def stop_auxiliary_loss(
    stop_logits: torch.Tensor | None,
    labels: torch.Tensor,
    positive_weight: float,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary STOP-vs-non-STOP loss for the optional auxiliary head."""
    if stop_logits is None:
        return labels.new_zeros((), dtype=torch.float32)
    if positive_weight <= 0:
        raise ValueError("stop_aux_positive_weight must be positive")
    targets = (labels == 0).to(dtype=torch.float32)
    return F.binary_cross_entropy_with_logits(
        stop_logits.float(),
        targets,
        pos_weight=stop_logits.new_tensor(float(positive_weight), dtype=torch.float32),
        reduction=reduction,
    )


def transition_targets(
    labels: torch.Tensor,
    prev_actions: torch.Tensor | None,
    num_actions: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build GT stay/switch targets and exclude episode-start samples.

    ``prev_actions`` is supervision metadata already emitted by the dataset;
    it is not an input feature for S0.  Values outside ``[0, num_actions)``
    are START/padding sentinels and therefore have no transition target.
    """
    if prev_actions is None:
        raise ValueError("Transition supervision requires GT prev_actions metadata")
    valid = (prev_actions >= 0) & (prev_actions < num_actions)
    targets = (labels != prev_actions).to(dtype=torch.float32)
    return targets, valid


def transition_auxiliary_loss(
    transition_logits: torch.Tensor | None,
    labels: torch.Tensor,
    prev_actions: torch.Tensor | None,
    num_actions: int,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary stay/switch BCE over non-initial episode timesteps."""
    if transition_logits is None:
        raise ValueError("Enabled transition supervision produced no logits")
    targets, valid = transition_targets(labels, prev_actions, num_actions)
    if not valid.any():
        return transition_logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(
        transition_logits.float()[valid], targets[valid], reduction=reduction
    )


def movement_action_loss(
    move_logits: torch.Tensor | None,
    labels: torch.Tensor,
    prev_actions: torch.Tensor | None,
    class_weights: torch.Tensor | None,
    inflection_coef: float,
    reduction: str = "mean",
) -> torch.Tensor:
    """Seven-way movement CE, evaluated only where the GT action is not STOP."""
    if move_logits is None:
        raise ValueError("Hierarchical STOP training requires raw move_logits")
    moving = labels != 0
    if not moving.any():
        return move_logits.sum() * 0.0
    # AirVLN action 0 is STOP; movement IDs 1..7 map to classifier IDs 0..6.
    move_targets = labels[moving] - 1
    losses = F.cross_entropy(move_logits.float()[moving], move_targets, reduction="none")
    weights = torch.ones_like(losses)
    if inflection_coef != 1.0:
        if prev_actions is None:
            raise ValueError("Inflection weighting requires prev_actions")
        weights = torch.where(
            labels[moving] != prev_actions[moving],
            weights.new_tensor(inflection_coef),
            weights,
        )
    if class_weights is not None:
        weights = weights * class_weights.to(losses.dtype)[labels[moving]]
    weighted = losses * weights
    if reduction == "none":
        return weighted
    if reduction == "sum":
        return weighted.sum()
    return weighted.sum() / weights.sum().clamp_min(1e-12)


def progress_auxiliary_loss(
    progress: torch.Tensor | None,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None,
    beta: float,
    reduction: str = "mean",
) -> torch.Tensor:
    """Smooth-L1 progress loss over samples carrying an official TF label."""
    if progress is None:
        return targets.new_zeros((), dtype=torch.float32)
    losses = F.smooth_l1_loss(
        progress.float(), targets.float(), reduction="none", beta=float(beta)
    )
    if valid_mask is not None:
        losses = losses[valid_mask.bool()]
    if losses.numel() == 0:
        return progress.sum() * 0.0
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()


def move_batch(batch: dict[str, Any], device: torch.device, bf16: bool) -> dict[str, Any]:
    """Move all tensor values in a batch dict to the given device.

    ``pixel_values`` and ``features`` are cast to the appropriate float
    dtype.  Non-tensor values (lists, strings) are left as-is.
    """
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if key in (
                "pixel_values", "features", "omtrack_history_features",
                "native1t_current_premerger_features",
                "nativemem_premerger_features",
                "nativemem_history_latents",
            ) and bf16:
                value = value.to(device, dtype=torch.bfloat16, non_blocking=True)
            else:
                value = value.to(device, non_blocking=True)
        moved[key] = value
    return moved


def _uses_stop_aux_head(model: torch.nn.Module) -> bool:
    base = model.module if hasattr(model, "module") else model
    return bool(getattr(getattr(base, "config", None), "use_stop_aux_head", False))


def _uses_hierarchical_stop(model: torch.nn.Module) -> bool:
    base = model.module if hasattr(model, "module") else model
    return bool(getattr(getattr(base, "config", None), "use_hierarchical_stop", False))


def _uses_progress_aux_head(model: torch.nn.Module) -> bool:
    base = model.module if hasattr(model, "module") else model
    return bool(getattr(getattr(base, "config", None), "use_progress_aux_head", False))


def _uses_transition_aux_head(model: torch.nn.Module) -> bool:
    base = model.module if hasattr(model, "module") else model
    return bool(
        getattr(getattr(base, "config", None), "use_transition_aux_head", False)
    )


def _uses_segment_task_state(model: torch.nn.Module) -> bool:
    base = model.module if hasattr(model, "module") else model
    return getattr(getattr(base, "config", None), "task_state_mode", "none") != "none"


def _uses_temporal_memory(model: torch.nn.Module) -> bool:
    base = model.module if hasattr(model, "module") else model
    return bool(getattr(getattr(base, "config", None), "use_temporal_memory", False))


def _model_tensor_kwargs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    allowed = {
        "input_ids", "attention_mask", "pixel_values", "image_grid_thw",
        "mm_token_type_ids", "prev_actions",
        "instruction_token_mask", "frame_valid_mask", "frame_batch_indices",
        "frame_slot_indices", "precomputed_vision_features",
        "multimodal_left_padding", "segment_token_ids",
        "cached_memory_tokens", "history_valid_mask",
        "native1t_memory_pixel_values", "native1t_memory_features",
        "native1t_memory_grid_thw", "native1t_memory_valid_mask",
        "native1t_current_premerger_features",
        "nativemem_premerger_features", "nativemem_history_valid_mask",
        "nativemem_history_latents",
        "nativemem_history_tokens",
        "native1t_action_history", "native1t_action_history_valid_mask",
        "omtrack_history_features", "omtrack_history_valid_mask",
        "h16_frozen_pool_history_features", "h16_frozen_pool_history_valid_mask",
        "openfly_action_token_targets",
    }
    return {key: value for key, value in batch.items() if key in allowed}


def _forward_outputs(
    batch: dict[str, Any], model: torch.nn.Module
) -> dict[str, torch.Tensor | None]:
    tensor_batch = _model_tensor_kwargs(batch)
    base = model.module if hasattr(model, "module") else model
    if not bool(getattr(getattr(base, "config", None), "use_prev_action", False)):
        # S0-LORA-TS may use GT prev_actions to construct its auxiliary label,
        # but the previous action must never enter the policy forward pass.
        tensor_batch.pop("prev_actions", None)
    if _uses_openfly_action_tokens_ar(model):
        output = model(**tensor_batch)
        if not isinstance(output, dict):
            raise RuntimeError(
                "OpenFly action_tokens_ar_v2 must return teacher-forced LM details"
            )
        return output
    if (
        _uses_progress_aux_head(model)
        or _uses_hierarchical_stop(model)
        or _uses_transition_aux_head(model)
        or _uses_segment_task_state(model)
    ):
        output = model(**tensor_batch, return_auxiliary=True)
        if not isinstance(output, dict):
            raise RuntimeError("Auxiliary/hierarchical model must return an output dictionary")
        return output
    if _uses_stop_aux_head(model):
        output = model(**tensor_batch, return_stop_aux=True)
        if not isinstance(output, tuple) or len(output) not in (2, 3):
            raise RuntimeError(
                "STOP auxiliary model must return (action_logits, stop_logits) "
                "or (action_logits, memory_state, stop_logits)"
            )
        return {
            "action_logits": output[0],
            "stop_logits": output[-1],
            "progress": None,
            "transition_logits": None,
        }
    output = model(**tensor_batch)
    return {
        "action_logits": output[0] if isinstance(output, tuple) else output,
        "stop_logits": None,
        "progress": None,
        "transition_logits": None,
    }


def _forward_kwargs_with_stop_aux(
    batch: dict[str, Any],
    model: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return action logits and optional binary STOP logits."""
    output = _forward_outputs(batch, model)
    return output["action_logits"], output["stop_logits"]


def _forward_kwargs(batch: dict[str, Any], model: torch.nn.Module) -> torch.Tensor:
    """Call ``model.forward(**batch)``, dropping non-tensor keys.

    The model's ``forward`` is expected to accept only the kwargs it needs.
    """
    logits, _ = _forward_kwargs_with_stop_aux(batch, model)
    return logits


def _uses_openfly_action_tokens(model: torch.nn.Module) -> bool:
    """Whether a model uses either supported OpenFly 8-D token route."""
    base = model.module if hasattr(model, "module") else model
    return getattr(
        getattr(base, "config", None), "openfly_action_head_type", "categorical"
    ) in {"action_tokens_v1", "action_tokens_ar_v2"}


def _uses_openfly_action_tokens_ar(model: torch.nn.Module) -> bool:
    base = model.module if hasattr(model, "module") else model
    return (
        getattr(getattr(base, "config", None), "openfly_action_head_type", "categorical")
        == "action_tokens_ar_v2"
    )


def _openfly_action_vocab_size(model: torch.nn.Module) -> int:
    base = model.module if hasattr(model, "module") else model
    value = getattr(base, "_openfly_action_vocab_size", None)
    if not isinstance(value, int) or value <= 256:
        raise RuntimeError("OpenFly autoregressive model did not expose a valid action vocabulary size")
    return value


def openfly_action_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """OpenFly's eight-coordinate 256-bin token cross entropy."""
    if logits.ndim != 3 or tuple(logits.shape[-2:]) != (8, 256):
        raise ValueError(
            "OpenFly action-token logits must have shape [B,8,256], got "
            f"{tuple(logits.shape)}"
        )
    if tuple(targets.shape) != tuple(logits.shape[:2]):
        raise ValueError(
            "OpenFly action-token targets must have shape [B,8], got "
            f"{tuple(targets.shape)} for logits {tuple(logits.shape)}"
        )
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.long().reshape(-1),
        reduction=reduction,
    )


def _openfly_token_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    codec: Any,
    *,
    predicted_token_ids: torch.Tensor | None = None,
    action_vocab_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return official token metrics plus explicitly diagnostic codebook data.

    OpenFly's trainer computes full-vocabulary teacher-forced top-1 accuracy
    only over the eight action-tail tokens and reports an L1 loss between
    their *normalized* ActionTokenizer bin centers.  It does not evaluate
    rounded physical action vectors or 10-way codebook agreement here.
    """
    if predicted_token_ids is None:
        predictions, _physical_vectors, matched = codec.decode_logits_tensor(logits)
        predicted_classes = logits.argmax(dim=-1)
        token_correct = (predicted_classes == targets).sum()
    else:
        if action_vocab_size is None:
            raise ValueError("OpenFly full-vocabulary token metrics require action_vocab_size")
        predictions, _physical_vectors, matched = codec.decode_token_ids_tensor(
            predicted_token_ids, action_vocab_size=action_vocab_size
        )
        target_token_ids = int(action_vocab_size) - targets.long() - 1
        token_correct = (predicted_token_ids.long() == target_token_ids).sum()
        predicted_classes = int(action_vocab_size) - predicted_token_ids.long() - 1
    predicted_vectors = codec.decode_token_classes_normalized_tensor(predicted_classes)
    target_vectors = codec.decode_token_classes_normalized_tensor(targets)
    token_examples = torch.tensor(
        targets.numel(), dtype=torch.float64, device=logits.device
    )
    vector_abs_sum = (predicted_vectors - target_vectors).abs().sum()
    unmatched = (~matched).sum()
    return predictions, token_correct, token_examples, vector_abs_sum, unmatched


def _sequence_forward(
    batch: dict[str, Any],
    model: torch.nn.Module,
    class_weights: torch.Tensor | None,
    inflection_coef: float = 1.0,
    stop_aux_loss_weight: float = 0.0,
    stop_aux_positive_weight: float = 1.0,
    task_gru_state_prev: torch.Tensor | None = None,
    episode_stream: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Unroll an episode-contiguous window in chronological order."""
    memory_state = None
    segment_belief = None
    task_gru_state = task_gru_state_prev
    base = model.module if hasattr(model, "module") else model
    use_task_gru = bool(getattr(base.config, "task_state_use_gru", False))
    task_gru_burn_in = int(getattr(base.config, "task_gru_burn_in", 0))
    task_gru_loss_window = int(getattr(base.config, "task_gru_loss_window", 0))
    if use_task_gru and batch["action_id"].shape[1] != (
        task_gru_burn_in + task_gru_loss_window
    ):
        raise ValueError(
            "TaskGRU sequence length must equal burn-in + loss window"
        )
    total_loss = torch.zeros((), device=batch["action_id"].device)
    total_weight = torch.zeros((), device=batch["action_id"].device)
    total_examples = 0
    total_stop_aux_loss = torch.zeros((), device=batch["action_id"].device)
    predictions: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    loss_anchor = torch.zeros((), device=batch["action_id"].device)
    for time_index in range(batch["action_id"].shape[1]):
        sequence_valid = batch["sequence_mask"][:, time_index]
        if episode_stream:
            if batch["stream_reset"].ndim != 2:
                raise ValueError("TaskGRU episode stream requires [B,T] reset masks")
            reset = batch["stream_reset"][:, time_index]
            if task_gru_state is not None:
                task_gru_state = torch.where(
                    reset[:, None], torch.zeros_like(task_gru_state), task_gru_state
                )
        previous_task_gru_state = task_gru_state
        previous_segment_belief = segment_belief
        model_kwargs = dict(
            input_ids=batch["input_ids"][:, time_index],
            attention_mask=batch["attention_mask"][:, time_index],
            pixel_values=batch["pixel_values"][:, time_index],
            prev_actions=batch["prev_actions"][:, time_index],
            memory_state=memory_state,
        )
        if "image_grid_thw" in batch:
            model_kwargs["image_grid_thw"] = batch["image_grid_thw"][:, time_index]
        if "mm_token_type_ids" in batch:
            model_kwargs["mm_token_type_ids"] = batch["mm_token_type_ids"][:, time_index]
        if "frame_valid_mask" in batch:
            model_kwargs["frame_valid_mask"] = batch["frame_valid_mask"][:, time_index]
        if _uses_segment_task_state(model):
            model_kwargs["instruction_token_mask"] = batch[
                "instruction_token_mask"
            ][:, time_index]
            model_kwargs["segment_token_ids"] = batch[
                "segment_token_ids"
            ][:, time_index]
            model_kwargs["segment_belief_prev"] = segment_belief
            model_kwargs["task_gru_state_prev"] = task_gru_state
            output = model(**model_kwargs, return_auxiliary=True)
            if not isinstance(output, dict):
                raise RuntimeError("Sequence SegTS model must return auxiliary outputs")
            logits = output["action_logits"]
            loss_anchor = loss_anchor + logits.sum() * 0.0
            segment_belief = output.get("segment_belief")
            if segment_belief is None:
                raise RuntimeError("Sequence SegTS output is missing segment belief")
            task_gru_state = output.get("task_gru_state")
            if use_task_gru and task_gru_state is None:
                raise RuntimeError("Sequence TaskGRU output is missing recurrent state")
            if task_gru_state is not None:
                if previous_task_gru_state is None:
                    previous_task_gru_state = torch.zeros_like(task_gru_state)
                task_gru_state = torch.where(
                    sequence_valid[:, None], task_gru_state, previous_task_gru_state
                )
            if segment_belief is not None and previous_segment_belief is not None:
                segment_belief = torch.where(
                    sequence_valid[:, None], segment_belief, previous_segment_belief
                )
            if bool(base.config.task_state_detach_previous_belief):
                segment_belief = segment_belief.detach()
            stop_logits = output.get("stop_logits")
        elif _uses_stop_aux_head(model):
            output = model(**model_kwargs, return_stop_aux=True)
            if not isinstance(output, tuple) or len(output) != 3:
                raise RuntimeError(
                    "Sequence STOP auxiliary model must return "
                    "(logits, memory_state, stop_logits)"
                )
            logits, memory_state, stop_logits = output
        else:
            output = model(**model_kwargs)
            stop_logits = None
        if not _uses_segment_task_state(model) and not isinstance(output, tuple):
            raise RuntimeError("Sequence training requires a model that returns (logits, memory_state)")
        if not _uses_segment_task_state(model) and not _uses_stop_aux_head(model):
            logits, memory_state = output
        valid = (
            batch["loss_mask"][:, time_index]
            if use_task_gru
            else sequence_valid
        )
        if use_task_gru and time_index < task_gru_burn_in:
            # Burn-in only establishes the recurrent state.  Its graph remains
            # connected so loss timesteps backpropagate through all 12 steps.
            continue
        step_labels = batch["action_id"][:, time_index]
        step_loss = F.cross_entropy(logits.float(), step_labels, reduction="none")
        # Mirror weighted_step_loss: normalising by the weight sum (not the
        # step count) keeps the loss scale identical to the per-step path.
        step_weight = torch.ones_like(step_loss)
        if inflection_coef != 1.0:
            step_prev = batch["prev_actions"][:, time_index]
            step_weight = torch.where(
                step_labels != step_prev,
                step_weight.new_tensor(inflection_coef),
                step_weight.new_tensor(1.0),
            )
        if class_weights is not None:
            step_weight = step_weight * class_weights.to(step_loss.dtype)[step_labels]
        total_loss = total_loss + (step_loss * step_weight)[valid].sum()
        if stop_logits is not None:
            aux_losses = stop_auxiliary_loss(
                stop_logits,
                step_labels,
                stop_aux_positive_weight,
                reduction="none",
            )
            total_stop_aux_loss = total_stop_aux_loss + aux_losses[valid].sum()
        total_weight = total_weight + step_weight[valid].sum()
        total_examples += int(valid.sum().item())
        predictions.append(logits.argmax(dim=-1)[valid])
        labels.append(step_labels[valid])
        if episode_stream and task_gru_state is not None:
            finished = batch["stream_end"][:, time_index]
            task_gru_state = torch.where(
                finished[:, None], torch.zeros_like(task_gru_state), task_gru_state
            )
    if total_examples == 0:
        empty = batch["action_id"].new_empty((0,))
        return loss_anchor, empty, empty, (
            task_gru_state.detach() if task_gru_state is not None else None
        )
    # The window ends here, so the next window begins with detached/reset memory.
    mean_loss = total_loss / total_weight.clamp_min(1e-12)
    if stop_aux_loss_weight:
        mean_loss = mean_loss + float(stop_aux_loss_weight) * (
            total_stop_aux_loss / max(total_examples, 1)
        )
    next_task_gru_state = (
        task_gru_state.detach()
        if episode_stream and task_gru_state is not None
        else None
    )
    return mean_loss, torch.cat(predictions), torch.cat(labels), next_task_gru_state


def _temporal_stream_forward(
    batch: dict[str, Any],
    model: torch.nn.Module,
    stream_state: dict[str, tuple[torch.Tensor, torch.Tensor]],
    class_weights: torch.Tensor | None,
    inflection_coef: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorize a TM-VLA chunk and carry detached memory across chunks.

    M_t depends only on the frozen feature F_t.  We therefore encode and
    tokenize every frame in a [B,T] window together, build the causal queues
    with tensor operations, and run one B*T VLA forward.  This preserves the
    exact online ordering while avoiding T small serial model calls.
    """
    if "action_chunk_targets" in batch:
        raise RuntimeError("Native TM-VLA batches must not contain AC action-chunk targets")
    base = model.module if hasattr(model, "module") else model
    config = base.config
    batch_size, sequence_length = batch["action_id"].shape
    feature_dim = int(base._vision_feature_size)
    history_length = int(config.temporal_memory_num_history)
    device = batch["pixel_values"].device
    memory_dtype = next(base.temporal_memory_tokenizer.parameters()).dtype

    memory = torch.zeros(
        batch_size, history_length, feature_dim, device=device, dtype=memory_dtype
    )
    memory_mask = torch.zeros(
        batch_size, history_length, device=device, dtype=torch.bool
    )
    reset_flags = batch["stream_reset"].bool()
    for row, episode_id in enumerate(batch["episode_id"]):
        if reset_flags[row]:
            stream_state.pop(episode_id, None)
        previous = stream_state.get(episode_id)
        if previous is not None:
            memory[row] = previous[0]
            memory_mask[row] = previous[1]

    pixel_values = batch["pixel_values"]
    if pixel_values.ndim == 6:
        if pixel_values.shape[2] != 1:
            raise ValueError("Native TM-VLA windows require one RGB per timestep")
        pixel_values = pixel_values[:, :, 0]
    if pixel_values.ndim != 5:
        raise ValueError("Native TM-VLA pixel_values must have shape [B,T,C,H,W]")

    flat_pixels = pixel_values.reshape(
        batch_size * sequence_length, *pixel_values.shape[2:]
    )
    flat_features = base.encode_frozen_vision(flat_pixels)
    if flat_features.ndim != 3 or flat_features.shape[-1] != feature_dim:
        raise ValueError("Frozen vision feature shape does not match TM-VLA")
    patch_count = flat_features.shape[1]
    current_tokens = base.temporal_memory_tokenizer(
        flat_features.reshape(
            batch_size, sequence_length, patch_count, feature_dim
        )
    )

    # Snapshot the queue before appending M_t.  Thus action_t sees only
    # [M_(t-H), ..., M_(t-1)], exactly matching online inference.
    history_by_time: list[torch.Tensor] = []
    history_mask_by_time: list[torch.Tensor] = []
    sequence_valid = batch["sequence_mask"].bool()
    for time_index in range(sequence_length):
        history_by_time.append(memory)
        history_mask_by_time.append(memory_mask)
        valid = sequence_valid[:, time_index]
        shifted_memory = torch.cat(
            [memory[:, 1:], current_tokens[:, time_index : time_index + 1]], dim=1
        )
        shifted_mask = torch.cat(
            [
                memory_mask[:, 1:],
                torch.ones(batch_size, 1, device=device, dtype=torch.bool),
            ],
            dim=1,
        )
        memory = torch.where(valid[:, None, None], shifted_memory, memory)
        memory_mask = torch.where(valid[:, None], shifted_mask, memory_mask)

    flat_history = torch.stack(history_by_time, dim=1).reshape(
        batch_size * sequence_length, history_length, feature_dim
    )
    flat_history_mask = torch.stack(history_mask_by_time, dim=1).reshape(
        batch_size * sequence_length, history_length
    )
    flat_current_tokens = current_tokens.reshape(
        batch_size * sequence_length, feature_dim
    )
    output = model(
        input_ids=batch["input_ids"].reshape(
            batch_size * sequence_length, *batch["input_ids"].shape[2:]
        ),
        attention_mask=batch["attention_mask"].reshape(
            batch_size * sequence_length, *batch["attention_mask"].shape[2:]
        ),
        pixel_values=None,
        precomputed_vision_features=flat_features,
        precomputed_current_memory_token=flat_current_tokens,
        cached_memory_tokens=flat_history,
        history_valid_mask=flat_history_mask,
        instruction_token_mask=(
            batch["instruction_token_mask"].reshape(
                batch_size * sequence_length,
                *batch["instruction_token_mask"].shape[2:],
            )
            if "instruction_token_mask" in batch else None
        ),
        segment_token_ids=(
            batch["segment_token_ids"].reshape(
                batch_size * sequence_length,
                *batch["segment_token_ids"].shape[2:],
            )
            if "segment_token_ids" in batch else None
        ),
        return_auxiliary=_uses_segment_task_state(model),
    )
    if isinstance(output, dict):
        flat_logits = output["action_logits"]
        returned_current_tokens = output.get("current_memory_token")
        if returned_current_tokens is None:
            raise RuntimeError("TM-VLA auxiliary output is missing current_memory_token")
    elif isinstance(output, tuple) and len(output) == 2:
        flat_logits, returned_current_tokens = output
    else:
        raise RuntimeError("Native TM-VLA must return logits and current_memory_token")
    flat_labels = batch["action_id"].reshape(-1)
    flat_prev_actions = batch["prev_actions"].reshape(-1)
    flat_valid = sequence_valid.reshape(-1)
    per_step_loss = F.cross_entropy(flat_logits.float(), flat_labels, reduction="none")
    step_weight = torch.ones_like(per_step_loss)
    if inflection_coef != 1.0:
        step_weight = torch.where(
            flat_labels != flat_prev_actions,
            step_weight.new_tensor(inflection_coef),
            step_weight.new_tensor(1.0),
        )
    if class_weights is not None:
        step_weight = step_weight * class_weights.to(per_step_loss.dtype)[flat_labels]
    total_weight = step_weight[flat_valid].sum()
    total_loss = (per_step_loss * step_weight)[flat_valid].sum()
    # Keep every TM component in DDP's graph on rank-balancing padding batches
    # and for the final token, which no later action consumes.
    total_loss = total_loss + returned_current_tokens.sum() * 0.0

    end_flags = batch["stream_end"].bool()
    for row, episode_id in enumerate(batch["episode_id"]):
        if end_flags[row]:
            stream_state.pop(episode_id, None)
        else:
            stream_state[episode_id] = (
                memory[row].detach(), memory_mask[row].detach()
            )
    return (
        total_loss / total_weight.clamp_min(1e-12),
        flat_logits.argmax(dim=-1)[flat_valid],
        flat_labels[flat_valid],
    )


def cosine_with_warmup(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    bf16: bool,
    num_actions: int,
    class_weights: torch.Tensor | None = None,
    inflection_coef: float = 1.0,
    stop_aux_loss_weight: float = 0.0,
    stop_aux_positive_weight: float = 1.0,
    progress_loss_weight: float = 0.0,
    progress_smooth_l1_beta: float = 0.1,
    transition_loss_weight: float = 0.0,
    openfly_action_codec: Any | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0.0, device=device)
    total_examples = torch.tensor(0.0, device=device)
    total_loss_examples = torch.tensor(0.0, device=device)
    progress_abs_error = torch.tensor(0.0, device=device)
    progress_pred_sum = torch.tensor(0.0, device=device)
    progress_target_sum = torch.tensor(0.0, device=device)
    progress_examples = torch.tensor(0.0, device=device)
    transition_loss_sum = torch.tensor(0.0, device=device)
    transition_correct = torch.tensor(0.0, device=device)
    transition_examples = torch.tensor(0.0, device=device)
    segment_entropy_sum = torch.tensor(0.0, device=device)
    segment_max_sum = torch.tensor(0.0, device=device)
    segment_examples = torch.tensor(0.0, device=device)
    confusion = torch.zeros((num_actions, num_actions), dtype=torch.float64, device=device)
    base_model = model.module if hasattr(model, "module") else model
    chunk_horizon = int(
        getattr(getattr(base_model, "config", None), "action_chunk_horizon", 1)
    )
    chunk_slot_correct = torch.zeros(
        chunk_horizon, dtype=torch.float64, device=device
    )
    chunk_slot_examples = torch.zeros(
        chunk_horizon, dtype=torch.float64, device=device
    )
    chunk_exact_correct = torch.tensor(0.0, device=device)
    chunk_exact_examples = torch.tensor(0.0, device=device)
    future_transition_correct = torch.tensor(0.0, device=device)
    future_transition_examples = torch.tensor(0.0, device=device)
    openfly_token_correct = torch.tensor(0.0, device=device)
    openfly_token_examples = torch.tensor(0.0, device=device)
    openfly_vector_abs_sum = torch.tensor(0.0, dtype=torch.float64, device=device)
    openfly_unmatched = torch.tensor(0.0, device=device)
    openfly_codebook_correct = torch.tensor(0.0, device=device)
    openfly_codebook_examples = torch.tensor(0.0, device=device)
    temporal_stream_state: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    use_openfly_action_tokens = _uses_openfly_action_tokens(model)
    if use_openfly_action_tokens and openfly_action_codec is None:
        raise ValueError("OpenFly action-token validation requires its immutable codec")

    for batch in loader:
        batch = move_batch(batch, device, bf16)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            if "sequence_mask" in batch:
                if _uses_temporal_memory(model):
                    mean_loss, predictions, labels = _temporal_stream_forward(
                        batch,
                        model,
                        temporal_stream_state,
                        class_weights,
                        inflection_coef,
                    )
                else:
                    mean_loss, predictions, labels, _ = _sequence_forward(
                        batch,
                        model,
                        class_weights,
                        inflection_coef,
                        stop_aux_loss_weight,
                        stop_aux_positive_weight,
                    )
                loss = mean_loss * labels.numel()
                loss_examples = labels.numel()
            else:
                outputs = _forward_outputs(batch, model)
                logits = outputs["action_logits"]
                stop_logits = outputs["stop_logits"]
                labels = batch["action_id"]
                if use_openfly_action_tokens:
                    targets = batch.get("openfly_action_token_targets")
                    if targets is None:
                        raise RuntimeError("OpenFly action-token batch lacks token targets")
                    if _uses_openfly_action_tokens_ar(model):
                        loss = outputs.get("openfly_lm_loss_sum")
                        loss_tokens = outputs.get("openfly_lm_loss_tokens")
                        predicted_token_ids = outputs.get("openfly_predicted_token_ids")
                        if loss is None or loss_tokens is None or predicted_token_ids is None:
                            raise RuntimeError(
                                "OpenFly autoregressive validation output lacks LM loss or token IDs"
                            )
                        loss_examples = int(loss_tokens.item())
                        metric_kwargs = {
                            "predicted_token_ids": predicted_token_ids,
                            "action_vocab_size": _openfly_action_vocab_size(model),
                        }
                    else:
                        loss = openfly_action_token_loss(logits, targets, reduction="sum")
                        loss_examples = int(targets.numel())
                        metric_kwargs = {}
                    (
                        predictions,
                        token_correct,
                        token_examples,
                        vector_abs_sum,
                        unmatched,
                    ) = _openfly_token_metrics(
                        logits, targets, openfly_action_codec, **metric_kwargs
                    )
                    openfly_token_correct += token_correct
                    openfly_token_examples += token_examples
                    openfly_vector_abs_sum += vector_abs_sum
                    openfly_unmatched += unmatched
                    openfly_codebook_correct += (predictions == labels).sum()
                    openfly_codebook_examples += labels.numel()
                elif logits.ndim == 3:
                    targets = batch["action_chunk_targets"]
                    valid_chunk = targets != -100
                    if _uses_hierarchical_stop(model):
                        move_loss, stop_loss = hierarchical_action_chunk_losses(
                            outputs["move_logits"],
                            stop_logits,
                            targets,
                            stop_aux_positive_weight,
                            reduction="sum",
                        )
                        loss = move_loss + float(stop_aux_loss_weight) * stop_loss
                    else:
                        loss = action_chunk_loss(logits, targets, reduction="sum")
                    loss_examples = int(valid_chunk.sum().item())
                    chunk_predictions = logits.argmax(dim=-1)
                    for slot in range(logits.shape[1]):
                        slot_valid = valid_chunk[:, slot]
                        chunk_slot_correct[slot] += (
                            chunk_predictions[:, slot][slot_valid]
                            == targets[:, slot][slot_valid]
                        ).sum()
                        chunk_slot_examples[slot] += slot_valid.sum()
                    full_valid = valid_chunk.all(dim=1)
                    chunk_exact_correct += (
                        (chunk_predictions == targets).all(dim=1) & full_valid
                    ).sum()
                    chunk_exact_examples += full_valid.sum()
                    for slot in range(1, logits.shape[1]):
                        transition_valid = (
                            valid_chunk[:, slot]
                            & valid_chunk[:, slot - 1]
                            & (targets[:, slot] != targets[:, slot - 1])
                        )
                        future_transition_correct += (
                            chunk_predictions[:, slot][transition_valid]
                            == targets[:, slot][transition_valid]
                        ).sum()
                        future_transition_examples += transition_valid.sum()
                    predictions = chunk_predictions[:, 0]
                else:
                    loss_examples = labels.numel()
                if outputs.get("segment_alpha") is not None:
                    segment_alpha = outputs["segment_alpha"].float()
                    segment_entropy = outputs["segment_entropy"].float()
                    segment_entropy_sum += segment_entropy.sum()
                    segment_max_sum += segment_alpha.max(dim=-1).values.sum()
                    segment_examples += segment_alpha.shape[0]
                if use_openfly_action_tokens:
                    pass
                elif logits.ndim == 3:
                    pass
                elif _uses_hierarchical_stop(model):
                    loss = movement_action_loss(
                        outputs.get("move_logits"), labels, batch.get("prev_actions"),
                        class_weights, inflection_coef, reduction="sum",
                    )
                else:
                    loss = weighted_step_loss(
                        logits.float(),
                        labels,
                        batch.get("prev_actions"),
                        class_weights,
                        inflection_coef,
                        reduction="sum",
                    )
                if (not use_openfly_action_tokens) and stop_aux_loss_weight and not (
                    logits.ndim == 3 and _uses_hierarchical_stop(model)
                ):
                    loss = loss + float(stop_aux_loss_weight) * stop_auxiliary_loss(
                        stop_logits,
                        labels,
                        stop_aux_positive_weight,
                        reduction="sum",
                    )
                if (not use_openfly_action_tokens) and progress_loss_weight:
                    progress = outputs["progress"]
                    targets = batch["progress_targets"]
                    valid = batch.get("progress_valid_mask")
                    loss = loss + float(progress_loss_weight) * progress_auxiliary_loss(
                        progress, targets, valid, progress_smooth_l1_beta, reduction="sum"
                    )
                    selected = valid.bool() if valid is not None else torch.ones_like(targets, dtype=torch.bool)
                    progress_abs_error += (progress.float()[selected] - targets.float()[selected]).abs().sum()
                    progress_pred_sum += progress.float()[selected].sum()
                    progress_target_sum += targets.float()[selected].sum()
                    progress_examples += selected.sum()
                if (not use_openfly_action_tokens) and transition_loss_weight:
                    transition_logits = outputs["transition_logits"]
                    transition_targets_batch, transition_valid = transition_targets(
                        labels, batch.get("prev_actions"), num_actions
                    )
                    transition_loss_sum += transition_auxiliary_loss(
                        transition_logits,
                        labels,
                        batch.get("prev_actions"),
                        num_actions,
                        reduction="sum",
                    )
                    transition_predictions = transition_logits >= 0
                    transition_correct += (
                        transition_predictions[transition_valid]
                        == transition_targets_batch[transition_valid].bool()
                    ).sum()
                    transition_examples += transition_valid.sum()
                if (not use_openfly_action_tokens) and logits.ndim != 3:
                    predictions = logits.argmax(dim=-1)
        total_loss += loss
        total_loss_examples += loss_examples
        if use_openfly_action_tokens:
            # Match OpenFly strategy.py: aggregate only teacher-forced action
            # tail-token top-1 results (EOS is not part of this denominator).
            total_correct += token_correct
            total_examples += token_examples
        else:
            total_correct += (predictions == labels).sum()
            total_examples += labels.numel()
        indices = labels * num_actions + predictions
        confusion += torch.bincount(indices, minlength=num_actions * num_actions).reshape(num_actions, num_actions)

    for tensor in (
        total_loss, total_correct, total_examples, total_loss_examples, confusion,
        progress_abs_error,
        progress_pred_sum, progress_target_sum, progress_examples,
        transition_loss_sum, transition_correct, transition_examples,
        segment_entropy_sum, segment_max_sum, segment_examples,
        chunk_slot_correct, chunk_slot_examples, chunk_exact_correct,
        chunk_exact_examples, future_transition_correct, future_transition_examples,
        openfly_token_correct, openfly_token_examples, openfly_vector_abs_sum,
        openfly_unmatched, openfly_codebook_correct, openfly_codebook_examples,
    ):
        reduce_sum(tensor)
    model.train()
    count = max(float(total_examples.item()), 1.0)
    metrics = {
        "loss": float(
            total_loss.item() / max(float(total_loss_examples.item()), 1.0)
            + float(transition_loss_weight)
            * transition_loss_sum.item()
            / max(float(transition_examples.item()), 1.0)
        ),
        "accuracy": float(total_correct.item() / count),
        "examples": float(total_examples.item()),
    }
    if use_openfly_action_tokens:
        metrics["codebook_confusion"] = confusion.cpu().tolist()
    else:
        metrics["confusion"] = confusion.cpu().tolist()
    if progress_examples.item() > 0:
        progress_count = float(progress_examples.item())
        metrics.update({
            "progress_mae": float(progress_abs_error.item() / progress_count),
            "progress_pred_mean": float(progress_pred_sum.item() / progress_count),
            "progress_target_mean": float(progress_target_sum.item() / progress_count),
        })
    if transition_examples.item() > 0:
        transition_count = float(transition_examples.item())
        metrics.update({
            "transition_loss": float(transition_loss_sum.item() / transition_count),
            "transition_accuracy": float(transition_correct.item() / transition_count),
            "transition_examples": transition_count,
        })
    if segment_examples.item() > 0:
        segment_count = float(segment_examples.item())
        metrics.update({
            "task_state_mean_entropy": float(segment_entropy_sum.item() / segment_count),
            "task_state_mean_max_probability": float(segment_max_sum.item() / segment_count),
        })
    if chunk_slot_examples.sum().item() > 0:
        for slot in range(chunk_horizon):
            metrics[f"chunk_accuracy_{slot}"] = float(
                chunk_slot_correct[slot].item()
                / max(chunk_slot_examples[slot].item(), 1.0)
            )
            metrics[f"chunk_examples_{slot}"] = float(chunk_slot_examples[slot].item())
        metrics["chunk_exact_match"] = float(
            chunk_exact_correct.item() / max(chunk_exact_examples.item(), 1.0)
        )
        metrics["chunk_exact_examples"] = float(chunk_exact_examples.item())
        metrics["future_transition_accuracy"] = float(
            future_transition_correct.item()
            / max(future_transition_examples.item(), 1.0)
        )
        metrics["future_transition_examples"] = float(future_transition_examples.item())
    if use_openfly_action_tokens:
        token_count = max(float(openfly_token_examples.item()), 1.0)
        action_count = max(float(openfly_codebook_examples.item()), 1.0)
        action_accuracy = float(openfly_token_correct.item() / token_count)
        l1_loss = float(openfly_vector_abs_sum.item() / token_count)
        metrics.update({
            "openfly_action_head_type": getattr(
                getattr(model.module if hasattr(model, "module") else model, "config", None),
                "openfly_action_head_type",
                "unknown",
            ),
            "openfly_metric_protocol": "official_teacher_forced_action_tokens",
            # ``action_accuracy`` and ``l1_loss`` use the same definitions
            # and names committed by OpenFly's training strategy.
            "accuracy": action_accuracy,
            "action_accuracy": action_accuracy,
            "l1_loss": l1_loss,
            # Keep the generic fields truthful for generic trainer consumers.
            "action_token_accuracy": action_accuracy,
            "action_l1_loss": l1_loss,
            "action_token_examples": float(openfly_token_examples.item()),
            "action_examples": float(openfly_codebook_examples.item()),
            # These require physical-vector unnormalization/rounding and are
            # useful diagnostics only; they are not OpenFly's accuracy metric.
            "codebook_exact_agreement": float(
                openfly_codebook_correct.item() / action_count
            ),
            "decoded_unmatched_rate": float(openfly_unmatched.item() / action_count),
        })
    return metrics


def config_to_dict(cfg: TrainConfig) -> dict[str, Any]:
    """Safely convert a slots-based or dict-based dataclass to a plain dict."""
    if hasattr(cfg, "__dict__"):
        return {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")}
    result = {}
    for cls in type(cfg).__mro__:
        for slot in getattr(cls, "__slots__", ()):
            result[slot] = getattr(cfg, slot)
    return result


def rotate_checkpoints(output_dir: Path, keep: int) -> None:
    """Remove old checkpoints, keeping only the most recent ``keep``."""
    if keep <= 0:
        return
    checkpoint_dirs = sorted(
        [
            path for path in output_dir.iterdir()
            if path.is_dir() and (path.name.startswith("step-") or path.name.startswith("epoch-"))
        ],
        key=lambda path: path.stat().st_mtime,
    )
    while len(checkpoint_dirs) > keep:
        shutil.rmtree(checkpoint_dirs.pop(0))


def should_save_step_checkpoint(
    global_step: int,
    save_every_steps: int,
    validation_checkpoint_saved: bool,
    save_at_steps: list[int] | None = None,
) -> bool:
    """Return whether a non-validation periodic checkpoint is due."""
    return (
        global_step > 0
        and (
            (save_every_steps > 0 and global_step % save_every_steps == 0)
            or global_step in set(save_at_steps or [])
        )
        and not validation_checkpoint_saved
    )


def find_latest_checkpoint(output_dir: str | Path) -> tuple[Path | None, dict[str, Any] | None]:
    """Locate the most recent checkpoint for training resume.

    Prefers the ``latest`` symlink; a dangling link falls back to scanning
    both ``step-*`` and ``epoch-*`` directories, picking the highest recorded
    ``global_step`` (mtime as tiebreaker).
    """
    output = Path(output_dir)
    candidates: list[Path] = []
    link = output / "latest"
    if link.is_symlink() or link.is_dir():
        resolved = link.resolve()
        if resolved.is_dir():
            candidates = [resolved]
        else:
            print(f"[resume] warning: {link} is dangling; scanning checkpoint directories instead")
    scanning = not candidates
    if scanning:
        candidates = [
            path
            for pattern in ("step-*", "epoch-*")
            for path in output.glob(pattern)
            if path.is_dir()
        ]

    best: tuple[int, float, Path, dict[str, Any] | None] | None = None
    for path in candidates:
        state_path = path / "training_state.pt"
        extra = None
        if state_path.is_file():
            try:
                extra = torch.load(state_path, map_location="cpu", weights_only=False)
            except Exception as exc:  # truncated by a crash or full disk
                print(f"[resume] warning: unreadable training_state.pt in {path}: {exc}")
                extra = None
        if scanning and extra is None:
            # A directory without a readable training_state.pt is a half-written
            # checkpoint (the state file is saved last); resuming from it would
            # silently restart at global_step 0 with a stale optimizer.
            print(f"[resume] warning: skipping incomplete checkpoint {path}")
            continue
        step = int(extra.get("global_step", -1)) if extra else -1
        key = (step, path.stat().st_mtime, path, extra)
        if best is None or key[:2] > best[:2]:
            best = key
    if best is None:
        return None, None
    return best[2], best[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    cfg = load_yaml_dataclass(cli.config, TrainConfig)
    openfly_action_codec: Any | None = None
    if cfg.openfly_action_head_type in {"action_tokens_v1", "action_tokens_ar_v2"}:
        from aerial_openvla.openfly.action_codec import (
            OPENFLY_ACTION_BINS as OPENFLY_ACTION_TOKEN_BINS,
            OPENFLY_ACTION_DIM as OPENFLY_ACTION_TOKEN_DIM,
            load_openfly_action_codec,
        )

        incompatible = {
            "logical num_actions != 10": cfg.num_actions != 10,
            "action_chunk_horizon != 1": cfg.action_chunk_horizon != 1,
            "class weighting enabled": cfg.class_weights_path is not None
            or cfg.inflection_weight_coef != 1.0,
            "STOP auxiliary enabled": cfg.use_stop_aux_head
            or cfg.use_hierarchical_stop,
            "progress auxiliary enabled": cfg.use_progress_aux_head,
            "transition auxiliary enabled": cfg.use_transition_aux_head,
            "sequence training enabled": cfg.sequence_length != 1,
            "wrong token dimension": (
                cfg.openfly_action_token_dim != OPENFLY_ACTION_TOKEN_DIM
            ),
            "wrong token-bin count": (
                cfg.openfly_action_token_bins != OPENFLY_ACTION_TOKEN_BINS
            ),
            "missing codec path": not cfg.openfly_action_codec_path,
            "missing codec digest": not cfg.openfly_action_codec_sha256,
        }
        enabled = [name for name, value in incompatible.items() if value]
        if enabled:
            raise ValueError(
                f"OpenFly {cfg.openfly_action_head_type} has a fixed 8x256 action-token "
                "contract; incompatible: " + ", ".join(enabled)
            )
        openfly_action_codec = load_openfly_action_codec(
            cfg.openfly_action_codec_path,
            expected_sha256=cfg.openfly_action_codec_sha256,
        )
    elif cfg.openfly_action_head_type != "categorical":
        raise ValueError(
            "openfly_action_head_type must be 'categorical', 'action_tokens_v1', "
            "or 'action_tokens_ar_v2'"
        )
    resume_contract = training_contract(cfg)

    model_type = getattr(cfg, "model_type", "openvla")

    # Per-model guard rails (keep existing safety checks)
    if model_type in {"openvla", "qwen35_native"}:
        if not getattr(cfg, "use_lora", True):
            raise NotImplementedError("This trainer currently supports LoRA fine-tuning only")
        if not getattr(cfg, "freeze_vision", True):
            raise NotImplementedError("Vision-tower fine-tuning is intentionally disabled")

    if model_type == "qwen35_native":
        from aerial_openvla.data.qwen35_relative_state import (
            QWEN35_RELATIVE_NONE,
            validate_qwen35_relative_state_mode,
        )

        relative_state_mode = validate_qwen35_relative_state_mode(
            getattr(cfg, "qwen35_relative_state_mode", QWEN35_RELATIVE_NONE)
        )
        if relative_state_mode != QWEN35_RELATIVE_NONE:
            incompatible = {
                "input_mode != image": cfg.input_mode != "image",
                "TaskGRU/SegTS enabled": cfg.task_state_mode != "none",
                "Native1T/H16 enabled": cfg.use_native1t_postfusion,
                "legacy Relative enabled": (
                    cfg.use_relative_state_text or cfg.use_relative_planar_state_text
                ),
                "offline validation enabled": cfg.val_manifest is not None,
                "action chunking enabled": cfg.action_chunk_horizon != 1,
                "weighted action loss enabled": (
                    cfg.class_weights_path is not None
                    or cfg.inflection_weight_coef != 1.0
                ),
                "learned ACT disabled": not cfg.use_learned_action_query,
            }
            enabled = [name for name, value in incompatible.items() if value]
            if enabled:
                raise ValueError(
                    "Pure Qwen3.5 append-planar Relative requires single-frame "
                    "unweighted learned-ACT training only; incompatible: "
                    + ", ".join(enabled)
                )
            if not getattr(cfg, "qwen35_relative_state_cache", None):
                raise ValueError(
                    "Pure Qwen3.5 append-planar Relative requires "
                    "qwen35_relative_state_cache"
                )

    if model_type == "qwen35_nativemem":
        # NativeMEM uses cached pre-merger features. The single-stage V2 branch
        # is a separate contract from the existing two-stage implementation.
        # Keep these guards before model loading so incompatible jobs cannot
        # allocate GPUs.
        if cfg.input_mode != "feature":
            raise ValueError("NativeMEM v2 training requires input_mode=feature")
        if cfg.nativemem_stage not in (1, 2):
            raise ValueError("NativeMEM v2 stage must be 1 or 2")
        if cfg.nativemem_single_stage and cfg.nativemem_stage != 2:
            raise ValueError("NativeMEM single-stage V2 requires strict causal H16 (stage=2)")
        if cfg.nativemem_single_stage and cfg.init_checkpoint and not cli.resume:
            raise ValueError("NativeMEM single-stage V2 must start from official Qwen, not Stage1")
        if cfg.nativemem_history_length != 16 or cfg.nativemem_stage1_frames != 9:
            raise ValueError("NativeMEM v2 fixes H16 and nine Stage1 slots")
        if not cfg.nativemem_premerger_train_cache:
            raise ValueError("NativeMEM v2 requires the train pre-merger cache")
        if (
            cfg.nativemem_stage == 2
            and not cfg.nativemem_single_stage
            and not getattr(cfg, "nativemem_1t_train_cache", None)
        ):
            raise ValueError("NativeMEM v2 Stage2 requires the matched 1T cache")
        if cfg.val_manifest is not None:
            raise ValueError("NativeMEM v2 formal training does not run offline validation")
        if cfg.action_chunk_horizon != 1:
            raise ValueError("NativeMEM v2 uses single-step 8-way CE")
        if cfg.class_weights_path is not None or cfg.inflection_weight_coef != 1.0:
            raise ValueError("NativeMEM v2 uses unweighted 8-way action CE")
        if not cfg.use_lora:
            raise ValueError("NativeMEM v2 requires a loaded LoRA adapter (frozen in Stage1, trainable in Stage2)")
        if (
            cfg.nativemem_stage == 2
            and not cfg.nativemem_single_stage
            and not cfg.init_checkpoint
            and not cli.resume
        ):
            raise ValueError("NativeMEM Stage2 requires a Stage1 init_checkpoint")

    # Fail before model loading: only the multiframe pipeline's collators emit
    # the prev_actions that inflection weighting needs.
    if cfg.inflection_weight_coef != 1.0 and cfg.input_mode != "multiframe_image":
        raise ValueError(
            "inflection_weight_coef requires input_mode=multiframe_image; "
            f"input_mode={cfg.input_mode!r} batches do not carry prev_actions"
        )
    if cfg.action_chunk_loss_weight <= 0:
        raise ValueError("action_chunk_loss_weight must be positive")
    if cfg.max_train_steps is not None and cfg.max_train_steps <= 0:
        raise ValueError("max_train_steps must be positive when set")
    if cfg.save_at_steps is not None:
        if any(step <= 0 for step in cfg.save_at_steps):
            raise ValueError("save_at_steps values must be positive")
        if len(set(cfg.save_at_steps)) != len(cfg.save_at_steps):
            raise ValueError("save_at_steps must not contain duplicates")
    if (
        cfg.scheduler_horizon_epochs is not None
        and cfg.scheduler_horizon_epochs <= 0
    ):
        raise ValueError("scheduler_horizon_epochs must be positive when set")
    if cfg.use_native1t_postfusion:
        if cfg.model_type == "airminivla" and cfg.airminivla_variant != "s0":
            raise ValueError("Native1T-PostFusion requires AirMiniVLA S0")
        if cfg.model_type not in {"airminivla", "qwen35_native"}:
            raise ValueError("Native1T requires AirMiniVLA S0 or Qwen3.5-native")
        if cfg.native1t_stage not in (1, 2):
            raise ValueError("Native1T stage must be 1 or 2")
        if cfg.native1t_history_length not in (8, 16, 31):
            raise ValueError(
                "Native1T-PostFusion supports history_length in {8, 16, 31}"
            )
        if cfg.native1t_stage1_memory_frames != 9:
            raise ValueError("Native1T Stage 1 fixes memory_frames=9")
        if cfg.val_manifest is not None:
            raise ValueError("Native1T training forbids offline validation")
        if cfg.use_temporal_memory or cfg.use_h16_frozen_pool or cfg.use_omtrack:
            raise ValueError("Native1T is independent of TMVLA/FrozenPool/OmTrack")
        if cfg.action_chunk_horizon not in (1, 4):
            raise ValueError("Native1T supports action_chunk_horizon in {1, 4}")
        if cfg.action_chunk_horizon > 1 and cfg.native1t_stage != 2:
            raise ValueError("Native1T AC4 is supported only for Stage 2")
        if cfg.class_weights_path is not None or cfg.inflection_weight_coef != 1.0:
            raise ValueError("Native1T uses unweighted action CE")
        if (
            cfg.model_type == "airminivla"
            and cfg.native1t_stage == 1
            and cfg.max_train_steps not in (None, 347)
        ):
            raise ValueError(
                "Native1T Stage 1 supports either the 800,000-image schedule "
                "(max_train_steps=347) or a complete epoch (max_train_steps unset)"
            )
        if cfg.native1t_stage == 2 and not cfg.native1t_train_cache:
            raise ValueError("Native1T Stage 2 requires its frozen-tokenizer cache")
        if cfg.native1t_stage == 2 and not (
            cfg.native1t_stage1_checkpoint or cfg.init_checkpoint
        ):
            raise ValueError("Native1T Stage 2 requires a Stage-1 tokenizer source")
        if cfg.native1t_stage == 1 and cfg.native1t_stage1_checkpoint:
            raise ValueError("Native1T Stage 1 cannot load a Stage-1 module source")
        if cfg.use_native1t_action_history:
            if cfg.native1t_stage != 2:
                raise ValueError("Native1T AH4 is supported only in Stage 2")
            if cfg.native1t_action_history_length != 4:
                raise ValueError("Native1T AH4 fixes action_history_length=4")
            if cfg.native1t_action_embedding_dim != 16:
                raise ValueError("Native1T AH4 fixes action_embedding_dim=16")
    if cfg.use_omtrack:
        if cfg.model_type != "airminivla":
            raise ValueError("OmTrack requires model_type=airminivla")
        if cfg.input_mode != "multiframe_image" or cfg.sequence_length != 1:
            raise ValueError("OmTrack uses single-sample multiframe_image training")
        if not cfg.omtrack_train_sparse_cache:
            raise ValueError("OmTrack requires a train sparse-cache directory")
        if cfg.val_manifest is not None or cfg.omtrack_val_sparse_cache is not None:
            raise ValueError(
                "OmTrack forbids offline val history: validate only with online "
                "episode-local cache in closed-loop AirSim evaluation"
            )
        if cfg.class_weights_path is not None or cfg.inflection_weight_coef != 1.0:
            raise ValueError("OmTrack uses unweighted equal-slot action CE")
    if cfg.task_state_use_gru:
        if cfg.model_type not in {"airminivla", "qwen35_native"}:
            raise ValueError("TaskGRU requires AirMiniVLA or pure qwen35_native")
        if cfg.task_state_mode != "segment_only" or not cfg.task_state_position_key:
            raise ValueError("TaskGRU requires the SegTS-v5 PosKey baseline")
        if cfg.task_state_sticky_belief or cfg.task_state_hierarchical_ia:
            raise ValueError("TaskGRU must not be combined with Sticky or HierIA")
        if cfg.task_gru_hidden_dim != 256:
            raise ValueError("Relative-SegTS-v6-TaskGRU fixes hidden_dim=256")
        if cfg.task_gru_training_mode != "episode_stream":
            raise ValueError("TaskGRU formal training requires episode_stream mode")
        if cfg.task_gru_tbptt_steps != 4:
            raise ValueError("TaskGRU episode stream fixes tbptt_steps=4")
        if cfg.task_gru_burn_in != 0 or cfg.task_gru_loss_window != 4:
            raise ValueError("TaskGRU episode stream forbids burn-in and supervises all 4 steps")
        if cfg.sequence_length != cfg.task_gru_tbptt_steps:
            raise ValueError("TaskGRU sequence_length must equal tbptt_steps=4")
        if cfg.input_mode != "multiframe_image" or cfg.num_frames != 1:
            raise ValueError("TaskGRU uses contiguous single-frame episode windows")
        if cfg.val_manifest is not None:
            raise ValueError("TaskGRU formal training does not run offline val")
        if cfg.action_chunk_horizon != 1:
            raise ValueError("TaskGRU uses the original single-step action head")
        if cfg.class_weights_path is not None or cfg.inflection_weight_coef != 1.0:
            raise ValueError("TaskGRU uses unweighted 8-way action CE only")
        if cfg.model_type == "qwen35_native":
            if cfg.use_native1t_postfusion or cfg.qwen35_relative_state_mode != "none":
                raise ValueError("Qwen3.5 TaskGRU is pure current-frame; H16/NativeMEM is forbidden")
            if cfg.use_relative_state_text or cfg.use_relative_planar_state_text:
                raise ValueError("Qwen3.5 TaskGRU must not add Relative state text")
            if not cfg.use_learned_action_query:
                raise ValueError("Qwen3.5 TaskGRU requires learned ACT")

    if cfg.stop_aux_loss_weight < 0:
        raise ValueError("stop_aux_loss_weight must be non-negative")
    plateau_fields = (
        cfg.train_loss_plateau_window_steps,
        cfg.train_loss_plateau_patience_steps,
        cfg.train_loss_plateau_min_steps,
    )
    if any(value < 0 for value in plateau_fields) or cfg.train_loss_plateau_min_delta < 0:
        raise ValueError("Train-loss plateau settings must be non-negative")
    plateau_enabled = cfg.train_loss_plateau_window_steps > 0
    if plateau_enabled:
        if cfg.train_loss_plateau_patience_steps <= 0:
            raise ValueError("Train-loss plateau stopping requires positive patience")
        if cfg.train_loss_plateau_window_steps % cfg.log_every_steps != 0:
            raise ValueError("Plateau window must be divisible by log_every_steps")
        if cfg.train_loss_plateau_patience_steps % cfg.log_every_steps != 0:
            raise ValueError("Plateau patience must be divisible by log_every_steps")
    if cfg.stop_aux_positive_weight <= 0:
        raise ValueError("stop_aux_positive_weight must be positive")
    if cfg.stop_aux_logit_weight < 0:
        raise ValueError("stop_aux_logit_weight must be non-negative")
    if (
        cfg.use_stop_aux_head
        and cfg.stop_aux_loss_weight == 0
        and cfg.stop_aux_logit_weight == 0
    ):
        raise ValueError(
            "Enabled STOP auxiliary head needs a positive stop_aux_loss_weight "
            "or stop_aux_logit_weight"
        )
    if cfg.use_stop_aux_head and model_type != "airminivla":
        raise ValueError("use_stop_aux_head is currently supported only for model_type=airminivla")
    if cfg.use_hierarchical_stop and model_type != "airminivla":
        raise ValueError("use_hierarchical_stop is supported only for model_type=airminivla")
    if cfg.use_hierarchical_stop and cfg.use_stop_aux_head:
        raise ValueError("use_hierarchical_stop and use_stop_aux_head are mutually exclusive")
    if cfg.use_hierarchical_stop and cfg.stop_aux_loss_weight <= 0:
        raise ValueError("Hierarchical STOP requires positive stop_aux_loss_weight")
    if not 0.0 < cfg.hierarchical_stop_threshold < 1.0:
        raise ValueError("hierarchical_stop_threshold must be in (0, 1)")
    if not 0.0 < cfg.hierarchical_stop_initial_probability < 1.0:
        raise ValueError("hierarchical_stop_initial_probability must be in (0, 1)")
    if cfg.use_progress_aux_head and model_type != "airminivla":
        raise ValueError("use_progress_aux_head is supported only for model_type=airminivla")
    if cfg.progress_mode not in {"direct", "text_query"}:
        raise ValueError("progress_mode must be 'direct' or 'text_query'")
    if cfg.progress_loss_weight < 0:
        raise ValueError("progress_loss_weight must be non-negative")
    if cfg.progress_smooth_l1_beta <= 0:
        raise ValueError("progress_smooth_l1_beta must be positive")
    if cfg.use_progress_aux_head and cfg.progress_loss_weight == 0:
        raise ValueError("Enabled progress auxiliary head needs positive progress_loss_weight")
    if cfg.use_progress_aux_head and cfg.sequence_length > 1:
        raise ValueError("Progress auxiliary training currently requires sequence_length=1")
    if cfg.use_hierarchical_stop and cfg.sequence_length > 1:
        raise ValueError("Hierarchical STOP training currently requires sequence_length=1")
    if cfg.transition_loss_weight < 0:
        raise ValueError("transition_loss_weight must be non-negative")
    if cfg.use_transition_aux_head and model_type != "airminivla":
        raise ValueError(
            "use_transition_aux_head is supported only for model_type=airminivla"
        )
    if cfg.use_transition_aux_head and cfg.transition_loss_weight == 0:
        raise ValueError(
            "Enabled transition auxiliary head needs positive transition_loss_weight"
        )
    if cfg.use_transition_aux_head and cfg.sequence_length > 1:
        raise ValueError("Transition auxiliary training currently requires sequence_length=1")
    if cfg.use_temporal_memory:
        if cfg.input_mode != "multiframe_image" or cfg.num_frames != 1:
            raise ValueError(
                "Native TM-VLA requires multiframe_image with num_frames=1"
            )
        if cfg.sequence_length < 2:
            raise ValueError("Native TM-VLA requires sequence_length >= 2")
        if cfg.action_chunk_horizon != 1:
            raise ValueError("Native TM-VLA is single-step and forbids action chunking")
        if cfg.class_weights_path is not None or cfg.inflection_weight_coef != 1.0:
            raise ValueError("Native TM-VLA uses unweighted 8-class action CE only")

    context = init_distributed()
    seed_everything(cfg.seed, context.rank)

    if context.is_main:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        print(json.dumps(config_to_dict(cfg), indent=2, default=str))

    # ---- Model (factory) --------------------------------------------------
    processor, model = create_model(cfg, device=context.device)
    if cfg.model_type == "qwen35_native" and cfg.task_state_mode != "none":
        model.enforce_qwen35_segts_trainable_contract()

    # A later stage can change the trainable parameter set (for example, add
    # LoRA adapters or unfreeze a vision tail).  Load its predecessor's model
    # weights before constructing the optimizer, but deliberately start a new
    # optimizer and scheduler for this stage.  --resume takes priority: once
    # this stage has its own checkpoint, continue from it rather than
    # re-seeding from the previous stage, so stage 2/3 jobs can always be
    # resubmitted with --resume.
    init_checkpoint = getattr(cfg, "init_checkpoint", None)
    resume_ckpt: Path | None = None
    resume_extra: dict[str, Any] | None = None
    if cli.resume:
        resume_ckpt, resume_extra = find_latest_checkpoint(cfg.output_dir)
        if resume_ckpt is None and not init_checkpoint:
            raise FileNotFoundError(
                f"--resume requested but no usable checkpoint was found in {cfg.output_dir}; "
                "refusing to silently restart from scratch"
            )
    if init_checkpoint and resume_ckpt is None:
        init_path = Path(init_checkpoint)
        if not init_path.is_dir():
            raise FileNotFoundError(f"Initial checkpoint not found: {init_path}")
        model.load_classifier_checkpoint(
            init_path,
            allow_missing_native1t_action_history=bool(
                cfg.use_native1t_action_history and cfg.native1t_stage == 2
            ),
        )
        if (
            cfg.use_native1t_postfusion
            and cfg.native1t_stage == 2
            and cfg.native1t_stage1_checkpoint
        ):
            native1t_source = Path(cfg.native1t_stage1_checkpoint)
            if not native1t_source.is_dir():
                raise FileNotFoundError(
                    f"Native1T Stage-1 module source not found: {native1t_source}"
                )
            model.load_native1t_postfusion_modules(native1t_source)
            if context.is_main:
                print(f"[init] overlaid Native1T modules from {native1t_source}")
        if cfg.use_native1t_postfusion:
            model.enforce_native1t_trainable_contract()
        if cfg.model_type == "qwen35_nativemem":
            model.enforce_nativemem_trainable_contract()
        if cfg.model_type == "qwen35_native" and cfg.task_state_mode != "none":
            model.enforce_qwen35_segts_trainable_contract()
        if context.is_main:
            print(f"[init] loaded model weights from {init_path}; optimizer is fresh")
    elif (
        cfg.use_native1t_postfusion
        and cfg.native1t_stage == 2
        and cfg.native1t_stage1_checkpoint
        and resume_ckpt is None
    ):
        native1t_source = Path(cfg.native1t_stage1_checkpoint)
        if not native1t_source.is_dir():
            raise FileNotFoundError(
                f"Native1T Stage-1 module source not found: {native1t_source}"
            )
        model.load_native1t_postfusion_modules(native1t_source)
        model.enforce_native1t_trainable_contract()
        if context.is_main:
            print(
                "[init] loaded only Native1T tokenizer/projector/MEM_BOS from "
                f"{native1t_source}; policy, Qwen LoRA, original projector and "
                "action path retain fresh initialization"
            )

    native1t_trainable_names = native1t_trainable_contract_names(model, cfg)
    if cfg.use_native1t_postfusion and context.is_main:
        print(json.dumps({
            "native1t_stage": cfg.native1t_stage,
            "native1t_trainable_parameters": native1t_trainable_names,
            "native1t_trainable_count": len(native1t_trainable_names),
        }, sort_keys=True))

    # ---- Data pipeline (factory) -------------------------------------------
    pipeline = create_pipeline(cfg)

    train_sampler: OffsetSampler | None = None
    task_gru_episode_stream = bool(
        cfg.task_state_use_gru
        and cfg.task_gru_training_mode == "episode_stream"
    )
    if cfg.use_temporal_memory or task_gru_episode_stream:
        from aerial_openvla.data.multiframe_dataset import (
            TaskGRUEpisodeStreamBatchSampler,
            TemporalStreamBatchSampler,
        )

        sampler_class = (
            TaskGRUEpisodeStreamBatchSampler
            if task_gru_episode_stream
            else TemporalStreamBatchSampler
        )
        base_train_sampler = sampler_class(
            pipeline.dataset,
            cfg.batch_size,
            rank=context.rank,
            world_size=context.world_size,
            seed=cfg.seed,
            shuffle=True,
        )
        train_loader = DataLoader(
            pipeline.dataset,
            batch_sampler=base_train_sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=cfg.num_workers > 0,
            collate_fn=pipeline.collator,
        )
    else:
        base_train_sampler = (
            DistributedSampler(pipeline.dataset, shuffle=True, seed=cfg.seed)
            if context.distributed
            else EpochRandomSampler(pipeline.dataset, cfg.seed)
        )
        train_sampler = OffsetSampler(base_train_sampler)
        train_loader = DataLoader(
            pipeline.dataset,
            batch_size=cfg.batch_size,
            sampler=train_sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=cfg.num_workers > 0,
            collate_fn=pipeline.collator,
            drop_last=True,
        )

    val_loader = None
    val_sampler = None
    if cfg.val_manifest:
        val_dataset = create_dataset(cfg, cfg.val_manifest)
        if cfg.use_temporal_memory or task_gru_episode_stream:
            from aerial_openvla.data.multiframe_dataset import (
                TaskGRUEpisodeStreamBatchSampler,
                TemporalStreamBatchSampler,
            )

            sampler_class = (
                TaskGRUEpisodeStreamBatchSampler
                if task_gru_episode_stream
                else TemporalStreamBatchSampler
            )
            val_sampler = sampler_class(
                val_dataset,
                cfg.batch_size,
                rank=context.rank,
                world_size=context.world_size,
                seed=cfg.seed,
                shuffle=False,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_sampler=val_sampler,
                num_workers=max(1, cfg.num_workers // 2),
                pin_memory=True,
                collate_fn=pipeline.collator,
            )
        else:
            val_sampler = DistributedSampler(val_dataset, shuffle=False) if context.distributed else None
            val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.batch_size,
                sampler=val_sampler,
                shuffle=False,
                num_workers=max(1, cfg.num_workers // 2),
                pin_memory=True,
                collate_fn=pipeline.collator,
                drop_last=False,
            )

    # ---- Optimizer & scheduler --------------------------------------------
    lr_kwargs = collect_lr_kwargs(cfg, model_type)
    parameter_groups = model.trainable_parameter_groups(
        head_lr=cfg.learning_rate_head,
        weight_decay=cfg.weight_decay,
        **lr_kwargs,
    )
    optimizer = AdamW(parameter_groups)
    updates_per_epoch = math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
    scheduler_horizon_epochs = (
        cfg.scheduler_horizon_epochs
        if cfg.scheduler_horizon_epochs is not None
        else cfg.epochs
    )
    total_updates = max(1, updates_per_epoch * scheduler_horizon_epochs)
    # Preserve legacy schedules unless an explicit horizon is requested. This
    # lets H16 Stage2 stop at epoch 2 while retaining the approved 4-epoch
    # warmup and cosine denominator.
    if cfg.max_train_steps is not None and cfg.scheduler_horizon_epochs is None:
        total_updates = min(total_updates, cfg.max_train_steps)
    warmup_steps = int(total_updates * cfg.warmup_ratio)
    scheduler = LambdaLR(
        optimizer,
        lambda step: cosine_with_warmup(step, warmup_steps=warmup_steps, total_steps=total_updates),
    )
    class_weights = load_class_weights(
        cfg.class_weights_path,
        context.device,
        cfg.num_actions,
        max_weight=getattr(cfg, "class_weight_max", None),
    )

    # ---- Resume -----------------------------------------------------------
    start_epoch = 0
    global_step = 0
    if resume_ckpt is not None:
        model.load_classifier_checkpoint(resume_ckpt, optimizer=optimizer, scheduler=scheduler)
        if cfg.use_native1t_postfusion:
            model.enforce_native1t_trainable_contract()
        if cfg.model_type == "qwen35_nativemem":
            model.enforce_nativemem_trainable_contract()
        if cfg.model_type == "qwen35_native" and cfg.task_state_mode != "none":
            model.enforce_qwen35_segts_trainable_contract()
        if resume_extra:
            saved_contract = resume_extra.get("training_contract")
            if saved_contract is None:
                if context.is_main:
                    print("[resume] legacy checkpoint has no manifest contract; recording one on the next save")
            elif not contracts_compatible(saved_contract, resume_contract):
                raise ValueError(
                    "Checkpoint training configuration or manifest content differs from this run; "
                    "refusing an unsafe resume."
                )
            saved_world_size = resume_extra.get("world_size")
            if saved_world_size is not None and saved_world_size != context.world_size:
                raise ValueError(
                    f"Checkpoint was written with world_size={saved_world_size} but this run "
                    f"has world_size={context.world_size}; updates_per_epoch and the sample "
                    "offset would silently diverge. Resume with the same number of processes."
                )
            global_step = resume_extra.get("global_step", 0)
            # Derive the epoch from global_step alone; the stored epoch field
            # is ambiguous exactly at epoch boundaries (0-based for step
            # checkpoints, 1-based for epoch checkpoints).
            start_epoch = global_step // updates_per_epoch
            saved_epoch = resume_extra.get("epoch")
            if saved_epoch is not None and not (saved_epoch <= start_epoch <= saved_epoch + 1):
                raise ValueError(
                    f"Derived epoch {start_epoch} is inconsistent with the checkpoint's "
                    f"recorded epoch {saved_epoch}; refusing to resume with a mismatched "
                    "updates_per_epoch."
                )
            ranked_states = resume_extra.get("rng_states", {})
            restore_rng_state(ranked_states.get(str(context.rank), resume_extra.get("rng_state")))
        if context.is_main:
            print(f"[resume] loaded checkpoint {resume_ckpt}, epoch={start_epoch}, step={global_step}")

    wrapped: torch.nn.Module = model
    if context.distributed:
        wrapped = DDP(
            model,
            device_ids=[context.local_rank],
            # The scoped Native1T SigLIP tail contains only blocks that
            # participate in Prismatic's emitted visual feature; unlike the
            # legacy all-tower option, it does not require unused-param scans.
            find_unused_parameters=cfg.vision_train_last_n_layers > 0,
        )

    stage1_profiler = Stage1TimingProfiler(
        rank=context.rank,
        world_size=context.world_size,
        device=context.device,
    )
    from aerial_openvla.models.airminivla_classifier import _get_internal_decoder

    stage1_profiler.install_module_hook(
        "frozen_dual_vision_backbone", model.backbone.vision_backbone
    )
    stage1_profiler.install_module_hook(
        "native1t_memory_tokenizer", model.native1t_memory_tokenizer
    )
    stage1_profiler.install_module_hook(
        "current_prism_projector", model.backbone.projector
    )
    stage1_profiler.install_module_hook(
        "native1t_memory_projector", model.native1t_memory_projector
    )
    stage1_profiler.install_module_hook(
        "qwen_decoder", _get_internal_decoder(model.backbone)
    )
    stage1_profiler.install_module_hook(
        "categorical_action_head", getattr(model, "action_head", None)
    )

    optimizer.zero_grad(set_to_none=True)
    metrics_log_path = Path(cfg.output_dir) / "train_metrics.jsonl"
    metrics_log_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_log = metrics_log_path.open("a", encoding="utf-8")
    if context.is_main:
        # Run-boundary marker: downstream tooling can split segments instead of
        # guessing them from global_step regressions.  The TensorBoard bridge
        # ignores records without loss/validation keys.
        metrics_log.write(json.dumps({
            "run_start": True,
            "global_step": global_step,
            "epoch": start_epoch + 1,
            "batch_size": cfg.batch_size,
            "grad_accum": cfg.gradient_accumulation_steps,
            "inflection_weight_coef": cfg.inflection_weight_coef,
            "class_weights": cfg.class_weights_path is not None,
            "transition_loss_weight": (
                cfg.transition_loss_weight if cfg.use_transition_aux_head else 0.0
            ),
        }) + "\n")
        metrics_log.flush()
        print(json.dumps({
            "updates_per_epoch": updates_per_epoch,
            "physical_max_train_steps": cfg.max_train_steps,
            "scheduler_horizon_epochs": scheduler_horizon_epochs,
            "scheduler_total_updates": total_updates,
            "scheduler_warmup_steps": warmup_steps,
        }, sort_keys=True))

    # ---- Main loop ---------------------------------------------------------
    plateau_losses: list[float] = []
    plateau_best: float | None = None
    plateau_last_improvement_step = 0
    stopped_for_plateau = False
    plateau_window_mean: float | None = None
    for epoch in range(start_epoch, cfg.epochs):
        if hasattr(base_train_sampler, "set_epoch"):
            base_train_sampler.set_epoch(epoch)
        if train_sampler is not None:
            train_sampler.offset = 0
        if train_sampler is not None and epoch == start_epoch and global_step > 0:
            train_sampler.offset = (
                global_step % updates_per_epoch
            ) * cfg.gradient_accumulation_steps * cfg.batch_size
        wrapped.train()
        epoch_update_offset = global_step % updates_per_epoch
        progress = tqdm(
            total=updates_per_epoch,
            initial=epoch_update_offset,
            disable=not context.is_main,
            desc=f"optimizer steps epoch {epoch + 1}/{cfg.epochs}",
            unit="update",
        )
        running_loss = 0.0
        running_loss_examples = 0
        running_correct = 0
        running_examples = 0
        running_progress_loss = 0.0
        running_progress_pred = 0.0
        running_progress_target = 0.0
        running_progress_examples = 0
        running_move_loss = 0.0
        running_move_examples = 0
        running_stop_loss = 0.0
        running_transition_loss = 0.0
        running_transition_correct = 0
        running_transition_examples = 0
        running_openfly_vector_abs_sum = 0.0
        running_openfly_unmatched = 0
        running_openfly_codebook_correct = 0
        running_openfly_codebook_examples = 0
        temporal_stream_state: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        task_gru_stream_state: torch.Tensor | None = None

        for micro_step, batch in enumerate(train_loader):
            if cfg.max_train_steps is not None and global_step >= cfg.max_train_steps:
                break
            stage1_profiler.begin_microbatch(global_step)
            h2d_started = (
                stage1_profiler.timed_cuda_start()
                if stage1_profiler.active else 0.0
            )
            batch = move_batch(batch, context.device, cfg.bf16)
            if stage1_profiler.active:
                stage1_profiler.timed_cuda_end("host_to_device", h2d_started)
            window_start = (micro_step // cfg.gradient_accumulation_steps) * cfg.gradient_accumulation_steps
            window_end = min(window_start + cfg.gradient_accumulation_steps, len(train_loader))
            window_size = window_end - window_start
            should_update = micro_step + 1 == window_end
            sync_context = nullcontext()
            if context.distributed and not should_update:
                sync_context = wrapped.no_sync()  # type: ignore[attr-defined]

            forward_started = (
                stage1_profiler.timed_cuda_start()
                if stage1_profiler.active else 0.0
            )
            with sync_context:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16):
                    if "sequence_mask" in batch:
                        if cfg.use_temporal_memory:
                            loss, predictions, labels = _temporal_stream_forward(
                                batch,
                                wrapped,
                                temporal_stream_state,
                                class_weights,
                                cfg.inflection_weight_coef,
                            )
                        else:
                            (
                                loss,
                                predictions,
                                labels,
                                task_gru_stream_state,
                            ) = _sequence_forward(
                                batch,
                                wrapped,
                                class_weights,
                                cfg.inflection_weight_coef,
                                cfg.stop_aux_loss_weight if (
                                    cfg.use_stop_aux_head or cfg.use_hierarchical_stop
                                ) else 0.0,
                                cfg.stop_aux_positive_weight,
                                task_gru_state_prev=task_gru_stream_state,
                                episode_stream=task_gru_episode_stream,
                            )
                        loss_examples = int(labels.numel())
                    else:
                        outputs = _forward_outputs(batch, wrapped)
                        logits = outputs["action_logits"]
                        stop_logits = outputs["stop_logits"]
                        if openfly_action_codec is not None:
                            targets = batch.get("openfly_action_token_targets")
                            if targets is None:
                                raise RuntimeError(
                                    "OpenFly action-token batch lacks token targets"
                                )
                            labels = batch["action_id"]
                            if _uses_openfly_action_tokens_ar(wrapped):
                                loss_sum = outputs.get("openfly_lm_loss_sum")
                                loss_tokens = outputs.get("openfly_lm_loss_tokens")
                                predicted_token_ids = outputs.get("openfly_predicted_token_ids")
                                if loss_sum is None or loss_tokens is None or predicted_token_ids is None:
                                    raise RuntimeError(
                                        "OpenFly autoregressive training output lacks LM loss or token IDs"
                                    )
                                loss_examples = int(loss_tokens.item())
                                loss = loss_sum / max(loss_examples, 1)
                                metric_kwargs = {
                                    "predicted_token_ids": predicted_token_ids,
                                    "action_vocab_size": _openfly_action_vocab_size(wrapped),
                                }
                            else:
                                loss = openfly_action_token_loss(logits, targets)
                                loss_examples = int(targets.numel())
                                metric_kwargs = {}
                            (
                                predictions,
                                token_correct,
                                token_examples,
                                vector_abs_sum,
                                unmatched,
                            ) = _openfly_token_metrics(
                                logits, targets, openfly_action_codec, **metric_kwargs
                            )
                        elif cfg.action_chunk_horizon > 1:
                            targets = batch["action_chunk_targets"]
                            if cfg.use_hierarchical_stop:
                                move_loss, stop_loss = hierarchical_action_chunk_losses(
                                    outputs["move_logits"],
                                    stop_logits,
                                    targets,
                                    cfg.stop_aux_positive_weight,
                                )
                                loss = move_loss + cfg.stop_aux_loss_weight * stop_loss
                            else:
                                loss = cfg.action_chunk_loss_weight * action_chunk_loss(
                                    logits, targets
                                )
                            predictions = logits[:, 0].argmax(dim=-1)
                            labels = targets[:, 0]
                            loss_examples = int((targets != -100).sum().item())
                        elif cfg.use_hierarchical_stop:
                            move_loss = movement_action_loss(
                                outputs.get("move_logits"),
                                batch["action_id"],
                                batch.get("prev_actions"),
                                class_weights,
                                cfg.inflection_weight_coef,
                            )
                            loss = move_loss
                        else:
                            loss = weighted_step_loss(
                                logits.float(),
                                batch["action_id"],
                                batch.get("prev_actions"),
                                class_weights,
                                cfg.inflection_weight_coef,
                            )
                        if openfly_action_codec is None and (
                            cfg.use_stop_aux_head or cfg.use_hierarchical_stop
                        ) and not (
                            cfg.action_chunk_horizon > 1 and cfg.use_hierarchical_stop
                        ):
                            stop_loss = stop_auxiliary_loss(
                                stop_logits,
                                batch["action_id"],
                                cfg.stop_aux_positive_weight,
                            )
                            loss = loss + cfg.stop_aux_loss_weight * stop_loss
                        if openfly_action_codec is None and cfg.use_progress_aux_head:
                            progress_predictions = outputs["progress"]
                            progress_targets = batch["progress_targets"]
                            progress_valid = batch.get("progress_valid_mask")
                            progress_loss = progress_auxiliary_loss(
                                progress_predictions,
                                progress_targets,
                                progress_valid,
                                cfg.progress_smooth_l1_beta,
                            )
                            loss = loss + cfg.progress_loss_weight * progress_loss
                        if openfly_action_codec is None and cfg.use_transition_aux_head:
                            transition_logits = outputs["transition_logits"]
                            transition_loss = transition_auxiliary_loss(
                                transition_logits,
                                batch["action_id"],
                                batch.get("prev_actions"),
                                cfg.num_actions,
                            )
                            loss = loss + cfg.transition_loss_weight * transition_loss
                        if openfly_action_codec is None and cfg.action_chunk_horizon == 1:
                            predictions = logits.argmax(dim=-1)
                            labels = batch["action_id"]
                            loss_examples = int(labels.numel())
                    scaled_loss = loss / window_size
                if stage1_profiler.active:
                    stage1_profiler.timed_cuda_end("forward_and_loss", forward_started)
                    stage1_profiler.flush_module_events()
                backward_started = (
                    stage1_profiler.timed_cuda_start()
                    if stage1_profiler.active else 0.0
                )
                scaled_loss.backward()
                if stage1_profiler.active:
                    stage1_profiler.timed_cuda_end("backward", backward_started)

            metric_started = time.perf_counter()
            running_loss += float(loss.item()) * loss_examples
            running_loss_examples += loss_examples
            if openfly_action_codec is not None:
                # The canonical training accuracy is the OpenFly
                # teacher-forced action-token metric, not codebook agreement.
                running_correct += int(token_correct.item())
                running_examples += int(token_examples.item())
                running_openfly_vector_abs_sum += float(vector_abs_sum.item())
                running_openfly_unmatched += int(unmatched.item())
                running_openfly_codebook_correct += int(
                    (predictions == labels).sum().item()
                )
                running_openfly_codebook_examples += int(labels.numel())
            else:
                running_correct += int((predictions == labels).sum().item())
                running_examples += int(labels.numel())
            if cfg.use_hierarchical_stop:
                metric_targets = (
                    targets if cfg.action_chunk_horizon > 1 else labels
                )
                metric_valid = metric_targets != -100
                moving_count = int((metric_valid & (metric_targets != 0)).sum().item())
                stop_count = int(metric_valid.sum().item())
                running_move_loss += float(move_loss.item()) * moving_count
                running_move_examples += moving_count
                running_stop_loss += float(stop_loss.item()) * stop_count
            if cfg.use_progress_aux_head:
                valid = batch.get("progress_valid_mask")
                selected = valid.bool() if valid is not None else torch.ones_like(
                    batch["progress_targets"], dtype=torch.bool
                )
                count = int(selected.sum().item())
                running_progress_loss += float(progress_loss.item()) * count
                running_progress_pred += float(progress_predictions.float()[selected].sum().item())
                running_progress_target += float(batch["progress_targets"].float()[selected].sum().item())
                running_progress_examples += count
            if cfg.use_transition_aux_head:
                transition_targets_batch, transition_valid = transition_targets(
                    labels, batch.get("prev_actions"), cfg.num_actions
                )
                valid_count = int(transition_valid.sum().item())
                running_transition_loss += float(transition_loss.item()) * valid_count
                running_transition_correct += int((
                    (transition_logits >= 0)[transition_valid]
                    == transition_targets_batch[transition_valid].bool()
                ).sum().item())
                running_transition_examples += valid_count
            if stage1_profiler.active:
                stage1_profiler.add(
                    "metric_bookkeeping", time.perf_counter() - metric_started
                )

            if should_update:
                grad_clip_started = (
                    stage1_profiler.timed_cuda_start()
                    if stage1_profiler.active else 0.0
                )
                torch.nn.utils.clip_grad_norm_(
                    [p for p in wrapped.parameters() if p.requires_grad],
                    cfg.max_grad_norm,
                )
                if stage1_profiler.active:
                    stage1_profiler.timed_cuda_end("gradient_clip", grad_clip_started)
                optimizer_started = (
                    stage1_profiler.timed_cuda_start()
                    if stage1_profiler.active else 0.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if stage1_profiler.active:
                    stage1_profiler.timed_cuda_end(
                        "optimizer_scheduler_zero_grad", optimizer_started
                    )
                global_step += 1
                progress.update(1)

                if global_step % cfg.log_every_steps == 0:
                    log_totals = torch.tensor(
                        [running_loss, running_loss_examples, running_correct, running_examples,
                         running_progress_loss, running_progress_pred,
                         running_progress_target, running_progress_examples,
                         running_move_loss, running_move_examples, running_stop_loss,
                         running_transition_loss, running_transition_correct,
                         running_transition_examples,
                         running_openfly_vector_abs_sum, running_openfly_unmatched,
                         running_openfly_codebook_correct,
                         running_openfly_codebook_examples],
                        dtype=torch.float64,
                        device=context.device,
                    )
                    reduce_sum(log_totals)
                    (total_loss, total_loss_examples, total_correct, total_examples,
                     total_progress_loss,
                     total_progress_pred, total_progress_target,
                     total_progress_examples, total_move_loss,
                     total_move_examples, total_stop_loss,
                     total_transition_loss, total_transition_correct,
                     total_transition_examples,
                     total_openfly_vector_abs_sum, total_openfly_unmatched,
                     total_openfly_codebook_correct,
                     total_openfly_codebook_examples) = log_totals.tolist()
                    current_loss = total_loss / max(total_loss_examples, 1)
                    current_acc = total_correct / max(total_examples, 1)
                    if context.is_main:
                        current_lr = scheduler.get_last_lr()[0]
                        progress_fields: dict[str, float] = {
                            "loss": current_loss,
                            "lr": current_lr,
                        }
                        progress_fields[
                            "action_acc" if openfly_action_codec is not None else "acc"
                        ] = current_acc
                        progress.set_postfix(**progress_fields)
                        log_record = {
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "loss": round(current_loss, 6),
                            "accuracy": round(current_acc, 6),
                            "examples": int(total_examples),
                            "lr": current_lr,
                            "batch_size": cfg.batch_size,
                        }
                        if openfly_action_codec is not None:
                            action_count = max(total_openfly_codebook_examples, 1)
                            l1_loss = total_openfly_vector_abs_sum / max(total_examples, 1)
                            log_record.update({
                                "openfly_metric_protocol": (
                                    "official_teacher_forced_action_tokens"
                                ),
                                # Canonical OpenFly names.  ``accuracy`` above
                                # is deliberately the same action-token value.
                                "action_accuracy": round(current_acc, 6),
                                "l1_loss": round(l1_loss, 6),
                                "action_token_accuracy": round(current_acc, 6),
                                "action_l1_loss": round(l1_loss, 6),
                                "action_token_examples": int(total_examples),
                                "action_examples": int(total_openfly_codebook_examples),
                                "codebook_exact_agreement": round(
                                    total_openfly_codebook_correct / action_count, 6
                                ),
                                "decoded_unmatched_rate": round(
                                    total_openfly_unmatched / action_count, 6
                                ),
                            })
                        if total_progress_examples > 0:
                            log_record.update({
                                "progress_loss": round(total_progress_loss / total_progress_examples, 6),
                                "progress_pred_mean": round(total_progress_pred / total_progress_examples, 6),
                                "progress_target_mean": round(total_progress_target / total_progress_examples, 6),
                            })
                        if cfg.use_hierarchical_stop:
                            log_record.update({
                                "move_loss": round(
                                    total_move_loss / max(total_move_examples, 1), 6
                                ),
                                "stop_loss": round(
                                    total_stop_loss / max(total_examples, 1), 6
                                ),
                                "moving_examples": int(total_move_examples),
                            })
                        if total_transition_examples > 0:
                            log_record.update({
                                "transition_loss": round(
                                    total_transition_loss / total_transition_examples, 6
                                ),
                                "transition_accuracy": round(
                                    total_transition_correct / total_transition_examples, 6
                                ),
                                "transition_examples": int(total_transition_examples),
                            })
                        metrics_log.write(json.dumps(log_record) + "\n")
                        metrics_log.flush()
                    if plateau_enabled:
                        plateau_losses.append(current_loss)
                        window_points = (
                            cfg.train_loss_plateau_window_steps // cfg.log_every_steps
                        )
                        if len(plateau_losses) >= window_points:
                            plateau_window_mean = sum(
                                plateau_losses[-window_points:]
                            ) / window_points
                            if (
                                plateau_best is None
                                or plateau_window_mean
                                < plateau_best - cfg.train_loss_plateau_min_delta
                            ):
                                plateau_best = plateau_window_mean
                                plateau_last_improvement_step = global_step
                            elif (
                                global_step >= cfg.train_loss_plateau_min_steps
                                and global_step - plateau_last_improvement_step
                                >= cfg.train_loss_plateau_patience_steps
                            ):
                                stopped_for_plateau = True
                                if context.is_main:
                                    metrics_log.write(json.dumps({
                                        "early_stop": "train_loss_plateau",
                                        "global_step": global_step,
                                        "window_mean_loss": round(
                                            plateau_window_mean, 6
                                        ),
                                        "best_window_mean_loss": round(
                                            plateau_best, 6
                                        ),
                                        "window_steps": (
                                            cfg.train_loss_plateau_window_steps
                                        ),
                                        "patience_steps": (
                                            cfg.train_loss_plateau_patience_steps
                                        ),
                                        "min_delta": cfg.train_loss_plateau_min_delta,
                                    }) + "\n")
                                    metrics_log.flush()
                    running_loss = 0.0
                    running_loss_examples = 0
                    running_correct = 0
                    running_examples = 0
                    running_progress_loss = 0.0
                    running_progress_pred = 0.0
                    running_progress_target = 0.0
                    running_progress_examples = 0
                    running_move_loss = 0.0
                    running_move_examples = 0
                    running_stop_loss = 0.0
                    running_transition_loss = 0.0
                    running_transition_correct = 0
                    running_transition_examples = 0
                    running_openfly_vector_abs_sum = 0.0
                    running_openfly_unmatched = 0
                    running_openfly_codebook_correct = 0
                    running_openfly_codebook_examples = 0

                validation_checkpoint_saved = (
                    val_loader is not None
                    and cfg.val_every_steps > 0
                    and global_step % cfg.val_every_steps == 0
                )
                if validation_checkpoint_saved:
                    if val_sampler is not None:
                        val_sampler.set_epoch(epoch)
                    validation = validate(
                        model, val_loader, context.device, cfg.bf16, cfg.num_actions,
                        class_weights, cfg.inflection_weight_coef,
                        cfg.stop_aux_loss_weight if (
                            cfg.use_stop_aux_head or cfg.use_hierarchical_stop
                        ) else 0.0,
                        cfg.stop_aux_positive_weight,
                        cfg.progress_loss_weight if cfg.use_progress_aux_head else 0.0,
                        cfg.progress_smooth_l1_beta,
                        cfg.transition_loss_weight if cfg.use_transition_aux_head else 0.0,
                        openfly_action_codec,
                    )
                    checkpoint_extra = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "world_size": context.world_size,
                        "training_contract": resume_contract,
                        "rng_states": capture_ranked_rng_states(
                            context.rank, context.world_size, context.distributed
                        ),
                    }
                    if context.is_main:
                        validation_record = {
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "validation_loss": round(validation["loss"], 6),
                            "validation_accuracy": round(validation["accuracy"], 6),
                            "validation_examples": int(validation["examples"]),
                        }
                        if "transition_loss" in validation:
                            validation_record.update({
                                "validation_transition_loss": round(
                                    validation["transition_loss"], 6
                                ),
                                "validation_transition_accuracy": round(
                                    validation["transition_accuracy"], 6
                                ),
                                "validation_transition_examples": int(
                                    validation["transition_examples"]
                                ),
                            })
                        if "action_accuracy" in validation:
                            validation_record.update({
                                "validation_action_accuracy": round(
                                    validation["action_accuracy"], 6
                                ),
                                "validation_l1_loss": round(
                                    validation["l1_loss"], 6
                                ),
                                "validation_action_token_examples": int(
                                    validation["action_token_examples"]
                                ),
                                "validation_action_examples": int(
                                    validation["action_examples"]
                                ),
                                "validation_codebook_exact_agreement": round(
                                    validation["codebook_exact_agreement"], 6
                                ),
                                "validation_decoded_unmatched_rate": round(
                                    validation["decoded_unmatched_rate"], 6
                                ),
                            })
                        metrics_log.write(json.dumps(validation_record) + "\n")
                        metrics_log.flush()
                        print(json.dumps({"global_step": global_step, "validation": validation}, indent=2))
                        checkpoint = Path(cfg.output_dir) / f"step-{global_step:07d}"
                        model.save_classifier_checkpoint(
                            checkpoint,
                            processor,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            extra=checkpoint_extra,
                        )
                        update_latest_symlink(checkpoint, Path(cfg.output_dir) / "latest")
                        rotate_checkpoints(Path(cfg.output_dir), cfg.keep_last_checkpoints)
                    barrier()

                if should_save_step_checkpoint(
                    global_step,
                    cfg.save_every_steps,
                    validation_checkpoint_saved,
                    cfg.save_at_steps,
                ):
                    checkpoint_extra = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "world_size": context.world_size,
                        "training_contract": resume_contract,
                        "rng_states": capture_ranked_rng_states(
                            context.rank, context.world_size, context.distributed
                        ),
                    }
                    if context.is_main:
                        checkpoint = Path(cfg.output_dir) / f"step-{global_step:07d}"
                        model.save_classifier_checkpoint(
                            checkpoint,
                            processor,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            extra=checkpoint_extra,
                        )
                        update_latest_symlink(checkpoint, Path(cfg.output_dir) / "latest")
                        rotate_checkpoints(Path(cfg.output_dir), cfg.keep_last_checkpoints)
                        print(json.dumps({
                            "global_step": global_step,
                            "checkpoint": str(checkpoint),
                            "reason": (
                                "save_at_steps"
                                if global_step in set(cfg.save_at_steps or [])
                                else "save_every_steps"
                            ),
                        }))
                    barrier()

                if stopped_for_plateau:
                    stage1_profiler.end_microbatch(did_update=should_update)
                    break

            stage1_profiler.end_microbatch(did_update=should_update)

        progress.close()

        metrics: dict[str, Any] = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "early_stopped": stopped_for_plateau,
        }
        if stopped_for_plateau:
            metrics.update({
                "early_stop_reason": "train_loss_plateau",
                "window_mean_loss": plateau_window_mean,
                "best_window_mean_loss": plateau_best,
            })
        if val_loader is not None and (
            cfg.val_every_steps == 0 or global_step % cfg.val_every_steps != 0
        ):
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)
            metrics["validation"] = validate(
                model, val_loader, context.device, cfg.bf16, cfg.num_actions,
                class_weights, cfg.inflection_weight_coef,
                cfg.stop_aux_loss_weight if (
                    cfg.use_stop_aux_head or cfg.use_hierarchical_stop
                ) else 0.0,
                cfg.stop_aux_positive_weight,
                cfg.progress_loss_weight if cfg.use_progress_aux_head else 0.0,
                cfg.progress_smooth_l1_beta,
                cfg.transition_loss_weight if cfg.use_transition_aux_head else 0.0,
                openfly_action_codec,
            )
            if context.is_main:
                validation_record = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "validation_loss": round(metrics["validation"]["loss"], 6),
                    "validation_accuracy": round(metrics["validation"]["accuracy"], 6),
                    "validation_examples": int(metrics["validation"]["examples"]),
                }
                if "transition_loss" in metrics["validation"]:
                    validation_record.update({
                        "validation_transition_loss": round(
                            metrics["validation"]["transition_loss"], 6
                        ),
                        "validation_transition_accuracy": round(
                            metrics["validation"]["transition_accuracy"], 6
                        ),
                        "validation_transition_examples": int(
                            metrics["validation"]["transition_examples"]
                        ),
                    })
                if "action_accuracy" in metrics["validation"]:
                    validation_record.update({
                        "validation_action_accuracy": round(
                            metrics["validation"]["action_accuracy"], 6
                        ),
                        "validation_l1_loss": round(
                            metrics["validation"]["l1_loss"], 6
                        ),
                        "validation_action_token_examples": int(
                            metrics["validation"]["action_token_examples"]
                        ),
                        "validation_action_examples": int(
                            metrics["validation"]["action_examples"]
                        ),
                        "validation_codebook_exact_agreement": round(
                            metrics["validation"]["codebook_exact_agreement"], 6
                        ),
                        "validation_decoded_unmatched_rate": round(
                            metrics["validation"]["decoded_unmatched_rate"], 6
                        ),
                    })
                metrics_log.write(json.dumps(validation_record) + "\n")
                metrics_log.flush()
        barrier()
        checkpoint_extra = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "world_size": context.world_size,
            "training_contract": resume_contract,
            "rng_states": capture_ranked_rng_states(
                context.rank, context.world_size, context.distributed
            ),
        }
        if context.is_main:
            checkpoint_started = time.perf_counter()
            checkpoint_name = (
                f"plateau-step-{global_step:07d}"
                if stopped_for_plateau
                else f"epoch-{epoch + 1:03d}"
            )
            checkpoint = Path(cfg.output_dir) / checkpoint_name
            model.save_classifier_checkpoint(
                checkpoint,
                processor,
                optimizer=optimizer,
                scheduler=scheduler,
                extra=checkpoint_extra,
            )
            update_latest_symlink(checkpoint, Path(cfg.output_dir) / "latest")
            rotate_checkpoints(Path(cfg.output_dir), cfg.keep_last_checkpoints)
            with (checkpoint / "metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)
            stage1_profiler.add(
                "final_checkpoint_save", time.perf_counter() - checkpoint_started
            )
            print(json.dumps(metrics, indent=2))
        barrier()

        if cfg.max_train_steps is not None and global_step >= cfg.max_train_steps:
            break
        if stopped_for_plateau:
            break

    stage1_profiler.write(Path(cfg.output_dir), cfg, global_step)
    barrier()
    metrics_log.close()
    cleanup_distributed()


if __name__ == "__main__":
    main()
