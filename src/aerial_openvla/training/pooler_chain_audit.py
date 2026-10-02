"""Runtime evidence for the isolated fresh OpenFly 32T training chains."""
from collections import deque
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist


class ChainAudit:
    def __init__(self, model, cfg, context):
        self.model, self.cfg, self.context = model, cfg, context
        self.stage = os.environ["OPENFLY_CHAIN_STAGE"]
        self.output = Path(cfg.output_dir)
        self.total, self.count = 0.0, 0
        self.rolling = deque(maxlen=100)
        self.cumulative, self.cumulative_examples = 0.0, 0
        self.steps = 0
        self.epoch_start = None
        self.shapes = {}
        self.gradient_seen = set()
        self.handles = []
        self.pooler_name = ("instruction_local_pooler" if cfg.use_instruction_local_pooler
                            else "spatial_anchored_residual_pooler")
        assert cfg.visual_tokens_per_frame == 32 and cfg.num_frames == 1
        assert cfg.use_instruction_local_pooler != cfg.use_spatial_anchored_residual_pooler
        assert not cfg.use_task_aware_pooler
        assert cfg.openfly_action_policy is None and cfg.num_actions == 8
        assert not cfg.openfly_premerger_train_cache
        assert cfg.keep_last_checkpoints == 0
        if self.stage == "base":
            assert cfg.init_checkpoint is None and not cfg.use_native1t_postfusion
            assert cfg.native1t_train_cache is None and cfg.native1t_stage1_checkpoint is None
            assert model.native1t_memory_tokenizer is None
            pooler = getattr(model, self.pooler_name)
            zero = pooler.score_weight if cfg.use_instruction_local_pooler else pooler.alpha
            assert torch.count_nonzero(zero).item() == 0
        else:
            assert cfg.use_native1t_postfusion
            assert cfg.native1t_stage == (1 if self.stage == "stage1" else 2)
        self.required = ({"tokenizer", "memory_projector", "mem_bos"} if self.stage == "stage1"
                         else {"projector", "pooler", "lora", "head", "fuse", "action_query"})
        if self.stage == "stage2":
            self.required |= {"memory_projector", "mem_bos"}
        if getattr(cfg, "history_conditioned_local_pooler", False):
            if cfg.local_history_variant == "instruction_selection_residual":
                self.required.add("history_text_key")
            self.required |= ({"history_temporal", "history_projection"}
                              if cfg.local_history_variant == "instruction_selection" else
                              {"novelty_current", "novelty_history", "history_gamma"}
                              if cfg.local_history_variant == "novelty" else
                              {"history_temporal", "history_projection", "history_gamma"})
            if cfg.local_history_variant == "instruction_selection":
                assert not hasattr(model.instruction_local_pooler, "gamma")
            else:
                assert model.instruction_local_pooler.gamma.item() == 0
        for name, parameter in model.named_parameters():
            group = self.group(name)
            assert parameter.requires_grad == (group in self.required), (name, group, parameter.requires_grad)
        self.handles.append(model.register_forward_pre_hook(self.before_forward, with_kwargs=True))
        self.handles.append(getattr(model, self.pooler_name).register_forward_hook(self.pooler_hook))
        if model.native1t_memory_tokenizer is not None:
            self.handles.append(model.native1t_memory_tokenizer.register_forward_hook(self.memory_hook))
        self.write("initialization_audit.json", {
            "stage": self.stage, "init_checkpoint": cfg.init_checkpoint,
            "foundation": cfg.model_name_or_path, "pooler": self.pooler_name,
            "native1t_stage1_checkpoint": cfg.native1t_stage1_checkpoint,
            "trainable_groups": sorted(self.required), "fresh_optimizer": True,
            "physical_epochs": cfg.epochs, "scheduler_horizon_epochs": cfg.scheduler_horizon_epochs,
        })

    def group(self, name):
        if name.startswith("instruction_local_pooler.text_key_projection."):
            return "history_text_key"
        for prefix, group in (("cur_norm.", "novelty_current"), ("cur_proj.", "novelty_current"),
                              ("hist_norm.", "novelty_history"), ("hist_proj.", "novelty_history")):
            if name.startswith("instruction_local_pooler." + prefix):
                return group
        for prefix, group in (("instruction_local_pooler.temporal_mlp.", "history_temporal"),
                              ("instruction_local_pooler.history_projection.", "history_projection"),
                              ("instruction_local_pooler.gamma", "history_gamma")):
            if name.startswith(prefix):
                return group
        for prefix, group in (
            ("native1t_memory_tokenizer.", "tokenizer"),
            ("native1t_memory_projector.", "memory_projector"),
            ("backbone.projector.", "projector"),
            (self.pooler_name + ".", "pooler"),
            ("action_head.", "head"), ("fuse_proj.", "fuse"),
        ):
            if name.startswith(prefix):
                return group
        if "lora_" in name:
            return "lora"
        return {"native1t_mem_bos": "mem_bos", "action_query": "action_query"}.get(name, "frozen")

    def write(self, filename, value):
        if self.context.is_main:
            tmp = self.output / ("." + filename + ".tmp")
            tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
            tmp.replace(self.output / filename)

    def before_forward(self, module, args, kw):
        mask = kw.get("instruction_token_mask")
        assert mask is not None and mask.bool().any(dim=1).all().item(), "instruction-only mask missing"
        assert not torch.equal(mask.bool(), kw["attention_mask"].bool()), "prompt included in instruction pool"
        pixels = kw.get("pixel_values")
        memory_pixels = kw.get("native1t_memory_pixel_values")
        history = kw.get("native1t_memory_features")
        if self.stage == "base":
            assert memory_pixels is None and history is None
            assert pixels is not None and pixels.ndim in (4, 5)
            assert pixels.ndim == 4 or pixels.shape[1] == 1
        elif self.stage == "stage1":
            assert memory_pixels is not None and memory_pixels.shape[1] == 9
            assert history is None
        else:
            assert memory_pixels is None and history.shape[1:] == (16, 2176)
        source = memory_pixels if self.stage == "stage1" else pixels
        assert tuple(source.shape[-2:]) == (224, 224)
        self.shapes["image_batch"] = list(source.shape)
        self.shapes["history_slots"] = 0 if self.stage == "base" else (9 if self.stage == "stage1" else 16)

    def pooler_hook(self, module, args, output):
        features = args[0]
        assert tuple(features.shape[1:]) == (256, 896)
        assert tuple(output.shape[1:]) == (32, 896)
        self.shapes["current_pooler_input"] = list(features.shape)
        self.shapes["current_pooler_output"] = list(output.shape)

    def memory_hook(self, module, args, output):
        assert tuple(args[0].shape[-2:]) == (256, 2176), "memory must take backbone patches directly"
        assert output.shape == args[0].shape[:2] + (2176,)
        self.shapes["tokenizer_input"] = list(args[0].shape)
        self.shapes["tokenizer_output"] = list(output.shape)

    def start_epoch(self):
        self.epoch_start = time.monotonic()

    def record_micro(self, loss, examples):
        value = float(loss.detach().item())
        if not __import__("math").isfinite(value):
            raise RuntimeError("Non-finite training loss")
        self.total += value * examples
        self.count += examples

    def before_update(self, epoch, step):
        values = torch.tensor([self.total, self.count], device=self.context.device, dtype=torch.float64)
        if self.context.distributed:
            dist.all_reduce(values)
        loss_sum, count = values.tolist()
        self.total, self.count = 0.0, 0
        mean = loss_sum / count
        self.rolling.append((loss_sum, count))
        self.cumulative += loss_sum
        self.cumulative_examples += count
        elapsed = time.monotonic() - self.epoch_start
        record = {"epoch": epoch + 1, "global_step": step,
                  "optimizer_step_loss": mean, "examples": int(count),
                  "trailing_100_mean": sum(x[0] for x in self.rolling) / sum(x[1] for x in self.rolling),
                  "trailing_window_steps": len(self.rolling),
                  "cumulative_mean": self.cumulative / self.cumulative_examples,
                  "epoch_elapsed_seconds": elapsed}
        if self.context.is_main:
            with (self.output / "optimizer_loss.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
        if self.stage != "stage1" and elapsed >= 9000:
            self.write("epoch_time_failure.json", record)
            raise RuntimeError("Base/Stage2 epoch exceeded the user's 2.5-hour limit")
        if self.steps < 3:
            for name, parameter in self.model.named_parameters():
                grad = parameter.grad
                if not parameter.requires_grad:
                    assert grad is None, ("frozen parameter has gradient", name)
                elif grad is not None:
                    assert torch.isfinite(grad).all().item(), ("non-finite gradient", name)
                    if torch.count_nonzero(grad).item():
                        self.gradient_seen.add(self.group(name))
            self.steps += 1
            if self.steps == 3:
                assert self.required <= self.gradient_seen, ("missing gradient groups", self.required-self.gradient_seen)
                assert "current_pooler_output" in self.shapes
                if self.stage != "base":
                    assert "tokenizer_input" in self.shapes
                self.write("runtime_audit.json", {
                    "status": "PASS", "stage": self.stage, "ranks": self.context.world_size,
                    "shapes": self.shapes, "gradient_groups": sorted(self.gradient_seen),
                    "gradient_checked_optimizer_steps": 3,
                    "memory_source": "backbone_256_patches_before_visual_projector_and_current_pooler",
                    "current_tokens": 32, "history_slots": self.shapes["history_slots"],
                })
                for handle in self.handles:
                    handle.remove()


def maybe_audit(model, cfg, context):
    return ChainAudit(model, cfg, context) if os.environ.get("OPENFLY_CHAIN_STAGE") else None
