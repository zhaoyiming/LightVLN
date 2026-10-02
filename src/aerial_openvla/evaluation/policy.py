"""Evaluation-time policy interface and factory.

Every policy exposes a single-step ``act(image, instruction) → (action_id, probabilities)``
method used by the rollout loop.  The factory ``create_policy(cfg, device)`` dispatches on
``cfg.model_type`` so new model types can be plugged in without touching rollout.py.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any, Protocol

import torch
from PIL import Image

from aerial_openvla.openfly.action_codec import (
    OPENFLY_OBSERVED8_ACTION_IDS,
    OPENFLY_OBSERVED8_ACTION_POLICY,
)


def ensemble_action_chunk_probabilities(
    predictions: list[tuple[int, torch.Tensor]] | deque[tuple[int, torch.Tensor]],
    current_step: int,
    decay: float,
) -> torch.Tensor:
    """ACT-style exponential ensemble of overlapping categorical chunks."""
    if decay < 0:
        raise ValueError("decay must be non-negative")
    weighted: torch.Tensor | None = None
    weight_sum = 0.0
    for start_step, prediction in predictions:
        horizon_index = current_step - start_step
        if 0 <= horizon_index < prediction.shape[0]:
            weight = math.exp(-decay * horizon_index)
            contribution = prediction[horizon_index] * weight
            weighted = contribution if weighted is None else weighted + contribution
            weight_sum += weight
    if weighted is None or weight_sum <= 0:
        raise ValueError("No action chunks overlap the current step")
    return weighted / weight_sum


def select_hierarchical_action(
    navigation_probabilities: torch.Tensor,
    stop_probability: torch.Tensor,
    stop_threshold: float,
) -> int:
    """Apply a true STOP gate before the seven-way navigation argmax."""
    if navigation_probabilities.ndim != 1:
        raise ValueError("navigation_probabilities must be one-dimensional")
    if stop_probability.numel() != 1:
        raise ValueError("stop_probability must be scalar")
    if not 0.0 < stop_threshold < 1.0:
        raise ValueError("stop_threshold must be in (0, 1)")
    if float(stop_probability.item()) > stop_threshold:
        return 0
    return 1 + int(navigation_probabilities.argmax().item())


# ---------------------------------------------------------------------------
# Duck-typing protocol (documentation; no runtime enforcement)
# ---------------------------------------------------------------------------

class PolicyProtocol(Protocol):
    """Interface that every evaluation policy must satisfy."""

    def act(self, image: Image.Image, instruction: str) -> tuple[int, list[float]]:
        """Return (action_id, probabilities) for a single step."""
        ...


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_policy(cfg: Any, device: torch.device) -> Any:
    """Build an evaluation policy for the configured ``model_type``.

    Parameters
    ----------
    cfg : EvalConfig (or any object with matching fields)
        Evaluation configuration.  Must expose at least ``model_type``,
        ``model_name_or_path``, ``checkpoint_dir``, ``prompt_template``,
        ``max_text_tokens``, ``use_flash_attention_2``, and ``bf16``.
    device : torch.device
        CUDA device for inference.

    Returns
    -------
    policy : object
        Any object with an ``act(image: PIL.Image, instruction: str)
        → tuple[int, list[float]]`` method.
    """
    model_type = getattr(cfg, "model_type", "openvla")

    if model_type == "openvla":
        return OpenVLAClassifierPolicy(
            model_name_or_path=cfg.model_name_or_path,
            checkpoint_dir=cfg.checkpoint_dir,
            prompt_template=cfg.prompt_template,
            max_text_tokens=cfg.max_text_tokens,
            device=device,
            use_flash_attention_2=cfg.use_flash_attention_2,
            bf16=cfg.bf16,
        )

    if model_type == "airminivla":
        return AirMiniVLAClassifierPolicy(
            model_name_or_path=cfg.model_name_or_path,
            checkpoint_dir=cfg.checkpoint_dir,
            prompt_template=cfg.prompt_template,
            max_text_tokens=cfg.max_text_tokens,
            instruction_only_text_budget=getattr(
                cfg, "instruction_only_text_budget", False
            ),
            device=device,
            use_flash_attention_2=cfg.use_flash_attention_2,
            bf16=cfg.bf16,
            num_frames=getattr(cfg, "num_frames", 4),
            variant=getattr(cfg, "airminivla_variant", "m1"),
            use_prev_action=getattr(cfg, "use_prev_action", True),
            use_previous_actions_text=getattr(cfg, "use_previous_actions_text", False),
            use_relative_state_text=getattr(cfg, "use_relative_state_text", False),
            use_relative_planar_state_text=getattr(
                cfg, "use_relative_planar_state_text", False
            ),
            use_memory=getattr(cfg, "use_memory", False),
            visual_history_mode=getattr(cfg, "visual_history_mode", "none"),
            history_tokens_per_frame=getattr(cfg, "history_tokens_per_frame", 8),
            frame_history_stride=getattr(cfg, "frame_history_stride", 2),
            use_stop_aux_head=getattr(cfg, "use_stop_aux_head", False),
            stop_aux_logit_weight=getattr(cfg, "stop_aux_logit_weight", 1.0),
            use_frame_slot_embedding=getattr(cfg, "use_frame_slot_embedding", False),
            use_temporal_attention=getattr(cfg, "use_temporal_attention", False),
            temporal_attention_dim=getattr(cfg, "temporal_attention_dim", 256),
            temporal_attention_heads=getattr(cfg, "temporal_attention_heads", 4),
            temporal_attention_ffn_dim=getattr(cfg, "temporal_attention_ffn_dim", 1024),
            temporal_attention_dropout=getattr(cfg, "temporal_attention_dropout", 0.1),
            temporal_attention_initial_scale=getattr(
                cfg, "temporal_attention_initial_scale", 0.1
            ),
            use_motion_instruction_attention=getattr(
                cfg, "use_motion_instruction_attention", False
            ),
            motion_instruction_attention_heads=getattr(
                cfg, "motion_instruction_attention_heads", 4
            ),
            motion_instruction_attention_dropout=getattr(
                cfg, "motion_instruction_attention_dropout", 0.0
            ),
            action_chunk_horizon=getattr(cfg, "action_chunk_horizon", 1),
            action_chunk_temporal_ensemble=getattr(
                cfg, "action_chunk_temporal_ensemble", False
            ),
            action_chunk_temporal_ensemble_decay=getattr(
                cfg, "action_chunk_temporal_ensemble_decay", 0.5
            ),
            task_state_pointer_mode=getattr(
                cfg, "task_state_pointer_mode", "vanilla"
            ),
            use_temporal_memory=getattr(cfg, "use_temporal_memory", False),
            temporal_memory_num_history=getattr(cfg, "temporal_memory_num_history", 7),
            temporal_memory_attention_heads=getattr(
                cfg, "temporal_memory_attention_heads", 8
            ),
            use_native1t_postfusion=getattr(cfg, "use_native1t_postfusion", False),
            native1t_stage=getattr(cfg, "native1t_stage", 2),
            native1t_history_length=getattr(cfg, "native1t_history_length", 16),
            native1t_stage1_memory_frames=getattr(
                cfg, "native1t_stage1_memory_frames", 9
            ),
            use_native1t_action_history=getattr(
                cfg, "use_native1t_action_history", False
            ),
            native1t_action_history_length=getattr(
                cfg, "native1t_action_history_length", 4
            ),
            native1t_action_embedding_dim=getattr(
                cfg, "native1t_action_embedding_dim", 16
            ),
            use_h16_frozen_pool=getattr(cfg, "use_h16_frozen_pool", False),
            h16_frozen_pool_history_length=getattr(
                cfg, "h16_frozen_pool_history_length", 16
            ),
            use_omtrack=getattr(cfg, "use_omtrack", False),
            omtrack_history_length=getattr(cfg, "omtrack_history_length", 31),
            omtrack_history_tokens_per_frame=getattr(
                cfg, "omtrack_history_tokens_per_frame", 4
            ),
            omtrack_current_tokens=getattr(cfg, "omtrack_current_tokens", 64),
            omtrack_use_relative_time_embedding=getattr(
                cfg, "omtrack_use_relative_time_embedding", True
            ),
        )

    if model_type == "qwen35_nativemem":
        return Qwen35NativeMEMPolicy(
            model_name_or_path=cfg.model_name_or_path,
            checkpoint_dir=cfg.checkpoint_dir,
            prompt_template=cfg.prompt_template,
            max_text_tokens=cfg.max_text_tokens,
            device=device,
            use_flash_attention_2=cfg.use_flash_attention_2,
            bf16=cfg.bf16,
            relative_state_mode=getattr(
                cfg, "qwen35_relative_state_mode", "none"
            ),
        )

    if model_type == "qwen35_native":
        return Qwen35NativePolicy(
            model_name_or_path=cfg.model_name_or_path,
            checkpoint_dir=cfg.checkpoint_dir,
            prompt_template=cfg.prompt_template,
            max_text_tokens=cfg.max_text_tokens,
            device=device,
            use_flash_attention_2=cfg.use_flash_attention_2,
            bf16=cfg.bf16,
            relative_state_mode=getattr(
                cfg, "qwen35_relative_state_mode", "none"
            ),
        )

    if model_type == "cma":
        raise NotImplementedError(
            "CMA evaluation policy is not yet implemented. "
            "Set model_type='openvla' to use the OpenVLA-based policy."
        )

    raise ValueError(
        f"Unknown model_type={model_type!r}. "
        f"Supported types: openvla, airminivla, qwen35_native, qwen35_nativemem."
    )


# ---------------------------------------------------------------------------
# Built-in policies
# ---------------------------------------------------------------------------

class Qwen35NativePolicy:
    """Closed-loop policy for pure current-frame Qwen3.5 classifiers.

    It covers both the ordinary append-planar Relative experiment and the
    SegTS-v6 TaskGRU experiment.  Relative state is updated only after an
    action is predicted; TaskGRU state is reset at every episode boundary.
    """

    def __init__(
        self,
        *,
        model_name_or_path: str,
        checkpoint_dir: str | Path,
        prompt_template: str,
        max_text_tokens: int,
        device: torch.device,
        use_flash_attention_2: bool,
        bf16: bool,
        relative_state_mode: str = "none",
    ) -> None:
        from dataclasses import fields

        from aerial_openvla.data.qwen35_relative_state import (
            QWEN35_RELATIVE_NONE,
            validate_qwen35_relative_state_mode,
        )
        from aerial_openvla.models.qwen35_native_classifier import (
            Qwen35NativeBuildConfig,
            build_qwen35_native_classifier,
        )

        checkpoint = Path(checkpoint_dir).resolve()
        meta_path = checkpoint / "classifier_meta.pt"
        if not meta_path.is_file():
            raise FileNotFoundError(meta_path)
        checkpoint_meta = torch.load(
            meta_path, map_location="cpu", weights_only=False
        )
        allowed = {field.name for field in fields(Qwen35NativeBuildConfig)}
        build_args = {
            key: value for key, value in checkpoint_meta.items() if key in allowed
        }
        build_args.update(
            model_name_or_path=model_name_or_path,
            use_flash_attention_2=use_flash_attention_2,
            bf16=bf16,
        )
        relative_state_mode = validate_qwen35_relative_state_mode(
            relative_state_mode
        )
        checkpoint_relative = validate_qwen35_relative_state_mode(
            build_args.get("qwen35_relative_state_mode", QWEN35_RELATIVE_NONE)
        )
        if relative_state_mode != checkpoint_relative:
            raise RuntimeError(
                "Qwen3.5 Native Relative eval configuration does not match "
                f"checkpoint: config={relative_state_mode!r}, "
                f"checkpoint={checkpoint_relative!r}"
            )
        if bool(build_args.get("use_native1t_postfusion", False)):
            raise ValueError("Qwen35NativePolicy is current-frame only; use NativeMEM policy for H16")
        self.processor, self.model = build_qwen35_native_classifier(
            Qwen35NativeBuildConfig(**build_args), device=device
        )
        self.model.load_classifier_checkpoint(checkpoint)
        self.model.eval()
        self.device = device
        self.bf16 = bf16
        self.relative_state_mode = relative_state_mode
        self.uses_taskgru = self.model.segment_task_state is not None

        from aerial_openvla.data.qwen35_native_collator import (
            Qwen35NativeClassificationCollator,
        )

        self.collator = Qwen35NativeClassificationCollator(
            self.processor,
            prompt_template=prompt_template,
            max_text_tokens=max_text_tokens,
            qwen35_relative_state_mode=relative_state_mode,
            include_segment_token_ids=self.uses_taskgru,
            task_state_max_segments=int(
                build_args.get("task_state_max_segments", 10)
            ),
            # Formal training uses cached segments.  Eval instructions are
            # generated at runtime, so use the same deterministic segmenter.
            allow_online_segment_fallback=self.uses_taskgru,
        )
        self.reset()

    def reset(self) -> None:
        self.relative_heading_degrees = 0
        self.relative_altitude_meters = 0
        self.relative_forward_meters = 0.0
        self.relative_left_meters = 0.0
        self.segment_belief: torch.Tensor | None = None
        self.task_gru_state: torch.Tensor | None = None

    @torch.inference_mode()
    def act(self, image: Image.Image, instruction: str) -> tuple[int, list[float]]:
        sample: dict[str, Any] = {
            "image": image.convert("RGB"),
            "instruction": instruction,
            "action_id": 0,
            "episode_id": "inference",
            "step_id": 0,
            "image_path": "<simulator>",
        }
        if self.relative_state_mode != "none":
            sample.update({
                "relative_heading_degrees": self.relative_heading_degrees,
                "relative_altitude_meters": self.relative_altitude_meters,
                "relative_forward_meters": self.relative_forward_meters,
                "relative_left_meters": self.relative_left_meters,
            })
        batch = self.collator([sample])
        kwargs: dict[str, Any] = {
            key: batch[key].to(self.device)
            for key in (
                "input_ids", "attention_mask", "image_grid_thw",
                "mm_token_type_ids", "instruction_token_mask",
                "segment_token_ids",
            )
            if key in batch
        }
        kwargs["pixel_values"] = batch["pixel_values"].to(
            self.device, dtype=torch.bfloat16 if self.bf16 else torch.float32
        )
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            if self.uses_taskgru:
                output = self.model(
                    **kwargs,
                    segment_belief_prev=self.segment_belief,
                    task_gru_state_prev=self.task_gru_state,
                    return_auxiliary=True,
                )
                if not isinstance(output, dict):
                    raise RuntimeError("Qwen3.5 TaskGRU eval requires auxiliary outputs")
                logits = output["action_logits"]
                self.segment_belief = output["segment_belief"].detach()
                self.task_gru_state = output["task_gru_state"].detach()
            else:
                logits = self.model(**kwargs)
        probabilities = torch.softmax(logits.float(), dim=-1)[0]
        action_id = int(probabilities.argmax().item())
        if self.relative_state_mode != "none":
            from aerial_openvla.data.qwen35_relative_state import (
                update_qwen35_relative_state,
            )
            (
                self.relative_heading_degrees,
                self.relative_altitude_meters,
                self.relative_forward_meters,
                self.relative_left_meters,
            ) = update_qwen35_relative_state(
                self.relative_heading_degrees,
                self.relative_altitude_meters,
                self.relative_forward_meters,
                self.relative_left_meters,
                action_id,
            )
        return action_id, probabilities.cpu().tolist()

class Qwen35NativeMEMPolicy:
    """Closed-loop NativeMEM policy with episode-local predict-before-append H16."""

    def __init__(
        self, *, model_name_or_path: str, checkpoint_dir: str | Path,
        prompt_template: str, max_text_tokens: int, device: torch.device,
        use_flash_attention_2: bool, bf16: bool, relative_state_mode: str = "none",
    ) -> None:
        from dataclasses import fields
        from aerial_openvla.models.qwen35_nativemem_v2_classifier import (
            CONTRACT_VERSION,
            SINGLE_STAGE_V2_CONTRACT_VERSION,
            Qwen35NativeMEMV2BuildConfig,
            build_qwen35_nativemem_v2_classifier,
        )

        self.device=device; self.bf16=bf16; self.prompt_template=prompt_template
        self.max_text_tokens=max_text_tokens; self.history_length=16
        checkpoint=Path(checkpoint_dir)
        meta_path=checkpoint/'classifier_meta.pt'
        if not meta_path.is_file(): raise FileNotFoundError(meta_path)
        meta=torch.load(meta_path,map_location='cpu',weights_only=False)
        contract=meta.get('contract') or {}
        from aerial_openvla.data.qwen35_relative_state import (
            QWEN35_RELATIVE_NONE,
            validate_qwen35_relative_state_mode,
        )
        relative_state_mode = validate_qwen35_relative_state_mode(
            relative_state_mode
        )
        checkpoint_relative = (contract.get("relative_state") or {}).get(
            "mode", QWEN35_RELATIVE_NONE
        )
        if relative_state_mode != checkpoint_relative:
            raise RuntimeError(
                "Qwen3.5 Relative eval configuration does not match checkpoint: "
                f"config={relative_state_mode!r}, checkpoint={checkpoint_relative!r}"
            )
        version=contract.get('version')
        single_stage=bool(contract.get('single_stage', False))
        if contract.get('stage')!=2 or version not in {CONTRACT_VERSION, SINGLE_STAGE_V2_CONTRACT_VERSION}:
            raise RuntimeError('NativeMEM closed-loop eval requires a formal H16 NativeMEM checkpoint')
        if single_stage != (version == SINGLE_STAGE_V2_CONTRACT_VERSION):
            raise RuntimeError('NativeMEM checkpoint has inconsistent single-stage contract metadata')
        allowed={field.name for field in fields(Qwen35NativeMEMV2BuildConfig)}
        build_args={key:value for key,value in meta.items() if key in allowed}
        build_args.update(model_name_or_path=model_name_or_path,stage=2,single_stage=single_stage,use_flash_attention_2=use_flash_attention_2,bf16=bf16)
        # Training is cache-only and therefore records load_vision_backbone=false.
        # Closed-loop evaluation receives a new RGB observation at every step, so
        # it must restore the frozen ViT to compute the current pre-merger feature.
        # Do not mutate the training config/meta; override only this eval build.
        build_args["load_vision_backbone"] = True
        self.processor,self.model=build_qwen35_nativemem_v2_classifier(Qwen35NativeMEMV2BuildConfig(**build_args),device=device)
        self.model.load_classifier_checkpoint(checkpoint)
        self.model.eval(); self.memory_bank:deque[torch.Tensor]=deque(maxlen=self.history_length)
        self.relative_state_mode = relative_state_mode
        self.relative_heading_degrees = 0
        self.relative_altitude_meters = 0
        self.relative_forward_meters = 0.0
        self.relative_left_meters = 0.0

    def reset(self) -> None:
        self.memory_bank.clear()
        self.relative_heading_degrees = 0
        self.relative_altitude_meters = 0
        self.relative_forward_meters = 0.0
        self.relative_left_meters = 0.0

    @torch.inference_mode()
    def act(self,image:Image.Image,instruction:str)->tuple[int,list[float]]:
        rendered_instruction = instruction.strip()
        relative_state_suffix = ""
        if self.relative_state_mode != "none":
            from aerial_openvla.data.qwen35_relative_state import (
                append_qwen35_relative_state_text,
            )
            relative_state_suffix = append_qwen35_relative_state_text(
                "",
                self.relative_heading_degrees,
                self.relative_altitude_meters,
                self.relative_forward_meters,
                self.relative_left_meters,
            )
        prompt=self.prompt_template.format(instruction=rendered_instruction)
        # Keep the original Qwen prompt + Instruction at its trained budget.
        # Relative State is appended afterwards and does not consume it.
        tokenizer=getattr(self.processor,'tokenizer',self.processor)
        prompt_ids=tokenizer.encode(prompt,add_special_tokens=False)
        if len(prompt_ids)>self.max_text_tokens:
            prompt=tokenizer.decode(prompt_ids[:self.max_text_tokens],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        if relative_state_suffix:
            prompt=f"{prompt}{relative_state_suffix}"
        if self.relative_state_mode != "none" and "Current state:" not in prompt:
            raise RuntimeError(
                "Qwen3.5 append-planar Relative State was not appended after "
                "the Instruction text budget was applied"
            )
        conversations=[[{'role':'user','content':[{'type':'image','image':image.convert('RGB')},{'type':'text','text':prompt}]}]]
        text=self.processor.apply_chat_template(conversations,tokenize=True,add_generation_prompt=False,return_dict=True,return_tensors='pt',processor_kwargs={'padding':True,'truncation':False})
        dtype=torch.bfloat16 if self.bf16 else torch.float32
        history=torch.zeros(1,self.history_length,1024,device=self.device,dtype=dtype)
        history_mask=torch.zeros(1,self.history_length,device=self.device,dtype=torch.bool)
        cached=list(self.memory_bank); first=self.history_length-len(cached)
        if cached:
            history[0,first:]=torch.cat(cached,dim=0).to(dtype=dtype)
            history_mask[0,first:]=True
        # The current frame is absent from history here.  It is appended only
        # after the action logits have been computed.
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.bf16):
            logits,current_memory=self.model(
                input_ids=text['input_ids'].to(self.device),
                attention_mask=text['attention_mask'].to(self.device),
                mm_token_type_ids=text['mm_token_type_ids'].to(self.device),
                nativemem_history_valid_mask=history_mask,
                nativemem_history_tokens=history,
                pixel_values=text['pixel_values'].to(self.device,dtype=dtype),
                image_grid_thw=text['image_grid_thw'].to(self.device),
                return_nativemem_memory=True,
            )
        probabilities=torch.softmax(logits.float(),dim=-1)[0]
        action_id=int(probabilities.argmax().item())
        self.memory_bank.append(current_memory.detach())
        if self.relative_state_mode != "none":
            from aerial_openvla.data.qwen35_relative_state import (
                update_qwen35_relative_state,
            )
            (
                self.relative_heading_degrees,
                self.relative_altitude_meters,
                self.relative_forward_meters,
                self.relative_left_meters,
            ) = update_qwen35_relative_state(
                self.relative_heading_degrees,
                self.relative_altitude_meters,
                self.relative_forward_meters,
                self.relative_left_meters,
                action_id,
            )
        return action_id,probabilities.cpu().tolist()

class OpenVLAClassifierPolicy:
    """OpenVLA-based action classifier for evaluation rollouts."""

    def __init__(
        self,
        *,
        model_name_or_path: str,
        checkpoint_dir: str | Path,
        prompt_template: str,
        max_text_tokens: int,
        device: torch.device,
        use_flash_attention_2: bool,
        bf16: bool,
    ) -> None:
        # Accelerate small-batch matmul (TF32 uses Tensor Cores for fp32 matmul) &
        # enable cuDNN autotuner for fastest convolution algorithms.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        from aerial_openvla.data.collator import OpenVLAClassificationCollator
        from aerial_openvla.models.openvla_classifier import ClassifierBuildConfig, build_classifier

        checkpoint = Path(checkpoint_dir).resolve()
        lora_dir = checkpoint / "lora"
        config = ClassifierBuildConfig(
            model_name_or_path=model_name_or_path,
            num_actions=8,
            use_lora=False,
            train_projector=False,
            freeze_vision=True,
            gradient_checkpointing=False,
            use_flash_attention_2=use_flash_attention_2,
            bf16=bf16,
        )
        self.processor, self.model = build_classifier(
            config,
            device=device,
            lora_adapter_dir=lora_dir if lora_dir.is_dir() else None,
        )
        self.model.load_classifier_checkpoint(checkpoint)
        self.model.to(device).eval()

        self.collator = OpenVLAClassificationCollator(
            self.processor, prompt_template=prompt_template, max_text_tokens=max_text_tokens
        )
        self.device = device
        self.bf16 = bf16

        # Prime CUDA context, kernel cache, and cuDNN autotuner so the first
        # real evaluation step doesn"t incur cold-start latency.
        _dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            _ = self.act(_dummy, "warmup")

    @torch.inference_mode()
    def act(self, image: Image.Image, instruction: str) -> tuple[int, list[float]]:
        batch = self.collator(
            [
                {
                    "image": image,
                    "instruction": instruction,
                    "action_id": 0,
                    "episode_id": "inference",
                    "step_id": 0,
                    "image_path": "<simulator>",
                }
            ]
        )
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        pixel_values = batch["pixel_values"].to(
            self.device, dtype=torch.bfloat16 if self.bf16 else torch.float32
        )
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )
        probabilities = torch.softmax(logits.float(), dim=-1)[0]
        action_id = int(probabilities.argmax().item())
        return action_id, probabilities.cpu().tolist()


class AirMiniVLAClassifierPolicy:
    """AirMiniVLA-based action classifier for evaluation rollouts.

    Maintains a frame buffer for multi-frame inference and tracks the
    previous action for the next step.
    """

    def __init__(
        self,
        *,
        model_name_or_path: str,
        checkpoint_dir: str | Path,
        prompt_template: str,
        max_text_tokens: int,
        instruction_only_text_budget: bool = False,
        device: torch.device,
        use_flash_attention_2: bool,
        bf16: bool,
        num_frames: int = 4,
        use_prev_action: bool = True,
        use_previous_actions_text: bool = False,
        use_relative_state_text: bool = False,
        use_relative_planar_state_text: bool = False,
        use_memory: bool = False,
        visual_history_mode: str = "none",
        history_tokens_per_frame: int = 8,
        variant: str = "m1",
        frame_history_stride: int = 2,
        use_stop_aux_head: bool = False,
        stop_aux_logit_weight: float = 1.0,
        use_frame_slot_embedding: bool = False,
        use_temporal_attention: bool = False,
        temporal_attention_dim: int = 256,
        temporal_attention_heads: int = 4,
        temporal_attention_ffn_dim: int = 1024,
        temporal_attention_dropout: float = 0.1,
        temporal_attention_initial_scale: float = 0.1,
        use_motion_instruction_attention: bool = False,
        motion_instruction_attention_heads: int = 4,
        motion_instruction_attention_dropout: float = 0.0,
        action_chunk_horizon: int = 1,
        action_chunk_temporal_ensemble: bool = False,
        action_chunk_temporal_ensemble_decay: float = 0.5,
        task_state_pointer_mode: str = "vanilla",
        use_temporal_memory: bool = False,
        temporal_memory_num_history: int = 7,
        temporal_memory_attention_heads: int = 8,
        use_native1t_postfusion: bool = False,
        native1t_stage: int = 2,
        native1t_history_length: int = 16,
        native1t_stage1_memory_frames: int = 9,
        use_native1t_action_history: bool = False,
        native1t_action_history_length: int = 4,
        native1t_action_embedding_dim: int = 16,
        use_h16_frozen_pool: bool = False,
        h16_frozen_pool_history_length: int = 16,
        use_omtrack: bool = False,
        omtrack_history_length: int = 31,
        omtrack_history_tokens_per_frame: int = 4,
        omtrack_current_tokens: int = 64,
        omtrack_use_relative_time_embedding: bool = True,
    ) -> None:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        from aerial_openvla.models.airminivla_classifier import (
            AirMiniVLAConfig,
            build_airminivla_classifier,
        )

        checkpoint = Path(checkpoint_dir).resolve()
        lora_dir = checkpoint / "lora"

        config = AirMiniVLAConfig(
            model_name_or_path=model_name_or_path,
            variant=variant,
            num_actions=8,
            num_frames=num_frames,
            use_prev_action=use_prev_action,
            use_previous_actions_text=use_previous_actions_text,
            use_relative_state_text=use_relative_state_text,
            use_relative_planar_state_text=use_relative_planar_state_text,
            instruction_only_text_budget=instruction_only_text_budget,
            use_memory=use_memory,
            visual_history_mode=visual_history_mode,
            history_tokens_per_frame=history_tokens_per_frame,
            use_stop_aux_head=use_stop_aux_head,
            stop_aux_logit_weight=stop_aux_logit_weight,
            use_frame_slot_embedding=use_frame_slot_embedding,
            use_temporal_attention=use_temporal_attention,
            temporal_attention_dim=temporal_attention_dim,
            temporal_attention_heads=temporal_attention_heads,
            temporal_attention_ffn_dim=temporal_attention_ffn_dim,
            temporal_attention_dropout=temporal_attention_dropout,
            temporal_attention_initial_scale=temporal_attention_initial_scale,
            use_motion_instruction_attention=use_motion_instruction_attention,
            motion_instruction_attention_heads=motion_instruction_attention_heads,
            motion_instruction_attention_dropout=motion_instruction_attention_dropout,
            action_chunk_horizon=action_chunk_horizon,
            task_state_pooling_mode="mean",
            use_temporal_memory=use_temporal_memory,
            temporal_memory_num_history=temporal_memory_num_history,
            temporal_memory_attention_heads=temporal_memory_attention_heads,
            use_native1t_postfusion=use_native1t_postfusion,
            native1t_stage=native1t_stage,
            native1t_history_length=native1t_history_length,
            native1t_stage1_memory_frames=native1t_stage1_memory_frames,
            use_native1t_action_history=use_native1t_action_history,
            native1t_action_history_length=native1t_action_history_length,
            native1t_action_embedding_dim=native1t_action_embedding_dim,
            use_h16_frozen_pool=use_h16_frozen_pool,
            h16_frozen_pool_history_length=h16_frozen_pool_history_length,
            use_omtrack=use_omtrack,
            omtrack_history_length=omtrack_history_length,
            omtrack_history_tokens_per_frame=omtrack_history_tokens_per_frame,
            omtrack_current_tokens=omtrack_current_tokens,
            omtrack_use_relative_time_embedding=omtrack_use_relative_time_embedding,
            freeze_vision=True,
            train_projector=False,
            use_lora=False,
            gradient_checkpointing=False,
            use_flash_attention_2=use_flash_attention_2,
            bf16=bf16,
        )

        # The builder applies the checkpoint's classifier_meta.pt over the
        # eval config before constructing the model, then loads the weights —
        # evaluation always rebuilds the exact trained architecture.
        self.processor, self.model = build_airminivla_classifier(
            config,
            device=device,
            lora_adapter_dir=lora_dir if lora_dir.is_dir() else None,
            checkpoint_dir=checkpoint,
        )
        self.model.to(device).eval()

        self.prompt_template = prompt_template
        self.max_text_tokens = max_text_tokens
        self.device = device
        self.bf16 = bf16
        # Runtime structure follows the model actually built (checkpoint meta
        # may have overridden the eval config), never the raw eval config.
        model_config = self.model.config
        self.use_instruction_local_pooler = bool(
            getattr(model_config, "use_instruction_local_pooler", False)
        )
        self.use_spatial_anchored_residual_pooler = bool(
            getattr(model_config, "use_spatial_anchored_residual_pooler", False)
        )
        self.num_actions = int(model_config.num_actions)
        self.openfly_action_codec = None
        self.openfly_action_vocab_size: int | None = None
        raw_openfly_action_policy = getattr(model_config, "openfly_action_policy", None)
        self.openfly_action_policy = (
            None if raw_openfly_action_policy is None else str(raw_openfly_action_policy)
        )
        self.openfly_action_head_type = str(
            getattr(model_config, "openfly_action_head_type", "categorical")
        )
        if self.openfly_action_head_type in {"action_tokens_v1", "action_tokens_ar_v2"}:
            from aerial_openvla.openfly.action_codec import (
                OPENFLY_ACTION_BINS,
                OPENFLY_ACTION_DIM,
                load_openfly_action_codec,
            )

            if (
                self.num_actions != 10
                or int(model_config.openfly_action_token_dim) != OPENFLY_ACTION_DIM
                or int(model_config.openfly_action_token_bins) != OPENFLY_ACTION_BINS
            ):
                raise ValueError(
                    "OpenFly action-token checkpoint has an incompatible logical "
                    "action/code shape"
                )
            self.openfly_action_codec = load_openfly_action_codec(
                checkpoint / "openfly_action_codec.json",
                expected_sha256=model_config.openfly_action_codec_sha256,
            )
            if self.openfly_action_head_type == "action_tokens_ar_v2":
                vocab_size = getattr(self.model, "_openfly_action_vocab_size", None)
                if not isinstance(vocab_size, int) or vocab_size <= OPENFLY_ACTION_BINS:
                    raise RuntimeError(
                        "OpenFly autoregressive checkpoint did not rebuild a valid action vocabulary"
                    )
                self.openfly_action_vocab_size = vocab_size
        elif self.openfly_action_head_type != "categorical":
            raise ValueError(
                "Unknown AirMiniVLA action head: "
                f"{self.openfly_action_head_type!r}"
            )
        elif self.openfly_action_policy in {
            "openfly_10",
            OPENFLY_OBSERVED8_ACTION_POLICY,
        }:
            expected_num_actions = (
                10
                if self.openfly_action_policy == "openfly_10"
                else len(OPENFLY_OBSERVED8_ACTION_IDS)
            )
            incompatible = {
                "num_actions": self.num_actions != expected_num_actions,
                "action-token dimension": int(model_config.openfly_action_token_dim) != 0,
                "action-token bins": int(model_config.openfly_action_token_bins) != 0,
                "action codec digest": bool(model_config.openfly_action_codec_sha256),
            }
            enabled = [name for name, value in incompatible.items() if value]
            if enabled:
                raise ValueError(
                    "OpenFly direct categorical checkpoint has an incompatible "
                    "direct-action "
                    "contract: " + ", ".join(enabled)
                )
        self.prev_action_start_id = int(model_config.prev_action_vocab) - 1
        if model_config.use_prev_action and self.prev_action_start_id != self.num_actions:
            raise ValueError(
                "Previous-action vocabulary must reserve its final ID as the "
                f"start token: num_actions={self.num_actions}, "
                f"prev_action_vocab={model_config.prev_action_vocab}"
            )
        self.num_frames = model_config.num_frames
        self.instruction_only_text_budget = bool(
            model_config.instruction_only_text_budget
        )
        self.use_prev_action = model_config.use_prev_action
        self.use_previous_actions_text = model_config.use_previous_actions_text
        self.use_relative_state_text = model_config.use_relative_state_text
        self.use_relative_planar_state_text = (
            model_config.use_relative_planar_state_text
        )
        self.use_memory = model_config.use_memory
        self.visual_history_mode = model_config.visual_history_mode
        self.use_text_conditioned_stop = (
            model_config.use_hierarchical_stop
            and model_config.hierarchical_stop_mode == "text_query"
        )
        self.use_motion_instruction_attention = (
            model_config.use_motion_instruction_attention
        )
        self.task_state_mode = model_config.task_state_mode
        self.task_state_max_segments = model_config.task_state_max_segments
        self.task_state_pooling_mode = model_config.task_state_pooling_mode
        self.task_state_pointer_mode = str(task_state_pointer_mode).lower()
        if self.task_state_pointer_mode not in {"vanilla", "hardptr"}:
            raise ValueError(
                "task_state_pointer_mode must be 'vanilla' or 'hardptr'"
            )
        if self.task_state_pointer_mode == "hardptr":
            incompatible = {
                "task_state_mode": self.task_state_mode != "segment_only",
                "Sticky belief": bool(model_config.task_state_sticky_belief),
                "TaskGRU": bool(model_config.task_state_use_gru),
                "HierIA": bool(model_config.task_state_hierarchical_ia),
                "Relative text": bool(model_config.use_relative_state_text),
                "Native1T": bool(model_config.use_native1t_postfusion),
                "action chunking": model_config.action_chunk_horizon != 1,
            }
            enabled = [name for name, value in incompatible.items() if value]
            if enabled:
                raise ValueError(
                    "HardPtr evaluator is incompatible with: " + ", ".join(enabled)
                )
        self.use_task_gru = bool(model_config.task_state_use_gru)
        self.action_chunk_horizon = model_config.action_chunk_horizon
        self.use_hierarchical_stop = bool(model_config.use_hierarchical_stop)
        self.hierarchical_stop_threshold = float(
            model_config.hierarchical_stop_threshold
        )
        self.use_temporal_memory = model_config.use_temporal_memory
        self.temporal_memory_num_history = model_config.temporal_memory_num_history
        self.use_native1t_postfusion = model_config.use_native1t_postfusion
        self.native1t_history_length = model_config.native1t_history_length
        self.native1t_history_mode = getattr(
            model_config, "native1t_history_mode", "fifo"
        )
        self.use_native1t_action_history = bool(
            model_config.use_native1t_action_history
        )
        self.native1t_action_history_length = (
            model_config.native1t_action_history_length
        )
        if self.use_native1t_postfusion and model_config.native1t_stage != 2:
            raise ValueError("Closed-loop Native1T evaluation requires a Stage-2 checkpoint")
        if self.use_native1t_postfusion and self.native1t_history_mode not in {
            "fifo", "event_h16_v1"
        }:
            raise ValueError(
                f"Unknown Native1T history mode: {self.native1t_history_mode!r}"
            )
        self.native1t_event_manager = None
        self.native1t_event_step = 0
        if (
            self.use_native1t_postfusion
            and self.native1t_history_mode == "event_h16_v1"
        ):
            from aerial_openvla.data.event_h16 import EventMemoryManager

            self.native1t_event_manager = EventMemoryManager(
                tau_local=float(model_config.native1t_event_tau_local),
                tau_anchor=float(model_config.native1t_event_tau_anchor),
                max_event_length=int(model_config.native1t_event_max_length),
                max_nodes=int(model_config.native1t_event_max_nodes),
                recent_protected=int(model_config.native1t_event_recent_protected),
            )
        self.use_h16_frozen_pool = model_config.use_h16_frozen_pool
        self.h16_frozen_pool_history_length = model_config.h16_frozen_pool_history_length
        self.use_omtrack = model_config.use_omtrack
        self.omtrack_history_length = model_config.omtrack_history_length
        self.action_chunk_temporal_ensemble = bool(action_chunk_temporal_ensemble)
        self.action_chunk_temporal_ensemble_decay = float(
            action_chunk_temporal_ensemble_decay
        )
        if self.action_chunk_temporal_ensemble and self.action_chunk_horizon <= 1:
            raise ValueError("Temporal ensembling requires an action-chunk checkpoint")
        if self.action_chunk_temporal_ensemble_decay < 0:
            raise ValueError("action_chunk_temporal_ensemble_decay must be non-negative")
        self.frame_history_stride = frame_history_stride

        from aerial_openvla.data.multiframe_dataset import MultiFrameCollator

        self.collator = MultiFrameCollator(
            self.processor,
            prompt_template=prompt_template,
            max_text_tokens=max_text_tokens,
            instruction_only_text_budget=self.instruction_only_text_budget,
            num_frames=self.num_frames,
            include_instruction_token_mask=(
                self.use_text_conditioned_stop or self.use_motion_instruction_attention
                or self.task_state_mode != "none"
                or self.use_instruction_local_pooler
                or self.use_spatial_anchored_residual_pooler
            ),
            include_segment_token_ids=self.task_state_mode != "none",
            task_state_max_segments=self.task_state_max_segments,
            use_previous_actions_text=self.use_previous_actions_text,
            use_relative_state_text=self.use_relative_state_text,
            use_relative_planar_state_text=self.use_relative_planar_state_text,
            action_chunk_horizon=self.action_chunk_horizon,
            visual_history_mode=self.visual_history_mode,
        )

        # Stateful evaluation
        self.frame_buffer: deque[Image.Image] = deque(
            maxlen=(self.num_frames - 1) * frame_history_stride + 1
        )
        self.vision_feature_buffer: deque[torch.Tensor] = deque(
            maxlen=(
                self.omtrack_history_length
                if self.use_omtrack
                else (
                    self.h16_frozen_pool_history_length
                    if self.use_h16_frozen_pool
                    else (
                    self.temporal_memory_num_history
                    if self.use_temporal_memory
                    else (self.native1t_history_length if self.use_native1t_postfusion else 3)
                    )
                )
            )
        )
        self.temporal_memory_buffer: deque[torch.Tensor] = deque(
            maxlen=self.temporal_memory_num_history
        )
        self.native1t_memory_buffer: deque[torch.Tensor] = deque(
            maxlen=self.native1t_history_length
        )
        self.native1t_action_buffer: deque[int] = deque(
            maxlen=self.native1t_action_history_length
        )
        self.prev_action: int = self.prev_action_start_id
        self.previous_actions: deque[int] = deque(maxlen=3)
        self.relative_heading_degrees = 0
        self.relative_altitude_meters = 0
        self.relative_forward_meters = 0.0
        self.relative_left_meters = 0.0
        self.memory_state: torch.Tensor | None = None
        self.last_task_state: dict[str, object] | None = None
        self.segment_belief: torch.Tensor | None = None
        self.current_stage = 0
        self.task_gru_state: torch.Tensor | None = None
        self.action_chunk_step = 0
        self.action_chunk_predictions: deque[tuple[int, torch.Tensor]] = deque(
            maxlen=max(self.action_chunk_horizon, 1)
        )
        self.last_action_chunk_probabilities: list[list[float]] | None = None
        self.last_action_chunk_slot0_probabilities: list[float] | None = None
        self.last_action_chunk_navigation_probabilities: list[list[float]] | None = None
        self.last_action_chunk_stop_probabilities: list[float] | None = None
        self.last_stop_probability: float | None = None
        self.last_openfly_action: dict[str, object] | None = None

        # Warmup
        _dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
        self.reset()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            _ = self.act(_dummy, "warmup")
        self.reset()

    def reset(self) -> None:
        """Reset state for a new episode."""
        self.frame_buffer.clear()
        self.vision_feature_buffer.clear()
        self.temporal_memory_buffer.clear()
        self.native1t_memory_buffer.clear()
        if self.native1t_event_manager is not None:
            self.native1t_event_manager.reset()
            self.native1t_event_step = 0
        self.native1t_action_buffer.clear()
        self.prev_action = self.prev_action_start_id
        self.previous_actions.clear()
        self.relative_heading_degrees = 0
        self.relative_altitude_meters = 0
        self.relative_forward_meters = 0.0
        self.relative_left_meters = 0.0
        self.memory_state = None
        self.last_task_state = None
        self.segment_belief = None
        self.current_stage = 0
        self.task_gru_state = None
        self.action_chunk_step = 0
        self.action_chunk_predictions.clear()
        self.last_action_chunk_probabilities = None
        self.last_action_chunk_slot0_probabilities = None
        self.last_action_chunk_navigation_probabilities = None
        self.last_action_chunk_stop_probabilities = None
        self.last_stop_probability = None
        self.last_openfly_action = None

    def record_executed_action(self, action_id: int) -> None:
        """Commit an action to AH4 only after the environment executes it."""
        if self.use_native1t_action_history:
            self.native1t_action_buffer.append(int(action_id))

    @torch.inference_mode()
    def act(self, image: Image.Image, instruction: str) -> tuple[int, list[float]]:
        image_processor = getattr(self.processor, "image_processor", self.processor)
        frame_batch_indices = None
        frame_slot_indices = None
        precomputed_vision_features = None
        current_feature = None
        cached_memory_tokens = None
        history_valid_mask = None
        omtrack_history_features = None
        omtrack_history_valid_mask = None
        h16_frozen_pool_history_features = None
        h16_frozen_pool_history_valid_mask = None
        native1t_memory_features = None
        native1t_memory_valid_mask = None
        native1t_action_history = None
        native1t_action_history_valid_mask = None
        if self.use_native1t_postfusion:
            encoded = image_processor([image], return_tensors="pt")
            pixel_values = encoded["pixel_values"].to(
                self.device, dtype=torch.bfloat16 if self.bf16 else torch.float32
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
                current_feature = self.model.encode_frozen_vision(pixel_values)
            cached = (
                self.native1t_event_manager.get_tokens()
                if self.native1t_event_manager is not None
                else list(self.native1t_memory_buffer)
            )
            native1t_memory_features = current_feature.new_zeros(
                1, self.native1t_history_length, self.model._vision_feature_size
            )
            native1t_memory_valid_mask = torch.zeros(
                1, self.native1t_history_length, dtype=torch.bool, device=self.device
            )
            first_slot = self.native1t_history_length - len(cached)
            if cached:
                native1t_memory_features[0, first_slot:] = torch.cat(cached, dim=0)
                native1t_memory_valid_mask[0, first_slot:] = True
            precomputed_vision_features = current_feature
            frame_valid_mask = None
            if self.use_native1t_action_history:
                action_history = list(self.native1t_action_buffer)
                native1t_action_history = torch.zeros(
                    1,
                    self.native1t_action_history_length,
                    dtype=torch.long,
                    device=self.device,
                )
                native1t_action_history_valid_mask = torch.zeros(
                    1,
                    self.native1t_action_history_length,
                    dtype=torch.bool,
                    device=self.device,
                )
                action_first_slot = (
                    self.native1t_action_history_length - len(action_history)
                )
                if action_history:
                    native1t_action_history[0, action_first_slot:] = torch.tensor(
                        action_history, dtype=torch.long, device=self.device
                    )
                    native1t_action_history_valid_mask[0, action_first_slot:] = True
        elif self.use_h16_frozen_pool:
            encoded = image_processor([image], return_tensors="pt")
            pixel_values = encoded["pixel_values"].to(
                self.device, dtype=torch.bfloat16 if self.bf16 else torch.float32
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
                current_feature = self.model.encode_frozen_vision(pixel_values)
            cached = list(self.vision_feature_buffer)
            h16_frozen_pool_history_features = current_feature.new_zeros(
                1, self.h16_frozen_pool_history_length, self.model._vision_feature_size
            )
            h16_frozen_pool_history_valid_mask = torch.zeros(
                1, self.h16_frozen_pool_history_length, dtype=torch.bool, device=self.device
            )
            first_slot = self.h16_frozen_pool_history_length - len(cached)
            if cached:
                h16_frozen_pool_history_features[0, first_slot:] = torch.cat(cached, dim=0)
                h16_frozen_pool_history_valid_mask[0, first_slot:] = True
            precomputed_vision_features = current_feature
            frame_valid_mask = None
        elif self.use_omtrack:
            encoded = image_processor([image], return_tensors="pt")
            pixel_values = encoded["pixel_values"].to(
                self.device, dtype=torch.bfloat16 if self.bf16 else torch.float32
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
                current_feature = self.model.encode_frozen_vision(pixel_values)
            cached = list(self.vision_feature_buffer)
            omtrack_history_features = current_feature.new_zeros(
                1,
                self.omtrack_history_length,
                self.model.config.omtrack_history_tokens_per_frame,
                self.model._vision_feature_size,
            )
            omtrack_history_valid_mask = torch.zeros(
                1, self.omtrack_history_length, dtype=torch.bool, device=self.device
            )
            first_slot = self.omtrack_history_length - len(cached)
            if cached:
                omtrack_history_features[0, first_slot:] = torch.cat(cached, dim=0)
                omtrack_history_valid_mask[0, first_slot:] = True
            precomputed_vision_features = current_feature
            frame_valid_mask = None
        elif self.use_temporal_memory:
            encoded = image_processor([image], return_tensors="pt")
            pixel_values = encoded["pixel_values"].to(
                self.device, dtype=torch.bfloat16 if self.bf16 else torch.float32
            )
            cached = list(self.temporal_memory_buffer)
            cached_memory_tokens = pixel_values.new_zeros(
                1,
                self.temporal_memory_num_history,
                self.model._vision_feature_size,
            )
            history_valid_mask = torch.zeros(
                1,
                self.temporal_memory_num_history,
                dtype=torch.bool,
                device=self.device,
            )
            first_slot = self.temporal_memory_num_history - len(cached)
            if cached:
                cached_memory_tokens[0, first_slot:] = torch.cat(cached, dim=0)
                history_valid_mask[0, first_slot:] = True
            frame_valid_mask = None
        elif self.visual_history_mode in {"recent3", "keyframe3", "motion_recent3"}:
            encoded = image_processor([image], return_tensors="pt")
            pixel_values = encoded["pixel_values"].to(
                self.device, dtype=torch.bfloat16 if self.bf16 else torch.float32
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
                current_feature = self.model.encode_frozen_vision(pixel_values)
            cached = list(self.vision_feature_buffer)
            precomputed_vision_features = torch.cat([*cached, current_feature], dim=0)
            first_slot = self.num_frames - precomputed_vision_features.shape[0]
            frame_slot_indices = torch.arange(
                first_slot, self.num_frames, dtype=torch.long, device=self.device
            )
            frame_batch_indices = torch.zeros_like(frame_slot_indices)
            frame_valid_mask = torch.zeros(1, self.num_frames, dtype=torch.bool, device=self.device)
            frame_valid_mask[0, frame_slot_indices] = True
        else:
            self.frame_buffer.append(image)
            available = list(self.frame_buffer)
            current_index = len(available) - 1
            relative_offsets = [
                -self.frame_history_stride * (self.num_frames - 1 - index)
                for index in range(self.num_frames)
            ]
            frames: list[Image.Image] = []
            frame_valid: list[bool] = []
            for offset in relative_offsets:
                source_index = current_index + offset
                is_valid = source_index >= 0
                frames.append(available[source_index] if is_valid else available[0])
                frame_valid.append(is_valid)
            frame_valid_mask = torch.tensor([frame_valid], dtype=torch.bool, device=self.device)
            all_pixel_values: list[torch.Tensor] = []
            for frame in frames:
                encoded = image_processor([frame], return_tensors="pt")
                pv = encoded["pixel_values"]
                if pv.dim() == 4:
                    pv = pv.unsqueeze(1)
                all_pixel_values.append(pv)
            pixel_values = torch.cat(all_pixel_values, dim=1)

        # Encode text
        from aerial_openvla.data.multiframe_dataset import (
            truncate_instruction_to_token_budget,
        )

        source_instruction = instruction.strip()
        if self.instruction_only_text_budget:
            source_instruction = truncate_instruction_to_token_budget(
                self.processor.tokenizer, source_instruction, self.max_text_tokens
            )
        rendered_instruction = source_instruction
        if self.use_previous_actions_text:
            from aerial_openvla.data.multiframe_dataset import append_previous_actions_text

            rendered_instruction = append_previous_actions_text(
                rendered_instruction, list(self.previous_actions)
            )
        elif self.use_relative_state_text:
            from aerial_openvla.data.multiframe_dataset import prepend_relative_state_text

            rendered_instruction = prepend_relative_state_text(
                rendered_instruction,
                self.relative_heading_degrees,
                self.relative_altitude_meters,
                (
                    self.relative_forward_meters
                    if self.use_relative_planar_state_text
                    else None
                ),
                (
                    self.relative_left_meters
                    if self.use_relative_planar_state_text
                    else None
                ),
            )
        prompt = self.prompt_template.format(instruction=rendered_instruction)
        needs_instruction_mask = (
            self.use_text_conditioned_stop or self.use_motion_instruction_attention
            or self.task_state_mode != "none"
            or self.use_instruction_local_pooler
            or self.use_spatial_anchored_residual_pooler
        )
        if self.use_previous_actions_text:
            from aerial_openvla.data.multiframe_dataset import tokenize_s2_prompts

            text_encoded = tokenize_s2_prompts(
                self.processor.tokenizer,
                [prompt],
                max_length=self.max_text_tokens,
                return_offsets_mapping=needs_instruction_mask,
            )
        else:
            tokenizer_kwargs = dict(
                padding=True,
                truncation=not self.instruction_only_text_budget,
                return_tensors="pt",
            )
            if not self.instruction_only_text_budget:
                tokenizer_kwargs["max_length"] = self.max_text_tokens
            if needs_instruction_mask:
                tokenizer_kwargs["return_offsets_mapping"] = True
            text_encoded = self.processor.tokenizer([prompt], **tokenizer_kwargs)
        offsets = text_encoded.pop("offset_mapping", None)

        input_ids = text_encoded["input_ids"].to(self.device)
        attention_mask = text_encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(self.device)
        instruction_token_mask = None
        segment_token_ids = None
        if needs_instruction_mask:
            if offsets is None:
                raise RuntimeError(
                    "Instruction attention requires a fast tokenizer with offset mappings"
                )
            instruction_text = (
                source_instruction if self.instruction_only_text_budget else rendered_instruction
            )
            char_start = prompt.find(instruction_text)
            if char_start < 0:
                raise ValueError("Rendered prompt does not contain its instruction")
            char_end = char_start + len(instruction_text)
            instruction_token_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
            for column, (token_start, token_end) in enumerate(offsets[0].tolist()):
                if (
                    token_end > char_start
                    and token_start < char_end
                    and prompt[token_start:token_end].strip()
                ):
                    instruction_token_mask[0, column] = True
            instruction_token_mask &= attention_mask.bool()
            if not instruction_token_mask.any():
                raise ValueError("Instruction was fully truncated; increase max_text_tokens")
            instruction_token_mask = instruction_token_mask.to(self.device)
            if self.task_state_mode != "none":
                from aerial_openvla.data.instruction_segments import segment_instruction

                segment_token_ids = torch.full_like(input_ids, -1, dtype=torch.long)
                segments = segment_instruction(
                    instruction_text, max_segments=self.task_state_max_segments
                )
                for segment_index, segment in enumerate(segments):
                    span_start = char_start + segment.char_start
                    span_end = char_start + segment.char_end
                    for column, (token_start, token_end) in enumerate(offsets[0].tolist()):
                        if (
                            instruction_token_mask[0, column]
                            and token_end > span_start
                            and token_start < span_end
                        ):
                            segment_token_ids[0, column] = segment_index
                if torch.any(instruction_token_mask & (segment_token_ids < 0)):
                    raise ValueError("Segment spans leave uncovered instruction tokens")
                segment_token_ids = segment_token_ids.to(self.device)
        pixel_values = pixel_values.to(
            self.device, dtype=torch.bfloat16 if self.bf16 else torch.float32
        )
        prev_actions = torch.tensor([self.prev_action], dtype=torch.long, device=self.device)
        hard_pointer_stage = (
            torch.tensor([self.current_stage], dtype=torch.long, device=self.device)
            if self.task_state_pointer_mode == "hardptr"
            else None
        )

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                frame_valid_mask=frame_valid_mask,
                frame_batch_indices=frame_batch_indices,
                frame_slot_indices=frame_slot_indices,
                precomputed_vision_features=precomputed_vision_features,
                cached_memory_tokens=cached_memory_tokens,
                history_valid_mask=history_valid_mask,
                omtrack_history_features=omtrack_history_features,
                omtrack_history_valid_mask=omtrack_history_valid_mask,
                h16_frozen_pool_history_features=h16_frozen_pool_history_features,
                h16_frozen_pool_history_valid_mask=h16_frozen_pool_history_valid_mask,
                native1t_memory_features=native1t_memory_features,
                native1t_memory_valid_mask=native1t_memory_valid_mask,
                native1t_action_history=native1t_action_history,
                native1t_action_history_valid_mask=(
                    native1t_action_history_valid_mask
                ),
                prev_actions=prev_actions,
                memory_state=self.memory_state,
                instruction_token_mask=instruction_token_mask,
                segment_token_ids=segment_token_ids,
                segment_belief_prev=self.segment_belief,
                task_gru_state_prev=self.task_gru_state,
                hard_pointer_stage=hard_pointer_stage,
                return_auxiliary=(
                    self.task_state_mode != "none"
                    or (self.action_chunk_horizon > 1 and self.use_hierarchical_stop)
                ),
            )

        move_logits = None
        stop_logits = None
        openfly_generated_token_ids: torch.Tensor | None = None
        if isinstance(output, dict):
            logits = output["action_logits"]
            move_logits = output.get("move_logits")
            stop_logits = output.get("stop_logits")
            openfly_generated_token_ids = output.get("openfly_generated_token_ids")
            if output.get("segment_alpha") is not None:
                previous_task_gru_state = self.task_gru_state
                returned_task_gru_state = output.get("task_gru_state")
                if self.use_task_gru:
                    if returned_task_gru_state is None:
                        raise RuntimeError("TaskGRU auxiliary output is missing recurrent state")
                    self.task_gru_state = returned_task_gru_state.detach()
                alpha = output["segment_alpha"].float()[0]
                belief_full = output.get("segment_belief")
                if belief_full is None:
                    raise RuntimeError("SegTS auxiliary output is missing segment belief")
                self.segment_belief = belief_full.detach()
                belief = belief_full.float()[0]
                context_alpha_full = output.get("segment_context_alpha")
                if context_alpha_full is None:
                    raise RuntimeError("SegTS auxiliary output is missing context alpha")
                context_alpha = context_alpha_full.float()[0]
                valid = output["segment_valid_mask"][0].bool()
                alpha = alpha[valid]
                belief = belief[valid]
                context_alpha = context_alpha[valid]
                self.last_task_state = {
                    "num_segments": int(valid.sum().item()),
                    "alpha": alpha.cpu().tolist(),
                    "belief": belief.cpu().tolist(),
                    "context_alpha": context_alpha.cpu().tolist(),
                    "raw_argmax_segment": int(alpha.argmax().item()),
                    "sticky_argmax_segment": int(belief.argmax().item()),
                    "argmax_segment": int(belief.argmax().item()),
                    "alpha_max": float(alpha.max().item()),
                    "alpha_entropy": float(output["segment_entropy"][0].float().item()),
                    "pooling_mode": self.task_state_pooling_mode,
                    "pointer_mode": self.task_state_pointer_mode,
                }
                if self.task_state_pointer_mode == "hardptr":
                    pointer_stage = output.get("segment_pointer_stage")
                    pointer_next = output.get("segment_pointer_next_stage")
                    pointer_advance = output.get("segment_pointer_advance")
                    if (
                        pointer_stage is None
                        or pointer_next is None
                        or pointer_advance is None
                        or pointer_stage.numel() != 1
                        or pointer_next.numel() != 1
                        or pointer_advance.numel() != 1
                    ):
                        raise RuntimeError("HardPtr auxiliary output is incomplete")
                    stage_before = int(pointer_stage.item())
                    stage_next = int(pointer_next.item())
                    advances = bool(pointer_advance.item())
                    if stage_before != self.current_stage:
                        raise RuntimeError(
                            "HardPtr model/policy stage drift: "
                            f"model={stage_before}, policy={self.current_stage}"
                        )
                    self.current_stage = stage_next if advances else stage_before
                    self.last_task_state.update({
                        "current_stage_before": stage_before,
                        "current_stage_after": self.current_stage,
                        "candidate_next_stage": stage_next,
                        "advance": advances,
                    })
                if self.task_gru_state is not None:
                    current_state = self.task_gru_state.float()[0]
                    cosine_prev = None
                    if previous_task_gru_state is not None:
                        cosine_prev = float(
                            torch.nn.functional.cosine_similarity(
                                current_state.unsqueeze(0),
                                previous_task_gru_state.float(),
                                dim=-1,
                            ).item()
                        )
                    self.last_task_state["task_gru_state_norm"] = float(
                        current_state.norm().item()
                    )
                    self.last_task_state["task_gru_state_cosine_prev"] = cosine_prev
            if self.use_temporal_memory:
                current_memory_token = output.get("current_memory_token")
                if current_memory_token is None:
                    raise RuntimeError("TM-VLA auxiliary output is missing current memory token")
                self.temporal_memory_buffer.append(current_memory_token.detach())
            if self.use_native1t_postfusion:
                current_memory_token = output.get("current_memory_token")
                if current_memory_token is None:
                    raise RuntimeError("Native1T autoregressive output is missing current memory token")
                if self.native1t_event_manager is not None:
                    self.native1t_event_manager.update(
                        current_memory_token.detach(),
                        self.native1t_event_step,
                        token_idx=self.native1t_event_step,
                    )
                    self.native1t_event_step += 1
                else:
                    self.native1t_memory_buffer.append(current_memory_token.detach())
        elif isinstance(output, tuple) and self.use_native1t_postfusion:
            logits, current_memory_token = output
            if self.native1t_event_manager is not None:
                self.native1t_event_manager.update(
                    current_memory_token.detach(),
                    self.native1t_event_step,
                    token_idx=self.native1t_event_step,
                )
                self.native1t_event_step += 1
            else:
                self.native1t_memory_buffer.append(current_memory_token.detach())
        elif isinstance(output, tuple) and self.use_temporal_memory:
            logits, current_memory_token = output
            self.temporal_memory_buffer.append(current_memory_token.detach())
        elif isinstance(output, tuple):
            logits, new_memory = output
            self.memory_state = new_memory
        else:
            logits = output

        if self.use_h16_frozen_pool:
            if current_feature is None:
                raise RuntimeError("H16-1T-FrozenPool did not produce current frozen features")
            self.vision_feature_buffer.append(current_feature.mean(dim=1).detach())

        hierarchical_action_id = None
        openfly_action_id: int | None = None
        if self.openfly_action_codec is not None:
            if tuple(logits.shape[-2:]) != (8, 256):
                raise RuntimeError(
                    "OpenFly action-token checkpoint returned unexpected logits: "
                    f"{tuple(logits.shape)}"
                )
            if self.openfly_action_head_type == "action_tokens_ar_v2":
                if openfly_generated_token_ids is None or self.openfly_action_vocab_size is None:
                    raise RuntimeError(
                        "OpenFly autoregressive checkpoint did not return generated full-vocabulary token IDs"
                    )
                decoded_ids, decoded_vectors, matched = (
                    self.openfly_action_codec.decode_token_ids_tensor(
                        openfly_generated_token_ids,
                        action_vocab_size=self.openfly_action_vocab_size,
                    )
                )
                # OpenFly itself executes the decoded greedy sequence; it does
                # not derive the action from an independent 10-way posterior.
                codebook_probabilities = torch.nn.functional.one_hot(
                    decoded_ids, num_classes=self.num_actions
                ).to(dtype=torch.float32)[0]
                token_classes = (
                    self.openfly_action_vocab_size - openfly_generated_token_ids.long() - 1
                )[0]
            else:
                decoded_ids, decoded_vectors, matched = (
                    self.openfly_action_codec.decode_logits_tensor(logits)
                )
                codebook_probabilities = self.openfly_action_codec.codebook_probabilities_tensor(
                    logits
                )[0]
                token_classes = logits.argmax(dim=-1)[0]
            openfly_action_id = int(decoded_ids[0].item())
            probabilities = codebook_probabilities
            self.last_openfly_action = {
                "action_id": openfly_action_id,
                "vector": decoded_vectors[0].float().cpu().tolist(),
                "matched_codebook": bool(matched[0].item()),
                "token_classes": token_classes.cpu().tolist(),
                "generated_token_ids": (
                    None
                    if openfly_generated_token_ids is None
                    else openfly_generated_token_ids[0].cpu().tolist()
                ),
                "codebook_probabilities": codebook_probabilities.cpu().tolist(),
                "probability_semantics": (
                    "generated_action_one_hot"
                    if self.openfly_action_head_type == "action_tokens_ar_v2"
                    else "parallel_codebook_posterior"
                ),
                "codec_sha256": self.openfly_action_codec.sha256,
            }
        elif logits.ndim == 3 and self.use_hierarchical_stop:
            if move_logits is None or stop_logits is None:
                raise RuntimeError("AC4-HSTOP output is missing MOVE or STOP logits")
            navigation_probabilities = torch.softmax(move_logits.float(), dim=-1)[0]
            stop_probabilities = torch.sigmoid(stop_logits.float())[0]
            if navigation_probabilities.shape[0] != self.action_chunk_horizon:
                raise RuntimeError("AC4-HSTOP horizon disagrees with checkpoint metadata")
            self.action_chunk_predictions.append(
                (self.action_chunk_step, navigation_probabilities.detach())
            )
            navigation_decision = navigation_probabilities[0]
            if self.action_chunk_temporal_ensemble:
                navigation_decision = ensemble_action_chunk_probabilities(
                    self.action_chunk_predictions,
                    self.action_chunk_step,
                    self.action_chunk_temporal_ensemble_decay,
                )
            current_stop_probability = stop_probabilities[0]
            hierarchical_action_id = select_hierarchical_action(
                navigation_decision,
                current_stop_probability,
                self.hierarchical_stop_threshold,
            )

            # Normalized probabilities are diagnostics only. The action above is
            # selected by the explicit hierarchy, never by joint eight-way argmax.
            probabilities = torch.cat(
                [
                    current_stop_probability.unsqueeze(0),
                    (1.0 - current_stop_probability) * navigation_decision,
                ]
            )
            chunk_probabilities = torch.cat(
                [
                    stop_probabilities.unsqueeze(-1),
                    (1.0 - stop_probabilities).unsqueeze(-1)
                    * navigation_probabilities,
                ],
                dim=-1,
            )
            self.last_action_chunk_probabilities = chunk_probabilities.cpu().tolist()
            self.last_action_chunk_slot0_probabilities = (
                chunk_probabilities[0].cpu().tolist()
            )
            self.last_action_chunk_navigation_probabilities = (
                navigation_probabilities.cpu().tolist()
            )
            self.last_action_chunk_stop_probabilities = stop_probabilities.cpu().tolist()
            self.last_stop_probability = float(current_stop_probability.item())
            self.action_chunk_step += 1
        elif logits.ndim == 3:
            chunk_probabilities = torch.softmax(logits.float(), dim=-1)[0]
            if chunk_probabilities.shape[0] != self.action_chunk_horizon:
                raise RuntimeError("AC output horizon disagrees with checkpoint metadata")
            self.action_chunk_predictions.append(
                (self.action_chunk_step, chunk_probabilities.detach())
            )
            slot0_probabilities = chunk_probabilities[0]
            probabilities = slot0_probabilities
            if self.action_chunk_temporal_ensemble:
                probabilities = ensemble_action_chunk_probabilities(
                    self.action_chunk_predictions,
                    self.action_chunk_step,
                    self.action_chunk_temporal_ensemble_decay,
                )
            self.last_action_chunk_probabilities = chunk_probabilities.cpu().tolist()
            self.last_action_chunk_slot0_probabilities = (
                slot0_probabilities.cpu().tolist()
            )
            self.action_chunk_step += 1
        else:
            probabilities = torch.softmax(logits.float(), dim=-1)[0]
        action_id = (
            openfly_action_id
            if openfly_action_id is not None
            else (
                hierarchical_action_id
                if hierarchical_action_id is not None
                else int(probabilities.argmax().item())
            )
        )
        previous_action = self.prev_action
        self.prev_action = action_id
        if current_feature is not None:
            if self.use_omtrack:
                sparse = self.model._pool_vision_grid(current_feature, (2, 2))
                self.vision_feature_buffer.append(sparse.detach())
            elif self.visual_history_mode in {"recent3", "motion_recent3"}:
                self.vision_feature_buffer.append(current_feature.detach())
            elif previous_action != self.prev_action_start_id and action_id != previous_action:
                self.vision_feature_buffer.append(current_feature.detach())
        self.previous_actions.append(action_id)
        if self.use_relative_state_text:
            if self.use_relative_planar_state_text:
                from aerial_openvla.data.multiframe_dataset import (
                    update_relative_planar_state,
                )

                (
                    self.relative_heading_degrees,
                    self.relative_altitude_meters,
                    self.relative_forward_meters,
                    self.relative_left_meters,
                ) = update_relative_planar_state(
                    self.relative_heading_degrees,
                    self.relative_altitude_meters,
                    self.relative_forward_meters,
                    self.relative_left_meters,
                    action_id,
                )
            else:
                from aerial_openvla.data.multiframe_dataset import update_relative_state

                self.relative_heading_degrees, self.relative_altitude_meters = (
                    update_relative_state(
                        self.relative_heading_degrees,
                        self.relative_altitude_meters,
                        action_id,
                    )
                )
        return action_id, probabilities.cpu().tolist()
