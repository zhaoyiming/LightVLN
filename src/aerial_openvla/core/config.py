from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")

# Aliases that let older configs use deprecated field names transparently.
_FIELD_ALIASES: dict[str, str] = {
    "policy_type": "model_type",
}


def _apply_aliases(raw: dict[str, Any]) -> dict[str, Any]:
    """Translate deprecated field names so strict parsing still passes."""
    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        canonical = _FIELD_ALIASES.get(key, key)
        if canonical in normalized:
            raise ValueError(
                f"Both {canonical!r} and {key!r} are set — pick one."
            )
        normalized[canonical] = value
    return normalized


def _strict_dataclass_from_dict(cls: type[T], raw: dict[str, Any]) -> T:
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {unknown}")
    return cls(**raw)


def load_yaml_dataclass(path: str | Path, cls: type[T]) -> T:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    raw = _apply_aliases(raw)
    return _strict_dataclass_from_dict(cls, raw)


@dataclass(slots=True)
class TrainConfig:
    model_type: str = "openvla"
    input_mode: str = "image"
    feature_lmdb_dir: str = "data/features/collect/cma/train"
    model_name_or_path: str = "openvla/openvla-7b"
    train_manifest: str = "data/manifests/train.jsonl"
    val_manifest: str | None = "data/manifests/val_seen.jsonl"
    output_dir: str = "output/checkpoints/openvla-lora"
    # Isolated OpenFly adapter metadata.  This is intentionally separate from
    # the AirVLN manifest fields so an OpenFly Native1T checkpoint records the
    # action contract that produced it.
    openfly_action_policy: str | None = None
    # ``action_tokens_v1`` is the historical parallel 8-D / 256-bin head;
    # ``action_tokens_ar_v2`` uses the same codec through Qwen's full causal
    # LM head. ``categorical`` remains the legacy AirVLN default, and is also
    # used by the explicit OpenFly categorical10 entry point when paired with
    # ``openfly_action_policy: openfly_10``.
    openfly_action_head_type: str = "categorical"
    openfly_action_token_dim: int = 8
    openfly_action_token_bins: int = 256
    openfly_action_codec_path: str | None = None
    openfly_action_codec_sha256: str | None = None
    openfly_native1t_stage1_context_window: int = 16
    prompt_template: str = (
        "In: What primitive action should the aerial robot take to follow the instruction: "
        "{instruction}?\nOut:"
    )
    num_actions: int = 8
    pooling: str = "last_token"
    epochs: int = 3
    batch_size: int = 2
    num_workers: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate_lora: float = 5e-5
    learning_rate_head: float = 2e-4
    learning_rate_projector: float = 1e-5
    learning_rate_memory_tokenizer: float = 1e-4
    learning_rate_memory_projector: float = 5e-5
    learning_rate_action_history: float = 2e-4
    learning_rate_vision: float = 1e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    # When enabled, the 256-token budget applies to the original instruction
    # only; fixed prompt tokens and visual tokens are outside this budget.
    # It is intentionally restricted by the collators to raw, non-Relative
    # instruction input.
    instruction_only_text_budget: bool = False
    max_text_tokens: int = 256
    use_lora: bool = True
    train_language_model: bool = False
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    train_projector: bool = False
    freeze_vision: bool = True
    gradient_checkpointing: bool = True
    use_flash_attention_2: bool = True
    bf16: bool = True
    use_learned_action_query: bool = False  # Qwen3.5: replace final chat newline with learned ACT embedding
    class_weights_path: str | None = None
    class_weight_max: float | None = None  # cap per-class weights (rare classes can reach 14x raw)
    inflection_weight_coef: float = 1.0   # 1.0 disables; AirVLN baselines use 1.9 on action-change steps
    seed: int = 7
    log_every_steps: int = 10
    val_every_steps: int = 0  # 0 means validate only at epoch boundaries
    save_every_steps: int = 1000  # legacy field; checkpoints are saved after validation
    save_at_steps: list[int] | None = None  # exact additional optimizer-step checkpoints
    # OpenFly retention policy: keep the three most recent checkpoints.
    keep_last_checkpoints: int = 3
    max_train_steps: int | None = None
    # Optional LR schedule horizon independent of the physical stopping step.
    # Native1T Stage2 uses four epoch horizons while normally stopping after
    # the completed second epoch.
    scheduler_horizon_epochs: int | None = None
    train_loss_plateau_window_steps: int = 0
    train_loss_plateau_patience_steps: int = 0
    train_loss_plateau_min_steps: int = 0
    train_loss_plateau_min_delta: float = 0.0

    # ---- AirMiniVLA fields (only used when model_type="airminivla") ----
    airminivla_variant: str = "m1"       # s0/s1/s2 or visual-history variants
    num_frames: int = 4                  # 1 for S0/S1, 4 for M0/M1/M2
    use_prev_action: bool = True          # False for S0/M0
    use_previous_actions_text: bool = False  # S2: append last 3 actions to instruction
    use_relative_state_text: bool = False  # S2-Relative: prepend relative yaw/altitude
    use_relative_planar_state_text: bool = False  # add start-local forward/left position
    use_memory: bool = False              # True only for M2
    prev_action_embedding_dim: int = 64
    memory_hidden_dim: int = 512          # M2 only
    action_head_hidden: int = 512
    action_chunk_horizon: int = 1       # 1=standard policy; AC-4 fixes this to 4
    action_chunk_loss_weight: float = 1.0
    use_motion_instruction_attention: bool = False  # S0-IA / MCIA residual text retrieval
    motion_instruction_attention_heads: int = 4
    motion_instruction_attention_dropout: float = 0.0
    task_state_mode: str = "none"       # none, segment_only, segment_mcia
    task_state_max_segments: int = 10
    task_state_attention_dim: int = 256
    task_state_dropout: float = 0.0
    task_state_position_key: bool = False
    task_state_pooling_mode: str = "mean"  # mean or learnpool
    task_state_use_gru: bool = False
    task_gru_hidden_dim: int = 256
    task_gru_training_mode: str = "window"  # window or episode_stream
    task_gru_tbptt_steps: int = 0
    task_gru_burn_in: int = 0
    task_gru_loss_window: int = 0
    task_state_hierarchical_ia: bool = False
    task_state_sticky_belief: bool = False
    task_state_belief_momentum: float = 0.8
    task_state_detach_previous_belief: bool = True
    task_state_segment_prior_beta: float = 1.0
    task_state_eps: float = 1.0e-6
    task_state_entropy_weight: float = 0.0
    task_state_metadata_path: str | None = None
    use_transition_aux_head: bool = False # Training-only GT action-transition head
    transition_loss_weight: float = 0.2   # BCE coefficient; never gates action inference
    use_stop_aux_head: bool = False       # Optional binary STOP-vs-non-STOP auxiliary head
    stop_aux_loss_weight: float = 1.0     # Weight of auxiliary BCE in the training objective
    stop_aux_positive_weight: float = 1.0 # Positive-class weight for rare STOP labels
    stop_aux_logit_weight: float = 1.0    # Auxiliary STOP log-odds added to action-0 logit
    use_hierarchical_stop: bool = False   # Binary STOP gate + 7-way MOVE classifier
    hierarchical_stop_mode: str = "navigation"  # navigation (V2) or text_query (V3)
    hierarchical_stop_threshold: float = 0.5
    hierarchical_stop_initial_probability: float = 0.01
    stop_attention_dim: int = 256
    stop_attention_heads: int = 4
    stop_hidden_dim: int = 256
    stop_dropout: float = 0.1
    use_progress_aux_head: bool = False    # Optional official-TF progress supervision
    progress_mode: str = "direct"         # direct or text_query
    progress_loss_weight: float = 0.1
    progress_smooth_l1_beta: float = 0.1
    progress_hidden_dim: int = 256
    progress_dropout: float = 0.1
    progress_attention_dim: int = 256
    progress_attention_heads: int = 4
    progress_annotations_dir: str = "third_party/AirVLN/DATA/data/aerialvln-s"
    history_pool_size: int = 2
    history_pool_stride: int = 2
    visual_tokens_per_frame: int = 256
    # Task-Aware Attention Pooling: current-frame 256 visual tokens are
    # compressed to a fixed 32-token interface using instruction-conditioned
    # queries.  The pooler is opt-in so the legacy Native1T/direct paths keep
    # their exact behavior by default.
    use_task_aware_pooler: bool = False
    task_aware_pooler_tokens: int = 32
    task_aware_pooler_heads: int = 8
    task_aware_pooler_only: bool = False
    # Instruction-guided local weighted pooling: exact 16x16 -> 4x8
    # partition, with one zero-initialized bilinear score matrix.
    use_instruction_local_pooler: bool = False
    instruction_local_pooler_tokens: int = 32
    history_conditioned_local_pooler: bool = False
    local_history_variant: str = "query"
    # Spatially anchored residual Task-Aware pooling: AvgPool32 tokens remain
    # the fixed spatial anchors and an instruction-conditioned 8-head cross
    # attention contributes only through a zero-initialized scalar residual.
    use_spatial_anchored_residual_pooler: bool = False
    spatial_anchored_pooler_tokens: int = 32
    spatial_anchored_pooler_heads: int = 8
    visual_history_mode: str = "none"    # none, recent3, keyframe3, motion_recent3
    history_tokens_per_frame: int = 8    # AdaptiveAvgPool2d((2, 4))
    use_frame_slot_embedding: bool = False # M0-FE: learned fixed-slot identity
    use_temporal_attention: bool = False   # M0-FE-TA: current queries history
    use_temporal_memory: bool = False       # TM-VLA: seven cached history tokens
    temporal_memory_num_history: int = 7
    temporal_memory_attention_heads: int = 8
    use_native1t_postfusion: bool = False
    native1t_stage: int = 2
    native1t_history_length: int = 16
    native1t_stage1_memory_frames: int = 9
    native1t_history_mode: str = "fifo"  # fifo or event_h16_v1
    native1t_event_tau_local: float = 0.757488
    native1t_event_tau_anchor: float = 0.765
    native1t_event_max_length: int = 8
    native1t_event_max_nodes: int = 16
    native1t_event_recent_protected: int = 4
    native1t_event_train_indices: str | None = None
    native1t_event_val_indices: str | None = None
    native1t_segts_mode: str = "none"  # none, vanilla, or history
    use_native1t_action_history: bool = False
    native1t_action_history_length: int = 4
    native1t_action_embedding_dim: int = 16
    # Optional Stage-1 source for Native1T modules only.  Unlike
    # init_checkpoint, this never loads policy/LLM/projector/action weights.
    native1t_stage1_checkpoint: str | None = None
    native1t_train_cache: str | None = None
    native1t_val_cache: str | None = None
    # Optional frozen Qwen Vision-Encoder output.  The trainable visual merger
    # remains inside the model and is never part of this cache.
    qwen35_premerger_train_cache: str | None = None
    qwen35_premerger_val_cache: str | None = None
    # When set, require the exact on-disk pre-merger cache contract.  The
    # manifest_row_image_path_v2 contract is deliberately non-deduplicated:
    # every manifest row owns its current-frame feature.
    qwen35_premerger_cache_contract: str | None = None
    # Legacy OpenFly direct S0: one frozen raw DINO+SigLIP feature grid per
    # row, consumed before the trainable Prismatic projector.  This is kept
    # separate from the Qwen3.5 cache fields so the direct route cannot be
    # mistaken for Native1T or NativeMEM.
    openfly_premerger_train_cache: str | None = None
    # Qwen3.5-0.8B NativeMEM is a separate replacement for every legacy
    # Qwen Native1T/multiframe variant.
    nativemem_stage: int = 1
    nativemem_history_length: int = 16
    nativemem_stage1_frames: int = 9
    nativemem_single_stage: bool = False
    nativemem_load_vision_backbone: bool = False
    nativemem_premerger_train_cache: str | None = None
    nativemem_premerger_val_cache: str | None = None
    # Optional frozen Native1T tokenizer output cache used by NativeMEM v2
    # Stage 2.  It is aligned with the pre-merger cache mapping and stores
    # one [768] latent per unique (trajectory_id, step_id).
    nativemem_1t_train_cache: str | None = None
    nativemem_1t_val_cache: str | None = None
    learning_rate_nativemem: float = 1e-4
    nativemem_train_action_query: bool = False
    # Qwen3.5 NativeMEM only: append the signed start-relative state block
    # after the instruction. This is distinct from legacy prepended Relative.
    qwen35_relative_state_mode: str = "none"
    qwen35_relative_state_cache: str | None = None
    use_h16_frozen_pool: bool = False
    h16_frozen_pool_history_length: int = 16
    h16_frozen_pool_train_cache: str | None = None
    h16_frozen_pool_val_cache: str | None = None
    use_omtrack: bool = False
    omtrack_history_length: int = 31
    omtrack_history_tokens_per_frame: int = 4
    omtrack_current_tokens: int = 64
    omtrack_use_relative_time_embedding: bool = True
    omtrack_train_sparse_cache: str | None = None
    omtrack_val_sparse_cache: str | None = None
    temporal_attention_dim: int = 256
    temporal_attention_heads: int = 4
    temporal_attention_ffn_dim: int = 1024
    temporal_attention_dropout: float = 0.1
    temporal_attention_initial_scale: float = 0.1
    frame_history_stride: int = 2
    missing_history: str = "error"       # error (dense manifests) or nearest_previous (sparse cache)
    prev_action_vocab: int = 9            # 8 actions + 1 start
    training_stage: int = 1               # 1, 2, or 3 for staged training
    init_checkpoint: str | None = None    # previous-stage weights; optimizer is intentionally fresh
    # Legacy MiniVLA only: strict source for initializing the DINO half of the
    # fused Prism vision tower.  This is provenance, not an evaluator path;
    # checkpoints persist the resulting DINO state independently.
    dino_teacher_init_checkpoint: str | None = None
    dino_teacher_init_sha256: str | None = None
    sequence_length: int = 1              # M2: consecutive steps per truncated-BPTT window
    vision_train_last_n_layers: int = 0   # Stage 3: unfreeze final N blocks per vision tower
    # Native1T Stage 2 opt-in: unfreeze only the final N *output-contributing*
    # blocks of the fused SigLIP tower.  Prismatic returns penultimate-block
    # features, so its registered terminal block is intentionally excluded.
    # Kept separate from vision_train_last_n_layers, which opens both towers.
    vision_train_siglip_last_n_layers: int = 0


@dataclass(slots=True)
class EvalConfig:
    model_type: str = "openvla"
    model_name_or_path: str = "openvla/openvla-7b"
    checkpoint_dir: str = "output/checkpoints/openvla-lora-2gpu/latest"
    fast_manifest: str | None = None
    openvla_root: str = "third_party/OpenVLA"
    airvln_root: str = "third_party/AirVLN"
    airvln_project_root: str = "."
    aerialvln_s_data_dir: str = "data/annotations/aerialvln-s"
    split: str = "val_seen"
    output_dir: str = "output/reports/rollouts/val_seen"
    prompt_template: str = (
        "In: What primitive action should the aerial robot take to follow the instruction: "
        "{instruction}?\nOut:"
    )
    instruction_only_text_budget: bool = False
    max_text_tokens: int = 256
    max_actions: int = 500
    eval_num: int = -1
    simulator_port: int = 30000
    simulator_gpu_ids: str = "0"
    image_height_rgb: int = 224
    image_width_rgb: int = 224
    reset_warmup_seconds: float = 1.0
    reset_discard_frames: int = 0
    use_flash_attention_2: bool = True
    bf16: bool = True
    seed: int = 7
    deterministic: bool = True
    resume: bool = True

    # ---- AirMiniVLA eval fields ----
    airminivla_variant: str = "m1"
    num_frames: int = 4
    frame_history_stride: int = 2
    use_prev_action: bool = True
    use_previous_actions_text: bool = False
    use_relative_state_text: bool = False
    use_relative_planar_state_text: bool = False
    qwen35_relative_state_mode: str = "none"
    use_memory: bool = False
    visual_history_mode: str = "none"
    history_tokens_per_frame: int = 8
    missing_history: str = "error"
    use_stop_aux_head: bool = False
    stop_aux_logit_weight: float = 1.0
    use_motion_instruction_attention: bool = False
    motion_instruction_attention_heads: int = 4
    motion_instruction_attention_dropout: float = 0.0
    use_frame_slot_embedding: bool = False
    use_temporal_attention: bool = False
    use_temporal_memory: bool = False
    temporal_memory_num_history: int = 7
    temporal_memory_attention_heads: int = 8
    use_native1t_postfusion: bool = False
    native1t_stage: int = 2
    native1t_history_length: int = 16
    native1t_stage1_memory_frames: int = 9
    native1t_history_mode: str = "fifo"
    native1t_event_tau_local: float = 0.757488
    native1t_event_tau_anchor: float = 0.765
    native1t_event_max_length: int = 8
    native1t_event_max_nodes: int = 16
    native1t_event_recent_protected: int = 4
    native1t_segts_mode: str = "none"
    use_native1t_action_history: bool = False
    native1t_action_history_length: int = 4
    native1t_action_embedding_dim: int = 16
    use_h16_frozen_pool: bool = False
    h16_frozen_pool_history_length: int = 16
    temporal_attention_dim: int = 256
    temporal_attention_heads: int = 4
    temporal_attention_ffn_dim: int = 1024
    temporal_attention_dropout: float = 0.1
    temporal_attention_initial_scale: float = 0.1
    action_chunk_horizon: int = 1
    use_omtrack: bool = False
    omtrack_history_length: int = 31
    omtrack_history_tokens_per_frame: int = 4
    omtrack_current_tokens: int = 64
    omtrack_use_relative_time_embedding: bool = True
    action_chunk_temporal_ensemble: bool = False
    action_chunk_temporal_ensemble_decay: float = 0.5
    # Runtime-only SegTS evaluation selector.  The checkpoint remains the
    # authority for pooling and all train-time architecture fields.
    task_state_pooling_mode: str = "mean"
    task_state_pointer_mode: str = "vanilla"  # vanilla or hardptr
