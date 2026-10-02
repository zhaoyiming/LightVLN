"""AirMiniVLA — multi-frame VLA action classifier for 8-class drone action classification.

Based on MiniVLA"s Prismatic VLM with Qwen2.5-0.5B backbone, DINOv2 + SigLIP
dual vision towers at 224px.

Variants (controlled via AirMiniVLAConfig):
  S0: single frame, 896→512→8
  S1: single frame + prev_action embedding, 960→512→8
  S2: single frame + three previous actions rendered as prompt text, 896→512→8
  M0: 4-frame visual history, 896→512→8
  M1: 4-frame + prev_action, 960→512→8   ← recommended primary model
  M2: M1 + GRU memory cell, 960→512↔GRU→512→256→8
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForVision2Seq

from aerial_openvla.openfly.action_codec import (
    OPENFLY_ACTION_NAMES,
    OPENFLY_OBSERVED8_ACTION_IDS,
    OPENFLY_OBSERVED8_ACTION_POLICY,
    OPENFLY_OBSERVED8_CLASS_TO_ACTION_ID,
)
from aerial_openvla.models.temporal_memory import MemoryProjector, TemporalMemoryTokenizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AirMiniVLAConfig:
    """All tunable knobs for AirMiniVLA.

    Defaults target the M1 variant (recommended primary model).
    """
    # ---- Model identity ----
    model_name_or_path: str = ""        # path to prism-qwen25-0.5b-hf checkpoint
    variant: str = "m1"                 # s0, s1, m0, m1, m2
    num_actions: int = 8
    # Optional AirSim-adapted DINOv2 teacher.  It is loaded strictly into the
    # DINO half of the fused Prism tower; SigLIP and every policy module keep
    # their normal Base initialization.
    dino_teacher_init_checkpoint: str | None = None
    dino_teacher_init_sha256: str | None = None

    # ---- Multi-frame ----
    num_frames: int = 4                 # 1 for S0/S1, 4 for M0/M1/M2
    history_pool_size: int = 2          # avg_pool_2d kernel size
    history_pool_stride: int = 2        # avg_pool_2d stride
    visual_tokens_per_frame: int = 256  # post-projection tokens per frame; native 16×16, visual64 8×8
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
    # fixed spatial anchors and instruction-conditioned cross-attention is
    # introduced only through a zero-initialized scalar residual.
    use_spatial_anchored_residual_pooler: bool = False
    spatial_anchored_pooler_tokens: int = 32
    spatial_anchored_pooler_heads: int = 8
    visual_history_mode: str = "none"   # none, recent3, keyframe3, motion_recent3
    history_tokens_per_frame: int = 8   # fixed 2x4 pooled history grid
    use_frame_slot_embedding: bool = False
    use_temporal_attention: bool = False
    temporal_attention_dim: int = 256
    temporal_attention_heads: int = 4
    temporal_attention_ffn_dim: int = 1024
    temporal_attention_dropout: float = 0.1
    temporal_attention_initial_scale: float = 0.1
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
    native1t_event_calibration_sha256: str | None = None
    native1t_segts_mode: str = "none"
    use_native1t_action_history: bool = False
    native1t_action_history_length: int = 4
    native1t_action_embedding_dim: int = 16
    use_h16_frozen_pool: bool = False
    h16_frozen_pool_history_length: int = 16
    use_omtrack: bool = False
    omtrack_history_length: int = 31
    omtrack_history_tokens_per_frame: int = 4
    omtrack_current_tokens: int = 64
    omtrack_use_relative_time_embedding: bool = True

    # ---- Previous action embedding (S1/M1/M2) ----
    use_prev_action: bool = True        # False for S0/M0
    use_previous_actions_text: bool = False  # True only for S2
    use_relative_state_text: bool = False  # True only for S2-Relative
    use_relative_planar_state_text: bool = False
    instruction_only_text_budget: bool = False
    prev_action_vocab: int = 9          # 8 actions + 1 start token
    prev_action_dim: int = 64

    # ---- GRU memory (M2 only) ----
    use_memory: bool = False            # True only for M2
    memory_hidden_dim: int = 512

    # ---- Action head ----
    action_head_hidden: int = 512       # intermediate dim of the 2-layer action MLP
    # The legacy default is an N-way classifier.  When
    # ``openfly_action_policy == 'openfly_10'`` and this head is categorical,
    # it is a direct [B, 10] OpenFly classifier: cross-entropy on the released
    # action ID, then argmax into the same ID in closed loop.  The two token
    # heads below are separate historical experiments.
    openfly_action_policy: str | None = None
    openfly_action_head_type: str = "categorical"
    openfly_action_token_dim: int = 8
    openfly_action_token_bins: int = 256
    # Training supplies a local immutable codec path.  Checkpoint metadata
    # intentionally persists only the digest; each checkpoint bundles the
    # codec itself so evaluation is portable.
    openfly_action_codec_path: str | None = None
    openfly_action_codec_sha256: str | None = None
    action_chunk_horizon: int = 1       # AC-4 uses four categorical future slots
    use_motion_instruction_attention: bool = False
    motion_instruction_attention_heads: int = 4
    motion_instruction_attention_dropout: float = 0.0
    task_state_mode: str = "none"
    task_state_max_segments: int = 10
    task_state_attention_dim: int = 256
    task_state_dropout: float = 0.0
    task_state_position_key: bool = False
    # Segment feature pooling is part of the checkpoint architecture.  The
    # default exactly preserves the historical SegTS mean-pooling contract.
    task_state_pooling_mode: str = "mean"  # mean or learnpool
    task_state_use_gru: bool = False
    task_gru_hidden_dim: int = 256
    task_gru_burn_in: int = 0
    task_gru_loss_window: int = 0
    task_state_hierarchical_ia: bool = False
    task_state_sticky_belief: bool = False
    task_state_belief_momentum: float = 0.8
    task_state_detach_previous_belief: bool = True
    task_state_segment_prior_beta: float = 1.0
    task_state_eps: float = 1.0e-6
    task_state_entropy_weight: float = 0.0
    use_transition_aux_head: bool = False  # training-only stay/switch supervision
    use_stop_aux_head: bool = False     # binary STOP-vs-non-STOP auxiliary classifier
    stop_aux_logit_weight: float = 1.0  # auxiliary log-odds contribution to action-0
    use_hierarchical_stop: bool = False # binary STOP gate + 7-way movement classifier
    hierarchical_stop_mode: str = "navigation"  # navigation (V2) or text_query (V3)
    hierarchical_stop_threshold: float = 0.5
    hierarchical_stop_initial_probability: float = 0.01
    stop_attention_dim: int = 256
    stop_attention_heads: int = 4
    stop_hidden_dim: int = 256
    stop_dropout: float = 0.1
    use_progress_aux_head: bool = False
    progress_mode: str = "direct"       # direct or text_query
    progress_hidden_dim: int = 256
    progress_dropout: float = 0.1
    progress_attention_dim: int = 256
    progress_attention_heads: int = 4

    # ---- Freeze / train flags (per stage) ----
    freeze_vision: bool = True
    vision_train_last_n_layers: int = 0
    # Native1T Stage-2-only opt-in: tune the final N output-contributing fused
    # SigLIP blocks without opening the DINO tower.
    vision_train_siglip_last_n_layers: int = 0
    train_projector: bool = True
    use_lora: bool = True
    train_language_model: bool = False
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = True
    use_flash_attention_2: bool = True
    bf16: bool = True

    @property
    def is_multi_frame(self) -> bool:
        return self.num_frames > 1

    @property
    def needs_memory_state(self) -> bool:
        return self.use_memory


# Structural fields that define the checkpoint's architecture.  They are saved
# into classifier_meta.pt and take precedence over eval-side configuration when
# a checkpoint is loaded, so evaluation can never silently rebuild a different
# model than the one that was trained.
STRUCTURAL_META_FIELDS = (
    "variant",
    "num_actions",
    "num_frames",
    "use_prev_action",
    "use_previous_actions_text",
    "use_relative_state_text",
    "use_relative_planar_state_text",
    "instruction_only_text_budget",
    "use_memory",
    "action_head_hidden",
    "openfly_action_policy",
    "openfly_action_head_type",
    "openfly_action_token_dim",
    "openfly_action_token_bins",
    "openfly_action_codec_sha256",
    "action_chunk_horizon",
    "use_motion_instruction_attention",
    "motion_instruction_attention_heads",
    "motion_instruction_attention_dropout",
    "task_state_mode",
    "task_state_max_segments",
    "task_state_attention_dim",
    "task_state_dropout",
    "task_state_position_key",
    "task_state_pooling_mode",
    "task_state_use_gru",
    "task_gru_hidden_dim",
    "task_gru_burn_in",
    "task_gru_loss_window",
    "task_state_hierarchical_ia",
    "task_state_sticky_belief",
    "task_state_belief_momentum",
    "task_state_detach_previous_belief",
    "task_state_segment_prior_beta",
    "task_state_eps",
    "task_state_entropy_weight",
    "use_transition_aux_head",
    "use_stop_aux_head",
    "stop_aux_logit_weight",
    "use_hierarchical_stop",
    "hierarchical_stop_mode",
    "hierarchical_stop_threshold",
    "stop_attention_dim",
    "stop_attention_heads",
    "stop_hidden_dim",
    "stop_dropout",
    "use_progress_aux_head",
    "progress_mode",
    "progress_hidden_dim",
    "progress_dropout",
    "progress_attention_dim",
    "progress_attention_heads",
    "prev_action_vocab",
    "prev_action_dim",
    "memory_hidden_dim",
    "history_pool_size",
    "history_pool_stride",
    "visual_tokens_per_frame",
    "use_task_aware_pooler",
    "task_aware_pooler_tokens",
    "task_aware_pooler_heads",
    "use_instruction_local_pooler",
    "instruction_local_pooler_tokens",
    "history_conditioned_local_pooler",
    "local_history_variant",
    "use_spatial_anchored_residual_pooler",
    "spatial_anchored_pooler_tokens",
    "spatial_anchored_pooler_heads",
    "visual_history_mode",
    "history_tokens_per_frame",
    "use_frame_slot_embedding",
    "use_temporal_attention",
    "temporal_attention_dim",
    "temporal_attention_heads",
    "temporal_attention_ffn_dim",
    "temporal_attention_dropout",
    "temporal_attention_initial_scale",
    "use_temporal_memory",
    "temporal_memory_num_history",
    "temporal_memory_attention_heads",
    "use_native1t_postfusion",
    "native1t_stage",
    "native1t_history_length",
    "native1t_stage1_memory_frames",
    "native1t_history_mode",
    "native1t_event_tau_local",
    "native1t_event_tau_anchor",
    "native1t_event_max_length",
    "native1t_event_max_nodes",
    "native1t_event_recent_protected",
    "native1t_event_calibration_sha256",
    "native1t_segts_mode",
    "use_native1t_action_history",
    "native1t_action_history_length",
    "native1t_action_embedding_dim",
    "use_h16_frozen_pool",
    "h16_frozen_pool_history_length",
    "use_omtrack",
    "omtrack_history_length",
    "omtrack_history_tokens_per_frame",
    "omtrack_current_tokens",
    "omtrack_use_relative_time_embedding",
    "vision_train_siglip_last_n_layers",
)

# Canonical variant names and the exact flag combination each one means.
VARIANT_FLAGS: dict[str, tuple[int, bool, bool, bool]] = {
    "s0": (1, False, False, False),
    "s1": (1, True, False, False),
    "s2": (1, False, False, True),
    "s2_relative": (1, False, False, False),
    "m0": (4, False, False, False),
    "m0_key": (4, False, False, False),
    "m0_motion": (4, False, False, False),
    "m1": (4, True, False, False),
    "m2": (4, True, True, False),
    "omtrack": (1, False, False, False),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_internal_decoder(backbone: nn.Module) -> nn.Module:
    """Robustly locate the internal transformer decoder inside a Prismatic model."""
    lm = backbone.language_model
    # Unwrap PEFT wrapper
    base = lm.model if hasattr(lm, "model") else lm
    base = base.get_base_model() if hasattr(base, "get_base_model") else base
    if hasattr(base, "layers"):
        return base
    for attr in ("model", "transformer", "decoder"):
        candidate = getattr(base, attr, None)
        if isinstance(candidate, nn.Module):
            return candidate
    decoder = base.get_decoder() if hasattr(base, "get_decoder") else None
    if isinstance(decoder, nn.Module):
        return decoder
    raise AttributeError(
        f"Cannot locate internal decoder inside language_model of type {type(lm)}"
    )


class FrameSlotEmbedding(nn.Module):
    """Learned embeddings for fixed temporal slots with unequal token counts."""

    def __init__(self, hidden_dim: int, num_frame_slots: int = 4) -> None:
        super().__init__()
        if hidden_dim <= 0 or num_frame_slots <= 0:
            raise ValueError("hidden_dim and num_frame_slots must be positive")
        self.hidden_dim = hidden_dim
        self.num_frame_slots = num_frame_slots
        # An FE model starts exactly as M0-Mask and learns slot identity later.
        self.frame_embeddings = nn.Parameter(
            torch.zeros(1, num_frame_slots, 1, hidden_dim)
        )

    def forward(self, frame_token_blocks: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(frame_token_blocks) != self.num_frame_slots:
            raise ValueError(
                f"Expected {self.num_frame_slots} frame blocks, "
                f"got {len(frame_token_blocks)}"
            )
        outputs: list[torch.Tensor] = []
        for slot_index, tokens in enumerate(frame_token_blocks):
            if tokens.ndim != 3 or tokens.shape[-1] != self.hidden_dim:
                raise ValueError(
                    "Each frame-token block must have shape "
                    f"[B, N, {self.hidden_dim}]"
                )
            embedding = self.frame_embeddings[:, slot_index].to(
                device=tokens.device, dtype=tokens.dtype
            )
            outputs.append(tokens + embedding)
        return outputs

    def add_slot(self, tokens: torch.Tensor, slot_index: int) -> torch.Tensor:
        """Add one of old/middle/recent/current embeddings to one real frame."""
        if not 0 <= slot_index < self.num_frame_slots:
            raise ValueError(f"Invalid frame slot {slot_index}")
        if tokens.ndim != 3 or tokens.shape[-1] != self.hidden_dim:
            raise ValueError("Frame tokens must have shape [B,N,hidden_dim]")
        embedding = self.frame_embeddings[:, slot_index].to(
            device=tokens.device, dtype=tokens.dtype
        )
        return tokens + embedding


class LightweightTemporalCrossAttention(nn.Module):
    """Let current-frame tokens query all valid compressed history tokens."""

    def __init__(
        self,
        token_dim: int,
        temporal_dim: int = 256,
        num_heads: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        initial_residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if temporal_dim <= 0 or temporal_dim % num_heads:
            raise ValueError("temporal_attention_dim must be divisible by its head count")
        if ffn_dim <= 0:
            raise ValueError("temporal_attention_ffn_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("temporal_attention_dropout must be in [0, 1)")
        if not 0.0 < initial_residual_scale < 1.0:
            raise ValueError("temporal_attention_initial_scale must be in (0, 1)")
        self.token_dim = token_dim
        self.temporal_dim = temporal_dim
        self.current_projection = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, temporal_dim)
        )
        self.history_projection = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, temporal_dim)
        )
        self.temporal_attention = nn.MultiheadAttention(
            temporal_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.context_norm = nn.LayerNorm(temporal_dim)
        self.context_ffn = nn.Sequential(
            nn.Linear(temporal_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, temporal_dim),
            nn.Dropout(dropout),
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(temporal_dim), nn.Linear(temporal_dim, token_dim)
        )
        initial_logit = math.log(
            initial_residual_scale / (1.0 - initial_residual_scale)
        )
        self.residual_gate_logit = nn.Parameter(torch.tensor(initial_logit))

    def forward(
        self,
        history_token_blocks: Sequence[torch.Tensor],
        current_tokens: torch.Tensor,
        frame_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(history_token_blocks) != 3:
            raise ValueError("Temporal attention expects exactly three history blocks")
        if current_tokens.ndim != 3 or current_tokens.shape[-1] != self.token_dim:
            raise ValueError(
                f"current_tokens must have shape [B, N, {self.token_dim}]"
            )
        batch_size = current_tokens.shape[0]
        if tuple(frame_valid_mask.shape) != (batch_size, 4):
            raise ValueError(
                f"frame_valid_mask must have shape {(batch_size, 4)}, "
                f"got {tuple(frame_valid_mask.shape)}"
            )
        history_counts: list[int] = []
        for tokens in history_token_blocks:
            if (
                tokens.ndim != 3
                or tokens.shape[0] != batch_size
                or tokens.shape[-1] != self.token_dim
            ):
                raise ValueError("History blocks must have matching [B, N, D] shapes")
            history_counts.append(tokens.shape[1])

        history_tokens = torch.cat(list(history_token_blocks), dim=1)
        frame_valid_mask = frame_valid_mask.to(
            current_tokens.device, dtype=torch.bool
        )
        history_valid = torch.cat(
            [
                frame_valid_mask[:, index:index + 1].expand(-1, token_count)
                for index, token_count in enumerate(history_counts)
            ],
            dim=1,
        )
        has_valid_history = history_valid.any(dim=1)
        current_features = self.current_projection(current_tokens)
        history_features = self.history_projection(history_tokens)
        history_padding_mask = ~history_valid

        # PyTorch MHA may return NaN if a row has every key masked.  Expose one
        # zero dummy key for those rows, then force their residual to zero.
        no_history = ~has_valid_history
        if no_history.any():
            history_features = history_features.clone()
            history_padding_mask = history_padding_mask.clone()
            history_features[no_history, 0] = 0.0
            history_padding_mask[no_history, 0] = False

        temporal_context, _ = self.temporal_attention(
            query=current_features,
            key=history_features,
            value=history_features,
            key_padding_mask=history_padding_mask,
            need_weights=False,
        )
        temporal_context = temporal_context + self.context_ffn(
            self.context_norm(temporal_context)
        )
        valid_scale = has_valid_history[:, None, None].to(temporal_context.dtype)
        temporal_context = temporal_context * valid_scale
        temporal_delta = self.output_projection(temporal_context) * valid_scale
        enhanced = current_tokens + torch.sigmoid(
            self.residual_gate_logit
        ) * temporal_delta
        return enhanced, temporal_context


class DirectProgressHead(nn.Module):
    """Predict bounded route progress from the final navigation feature."""

    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(features).squeeze(-1))


class TextQueryProgressHead(nn.Module):
    """Let the navigation feature query instruction-token hidden states."""

    def __init__(
        self,
        navigation_dim: int,
        text_dim: int,
        attention_dim: int,
        attention_heads: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if attention_dim % attention_heads:
            raise ValueError("progress_attention_dim must be divisible by progress_attention_heads")
        self.navigation_projection = nn.Sequential(
            nn.LayerNorm(navigation_dim), nn.Linear(navigation_dim, attention_dim)
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_dim), nn.Linear(text_dim, attention_dim)
        )
        self.cross_attention = nn.MultiheadAttention(
            attention_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.predictor = nn.Sequential(
            nn.LayerNorm(attention_dim * 4),
            nn.Linear(attention_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        navigation_features: torch.Tensor,
        instruction_tokens: torch.Tensor,
        instruction_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        instruction_mask = instruction_mask.bool()
        if instruction_mask.shape != instruction_tokens.shape[:2]:
            raise ValueError("instruction_token_mask shape does not match instruction tokens")
        if torch.any(instruction_mask.sum(dim=1) == 0):
            raise ValueError("Every sample needs at least one non-truncated instruction token")
        query = self.navigation_projection(navigation_features)
        text = self.text_projection(instruction_tokens)
        context, attention = self.cross_attention(
            query=query.unsqueeze(1),
            key=text,
            value=text,
            key_padding_mask=~instruction_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        context = context.squeeze(1)
        fused = torch.cat(
            [query, context, query * context, torch.abs(query - context)], dim=-1
        )
        progress = torch.sigmoid(self.predictor(fused).squeeze(-1))
        return progress, attention.squeeze(1)


class SegmentTaskState(nn.Module):
    """Infer an interpretable distribution over ordered instruction segments."""

    def __init__(
        self,
        hidden_dim: int,
        attention_dim: int = 256,
        max_segments: int = 10,
        dropout: float = 0.0,
        use_value_context: bool = True,
        use_position_key: bool = False,
        pooling_mode: str = "mean",
        use_task_gru: bool = False,
        task_gru_hidden_dim: int = 256,
        use_hierarchical_ia: bool = False,
        use_sticky_belief: bool = False,
        belief_momentum: float = 0.8,
        history_conditioned: bool = False,
        history_length: int = 16,
    ) -> None:
        super().__init__()
        if attention_dim <= 0 or max_segments <= 0:
            raise ValueError("Segment attention dimensions must be positive")
        if dropout != 0.0:
            raise ValueError("First-version Segment Task State requires dropout=0")
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.max_segments = max_segments
        if pooling_mode not in {"mean", "learnpool"}:
            raise ValueError("Segment pooling_mode must be 'mean' or 'learnpool'")
        self.pooling_mode = str(pooling_mode)
        if self.pooling_mode == "learnpool" and self.hidden_dim != 896:
            raise ValueError(
                "SegTS-v5 LearnPool is fixed to the legacy AirMiniVLA hidden_dim=896"
            )
        self.use_task_gru = bool(use_task_gru)
        self.task_gru_hidden_dim = int(task_gru_hidden_dim)
        if self.use_task_gru and self.task_gru_hidden_dim <= 0:
            raise ValueError("TaskGRU hidden dimension must be positive")
        self.use_sticky_belief = bool(use_sticky_belief)
        self.belief_momentum = float(belief_momentum)
        if not 0.0 <= self.belief_momentum < 1.0:
            raise ValueError("Segment belief momentum must be in [0, 1)")
        self.query = nn.Linear(hidden_dim, attention_dim)
        self.history_conditioned = bool(history_conditioned)
        self.history_length = int(history_length)
        if self.history_conditioned:
            if self.history_length <= 0:
                raise ValueError("History-conditioned SegTS needs a positive history length")
            self.history_position_embedding = nn.Embedding(
                self.history_length, hidden_dim
            )
            self.history_key = nn.Linear(hidden_dim, attention_dim)
            self.history_value = nn.Linear(hidden_dim, hidden_dim)
            self.segment_query = nn.Linear(
                attention_dim + hidden_dim, attention_dim
            )
        else:
            self.history_position_embedding = None
            self.history_key = None
            self.history_value = None
            self.segment_query = None
        if self.use_task_gru:
            self.task_state_input_proj = nn.Sequential(
                nn.Linear(hidden_dim, self.task_gru_hidden_dim),
                nn.LayerNorm(self.task_gru_hidden_dim),
            )
            self.task_gru = nn.GRUCell(
                self.task_gru_hidden_dim, self.task_gru_hidden_dim
            )
            self.task_query_proj = nn.Linear(
                self.task_gru_hidden_dim, attention_dim
            )
            nn.init.normal_(self.task_query_proj.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.task_query_proj.bias)
            # The v5 direct query remains checkpoint-compatible but is not used
            # by TaskGRU and must not enter DDP's trainable parameter set.
            self.query.requires_grad_(False)
        else:
            self.task_state_input_proj = None
            self.task_gru = None
            self.task_query_proj = None
        self.key = nn.Linear(hidden_dim, attention_dim)
        self.segment_pos_embed = (
            nn.Embedding(max_segments, attention_dim) if use_position_key else None
        )
        if self.segment_pos_embed is not None:
            # A v5 model loaded from v4 must initially reproduce v4 exactly.
            nn.init.zeros_(self.segment_pos_embed.weight)
        self.segment_pooler: nn.Sequential | None = None
        if self.pooling_mode == "learnpool":
            # Shared across all segments.  The zero score head makes every
            # in-segment token equiprobable at initialization, while keeping
            # a non-zero gradient for the final score layer on the first CE
            # update.  The hidden scorer becomes trainable after that update.
            self.segment_pooler = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.GELU(),
                nn.Linear(128, 1),
            )
            nn.init.zeros_(self.segment_pooler[2].weight)
            nn.init.zeros_(self.segment_pooler[2].bias)
        self.value = nn.Linear(hidden_dim, hidden_dim) if use_value_context else None
        self.token_query = (
            nn.Linear(hidden_dim, attention_dim) if use_hierarchical_ia else None
        )
        self.token_key = (
            nn.Linear(hidden_dim, attention_dim) if use_hierarchical_ia else None
        )
        if self.token_query is not None:
            # Uniform intra-segment attention exactly recovers v4 MeanPool at
            # initialization, while the randomly initialized token keys give
            # the zero query a non-zero gradient on the first update.
            nn.init.zeros_(self.token_query.weight)
            nn.init.zeros_(self.token_query.bias)
        self.norm = nn.LayerNorm(hidden_dim) if use_value_context else None

    def forward(
        self,
        state: torch.Tensor,
        instruction_tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        instruction_mask: torch.Tensor,
        belief_prev: torch.Tensor | None = None,
        task_gru_state_prev: torch.Tensor | None = None,
        history_tokens: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
        hard_pointer_stage: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if state.ndim != 2 or instruction_tokens.ndim != 3:
            raise ValueError("Expected state [B,D] and instruction_tokens [B,L,D]")
        if segment_ids.shape != instruction_tokens.shape[:2]:
            raise ValueError("segment_token_ids shape does not match instruction tokens")
        instruction_mask = instruction_mask.bool()
        valid_tokens = instruction_mask & (segment_ids >= 0)
        if torch.any(valid_tokens.sum(dim=1) != instruction_mask.sum(dim=1)):
            raise ValueError("Every valid instruction token must belong to one segment")
        if torch.any(segment_ids[valid_tokens] >= self.max_segments):
            raise ValueError("segment_token_ids exceed task_state_max_segments")

        safe_ids = segment_ids.clamp(min=0, max=self.max_segments - 1)
        one_hot = F.one_hot(safe_ids, num_classes=self.max_segments).to(
            instruction_tokens.dtype
        )
        one_hot = one_hot * valid_tokens.unsqueeze(-1).to(one_hot.dtype)
        lengths = one_hot.sum(dim=1)
        segment_valid = lengths > 0
        if torch.any(segment_valid.sum(dim=1) == 0):
            raise ValueError("Every sample needs at least one valid segment")
        mean_pool_alpha = one_hot.transpose(1, 2) / lengths.clamp_min(1).unsqueeze(-1)
        mean_pooled = torch.einsum("bkl,bld->bkd", mean_pool_alpha, instruction_tokens)
        if self.segment_pooler is None:
            pool_alpha = mean_pool_alpha
            pooled = mean_pooled
        else:
            # One scalar per instruction token, shared for all segments.  The
            # membership mask turns that into independent segment-local
            # softmaxes.  Empty K slots are set to zero after softmax so they
            # never create NaNs or affect a valid segment.
            segment_membership = one_hot.transpose(1, 2).bool()
            pool_scores = self.segment_pooler(instruction_tokens).squeeze(-1)
            pool_logits = pool_scores.unsqueeze(1).expand(-1, self.max_segments, -1)
            pool_logits = pool_logits.masked_fill(
                ~segment_membership, torch.finfo(pool_logits.dtype).min
            )
            pool_logits = torch.where(
                segment_valid.unsqueeze(-1), pool_logits, torch.zeros_like(pool_logits)
            )
            pool_alpha = torch.softmax(pool_logits.float(), dim=-1).to(
                dtype=instruction_tokens.dtype
            )
            pool_alpha = pool_alpha * segment_membership.to(pool_alpha.dtype)
            pool_alpha = pool_alpha / pool_alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            # Express LearnPool as an additive correction to the historical
            # mean.  At zero score-head initialization the correction is
            # exactly zero, so the initial segment feature is the old mean
            # path—not merely an approximately equal weighted sum.
            pooled = mean_pooled + torch.einsum(
                "bkl,bld->bkd", pool_alpha - mean_pool_alpha, instruction_tokens
            )

        task_gru_state = None
        if (
            self.task_state_input_proj is not None
            and self.task_gru is not None
            and self.task_query_proj is not None
        ):
            task_input = self.task_state_input_proj(state)
            if task_gru_state_prev is None:
                task_gru_state_prev = task_input.new_zeros(
                    state.shape[0], self.task_gru_hidden_dim
                )
            elif task_gru_state_prev.shape != (
                state.shape[0], self.task_gru_hidden_dim
            ):
                raise ValueError(
                    "Previous TaskGRU state must have shape "
                    f"[B,{self.task_gru_hidden_dim}]"
                )
            task_gru_state = self.task_gru(
                task_input, task_gru_state_prev.to(task_input)
            )
            query = self.task_query_proj(task_gru_state)
        else:
            if task_gru_state_prev is not None:
                raise ValueError("TaskGRU state was supplied to a non-TaskGRU model")
            query = self.query(state)
        history_attention = torch.empty(0, device=state.device, dtype=state.dtype)
        if self.history_conditioned:
            if (
                self.history_position_embedding is None
                or self.history_key is None
                or self.history_value is None
                or self.segment_query is None
            ):
                raise RuntimeError("History-conditioned SegTS modules are incomplete")
            if history_tokens is None or history_mask is None:
                raise ValueError("History-conditioned SegTS requires H16 tokens and mask")
            expected = (state.shape[0], self.history_length)
            if tuple(history_tokens.shape[:2]) != expected:
                raise ValueError(f"History tokens must begin with shape {expected}")
            if history_tokens.shape[-1] != self.hidden_dim:
                raise ValueError("History token hidden dimension does not match SegTS")
            if tuple(history_mask.shape) != expected:
                raise ValueError(f"History mask must have shape {expected}")
            history_mask = history_mask.to(device=state.device, dtype=torch.bool)
            positions = torch.arange(self.history_length, device=state.device)
            positioned_history = history_tokens.to(state) + self.history_position_embedding(
                positions
            ).to(state).unsqueeze(0)
            history_scores = torch.einsum(
                "bd,bhd->bh", query, self.history_key(positioned_history)
            ) / math.sqrt(self.attention_dim)
            history_scores = history_scores.masked_fill(~history_mask, float("-inf"))
            no_history = ~history_mask.any(dim=-1)
            history_scores[no_history] = 0.0
            history_attention = torch.softmax(history_scores, dim=-1)
            history_attention = history_attention * history_mask.to(
                history_attention.dtype
            )
            history_attention = history_attention / history_attention.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            history_context = torch.einsum(
                "bh,bhd->bd",
                history_attention,
                self.history_value(positioned_history),
            )
            query = self.segment_query(torch.cat([query, history_context], dim=-1))
        elif history_tokens is not None or history_mask is not None:
            raise ValueError("Vanilla SegTS must not receive explicit history inputs")
        keys = self.key(pooled)
        if self.segment_pos_embed is not None:
            position_ids = torch.arange(self.max_segments, device=keys.device)
            position_keys = self.segment_pos_embed(position_ids).to(dtype=keys.dtype)
            keys = keys + position_keys.unsqueeze(0)
        scores = torch.einsum("bd,bkd->bk", query, keys) / math.sqrt(self.attention_dim)
        scores = scores.masked_fill(~segment_valid, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)
        if self.use_sticky_belief:
            if belief_prev is None:
                belief = alpha
            else:
                if belief_prev.shape != alpha.shape:
                    raise ValueError(
                        "Previous segment belief must match current attention shape"
                    )
                momentum = self.belief_momentum
                belief = momentum * belief_prev.to(alpha) + (1.0 - momentum) * alpha
            belief = belief * segment_valid.to(belief.dtype)
            belief = belief / belief.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            belief = alpha
        context_alpha = belief
        pointer_stage = torch.empty(0, device=state.device, dtype=torch.long)
        pointer_next_stage = torch.empty(0, device=state.device, dtype=torch.long)
        pointer_advance = torch.empty(0, device=state.device, dtype=torch.bool)
        if hard_pointer_stage is not None:
            if self.use_sticky_belief:
                raise ValueError("HardPtr is incompatible with Sticky segment belief")
            if hard_pointer_stage.shape not in {(state.shape[0],), (state.shape[0], 1)}:
                raise ValueError("hard_pointer_stage must have shape [B] or [B,1]")
            # Pointer semantics require segment ids to form a contiguous
            # ordered prefix: [0, ..., K-1], followed only by padding.
            valid_counts = segment_valid.sum(dim=-1)
            expected_valid = (
                torch.arange(self.max_segments, device=state.device).unsqueeze(0)
                < valid_counts.unsqueeze(1)
            )
            if not torch.equal(segment_valid, expected_valid):
                raise ValueError("HardPtr requires contiguous ordered SegTS segment ids")
            pointer_stage = hard_pointer_stage.reshape(-1).to(
                device=state.device, dtype=torch.long
            ).clamp_min(0)
            last_stage = valid_counts.to(torch.long) - 1
            pointer_stage = torch.minimum(pointer_stage, last_stage)
            pointer_next_stage = torch.minimum(pointer_stage + 1, last_stage)
            pointer_mask = torch.zeros_like(segment_valid)
            pointer_mask.scatter_(1, pointer_stage.unsqueeze(1), True)
            pointer_mask.scatter_(1, pointer_next_stage.unsqueeze(1), True)
            context_alpha = alpha * pointer_mask.to(alpha.dtype)
            context_alpha = context_alpha / context_alpha.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            current_weight = alpha.gather(1, pointer_stage.unsqueeze(1)).squeeze(1)
            next_weight = alpha.gather(1, pointer_next_stage.unsqueeze(1)).squeeze(1)
            pointer_advance = (pointer_next_stage > pointer_stage) & (
                next_weight > current_weight
            )
        if self.value is not None and self.norm is not None:
            if self.token_query is not None and self.token_key is not None:
                token_query = self.token_query(state)
                token_keys = self.token_key(instruction_tokens)
                token_scores = torch.einsum(
                    "bd,bld->bl", token_query, token_keys
                ) / math.sqrt(self.attention_dim)
                segment_membership = one_hot.transpose(1, 2).bool()
                intra_logits = token_scores.unsqueeze(1).expand(
                    -1, self.max_segments, -1
                )
                intra_logits = intra_logits.masked_fill(
                    ~segment_membership, torch.finfo(intra_logits.dtype).min
                )
                token_alpha = torch.softmax(intra_logits, dim=-1)
                token_alpha = token_alpha * segment_membership.to(token_alpha.dtype)
                token_alpha = token_alpha / token_alpha.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-12)
                token_values = self.value(instruction_tokens)
                values = torch.einsum("bkl,bld->bkd", token_alpha, token_values)
            else:
                token_alpha = torch.empty(0, device=state.device, dtype=state.dtype)
                values = self.value(pooled)
            context = torch.einsum("bk,bkd->bd", context_alpha, values)
            updated = self.norm(state + context)
        else:
            token_alpha = torch.empty(0, device=state.device, dtype=state.dtype)
            context = torch.zeros_like(state)
            updated = state
        entropy = -(alpha * alpha.clamp_min(1e-12).log()).sum(dim=-1)
        return {
            "updated": updated,
            "context": context,
            "alpha": alpha,
            "belief": belief,
            "context_alpha": context_alpha,
            "entropy": entropy,
            "segment_valid_mask": segment_valid,
            "segment_lengths": lengths,
            "pooled_segments": pooled,
            "pool_alpha": pool_alpha,
            "token_alpha": token_alpha,
            "task_gru_state": task_gru_state,
            "history_attention": history_attention,
            "pointer_stage": pointer_stage,
            "pointer_next_stage": pointer_next_stage,
            "pointer_advance": pointer_advance,
        }

    @staticmethod
    def token_prior_bias(
        alpha: torch.Tensor,
        segment_ids: torch.Tensor,
        segment_lengths: torch.Tensor,
        instruction_mask: torch.Tensor,
        *,
        beta: float = 1.0,
        eps: float = 1.0e-6,
    ) -> torch.Tensor:
        safe_ids = segment_ids.clamp(min=0, max=alpha.shape[1] - 1)
        token_alpha = torch.gather(alpha, 1, safe_ids)
        token_lengths = torch.gather(segment_lengths, 1, safe_ids).clamp_min(1)
        prior = token_alpha / token_lengths
        bias = float(beta) * torch.log(prior.clamp_min(float(eps)))
        return bias.masked_fill(~instruction_mask.bool(), float("-inf"))


class MotionInstructionAttention(nn.Module):
    """Let the current action-query state retrieve relevant instruction tokens."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError(
                "The language hidden size must be divisible by "
                "motion_instruction_attention_heads"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("motion_instruction_attention_dropout must be in [0, 1)")
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        motion_features: torch.Tensor,
        instruction_tokens: torch.Tensor,
        instruction_mask: torch.Tensor,
        *,
        token_prior_bias: torch.Tensor | None = None,
        return_context: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if motion_features.ndim != 2:
            raise ValueError("motion_features must have shape [B, D]")
        if instruction_tokens.ndim != 3:
            raise ValueError("instruction_tokens must have shape [B, L, D]")
        instruction_mask = instruction_mask.bool()
        if instruction_mask.shape != instruction_tokens.shape[:2]:
            raise ValueError("instruction_token_mask shape does not match instruction tokens")
        if motion_features.shape[0] != instruction_tokens.shape[0]:
            raise ValueError("Motion and instruction batches must have the same size")
        if motion_features.shape[-1] != instruction_tokens.shape[-1]:
            raise ValueError("Motion and instruction features must have the same hidden size")
        if torch.any(instruction_mask.sum(dim=1) == 0):
            raise ValueError("Every sample needs at least one non-truncated instruction token")

        attention_mask = None
        key_padding_mask = ~instruction_mask
        if token_prior_bias is not None:
            if token_prior_bias.shape != instruction_mask.shape:
                raise ValueError("token_prior_bias must have shape [B,L]")
            attention_mask = token_prior_bias[:, None, :].expand(
                -1, self.cross_attention.num_heads, -1
            ).reshape(-1, 1, instruction_tokens.shape[1])
            # The additive mask already assigns -inf to invalid tokens.
            key_padding_mask = None
        context, attention = self.cross_attention(
            query=motion_features.unsqueeze(1),
            key=instruction_tokens,
            value=instruction_tokens,
            key_padding_mask=key_padding_mask,
            attn_mask=attention_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        context = context.squeeze(1)
        updated = self.norm(motion_features + context)
        if return_context:
            return updated, context, attention.squeeze(1)
        return updated, attention.squeeze(1)


class MCIAContextStopHead(nn.Module):
    """Predict STOP directly from the pre-fusion state and retrieved task context."""

    def __init__(self, hidden_dim: int, stop_hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim * 2, stop_hidden_dim),
            nn.GELU(),
            nn.Linear(stop_hidden_dim, 1),
        )

    @property
    def output_layer(self) -> nn.Linear:
        return self.network[-1]

    def forward(
        self,
        motion_features: torch.Tensor,
        task_context: torch.Tensor,
    ) -> torch.Tensor:
        if motion_features.shape != task_context.shape:
            raise ValueError("h_t and c_t must have the same [B, D] shape")
        return self.network(torch.cat([motion_features, task_context], dim=-1)).squeeze(-1)


class TextConditionedStopHead(nn.Module):
    """Let the current navigation state query goal-relevant instruction tokens."""

    def __init__(
        self,
        navigation_dim: int,
        text_dim: int,
        attention_dim: int,
        attention_heads: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if attention_dim % attention_heads:
            raise ValueError("stop_attention_dim must be divisible by stop_attention_heads")
        self.navigation_projection = nn.Sequential(
            nn.LayerNorm(navigation_dim), nn.Linear(navigation_dim, attention_dim)
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_dim), nn.Linear(text_dim, attention_dim)
        )
        self.cross_attention = nn.MultiheadAttention(
            attention_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.predictor = nn.Sequential(
            nn.LayerNorm(attention_dim * 4),
            nn.Linear(attention_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def output_layer(self) -> nn.Linear:
        return self.predictor[-1]

    def forward(
        self,
        navigation_features: torch.Tensor,
        instruction_tokens: torch.Tensor,
        instruction_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        instruction_mask = instruction_mask.bool()
        if instruction_mask.shape != instruction_tokens.shape[:2]:
            raise ValueError("instruction_token_mask shape does not match instruction tokens")
        if torch.any(instruction_mask.sum(dim=1) == 0):
            raise ValueError("Every sample needs at least one non-truncated instruction token")
        query = self.navigation_projection(navigation_features)
        text = self.text_projection(instruction_tokens)
        context, attention = self.cross_attention(
            query=query.unsqueeze(1),
            key=text,
            value=text,
            key_padding_mask=~instruction_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        context = context.squeeze(1)
        fused = torch.cat(
            [query, context, query * context, torch.abs(query - context)], dim=-1
        )
        stop_logits = self.predictor(fused).squeeze(-1)
        return stop_logits, attention.squeeze(1)


def hierarchical_action_scores(
    move_logits: torch.Tensor,
    stop_logits: torch.Tensor,
    stop_threshold: float = 0.5,
) -> torch.Tensor:
    """Build argmax-compatible scores for a binary STOP gate and MOVE head."""
    if move_logits.ndim < 2:
        raise ValueError("Expected move_logits [..., A-1]")
    if stop_logits.shape != move_logits.shape[:-1]:
        raise ValueError("STOP logits must match every leading MOVE-logit dimension")
    if not 0.0 < stop_threshold < 1.0:
        raise ValueError("stop_threshold must be in (0, 1)")
    threshold_logit = math.log(stop_threshold / (1.0 - stop_threshold))
    gate_margin = stop_logits - stop_logits.new_tensor(threshold_logit)
    move_scores = move_logits - move_logits.max(dim=-1, keepdim=True).values
    return torch.cat([gate_margin.unsqueeze(-1), move_scores], dim=-1)


def factorized_action_scores(
    move_logits: torch.Tensor,
    stop_logits: torch.Tensor,
) -> torch.Tensor:
    """Joint STOP/navigation log-probabilities without a hand-set threshold."""
    if move_logits.ndim != 2 or stop_logits.ndim != 1:
        raise ValueError("Expected move_logits [B, A-1] and stop_logits [B]")
    if move_logits.shape[0] != stop_logits.shape[0]:
        raise ValueError("STOP and MOVE logits must have the same batch size")
    stop_log_probability = F.logsigmoid(stop_logits).unsqueeze(-1)
    move_log_probabilities = F.log_softmax(move_logits, dim=-1) + F.logsigmoid(
        -stop_logits
    ).unsqueeze(-1)
    return torch.cat([stop_log_probability, move_log_probabilities], dim=-1)


class TaskAwareAttentionPooler(nn.Module):
    """Pool a current-frame visual grid with instruction-conditioned queries.

    The module implements the exact requested interface:

      e_L = MeanPool(E_L)
      Q   = Q_0 + W_q e_L
      Z   = MHA(Q, F_t, F_t)

    ``Q_0`` contains one learnable query per output token.  The input
    instruction embeddings are the frozen Qwen input-embedding representations
    already used by the existing policy prompt; ``instruction_mask`` removes
    right-padding before the mean.  No vision or text backbone is added.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        instruction_dim: int,
        output_tokens: int = 32,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if output_tokens not in {32, 64, 256}:
            raise ValueError(
                "Task-Aware Attention Pooling supports 32, 64, or 256 output tokens"
            )
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "Task-Aware pooler hidden_dim must be divisible by num_heads: "
                f"{hidden_dim} % {num_heads} != 0"
            )
        self.hidden_dim = int(hidden_dim)
        self.instruction_dim = int(instruction_dim)
        self.output_tokens = int(output_tokens)
        self.num_heads = int(num_heads)
        self.query0 = nn.Parameter(torch.empty(1, output_tokens, hidden_dim))
        nn.init.normal_(self.query0, std=0.02)
        self.query_projection = nn.Linear(instruction_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        instruction_embeddings: torch.Tensor,
        instruction_mask: torch.Tensor,
    ) -> torch.Tensor:
        if visual_tokens.ndim != 3:
            raise ValueError(
                "Task-Aware pooler visual input must have shape [B,256,D], got "
                f"{tuple(visual_tokens.shape)}"
            )
        if visual_tokens.shape[1] != 256 or visual_tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                "Task-Aware pooler expects current visual tokens [B,256,D], got "
                f"{tuple(visual_tokens.shape)}"
            )
        if instruction_embeddings.ndim != 3:
            raise ValueError(
                "Task-Aware pooler instruction input must have shape [B,L,Dt], got "
                f"{tuple(instruction_embeddings.shape)}"
            )
        if instruction_embeddings.shape[0] != visual_tokens.shape[0]:
            raise ValueError("Visual and instruction batches must have the same size")
        if instruction_embeddings.shape[-1] != self.instruction_dim:
            raise ValueError(
                "Task-Aware pooler instruction embedding dimension mismatch: "
                f"expected {self.instruction_dim}, got {instruction_embeddings.shape[-1]}"
            )
        if instruction_mask.shape != instruction_embeddings.shape[:2]:
            raise ValueError(
                "Task-Aware pooler instruction mask must have shape [B,L], got "
                f"{tuple(instruction_mask.shape)}"
            )
        instruction_mask = instruction_mask.to(
            device=instruction_embeddings.device, dtype=torch.bool
        )
        lengths = instruction_mask.sum(dim=1)
        if torch.any(lengths == 0):
            raise ValueError("Every sample needs at least one valid instruction token")
        valid = instruction_mask.unsqueeze(-1).to(instruction_embeddings.dtype)
        instruction_mean = (instruction_embeddings * valid).sum(dim=1) / lengths.clamp_min(1).unsqueeze(-1).to(
            instruction_embeddings.dtype
        )
        instruction_query = self.query_projection(instruction_mean).unsqueeze(1)
        queries = self.query0.to(
            device=visual_tokens.device, dtype=visual_tokens.dtype
        ) + instruction_query.to(dtype=visual_tokens.dtype)
        pooled, _ = self.cross_attention(
            query=queries,
            key=visual_tokens,
            value=visual_tokens,
            need_weights=False,
        )
        return pooled


class InstructionGuidedLocalWeightedPooler(nn.Module):
    """Instruction-guided weighted pooling with the exact AvgPool32 cells.

    The source grid is 16x16 and is partitioned into a 4x8 output grid. Each
    output cell therefore contains four source rows by two source columns,
    i.e. eight source tokens. The only trainable pooling parameter is the
    bilinear score matrix W. W starts at zero, making every local softmax
    uniform and reducing the forward pass to AvgPool32.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        instruction_dim: int,
        output_tokens: int = 32,
        history_conditioned: bool = False,
        history_variant: str = "query",
    ) -> None:
        super().__init__()
        if output_tokens != 32:
            raise ValueError(
                "Instruction-guided local pooling requires exactly 32 output tokens"
            )
        if hidden_dim <= 0 or instruction_dim <= 0:
            raise ValueError("Pooler dimensions must be positive")
        self.history_conditioned = history_conditioned
        self.history_variant = history_variant
        if history_variant not in ("query", "novelty", "dual_prior", "instruction_residual", "instruction_attention", "instruction_selection", "instruction_selection_residual"):
            raise ValueError(f"Unknown local history variant: {history_variant}")
        if history_conditioned:
            if history_variant == "novelty":
                self.cur_norm = nn.LayerNorm(hidden_dim)
                self.hist_norm = nn.LayerNorm(hidden_dim)
                self.cur_proj = nn.Linear(hidden_dim, 32)
                self.hist_proj = nn.Linear(hidden_dim, 32)
                self.tau = 0.1
            else:
                self.temporal_mlp = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 1))
                self.history_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
            if history_variant == "instruction_selection_residual":
                self.text_key_projection = nn.Linear(instruction_dim, hidden_dim, bias=False)
            if history_variant != "instruction_selection":
                self.gamma = nn.Parameter(torch.zeros(()))
        self.hidden_dim = int(hidden_dim)
        self.instruction_dim = int(instruction_dim)
        self.output_tokens = int(output_tokens)
        self.score_weight = nn.Parameter(
            torch.zeros(self.hidden_dim, self.instruction_dim)
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        instruction_embeddings: torch.Tensor,
        instruction_mask: torch.Tensor,
        history_tokens: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if visual_tokens.ndim != 3 or visual_tokens.shape[1] != 256:
            raise ValueError(
                "Instruction-guided local pooler expects visual tokens [B,256,D], "
                f"got {tuple(visual_tokens.shape)}"
            )
        if visual_tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                "Instruction-guided local pooler visual dimension mismatch: "
                f"expected {self.hidden_dim}, got {visual_tokens.shape[-1]}"
            )
        if instruction_embeddings.ndim != 3:
            raise ValueError(
                "Instruction-guided local pooler expects instruction embeddings [B,L,Dt]"
            )
        if instruction_embeddings.shape[0] != visual_tokens.shape[0]:
            raise ValueError("Visual and instruction batches must have the same size")
        if instruction_embeddings.shape[-1] != self.instruction_dim:
            raise ValueError(
                "Instruction-guided local pooler instruction dimension mismatch: "
                f"expected {self.instruction_dim}, got {instruction_embeddings.shape[-1]}"
            )
        if instruction_mask.shape != instruction_embeddings.shape[:2]:
            raise ValueError(
                "Instruction-guided local pooler mask must have shape [B,L], got "
                f"{tuple(instruction_mask.shape)}"
            )

        # [B,16,16,D] -> [B,4,8,8,D]. The two inner factors are the exact
        # four source rows and two source columns belonging to each cell.
        batch = visual_tokens.shape[0]
        cells = (
            visual_tokens.reshape(batch, 16, 16, self.hidden_dim)
            .reshape(batch, 4, 4, 8, 2, self.hidden_dim)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, 32, 8, self.hidden_dim)
        )

        instruction_mask = instruction_mask.to(
            device=instruction_embeddings.device, dtype=torch.bool
        )
        lengths = instruction_mask.sum(dim=1)
        if torch.any(lengths == 0):
            raise ValueError("Every sample needs at least one valid instruction token")
        if self.history_variant in ("instruction_selection", "instruction_selection_residual"):
            if not self.history_conditioned:
                raise ValueError("Instruction selection requires history conditioning")
            if history_tokens is None or history_tokens.shape != (batch, 16, self.hidden_dim):
                raise ValueError("Instruction selection requires projected H16")
            if history_mask is None or history_mask.shape != (batch, 16):
                raise ValueError("Instruction selection requires existing history mask")
            if self.instruction_dim != self.hidden_dim:
                raise ValueError("Instruction selection expects matching text/history dimensions")
            hm = history_mask.to(device=history_tokens.device, dtype=torch.bool)
            masked_history = history_tokens * hm.unsqueeze(-1).to(history_tokens.dtype)
            state = self.temporal_mlp(masked_history.transpose(1, 2)).squeeze(-1)
            # Existing history_projection is q_H; score_weight is q_V. Keeping
            # the latter key lets the verified Local Weighted parent supply W_V.
            query_h = self.history_projection(state)
            query_h = query_h * hm.any(-1, keepdim=True).to(query_h.dtype)
            keys = (self.text_key_projection(instruction_embeddings)
                    if self.history_variant == "instruction_selection_residual" else instruction_embeddings)
            text_scores = torch.einsum("bld,bd->bl", keys.float(), query_h.float())
            if self.history_variant == "instruction_selection_residual":
                text_scores = text_scores / math.sqrt(self.hidden_dim)
            text_weights = text_scores.masked_fill(~instruction_mask, float("-inf")).softmax(-1)
            context = torch.einsum("bl,bld->bd", text_weights, instruction_embeddings.float())
            if self.history_variant == "instruction_selection_residual":
                valid = instruction_mask.unsqueeze(-1).to(instruction_embeddings.dtype)
                global_instruction = (instruction_embeddings * valid).sum(1) / lengths.unsqueeze(-1).to(instruction_embeddings.dtype)
                refined = global_instruction + self.gamma * (context.to(global_instruction.dtype) - global_instruction)
                query_v = F.linear(refined, self.score_weight.to(refined.dtype)).to(visual_tokens.dtype)
                local_scores = torch.einsum("bcnd,bd->bcn", cells, query_v) / math.sqrt(self.hidden_dim)
                local_weights = local_scores.float().softmax(-1).to(cells.dtype)
                return (local_weights.unsqueeze(-1) * cells).sum(2)
            query_v = F.linear(context, self.score_weight.float()).to(visual_tokens.dtype)
            local_scores = torch.einsum("bcnd,bd->bcn", cells, query_v) / math.sqrt(self.hidden_dim)
            local_weights = local_scores.float().softmax(-1).to(cells.dtype)
            return (local_weights.unsqueeze(-1) * cells).sum(2)
        valid = instruction_mask.unsqueeze(-1).to(instruction_embeddings.dtype)
        instruction_mean = (instruction_embeddings * valid).sum(dim=1) / (
            lengths.clamp_min(1).unsqueeze(-1).to(instruction_embeddings.dtype)
        )
        score_weight = self.score_weight.to(
            device=instruction_mean.device, dtype=instruction_mean.dtype
        )
        if self.history_conditioned and self.history_variant in ("instruction_residual", "instruction_attention"):
            if history_tokens is None or history_tokens.shape[1:] != (16, self.hidden_dim):
                raise ValueError("Instruction refinement requires H16")
            if history_mask is None or history_mask.shape != history_tokens.shape[:2]:
                raise ValueError("Instruction refinement requires existing history mask")
            valid_history = history_mask.to(history_tokens.device, dtype=torch.bool)
            masked_history = history_tokens * valid_history.unsqueeze(-1).to(history_tokens.dtype)
            state = self.temporal_mlp(masked_history.transpose(1, 2)).squeeze(-1)
            state = self.history_projection(state)
            state = state * valid_history.any(-1, keepdim=True).to(state.dtype)
            if self.history_variant == "instruction_residual":
                instruction_mean = instruction_mean + self.gamma * state.to(instruction_mean.dtype)
            else:
                # Zero-initialized scalar gives uniform instruction attention.
                logits = torch.einsum("bld,bd->bl", instruction_embeddings.float(), state.float()) * self.gamma.float()
                logits = logits.masked_fill(~instruction_mask, float("-inf"))
                probabilities = logits.softmax(-1)
                # Centering preserves the original mean exactly at initialization.
                uniform = instruction_mask.float() / lengths.unsqueeze(-1).float()
                instruction_mean = instruction_mean + ((probabilities - uniform).unsqueeze(-1) * instruction_embeddings.float()).sum(1).to(instruction_mean.dtype)
        conditioned_instruction = F.linear(
            instruction_mean, score_weight
        ).to(dtype=visual_tokens.dtype)
        scores = torch.einsum(
            "bcnd,bd->bcn", cells, conditioned_instruction
        ) / math.sqrt(self.hidden_dim)
        if self.history_conditioned and self.history_variant not in ("instruction_residual", "instruction_attention"):
            if history_tokens is None or history_tokens.shape[1:] != (16, self.hidden_dim):
                raise ValueError("History-conditioned pooling requires existing masked H16 embeddings")
            if self.history_variant == "novelty":
                if history_mask is None or history_mask.shape != history_tokens.shape[:2]:
                    raise ValueError("Novelty requires the existing H16 valid mask")
                cur = F.normalize(self.cur_proj(self.cur_norm(cells)).float(), dim=-1)
                hist = F.normalize(self.hist_proj(self.hist_norm(history_tokens)).float(), dim=-1)
                valid_history = history_mask.to(device=hist.device, dtype=torch.bool)
                similarity = torch.einsum("bcnd,bhd->bcnh", cur, hist) / self.tau
                # A finite sentinel and explicit mask also handle empty FIFO histories.
                attention = similarity.masked_fill(~valid_history[:, None, None, :], -1e4).softmax(-1)
                attention = attention * valid_history[:, None, None, :]
                attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-6)
                reconstruction = torch.einsum("bcnh,bhd->bcnd", attention, hist)
                novelty = (cur - reconstruction).square().sum(-1)
                novelty = (novelty - novelty.mean(-1, keepdim=True)) / (novelty.std(-1, keepdim=True) + 1e-6)
                novelty = novelty * valid_history.any(-1)[:, None, None]
                scores = scores + self.gamma * novelty
            else:
                temporal = self.temporal_mlp(history_tokens.transpose(1, 2)).squeeze(-1)
                correction = self.history_projection(temporal)
                if self.history_variant == "dual_prior":
                    history_scores = torch.einsum("bcnd,bd->bcn", cells, correction) / math.sqrt(self.hidden_dim)
                    scores = scores + self.gamma * history_scores
                else:
                    scores = torch.einsum("bcnd,bd->bcn", cells, conditioned_instruction + self.gamma * correction) / math.sqrt(self.hidden_dim)
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=cells.dtype)
        return (weights.unsqueeze(-1) * cells).sum(dim=2)



class SpatialAnchoredResidualTaskAwarePooler(nn.Module):
    """Instruction-conditioned cross-attention with AvgPool32 anchors.

    The 32-token AvgPool32 output is retained as the spatially anchored query
    grid.  Instruction conditioning only supplies a shared query offset, and
    the cross-attention result is introduced through a single scalar residual
    initialized to zero.  Consequently the initial forward pass is exactly
    AvgPool32 and there is no free query bank.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        instruction_dim: int,
        output_tokens: int = 32,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if output_tokens != 32:
            raise ValueError(
                "Spatially anchored residual pooling requires exactly 32 output tokens"
            )
        if num_heads != 8:
            raise ValueError(
                "Spatially anchored residual pooling is defined with exactly 8 heads"
            )
        if hidden_dim <= 0 or instruction_dim <= 0:
            raise ValueError("Pooler dimensions must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "Spatially anchored pooler hidden_dim must be divisible by num_heads: "
                f"{hidden_dim} % {num_heads} != 0"
            )
        self.hidden_dim = int(hidden_dim)
        self.instruction_dim = int(instruction_dim)
        self.output_tokens = int(output_tokens)
        self.num_heads = int(num_heads)
        # Bias-free projection implements c = Wq(e_L) exactly.
        self.query_projection = nn.Linear(
            instruction_dim, hidden_dim, bias=False
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.alpha = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _avg_pool32(visual_tokens: torch.Tensor) -> torch.Tensor:
        batch, _, hidden_dim = visual_tokens.shape
        grid = visual_tokens.reshape(batch, 16, 16, hidden_dim).permute(
            0, 3, 1, 2
        )
        pooled = F.adaptive_avg_pool2d(grid, (4, 8))
        return pooled.permute(0, 2, 3, 1).reshape(batch, 32, hidden_dim)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        instruction_embeddings: torch.Tensor,
        instruction_mask: torch.Tensor,
    ) -> torch.Tensor:
        if visual_tokens.ndim != 3 or visual_tokens.shape[1] != 256:
            raise ValueError(
                "Spatially anchored residual pooler expects visual tokens "
                f"[B,256,D], got {tuple(visual_tokens.shape)}"
            )
        if visual_tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                "Spatially anchored residual pooler visual dimension mismatch: "
                f"expected {self.hidden_dim}, got {visual_tokens.shape[-1]}"
            )
        if instruction_embeddings.ndim != 3:
            raise ValueError(
                "Spatially anchored residual pooler expects instruction embeddings "
                "[B,L,Dt]"
            )
        if instruction_embeddings.shape[0] != visual_tokens.shape[0]:
            raise ValueError("Visual and instruction batches must have the same size")
        if instruction_embeddings.shape[-1] != self.instruction_dim:
            raise ValueError(
                "Spatially anchored residual pooler instruction dimension mismatch: "
                f"expected {self.instruction_dim}, got {instruction_embeddings.shape[-1]}"
            )
        if instruction_mask.shape != instruction_embeddings.shape[:2]:
            raise ValueError(
                "Spatially anchored residual pooler instruction mask must have "
                f"shape [B,L], got {tuple(instruction_mask.shape)}"
            )

        base = self._avg_pool32(visual_tokens)
        instruction_mask = instruction_mask.to(
            device=instruction_embeddings.device, dtype=torch.bool
        )
        lengths = instruction_mask.sum(dim=1)
        if torch.any(lengths == 0):
            raise ValueError("Every sample needs at least one valid instruction token")
        valid = instruction_mask.unsqueeze(-1).to(instruction_embeddings.dtype)
        instruction_mean = (instruction_embeddings * valid).sum(dim=1) / (
            lengths.clamp_min(1).unsqueeze(-1).to(instruction_embeddings.dtype)
        )
        query_weight = self.query_projection.weight.to(
            device=instruction_mean.device, dtype=instruction_mean.dtype
        )
        c = F.linear(instruction_mean, query_weight).to(dtype=base.dtype)
        queries = base + c.unsqueeze(1)

        # Newly created modules remain float32 while the frozen Prismatic
        # backbone is normally bfloat16.  Run MHA in its own parameter dtype,
        # then return to the visual feature dtype without changing the model
        # interface or the residual initialization.
        attention_dtype = next(self.cross_attention.parameters()).dtype
        retrieved, _ = self.cross_attention(
            query=queries.to(dtype=attention_dtype),
            key=visual_tokens.to(dtype=attention_dtype),
            value=visual_tokens.to(dtype=attention_dtype),
            need_weights=False,
        )
        retrieved = retrieved.to(dtype=base.dtype)
        return base + self.alpha.to(dtype=base.dtype) * retrieved


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AirMiniVLAActionClassifier(nn.Module):
    """Multi-frame VLA action classifier (variants S0→M2).

    Implements the model protocol required by train.py:
      - forward(**tensor_kwargs) → Tensor[B, num_actions] or Tensor[B,8,256]
      - trainable_parameter_groups(head_lr=..., **extra) → list[dict]
      - save_checkpoint(output_dir, processor, ...)
      - load_checkpoint(checkpoint_dir, ...) → dict | None
    """

    def __init__(self, backbone: nn.Module, config: AirMiniVLAConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        # Set by build_airminivla_classifier when an adapted DINO teacher is
        # loaded.  It is persisted with checkpoints, but never used as an
        # evaluation-time external path.
        self.dino_teacher_init_report: dict[str, Any] | None = None

        # The variant string is metadata, but a mismatch with the structural
        # flags always means a config error — fail instead of silently
        # building a different variant than the name claims.
        expected = VARIANT_FLAGS.get(config.variant)
        actual = (
            config.num_frames,
            config.use_prev_action,
            config.use_memory,
            config.use_previous_actions_text,
        )
        if expected is not None and expected != actual:
            raise ValueError(
                f"variant={config.variant!r} implies (num_frames, use_prev_action, "
                f"use_memory, use_previous_actions_text)={expected} but the config sets {actual}"
            )

        if config.use_relative_planar_state_text and not config.use_relative_state_text:
            raise ValueError("Planar relative text requires relative-state text")

        self._uses_openfly_action_tokens_ar = (
            config.openfly_action_head_type == "action_tokens_ar_v2"
        )
        self._uses_openfly_action_tokens = (
            config.openfly_action_head_type in {"action_tokens_v1", "action_tokens_ar_v2"}
        )
        if config.openfly_action_head_type not in {
            "categorical", "action_tokens_v1", "action_tokens_ar_v2"
        }:
            raise ValueError(
                "openfly_action_head_type must be 'categorical', 'action_tokens_v1', "
                "or 'action_tokens_ar_v2'"
            )
        if config.openfly_action_policy not in {
            None,
            "openfly_10",
            OPENFLY_OBSERVED8_ACTION_POLICY,
        }:
            raise ValueError(
                "AirMiniVLA only recognizes OpenFly action policies "
                f"'openfly_10', {OPENFLY_OBSERVED8_ACTION_POLICY!r}, or None; "
                f"got {config.openfly_action_policy!r}"
            )
        if (
            config.openfly_action_policy
            in {"openfly_10", OPENFLY_OBSERVED8_ACTION_POLICY}
            and config.openfly_action_head_type == "categorical"
        ):
            expected_num_actions = (
                10
                if config.openfly_action_policy == "openfly_10"
                else len(OPENFLY_OBSERVED8_ACTION_IDS)
            )
            incompatible = {
                "non-S0 variant": config.variant != "s0",
                "multi-frame RGB": config.num_frames != 1,
                "previous-action input": config.use_prev_action,
                "previous-action text": config.use_previous_actions_text,
                "GRU memory": config.use_memory,
                "temporal memory": config.use_temporal_memory,
                "H16 frozen pool": config.use_h16_frozen_pool,
                "OmTrack": config.use_omtrack,
                "visual history": config.visual_history_mode != "none",
                "action chunking": config.action_chunk_horizon != 1,
                "transition auxiliary head": config.use_transition_aux_head,
                "STOP auxiliary head": config.use_stop_aux_head,
                "hierarchical STOP": config.use_hierarchical_stop,
                "progress auxiliary head": config.use_progress_aux_head,
                "wrong categorical class count": config.num_actions != expected_num_actions,
                "action-token dimension is not disabled": (
                    config.openfly_action_token_dim != 0
                ),
                "action-token bins are not disabled": (
                    config.openfly_action_token_bins != 0
                ),
                "action codec path": bool(config.openfly_action_codec_path),
                "action codec hash": bool(config.openfly_action_codec_sha256),
            }
            enabled = [name for name, value in incompatible.items() if value]
            if enabled:
                raise ValueError(
                    "OpenFly direct categorical-action contract is incompatible with: "
                    + ", ".join(enabled)
                )
        if self._uses_openfly_action_tokens:
            incompatible = {
                "non-S0 variant": config.variant != "s0",
                "multi-frame RGB": config.num_frames != 1,
                "previous-action input": config.use_prev_action,
                "previous-action text": config.use_previous_actions_text,
                "GRU memory": config.use_memory,
                "temporal memory": config.use_temporal_memory,
                "H16 frozen pool": config.use_h16_frozen_pool,
                "OmTrack": config.use_omtrack,
                "action chunking": config.action_chunk_horizon != 1,
                "transition auxiliary head": config.use_transition_aux_head,
                "STOP auxiliary head": config.use_stop_aux_head,
                "hierarchical STOP": config.use_hierarchical_stop,
                "progress auxiliary head": config.use_progress_aux_head,
                "logical codebook size": config.num_actions != 10,
                "wrong action dimension": config.openfly_action_token_dim != 8,
                "wrong bin count": config.openfly_action_token_bins != 256,
                "missing action codec hash": not config.openfly_action_codec_sha256,
                "non-openfly_10 policy": config.openfly_action_policy != "openfly_10",
            }
            enabled = [name for name, value in incompatible.items() if value]
            if enabled:
                raise ValueError(
                    "OpenFly action-token head contract is incompatible with: "
                    + ", ".join(enabled)
                )

        if config.vision_train_siglip_last_n_layers < 0:
            raise ValueError("vision_train_siglip_last_n_layers must be non-negative")
        if (
            config.vision_train_siglip_last_n_layers > 0
            and not config.use_native1t_postfusion
        ):
            raise ValueError(
                "SigLIP-tail vision tuning is implemented only for Native1T PostFusion"
            )

        if config.task_aware_pooler_only and not config.use_task_aware_pooler:
            raise ValueError(
                "task_aware_pooler_only requires use_task_aware_pooler=true"
            )
        enabled_poolers = [
            name
            for name, enabled in (
                ("task-aware MHA", config.use_task_aware_pooler),
                ("instruction-guided local", config.use_instruction_local_pooler),
                (
                    "spatially anchored residual",
                    config.use_spatial_anchored_residual_pooler,
                ),
            )
            if enabled
        ]
        if len(enabled_poolers) > 1:
            raise ValueError(
                "Visual poolers are mutually exclusive; enabled: "
                + ", ".join(enabled_poolers)
            )
        if config.use_task_aware_pooler:
            if config.visual_tokens_per_frame not in {32, 64, 256}:
                raise ValueError(
                    "Task-Aware Attention Pooling requires visual_tokens_per_frame to be 32, 64, or 256"
                )
            if config.task_aware_pooler_tokens != config.visual_tokens_per_frame:
                raise ValueError(
                    "Task-Aware pooler output must equal visual_tokens_per_frame"
                )
            if config.task_aware_pooler_heads <= 0:
                raise ValueError("task_aware_pooler_heads must be positive")
        if config.history_conditioned_local_pooler and not (config.use_instruction_local_pooler and config.use_native1t_postfusion and config.native1t_stage == 2 and config.native1t_history_length == 16):
            raise ValueError("History-conditioned local pooling requires Local32 Native1T Stage2 H16")
        if config.use_instruction_local_pooler:
            if config.visual_tokens_per_frame != 32:
                raise ValueError(
                    "Instruction-guided local pooling requires visual_tokens_per_frame=32"
                )
            if config.instruction_local_pooler_tokens != 32:
                raise ValueError(
                    "Instruction-guided local pooling requires "
                    "instruction_local_pooler_tokens=32"
                )
            if config.task_aware_pooler_only:
                raise ValueError(
                    "task_aware_pooler_only cannot be combined with "
                    "instruction-guided local pooling"
                )
        if config.use_spatial_anchored_residual_pooler:
            if config.visual_tokens_per_frame != 32:
                raise ValueError(
                    "Spatially anchored residual pooling requires "
                    "visual_tokens_per_frame=32"
                )
            if config.spatial_anchored_pooler_tokens != 32:
                raise ValueError(
                    "Spatially anchored residual pooling requires "
                    "spatial_anchored_pooler_tokens=32"
                )
            if config.spatial_anchored_pooler_heads != 8:
                raise ValueError(
                    "Spatially anchored residual pooling requires exactly 8 heads"
                )
            if config.task_aware_pooler_only:
                raise ValueError(
                    "task_aware_pooler_only cannot be combined with "
                    "spatially anchored residual pooling"
                )

        if config.task_state_mode not in {"none", "segment_only", "segment_mcia"}:
            raise ValueError("task_state_mode must be none, segment_only, or segment_mcia")
        if config.task_state_pooling_mode not in {"mean", "learnpool"}:
            raise ValueError("task_state_pooling_mode must be mean or learnpool")
        if config.native1t_segts_mode not in {"none", "vanilla", "history"}:
            raise ValueError("native1t_segts_mode must be none, vanilla, or history")
        if config.native1t_segts_mode != "none":
            incompatible = {
                "Native1T disabled": not config.use_native1t_postfusion,
                "unsupported Native1T stage": config.native1t_stage not in {1, 2},
                # Stage 1 has no dense cached H16 sequence.  It can still
                # preserve a loaded vanilla SegTS/PosKey policy while learning
                # the Native1T modules from RGB, but history-conditioned SegTS
                # is valid only once Stage 2 supplies causal H16 tokens.
                "Stage 1 requires vanilla SegTS": (
                    config.native1t_stage == 1
                    and config.native1t_segts_mode != "vanilla"
                ),
                "wrong task-state mode": config.task_state_mode != "segment_only",
                "TaskGRU": config.task_state_use_gru,
                "Sticky belief": config.task_state_sticky_belief,
                "HierIA": config.task_state_hierarchical_ia,
                "MCIA": config.use_motion_instruction_attention,
                "auxiliary heads": (
                    config.use_transition_aux_head
                    or config.use_stop_aux_head
                    or config.use_hierarchical_stop
                    or config.use_progress_aux_head
                ),
            }
            enabled = [name for name, value in incompatible.items() if value]
            if enabled:
                raise ValueError(
                    "Invalid Native1T SegTS configuration: " + ", ".join(enabled)
                )
        elif config.use_native1t_postfusion and config.task_state_mode != "none":
            raise ValueError(
                "Native1T Segment Task State requires native1t_segts_mode=vanilla or history"
            )
        if config.task_state_mode == "segment_only" and config.use_motion_instruction_attention:
            raise ValueError("SegTS-only must not enable token-level MCIA")
        if config.task_state_mode == "segment_mcia" and not config.use_motion_instruction_attention:
            raise ValueError("SegTS-MCIA requires use_motion_instruction_attention=true")
        if config.task_state_max_segments != 10:
            raise ValueError("First-version Segment Task State fixes Kmax=10")
        if config.task_state_attention_dim != 256 or config.task_state_dropout != 0.0:
            raise ValueError("First-version Segment Task State fixes dim=256 and dropout=0")
        if config.task_state_entropy_weight != 0.0:
            raise ValueError("First-version Segment Task State uses action CE only")
        if config.task_state_position_key and config.task_state_mode != "segment_only":
            raise ValueError("Segment position keys require task_state_mode=segment_only")
        if config.task_state_pooling_mode == "learnpool":
            incompatible = {
                "wrong task-state mode": config.task_state_mode != "segment_only",
                "missing PosKey": not config.task_state_position_key,
                "Relative text": config.use_relative_state_text,
                "Native1T": config.use_native1t_postfusion,
                "MCIA": config.use_motion_instruction_attention,
                "TaskGRU": config.task_state_use_gru,
                "Sticky belief": config.task_state_sticky_belief,
                "HierIA": config.task_state_hierarchical_ia,
                "action chunking": config.action_chunk_horizon != 1,
            }
            enabled = [name for name, value in incompatible.items() if value]
            if enabled:
                raise ValueError(
                    "SegTS-v5 PosKey LearnPool contract is incompatible with: "
                    + ", ".join(enabled)
                )
        if config.task_state_hierarchical_ia and config.task_state_mode != "segment_only":
            raise ValueError("Hierarchical segment IA requires task_state_mode=segment_only")
        if config.task_state_position_key and config.task_state_hierarchical_ia:
            raise ValueError("PosKey and HierIA are independent experiments")
        if config.task_state_sticky_belief and not config.task_state_position_key:
            raise ValueError("Sticky segment belief requires SegTS PosKey")
        if config.task_state_use_gru:
            if config.task_state_mode != "segment_only" or not config.task_state_position_key:
                raise ValueError("TaskGRU requires Relative SegTS-only PosKey")
            if config.task_state_sticky_belief or config.task_state_hierarchical_ia:
                raise ValueError("TaskGRU is independent of Sticky belief and HierIA")
            if config.task_gru_hidden_dim != 256:
                raise ValueError("First TaskGRU experiment fixes hidden_dim=256")
            if config.task_gru_burn_in < 0 or config.task_gru_loss_window <= 0:
                raise ValueError("TaskGRU requires non-negative burn-in and positive loss window")
        if not 0.0 <= config.task_state_belief_momentum < 1.0:
            raise ValueError("task_state_belief_momentum must be in [0, 1)")

        if config.action_chunk_horizon not in {1, 4, 8}:
            raise ValueError("action_chunk_horizon must be 1, 4, or 8")
        if config.use_omtrack:
            incompatible = {
                "wrong variant": config.variant != "omtrack",
                "multi-frame RGB": config.num_frames != 1,
                "visual history": config.visual_history_mode != "none",
                "previous-action input": config.use_prev_action,
                "action-history text": config.use_previous_actions_text,
                "GRU memory": config.use_memory,
                "MCIA": config.use_motion_instruction_attention,
                "Segment Task State": config.task_state_mode != "none",
                "TM-VLA memory": config.use_temporal_memory,
                "hierarchical STOP": config.use_hierarchical_stop,
                "auxiliary heads": (
                    config.use_transition_aux_head
                    or config.use_stop_aux_head
                    or config.use_progress_aux_head
                ),
                "temporal attention": config.use_temporal_attention,
                "wrong action horizon": config.action_chunk_horizon != 8,
                "non-8-class output": config.num_actions != 8,
                "unfrozen vision": (
                    not config.freeze_vision
                    or config.vision_train_last_n_layers > 0
                    or config.vision_train_siglip_last_n_layers > 0
                ),
                "wrong history length": config.omtrack_history_length not in {16, 31},
                "wrong history tokens": config.omtrack_history_tokens_per_frame != 4,
                "wrong current tokens": config.omtrack_current_tokens not in {64, 256},
                "missing relative time": not config.omtrack_use_relative_time_embedding,
            }
            enabled = [name for name, value in incompatible.items() if value]
            if enabled:
                raise ValueError("Invalid OmTrack configuration: " + ", ".join(enabled))
        elif config.action_chunk_horizon > 1:
            incompatible = {
                "multi-frame input": config.num_frames != 1,
                "visual history": config.visual_history_mode != "none",
                "previous-action input": config.use_prev_action,
                "action-history text": config.use_previous_actions_text,
                "GRU memory": config.use_memory,
                "MCIA": config.use_motion_instruction_attention,
                "Segment Task State": config.task_state_mode != "none",
                "transition supervision": config.use_transition_aux_head,
                "STOP auxiliary head": config.use_stop_aux_head,
                "non-navigation hierarchical STOP": (
                    config.use_hierarchical_stop
                    and config.hierarchical_stop_mode != "navigation"
                ),
                "progress supervision": config.use_progress_aux_head,
                "temporal attention": config.use_temporal_attention,
                "non-8-class output": config.num_actions != 8,
            }
            enabled = [name for name, value in incompatible.items() if value]
            if enabled:
                raise ValueError(
                    "AC-4 supports S0 action CE and navigation hierarchical STOP; "
                    "incompatible settings: "
                    + ", ".join(enabled)
                )

        if config.use_temporal_memory:
            incompatible = {
                "multi-frame image path": config.num_frames != 1,
                "visual history": config.visual_history_mode != "none",
                "previous-action input": config.use_prev_action,
                "action-history text": config.use_previous_actions_text,
                "GRU memory": config.use_memory,
                "transition supervision": config.use_transition_aux_head,
                "STOP auxiliary head": config.use_stop_aux_head,
                "hierarchical STOP": config.use_hierarchical_stop,
                "progress supervision": config.use_progress_aux_head,
                "temporal attention": config.use_temporal_attention,
                "action chunking": config.action_chunk_horizon != 1,
                "non-8-class output": config.num_actions != 8,
                "unfrozen vision": (
                    not config.freeze_vision
                    or config.vision_train_last_n_layers > 0
                    or config.vision_train_siglip_last_n_layers > 0
                ),
            }
            enabled = [name for name, value in incompatible.items() if value]
            if enabled:
                raise ValueError(
                    "Invalid TM-VLA combination; incompatible settings: "
                    + ", ".join(enabled)
                )
            if config.temporal_memory_num_history < 1:
                raise ValueError("TM-VLA requires temporal_memory_num_history >= 1")

        if config.use_native1t_postfusion:
            incompatible = {
                "non-S0 variant": config.variant != "s0",
                "multi-frame current path": config.num_frames != 1,
                "visual history": config.visual_history_mode != "none",
                "previous-action input": config.use_prev_action,
                "action-history text": config.use_previous_actions_text,
                "GRU memory": config.use_memory,
                "TM-VLA": config.use_temporal_memory,
                "FrozenPool": config.use_h16_frozen_pool,
                "OmTrack": config.use_omtrack,
                "MCIA": config.use_motion_instruction_attention,
                "Segment Task State": (
                    config.task_state_mode != "none"
                    and config.native1t_segts_mode == "none"
                ),
                "unsupported action horizon": config.action_chunk_horizon not in {1, 4},
                "AC4 outside Stage 2": (
                    config.action_chunk_horizon > 1 and config.native1t_stage != 2
                ),
                "auxiliary heads": (
                    config.use_transition_aux_head or config.use_stop_aux_head
                    or config.use_hierarchical_stop or config.use_progress_aux_head
                ),
                "full vision unfreeze": not config.freeze_vision,
                "all-tower vision tail": config.vision_train_last_n_layers > 0,
                "SigLIP tail outside Stage 2": (
                    config.vision_train_siglip_last_n_layers > 0
                    and config.native1t_stage != 2
                ),
                "unsupported history length": (
                    config.native1t_history_length not in (8, 16, 31)
                ),
                "wrong Stage-1 frame count": config.native1t_stage1_memory_frames != 9,
                "wrong stage": config.native1t_stage not in (1, 2),
                "unknown history mode": config.native1t_history_mode not in {
                    "fifo", "event_h16_v1"
                },
                "Event-H16 outside Stage 2": (
                    config.native1t_history_mode == "event_h16_v1"
                    and config.native1t_stage != 2
                ),
                "Event-H16 node count": (
                    config.native1t_history_mode == "event_h16_v1"
                    and config.native1t_event_max_nodes != config.native1t_history_length
                ),
            }
            enabled = [name for name, value in incompatible.items() if value]
            if enabled:
                raise ValueError(
                    "Invalid Native1T-PostFusion-H16 configuration: "
                    + ", ".join(enabled)
                )
        elif config.use_native1t_action_history:
            raise ValueError("Native1T AH4 requires use_native1t_postfusion=true")

        if config.use_h16_frozen_pool:
            incompatible = {
                "non-S0 variant": config.variant != "s0",
                "multi-frame image path": config.num_frames != 1,
                "visual history": config.visual_history_mode != "none",
                "TM-VLA": config.use_temporal_memory,
                "OmTrack": config.use_omtrack,
                "MCIA": config.use_motion_instruction_attention,
                "Segment Task State": config.task_state_mode != "none",
                "action chunking": config.action_chunk_horizon != 1,
                "auxiliary heads": (
                    config.use_transition_aux_head or config.use_stop_aux_head
                    or config.use_hierarchical_stop or config.use_progress_aux_head
                ),
                "unfrozen vision": (
                    not config.freeze_vision
                    or config.vision_train_last_n_layers > 0
                    or config.vision_train_siglip_last_n_layers > 0
                ),
                "wrong history length": config.h16_frozen_pool_history_length != 16,
            }
            enabled = [name for name, value in incompatible.items() if value]
            if enabled:
                raise ValueError(
                    "Invalid H16-1T-FrozenPool configuration: " + ", ".join(enabled)
                )

        if config.use_motion_instruction_attention or config.task_state_mode != "none":
            incompatible = {
                "multi-frame input": config.num_frames != 1,
                "visual history": config.visual_history_mode != "none",
                "previous-action input": config.use_prev_action,
                "action-history text": config.use_previous_actions_text,
                "GRU memory": config.use_memory,
                "transition supervision": config.use_transition_aux_head,
                "STOP auxiliary head": config.use_stop_aux_head,
                "hierarchical STOP": (
                    config.use_hierarchical_stop
                    and not (
                        (
                            config.use_motion_instruction_attention
                            and config.hierarchical_stop_mode == "mcia_context"
                        )
                        or (
                            config.task_state_mode == "segment_only"
                            and config.hierarchical_stop_mode == "text_query"
                        )
                    )
                ),
                "progress supervision": config.use_progress_aux_head,
                "temporal attention": config.use_temporal_attention,
            }
            enabled = [name for name, value in incompatible.items() if value]
            if enabled:
                raise ValueError(
                    "First-version MCIA/Segment Task State must be a single-frame action-loss-only "
                    f"experiment; incompatible settings: {', '.join(enabled)}"
                )

        # ---- Freeze everything first ----
        for p in self.backbone.parameters():
            p.requires_grad = False

        # ---- Verify backbone components ----
        for attr in ("vision_backbone", "projector", "language_model"):
            if not hasattr(self.backbone, attr):
                raise AttributeError(
                    f"Backbone missing {attr!r}; expected a Prismatic model"
                )

        hidden_size = int(self.backbone.config.text_config.hidden_size)  # 896 for Qwen2.5-0.5B
        self._hidden_size = hidden_size
        self._openfly_action_vocab_size: int | None = None
        self._openfly_action_eos_token_id: int | None = None
        self._openfly_action_lm_contract: dict[str, int | str] | None = None
        if self._uses_openfly_action_tokens_ar:
            self._initialize_openfly_action_lm_contract()

        projector_linear = next(
            (module for module in self.backbone.projector.modules() if isinstance(module, nn.Linear)),
            None,
        )
        if projector_linear is None:
            raise ValueError("Prismatic projector does not expose a Linear input layer")
        self._vision_feature_size = int(projector_linear.in_features)
        self.task_aware_pooler: TaskAwareAttentionPooler | None = (
            TaskAwareAttentionPooler(
                hidden_dim=hidden_size,
                instruction_dim=hidden_size,
                output_tokens=config.task_aware_pooler_tokens,
                num_heads=config.task_aware_pooler_heads,
            )
            if config.use_task_aware_pooler
            else None
        )
        self.instruction_local_pooler: InstructionGuidedLocalWeightedPooler | None = (
            InstructionGuidedLocalWeightedPooler(
                hidden_dim=hidden_size,
                instruction_dim=hidden_size,
                output_tokens=config.instruction_local_pooler_tokens,
                history_conditioned=config.history_conditioned_local_pooler,
                history_variant=config.local_history_variant,
            )
            if config.use_instruction_local_pooler
            else None
        )
        self.spatial_anchored_residual_pooler: SpatialAnchoredResidualTaskAwarePooler | None = (
            SpatialAnchoredResidualTaskAwarePooler(
                hidden_dim=hidden_size,
                instruction_dim=hidden_size,
                output_tokens=config.spatial_anchored_pooler_tokens,
                num_heads=config.spatial_anchored_pooler_heads,
            )
            if config.use_spatial_anchored_residual_pooler
            else None
        )
        self.temporal_memory_tokenizer = None
        self.temporal_memory_projector = None
        self.temporal_memory_time_embedding = None
        if config.use_temporal_memory:
            self.temporal_memory_tokenizer = TemporalMemoryTokenizer(
                self._vision_feature_size,
                num_heads=config.temporal_memory_attention_heads,
            )
            self.temporal_memory_projector = MemoryProjector(
                self._vision_feature_size, hidden_size
            )
            self.temporal_memory_time_embedding = nn.Embedding(
                config.temporal_memory_num_history, self._vision_feature_size
            )
        self.native1t_memory_tokenizer = None
        self.native1t_memory_projector = None
        self.native1t_mem_bos = None
        self.native1t_action_embedding = None
        self.native1t_action_history_projector = None
        self.native1t_action_bos = None
        if config.use_native1t_postfusion:
            if config.native1t_stage not in (1, 2):
                raise ValueError("Native1T stage must be 1 or 2")
            if config.native1t_history_length not in (8, 16, 31):
                raise ValueError(
                    "Native1T-PostFusion supports history_length in {8, 16, 31}"
                )
            if config.native1t_stage1_memory_frames != 9:
                raise ValueError("Native1T Stage 1 fixes memory_frames=9")
            self.native1t_memory_tokenizer = TemporalMemoryTokenizer(
                self._vision_feature_size,
                num_heads=config.temporal_memory_attention_heads,
            )
            self.native1t_memory_projector = MemoryProjector(
                self._vision_feature_size, hidden_size
            )
            self.native1t_mem_bos = nn.Parameter(torch.empty(1, 1, hidden_size))
            nn.init.normal_(self.native1t_mem_bos, std=hidden_size**-0.5)
            if config.use_native1t_action_history:
                if config.native1t_stage != 2:
                    raise ValueError("Native1T AH4 is a Stage-2-only module")
                if config.native1t_action_history_length != 4:
                    raise ValueError("Native1T AH4 fixes action_history_length=4")
                if config.native1t_action_embedding_dim != 16:
                    raise ValueError("Native1T AH4 fixes action_embedding_dim=16")
                self.native1t_action_embedding = nn.Embedding(
                    config.num_actions, config.native1t_action_embedding_dim
                )
                action_history_input_dim = (
                    config.native1t_action_history_length
                    * config.native1t_action_embedding_dim
                    + config.native1t_action_history_length
                )
                self.native1t_action_history_projector = nn.Sequential(
                    nn.Linear(action_history_input_dim, 128),
                    nn.GELU(),
                    nn.Linear(128, hidden_size),
                    nn.LayerNorm(hidden_size),
                )
                self.native1t_action_bos = nn.Parameter(
                    torch.zeros(1, 1, hidden_size)
                )
                nn.init.normal_(self.native1t_action_bos, std=0.02)
        self.omtrack_time_embedding: nn.Embedding | None = None
        if config.use_omtrack:
            self.omtrack_time_embedding = nn.Embedding(
                config.omtrack_history_length + 1, hidden_size
            )
            nn.init.normal_(self.omtrack_time_embedding.weight, std=0.02)

        if config.use_temporal_attention and not config.use_frame_slot_embedding:
            raise ValueError(
                "M0-FE-TA requires use_frame_slot_embedding=true"
            )
        if (config.use_frame_slot_embedding or config.use_temporal_attention) and config.num_frames != 4:
            raise ValueError("Frame embedding and temporal attention require num_frames=4")
        if config.visual_history_mode not in {
            "none", "recent3", "keyframe3", "motion_recent3"
        }:
            raise ValueError(
                "visual_history_mode must be none, recent3, keyframe3, "
                "or motion_recent3"
            )
        if config.visual_history_mode != "none":
            if config.num_frames != 4:
                raise ValueError("Visual-history variants require four frame slots")
            if (
                config.visual_history_mode in {"recent3", "keyframe3"}
                and not config.use_frame_slot_embedding
            ):
                raise ValueError("Recent-3/Keyframe-3 require four slots and frame embeddings")
            if config.visual_history_mode == "motion_recent3" and config.use_frame_slot_embedding:
                raise ValueError("Motion-Recent3 encodes temporal order in delta channels")
            if config.use_temporal_attention or config.use_memory:
                raise ValueError("First visual-history version forbids TA and GRU memory")
            if (
                not config.freeze_vision
                or config.vision_train_last_n_layers
                or config.vision_train_siglip_last_n_layers
            ):
                raise ValueError("Cached pre-projector history requires a fully frozen vision tower")
            if config.history_tokens_per_frame != 8:
                raise ValueError("History compression is fixed to AdaptiveAvgPool2d((2,4))")
        expected_history_mode = {
            "m0": "recent3",
            "m0_key": "keyframe3",
            "m0_motion": "motion_recent3",
        }.get(config.variant)
        if expected_history_mode is not None and config.visual_history_mode != expected_history_mode:
            raise ValueError(
                f"variant={config.variant!r} requires visual_history_mode={expected_history_mode!r}"
            )
        self.frame_slot_embedding = (
            FrameSlotEmbedding(hidden_size, num_frame_slots=4)
            if config.use_frame_slot_embedding
            else None
        )
        self.motion_history_projector: nn.Module | None = None
        if config.visual_history_mode == "motion_recent3":
            projector_linears = [
                module
                for module in self.backbone.projector.modules()
                if isinstance(module, nn.Linear)
            ]
            if not projector_linears:
                raise ValueError("Cannot infer frozen vision dimension from projector")
            vision_dim = int(projector_linears[0].in_features)
            self.motion_history_projector = nn.Sequential(
                nn.LayerNorm(vision_dim * 3),
                nn.Linear(vision_dim * 3, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
        self.temporal_adapter = (
            LightweightTemporalCrossAttention(
                token_dim=hidden_size,
                temporal_dim=config.temporal_attention_dim,
                num_heads=config.temporal_attention_heads,
                ffn_dim=config.temporal_attention_ffn_dim,
                dropout=config.temporal_attention_dropout,
                initial_residual_scale=config.temporal_attention_initial_scale,
            )
            if config.use_temporal_attention
            else None
        )

        # ---- Learnable action_query token (inserted after vision+text in the sequence) ----
        self.action_query = nn.Parameter(torch.zeros(1, hidden_size))
        nn.init.normal_(self.action_query, std=0.02)
        self.action_chunk_horizon_embeddings: nn.Embedding | None = None
        self.action_chunk_norm: nn.LayerNorm | None = None
        if config.action_chunk_horizon > 1 and not config.use_omtrack:
            self.action_chunk_horizon_embeddings = nn.Embedding(
                config.action_chunk_horizon, hidden_size
            )
            nn.init.normal_(self.action_chunk_horizon_embeddings.weight, std=0.02)
            self.action_chunk_norm = nn.LayerNorm(hidden_size)
        self.omtrack_chunk_norm: nn.LayerNorm | None = None
        self.omtrack_chunk_head: nn.Linear | None = None
        if config.use_omtrack:
            self.omtrack_chunk_norm = nn.LayerNorm(hidden_size)
            self.omtrack_chunk_head = nn.Linear(
                hidden_size, config.action_chunk_horizon * config.num_actions
            )

        # ---- Previous action embedding (S1/M1/M2) ----
        if config.use_prev_action:
            self.prev_action_embed = nn.Embedding(
                config.prev_action_vocab, config.prev_action_dim
            )
            nn.init.normal_(self.prev_action_embed.weight, std=0.02)
        else:
            self.prev_action_embed = None

        # ---- Fuse projection (combine pooled hidden + prev_action_emb) ----
        fuse_input = hidden_size + (config.prev_action_dim if config.use_prev_action else 0)
        # The original OpenFly route reads Qwen's causal-LM head directly;
        # it must not silently retain MiniVLA's parallel classifier MLP.
        self.fuse_proj: nn.Module = (
            nn.Identity()
            if self._uses_openfly_action_tokens_ar
            else nn.Linear(fuse_input, config.action_head_hidden)
        )
        self.motion_instruction_attention: MotionInstructionAttention | None = (
            MotionInstructionAttention(
                hidden_dim=hidden_size,
                num_heads=config.motion_instruction_attention_heads,
                dropout=config.motion_instruction_attention_dropout,
            )
            if config.use_motion_instruction_attention
            else None
        )
        self.segment_task_state: SegmentTaskState | None = (
            SegmentTaskState(
                hidden_dim=hidden_size,
                attention_dim=config.task_state_attention_dim,
                max_segments=config.task_state_max_segments,
                dropout=config.task_state_dropout,
                use_value_context=config.task_state_mode == "segment_only",
                use_position_key=config.task_state_position_key,
                pooling_mode=config.task_state_pooling_mode,
                use_task_gru=config.task_state_use_gru,
                task_gru_hidden_dim=config.task_gru_hidden_dim,
                use_hierarchical_ia=config.task_state_hierarchical_ia,
                use_sticky_belief=config.task_state_sticky_belief,
                belief_momentum=config.task_state_belief_momentum,
                history_conditioned=config.native1t_segts_mode == "history",
                history_length=config.native1t_history_length,
            )
            if config.task_state_mode != "none"
            else None
        )

        # ---- GRU memory (M2 only) ----
        action_output_dim = (
            config.openfly_action_token_dim * config.openfly_action_token_bins
            if config.openfly_action_head_type == "action_tokens_v1"
            else (config.num_actions - 1 if config.use_hierarchical_stop else config.num_actions)
        )
        if self._uses_openfly_action_tokens_ar:
            # Kept as an empty module so generic checkpoint I/O continues to
            # have a stable filename; it is deliberately absent from v2's
            # forward and optimizer surfaces.
            self.gru_cell = None
            self.memory_norm = None
            self.action_head = nn.Identity()
        elif config.use_memory:
            self.gru_cell = nn.GRUCell(fuse_input, config.memory_hidden_dim)
            self.memory_norm = nn.LayerNorm(config.memory_hidden_dim)
            # Deeper action head for M2
            self.action_head = nn.Sequential(
                nn.LayerNorm(config.memory_hidden_dim),
                nn.Linear(config.memory_hidden_dim, 256),
                nn.GELU(),
                nn.Linear(256, action_output_dim),
            )
        else:
            self.gru_cell = None
            self.memory_norm = None
            # 2-layer MLP for S0/S1/M0/M1
            self.action_head = nn.Sequential(
                nn.LayerNorm(config.action_head_hidden),
                nn.Linear(config.action_head_hidden, action_output_dim),
            )

        transition_feature_dim = (
            config.memory_hidden_dim if config.use_memory else config.action_head_hidden
        )
        self.transition_head: nn.Module | None = (
            nn.Sequential(
                nn.LayerNorm(transition_feature_dim),
                nn.Linear(transition_feature_dim, 1),
            )
            if config.use_transition_aux_head
            else None
        )
        if self.transition_head is not None:
            # Start as a neutral p(switch)=0.5 auxiliary classifier.  This head
            # never changes action logits and is ignored by policy inference.
            nn.init.zeros_(self.transition_head[-1].weight)
            nn.init.zeros_(self.transition_head[-1].bias)

        stop_feature_dim = (
            config.memory_hidden_dim if config.use_memory else config.action_head_hidden
        )
        if config.hierarchical_stop_mode not in {
            "navigation", "text_query", "mcia_context"
        }:
            raise ValueError(
                "hierarchical_stop_mode must be 'navigation', 'text_query', "
                "or 'mcia_context'"
            )
        if (
            config.hierarchical_stop_mode in {"text_query", "mcia_context"}
            and not config.use_hierarchical_stop
        ):
            raise ValueError(
                f"hierarchical_stop_mode={config.hierarchical_stop_mode!r} requires "
                "use_hierarchical_stop=true"
            )
        if config.hierarchical_stop_mode == "mcia_context" and not config.use_motion_instruction_attention:
            raise ValueError(
                "hierarchical_stop_mode='mcia_context' requires "
                "use_motion_instruction_attention=true"
            )
        if config.use_hierarchical_stop and config.hierarchical_stop_mode == "mcia_context":
            self.stop_aux_head: nn.Module | None = MCIAContextStopHead(
                hidden_dim=hidden_size,
                stop_hidden_dim=config.stop_hidden_dim,
            )
        elif config.use_hierarchical_stop and config.hierarchical_stop_mode == "text_query":
            self.stop_aux_head: nn.Module | None = TextConditionedStopHead(
                navigation_dim=stop_feature_dim,
                text_dim=hidden_size,
                attention_dim=config.stop_attention_dim,
                attention_heads=config.stop_attention_heads,
                hidden_dim=config.stop_hidden_dim,
                dropout=config.stop_dropout,
            )
        elif config.use_stop_aux_head or config.use_hierarchical_stop:
            self.stop_aux_head = nn.Sequential(
                nn.LayerNorm(stop_feature_dim),
                nn.Linear(stop_feature_dim, 1),
            )
        else:
            self.stop_aux_head = None
        if self.stop_aux_head is not None:
            # Initialising from an old checkpoint must preserve action logits
            # until the new auxiliary branch receives training updates.
            stop_output_layer = (
                self.stop_aux_head.output_layer
                if isinstance(
                    self.stop_aux_head,
                    (TextConditionedStopHead, MCIAContextStopHead),
                )
                else self.stop_aux_head[-1]
            )
            nn.init.zeros_(stop_output_layer.weight)
            nn.init.zeros_(stop_output_layer.bias)
        if config.use_hierarchical_stop:
            if config.use_stop_aux_head:
                raise ValueError(
                    "use_hierarchical_stop and use_stop_aux_head are mutually exclusive"
                )
            if config.num_actions < 2:
                raise ValueError("Hierarchical STOP requires STOP plus at least one movement action")
            if not 0.0 < config.hierarchical_stop_threshold < 1.0:
                raise ValueError("hierarchical_stop_threshold must be in (0, 1)")
            if not 0.0 < config.hierarchical_stop_initial_probability < 1.0:
                raise ValueError("hierarchical_stop_initial_probability must be in (0, 1)")
            initial_probability = config.hierarchical_stop_initial_probability
            nn.init.constant_(
                stop_output_layer.bias,
                math.log(initial_probability / (1.0 - initial_probability)),
            )

        if config.progress_mode not in {"direct", "text_query"}:
            raise ValueError("progress_mode must be 'direct' or 'text_query'")
        if config.use_progress_aux_head and config.progress_mode == "direct":
            self.progress_head: nn.Module | None = DirectProgressHead(
                stop_feature_dim,
                config.progress_hidden_dim,
                config.progress_dropout,
            )
        elif config.use_progress_aux_head:
            self.progress_head = TextQueryProgressHead(
                navigation_dim=stop_feature_dim,
                text_dim=hidden_size,
                attention_dim=config.progress_attention_dim,
                attention_heads=config.progress_attention_heads,
                hidden_dim=config.progress_hidden_dim,
                dropout=config.progress_dropout,
            )
        else:
            self.progress_head = None

        # M2 sends the 960-dimensional fused input directly into its GRU.
        # Retain fuse_proj for checkpoint-format compatibility, but freeze it
        # so DDP does not wait for gradients from an unused branch.
        if config.use_memory:
            self.fuse_proj.requires_grad_(False)
        if config.use_omtrack:
            # These compatibility modules are serialized for the common
            # checkpoint format but are not on OmTrack's forward path.
            self.fuse_proj.requires_grad_(False)
            self.action_head.requires_grad_(False)

        # ---- Apply per-stage freeze/unfreeze ----
        self._configure_trainable()
        if config.use_native1t_postfusion:
            self.enforce_native1t_trainable_contract()

        # ---- Gradient checkpointing ----
        if config.gradient_checkpointing:
            lm = self.backbone.language_model
            if hasattr(lm, "gradient_checkpointing_enable"):
                lm.gradient_checkpointing_enable()
            if hasattr(lm, "enable_input_require_grads"):
                lm.enable_input_require_grads()
            if hasattr(lm, "config"):
                lm.config.use_cache = False

    # ------------------------------------------------------------------
    # Trainable parameter management
    # ------------------------------------------------------------------

    def _has_trainable_vision(self) -> bool:
        """Whether the live current-image path must retain vision autograd."""
        c = self.config
        return (
            not c.freeze_vision
            or c.vision_train_last_n_layers > 0
            or c.vision_train_siglip_last_n_layers > 0
        )

    def enforce_task_aware_pooler_trainable_contract(self) -> None:
        """Freeze every inherited policy parameter for the pooler-only ablation."""
        if not self.config.task_aware_pooler_only:
            return
        if self.task_aware_pooler is None:
            raise RuntimeError(
                "task_aware_pooler_only is enabled but the Task-Aware pooler is missing"
            )
        self.requires_grad_(False)
        self.task_aware_pooler.requires_grad_(True)

    def _configure_trainable(self) -> None:
        c = self.config

        # Vision backbone
        if not c.freeze_vision:
            for p in self.backbone.vision_backbone.parameters():
                p.requires_grad = True
        elif c.vision_train_last_n_layers > 0:
            self._unfreeze_vision_tail(c.vision_train_last_n_layers)
        elif c.vision_train_siglip_last_n_layers > 0:
            self._unfreeze_siglip_vision_tail(
                c.vision_train_siglip_last_n_layers
            )

        # Projector
        if c.train_projector:
            for p in self.backbone.projector.parameters():
                p.requires_grad = True

        if c.use_lora and c.train_language_model:
            raise ValueError("use_lora and train_language_model are mutually exclusive")

        # Stages without LoRA keep the pretrained decoder frozen unless full
        # language-model fine-tuning is explicitly requested.
        if c.use_lora:
            target_modules: str | list[str] = "all-linear"
            if c.use_temporal_memory or c.use_omtrack:
                target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ]
            lora_config = LoraConfig(
                r=c.lora_rank,
                lora_alpha=c.lora_alpha,
                lora_dropout=c.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
                init_lora_weights="gaussian",
            )
            self.backbone.language_model = get_peft_model(
                self.backbone.language_model, lora_config
            )
        elif c.train_language_model:
            for parameter in self.backbone.language_model.parameters():
                parameter.requires_grad = True

        # The Task-Aware experiment is an isolated ablation: the parent
        # checkpoint supplies the complete policy, and only the newly added
        # current-frame pooler may receive gradients.
        self.enforce_task_aware_pooler_trainable_contract()

    def enforce_native1t_trainable_contract(self) -> None:
        """Reapply the stage freeze contract, including after checkpoint load."""
        c = self.config
        if not c.use_native1t_postfusion:
            return
        if (
            self.native1t_memory_tokenizer is None
            or self.native1t_memory_projector is None
            or self.native1t_mem_bos is None
        ):
            raise RuntimeError("Native1T modules are missing")
        self.requires_grad_(False)
        self.native1t_memory_projector.requires_grad_(True)
        self.native1t_mem_bos.requires_grad_(True)
        if c.native1t_stage == 1:
            self.native1t_memory_tokenizer.requires_grad_(True)
            return
        self.backbone.projector.requires_grad_(True)
        for pooler in (self.instruction_local_pooler, self.spatial_anchored_residual_pooler):
            if pooler is not None:
                pooler.requires_grad_(True)
        if self._uses_openfly_action_tokens_ar:
            # The autoregressive action surface is Qwen's tied LM head.  The
            # learned [ACT] query remains the only non-backbone policy token.
            self.action_query.requires_grad_(True)
        else:
            self.action_head.requires_grad_(True)
            self.fuse_proj.requires_grad_(True)
        self.action_query.requires_grad_(True)
        if c.vision_train_siglip_last_n_layers > 0:
            self._unfreeze_siglip_vision_tail(
                c.vision_train_siglip_last_n_layers
            )
        if c.action_chunk_horizon > 1:
            if (
                self.action_chunk_horizon_embeddings is None
                or self.action_chunk_norm is None
            ):
                raise RuntimeError("Native1T AC4 modules are missing")
            self.action_chunk_horizon_embeddings.requires_grad_(True)
            self.action_chunk_norm.requires_grad_(True)
        if self.segment_task_state is not None:
            self.segment_task_state.requires_grad_(True)
        if c.use_native1t_action_history:
            if (
                self.native1t_action_embedding is None
                or self.native1t_action_history_projector is None
                or self.native1t_action_bos is None
            ):
                raise RuntimeError("Native1T AH4 modules are missing")
            self.native1t_action_embedding.requires_grad_(True)
            self.native1t_action_history_projector.requires_grad_(True)
            self.native1t_action_bos.requires_grad_(True)
        for name, parameter in self.backbone.language_model.named_parameters():
            if "lora_" in name:
                parameter.requires_grad = True

    def _unfreeze_vision_tail(self, n_layers: int) -> None:
        """Unfreeze the final transformer blocks of each Prismatic vision tower."""
        towers = [self.backbone.vision_backbone.featurizer]
        fused_tower = getattr(self.backbone.vision_backbone, "fused_featurizer", None)
        if fused_tower is not None:
            towers.append(fused_tower)
        for tower in towers:
            blocks = getattr(tower, "blocks", None)
            if blocks is None or len(blocks) < n_layers:
                raise ValueError(f"Vision tower does not expose {n_layers} transformer blocks")
            for block in blocks[-n_layers:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True

    def _unfreeze_siglip_vision_tail(self, n_layers: int) -> None:
        """Unfreeze final output-contributing blocks of fused SigLIP only.

        Prismatic patches fused_featurizer.forward to return
        get_intermediate_layers(n={len(blocks) - 2}), i.e. the output after
        the second-to-last registered block.  The terminal registered block
        is therefore outside the visual feature path and must stay frozen;
        unfreezing it would create a dead parameter group.
        """
        fused_tower = getattr(self.backbone.vision_backbone, "fused_featurizer", None)
        blocks = getattr(fused_tower, "blocks", None)
        # `active_stop` is exclusive: the final registered block is not part
        # of Prismatic's penultimate-layer feature output.
        active_stop = 0 if blocks is None else len(blocks) - 1
        if blocks is None or n_layers > active_stop:
            raise ValueError(
                "Fused SigLIP tower does not expose "
                f"{n_layers} output-contributing transformer blocks"
            )
        for block in blocks[active_stop - n_layers:active_stop]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    def trainable_parameter_groups(
        self,
        *,
        head_lr: float,
        weight_decay: float,
        **extra: float,
    ) -> list[dict[str, Any]]:
        if self.config.task_aware_pooler_only:
            if self.task_aware_pooler is None:
                raise RuntimeError("Task-Aware pooler-only training has no pooler module")
            pooler_params = [
                parameter
                for parameter in self.task_aware_pooler.parameters()
                if parameter.requires_grad
            ]
            if not pooler_params:
                raise RuntimeError("Task-Aware pooler-only training has no trainable parameters")
            return [{
                "params": pooler_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
            }]

        groups: list[dict[str, Any]] = []

        # Group 1: action head + action_query + fuse_proj (always trainable)
        if self.config.use_omtrack:
            if self.omtrack_chunk_norm is None or self.omtrack_chunk_head is None:
                raise RuntimeError("OmTrack chunk head is missing")
            head_params = (
                [self.action_query]
                + list(self.omtrack_chunk_norm.parameters())
                + list(self.omtrack_chunk_head.parameters())
            )
            if self.omtrack_time_embedding is not None:
                head_params += list(self.omtrack_time_embedding.parameters())
        else:
            head_params = (
                list(self.action_head.parameters())
                + [self.action_query]
                + list(self.fuse_proj.parameters())
            )
        if self.action_chunk_horizon_embeddings is not None:
            head_params += list(self.action_chunk_horizon_embeddings.parameters())
        if self.action_chunk_norm is not None:
            head_params += list(self.action_chunk_norm.parameters())
        if self.stop_aux_head is not None:
            head_params += list(self.stop_aux_head.parameters())
        if self.instruction_local_pooler is not None:
            head_params += list(self.instruction_local_pooler.parameters())
        if self.spatial_anchored_residual_pooler is not None:
            head_params += list(self.spatial_anchored_residual_pooler.parameters())
        if self.transition_head is not None:
            head_params += list(self.transition_head.parameters())
        if self.progress_head is not None:
            head_params += list(self.progress_head.parameters())
        if self.motion_instruction_attention is not None:
            head_params += list(self.motion_instruction_attention.parameters())
        if self.segment_task_state is not None:
            head_params += list(self.segment_task_state.parameters())
        if self.frame_slot_embedding is not None:
            head_params += list(self.frame_slot_embedding.parameters())
        if self.temporal_adapter is not None:
            head_params += list(self.temporal_adapter.parameters())
        if self.prev_action_embed is not None:
            head_params += list(self.prev_action_embed.parameters())
        if self.gru_cell is not None:
            head_params += list(self.gru_cell.parameters())
        if self.memory_norm is not None:
            head_params += list(self.memory_norm.parameters())
        groups.append({
            "params": [p for p in head_params if p.requires_grad],
            "lr": head_lr,
            "weight_decay": weight_decay,
        })

        if self.temporal_memory_tokenizer is not None:
            tokenizer_params = list(self.temporal_memory_tokenizer.parameters())
            if self.temporal_memory_time_embedding is not None:
                tokenizer_params += list(self.temporal_memory_time_embedding.parameters())
            groups.append({
                "params": [parameter for parameter in tokenizer_params if parameter.requires_grad],
                "lr": extra.get("memory_tokenizer_lr", 1e-4),
                "weight_decay": weight_decay,
            })

        if self.temporal_memory_projector is not None:
            groups.append({
                "params": [
                    parameter for parameter in self.temporal_memory_projector.parameters()
                    if parameter.requires_grad
                ],
                "lr": extra.get("memory_projector_lr", 5e-5),
                "weight_decay": weight_decay,
            })

        if self.native1t_memory_tokenizer is not None:
            tokenizer_params = [
                p for p in self.native1t_memory_tokenizer.parameters() if p.requires_grad
            ]
            if tokenizer_params:
                groups.append({
                    "params": tokenizer_params,
                    "lr": extra.get("memory_tokenizer_lr", 1e-4),
                    "weight_decay": weight_decay,
                })
        if self.native1t_memory_projector is not None:
            memory_params = [
                p for p in self.native1t_memory_projector.parameters() if p.requires_grad
            ]
            if self.native1t_mem_bos is not None and self.native1t_mem_bos.requires_grad:
                memory_params.append(self.native1t_mem_bos)
            if memory_params:
                groups.append({
                    "params": memory_params,
                    "lr": extra.get("memory_projector_lr", 5e-5),
                    "weight_decay": weight_decay,
                })
        if self.native1t_action_embedding is not None:
            if (
                self.native1t_action_history_projector is None
                or self.native1t_action_bos is None
            ):
                raise RuntimeError("Native1T AH4 parameter group is incomplete")
            action_history_params = (
                list(self.native1t_action_embedding.parameters())
                + list(self.native1t_action_history_projector.parameters())
                + [self.native1t_action_bos]
            )
            groups.append({
                "params": [p for p in action_history_params if p.requires_grad],
                "lr": extra.get("action_history_lr", 2e-4),
                "weight_decay": weight_decay,
            })

        # Group 2: projector
        proj_params = [p for p in self.backbone.projector.parameters() if p.requires_grad]
        if self.motion_history_projector is not None:
            proj_params += [
                p for p in self.motion_history_projector.parameters() if p.requires_grad
            ]
        if proj_params:
            projector_lr = extra.get("projector_lr", head_lr)
            groups.append({
                "params": proj_params,
                "lr": projector_lr,
                "weight_decay": weight_decay,
            })

        # Group 3: vision backbone (if unfrozen)
        vis_params = [p for p in self.backbone.vision_backbone.parameters() if p.requires_grad]
        if vis_params:
            vision_lr = extra.get("vision_lr", 1e-6)
            groups.append({
                "params": vis_params,
                "lr": vision_lr,
                "weight_decay": weight_decay,
            })

        # Group 4: language model (LoRA or full)
        lm_params = [p for p in self.backbone.language_model.parameters() if p.requires_grad]
        if lm_params:
            lora_lr = extra.get("lora_lr", 5e-5)
            groups.append({
                "params": lm_params,
                "lr": lora_lr,
                "weight_decay": weight_decay,
            })

        if not groups:
            raise RuntimeError("No trainable parameters found")
        return groups

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def encode_frozen_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode the live current frame, retaining autograd only when required.

        Native1T Stage 2 normally uses this for a frozen current image and a
        separately pre-extracted history cache.  The opt-in SigLIP-tail
        ablation keeps that history frozen but must let gradients reach the
        live current frame and the rollout-appended current memory token.
        """
        if self._has_trainable_vision():
            return self.backbone.vision_backbone(pixel_values)
        with torch.no_grad():
            return self.backbone.vision_backbone(pixel_values)

    def _compress_history_feature(self, feature: torch.Tensor) -> torch.Tensor:
        """16x16 frozen feature grid -> 2x4 grid, before the projector."""
        token_count = feature.shape[1]
        grid_size = math.isqrt(token_count)
        if grid_size * grid_size != token_count:
            raise ValueError(f"Expected a square patch grid, got {token_count} tokens")
        grid = feature.reshape(1, grid_size, grid_size, feature.shape[-1]).permute(0, 3, 1, 2)
        pooled = torch.nn.functional.adaptive_avg_pool2d(grid, (2, 4))
        return pooled.permute(0, 2, 3, 1).reshape(1, 8, feature.shape[-1])

    @staticmethod
    def _pool_vision_grid(feature: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        """Pool [B,P,D] spatial features without mixing patch/channel memory order."""
        if feature.ndim != 3:
            raise ValueError(f"Expected [B,P,D] vision features, got {tuple(feature.shape)}")
        batch, token_count, feature_dim = feature.shape
        grid_size = math.isqrt(token_count)
        if grid_size * grid_size != token_count:
            raise ValueError(f"Expected a square patch grid, got {token_count} tokens")
        grid = feature.reshape(batch, grid_size, grid_size, feature_dim).permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(grid, output_size)
        return pooled.permute(0, 2, 3, 1).reshape(
            batch, output_size[0] * output_size[1], feature_dim
        ).to(dtype=feature.dtype)

    def native1t_visual_reducer_contract(self) -> dict[str, Any]:
        """Describe the reducer actually constructed for Native1T current frames.

        This is deliberately derived from live module instances as well as
        config flags.  A config value alone is not evidence that the module
        is on the Native1T forward path.
        """
        c = self.config
        if not c.use_native1t_postfusion:
            raise RuntimeError(
                "native1t_visual_reducer_contract requires Native1T PostFusion"
            )
        declared = (
            bool(c.use_task_aware_pooler),
            bool(c.use_instruction_local_pooler),
            bool(c.use_spatial_anchored_residual_pooler),
        )
        live = (
            getattr(self, "task_aware_pooler", None) is not None,
            getattr(self, "instruction_local_pooler", None) is not None,
            getattr(self, "spatial_anchored_residual_pooler", None) is not None,
        )
        if declared != live:
            raise RuntimeError(
                "Native1T visual reducer config/module mismatch: "
                f"declared={declared}, live={live}"
            )
        active = [
            (name, module)
            for name, module in (
                ("task_aware_mha", self.task_aware_pooler),
                ("instruction_local_weighted32", self.instruction_local_pooler),
                ("spatial_anchored_residual32", self.spatial_anchored_residual_pooler),
            )
            if module is not None
        ]
        if len(active) > 1:
            raise RuntimeError(
                "Native1T has multiple live current-frame reducers: "
                + ", ".join(name for name, _ in active)
            )
        target_tokens = int(c.visual_tokens_per_frame)
        if active:
            reducer, module = active[0]
            output_tokens = int(module.output_tokens)
            if reducer == "task_aware_mha":
                reducer = f"task_aware_mha_{output_tokens}"
            if output_tokens != target_tokens:
                raise RuntimeError(
                    "Native1T reducer/output budget mismatch: "
                    f"reducer={output_tokens}, config={target_tokens}"
                )
        elif target_tokens == 32:
            reducer = "avgpool32"
        elif target_tokens in {64, 128, 256}:
            reducer = f"avgpool{target_tokens}"
        else:
            raise RuntimeError(
                f"Native1T has no supported reducer for target={target_tokens}"
            )
        stage = int(c.native1t_stage)
        memory_slots = (
            int(c.native1t_stage1_memory_frames)
            if stage == 1
            else int(c.native1t_history_length)
        )
        return {
            "format": "openfly_native1t_current_visual_contract_v2",
            "native1t_stage": stage,
            "active_reducer": reducer,
            "current_raw_tokens": 256,
            "current_projected_tokens": 256,
            "current_output_tokens": target_tokens,
            "memory_slots_in_sequence": memory_slots,
            "visual_tokens_in_sequence": target_tokens + memory_slots,
            "reducer_call": "native1t_postfusion_current_unconditional",
        }

    def _reduce_visual_tokens(
        self,
        feature: torch.Tensor,
        *,
        instruction_embeddings: torch.Tensor | None = None,
        instruction_mask: torch.Tensor | None = None,
        history_tokens: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reduce current-frame projected tokens with the configured reducer."""
        if feature.ndim != 3:
            raise ValueError(f"Expected [B,P,D] vision features, got {tuple(feature.shape)}")
        target_tokens = int(self.config.visual_tokens_per_frame)
        input_tokens = int(feature.shape[1])
        if target_tokens <= 0:
            raise ValueError("visual_tokens_per_frame must be positive")
        task_aware_pooler = getattr(self, "task_aware_pooler", None)
        instruction_local_pooler = getattr(self, "instruction_local_pooler", None)
        spatial_anchored_residual_pooler = getattr(
            self, "spatial_anchored_residual_pooler", None
        )
        if spatial_anchored_residual_pooler is not None:
            if target_tokens != spatial_anchored_residual_pooler.output_tokens:
                raise ValueError(
                    "Spatially anchored residual pooler output and visual token "
                    f"budget disagree: pooler={spatial_anchored_residual_pooler.output_tokens}, "
                    f"target={target_tokens}"
                )
            if instruction_embeddings is None or instruction_mask is None:
                raise ValueError(
                    "Spatially anchored residual visual reduction requires "
                    "instruction embeddings and mask"
                )
            return spatial_anchored_residual_pooler(
                feature,
                instruction_embeddings,
                instruction_mask,
            )
        if instruction_local_pooler is not None:
            if target_tokens != instruction_local_pooler.output_tokens:
                raise ValueError(
                    "Instruction-guided local pooler output and visual token budget "
                    f"disagree: pooler={instruction_local_pooler.output_tokens}, "
                    f"target={target_tokens}"
                )
            if instruction_embeddings is None or instruction_mask is None:
                raise ValueError(
                    "Instruction-guided local visual reduction requires "
                    "instruction embeddings and mask"
                )
            return instruction_local_pooler(
                feature, instruction_embeddings, instruction_mask,
                history_tokens=history_tokens,
                history_mask=history_mask,
            )
        if task_aware_pooler is not None:
            if target_tokens != task_aware_pooler.output_tokens:
                raise ValueError(
                    "Task-Aware pooler output and visual token budget disagree: "
                    f"pooler={task_aware_pooler.output_tokens}, target={target_tokens}"
                )
            if instruction_embeddings is None or instruction_mask is None:
                raise ValueError(
                    "Task-Aware visual reduction requires instruction embeddings and mask"
                )
            return task_aware_pooler(
                feature,
                instruction_embeddings,
                instruction_mask,
            )
        if target_tokens == input_tokens:
            return feature
        if target_tokens > input_tokens:
            raise ValueError(
                "visual_tokens_per_frame cannot expand the native visual sequence: "
                f"input={input_tokens}, target={target_tokens}"
            )
        input_grid = math.isqrt(input_tokens)
        if input_grid * input_grid != input_tokens:
            raise ValueError(
                "visual token average pooling requires a square input grid: "
                f"input={input_tokens}, target={target_tokens}"
            )
        # Preserve the established 8x8 visual64 route.  For 32 tokens, a
        # 4x8 grid is the smallest factorized layout that keeps spatial
        # coverage balanced while retaining exactly 32 pooled tokens.
        output_size = {
            32: (4, 8),
            64: (8, 8),
            128: (8, 16),
            256: (16, 16),
        }.get(target_tokens)
        if output_size is None or output_size[0] * output_size[1] != target_tokens:
            raise ValueError(
                "visual token average pooling supports target budgets "
                "32 (4x8), 64 (8x8), 128 (8x16), and 256 (16x16); "
                f"got target={target_tokens}"
            )
        return self._pool_vision_grid(feature, output_size)
    @staticmethod
    def _fixed_sinusoidal_embedding(
        positions: torch.Tensor, hidden_dim: int
    ) -> torch.Tensor:
        """Parameter-free sinusoidal embeddings for positive relative times."""
        half = hidden_dim // 2
        frequency = torch.exp(
            torch.arange(half, device=positions.device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        angles = positions.float().unsqueeze(-1) * frequency.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if hidden_dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding.to(dtype=positions.dtype)

    # ------------------------------------------------------------------
    # OpenFly causal action-token head (v2)
    # ------------------------------------------------------------------

    def _openfly_language_model_embeddings(self) -> tuple[nn.Module, nn.Module]:
        """Return Qwen's input embeddings and tied full-vocabulary LM head."""
        language_model = self.backbone.language_model
        input_embeddings = language_model.get_input_embeddings()
        output_embeddings = language_model.get_output_embeddings()
        if not isinstance(input_embeddings, nn.Module) or not isinstance(output_embeddings, nn.Module):
            raise RuntimeError(
                "OpenFly action_tokens_ar_v2 requires language_model input embeddings "
                "and a callable causal-LM output head"
            )
        input_weight = getattr(input_embeddings, "weight", None)
        output_weight = getattr(output_embeddings, "weight", None)
        if input_weight is None or output_weight is None or input_weight.ndim != 2 or output_weight.ndim != 2:
            raise RuntimeError("OpenFly action_tokens_ar_v2 requires 2-D Qwen embedding/LM-head weights")
        if input_weight.shape[0] != output_weight.shape[0]:
            raise RuntimeError(
                "OpenFly action_tokens_ar_v2 requires matching input/output vocabulary sizes: "
                f"input={tuple(input_weight.shape)}, output={tuple(output_weight.shape)}"
            )
        return input_embeddings, output_embeddings

    def _initialize_openfly_action_lm_contract(self) -> None:
        """Validate and register OpenFly's reserved final-256-token contract."""
        input_embeddings, output_embeddings = self._openfly_language_model_embeddings()
        vocab_size = int(getattr(output_embeddings, "weight").shape[0])
        token_dim = int(self.config.openfly_action_token_dim)
        token_bins = int(self.config.openfly_action_token_bins)
        if token_dim != 8 or token_bins != 256:
            raise ValueError(
                "action_tokens_ar_v2 fixes the OpenFly token surface at 8 dimensions and 256 bins"
            )
        if vocab_size <= token_bins:
            raise RuntimeError(
                f"OpenFly action-token vocabulary needs > {token_bins} entries, got {vocab_size}"
            )
        eos_token_id = getattr(self.backbone.config.text_config, "eos_token_id", None)
        if eos_token_id is None:
            raise RuntimeError("OpenFly action_tokens_ar_v2 requires text_config.eos_token_id")
        eos_token_id = int(eos_token_id)
        action_token_ids = torch.arange(
            vocab_size - 1,
            vocab_size - token_bins - 1,
            -1,
            dtype=torch.long,
        )
        if eos_token_id < 0 or eos_token_id >= vocab_size or torch.any(action_token_ids == eos_token_id):
            raise RuntimeError(
                "OpenFly action-token EOS must be a valid non-action token: "
                f"eos={eos_token_id}, action_range=[{int(action_token_ids[-1])}, {int(action_token_ids[0])}]"
            )
        if int(getattr(input_embeddings, "weight").shape[0]) != vocab_size:
            raise RuntimeError("OpenFly action-token input embedding vocabulary drifted during initialization")
        self.register_buffer("_openfly_action_token_ids", action_token_ids, persistent=False)
        self._openfly_action_vocab_size = vocab_size
        self._openfly_action_eos_token_id = eos_token_id
        self._openfly_action_lm_contract = {
            "format": "openfly_action_tokens_ar_v2",
            "vocab_size": vocab_size,
            "action_token_begin_idx": vocab_size - token_bins - 1,
            "action_token_end_idx": vocab_size - 1,
            "eos_token_id": eos_token_id,
            "action_dim": token_dim,
            "num_bins": token_bins,
        }

    def _openfly_action_ids_from_classes(self, token_classes: torch.Tensor) -> torch.Tensor:
        """Map codec classes 0..255 to original OpenFly tail-vocabulary IDs."""
        if self._openfly_action_vocab_size is None:
            raise RuntimeError("OpenFly autoregressive action vocabulary was not initialized")
        if token_classes.ndim != 2 or token_classes.shape[1] != self.config.openfly_action_token_dim:
            raise ValueError(
                "OpenFly autoregressive token targets must have shape "
                f"[B,{self.config.openfly_action_token_dim}], got {tuple(token_classes.shape)}"
            )
        token_classes = token_classes.long()
        if torch.any(token_classes < 0) or torch.any(token_classes >= self.config.openfly_action_token_bins):
            raise ValueError("OpenFly autoregressive token targets must be integer classes in [0,255]")
        return self._openfly_action_vocab_size - token_classes - 1

    @staticmethod
    def _openfly_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        return position_ids.masked_fill(~attention_mask.bool(), 0)

    @staticmethod
    def _openfly_decoder_hidden(outputs: Any) -> torch.Tensor:
        return outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

    def _openfly_teacher_forced_action_tokens(
        self,
        decoder: nn.Module,
        multimodal: torch.Tensor,
        full_mask: torch.Tensor,
        token_classes: torch.Tensor,
        *,
        current_memory_token: torch.Tensor | None,
    ) -> dict[str, torch.Tensor | None]:
        """Run OpenFly's full-vocabulary causal-LM CE over 8 action tokens + EOS."""
        input_embeddings, output_embeddings = self._openfly_language_model_embeddings()
        target_ids = self._openfly_action_ids_from_classes(token_classes)
        action_embeds = input_embeddings(target_ids).to(dtype=multimodal.dtype)
        sequence = torch.cat([multimodal, action_embeds], dim=1)
        sequence_mask = torch.cat(
            [
                full_mask,
                torch.ones(
                    target_ids.shape,
                    dtype=full_mask.dtype,
                    device=full_mask.device,
                ),
            ],
            dim=1,
        )
        outputs = decoder(
            inputs_embeds=sequence,
            attention_mask=sequence_mask,
            position_ids=self._openfly_position_ids(sequence_mask),
            use_cache=False,
        )
        action_length = int(self.config.openfly_action_token_dim)
        prediction_hidden = self._openfly_decoder_hidden(outputs)[:, -(action_length + 1):, :]
        full_vocab_logits = output_embeddings(prediction_hidden)
        if self._openfly_action_eos_token_id is None:
            raise RuntimeError("OpenFly autoregressive EOS token was not initialized")
        eos = torch.full(
            (target_ids.shape[0], 1),
            self._openfly_action_eos_token_id,
            dtype=torch.long,
            device=target_ids.device,
        )
        lm_targets = torch.cat([target_ids, eos], dim=1)
        lm_loss_sum = F.cross_entropy(
            full_vocab_logits.float().reshape(-1, full_vocab_logits.shape[-1]),
            lm_targets.reshape(-1),
            reduction="sum",
        )
        action_logits = full_vocab_logits[:, :action_length, :].index_select(
            -1, self._openfly_action_token_ids
        )
        predicted_token_ids = full_vocab_logits[:, :action_length, :].argmax(dim=-1)
        return {
            "action_logits": action_logits,
            "stop_logits": None,
            "progress": None,
            "transition_logits": None,
            "openfly_lm_loss_sum": lm_loss_sum,
            "openfly_lm_loss_tokens": torch.tensor(
                lm_targets.numel(), dtype=torch.long, device=lm_targets.device
            ),
            "openfly_predicted_token_ids": predicted_token_ids,
            "openfly_eos_predicted_token_ids": full_vocab_logits[:, -1, :].argmax(dim=-1),
            "current_memory_token": current_memory_token,
        }

    def _openfly_generate_action_tokens(
        self,
        decoder: nn.Module,
        multimodal: torch.Tensor,
        full_mask: torch.Tensor,
        *,
        current_memory_token: torch.Tensor | None,
    ) -> dict[str, torch.Tensor | None]:
        """Greedily generate eight OpenFly tokens through Qwen's full LM head.

        This mirrors ``OpenVLAForActionPrediction.generate(max_new_tokens=8)``:
        the argmax is deliberately over the *full* vocabulary, not a masked
        256-way classifier.  The codec performs OpenFly's later clipping and
        codebook fallback when those IDs are decoded.
        """
        input_embeddings, output_embeddings = self._openfly_language_model_embeddings()
        running_embeds = multimodal
        running_mask = full_mask
        outputs = decoder(
            inputs_embeds=running_embeds,
            attention_mask=running_mask,
            position_ids=self._openfly_position_ids(running_mask),
            use_cache=True,
        )
        hidden = self._openfly_decoder_hidden(outputs)
        past_key_values = getattr(outputs, "past_key_values", None)
        step_logits: list[torch.Tensor] = []
        generated_token_ids: list[torch.Tensor] = []
        action_length = int(self.config.openfly_action_token_dim)
        for step in range(action_length):
            full_vocab_logits = output_embeddings(hidden[:, -1, :])
            step_logits.append(full_vocab_logits.index_select(-1, self._openfly_action_token_ids))
            next_token_ids = full_vocab_logits.argmax(dim=-1)
            generated_token_ids.append(next_token_ids)
            if step + 1 == action_length:
                break
            next_embed = input_embeddings(next_token_ids[:, None]).to(dtype=running_embeds.dtype)
            running_mask = torch.cat(
                [
                    running_mask,
                    torch.ones(
                        (running_mask.shape[0], 1),
                        dtype=running_mask.dtype,
                        device=running_mask.device,
                    ),
                ],
                dim=1,
            )
            if past_key_values is None:
                # Keep a correct, backend-neutral fallback for a decoder that
                # does not expose a KV cache.
                running_embeds = torch.cat([running_embeds, next_embed], dim=1)
                outputs = decoder(
                    inputs_embeds=running_embeds,
                    attention_mask=running_mask,
                    position_ids=self._openfly_position_ids(running_mask),
                    use_cache=True,
                )
            else:
                outputs = decoder(
                    inputs_embeds=next_embed,
                    attention_mask=running_mask,
                    position_ids=self._openfly_position_ids(running_mask)[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            hidden = self._openfly_decoder_hidden(outputs)
            past_key_values = getattr(outputs, "past_key_values", None)
        return {
            "action_logits": torch.stack(step_logits, dim=1),
            "stop_logits": None,
            "progress": None,
            "transition_logits": None,
            "openfly_generated_token_ids": torch.stack(generated_token_ids, dim=1),
            "current_memory_token": current_memory_token,
        }

    def forward(
        self,
        input_ids: torch.Tensor,           # [B, T]
        attention_mask: torch.Tensor,       # [B, T]
        pixel_values: torch.Tensor | None = None,  # image path; omitted for cached TM-VLA features
        frame_valid_mask: torch.Tensor | None = None,  # [B, F], False for repeated/fallback history
        frame_batch_indices: torch.Tensor | None = None,  # [N real frames]
        frame_slot_indices: torch.Tensor | None = None,   # old/middle/recent/current
        precomputed_vision_features: torch.Tensor | None = None,
        precomputed_current_memory_token: torch.Tensor | None = None,
        cached_memory_tokens: torch.Tensor | None = None,
        history_valid_mask: torch.Tensor | None = None,
        native1t_memory_pixel_values: torch.Tensor | None = None,
        native1t_memory_features: torch.Tensor | None = None,
        native1t_memory_valid_mask: torch.Tensor | None = None,
        native1t_action_history: torch.Tensor | None = None,
        native1t_action_history_valid_mask: torch.Tensor | None = None,
        omtrack_history_features: torch.Tensor | None = None,
        omtrack_history_valid_mask: torch.Tensor | None = None,
        h16_frozen_pool_history_features: torch.Tensor | None = None,
        h16_frozen_pool_history_valid_mask: torch.Tensor | None = None,
        multimodal_left_padding: torch.Tensor | None = None,
        prev_actions: torch.Tensor | None = None,  # [B] action_id from previous step
        memory_state: torch.Tensor | None = None,   # [B, memory_hidden_dim] for M2
        return_stop_aux: bool = False,
        instruction_token_mask: torch.Tensor | None = None,
        segment_token_ids: torch.Tensor | None = None,
        segment_belief_prev: torch.Tensor | None = None,
        task_gru_state_prev: torch.Tensor | None = None,
        hard_pointer_stage: torch.Tensor | None = None,
        openfly_action_token_targets: torch.Tensor | None = None,
        return_auxiliary: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Return action logits [B, num_actions] (and optional new_memory_state for M2)."""
        c = self.config
        B = input_ids.shape[0]
        variable_visual_sequences: list[torch.Tensor] | None = None
        current_memory_token: torch.Tensor | None = None
        action_history_token: torch.Tensor | None = None
        native1t_postfusion = False
        native1t_reducer_applied = False
        native1t_reducer_contract: dict[str, Any] | None = None
        # Reuse the existing frozen Qwen token representations as the
        # instruction-conditioned query source.  This is computed once per
        # trajectory step and is shared by the current-frame pooler and the
        # unchanged multimodal sequence construction below.
        input_embeds = self.backbone.language_model.get_input_embeddings()(input_ids)

        # ---- Step 1: Multi-frame visual encoding ----
        if c.use_native1t_postfusion:
            native1t_postfusion = True
            if (
                self.native1t_memory_tokenizer is None
                or self.native1t_memory_projector is None
                or self.native1t_mem_bos is None
            ):
                raise RuntimeError("Native1T modules are missing")
            if c.native1t_stage == 1:
                if native1t_memory_pixel_values is None or native1t_memory_valid_mask is None:
                    raise ValueError("Native1T Stage 1 requires nine memory RGB frames and mask")
                expected = (B, c.native1t_stage1_memory_frames)
                if tuple(native1t_memory_valid_mask.shape) != expected:
                    raise ValueError(f"Native1T Stage 1 mask must have shape {expected}")
                if native1t_memory_pixel_values.ndim != 5 or tuple(
                    native1t_memory_pixel_values.shape[:2]
                ) != expected:
                    raise ValueError("Native1T Stage 1 RGB must have shape [B,9,C,H,W]")
                flat_pixels = native1t_memory_pixel_values.reshape(
                    B * c.native1t_stage1_memory_frames,
                    *native1t_memory_pixel_values.shape[2:],
                )
                flat_raw = self.encode_frozen_vision(flat_pixels)
                patch_count = flat_raw.shape[1]
                memory_raw = self.native1t_memory_tokenizer(flat_raw.reshape(
                    B, c.native1t_stage1_memory_frames, patch_count,
                    self._vision_feature_size,
                ))
                # I_t is the final Stage-1 memory frame.  Reuse its raw feature
                # for the full 256-token current path: one vision forward/frame.
                current_raw = flat_raw.reshape(
                    B, c.native1t_stage1_memory_frames, patch_count,
                    self._vision_feature_size,
                )[:, -1]
            else:
                if native1t_memory_features is None or native1t_memory_valid_mask is None:
                    raise ValueError("Native1T Stage 2 requires cached history and mask")
                expected = (B, c.native1t_history_length, self._vision_feature_size)
                if tuple(native1t_memory_features.shape) != expected:
                    raise ValueError(
                        f"Native1T cached memory must have shape {expected}, got "
                        f"{tuple(native1t_memory_features.shape)}"
                    )
                if tuple(native1t_memory_valid_mask.shape) != expected[:2]:
                    raise ValueError("Native1T Stage 2 mask has the wrong shape")
                if pixel_values is None and precomputed_vision_features is None:
                    raise ValueError("Native1T Stage 2 requires current RGB or raw feature")
                current_raw = (
                    precomputed_vision_features
                    if precomputed_vision_features is not None
                    else self.encode_frozen_vision(pixel_values)
                )
                memory_raw = native1t_memory_features.to(
                    device=current_raw.device,
                    dtype=self.native1t_memory_tokenizer.query.dtype,
                )
                # Compute M_t from the same current raw feature for the online
                # deque, but never include it in action_t's history.
                current_memory_token = self.native1t_memory_tokenizer(
                    current_raw.unsqueeze(1).to(
                        dtype=self.native1t_memory_tokenizer.query.dtype
                    )
                ).squeeze(1)
                if c.use_native1t_action_history:
                    if (
                        self.native1t_action_embedding is None
                        or self.native1t_action_history_projector is None
                        or self.native1t_action_bos is None
                    ):
                        raise RuntimeError("Native1T AH4 modules are missing")
                    expected_action_shape = (B, c.native1t_action_history_length)
                    if (
                        native1t_action_history is None
                        or native1t_action_history_valid_mask is None
                    ):
                        raise ValueError("Native1T AH4 requires actions and valid mask")
                    if tuple(native1t_action_history.shape) != expected_action_shape:
                        raise ValueError(
                            f"Native1T AH4 actions must have shape {expected_action_shape}"
                        )
                    if tuple(native1t_action_history_valid_mask.shape) != expected_action_shape:
                        raise ValueError(
                            f"Native1T AH4 mask must have shape {expected_action_shape}"
                        )
                    action_valid = native1t_action_history_valid_mask.to(
                        device=current_raw.device, dtype=torch.bool
                    )
                    action_ids = native1t_action_history.to(
                        device=current_raw.device, dtype=torch.long
                    )
                    if torch.any((action_ids[action_valid] < 0) | (action_ids[action_valid] >= c.num_actions)):
                        raise ValueError("Native1T AH4 contains an invalid action id")
                    safe_action_ids = action_ids.masked_fill(~action_valid, 0)
                    action_embeds = self.native1t_action_embedding(safe_action_ids)
                    action_embeds = action_embeds * action_valid.unsqueeze(-1).to(
                        action_embeds.dtype
                    )
                    action_input = torch.cat(
                        [
                            action_embeds.reshape(B, -1),
                            action_valid.to(action_embeds.dtype),
                        ],
                        dim=-1,
                    )
                    action_history_token = self.native1t_action_history_projector(
                        action_input
                    ).unsqueeze(1)
            if current_raw.ndim != 3 or int(current_raw.shape[1]) != 256:
                raise RuntimeError(
                    "Native1T current raw visual contract requires [B,256,D], got "
                    f"{tuple(current_raw.shape)}"
                )
            current = self.backbone.projector(current_raw)
            # Keep the current-frame interface identical to the selected
            # Base model.  This is a no-op for 256T, preserves an explicit
            # instruction-conditioned pooler when one is enabled, and applies
            # the ordinary 4x8 AvgPool32 reducer for the plain 32T baseline.
            memory = self.native1t_memory_projector(memory_raw)
            history_mask = native1t_memory_valid_mask.to(
                device=memory.device, dtype=torch.bool
            )
            memory = memory * history_mask.unsqueeze(-1).to(memory.dtype)
            native1t_reducer_contract = self.native1t_visual_reducer_contract()
            current = self._reduce_visual_tokens(
                current,
                history_tokens=memory if c.history_conditioned_local_pooler else None,
                history_mask=history_mask if c.history_conditioned_local_pooler else None,
                instruction_embeddings=input_embeds,
                instruction_mask=(
                    instruction_token_mask
                    if instruction_token_mask is not None
                    else attention_mask
                ),
            )
            native1t_reducer_applied = True
            all_visual = current
            visual_mask = torch.ones(
                B, current.shape[1], dtype=torch.bool, device=current.device
            )
        elif c.use_h16_frozen_pool:
            if (
                h16_frozen_pool_history_features is None
                or h16_frozen_pool_history_valid_mask is None
            ):
                raise ValueError("H16-1T-FrozenPool requires history features and mask")
            expected = (B, c.h16_frozen_pool_history_length, self._vision_feature_size)
            if tuple(h16_frozen_pool_history_features.shape) != expected:
                raise ValueError(
                    f"H16 frozen history must have shape {expected}, got "
                    f"{tuple(h16_frozen_pool_history_features.shape)}"
                )
            if tuple(h16_frozen_pool_history_valid_mask.shape) != (
                B, c.h16_frozen_pool_history_length
            ):
                raise ValueError("H16 frozen history mask has the wrong shape")
            if pixel_values is None and precomputed_vision_features is None:
                raise ValueError("H16-1T-FrozenPool requires current RGB or feature")
            if pixel_values is not None and pixel_values.ndim == 5:
                if pixel_values.shape[1] != 1:
                    raise ValueError("H16-1T-FrozenPool accepts one current RGB")
                pixel_values = pixel_values[:, 0]
            current_raw = (
                precomputed_vision_features
                if precomputed_vision_features is not None
                else self.encode_frozen_vision(pixel_values)
            )
            history_raw = h16_frozen_pool_history_features.to(
                device=current_raw.device, dtype=current_raw.dtype
            )
            history_projected = self.backbone.projector(history_raw)
            current_projected = self.backbone.projector(current_raw)
            history_dt = torch.arange(
                c.h16_frozen_pool_history_length, 0, -1,
                device=history_projected.device, dtype=history_projected.dtype,
            )
            history_projected = history_projected + self._fixed_sinusoidal_embedding(
                history_dt, self._hidden_size
            )[None]
            history_mask = h16_frozen_pool_history_valid_mask.to(
                history_projected.device, dtype=torch.bool
            )
            history_projected = history_projected * history_mask.unsqueeze(-1).to(
                history_projected.dtype
            )
            all_visual = torch.cat([history_projected, current_projected], dim=1)
            visual_mask = torch.cat([
                history_mask,
                torch.ones(B, current_projected.shape[1], dtype=torch.bool,
                           device=current_projected.device),
            ], dim=1)
        elif c.use_omtrack:
            if omtrack_history_features is None or omtrack_history_valid_mask is None:
                raise ValueError("OmTrack requires sparse history features and a valid mask")
            expected = (
                B,
                c.omtrack_history_length,
                c.omtrack_history_tokens_per_frame,
                self._vision_feature_size,
            )
            if tuple(omtrack_history_features.shape) != expected:
                raise ValueError(
                    f"omtrack_history_features must have shape {expected}, got "
                    f"{tuple(omtrack_history_features.shape)}"
                )
            if tuple(omtrack_history_valid_mask.shape) != (
                B, c.omtrack_history_length
            ):
                raise ValueError("OmTrack history mask has the wrong shape")
            if pixel_values is None and precomputed_vision_features is None:
                raise ValueError("OmTrack requires one current RGB or precomputed feature")
            if pixel_values is not None and pixel_values.ndim == 5:
                if pixel_values.shape[1] != 1:
                    raise ValueError("OmTrack accepts exactly one current RGB frame")
                pixel_values = pixel_values[:, 0]
            current_raw = (
                precomputed_vision_features
                if precomputed_vision_features is not None
                else self.encode_frozen_vision(pixel_values)
            )
            current_grid = math.isqrt(c.omtrack_current_tokens)
            if current_grid * current_grid != c.omtrack_current_tokens:
                raise ValueError("OmTrack current token count must be a square")
            current_raw = self._pool_vision_grid(
                current_raw, (current_grid, current_grid)
            )
            history_raw = omtrack_history_features.to(
                device=current_raw.device, dtype=current_raw.dtype
            )
            history_projected = self.backbone.projector(
                history_raw.reshape(
                    B * c.omtrack_history_length,
                    c.omtrack_history_tokens_per_frame,
                    self._vision_feature_size,
                )
            ).reshape(
                B,
                c.omtrack_history_length,
                c.omtrack_history_tokens_per_frame,
                self._hidden_size,
            )
            current_projected = self.backbone.projector(current_raw)
            if self.omtrack_time_embedding is None:
                raise RuntimeError("OmTrack relative-time embedding is missing")
            # Oldest history is dt=history_length, newest is dt=1; current is dt=0.
            history_dt = torch.arange(
                c.omtrack_history_length, 0, -1,
                device=history_projected.device, dtype=torch.long,
            )
            history_projected = history_projected + self.omtrack_time_embedding(
                history_dt
            )[None, :, None, :]
            current_projected = current_projected + self.omtrack_time_embedding.weight[0][
                None, None, :
            ]
            history_mask = omtrack_history_valid_mask.to(
                history_projected.device, dtype=torch.bool
            )
            history_projected = history_projected * history_mask[:, :, None, None].to(
                history_projected.dtype
            )
            all_visual = torch.cat(
                [history_projected.reshape(B, -1, self._hidden_size), current_projected],
                dim=1,
            )
            visual_mask = torch.cat(
                [
                    history_mask[:, :, None].expand(
                        B, c.omtrack_history_length, c.omtrack_history_tokens_per_frame
                    ).reshape(B, -1),
                    torch.ones(
                        B, c.omtrack_current_tokens, dtype=torch.bool,
                        device=history_projected.device,
                    ),
                ],
                dim=1,
            )
        elif c.use_temporal_memory:
            if (
                cached_memory_tokens is None
                or history_valid_mask is None
                or self.temporal_memory_tokenizer is None
                or self.temporal_memory_projector is None
                or self.temporal_memory_time_embedding is None
            ):
                raise ValueError(
                    "TM-VLA requires cached_memory_tokens and history_valid_mask"
                )
            if pixel_values is None and precomputed_vision_features is None:
                raise ValueError(
                    "Native TM-VLA requires current pixel_values or precomputed vision features"
                )
            if pixel_values is not None and pixel_values.ndim == 5:
                if pixel_values.shape[1] != 1:
                    raise ValueError("Native TM-VLA accepts exactly one current RGB frame")
                pixel_values = pixel_values[:, 0]
            expected_memory_shape = (
                B, c.temporal_memory_num_history, self._vision_feature_size,
            )
            if tuple(cached_memory_tokens.shape) != expected_memory_shape:
                raise ValueError(
                    f"cached_memory_tokens must have shape {expected_memory_shape}, "
                    f"got {tuple(cached_memory_tokens.shape)}"
                )
            if tuple(history_valid_mask.shape) != (B, c.temporal_memory_num_history):
                raise ValueError(
                    "history_valid_mask must have shape "
                    f"[B,{c.temporal_memory_num_history}]"
                )
            model_dtype = self.temporal_memory_tokenizer.query.dtype
            cached_memory_tokens = cached_memory_tokens.to(dtype=model_dtype)
            history_valid_mask = history_valid_mask.to(
                device=cached_memory_tokens.device, dtype=torch.bool
            )
            # The frozen tower runs exactly once for the current RGB. During
            # stream training an entire [B,T] window is encoded in one large
            # batch and supplied here, avoiding T small serial vision calls.
            vis_feats = (
                precomputed_vision_features
                if precomputed_vision_features is not None
                else self.encode_frozen_vision(pixel_values)
            )
            if vis_feats.ndim != 3 or vis_feats.shape[0] != B or vis_feats.shape[-1] != self._vision_feature_size:
                raise ValueError(
                    "TM-VLA vision features must have shape "
                    f"[B,P,{self._vision_feature_size}], got {tuple(vis_feats.shape)}"
                )
            current_memory_token = precomputed_current_memory_token
            if current_memory_token is None:
                current_memory_token = self.temporal_memory_tokenizer(
                    vis_feats.unsqueeze(1)
                ).squeeze(1)
            elif tuple(current_memory_token.shape) != (B, self._vision_feature_size):
                raise ValueError(
                    "precomputed_current_memory_token must have shape "
                    f"{(B, self._vision_feature_size)}, got "
                    f"{tuple(current_memory_token.shape)}"
                )
            time_indices = torch.arange(
                c.temporal_memory_num_history,
                device=cached_memory_tokens.device,
                dtype=torch.long,
            )
            raw_memory = (
                cached_memory_tokens
                + self.temporal_memory_time_embedding(time_indices)[None]
            )
            memory = self.temporal_memory_projector(raw_memory)
            memory = memory * history_valid_mask.unsqueeze(-1).to(memory.dtype)
            current = self.backbone.projector(vis_feats)
            all_visual = torch.cat([memory, current], dim=1)
            visual_mask = torch.cat(
                [
                    history_valid_mask,
                    torch.ones(
                        B, current.shape[1], dtype=torch.bool, device=current.device
                    ),
                ],
                dim=1,
            )
        elif c.visual_history_mode != "none":
            if pixel_values is None:
                raise ValueError("Visual-history input requires pixel_values")
            if frame_batch_indices is None or frame_slot_indices is None:
                raise ValueError("Variable visual history requires frame batch/slot indices")
            frame_batch_indices = frame_batch_indices.to(pixel_values.device, dtype=torch.long)
            frame_slot_indices = frame_slot_indices.to(pixel_values.device, dtype=torch.long)
            features = (
                precomputed_vision_features
                if precomputed_vision_features is not None
                else self.encode_frozen_vision(pixel_values)
            )
            if features.shape[0] != frame_batch_indices.numel() or features.shape[0] != frame_slot_indices.numel():
                raise ValueError("Real-frame feature count does not match frame indices")
            variable_visual_sequences = []
            for row in range(B):
                selected = (frame_batch_indices == row).nonzero(as_tuple=False).flatten()
                if not len(selected):
                    raise ValueError("Every sample must contain its real current frame")
                row_slots = frame_slot_indices[selected]
                if int(row_slots[-1].item()) != 3 or (row_slots[:-1] == 3).any():
                    raise ValueError("Current frame must be the final and only slot 3")
                expected_slots = torch.arange(
                    4 - len(selected), 4, device=row_slots.device, dtype=row_slots.dtype
                )
                if not torch.equal(row_slots, expected_slots):
                    raise ValueError("Recent-3 frame slots must be consecutive and right-aligned")

                if c.visual_history_mode == "motion_recent3":
                    if self.motion_history_projector is None:
                        raise RuntimeError("Motion-Recent3 has no motion projector")
                    raw_sequence = features[selected]
                    current = self.backbone.projector(raw_sequence[-1:])
                    # Earliest-pad only at the frozen feature level so the
                    # three delta slots are always [old, middle, recent].
                    # t=0: [F0,F0,F0,F0], t=1: [F0,F0,F0,F1],
                    # t=2: [F0,F0,F1,F2]. Missing motion evidence is exactly 0.
                    missing = 4 - len(selected)
                    if missing:
                        earliest = raw_sequence[:1].expand(missing, -1, -1)
                        raw_sequence = torch.cat([earliest, raw_sequence], dim=0)
                    pooled_deltas = [
                        self._compress_history_feature(
                            (raw_sequence[index + 1] - raw_sequence[index]).unsqueeze(0)
                        )
                        for index in range(3)
                    ]
                    motion_input = torch.cat(pooled_deltas, dim=-1)
                    motion = self.motion_history_projector(motion_input)
                    variable_visual_sequences.append(torch.cat([motion, current], dim=1))
                    continue

                if self.frame_slot_embedding is None:
                    raise RuntimeError("Appearance history requires frame-slot embeddings")
                blocks: list[torch.Tensor] = []
                for feature_index, slot_tensor in zip(selected.tolist(), row_slots):
                    slot = int(slot_tensor.item())
                    raw = features[feature_index:feature_index + 1]
                    projector_input = raw if slot == 3 else self._compress_history_feature(raw)
                    projected = self.backbone.projector(projector_input)
                    blocks.append(self.frame_slot_embedding.add_slot(projected, slot))
                variable_visual_sequences.append(torch.cat(blocks, dim=1))
        elif c.is_multi_frame and pixel_values.dim() == 5:
            # pixel_values: [B, F, C, H, W]
            num_input_frames = pixel_values.shape[1]
            flat_pv = pixel_values.reshape(B * num_input_frames, *pixel_values.shape[2:])

            # Stage 3 keeps the tower frozen except for its final blocks, so
            # vision autograd must remain active whenever that tail is enabled.
            with torch.set_grad_enabled(self._has_trainable_vision()):
                vis_feats = self.backbone.vision_backbone(flat_pv)  # [B*F, 256, vis_dim]
            projected = self.backbone.projector(vis_feats)             # [B*F, 256, hidden]

            projected = projected.reshape(B, num_input_frames, projected.shape[1], -1)

            if frame_valid_mask is None:
                # Backward-compatible for direct callers with complete history.
                frame_valid_mask = torch.ones(
                    B, num_input_frames, dtype=torch.bool, device=projected.device
                )
            else:
                if tuple(frame_valid_mask.shape) != (B, num_input_frames):
                    raise ValueError(
                        "frame_valid_mask must have shape "
                        f"{(B, num_input_frames)}, got {tuple(frame_valid_mask.shape)}"
                    )
                frame_valid_mask = frame_valid_mask.to(projected.device, dtype=torch.bool)
            if not frame_valid_mask[:, -1].all():
                raise ValueError("The current frame (last slot) must always be valid")

            # History frames (0..F-2): avg_pool_2d → reduce to 64 tokens each
            history = projected[:, :num_input_frames - 1]
            current = projected[:, num_input_frames - 1]

            # Derive the spatial grid from the actual token count (16x16 for
            # 224px / patch 14) so a resolution change cannot silently corrupt
            # the pooling reshape.
            tokens_per_frame = projected.shape[2]
            H_patches = W_patches = math.isqrt(tokens_per_frame)
            if H_patches * W_patches != tokens_per_frame:
                raise ValueError(
                    f"Visual tokens per frame ({tokens_per_frame}) is not a "
                    "square grid; cannot pool frame history"
                )
            num_history = num_input_frames - 1
            history_rs = history.reshape(B * num_history, H_patches, W_patches, -1).permute(0, 3, 1, 2)
            history_pooled = torch.nn.functional.avg_pool2d(
                history_rs, c.history_pool_size, c.history_pool_stride
            )  # [B*(F-1), hidden, 8, 8]
            pooled_tokens = history_pooled.shape[-2] * history_pooled.shape[-1]
            history_pooled = history_pooled.permute(0, 2, 3, 1).reshape(
                B, num_history, pooled_tokens, -1
            )

            frame_blocks = list(history_pooled.unbind(dim=1)) + [current]
            if self.frame_slot_embedding is not None:
                frame_blocks = self.frame_slot_embedding(frame_blocks)
            history_blocks = frame_blocks[:-1]
            current = frame_blocks[-1]
            if self.temporal_adapter is not None:
                current, _temporal_context = self.temporal_adapter(
                    history_blocks, current, frame_valid_mask
                )

            # Concat remains oldest history → newest history → current;
            # FE and TA do not alter M0's 448-token LLM interface.
            all_visual = torch.cat([*history_blocks, current], dim=1)
            visual_mask = torch.cat(
                [
                    frame_valid_mask[:, :-1, None].expand(B, num_history, pooled_tokens).reshape(B, -1),
                    frame_valid_mask[:, -1:, None].expand(B, 1, tokens_per_frame).reshape(B, -1),
                ],
                dim=1,
            )
            # Also zero invalid embeddings.  The attention mask is authoritative;
            # zeroing prevents accidental leakage in alternative decoder backends.
            all_visual = all_visual * visual_mask.unsqueeze(-1).to(all_visual.dtype)
        else:
            # Single-frame: either run the live Prismatic vision towers or
            # consume their exact frozen pre-merger output from the aligned
            # OpenFly cache.  The latter must never fall through to RGB.
            if precomputed_vision_features is not None:
                if pixel_values is not None:
                    raise ValueError(
                        "Single-frame input cannot provide both pixel_values "
                        "and precomputed_vision_features"
                    )
                vis_feats = precomputed_vision_features
            else:
                if pixel_values is None:
                    raise ValueError(
                        "Single-frame input requires pixel_values or "
                        "precomputed_vision_features"
                    )
                if pixel_values.dim() == 5:
                    if pixel_values.shape[1] != 1:
                        raise ValueError(
                            f"Single-frame model (num_frames={c.num_frames}) received "
                            f"{pixel_values.shape[1]} frames — the collator and model "
                            "config disagree"
                        )
                    pixel_values = pixel_values[:, -1]
                with torch.set_grad_enabled(self._has_trainable_vision()):
                    vis_feats = self.backbone.vision_backbone(pixel_values)  # [B, 256, vis_dim]
            if vis_feats.ndim != 3 or vis_feats.shape[-1] != self._vision_feature_size:
                raise ValueError(
                    "Single-frame pre-merger vision features must have shape "
                    f"[B, 256, {self._vision_feature_size}], got {tuple(vis_feats.shape)}"
                )
            all_visual = self.backbone.projector(vis_feats)               # [B, 256, hidden]
            all_visual = self._reduce_visual_tokens(
                all_visual,
                instruction_embeddings=input_embeds,
                instruction_mask=(
                    instruction_token_mask
                    if instruction_token_mask is not None
                    else attention_mask
                ),
            )
            visual_mask = torch.ones(
                B, all_visual.shape[1], dtype=torch.bool, device=all_visual.device
            )

        # ---- Step 2: Build multimodal embeddings ----
        if native1t_postfusion:
            if not native1t_reducer_applied or native1t_reducer_contract is None:
                raise RuntimeError(
                    "Native1T current reducer contract failed: reducer was not applied"
                )
            expected_tokens = int(c.visual_tokens_per_frame)
            if all_visual.ndim != 3 or int(all_visual.shape[1]) != expected_tokens:
                raise RuntimeError(
                    "Native1T current reducer contract failed: expected "
                    f"[B,{expected_tokens},D], got {tuple(all_visual.shape)}"
                )
            if native1t_reducer_contract["current_output_tokens"] != expected_tokens:
                raise RuntimeError(
                    "Native1T current reducer contract failed: reducer metadata "
                    f"reports {native1t_reducer_contract['current_output_tokens']} "
                    f"but config requests {expected_tokens}"
                )
        if native1t_postfusion:
            # Exact contract: BOS, current visual budget, instruction, MEM_BOS,
            # visual-memory-16, optional ACTION_BOS + AH4-1T, then [ACT].
            sequences = []
            masks = []
            for row in range(B):
                valid_text = input_embeds[row, attention_mask[row].bool()]
                aq = self.action_query.to(valid_text.dtype)
                mem_bos = self.native1t_mem_bos[0].to(valid_text.dtype)
                valid_memory = memory[row, history_mask[row]]
                sequence_parts = [
                    valid_text[:1], all_visual[row], valid_text[1:],
                    mem_bos, valid_memory,
                ]
                if c.use_native1t_action_history:
                    if action_history_token is None or self.native1t_action_bos is None:
                        raise RuntimeError("Native1T AH4 token was not constructed")
                    sequence_parts.extend([
                        self.native1t_action_bos[0].to(valid_text.dtype),
                        action_history_token[row].to(valid_text.dtype),
                    ])
                sequence_parts.append(aq)
                sequences.append(torch.cat(sequence_parts, dim=0))
                masks.append(torch.ones(
                    sequences[-1].shape[0], dtype=torch.bool,
                    device=sequences[-1].device,
                ))
            max_sequence = max(sequence.shape[0] for sequence in sequences)
            multimodal = input_embeds.new_zeros(B, max_sequence, self._hidden_size)
            full_mask = attention_mask.new_zeros(B, max_sequence)
            for row, (sequence, sequence_mask) in enumerate(zip(sequences, masks)):
                pad = max_sequence - sequence.shape[0]
                multimodal[row, pad:] = sequence
                full_mask[row, pad:] = sequence_mask.to(full_mask.dtype)
            position_ids = full_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(~full_mask.bool(), 0)
        elif c.use_omtrack:
            # OmTrack contract: complete instruction, fixed visual slots, [ACT].
            sequences = []
            masks = []
            for row in range(B):
                valid_text = input_embeds[row, attention_mask[row].bool()]
                visual = all_visual[row]
                aq = self.action_query.to(valid_text.dtype)
                sequences.append(torch.cat([valid_text, visual, aq], dim=0))
                masks.append(torch.cat([
                    torch.ones(len(valid_text), dtype=torch.bool, device=visual.device),
                    visual_mask[row],
                    torch.ones(1, dtype=torch.bool, device=visual.device),
                ]))
            max_sequence = max(sequence.shape[0] for sequence in sequences)
            multimodal = input_embeds.new_zeros(B, max_sequence, self._hidden_size)
            full_mask = attention_mask.new_zeros(B, max_sequence)
            for row, (sequence, sequence_mask) in enumerate(zip(sequences, masks)):
                pad = max_sequence - sequence.shape[0]
                multimodal[row, pad:] = sequence
                full_mask[row, pad:] = sequence_mask.to(full_mask.dtype)
            position_ids = full_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(~full_mask.bool(), 0)
        elif variable_visual_sequences is not None:
            # Build each real sequence first, then standard left-pad the batch.
            # Explicit position_ids make valid-token positions independent of
            # the amount of left padding.
            sequences: list[torch.Tensor] = []
            for row, visual in enumerate(variable_visual_sequences):
                valid_text = input_embeds[row, attention_mask[row].bool()]
                if not len(valid_text):
                    raise ValueError("Text sequence has no valid BOS token")
                aq = self.action_query.to(valid_text.dtype)
                sequences.append(torch.cat([valid_text[:1], visual[0], valid_text[1:], aq], dim=0))
            if multimodal_left_padding is None:
                max_sequence = max(sequence.shape[0] for sequence in sequences)
                left_padding = [max_sequence - sequence.shape[0] for sequence in sequences]
            else:
                if multimodal_left_padding.numel() != B:
                    raise ValueError("multimodal_left_padding must have one value per sample")
                left_padding = [int(value) for value in multimodal_left_padding.tolist()]
                padded_lengths = [
                    pad + sequence.shape[0] for pad, sequence in zip(left_padding, sequences)
                ]
                if len(set(padded_lengths)) != 1:
                    raise ValueError("Collator left-padding plan does not align the batch")
                max_sequence = padded_lengths[0]
            multimodal = input_embeds.new_zeros(B, max_sequence, self._hidden_size)
            full_mask = attention_mask.new_zeros(B, max_sequence)
            for row, (sequence, pad) in enumerate(zip(sequences, left_padding)):
                multimodal[row, pad:pad + sequence.shape[0]] = sequence
                full_mask[row, pad:pad + sequence.shape[0]] = 1
            position_ids = full_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(~full_mask.bool(), 0)
        else:
            # Prismatic pattern: [BOS][visual][rest of text][action_query]
            bos = input_embeds[:, :1, :]
            text = input_embeds[:, 1:, :]
            aq = self.action_query.unsqueeze(0).expand(B, -1, -1)
            multimodal = torch.cat([bos, all_visual, text, aq], dim=1)
            visual_mask = visual_mask.to(device=attention_mask.device, dtype=attention_mask.dtype)
            query_mask = torch.ones(B, 1, dtype=attention_mask.dtype, device=attention_mask.device)
            full_mask = torch.cat(
                [attention_mask[:, :1], visual_mask, attention_mask[:, 1:], query_mask], dim=1
            )
            position_ids = None

        decoder = _get_internal_decoder(self.backbone)
        if self._uses_openfly_action_tokens_ar:
            if openfly_action_token_targets is not None:
                return self._openfly_teacher_forced_action_tokens(
                    decoder,
                    multimodal,
                    full_mask,
                    openfly_action_token_targets,
                    current_memory_token=current_memory_token,
                )
            return self._openfly_generate_action_tokens(
                decoder,
                multimodal,
                full_mask,
                current_memory_token=current_memory_token,
            )

        # ---- Step 3: Run LLM decoder ----
        decoder_kwargs = dict(
            inputs_embeds=multimodal, attention_mask=full_mask, use_cache=False
        )
        if position_ids is not None:
            decoder_kwargs["position_ids"] = position_ids
        outputs = decoder(**decoder_kwargs)
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

        # ---- Step 4: Extract action_query position (last token) ----
        pooled = hidden[:, -1, :]  # [B, hidden]
        instruction_tokens = None
        needs_instruction_tokens = (
            c.use_motion_instruction_attention or c.task_state_mode != "none"
        ) or (
            c.use_hierarchical_stop and c.hierarchical_stop_mode == "text_query"
        ) or (
            return_auxiliary
            and self.progress_head is not None
            and c.progress_mode == "text_query"
        )
        if needs_instruction_tokens:
            if instruction_token_mask is None:
                raise ValueError(
                    "Text-conditioned attention requires instruction_token_mask"
                )
            # Map decoder states back to the original tokenizer sequence:
            # token 0 remains before visual tokens; tokens 1..T-1 follow them.
            text_tail_start = 1 + all_visual.shape[1]
            instruction_tokens = torch.cat(
                [hidden[:, :1], hidden[:, text_tail_start:text_tail_start + input_ids.shape[1] - 1]],
                dim=1,
            )

        motion_instruction_attention = None
        motion_features = None
        motion_task_context = None
        segment_state = None
        segment_prior_bias = None
        if self.segment_task_state is not None:
            if instruction_tokens is None or instruction_token_mask is None:
                raise ValueError("Segment Task State requires instruction token features and mask")
            if segment_token_ids is None:
                raise ValueError("Segment Task State requires segment_token_ids")
            segment_state = self.segment_task_state(
                pooled,
                instruction_tokens,
                segment_token_ids,
                instruction_token_mask,
                belief_prev=segment_belief_prev,
                task_gru_state_prev=task_gru_state_prev,
                history_tokens=(
                    memory if c.native1t_segts_mode == "history" else None
                ),
                history_mask=(
                    history_mask if c.native1t_segts_mode == "history" else None
                ),
                hard_pointer_stage=hard_pointer_stage,
            )
            if c.task_state_mode == "segment_only":
                pooled = segment_state["updated"]
            else:
                segment_prior_bias = self.segment_task_state.token_prior_bias(
                    segment_state["alpha"],
                    segment_token_ids,
                    segment_state["segment_lengths"],
                    instruction_token_mask,
                    beta=c.task_state_segment_prior_beta,
                    eps=c.task_state_eps,
                )
        if self.motion_instruction_attention is not None:
            if instruction_tokens is None or instruction_token_mask is None:
                raise ValueError("MCIA requires instruction token features and mask")
            motion_features = pooled
            if c.use_hierarchical_stop and c.hierarchical_stop_mode == "mcia_context":
                pooled, motion_task_context, motion_instruction_attention = (
                    self.motion_instruction_attention(
                        pooled,
                        instruction_tokens,
                        instruction_token_mask,
                        token_prior_bias=segment_prior_bias,
                        return_context=True,
                    )
                )
            else:
                pooled, motion_instruction_attention = self.motion_instruction_attention(
                    pooled,
                    instruction_tokens,
                    instruction_token_mask,
                    token_prior_bias=segment_prior_bias,
                )

        # ---- Step 5: Action head (variant-specific) ----
        if self.config.use_prev_action:
            if prev_actions is None:
                prev_actions = torch.full(
                    (B,), self.config.prev_action_vocab - 1, dtype=torch.long, device=pooled.device
                )
            prev_emb = self.prev_action_embed(prev_actions)  # [B, 64]
            fused_input = torch.cat([pooled, prev_emb], dim=-1)  # [B, hidden+64]
        else:
            fused_input = pooled  # [B, hidden]

        if c.use_omtrack:
            if self.omtrack_chunk_norm is None or self.omtrack_chunk_head is None:
                raise RuntimeError("OmTrack chunk head is missing")
            return self.omtrack_chunk_head(self.omtrack_chunk_norm(pooled)).reshape(
                B, c.action_chunk_horizon, c.num_actions
            )
        if c.action_chunk_horizon > 1:
            if self.action_chunk_horizon_embeddings is None or self.action_chunk_norm is None:
                raise RuntimeError("AC-4 chunk modules are missing")
            horizon_ids = torch.arange(
                c.action_chunk_horizon, device=pooled.device, dtype=torch.long
            )
            chunk_hidden = self.action_chunk_norm(
                pooled.unsqueeze(1)
                + self.action_chunk_horizon_embeddings(horizon_ids).unsqueeze(0)
            )
            chunk_fused = self.fuse_proj(chunk_hidden)
            head_logits = self.action_head(chunk_fused)
            if c.use_hierarchical_stop:
                logits, stop_logits, stop_attention = self._apply_decision_heads(
                    head_logits,
                    chunk_fused,
                    instruction_tokens,
                    instruction_token_mask,
                    motion_features,
                    motion_task_context,
                )
                if return_auxiliary:
                    return self._auxiliary_output(
                        logits,
                        stop_logits,
                        chunk_fused,
                        instruction_tokens,
                        instruction_token_mask,
                        None,
                        move_logits=head_logits,
                        stop_attention=stop_attention,
                        current_memory_token=current_memory_token,
                    )
                if return_stop_aux:
                    return logits, stop_logits
                return logits
            if c.use_native1t_postfusion and c.native1t_stage == 2:
                if current_memory_token is None:
                    raise RuntimeError(
                        "Native1T AC4 failed to create the current memory feature"
                    )
                return head_logits, current_memory_token
            return head_logits

        if self.gru_cell is not None:
            # M2: GRU memory
            if memory_state is None:
                memory_state = torch.zeros(B, self.config.memory_hidden_dim, device=fused_input.device)
            new_memory = self.memory_norm(self.gru_cell(fused_input, memory_state))
            head_logits = self.action_head(new_memory)
            logits, stop_logits, stop_attention = self._apply_decision_heads(
                head_logits,
                new_memory,
                instruction_tokens,
                instruction_token_mask,
                motion_features,
                motion_task_context,
            )
            if return_auxiliary:
                return self._auxiliary_output(
                    logits, stop_logits, new_memory, instruction_tokens,
                    instruction_token_mask, new_memory,
                    move_logits=head_logits if c.use_hierarchical_stop else None,
                    stop_attention=stop_attention,
                    motion_instruction_attention=motion_instruction_attention,
                    segment_state=segment_state,
                )
            if return_stop_aux and stop_logits is not None:
                return logits, new_memory, stop_logits
            return logits, new_memory

        fused = self.fuse_proj(fused_input)  # [B, action_head_hidden]
        head_logits = self.action_head(fused)
        if self._uses_openfly_action_tokens:
            token_logits = head_logits.reshape(
                B,
                c.openfly_action_token_dim,
                c.openfly_action_token_bins,
            )
            if return_auxiliary:
                # SegTS/PosKey/LearnPool still supplies the same navigation
                # feature; only the final action surface changes from a
                # categorical vector to eight OpenFly token distributions.
                return self._auxiliary_output(
                    token_logits,
                    None,
                    fused,
                    instruction_tokens,
                    instruction_token_mask,
                    None,
                    motion_instruction_attention=motion_instruction_attention,
                    segment_state=segment_state,
                    current_memory_token=current_memory_token,
                )
            if c.use_native1t_postfusion and c.native1t_stage == 2:
                if current_memory_token is None:
                    raise RuntimeError("Native1T failed to create the current memory feature")
                return token_logits, current_memory_token
            return token_logits
        logits, stop_logits, stop_attention = self._apply_decision_heads(
            head_logits,
            fused,
            instruction_tokens,
            instruction_token_mask,
            motion_features,
            motion_task_context,
        )
        if return_auxiliary:
            return self._auxiliary_output(
                logits, stop_logits, fused, instruction_tokens,
                instruction_token_mask, None,
                move_logits=head_logits if c.use_hierarchical_stop else None,
                stop_attention=stop_attention,
                motion_instruction_attention=motion_instruction_attention,
                segment_state=segment_state,
                current_memory_token=current_memory_token,
            )
        if return_stop_aux and stop_logits is not None:
            return logits, stop_logits
        if c.use_native1t_postfusion and c.native1t_stage == 2:
            if current_memory_token is None:
                raise RuntimeError("Native1T failed to create the current memory feature")
            return logits, current_memory_token
        if c.use_temporal_memory:
            if current_memory_token is None:
                raise RuntimeError("TM-VLA failed to produce the current memory token")
            return logits, current_memory_token
        return logits

    def _apply_stop_aux(
        self,
        action_logits: torch.Tensor,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Fuse optional binary STOP log-odds into action 0 only."""
        if self.stop_aux_head is None:
            return action_logits, None
        stop_logits = self.stop_aux_head(features).squeeze(-1)
        stop_delta = torch.zeros_like(action_logits)
        stop_delta[:, 0] = stop_logits.to(action_logits.dtype) * float(
            self.config.stop_aux_logit_weight
        )
        return action_logits + stop_delta, stop_logits

    def _apply_decision_heads(
        self,
        head_logits: torch.Tensor,
        features: torch.Tensor,
        instruction_tokens: torch.Tensor | None = None,
        instruction_mask: torch.Tensor | None = None,
        motion_features: torch.Tensor | None = None,
        motion_task_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if not self.config.use_hierarchical_stop:
            logits, stop_logits = self._apply_stop_aux(head_logits, features)
            return logits, stop_logits, None
        if self.stop_aux_head is None:
            raise RuntimeError("Hierarchical STOP model has no STOP gate")
        stop_attention = None
        if self.config.hierarchical_stop_mode == "mcia_context":
            if not isinstance(self.stop_aux_head, MCIAContextStopHead):
                raise RuntimeError("MCIA-context STOP gate has the wrong module type")
            if motion_features is None or motion_task_context is None:
                raise ValueError("MCIA-context STOP inputs h_t and c_t are missing")
            stop_logits = self.stop_aux_head(motion_features, motion_task_context)
        elif self.config.hierarchical_stop_mode == "text_query":
            if not isinstance(self.stop_aux_head, TextConditionedStopHead):
                raise RuntimeError("Text-conditioned STOP gate has the wrong module type")
            if instruction_tokens is None or instruction_mask is None:
                raise ValueError("Text-conditioned STOP inputs are missing")
            stop_logits, stop_attention = self.stop_aux_head(
                features, instruction_tokens, instruction_mask
            )
        else:
            stop_logits = self.stop_aux_head(features).squeeze(-1)
        # Binary STOP gate and seven-way navigation head are decoded as a
        # true hierarchy: when p(STOP) exceeds its configured threshold,
        # choose STOP; otherwise choose the navigation argmax.  The returned
        # scores preserve precisely that decision under one final argmax.
        decision_scores = hierarchical_action_scores(
            head_logits, stop_logits, self.config.hierarchical_stop_threshold
        )
        return decision_scores, stop_logits, stop_attention

    def _auxiliary_output(
        self,
        action_logits: torch.Tensor,
        stop_logits: torch.Tensor | None,
        navigation_features: torch.Tensor,
        instruction_tokens: torch.Tensor | None,
        instruction_mask: torch.Tensor | None,
        memory_state: torch.Tensor | None,
        move_logits: torch.Tensor | None = None,
        stop_attention: torch.Tensor | None = None,
        motion_instruction_attention: torch.Tensor | None = None,
        segment_state: dict[str, torch.Tensor] | None = None,
        current_memory_token: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        progress = None
        transition_logits = None
        instruction_attention = None
        if self.transition_head is not None:
            transition_logits = self.transition_head(navigation_features).squeeze(-1)
        if self.progress_head is not None:
            if self.config.progress_mode == "direct":
                progress = self.progress_head(navigation_features)
            else:
                if instruction_tokens is None or instruction_mask is None:
                    raise ValueError("text_query progress inputs are missing")
                progress, instruction_attention = self.progress_head(
                    navigation_features, instruction_tokens, instruction_mask
                )
        return {
            "action_logits": action_logits,
            "stop_logits": stop_logits,
            "progress": progress,
            "transition_logits": transition_logits,
            "instruction_attention": instruction_attention,
            "stop_attention": stop_attention,
            "motion_instruction_attention": motion_instruction_attention,
            "memory_state": memory_state,
            "current_memory_token": current_memory_token,
            "move_logits": move_logits,
            "segment_alpha": None if segment_state is None else segment_state["alpha"],
            "segment_belief": (
                None if segment_state is None else segment_state["belief"]
            ),
            "segment_context_alpha": (
                None if segment_state is None else segment_state["context_alpha"]
            ),
            "segment_entropy": None if segment_state is None else segment_state["entropy"],
            "segment_valid_mask": (
                None if segment_state is None else segment_state["segment_valid_mask"]
            ),
            "segment_history_attention": (
                None if segment_state is None else segment_state["history_attention"]
            ),
            "segment_pool_alpha": (
                None if segment_state is None else segment_state["pool_alpha"]
            ),
            "segment_pointer_stage": (
                None if segment_state is None else segment_state["pointer_stage"]
            ),
            "segment_pointer_next_stage": (
                None if segment_state is None else segment_state["pointer_next_stage"]
            ),
            "segment_pointer_advance": (
                None if segment_state is None else segment_state["pointer_advance"]
            ),
            "task_gru_state": (
                None if segment_state is None else segment_state["task_gru_state"]
            ),
        }

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        output_dir: str | Path,
        processor: Any,
        *,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        # Processor
        processor.save_pretrained(output / "processor")

        # Core weights
        torch.save(self.action_head.state_dict(), output / "action_head.pt")
        if self.task_aware_pooler is not None:
            torch.save(
                self.task_aware_pooler.state_dict(),
                output / "task_aware_pooler.pt",
            )
        if self.instruction_local_pooler is not None:
            torch.save(
                self.instruction_local_pooler.state_dict(),
                output / "instruction_local_pooler.pt",
            )
        if self.spatial_anchored_residual_pooler is not None:
            torch.save(
                self.spatial_anchored_residual_pooler.state_dict(),
                output / "spatial_anchored_residual_pooler.pt",
            )
        if self._uses_openfly_action_tokens:
            source_value = self.config.openfly_action_codec_path
            if not source_value:
                raise RuntimeError(
                    "OpenFly action-token checkpoint save requires "
                    "openfly_action_codec_path"
                )
            source = Path(source_value).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(f"OpenFly action codec is missing: {source}")
            from aerial_openvla.openfly.action_codec import OpenFlyActionCodec

            codec = OpenFlyActionCodec.load(
                source, expected_sha256=self.config.openfly_action_codec_sha256
            )
            target = output / "openfly_action_codec.json"
            temporary = output / ".openfly_action_codec.json.inprogress"
            if temporary.exists():
                raise FileExistsError(f"Incomplete OpenFly codec copy exists: {temporary}")
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
        if (
            self.omtrack_chunk_norm is not None
            and self.omtrack_chunk_head is not None
            and self.omtrack_time_embedding is not None
        ):
            torch.save(
                {
                    "norm": self.omtrack_chunk_norm.state_dict(),
                    "head": self.omtrack_chunk_head.state_dict(),
                    "time_embedding": self.omtrack_time_embedding.state_dict(),
                },
                output / "omtrack_head.pt",
            )
        if self.action_chunk_horizon_embeddings is not None and self.action_chunk_norm is not None:
            torch.save(
                {
                    "horizon_embeddings": self.action_chunk_horizon_embeddings.state_dict(),
                    "norm": self.action_chunk_norm.state_dict(),
                },
                output / "action_chunk_head.pt",
            )
        if self.transition_head is not None:
            torch.save(self.transition_head.state_dict(), output / "transition_head.pt")
        if self.stop_aux_head is not None:
            torch.save(self.stop_aux_head.state_dict(), output / "stop_aux_head.pt")
        if self.progress_head is not None:
            torch.save(self.progress_head.state_dict(), output / "progress_head.pt")
        if self.motion_instruction_attention is not None:
            torch.save(
                self.motion_instruction_attention.state_dict(),
                output / "motion_instruction_attention.pt",
            )
        if self.segment_task_state is not None:
            torch.save(self.segment_task_state.state_dict(), output / "segment_task_state.pt")
        if self.frame_slot_embedding is not None:
            torch.save(
                self.frame_slot_embedding.state_dict(), output / "frame_slot_embedding.pt"
            )
        if self.temporal_adapter is not None:
            torch.save(self.temporal_adapter.state_dict(), output / "temporal_adapter.pt")
        if (
            self.temporal_memory_tokenizer is not None
            and self.temporal_memory_projector is not None
            and self.temporal_memory_time_embedding is not None
        ):
            torch.save(
                {
                    "tokenizer": self.temporal_memory_tokenizer.state_dict(),
                    "projector": self.temporal_memory_projector.state_dict(),
                    "time_embedding": self.temporal_memory_time_embedding.state_dict(),
                },
                output / "temporal_memory.pt",
            )
        if (
            self.native1t_memory_tokenizer is not None
            and self.native1t_memory_projector is not None
            and self.native1t_mem_bos is not None
        ):
            torch.save({
                "tokenizer": self.native1t_memory_tokenizer.state_dict(),
                "projector": self.native1t_memory_projector.state_dict(),
                "mem_bos": self.native1t_mem_bos.detach().cpu(),
            }, output / "native1t_postfusion.pt")
        if self.native1t_action_embedding is not None:
            if (
                self.native1t_action_history_projector is None
                or self.native1t_action_bos is None
            ):
                raise RuntimeError("Native1T AH4 checkpoint modules are incomplete")
            torch.save(
                {
                    "action_embedding": self.native1t_action_embedding.state_dict(),
                    "action_history_projector": (
                        self.native1t_action_history_projector.state_dict()
                    ),
                    "action_bos": self.native1t_action_bos.detach().cpu(),
                },
                output / "native1t_action_history.pt",
            )
        if self.motion_history_projector is not None:
            torch.save(
                self.motion_history_projector.state_dict(),
                output / "motion_history_projector.pt",
            )
        torch.save(self.action_query.data, output / "action_query.pt")
        torch.save(self.fuse_proj.state_dict(), output / "fuse_proj.pt")

        if self.prev_action_embed is not None:
            torch.save(self.prev_action_embed.state_dict(), output / "prev_action_embed.pt")
        if self.gru_cell is not None:
            torch.save(self.gru_cell.state_dict(), output / "gru_cell.pt")
        if self.memory_norm is not None:
            torch.save(self.memory_norm.state_dict(), output / "memory_norm.pt")

        # Projector (if trained)
        if (
            self.config.train_projector
            or self.config.use_native1t_postfusion
            or self.config.use_task_aware_pooler
            or self.config.use_instruction_local_pooler
            or self.config.use_spatial_anchored_residual_pooler
        ):
            torch.save(self.backbone.projector.state_dict(), output / "projector.pt")

        # Any tail-tuned vision state must be persisted so evaluation never
        # silently falls back to the base tower.
        if (
            not self.config.freeze_vision
            or self.config.vision_train_last_n_layers > 0
            or self.config.vision_train_siglip_last_n_layers > 0
        ):
            torch.save(self.backbone.vision_backbone.state_dict(), output / "vision_backbone.pt")

        # A frozen but AirSim-adapted DINO tower is still part of the policy
        # contract.  Persist just that tower (not an unnecessary SigLIP copy)
        # so later resume/eval cannot silently fall back to the original Prism
        # DINO weights.
        if self.dino_teacher_init_report is not None:
            torch.save(
                self.backbone.vision_backbone.featurizer.state_dict(),
                output / "dino_featurizer.pt",
            )
            with (output / "dino_teacher_init_report.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(self.dino_teacher_init_report, handle, indent=2, sort_keys=True)

        # LoRA adapter
        if isinstance(self.backbone.language_model, PeftModel):
            self.backbone.language_model.save_pretrained(output / "lora")
        elif self.config.train_language_model:
            torch.save(
                self.backbone.language_model.state_dict(),
                output / "language_model.pt",
            )

        # Meta: every structural field, so evaluation rebuilds this exact
        # architecture from the checkpoint instead of trusting its own config.
        meta = {"model_name_or_path": self.config.model_name_or_path}
        for field_name in STRUCTURAL_META_FIELDS:
            meta[field_name] = getattr(self.config, field_name)
        if self.config.use_native1t_postfusion:
            meta["native1t_visual_reducer_contract"] = self.native1t_visual_reducer_contract()
        if self.config.openfly_action_policy == OPENFLY_OBSERVED8_ACTION_POLICY:
            action_map = {
                "format": "openfly_observed_8_action_map_v1",
                "action_policy": OPENFLY_OBSERVED8_ACTION_POLICY,
                "head_class_to_openfly_action_id": {
                    str(class_id): action_id
                    for class_id, action_id in OPENFLY_OBSERVED8_CLASS_TO_ACTION_ID.items()
                },
                "openfly_action_names": {
                    str(action_id): OPENFLY_ACTION_NAMES[action_id]
                    for action_id in OPENFLY_OBSERVED8_ACTION_IDS
                },
            }
            (output / "openfly_observed8_action_map.json").write_text(
                json.dumps(action_map, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            meta["openfly_observed8_action_ids"] = list(OPENFLY_OBSERVED8_ACTION_IDS)
        if self._uses_openfly_action_tokens_ar:
            if self._openfly_action_lm_contract is None:
                raise RuntimeError("OpenFly autoregressive action-token contract is missing")
            meta["openfly_action_lm_contract"] = dict(self._openfly_action_lm_contract)
        if self.dino_teacher_init_report is not None:
            meta["dino_teacher_init"] = self.dino_teacher_init_report
        torch.save(meta, output / "classifier_meta.pt")

        # Optimizer / scheduler / extra
        if optimizer is not None:
            torch.save(optimizer.state_dict(), output / "optimizer.pt")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), output / "scheduler.pt")
        if extra:
            torch.save(extra, output / "training_state.pt")

    def load_checkpoint(
        self,
        checkpoint_dir: str | Path,
        *,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        allow_missing_native1t_action_history: bool = False,
    ) -> dict[str, Any] | None:
        checkpoint = Path(checkpoint_dir)
        meta_path = checkpoint / "classifier_meta.pt"
        checkpoint_meta = (
            torch.load(meta_path, map_location="cpu", weights_only=True)
            if meta_path.is_file()
            else {}
        )
        checkpoint_declares_task_gru = bool(
            checkpoint_meta.get("task_state_use_gru", False)
        )
        checkpoint_declares_learnpool = (
            checkpoint_meta.get("task_state_pooling_mode") == "learnpool"
        )
        if self._uses_openfly_action_tokens:
            codec_path = checkpoint / "openfly_action_codec.json"
            if not codec_path.is_file():
                raise FileNotFoundError(
                    f"OpenFly action-token checkpoint lacks bundled codec: {codec_path}"
                )
            from aerial_openvla.openfly.action_codec import OpenFlyActionCodec

            expected_codec_hash = self.config.openfly_action_codec_sha256
            codec = OpenFlyActionCodec.load(
                codec_path, expected_sha256=expected_codec_hash
            )
            meta_codec_hash = checkpoint_meta.get("openfly_action_codec_sha256")
            if meta_codec_hash != codec.sha256:
                raise ValueError(
                    "OpenFly checkpoint classifier metadata/codec hash mismatch: "
                    f"meta={meta_codec_hash!r}, codec={codec.sha256}"
                )
        if self._uses_openfly_action_tokens_ar:
            saved_lm_contract = checkpoint_meta.get("openfly_action_lm_contract")
            if saved_lm_contract != self._openfly_action_lm_contract:
                raise ValueError(
                    "OpenFly autoregressive LM token contract mismatch: "
                    f"checkpoint={saved_lm_contract!r}, live={self._openfly_action_lm_contract!r}"
                )

        # Core weights (required)
        head_path = checkpoint / "action_head.pt"
        if not head_path.is_file():
            raise FileNotFoundError(f"Missing action_head.pt in {checkpoint}")
        self.action_head.load_state_dict(
            torch.load(head_path, map_location="cpu", weights_only=True)
        )

        task_aware_pooler_path = checkpoint / "task_aware_pooler.pt"
        if self.task_aware_pooler is not None and task_aware_pooler_path.is_file():
            self.task_aware_pooler.load_state_dict(
                torch.load(task_aware_pooler_path, map_location="cpu", weights_only=True)
            )
        elif self.task_aware_pooler is not None:
            print(
                f"[init] {checkpoint} has no task_aware_pooler.pt; "
                "initialising the new Task-Aware pooler from scratch"
            )

        instruction_local_pooler_path = checkpoint / "instruction_local_pooler.pt"
        if self.instruction_local_pooler is not None and instruction_local_pooler_path.is_file():
            local_state = torch.load(instruction_local_pooler_path, map_location="cpu", weights_only=True)
            if self.config.history_conditioned_local_pooler and set(local_state) == {"score_weight"}:
                self.instruction_local_pooler.score_weight.data.copy_(local_state["score_weight"])
            else:
                self.instruction_local_pooler.load_state_dict(local_state, strict=True)
        elif self.instruction_local_pooler is not None:
            print(
                f"[init] {checkpoint} has no instruction_local_pooler.pt; "
                "keeping the zero-initialized local weighting matrix"
            )

        spatial_anchored_residual_pooler_path = (
            checkpoint / "spatial_anchored_residual_pooler.pt"
        )
        if (
            self.spatial_anchored_residual_pooler is not None
            and spatial_anchored_residual_pooler_path.is_file()
        ):
            self.spatial_anchored_residual_pooler.load_state_dict(
                torch.load(
                    spatial_anchored_residual_pooler_path,
                    map_location="cpu",
                    weights_only=True,
                ),
                strict=True,
            )
        elif self.spatial_anchored_residual_pooler is not None:
            print(
                f"[init] {checkpoint} has no spatial_anchored_residual_pooler.pt; "
                "keeping the zero-initialized alpha and fresh attention modules"
            )

        omtrack_path = checkpoint / "omtrack_head.pt"
        if self.config.use_omtrack:
            if not omtrack_path.is_file():
                raise FileNotFoundError(f"Missing omtrack_head.pt in {checkpoint}")
            if (
                self.omtrack_chunk_norm is None
                or self.omtrack_chunk_head is None
                or self.omtrack_time_embedding is None
            ):
                raise RuntimeError("Live model is missing OmTrack modules")
            omtrack_state = torch.load(
                omtrack_path, map_location="cpu", weights_only=True
            )
            self.omtrack_chunk_norm.load_state_dict(omtrack_state["norm"])
            self.omtrack_chunk_head.load_state_dict(omtrack_state["head"])
            self.omtrack_time_embedding.load_state_dict(omtrack_state["time_embedding"])

        chunk_head_path = checkpoint / "action_chunk_head.pt"
        if chunk_head_path.is_file() and self.action_chunk_horizon_embeddings is not None:
            chunk_state = torch.load(chunk_head_path, map_location="cpu", weights_only=True)
            self.action_chunk_horizon_embeddings.load_state_dict(
                chunk_state["horizon_embeddings"]
            )
            if self.action_chunk_norm is None:
                raise RuntimeError("AC checkpoint has no live chunk LayerNorm")
            self.action_chunk_norm.load_state_dict(chunk_state["norm"])
        elif self.action_chunk_horizon_embeddings is not None:
            print(
                f"[init] {checkpoint} has no action_chunk_head.pt; "
                "initialising AC-4 horizon embeddings and LayerNorm from scratch"
            )

        transition_head_path = checkpoint / "transition_head.pt"
        if transition_head_path.is_file() and self.transition_head is not None:
            self.transition_head.load_state_dict(
                torch.load(transition_head_path, map_location="cpu", weights_only=True)
            )
        elif self.transition_head is not None:
            print(
                f"[init] {checkpoint} has no transition_head.pt; "
                "initialising the enabled transition head from scratch"
            )

        stop_head_path = checkpoint / "stop_aux_head.pt"
        if stop_head_path.is_file() and self.stop_aux_head is not None:
            self.stop_aux_head.load_state_dict(
                torch.load(stop_head_path, map_location="cpu", weights_only=True)
            )
        elif self.stop_aux_head is not None:
            print(
                f"[init] {checkpoint} has no stop_aux_head.pt; "
                "initialising the enabled STOP auxiliary head from scratch"
            )

        progress_head_path = checkpoint / "progress_head.pt"
        if progress_head_path.is_file() and self.progress_head is not None:
            self.progress_head.load_state_dict(
                torch.load(progress_head_path, map_location="cpu", weights_only=True)
            )
        elif self.progress_head is not None:
            print(
                f"[init] {checkpoint} has no progress_head.pt; "
                "initialising the enabled progress auxiliary head from scratch"
            )

        motion_attention_path = checkpoint / "motion_instruction_attention.pt"
        if motion_attention_path.is_file() and self.motion_instruction_attention is not None:
            self.motion_instruction_attention.load_state_dict(
                torch.load(motion_attention_path, map_location="cpu", weights_only=True)
            )
        elif self.motion_instruction_attention is not None:
            print(
                f"[init] {checkpoint} has no motion_instruction_attention.pt; "
                "initialising MCIA from scratch"
            )

        segment_state_path = checkpoint / "segment_task_state.pt"
        if segment_state_path.is_file() and self.segment_task_state is not None:
            segment_state = torch.load(
                segment_state_path, map_location="cpu", weights_only=True
            )
            allowed_missing: list[str] = []
            position_weight = "segment_pos_embed.weight"
            if (
                self.segment_task_state.segment_pos_embed is not None
                and position_weight not in segment_state
            ):
                allowed_missing.append(position_weight)
            pooler_missing = []
            if self.segment_task_state.segment_pooler is not None:
                pooler_missing = sorted(
                    name
                    for name in self.segment_task_state.state_dict()
                    if name.startswith("segment_pooler.") and name not in segment_state
                )
                if pooler_missing and checkpoint_declares_learnpool:
                    raise RuntimeError(
                        "Checkpoint declares LearnPool but its Segment Task State is "
                        f"incomplete: {pooler_missing}"
                    )
                allowed_missing.extend(pooler_missing)
            if self.segment_task_state.token_query is not None:
                for prefix in ("token_query", "token_key"):
                    for suffix in ("weight", "bias"):
                        name = f"{prefix}.{suffix}"
                        if name not in segment_state:
                            allowed_missing.append(name)
            task_gru_prefixes = (
                "task_state_input_proj.",
                "task_gru.",
                "task_query_proj.",
            )
            missing_task_gru = sorted(
                name
                for name in self.segment_task_state.state_dict()
                if name.startswith(task_gru_prefixes) and name not in segment_state
            )
            if missing_task_gru:
                if checkpoint_declares_task_gru:
                    raise RuntimeError(
                        "Checkpoint declares TaskGRU but its Segment Task State is "
                        f"incomplete: {missing_task_gru}"
                    )
                allowed_missing.extend(missing_task_gru)
            if allowed_missing:
                incompatible = self.segment_task_state.load_state_dict(
                    segment_state, strict=False
                )
                if (
                    set(incompatible.missing_keys) != set(allowed_missing)
                    or incompatible.unexpected_keys
                ):
                    raise RuntimeError(
                        "SegTS-v5 may upgrade v4 only with its declared new attention "
                        f"parameters; got missing={incompatible.missing_keys}, "
                        f"unexpected={incompatible.unexpected_keys}"
                    )
                if position_weight in allowed_missing:
                    nn.init.zeros_(self.segment_task_state.segment_pos_embed.weight)
                if any(name.startswith("token_query.") for name in allowed_missing):
                    nn.init.zeros_(self.segment_task_state.token_query.weight)
                    nn.init.zeros_(self.segment_task_state.token_query.bias)
                if pooler_missing:
                    if self.segment_task_state.segment_pooler is None:
                        raise RuntimeError("LearnPool upgrade is missing its pooler module")
                    nn.init.zeros_(self.segment_task_state.segment_pooler[2].weight)
                    nn.init.zeros_(self.segment_task_state.segment_pooler[2].bias)
                print(f"[init] upgraded SegTS checkpoint with new parameters: {allowed_missing}")
            else:
                self.segment_task_state.load_state_dict(segment_state)
        elif self.segment_task_state is not None:
            print(
                f"[init] {checkpoint} has no segment_task_state.pt; "
                "initialising Segment Task State from scratch"
            )

        frame_embedding_path = checkpoint / "frame_slot_embedding.pt"
        if frame_embedding_path.is_file() and self.frame_slot_embedding is not None:
            self.frame_slot_embedding.load_state_dict(
                torch.load(frame_embedding_path, map_location="cpu", weights_only=True)
            )
        elif self.frame_slot_embedding is not None:
            print(
                f"[init] {checkpoint} has no frame_slot_embedding.pt; "
                "using zero-initialised frame embeddings"
            )

        temporal_adapter_path = checkpoint / "temporal_adapter.pt"
        if temporal_adapter_path.is_file() and self.temporal_adapter is not None:
            self.temporal_adapter.load_state_dict(
                torch.load(temporal_adapter_path, map_location="cpu", weights_only=True)
            )
        elif self.temporal_adapter is not None:
            print(
                f"[init] {checkpoint} has no temporal_adapter.pt; "
                "initialising temporal attention from scratch"
            )

        temporal_memory_path = checkpoint / "temporal_memory.pt"
        if temporal_memory_path.is_file() and self.temporal_memory_tokenizer is not None:
            temporal_memory_state = torch.load(
                temporal_memory_path, map_location="cpu", weights_only=True
            )
            if self.temporal_memory_projector is None or self.temporal_memory_time_embedding is None:
                raise RuntimeError("TM-VLA checkpoint has incomplete live modules")
            self.temporal_memory_tokenizer.load_state_dict(temporal_memory_state["tokenizer"])
            self.temporal_memory_projector.load_state_dict(temporal_memory_state["projector"])
            self.temporal_memory_time_embedding.load_state_dict(
                temporal_memory_state["time_embedding"]
            )
        elif self.temporal_memory_tokenizer is not None:
            print(
                f"[init] {checkpoint} has no temporal_memory.pt; "
                "initialising TM-VLA modules from scratch"
            )

        native1t_path = checkpoint / "native1t_postfusion.pt"
        if native1t_path.is_file() and self.native1t_memory_tokenizer is not None:
            native1t_state = torch.load(
                native1t_path, map_location="cpu", weights_only=True
            )
            if self.native1t_memory_projector is None or self.native1t_mem_bos is None:
                raise RuntimeError("Native1T checkpoint has incomplete live modules")
            self.native1t_memory_tokenizer.load_state_dict(native1t_state["tokenizer"])
            self.native1t_memory_projector.load_state_dict(native1t_state["projector"])
            self.native1t_mem_bos.data.copy_(native1t_state["mem_bos"])
        elif self.native1t_memory_tokenizer is not None:
            print(
                f"[init] {checkpoint} has no native1t_postfusion.pt; "
                "initialising Native1T modules from scratch"
            )

        native1t_action_path = checkpoint / "native1t_action_history.pt"
        if native1t_action_path.is_file() and self.native1t_action_embedding is not None:
            if (
                self.native1t_action_history_projector is None
                or self.native1t_action_bos is None
            ):
                raise RuntimeError("Native1T AH4 checkpoint has incomplete live modules")
            action_state = torch.load(
                native1t_action_path, map_location="cpu", weights_only=True
            )
            self.native1t_action_embedding.load_state_dict(
                action_state["action_embedding"]
            )
            self.native1t_action_history_projector.load_state_dict(
                action_state["action_history_projector"]
            )
            self.native1t_action_bos.data.copy_(action_state["action_bos"])
        elif self.native1t_action_embedding is not None:
            if not allow_missing_native1t_action_history:
                raise FileNotFoundError(
                    f"AH4 checkpoint is missing {native1t_action_path}"
                )
            print(
                f"[init] Stage-1 checkpoint {checkpoint} has no "
                "native1t_action_history.pt; initialising Native1T AH4 from scratch"
            )

        motion_history_path = checkpoint / "motion_history_projector.pt"
        if motion_history_path.is_file() and self.motion_history_projector is not None:
            self.motion_history_projector.load_state_dict(
                torch.load(motion_history_path, map_location="cpu", weights_only=True)
            )
        elif self.motion_history_projector is not None:
            print(
                f"[init] {checkpoint} has no motion_history_projector.pt; "
                "initialising Motion-Recent3 projector from scratch"
            )

        aq_path = checkpoint / "action_query.pt"
        if aq_path.is_file():
            self.action_query.data.copy_(
                torch.load(aq_path, map_location="cpu", weights_only=True)
            )

        fp_path = checkpoint / "fuse_proj.pt"
        if fp_path.is_file():
            self.fuse_proj.load_state_dict(
                torch.load(fp_path, map_location="cpu", weights_only=True)
            )

        # Optional weights
        pae_path = checkpoint / "prev_action_embed.pt"
        if pae_path.is_file() and self.prev_action_embed is not None:
            self.prev_action_embed.load_state_dict(
                torch.load(pae_path, map_location="cpu", weights_only=True)
            )

        gc_path = checkpoint / "gru_cell.pt"
        if gc_path.is_file() and self.gru_cell is not None:
            self.gru_cell.load_state_dict(
                torch.load(gc_path, map_location="cpu", weights_only=True)
            )

        mn_path = checkpoint / "memory_norm.pt"
        if mn_path.is_file() and self.memory_norm is not None:
            self.memory_norm.load_state_dict(
                torch.load(mn_path, map_location="cpu", weights_only=True)
            )

        proj_path = checkpoint / "projector.pt"
        if proj_path.is_file():
            self.backbone.projector.load_state_dict(
                torch.load(proj_path, map_location="cpu", weights_only=True)
            )

        vision_path = checkpoint / "vision_backbone.pt"
        if vision_path.is_file():
            self.backbone.vision_backbone.load_state_dict(
                torch.load(vision_path, map_location="cpu", weights_only=True)
            )
        else:
            dino_path = checkpoint / "dino_featurizer.pt"
            if dino_path.is_file():
                _load_strict_dino_state(
                    self.backbone.vision_backbone,
                    torch.load(dino_path, map_location="cpu", weights_only=True),
                    source_description=str(dino_path),
                    source_sha256=None,
                    expected_sha256=None,
                )
                saved_report = checkpoint_meta.get("dino_teacher_init")
                self.dino_teacher_init_report = (
                    saved_report
                    if isinstance(saved_report, dict)
                    else {
                        "status": "checkpoint_dino_featurizer_loaded",
                        "checkpoint": str(dino_path),
                    }
                )

        language_model_path = checkpoint / "language_model.pt"
        if language_model_path.is_file():
            self.backbone.language_model.load_state_dict(
                torch.load(language_model_path, map_location="cpu", weights_only=True)
            )

        # LoRA adapter
        lora_path = checkpoint / "lora"
        if lora_path.is_dir():
            from peft import PeftModel as PM
            if isinstance(self.backbone.language_model, PM):
                # The LM is already PEFT-wrapped (use_lora stages).  Wrapping it
                # again with from_pretrained resets the live adapter and drops
                # the saved weights on mismatched key prefixes, so load the
                # weights in place instead: Parameter identity is preserved and
                # optimizer param groups collected earlier stay valid.
                load_result = self.backbone.language_model.load_adapter(
                    str(lora_path), adapter_name="default", is_trainable=True
                )
                unexpected = getattr(load_result, "unexpected_keys", None)
                if unexpected:
                    raise RuntimeError(
                        f"LoRA weights in {lora_path} did not match the wrapped model; "
                        f"refusing a silent partial load (unexpected keys: {sorted(unexpected)[:5]})"
                    )
            else:
                self.backbone.language_model = PM.from_pretrained(
                    self.backbone.language_model, lora_path, is_trainable=True
                )

        # Loading a parent LoRA adapter intentionally enables its adapter
        # parameters inside PEFT.  Reapply the experiment contract after that
        # load so the parent policy remains frozen and only the new pooler is
        # exposed to the optimizer.
        self.enforce_task_aware_pooler_trainable_contract()

        # Optimizer / scheduler
        extra = None
        opt_path = checkpoint / "optimizer.pt"
        if opt_path.is_file() and optimizer is not None:
            optimizer.load_state_dict(
                torch.load(opt_path, map_location="cpu", weights_only=True)
            )
        sched_path = checkpoint / "scheduler.pt"
        if sched_path.is_file() and scheduler is not None:
            scheduler.load_state_dict(
                torch.load(sched_path, map_location="cpu", weights_only=True)
            )
        state_path = checkpoint / "training_state.pt"
        if state_path.is_file():
            extra = torch.load(state_path, map_location="cpu", weights_only=False)

        return extra

    def load_native1t_postfusion_modules(self, checkpoint_dir: str | Path) -> None:
        """Load Stage-1 Native1T modules without touching the policy weights.

        This is the Stage-2-from-scratch boundary: the frozen tokenizer and its
        trained memory interface are inherited, while Qwen/LoRA, the original
        projector, fuse/action query and action head remain newly initialized.
        """
        checkpoint = Path(checkpoint_dir)
        native1t_path = checkpoint / "native1t_postfusion.pt"
        if not native1t_path.is_file():
            raise FileNotFoundError(
                f"Missing Stage-1 native1t_postfusion.pt in {checkpoint}"
            )
        if (
            self.native1t_memory_tokenizer is None
            or self.native1t_memory_projector is None
            or self.native1t_mem_bos is None
        ):
            raise RuntimeError("Native1T modules are incomplete in the live model")
        if (
            self.config.use_instruction_local_pooler
            or self.config.use_spatial_anchored_residual_pooler
        ):
            meta_path = checkpoint / "classifier_meta.pt"
            if not meta_path.is_file():
                raise RuntimeError(
                    "Instruction-conditioned Native1T Stage-1 checkpoint lacks classifier_meta.pt"
                )
            saved_contract = torch.load(
                meta_path, map_location="cpu", weights_only=True
            ).get("native1t_visual_reducer_contract")
            expected_contract = self.native1t_visual_reducer_contract()
            required_keys = (
                "active_reducer",
                "current_raw_tokens",
                "current_projected_tokens",
                "current_output_tokens",
                "reducer_call",
            )
            if not isinstance(saved_contract, dict) or any(
                saved_contract.get(key) != expected_contract.get(key)
                for key in required_keys
            ) or saved_contract.get("native1t_stage") != 1:
                raise RuntimeError(
                    "Instruction-conditioned Native1T Stage-1 checkpoint reducer "
                    f"contract mismatch: saved={saved_contract!r}, expected={expected_contract!r}"
                )
        state = torch.load(native1t_path, map_location="cpu", weights_only=True)
        if set(state) != {"tokenizer", "projector", "mem_bos"}:
            raise RuntimeError(
                "Stage-1 Native1T module checkpoint has unexpected keys: "
                f"{sorted(state)}"
            )
        self.native1t_memory_tokenizer.load_state_dict(state["tokenizer"], strict=True)
        self.native1t_memory_projector.load_state_dict(state["projector"], strict=True)
        if state["mem_bos"].shape != self.native1t_mem_bos.shape:
            raise RuntimeError(
                "Stage-1 MEM_BOS shape mismatch: "
                f"{tuple(state['mem_bos'].shape)} vs {tuple(self.native1t_mem_bos.shape)}"
            )
        self.native1t_mem_bos.data.copy_(state["mem_bos"].to(self.native1t_mem_bos))

    # Keep compatibility with the existing training and evaluation factories.
    save_classifier_checkpoint = save_checkpoint
    load_classifier_checkpoint = load_checkpoint


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

_DINO_INIT_SENTINELS = (
    "patch_embed.proj.weight",
    "blocks.11.attn.qkv.weight",
    "blocks.23.mlp.fc2.weight",
    "norm.weight",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.detach().float()
    return {
        "shape": list(values.shape),
        "dtype": str(tensor.dtype),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "l2_norm": float(values.norm()),
    }


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash a state dict without converting BF16 values through FP32."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"State tensor {name!r} is not a torch.Tensor")
        host = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(host.dtype).encode("utf-8"))
        digest.update(str(tuple(host.shape)).encode("utf-8"))
        digest.update(host.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _load_strict_dino_state(
    vision_backbone: nn.Module,
    source_state: Any,
    *,
    source_description: str,
    source_sha256: str | None,
    expected_sha256: str | None,
) -> dict[str, Any]:
    """Load a raw DINOv2 ViT-L/14 state into Prism's DINO sub-tower only."""
    if not isinstance(source_state, dict):
        raise TypeError(
            "DINO initialization must be a raw state_dict, not a wrapped training "
            f"checkpoint; got {type(source_state)!r} from {source_description}"
        )
    if expected_sha256 is not None and source_sha256 != expected_sha256:
        raise RuntimeError(
            "DINO teacher SHA256 mismatch: "
            f"expected {expected_sha256}, got {source_sha256} for {source_description}"
        )

    dino = getattr(vision_backbone, "featurizer", None)
    siglip = getattr(vision_backbone, "fused_featurizer", None)
    if dino is None or siglip is None:
        raise RuntimeError(
            "AirSim DINO initialization requires the fused Prism DINO+SigLIP "
            "vision backbone"
        )
    target_state = dino.state_dict()
    if (
        len(target_state) != 343
        or len(source_state) != 343
        or len(getattr(dino, "blocks", ())) != 24
        or tuple(target_state.get("patch_embed.proj.weight", torch.empty(0)).shape)
        != (1024, 3, 14, 14)
    ):
        raise RuntimeError(
            "DINO teacher/target is not the expected MiniVLA DINOv2 ViT-L/14 "
            f"contract: source_keys={len(source_state)} target_keys={len(target_state)} "
            f"blocks={len(getattr(dino, 'blocks', ()))} "
            f"patch_shape={tuple(target_state.get('patch_embed.proj.weight', torch.empty(0)).shape)}"
        )
    invalid_source = [name for name, value in source_state.items() if not isinstance(value, torch.Tensor)]
    if invalid_source:
        raise TypeError(
            "DINO source state contains non-tensors: " + ", ".join(invalid_source[:5])
        )
    missing = sorted(set(target_state).difference(source_state))
    unexpected = sorted(set(source_state).difference(target_state))
    shape_mismatches = [
        {
            "name": name,
            "source": list(source_state[name].shape),
            "target": list(target_state[name].shape),
        }
        for name in sorted(set(target_state).intersection(source_state))
        if tuple(source_state[name].shape) != tuple(target_state[name].shape)
    ]
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            "DINO teacher does not exactly match Prism DINO target: "
            + json.dumps(
                {
                    "missing": missing[:10],
                    "unexpected": unexpected[:10],
                    "shape_mismatches": shape_mismatches[:5],
                },
                sort_keys=True,
            )
        )

    before = {name: _tensor_stats(target_state[name]) for name in _DINO_INIT_SENTINELS}
    # The source loader receives only `dino`, never `siglip`.  Hash the entire
    # SigLIP state on the rank that writes checkpoints as a further invariant.
    hash_siglip = int(os.environ.get("RANK", "0")) == 0
    siglip_before = _state_sha256(siglip.state_dict()) if hash_siglip else None
    dino.load_state_dict(source_state, strict=True)
    after_state = dino.state_dict()
    mismatched_after_load: list[str] = []
    for name in sorted(target_state):
        expected = source_state[name].to(
            device=after_state[name].device, dtype=after_state[name].dtype
        )
        if not torch.equal(after_state[name], expected):
            mismatched_after_load.append(name)
            if len(mismatched_after_load) == 5:
                break
    if mismatched_after_load:
        raise RuntimeError(
            "DINO target did not equal the converted teacher state after strict load: "
            + ", ".join(mismatched_after_load)
        )
    siglip_after = _state_sha256(siglip.state_dict()) if hash_siglip else None
    if siglip_before is not None and siglip_before != siglip_after:
        raise RuntimeError("DINO initialization unexpectedly modified the SigLIP tower")

    report = {
        "status": "ok",
        "source_checkpoint": source_description,
        "source_sha256": source_sha256,
        "expected_source_sha256": expected_sha256,
        "target_module": "backbone.vision_backbone.featurizer",
        "architecture": "DINOv2 ViT-L/14 reg4 (1024d, 24 blocks, 224px)",
        "source_key_count": len(source_state),
        "target_key_count": len(target_state),
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": [],
        "all_target_tensors_equal_after_load": True,
        "siglip_sha256_before": siglip_before,
        "siglip_sha256_after": siglip_after,
        "siglip_unchanged": None if siglip_before is None else True,
        "sentinel_stats_before": before,
        "sentinel_stats_after": {
            name: _tensor_stats(after_state[name]) for name in _DINO_INIT_SENTINELS
        },
        "sentinel_stats_source": {
            name: _tensor_stats(source_state[name]) for name in _DINO_INIT_SENTINELS
        },
    }
    print("[dino-teacher-init] " + json.dumps(report, sort_keys=True))
    return report


def load_dino_teacher_into_prism(
    vision_backbone: nn.Module,
    checkpoint_path: str | Path,
    expected_sha256: str | None,
) -> dict[str, Any]:
    """Load and verify the immutable AirSim SSL EMA teacher before Base training."""
    source = Path(checkpoint_path)
    if not source.is_file():
        raise FileNotFoundError(f"DINO teacher checkpoint does not exist: {source}")
    if not expected_sha256:
        raise ValueError("DINO teacher initialization requires an explicit SHA256")
    source_sha256 = _sha256_file(source)
    source_state = torch.load(source, map_location="cpu", weights_only=True)
    return _load_strict_dino_state(
        vision_backbone,
        source_state,
        source_description=str(source.resolve()),
        source_sha256=source_sha256,
        expected_sha256=expected_sha256,
    )

def load_airminivla_processor(model_name_or_path: str) -> Any:
    """Load the Prismatic processor (Qwen tokenizer + image processor)."""
    # The local Prismatic processor avoids AutoProcessor falling back to a raw
    # tokenizer for converted checkpoints.
    from aerial_openvla.models.openvla_classifier import load_openvla_processor
    return load_openvla_processor(model_name_or_path)


def load_airminivla_backbone(
    model_name_or_path: str,
    *,
    device: torch.device,
    use_flash_attention_2: bool = True,
    bf16: bool = True,
) -> nn.Module:
    """Load the Prismatic VLM backbone from a local checkpoint path."""
    from aerial_openvla.models.openvla_classifier import register_openvla_auto_classes
    register_openvla_auto_classes()
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16 if bf16 else torch.float32,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if use_flash_attention_2:
        kwargs["attn_implementation"] = "flash_attention_2"
    try:
        backbone = AutoModelForVision2Seq.from_pretrained(model_name_or_path, **kwargs)
    except (ImportError, RuntimeError) as exc:
        if not use_flash_attention_2:
            raise
        print(
            f"[warning] FlashAttention load failed ({exc}); "
            f"retrying with the default attention backend"
        )
        kwargs.pop("attn_implementation", None)
        backbone = AutoModelForVision2Seq.from_pretrained(model_name_or_path, **kwargs)
    return backbone.to(device)


def apply_checkpoint_meta(
    config: AirMiniVLAConfig, checkpoint_dir: str | Path
) -> AirMiniVLAConfig:
    """Override structural fields from a checkpoint's classifier_meta.pt.

    The trained checkpoint is the authority on model architecture; the eval
    config only supplies runtime choices (device, precision, attention).
    Differences are printed so silent drift is impossible.
    """
    meta_path = Path(checkpoint_dir) / "classifier_meta.pt"
    if not meta_path.is_file():
        return config
    meta = torch.load(meta_path, map_location="cpu", weights_only=True)
    overrides: dict[str, Any] = {}
    for field_name in STRUCTURAL_META_FIELDS:
        if field_name in meta and getattr(config, field_name) != meta[field_name]:
            overrides[field_name] = meta[field_name]
    if not overrides:
        return config
    for field_name, value in overrides.items():
        print(
            f"[meta] {field_name}: config={getattr(config, field_name)!r} -> "
            f"checkpoint={value!r}"
        )
    return AirMiniVLAConfig(**{**config.__dict__, **overrides})


def build_airminivla_classifier(
    config: AirMiniVLAConfig,
    *,
    device: torch.device,
    lora_adapter_dir: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
) -> tuple[Any, AirMiniVLAActionClassifier]:
    """Build a complete (processor, model) pair for AirMiniVLA."""
    if checkpoint_dir is not None:
        config = apply_checkpoint_meta(config, checkpoint_dir)
    processor = load_airminivla_processor(config.model_name_or_path)
    backbone = load_airminivla_backbone(
        config.model_name_or_path,
        device=device,
        use_flash_attention_2=config.use_flash_attention_2,
        bf16=config.bf16,
    )
    dino_teacher_init_report = None
    if config.dino_teacher_init_checkpoint:
        dino_teacher_init_report = load_dino_teacher_into_prism(
            backbone.vision_backbone,
            config.dino_teacher_init_checkpoint,
            config.dino_teacher_init_sha256,
        )

    # If loading a LoRA adapter, temporarily disable use_lora during init
    build_config = config
    if lora_adapter_dir is not None:
        build_config = AirMiniVLAConfig(
            **{**config.__dict__, "use_lora": False}
        )

    model = AirMiniVLAActionClassifier(backbone, build_config).to(device)
    model.dino_teacher_init_report = dino_teacher_init_report

    if lora_adapter_dir is not None:
        adapter_path = Path(lora_adapter_dir)
        if adapter_path.is_dir():
            model.backbone.language_model = PeftModel.from_pretrained(
                model.backbone.language_model, adapter_path, is_trainable=False
            )

    if checkpoint_dir is not None:
        model.load_checkpoint(checkpoint_dir)

    return processor, model
