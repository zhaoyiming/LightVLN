"""Temporal-memory modules for the streaming TM-VLA policy."""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalMemoryTokenizer(nn.Module):
    """Compress every historical patch grid into one learned-query token."""

    def __init__(self, vision_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        if vision_dim % num_heads:
            raise ValueError("vision_dim must be divisible by num_heads")
        self.vision_dim = int(vision_dim)
        self.query = nn.Parameter(torch.empty(1, 1, vision_dim))
        self.cross_attention = nn.MultiheadAttention(
            vision_dim, num_heads, dropout=0.0, batch_first=True
        )
        nn.init.normal_(self.query, std=vision_dim**-0.5)

    def forward(self, history_features: torch.Tensor) -> torch.Tensor:
        if history_features.ndim != 4:
            raise ValueError("history_features must have shape [B,H,P,D]")
        batch, history, patches, feature_dim = history_features.shape
        if patches <= 0 or feature_dim != self.vision_dim:
            raise ValueError(
                f"Expected non-empty [B,H,P,{self.vision_dim}] history, "
                f"got {tuple(history_features.shape)}"
            )
        flattened = history_features.reshape(batch * history, patches, feature_dim)
        query = self.query.to(dtype=flattened.dtype).expand(batch * history, -1, -1)
        memory, _ = self.cross_attention(
            query=query, key=flattened, value=flattened, need_weights=False
        )
        return memory.reshape(batch, history, feature_dim)


class MemoryProjector(nn.Module):
    """Map pre-projector vision tokens into the language-model embedding space."""

    def __init__(self, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(vision_dim, llm_dim)
        self.norm = nn.LayerNorm(llm_dim)

    def forward(self, memory_tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(memory_tokens))
