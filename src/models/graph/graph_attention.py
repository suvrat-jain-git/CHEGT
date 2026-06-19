"""Single graph attention layer (Pre-LN Transformer block).

Stores the last attention weight matrix for post-hoc visualisation of
inter-evidence interactions.
"""

from typing import Tuple

import torch
import torch.nn as nn


class GraphAttentionLayer(nn.Module):
    """One Pre-LayerNorm Transformer block with attention-weight caching.

    Args:
        d_model:        Node feature dimensionality.
        nhead:          Number of attention heads.
        dim_feedforward:FFN hidden dimensionality.
        dropout:        Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # batch_first=True so tensors flow as (B, N, D)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # Cached for visualisation — set during forward, detached
        self._last_attn_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) — N node features.

        Returns:
            (B, N, D) — updated node features.
        """
        # ── Self-attention (Pre-LN) ───────────────────────────────────
        residual = x
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(
            x_norm, x_norm, x_norm,
            need_weights=True,
            average_attn_weights=True,   # average over heads → (B, N, N)
        )
        # Cache (detached) for visualisation
        self._last_attn_weights = attn_weights.detach()
        x = residual + attn_out

        # ── Feed-forward (Pre-LN) ─────────────────────────────────────
        x = x + self.ffn(self.norm2(x))
        return x

    @property
    def attention_weights(self) -> torch.Tensor | None:
        """Last (B, N, N) head-averaged attention matrix. Call after forward."""
        return self._last_attn_weights
