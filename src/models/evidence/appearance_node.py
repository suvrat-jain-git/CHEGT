"""Appearance Evidence Node.

Captures *what* a person looks like:
  - Clothing colour and texture (from DINOv2 semantics)
  - Body silhouette shape
  - Temporally-attended summary (selects most informative frames)
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig


class AppearanceNode(nn.Module):
    """Temporal attention pooling + appearance projection.

    Uses a two-step aggregation:
    1. Soft attention over T frames to find the most appearance-rich frames.
    2. A mean-pooled residual for robustness.

    Args:
        input_dim:  Temporal feature dimensionality D.
        hidden_dim: Internal projection width.
        output_dim: Output evidence dimensionality.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()

        # Temporal attention scorer
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Feature encoder applied to the weighted summary
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Mean-pooled residual encoder (global appearance stability)
        self.mean_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Store attention weights for visualisation
        self._attn_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)

        Returns:
            (B, output_dim) — appearance evidence descriptor.
        """
        # ── Temporal attention pooling ────────────────────────────────
        scores = self.attention(x)                         # (B, T, 1)
        weights = torch.softmax(scores, dim=1)             # (B, T, 1)
        self._attn_weights = weights.squeeze(-1).detach()  # (B, T) — for vis
        pooled_attn = (x * weights).sum(dim=1)             # (B, D)
        attn_feat = self.encoder(pooled_attn)              # (B, hidden)

        # ── Mean pooled residual ──────────────────────────────────────
        pooled_mean = x.mean(dim=1)                        # (B, D)
        mean_feat = self.mean_encoder(pooled_mean)         # (B, hidden//2)

        # ── Fuse and project ──────────────────────────────────────────
        combined = torch.cat([attn_feat, mean_feat], dim=-1)
        return self.projection(combined)                   # (B, output_dim)

    @property
    def temporal_attention_weights(self) -> torch.Tensor | None:
        """Last computed per-frame attention weights (B, T). Call after forward."""
        return self._attn_weights
