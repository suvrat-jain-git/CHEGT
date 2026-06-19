"""Evidence Pooling — collapses N graph nodes into a single descriptor.

Supports three aggregation modes:
    - ``attention``  Learned per-node importance weights (default, interpretable).
    - ``mean``       Simple uniform average.
    - ``max``        Element-wise maximum across nodes.

The attention weights are stored and exposed for the evidence importance
bar-chart visualisation.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class EvidencePooling(nn.Module):
    """Aggregate evidence node features into a single gait descriptor.

    Args:
        cfg:      Pooling config (type, d_model, hidden_dim).
        d_model:  Node feature dimensionality (from graph transformer output).
    """

    def __init__(self, cfg: DictConfig, d_model: int) -> None:
        super().__init__()
        self.pool_type = cfg.type
        self.d_model   = d_model

        if self.pool_type == "attention":
            # Scalar score per node
            self.scorer = nn.Sequential(
                nn.Linear(d_model, cfg.hidden_dim),
                nn.Tanh(),
                nn.Linear(cfg.hidden_dim, 1),
            )

        # Cached for visualisation
        self._last_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aggregate graph node features.

        Args:
            x: (B, N, D) — output of GraphTransformer.

        Returns:
            pooled  : (B, D) — single aggregate descriptor.
            weights : (B, N) — per-node importance weights (for visualisation).
        """
        if self.pool_type == "attention":
            scores  = self.scorer(x)                      # (B, N, 1)
            weights = torch.softmax(scores, dim=1)         # (B, N, 1)
            pooled  = (x * weights).sum(dim=1)            # (B, D)
            weights = weights.squeeze(-1)                  # (B, N)

        elif self.pool_type == "mean":
            pooled  = x.mean(dim=1)                       # (B, D)
            B, N, _ = x.shape
            weights = torch.full((B, N), 1.0 / N, device=x.device)

        elif self.pool_type == "max":
            pooled, _ = x.max(dim=1)                      # (B, D)
            # Pseudo-weights: argmax indicator
            idx     = x.norm(dim=-1).argmax(dim=1)        # (B,)
            B, N, _ = x.shape
            weights = F.one_hot(idx, num_classes=N).float()

        else:
            raise ValueError(f"Unknown pooling type: {self.pool_type!r}")

        # Cache for post-hoc inspection / visualisation
        self._last_weights = weights.detach()
        return pooled, weights

    @property
    def evidence_importance(self) -> torch.Tensor | None:
        """Per-node importance weights from the last forward pass (B, N)."""
        return self._last_weights
