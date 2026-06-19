"""Gait Embedding Head.

Projects the pooled evidence descriptor to the final L2-normalised
embedding space used for retrieval and verification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class GaitHead(nn.Module):
    """Two-layer MLP projection head with optional L2 normalisation.

    Args:
        cfg:       Head config (input_dim, hidden_dim, embedding_dim,
                   normalize, dropout).
        input_dim: Overrides cfg.input_dim when provided.
    """

    def __init__(self, cfg: DictConfig, input_dim: int | None = None) -> None:
        super().__init__()
        in_dim      = input_dim if input_dim is not None else cfg.input_dim
        hidden_dim  = cfg.hidden_dim
        embed_dim   = cfg.embedding_dim
        dropout     = cfg.dropout
        self.normalize = cfg.normalize

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Final BN before normalisation (BNNeck — standard in ReID)
        self.bn = nn.BatchNorm1d(embed_dim, affine=True)
        # BN bias is not needed when L2 normalising
        nn.init.constant_(self.bn.bias, 0)
        self.bn.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim)

        Returns:
            embedding: (B, embedding_dim) — L2-normalised if cfg.normalize=True.
        """
        x = self.net(x)
        x = self.bn(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
        return x
