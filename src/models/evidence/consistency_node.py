"""Consistency Evidence Node.

Captures *how stable* a person's gait signature is over time:
  - Feature variance across frames  (low = very consistent appearance)
  - Lag-1 autocorrelation           (measures temporal regularity / periodicity)
  - Mean absolute deviation         (robust measure of variability)
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig


class ConsistencyNode(nn.Module):
    """Temporal stability and periodicity encoder.

    Computes three statistics over the T-length feature sequence:
    variance, autocorrelation, and mean absolute deviation.  Each is
    projected and then fused into ``output_dim``.

    Args:
        input_dim:  Temporal feature dimensionality D.
        hidden_dim: Internal projection width.
        output_dim: Output evidence dimensionality.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()

        # Variance branch
        self.var_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 3),
            nn.LayerNorm(hidden_dim // 3),
            nn.GELU(),
        )

        # Autocorrelation branch
        self.autocorr_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 3),
            nn.LayerNorm(hidden_dim // 3),
            nn.GELU(),
        )

        # Mean absolute deviation branch
        self.mad_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 3),
            nn.LayerNorm(hidden_dim // 3),
            nn.GELU(),
        )

        fusion_dim = (hidden_dim // 3) * 3

        self.projection = nn.Sequential(
            nn.Linear(fusion_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)

        Returns:
            (B, output_dim) — consistency evidence descriptor.
        """
        B, T, D = x.shape

        # FIXED MAJOR-1: subtract mean for correct centered statistics
        mean = x.mean(dim=1, keepdim=True)           # (B, 1, D)
        x_centered = x - mean                         # (B, T, D)
        mean_sq = mean.squeeze(1)                     # (B, D)

        # ── Variance ──────────────────────────────────────────────────
        var = x_centered.var(dim=1, unbiased=False)   # (B, D)
        var_feat = self.var_encoder(var)               # (B, hidden//3)

        # ── Lag-1 autocorrelation (correctly centered) ────────────────
        if T > 1:
            x_t  = x_centered[:, :-1, :]              # (B, T-1, D)
            x_t1 = x_centered[:, 1:,  :]              # (B, T-1, D)
            cov      = (x_t * x_t1).mean(dim=1)       # E[(x_t-μ)(x_{t+1}-μ)]  (B,D)
            autocorr = cov / (var.clamp(min=1e-8))    # ρ ∈ [-1, 1]
            autocorr = autocorr.clamp(-1.0, 1.0)
        else:
            autocorr = torch.zeros(B, D, device=x.device)
        autocorr_feat = self.autocorr_encoder(autocorr)     # (B, hidden//3)

        # ── Mean Absolute Deviation ───────────────────────────────────
        mad = x_centered.abs().mean(dim=1)             # (B, D)
        mad_feat = self.mad_encoder(mad)               # (B, hidden//3)

        # ── Fuse ──────────────────────────────────────────────────────
        combined = torch.cat([var_feat, autocorr_feat, mad_feat], dim=-1)
        return self.projection(combined)                     # (B, output_dim)
