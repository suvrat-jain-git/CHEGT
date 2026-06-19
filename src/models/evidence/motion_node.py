"""Motion Evidence Node.

Captures *how* a person moves:
  - Inter-frame velocity (feature-space differences)
  - Motion periodicity via FFT (stride cadence)
  - Acceleration (second-order differences)

All three cues are fused and projected to ``output_dim``.
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig


class MotionNode(nn.Module):
    """
    Args:
        input_dim:  Dimensionality of temporal features (D from TemporalTransformer).
        hidden_dim: Internal projection width.
        output_dim: Output evidence dimensionality.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim

        # Velocity branch: summarises inter-frame differences
        self.velocity_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
        )

        # Frequency branch: captures periodicity (stride cadence)
        self.freq_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )

        # Acceleration branch: second-order temporal change
        self.accel_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
        )

        fusion_dim = hidden_dim // 2 + hidden_dim // 2 + hidden_dim // 4

        self.projection = nn.Sequential(
            nn.Linear(fusion_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) — temporally-encoded frame features.

        Returns:
            (B, output_dim) — motion evidence descriptor.
        """
        B, T, D = x.shape

        # ── Velocity: weighted mean of inter-frame differences ────────
        delta = x[:, 1:, :] - x[:, :-1, :]                # (B, T-1, D)
        speed = delta.norm(dim=-1, keepdim=True)            # (B, T-1, 1)
        weights = torch.softmax(speed, dim=1)               # (B, T-1, 1)
        vel_summary = (delta * weights).sum(dim=1)          # (B, D)
        vel_feat = self.velocity_encoder(vel_summary)       # (B, hidden//2)

        # ── Frequency: FFT magnitude spectrum over time ───────────────
        # rfft along T produces T//2+1 complex bins; take magnitude
        x_fft = torch.fft.rfft(x, dim=1)                   # (B, T//2+1, D)
        x_fft_mag = x_fft.abs()                            # (B, T//2+1, D)
        freq_summary = x_fft_mag.mean(dim=1)               # (B, D)
        freq_feat = self.freq_encoder(freq_summary)         # (B, hidden//2)

        # ── Acceleration: second-order differences ────────────────────
        if T >= 3:
            delta2 = delta[:, 1:, :] - delta[:, :-1, :]   # (B, T-2, D)
            accel_summary = delta2.abs().mean(dim=1)        # (B, D)
        else:
            accel_summary = torch.zeros(B, D, device=x.device)
        accel_feat = self.accel_encoder(accel_summary)      # (B, hidden//4)

        # ── Fuse ──────────────────────────────────────────────────────
        combined = torch.cat([vel_feat, freq_feat, accel_feat], dim=-1)
        return self.projection(combined)                    # (B, output_dim)
