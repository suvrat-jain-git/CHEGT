"""Temporal Transformer for gait sequence modelling.

Takes a sequence of per-frame DINOv2 features and applies multi-layer
self-attention across the time dimension, enabling the model to capture
long-range temporal dependencies (e.g. stride periodicity).
"""

import math

import torch
import torch.nn as nn
from omegaconf import DictConfig


class SinusoidalPositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                            # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embedding (GPT-style)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        x = x + self.embedding(positions)
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """Multi-layer Transformer over the temporal dimension.

    Inputs  : (B, T, D_bb)  — sequence of frame features from DINOv2
    Outputs : (B, T, D)     — temporally-enriched frame representations

    Args:
        cfg: Temporal config with:
            - ``d_model``         int   Input/output dim (= backbone.output_dim).
            - ``nhead``           int   Attention heads.
            - ``num_layers``      int   Transformer encoder layers.
            - ``dim_feedforward`` int   FFN hidden dim.
            - ``dropout``         float Dropout probability.
            - ``max_seq_len``     int   Maximum supported sequence length.
            - ``pos_encoding``    str   ``"sinusoidal"`` or ``"learned"``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        d_model        = cfg.d_model
        nhead          = cfg.nhead
        num_layers     = cfg.num_layers
        dim_feedforward= cfg.dim_feedforward
        dropout        = cfg.dropout
        max_seq_len    = cfg.max_seq_len

        # ── Positional encoding ───────────────────────────────────────
        if cfg.pos_encoding == "learned":
            self.pos_enc = LearnedPositionalEncoding(d_model, max_seq_len, dropout)
        else:
            self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        # ── Transformer encoder ───────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # (B, T, D) convention
            norm_first=True,    # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D_bb)

        Returns:
            out: (B, T, D_model)  — temporally contextualised features.
        """
        x = self.pos_enc(x)
        out = self.transformer(x)
        return out
