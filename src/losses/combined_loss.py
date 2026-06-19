"""Combined loss function for CHEGT training.

Currently: weighted triplet loss (primary).
Easily extensible to add cross-entropy / label smoothing if desired.
"""

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.losses.triplet_loss import TripletLoss


@dataclass
class LossOutput:
    total:    torch.Tensor
    triplet:  torch.Tensor
    frac_pos: torch.Tensor          # fraction of non-trivial triplets


class CombinedLoss(nn.Module):
    """Weighted sum of all configured loss components.

    Args:
        cfg: Loss config section (cfg.loss).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        trip_cfg = cfg.triplet

        self.triplet_weight = float(trip_cfg.weight)
        self.triplet_loss   = TripletLoss(
            margin=trip_cfg.margin,
            mining=trip_cfg.mining,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> LossOutput:
        """
        Args:
            embeddings: (B, D) — model output embeddings.
            labels:     (B,)   — integer identity labels.

        Returns:
            LossOutput with scalar ``total`` and component losses.
        """
        triplet, frac_pos = self.triplet_loss(embeddings, labels)
        total = self.triplet_weight * triplet

        return LossOutput(total=total, triplet=triplet, frac_pos=frac_pos)

    def to_dict(self, loss_out: LossOutput) -> Dict[str, float]:
        """Convert LossOutput to a plain dict of floats (for logging)."""
        return {
            "loss/total":    loss_out.total.item(),
            "loss/triplet":  loss_out.triplet.item(),
            "loss/frac_pos": loss_out.frac_pos.item(),
        }
