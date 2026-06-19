"""Combined loss — Triplet + ArcFace."""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.losses.triplet_loss  import TripletLoss
from src.losses.arcface_loss  import ArcFaceLoss


@dataclass
class LossOutput:
    total:    torch.Tensor
    triplet:  torch.Tensor
    arcface:  torch.Tensor
    frac_pos: torch.Tensor


class CombinedLoss(nn.Module):
    """Weighted Triplet + ArcFace loss.

    Args:
        cfg:         Loss config section (cfg.loss).
        embed_dim:   Embedding dimensionality (for ArcFace weight matrix).
        num_classes: Number of training identities (for ArcFace).
    """

    def __init__(
        self,
        cfg:         DictConfig,
        embed_dim:   int = 512,
        num_classes: int = 75,
    ) -> None:
        super().__init__()
        trip_cfg = cfg.triplet
        arc_cfg  = cfg.get("arcface", None)

        self.triplet_weight = float(trip_cfg.weight)
        self.triplet_loss   = TripletLoss(
            margin=trip_cfg.margin,
            mining=trip_cfg.mining,
        )

        # ArcFace — only instantiate if configured
        self.arcface_weight = float(arc_cfg.weight) if arc_cfg else 0.0
        self.arcface_loss   = ArcFaceLoss(
            embed_dim=embed_dim,
            num_classes=num_classes,
            margin=float(arc_cfg.margin) if arc_cfg else 0.5,
            scale=float(arc_cfg.scale)   if arc_cfg else 64.0,
            easy_margin=bool(arc_cfg.get("easy_margin", False)) if arc_cfg else False,
        ) if arc_cfg else None

    def forward(
        self,
        embeddings: torch.Tensor,
        labels:     torch.Tensor,
    ) -> LossOutput:
        """
        Args:
            embeddings: (B, embed_dim) — L2-normalised embeddings.
            labels:     (B,)           — integer identity labels.
        """
        triplet, frac_pos = self.triplet_loss(embeddings, labels)
        total = self.triplet_weight * triplet

        arcface = torch.tensor(0.0, device=embeddings.device)
        if self.arcface_loss is not None:
            arcface = self.arcface_loss(embeddings, labels)
            total   = total + self.arcface_weight * arcface

        return LossOutput(
            total=total,
            triplet=triplet,
            arcface=arcface,
            frac_pos=frac_pos,
        )

    def to_dict(self, loss_out: LossOutput) -> Dict[str, float]:
        return {
            "loss/total":    loss_out.total.item(),
            "loss/triplet":  loss_out.triplet.item(),
            "loss/arcface":  loss_out.arcface.item(),
            "loss/frac_pos": loss_out.frac_pos.item(),
        }
