"""Batch-hard triplet loss with optional semi-hard and all-pairs variants.

Reference: Hermans et al. "In Defense of the Triplet Loss" (2017).
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_distance(
    embeddings: torch.Tensor,
    squared: bool = False,
) -> torch.Tensor:
    """Compute all-pairs L2 distance matrix.

    Args:
        embeddings: (B, D) — L2-normalised embeddings.
        squared:    Return squared distances (avoids sqrt instability).

    Returns:
        dist_mat: (B, B) — symmetric distance matrix.
    """
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a^T*b
    dot = embeddings @ embeddings.t()              # (B, B)
    sq  = dot.diagonal().unsqueeze(0)             # (1, B)
    dist = sq.t() + sq - 2.0 * dot               # (B, B)
    dist = dist.clamp(min=0.0)

    if not squared:
        # Add small epsilon for numerical stability of sqrt
        mask = dist.eq(0.0).float()
        dist = (dist + mask * 1e-16).sqrt()
        dist = dist * (1.0 - mask)

    return dist


class TripletLoss(nn.Module):
    """Batch-hard triplet loss.

    For each anchor in a batch:
        - Hardest positive:  farthest sample with the same label.
        - Hardest negative:  closest sample with a different label.

    Args:
        margin: Triplet margin α.  Use ``None`` for soft-plus version.
        mining: ``"hard"`` | ``"semi_hard"`` | ``"all"``.
        squared_dist: Use squared L2 (avoids gradient singularity at 0).
    """

    def __init__(
        self,
        margin: float = 0.3,
        mining: str = "hard",
        squared_dist: bool = False,
    ) -> None:
        super().__init__()
        self.margin      = margin
        self.mining      = mining
        self.squared_dist = squared_dist

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: (B, D) — L2-normalised embeddings.
            labels:     (B,)   — integer identity labels.

        Returns:
            loss:          Scalar triplet loss.
            frac_positive: Fraction of non-trivial (positive) triplets.
        """
        dist_mat = pairwise_distance(embeddings, squared=self.squared_dist)

        if self.mining == "hard":
            return self._batch_hard(dist_mat, labels)
        elif self.mining == "semi_hard":
            return self._semi_hard(dist_mat, labels)
        else:
            return self._all_pairs(dist_mat, labels)

    # ── Mining strategies ─────────────────────────────────────────────

    def _batch_hard(
        self,
        dist_mat: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = dist_mat.size(0)
        labels = labels.view(B, 1)

        same_id   = labels.eq(labels.t())          # (B, B) bool
        diff_id   = ~same_id

        # Hardest positive: mask diagonal, take max among same-id
        pos_dist = dist_mat.clone()
        pos_dist[~same_id] = 0.0
        pos_dist.fill_diagonal_(0.0)
        hardest_pos, _ = pos_dist.max(dim=1)       # (B,)

        # Hardest negative: set same-id to very large, take min
        neg_dist = dist_mat.clone()
        neg_dist[same_id] = 1e9
        hardest_neg, _ = neg_dist.min(dim=1)       # (B,)

        triplet_loss = torch.clamp(hardest_pos - hardest_neg + self.margin, min=0.0)

        # Fraction of non-trivial triplets (loss > 0)
        frac = (triplet_loss > 0.0).float().mean()
        return triplet_loss.mean(), frac

    def _semi_hard(
        self,
        dist_mat: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Semi-hard negative mining: neg closer than pos + margin."""
        B = dist_mat.size(0)
        labels_row = labels.view(B, 1)

        same_id = labels_row.eq(labels_row.t())
        diff_id = ~same_id

        losses = []
        for i in range(B):
            pos_dists = dist_mat[i][same_id[i] & (torch.arange(B, device=dist_mat.device) != i)]
            if pos_dists.numel() == 0:
                continue
            d_pos = pos_dists.max()
            # Semi-hard: neg s.t. d_pos < d_neg < d_pos + margin
            neg_dists = dist_mat[i][diff_id[i]]
            mask = (neg_dists > d_pos) & (neg_dists < d_pos + self.margin)
            if mask.sum() > 0:
                d_neg = neg_dists[mask].min()
            else:
                d_neg = neg_dists.min()
            losses.append(torch.clamp(d_pos - d_neg + self.margin, min=0.0))

        if not losses:
            return dist_mat.sum() * 0.0, torch.tensor(0.0)

        loss_t = torch.stack(losses)
        return loss_t.mean(), (loss_t > 0).float().mean()

    def _all_pairs(
        self,
        dist_mat: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """All valid (a, p, n) combinations."""
        B = dist_mat.size(0)
        labels_row = labels.view(B, 1)

        same_id = labels_row.eq(labels_row.t())

        # Expand to (B, B, B) triplets
        anchor_pos = dist_mat.unsqueeze(2)      # (B, B, 1)  — a-p
        anchor_neg = dist_mat.unsqueeze(1)      # (B, 1, B)  — a-n

        valid_pos = same_id.unsqueeze(2) & (~torch.eye(B, dtype=torch.bool, device=dist_mat.device)).unsqueeze(2)
        valid_neg = (~same_id).unsqueeze(1)
        valid     = valid_pos & valid_neg

        loss_triplets = torch.clamp(anchor_pos - anchor_neg + self.margin, min=0.0)
        loss_triplets = loss_triplets[valid]

        if loss_triplets.numel() == 0:
            return dist_mat.sum() * 0.0, torch.tensor(0.0)

        frac = (loss_triplets > 0).float().mean()
        return loss_triplets.mean(), frac
