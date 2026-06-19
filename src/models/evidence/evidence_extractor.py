"""Evidence Extractor — orchestrates the three evidence nodes."""

from typing import Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.evidence.motion_node      import MotionNode
from src.models.evidence.appearance_node  import AppearanceNode
from src.models.evidence.consistency_node import ConsistencyNode


class EvidenceExtractor(nn.Module):
    """Runs the three evidence nodes in parallel.

    Returns a 3-tuple of evidence tensors:
        (motion, appearance, consistency)   — each (B, evidence_dim)

    The evidence_dim is taken from ``cfg.evidence.motion.output_dim``
    and must be identical across all three nodes so they can be stacked
    into a uniform node tensor for the graph transformer.

    Args:
        cfg:       Full model config (uses ``cfg.evidence``).
        input_dim: Temporal feature dim D (from TemporalTransformer).
    """

    def __init__(self, cfg: DictConfig, input_dim: int) -> None:
        super().__init__()
        ev_cfg = cfg  # cfg is already cfg.evidence at call site

        # All three nodes share the same output_dim
        out_dim = ev_cfg.motion.output_dim
        assert ev_cfg.appearance.output_dim  == out_dim, \
            "evidence.*.output_dim must match across all nodes."
        assert ev_cfg.consistency.output_dim == out_dim, \
            "evidence.*.output_dim must match across all nodes."

        # FIXED MAJOR-2: respect the enabled flag from config for clean ablations
        self.motion_node = MotionNode(
            input_dim=input_dim,
            hidden_dim=ev_cfg.motion.hidden_dim,
            output_dim=out_dim,
        ) if ev_cfg.motion.get("enabled", True) else None

        self.appearance_node = AppearanceNode(
            input_dim=input_dim,
            hidden_dim=ev_cfg.appearance.hidden_dim,
            output_dim=out_dim,
        ) if ev_cfg.appearance.get("enabled", True) else None

        self.consistency_node = ConsistencyNode(
            input_dim=input_dim,
            hidden_dim=ev_cfg.consistency.hidden_dim,
            output_dim=out_dim,
        ) if ev_cfg.consistency.get("enabled", True) else None

        self.output_dim = out_dim
        self._out_dim   = out_dim

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D) — output of TemporalTransformer.

        Returns:
            motion_feat     : (B, evidence_dim)
            appearance_feat : (B, evidence_dim)
            consistency_feat: (B, evidence_dim)
        """
        B = x.size(0)
        zeros = torch.zeros(B, self._out_dim, device=x.device, dtype=x.dtype)

        # FIXED MAJOR-2: disabled nodes return zero tensors (clean ablation)
        motion_feat      = self.motion_node(x)      if self.motion_node      else zeros
        appearance_feat  = self.appearance_node(x)  if self.appearance_node  else zeros
        consistency_feat = self.consistency_node(x) if self.consistency_node else zeros
        return motion_feat, appearance_feat, consistency_feat
