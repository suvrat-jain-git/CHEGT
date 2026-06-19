"""Evidence Extractor — orchestrates three evidence nodes.

Evidence types:
  Motion      — How the person moves (CLS token temporal dynamics)
  Appearance  — What they look like  (CLS token temporal attention)
  Morphology  — Body shape           (patch token spatial structure)  ← replaces Consistency

All three receive different input signals:
  Motion + Appearance: CLS tokens  (B, T, D_cls)
  Morphology:          patch tokens (B, T, N_patches, D_patch)
"""

from typing import Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.evidence.motion_node      import MotionNode
from src.models.evidence.appearance_node  import AppearanceNode
from src.models.evidence.morphology_node  import MorphologyNode


class EvidenceExtractor(nn.Module):
    """Run three evidence nodes in parallel.

    Returns (motion_feat, appearance_feat, morphology_feat)
    each of shape (B, evidence_dim).

    Args:
        cfg:            cfg.evidence section.
        cls_input_dim:  CLS token dimensionality (temporal transformer output).
        patch_dim:      Patch token dimensionality (DINOv2 patch tokens).
        num_patches:    Number of patches per frame.
        grid_h:         Patch grid height.
        grid_w:         Patch grid width.
    """

    def __init__(
        self,
        cfg:           DictConfig,
        cls_input_dim: int,
        patch_dim:     int  = 768,
        num_patches:   int  = 128,
        grid_h:        int  = 16,
        grid_w:        int  = 8,
    ) -> None:
        super().__init__()
        ev_cfg  = cfg

        out_dim = ev_cfg.motion.output_dim
        assert ev_cfg.appearance.output_dim  == out_dim, \
            "evidence.*.output_dim must match across all nodes."
        assert ev_cfg.morphology.output_dim  == out_dim, \
            "evidence.morphology.output_dim must match other nodes."

        self.motion_node = MotionNode(
            input_dim=cls_input_dim,
            hidden_dim=ev_cfg.motion.hidden_dim,
            output_dim=out_dim,
        ) if ev_cfg.motion.get("enabled", True) else None

        self.appearance_node = AppearanceNode(
            input_dim=cls_input_dim,
            hidden_dim=ev_cfg.appearance.hidden_dim,
            output_dim=out_dim,
        ) if ev_cfg.appearance.get("enabled", True) else None

        self.morphology_node = MorphologyNode(
            patch_dim=patch_dim,
            num_patches=num_patches,
            hidden_dim=ev_cfg.morphology.hidden_dim,
            output_dim=out_dim,
            grid_h=grid_h,
            grid_w=grid_w,
        ) if ev_cfg.morphology.get("enabled", True) else None

        self.output_dim = out_dim
        self._out_dim   = out_dim

    def forward(
        self,
        cls_tokens:    torch.Tensor,
        patch_tokens:  torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            cls_tokens:   (B, T, D_cls)      — temporal transformer output.
            patch_tokens: (B, T, N, D_patch) — raw patch tokens per frame.

        Returns:
            motion_feat:     (B, evidence_dim)
            appearance_feat: (B, evidence_dim)
            morphology_feat: (B, evidence_dim)
        """
        B = cls_tokens.size(0)
        zeros = torch.zeros(B, self._out_dim,
                            device=cls_tokens.device,
                            dtype=cls_tokens.dtype)

        motion_feat     = self.motion_node(cls_tokens)       if self.motion_node      else zeros
        appearance_feat = self.appearance_node(cls_tokens)   if self.appearance_node  else zeros
        morphology_feat = self.morphology_node(patch_tokens) if self.morphology_node  else zeros

        return motion_feat, appearance_feat, morphology_feat
