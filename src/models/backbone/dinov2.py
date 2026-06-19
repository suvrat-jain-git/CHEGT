"""DINOv2 Vision Transformer backbone.

Wraps Meta's DINOv2 models (loaded via torch.hub) and exposes a clean
``forward()`` that returns the CLS token as a compact frame descriptor.

Supported model names:
    - ``dinov2_vits14``  (ViT-S/14, 384-d)
    - ``dinov2_vitb14``  (ViT-B/14, 768-d)  ← default
    - ``dinov2_vitl14``  (ViT-L/14, 1024-d)
    - ``dinov2_vitg14``  (ViT-G/14, 1536-d)
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

_DINOV2_OUTPUT_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


class DINOv2Backbone(nn.Module):
    """DINOv2 backbone that returns CLS-token frame features.

    Args:
        cfg: Backbone config with fields:
            - ``name``         (str)  Model name, e.g. ``dinov2_vitb14``.
            - ``pretrained``   (bool) Load pretrained weights from hub.
            - ``freeze_layers``(int)  Freeze first N transformer blocks.
                                      ``-1`` freezes everything except head.
            - ``output_dim``   (int)  Expected output dimensionality (for assertion).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.model_name: str = cfg.name
        self.output_dim: int = cfg.output_dim

        # ── Load from hub ─────────────────────────────────────────────
        logger.info(f"Loading DINOv2 backbone: {self.model_name}")
        self.vit = torch.hub.load(
            "facebookresearch/dinov2",
            self.model_name,
            pretrained=cfg.pretrained,
        )

        # Verify output dim
        expected = _DINOV2_OUTPUT_DIMS.get(self.model_name)
        if expected and expected != cfg.output_dim:
            raise ValueError(
                f"backbone.output_dim={cfg.output_dim} but {self.model_name} "
                f"produces {expected}-d features. Fix configs/model.yaml."
            )

        # ── Freeze layers ─────────────────────────────────────────────
        n_freeze: int = cfg.freeze_layers
        if n_freeze == -1:
            # Freeze everything
            for param in self.vit.parameters():
                param.requires_grad_(False)
            logger.info("DINOv2: all layers frozen.")
        elif n_freeze > 0:
            # Always freeze patch embed + positional embed
            for param in self.vit.patch_embed.parameters():
                param.requires_grad_(False)
            self.vit.pos_embed.requires_grad_(False)
            self.vit.cls_token.requires_grad_(False)
            # Freeze first n_freeze transformer blocks
            for i, block in enumerate(self.vit.blocks):
                if i < n_freeze:
                    for param in block.parameters():
                        param.requires_grad_(False)
            n_total = len(self.vit.blocks)
            logger.info(
                f"DINOv2: frozen patch_embed + first {n_freeze}/{n_total} blocks."
            )
        else:
            logger.info("DINOv2: all layers trainable.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS-token features for a batch of frames.

        Args:
            x: (N, C, H, W) — can be B*T frames flattened.

        Returns:
            features: (N, output_dim) CLS-token descriptors.
        """
        # DINOv2's forward_features returns a dict with 'x_norm_clstoken'
        out = self.vit.forward_features(x)
        return out["x_norm_clstoken"]  # (N, D)

    def get_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return spatial patch tokens (N, num_patches, D) for optional use."""
        out = self.vit.forward_features(x)
        return out["x_norm_patchtokens"]

    def trainable_parameters(self):
        """Yield only trainable parameters (convenience for optimizer)."""
        return (p for p in self.parameters() if p.requires_grad)
