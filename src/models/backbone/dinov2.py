"""DINOv2 backbone — returns CLS token and patch tokens.

For 224×112 images with patch_size=14:
  CLS token   : (N, 768)
  Patch tokens: (N, 128, 768)  where 128 = 16×8 patch grid
"""

import logging
from typing import Tuple

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
    """DINOv2 backbone returning both CLS and patch tokens.

    Args:
        cfg: Backbone config — name, pretrained, freeze_layers, output_dim.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.model_name = cfg.name
        self.output_dim = cfg.output_dim

        logger.info(f"Loading DINOv2 backbone: {self.model_name}")
        self.vit = torch.hub.load(
            "facebookresearch/dinov2",
            self.model_name,
            pretrained=cfg.pretrained,
        )

        expected = _DINOV2_OUTPUT_DIMS.get(self.model_name)
        if expected and expected != cfg.output_dim:
            raise ValueError(
                f"backbone.output_dim={cfg.output_dim} but {self.model_name} "
                f"produces {expected}-d features."
            )

        # Freeze layers
        n_freeze = cfg.freeze_layers
        if n_freeze == -1:
            for param in self.vit.parameters():
                param.requires_grad_(False)
            logger.info("DINOv2: all layers frozen.")
        elif n_freeze > 0:
            for param in self.vit.patch_embed.parameters():
                param.requires_grad_(False)
            self.vit.pos_embed.requires_grad_(False)
            self.vit.cls_token.requires_grad_(False)
            for i, block in enumerate(self.vit.blocks):
                if i < n_freeze:
                    for param in block.parameters():
                        param.requires_grad_(False)
            n_total = len(self.vit.blocks)
            logger.info(f"DINOv2: frozen patch_embed + first {n_freeze}/{n_total} blocks.")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract CLS and patch tokens.

        Args:
            x: (N, C, H, W) — flattened batch of frames.

        Returns:
            cls_tokens:   (N, output_dim)       — global frame descriptor.
            patch_tokens: (N, num_patches, D)   — spatial patch descriptors.
        """
        out          = self.vit.forward_features(x)
        cls_tokens   = out["x_norm_clstoken"]       # (N, D)
        patch_tokens = out["x_norm_patchtokens"]     # (N, num_patches, D)
        return cls_tokens, patch_tokens
