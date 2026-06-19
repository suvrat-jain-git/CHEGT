"""Synchronized spatial augmentations for gait video sequences.

All spatial transforms (flip, crop) are applied identically to every
frame in a sequence.  Per-frame colour jitter is intentional — it
provides robustness to illumination changes across a sequence.
"""

import random
from typing import List, Tuple

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from omegaconf import DictConfig


class GaitTransform:
    """Transform pipeline for a list of PIL frames.

    Args:
        cfg:   Dataset config (image_size, color_jitter_* fields).
        split: ``"train"`` or ``"val"`` / ``"test"``.
    """

    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, cfg: DictConfig, split: str = "train") -> None:
        self.split = split
        h, w = cfg.image_size[0], cfg.image_size[1]
        self.target_size: Tuple[int, int] = (h, w)

        # Per-frame colour jitter (stochastic; applied independently)
        self.color_jitter = T.ColorJitter(
            brightness=cfg.get("color_jitter_brightness", 0.2),
            contrast=cfg.get("color_jitter_contrast",   0.2),
            saturation=cfg.get("color_jitter_saturation", 0.2),
            hue=cfg.get("color_jitter_hue", 0.05),
        ) if split == "train" else None

        # Random erasing applied to each tensor independently
        self.random_erasing = T.RandomErasing(p=0.3, scale=(0.02, 0.08)) \
            if split == "train" else None

        self.normalize = T.Normalize(self._IMAGENET_MEAN, self._IMAGENET_STD)

    def __call__(self, frames: List[Image.Image]) -> torch.Tensor:
        """Apply transforms to a list of PIL frames.

        Args:
            frames: T PIL Images (RGB).

        Returns:
            Tensor of shape (T, C, H, W).
        """
        # ── Shared spatial decisions ─────────────────────────────────
        do_hflip   = self.split == "train" and random.random() < 0.5
        # Random crop parameters (same across all frames)
        crop_params = self._get_crop_params() if self.split == "train" else None

        transformed: List[torch.Tensor] = []
        for frame in frames:
            # Resize
            frame = TF.resize(frame, list(self.target_size))

            # Consistent horizontal flip
            if do_hflip:
                frame = TF.hflip(frame)

            # Consistent random crop (applied after resize → slight padding)
            if crop_params is not None:
                frame = TF.pad(frame, padding=8)
                i, j, h, w = crop_params
                frame = TF.crop(frame, i, j, h, w)
                frame = TF.resize(frame, list(self.target_size))

            # Independent colour jitter per frame
            if self.color_jitter is not None and random.random() < 0.8:
                frame = self.color_jitter(frame)

            # To tensor and normalise
            t = TF.to_tensor(frame)
            t = self.normalize(t)

            # Independent random erasing
            if self.random_erasing is not None:
                t = self.random_erasing(t)

            transformed.append(t)

        return torch.stack(transformed, dim=0)  # (T, C, H, W)

    def _get_crop_params(self) -> Tuple[int, int, int, int]:
        """Compute a single crop window reused across all frames."""
        h, w = self.target_size[0] + 16, self.target_size[1] + 16  # padded dims
        crop_h, crop_w = self.target_size
        i = random.randint(0, h - crop_h)
        j = random.randint(0, w - crop_w)
        return i, j, crop_h, crop_w


def get_transforms(cfg: DictConfig, split: str = "train") -> GaitTransform:
    """Factory function."""
    return GaitTransform(cfg, split=split)
