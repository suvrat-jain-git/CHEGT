"""Morphology Evidence Node.

Captures body shape and spatial structure using DINOv2 patch tokens
rather than CLS tokens — genuinely orthogonal to Motion (temporal
dynamics) and Appearance (global CLS semantics).

For 224×112 images with patch_size=14:
  patch grid  = 16 (H) × 8 (W) = 128 patches per frame
  patch_dim   = 768 (DINOv2-B)

Three branches:
  1. Spatial attention pooling  → which patches carry body structure
  2. Vertical stripe pooling    → head / torso / leg proportions
  3. Temporal shape consistency → how stable body shape is across frames
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MorphologyNode(nn.Module):
    """Spatial body shape encoder from DINOv2 patch tokens.

    Args:
        patch_dim:   Dimensionality of each patch token (768 for DINOv2-B).
        num_patches: Number of spatial patches per frame (128 for 224×112/14).
        hidden_dim:  Internal projection width.
        output_dim:  Output evidence dimensionality.
        grid_h:      Patch grid height (16 for 224/14).
        grid_w:      Patch grid width  (8  for 112/14).
    """

    def __init__(
        self,
        patch_dim:   int,
        num_patches: int,
        hidden_dim:  int,
        output_dim:  int,
        grid_h:      int = 16,
        grid_w:      int = 8,
    ) -> None:
        super().__init__()
        self.patch_dim   = patch_dim
        self.num_patches = num_patches
        self.grid_h      = grid_h
        self.grid_w      = grid_w

        half   = hidden_dim // 2
        quart  = hidden_dim // 4

        # ── Branch 1: spatial attention pooling ───────────────────────
        # Scores each patch → weighted spatial summary
        self.patch_scorer  = nn.Sequential(
            nn.Linear(patch_dim, half),
            nn.GELU(),
            nn.Linear(half, 1),
        )
        self.spatial_proj  = nn.Sequential(
            nn.Linear(patch_dim, half),
            nn.LayerNorm(half),
            nn.GELU(),
        )

        # ── Branch 2: vertical stripe pooling ─────────────────────────
        # grid_h stripes, each is mean of grid_w patches (D-dim)
        # Encodes per-stripe then flattens → captures body proportions
        self.stripe_encoder = nn.Sequential(
            nn.Linear(patch_dim, half),
            nn.LayerNorm(half),
            nn.GELU(),
        )
        self.stripe_proj = nn.Sequential(
            nn.Linear(grid_h * half, half),
            nn.LayerNorm(half),
            nn.GELU(),
        )

        # ── Branch 3: temporal shape consistency ──────────────────────
        # Variance of per-frame spatial summary → shape stability
        self.consistency_proj = nn.Sequential(
            nn.Linear(patch_dim, quart),
            nn.LayerNorm(quart),
            nn.GELU(),
        )

        fusion_dim = half + half + quart

        self.projection = nn.Sequential(
            nn.Linear(fusion_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, T, N, D) — patch tokens per frame.
                          N = num_patches, D = patch_dim.

        Returns:
            (B, output_dim) — morphology evidence descriptor.
        """
        B, T, N, D = patch_tokens.shape

        # ── Temporal average → (B, N, D) ─────────────────────────────
        mean_patches = patch_tokens.mean(dim=1)                  # (B, N, D)

        # ── Branch 1: Spatial attention pooling ───────────────────────
        scores       = self.patch_scorer(mean_patches)           # (B, N, 1)
        weights      = torch.softmax(scores, dim=1)              # (B, N, 1)
        spatial_pool = (mean_patches * weights).sum(dim=1)       # (B, D)
        spatial_feat = self.spatial_proj(spatial_pool)           # (B, half)

        # ── Branch 2: Vertical stripe pooling ─────────────────────────
        # Reshape to (B, grid_h, grid_w, D), mean over width
        grid = mean_patches.view(B, self.grid_h, self.grid_w, D)
        stripes      = grid.mean(dim=2)                          # (B, H, D)
        stripe_enc   = self.stripe_encoder(stripes)              # (B, H, half)
        stripe_flat  = stripe_enc.reshape(B, -1)                 # (B, H*half)
        stripe_feat  = self.stripe_proj(stripe_flat)             # (B, half)

        # ── Branch 3: Temporal shape consistency ──────────────────────
        # Per-frame spatial summaries → (B, T, D)
        flat_patches = patch_tokens.view(B * T, N, D)
        flat_scores  = self.patch_scorer(flat_patches)           # (B*T, N, 1)
        flat_weights = torch.softmax(flat_scores, dim=1)
        frame_pool   = (flat_patches * flat_weights).sum(dim=1)  # (B*T, D)
        frame_pool   = frame_pool.view(B, T, D)                  # (B, T, D)
        shape_var    = frame_pool.var(dim=1, unbiased=False)     # (B, D)
        consist_feat = self.consistency_proj(shape_var)          # (B, quart)

        # ── Fuse ──────────────────────────────────────────────────────
        combined = torch.cat([spatial_feat, stripe_feat, consist_feat], dim=-1)
        return self.projection(combined)                         # (B, output_dim)
