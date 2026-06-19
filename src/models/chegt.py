"""CHEGT: Cross-clothes Human Evidence Graph Transformer.

Pipeline
--------
1. DINOv2 backbone    → CLS tokens (B,T,D_cls) + patch tokens (B,T,N,D_patch)
2. Temporal Transformer on CLS tokens → (B, T, D_temporal)
3. Evidence Extraction:
     Motion Node      (B, T, D_temporal)   → (B, D_e)
     Appearance Node  (B, T, D_temporal)   → (B, D_e)
     Morphology Node  (B, T, N, D_patch)   → (B, D_e)   ← uses patch tokens
4. Graph Transformer  (B, 3, D_e)          → (B, 3, D_e) + attn (B, 3, 3)
5. Mean Pooling       (B, 3, D_e)          → (B, D_e)
6. Gait Head          (B, D_e)             → (B, embed_dim)
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.backbone.dinov2              import DINOv2Backbone
from src.models.temporal.temporal_transformer import TemporalTransformer
from src.models.evidence.evidence_extractor  import EvidenceExtractor
from src.models.graph.graph_transformer      import GraphTransformer
from src.models.pooling.evidence_pooling     import EvidencePooling
from src.models.heads.gait_head              import GaitHead


@dataclass
class CHEGTOutput:
    embedding:           torch.Tensor   # (B, embed_dim)
    evidence_weights:    torch.Tensor   # (B, 3)
    graph_attention:     torch.Tensor   # (B, 3, 3)
    temporal_features:   torch.Tensor   # (B, T, D_t)
    motion_features:     torch.Tensor   # (B, D_e)
    appearance_features: torch.Tensor   # (B, D_e)
    morphology_features: torch.Tensor   # (B, D_e)


class CHEGT(nn.Module):
    """Cross-clothes Human Evidence Graph Transformer."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        m        = cfg

        self.backbone = DINOv2Backbone(m.backbone)
        self.temporal = TemporalTransformer(m.temporal)

        # Patch grid dimensions for 224×112 with patch_size=14
        grid_h = m.backbone.get("grid_h", 16)
        grid_w = m.backbone.get("grid_w", 8)
        num_patches = grid_h * grid_w

        self.evidence = EvidenceExtractor(
            cfg=m.evidence,
            cls_input_dim=m.temporal.d_model,
            patch_dim=m.backbone.output_dim,
            num_patches=num_patches,
            grid_h=grid_h,
            grid_w=grid_w,
        )

        self.graph   = GraphTransformer(m.graph)
        self.pooling = EvidencePooling(m.pooling, d_model=m.graph.d_model)
        self.head    = GaitHead(m.head, input_dim=m.pooling.d_model)

    def forward(self, x: torch.Tensor) -> CHEGTOutput:
        """
        Args:
            x: (B, T, C, H, W) — batch of video sequences.
        """
        B, T, C, H, W = x.shape

        # 1. Backbone — CLS + patch tokens per frame
        x_flat = x.view(B * T, C, H, W)
        cls_flat, patch_flat = self.backbone(x_flat)
        # cls_flat:   (B*T, D_cls)
        # patch_flat: (B*T, N, D_patch)

        N, D_patch = patch_flat.shape[1], patch_flat.shape[2]
        cls_tokens   = cls_flat.view(B, T, -1)           # (B, T, D_cls)
        patch_tokens = patch_flat.view(B, T, N, D_patch)  # (B, T, N, D_patch)

        # 2. Temporal transformer on CLS tokens
        temporal_feats = self.temporal(cls_tokens)        # (B, T, D_temporal)

        # 3. Evidence extraction
        motion_feat, app_feat, morph_feat = self.evidence(
            cls_tokens=temporal_feats,
            patch_tokens=patch_tokens,
        )

        # 4. Graph reasoning
        nodes = torch.stack([motion_feat, app_feat, morph_feat], dim=1)  # (B, 3, D_e)
        graph_out, attn_weights = self.graph(nodes)       # (B, 3, D_e), (B, 3, 3)

        # 5. Pooling
        pooled, evidence_weights = self.pooling(graph_out)  # (B, D_e), (B, 3)

        # 6. Gait head
        embedding = self.head(pooled)                     # (B, embed_dim)

        return CHEGTOutput(
            embedding=embedding,
            evidence_weights=evidence_weights,
            graph_attention=attn_weights,
            temporal_features=temporal_feats.detach(),
            motion_features=motion_feat.detach(),
            appearance_features=app_feat.detach(),
            morphology_features=morph_feat.detach(),
        )

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).embedding

    def parameter_groups(self, base_lr: float, backbone_lr_mult: float):
        backbone_params = list(self.backbone.parameters())
        backbone_ids    = {id(p) for p in backbone_params}
        non_backbone    = [p for p in self.parameters() if id(p) not in backbone_ids]
        return [
            {"params": [p for p in backbone_params if p.requires_grad],
             "lr": base_lr * backbone_lr_mult, "name": "backbone"},
            {"params": non_backbone,
             "lr": base_lr, "name": "non_backbone"},
        ]

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        n_train = self.num_parameters(trainable_only=True)
        n_total = self.num_parameters(trainable_only=False)
        return (
            f"CHEGT(\n"
            f"  backbone        = {self.cfg.backbone.name}\n"
            f"  evidence        = Motion + Appearance + Morphology\n"
            f"  temporal_layers = {self.cfg.temporal.num_layers}\n"
            f"  graph_layers    = {self.cfg.graph.num_layers}\n"
            f"  pooling         = {self.cfg.pooling.type}\n"
            f"  embed_dim       = {self.cfg.head.embedding_dim}\n"
            f"  trainable_params= {n_train:,}\n"
            f"  total_params    = {n_total:,}\n"
            f")"
        )
