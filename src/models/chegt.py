"""CHEGT: Cross-clothes Human Evidence Graph Transformer.

Pipeline
--------
1. DINOv2 backbone    (B*T, C, H, W) → (B, T, D_bb)
2. Temporal Transformer              → (B, T, D_temporal)
3. Evidence Extraction               → 3 × (B, D_evidence)
4. Graph Transformer                 → (B, 3, D_evidence) + attn (B, 3, 3)
5. Evidence Pooling                  → (B, D_evidence) + weights (B, 3)
6. Gait Head                         → (B, embed_dim)

The model returns a ``CHEGTOutput`` dataclass that exposes every
intermediate result needed for training, evaluation, and visualisation.
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


# ── Output container ─────────────────────────────────────────────────────────

@dataclass
class CHEGTOutput:
    """All outputs from a single CHEGT forward pass.

    Shapes (B = batch, T = frames, D_e = evidence_dim, E = embed_dim):
        embedding           (B, E)       — final L2-normalised descriptor.
        evidence_weights    (B, 3)       — per-node importance (sums to ≈1).
        graph_attention     (B, 3, 3)    — inter-node attention heatmap.
        temporal_features   (B, T, D_t)  — output of temporal transformer.
        motion_features     (B, D_e)     — raw motion evidence.
        appearance_features (B, D_e)     — raw appearance evidence.
        consistency_features(B, D_e)     — raw consistency evidence.
    """
    embedding:            torch.Tensor
    evidence_weights:     torch.Tensor
    graph_attention:      torch.Tensor
    temporal_features:    torch.Tensor
    motion_features:      torch.Tensor
    appearance_features:  torch.Tensor
    consistency_features: torch.Tensor


# ── Main model ────────────────────────────────────────────────────────────────

class CHEGT(nn.Module):
    """Cross-clothes Human Evidence Graph Transformer.

    Args:
        cfg: Full merged config (model + dataset + train sections).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        m = cfg  # model-level cfg is the full cfg in this project

        self.backbone  = DINOv2Backbone(m.backbone)
        self.temporal  = TemporalTransformer(m.temporal)
        # FIXED MAJOR-3: evidence nodes receive temporal transformer output;
        # use temporal.d_model as the canonical input dim, not backbone.output_dim
        self.evidence  = EvidenceExtractor(m.evidence, input_dim=m.temporal.d_model)
        self.graph     = GraphTransformer(m.graph)
        self.pooling   = EvidencePooling(m.pooling, d_model=m.graph.d_model)
        self.head      = GaitHead(m.head, input_dim=m.pooling.d_model)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> CHEGTOutput:
        """Full CHEGT forward pass.

        Args:
            x: (B, T, C, H, W) — batch of video sequences.

        Returns:
            CHEGTOutput containing embedding and all intermediate results.
        """
        B, T, C, H, W = x.shape

        # 1. Per-frame features via DINOv2
        x_flat      = x.view(B * T, C, H, W)
        frame_feats = self.backbone(x_flat)               # (B*T, D_bb)
        frame_feats = frame_feats.view(B, T, -1)          # (B, T, D_bb)

        # 2. Temporal self-attention
        temporal_feats = self.temporal(frame_feats)       # (B, T, D_t)

        # 3. Evidence extraction (three parallel nodes)
        motion_feat, app_feat, cons_feat = self.evidence(temporal_feats)
        # Each: (B, D_evidence)

        # 4. Graph reasoning over the three evidence nodes
        nodes    = torch.stack([motion_feat, app_feat, cons_feat], dim=1)  # (B, 3, D_e)
        graph_out, attn_weights = self.graph(nodes)       # (B, 3, D_e), (B, 3, 3)

        # 5. Attention-based pooling across nodes
        pooled, evidence_weights = self.pooling(graph_out)  # (B, D_e), (B, 3)

        # 6. Projection to embedding space
        embedding = self.head(pooled)                     # (B, E)

        return CHEGTOutput(
            embedding=embedding,
            evidence_weights=evidence_weights,
            graph_attention=attn_weights,
            temporal_features=temporal_feats.detach(),
            motion_features=motion_feat.detach(),
            appearance_features=app_feat.detach(),
            consistency_features=cons_feat.detach(),
        )

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: return only the final embedding tensor."""
        return self.forward(x).embedding

    # ── Parameter groups ──────────────────────────────────────────────────────

    def parameter_groups(self, base_lr: float, backbone_lr_mult: float):
        """Return parameter groups with different learning rates.

        The backbone uses a reduced LR to avoid catastrophic forgetting.
        """
        backbone_params  = list(self.backbone.parameters())
        backbone_ids     = {id(p) for p in backbone_params}
        non_backbone     = [p for p in self.parameters() if id(p) not in backbone_ids]

        return [
            {"params": [p for p in backbone_params if p.requires_grad],
             "lr": base_lr * backbone_lr_mult,
             "name": "backbone"},
            {"params": non_backbone,
             "lr": base_lr,
             "name": "non_backbone"},
        ]

    # ── Utility ───────────────────────────────────────────────────────────────

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        n_train = self.num_parameters(trainable_only=True)
        n_total = self.num_parameters(trainable_only=False)
        return (
            f"CHEGT(\n"
            f"  backbone       = {self.backbone.model_name}\n"
            f"  temporal_layers= {self.cfg.temporal.num_layers}\n"
            f"  graph_layers   = {self.cfg.graph.num_layers}\n"
            f"  embed_dim      = {self.cfg.head.embedding_dim}\n"
            f"  trainable_params= {n_train:,}\n"
            f"  total_params    = {n_total:,}\n"
            f")"
        )
