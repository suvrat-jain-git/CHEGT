"""Evidence Graph Transformer.

Stacks multiple GraphAttentionLayer blocks and applies learnable
node-type embeddings so the transformer can distinguish Motion,
Appearance, and Consistency nodes.

Returns both the refined node features and the final layer's attention
weights (B, N, N) for the 3 × 3 inter-evidence heatmap.
"""

from typing import Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.graph.graph_attention  import GraphAttentionLayer
from src.models.graph.node_embeddings  import NodeTypeEmbedding


class GraphTransformer(nn.Module):
    """Multi-layer graph attention over evidence nodes.

    Args:
        cfg: Graph config with:
            - ``num_nodes``      int   Number of evidence nodes (3).
            - ``d_model``        int   Node feature dim.
            - ``nhead``          int   Attention heads per layer.
            - ``num_layers``     int   Number of attention layers.
            - ``dim_feedforward``int   FFN hidden dim.
            - ``dropout``        float Dropout.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        d_model         = cfg.d_model
        nhead           = cfg.nhead
        num_layers      = cfg.num_layers
        dim_feedforward = cfg.dim_feedforward
        dropout         = cfg.dropout

        # Learnable node-type embeddings
        self.node_type_emb = NodeTypeEmbedding(cfg.num_nodes, d_model)

        # Stacked attention layers
        self.layers = nn.ModuleList([
            GraphAttentionLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, D) — N stacked evidence node features.

        Returns:
            out          : (B, N, D) — refined node features.
            attn_weights : (B, N, N) — last-layer head-averaged attention.
        """
        # Add learnable type identity
        x = self.node_type_emb(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        # Retrieve last-layer attention weights for visualisation
        attn_weights = self.layers[-1].attention_weights  # (B, N, N)
        if attn_weights is None:
            B, N, _ = x.shape
            attn_weights = torch.zeros(B, N, N, device=x.device)

        return x, attn_weights
