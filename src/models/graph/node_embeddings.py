"""Learnable node-type embeddings for the Evidence Graph.

Each of the three evidence node types (motion, appearance, consistency)
receives a distinct learned type embedding, analogous to segment
embeddings in BERT.  This allows the graph transformer to distinguish
node roles without relying solely on content.
"""

import torch
import torch.nn as nn


# Fixed node-type indices — must match the order in which nodes are stacked
# in CHEGT.forward():  0=motion, 1=appearance, 2=consistency
MOTION_IDX      = 0
APPEARANCE_IDX  = 1
CONSISTENCY_IDX = 2


class NodeTypeEmbedding(nn.Module):
    """Additive learnable type embeddings for N node types.

    Args:
        num_nodes: Number of distinct node types (3 for CHEGT).
        d_model:   Embedding / node feature dimensionality.
    """

    def __init__(self, num_nodes: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add type embeddings to node features.

        Args:
            x: (B, N, D) — stacked evidence node features.

        Returns:
            (B, N, D) — features with type embeddings added.
        """
        B, N, D = x.shape
        node_ids = torch.arange(N, device=x.device)        # (N,)
        type_emb = self.embedding(node_ids)                 # (N, D)
        return x + type_emb.unsqueeze(0)                    # (B, N, D)
