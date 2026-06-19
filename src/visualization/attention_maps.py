"""Graph attention heatmap visualisation.

Renders the 3 × 3 inter-evidence attention matrix as an annotated
heatmap — the key interpretability figure for the paper.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


_DEFAULT_LABELS = ["Motion", "Appearance", "Consistency"]


def plot_attention_heatmap(
    attn_matrix: np.ndarray,
    output_path: str,
    node_labels: List[str] = _DEFAULT_LABELS,
    title: str = "Evidence Graph Attention",
    cmap: str = "YlOrRd",
    annot: bool = True,
    dpi: int = 150,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """Render the inter-evidence attention weight matrix.

    Args:
        attn_matrix:  (N, N) numpy array of attention weights.
                      If (B, N, N), the batch mean is used.
        output_path:  Output PNG path.
        node_labels:  Labels for the N evidence nodes.
        title:        Figure title.
        cmap:         Seaborn / matplotlib colormap name.
        annot:        Annotate cells with numeric values.
        dpi:          Output resolution.
        vmin, vmax:   Colormap scale limits.
    """
    if attn_matrix.ndim == 3:
        attn_matrix = attn_matrix.mean(axis=0)          # average over batch

    N = attn_matrix.shape[0]
    assert attn_matrix.shape == (N, N), \
        f"Expected (N, N) matrix, got {attn_matrix.shape}"

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        attn_matrix,
        ax=ax,
        annot=annot,
        fmt=".3f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        xticklabels=node_labels[:N],
        yticklabels=node_labels[:N],
        square=True,
        cbar_kws={"label": "Attention Weight", "shrink": 0.8},
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Key (target node)", fontsize=10)
    ax.set_ylabel("Query (source node)", fontsize=10)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_attention_over_epochs(
    attn_matrices: List[np.ndarray],
    epochs: List[int],
    output_path: str,
    node_labels: List[str] = _DEFAULT_LABELS,
    dpi: int = 150,
) -> None:
    """Show how graph attention evolves during training.

    Args:
        attn_matrices: List of (N, N) arrays, one per recorded epoch.
        epochs:        Corresponding epoch numbers.
        output_path:   Output PNG path.
    """
    n_epochs = len(attn_matrices)
    fig, axes = plt.subplots(1, n_epochs, figsize=(4 * n_epochs, 4))
    if n_epochs == 1:
        axes = [axes]

    for ax, mat, ep in zip(axes, attn_matrices, epochs):
        if mat.ndim == 3:
            mat = mat.mean(axis=0)
        sns.heatmap(
            mat, ax=ax, annot=True, fmt=".2f", cmap="YlOrRd",
            vmin=0.0, vmax=1.0,
            xticklabels=node_labels, yticklabels=node_labels,
            linewidths=0.3, linecolor="white", square=True,
            cbar=False,
        )
        ax.set_title(f"Epoch {ep}", fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8, rotation=20)
        ax.tick_params(axis="y", labelsize=8, rotation=0)

    fig.suptitle("Graph Attention Evolution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
