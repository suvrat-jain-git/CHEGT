"""Evidence node importance bar-chart visualisation.

Renders the per-node attention weights from EvidencePooling as a
horizontal bar chart — a key figure for communicating what the model
has learned to rely on.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_DEFAULT_LABELS = ["Motion", "Appearance", "Consistency"]
_DEFAULT_COLORS = ["#2196F3", "#4CAF50", "#FF9800"]   # Blue, Green, Orange


def plot_evidence_importance(
    weights: np.ndarray,
    output_path: str,
    node_labels: List[str] = _DEFAULT_LABELS,
    colors: List[str] = _DEFAULT_COLORS,
    title: str = "Evidence Node Importance",
    bar_width: float = 0.5,
    dpi: int = 150,
    with_std: bool = True,
) -> None:
    """Horizontal bar chart of per-node evidence importance.

    Args:
        weights:     (B, N) or (N,) evidence importance weights.
                     Values should be in [0, 1] and sum to ≈ 1 per sample.
        output_path: Output PNG path.
        node_labels: Labels for each node.
        colors:      Bar colours (one per node).
        title:       Figure title.
        bar_width:   Bar height fraction.
        dpi:         Output resolution.
        with_std:    Show standard deviation error bars when weights is (B, N).
    """
    if weights.ndim == 2:
        means = weights.mean(axis=0)           # (N,)
        stds  = weights.std(axis=0)            # (N,)
    else:
        means = weights                        # (N,)
        stds  = np.zeros_like(means)

    N = len(means)
    labels = node_labels[:N]
    cols   = colors[:N]

    # Percentage representation
    pct = means * 100.0
    pct_std = stds * 100.0

    fig, ax = plt.subplots(figsize=(7, 3.5))

    bars = ax.barh(
        y=labels[::-1],
        width=pct[::-1],
        height=bar_width,
        color=cols[::-1],
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )

    if with_std and pct_std.sum() > 0:
        ax.errorbar(
            pct[::-1],
            np.arange(N),
            xerr=pct_std[::-1],
            fmt="none",
            color="black",
            capsize=4,
            linewidth=1.2,
        )

    # Annotate bars with percentage
    for bar, val in zip(bars, pct[::-1]):
        ax.text(
            val + 0.5,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.1f}%",
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Importance (%)", fontsize=11)
    ax.set_xlim(0, 110)
    ax.axvline(100.0 / N, color="grey", linestyle="--", alpha=0.5,
               linewidth=1.0, label="Uniform baseline")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_importance_over_epochs(
    weights_list: List[np.ndarray],
    epochs: List[int],
    output_path: str,
    node_labels: List[str] = _DEFAULT_LABELS,
    colors: List[str] = _DEFAULT_COLORS,
    dpi: int = 150,
) -> None:
    """Line chart: how evidence importance changes over training.

    Args:
        weights_list: List of (N,) mean weight arrays, one per epoch.
        epochs:       Corresponding epoch numbers.
    """
    weights_arr = np.stack(weights_list, axis=0)   # (E, N)
    E, N = weights_arr.shape

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (label, color) in enumerate(zip(node_labels[:N], colors[:N])):
        ax.plot(epochs, weights_arr[:, i] * 100.0,
                label=label, color=color, linewidth=2.0, marker="o", markersize=4)

    ax.set_title("Evidence Importance over Training", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Importance (%)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
