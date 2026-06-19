"""UMAP embedding visualisation.

UMAP generally preserves global cluster structure better than t-SNE and
is more publication-friendly for large datasets.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import umap
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False


def plot_umap(
    embeddings:   np.ndarray,
    labels:       np.ndarray,
    output_path:  str,
    num_classes:  int = 20,
    n_neighbors:  int = 15,
    min_dist:     float = 0.1,
    random_state: int = 42,
    point_size:   float = 10.0,
    alpha:        float = 0.8,
    title:        str = "UMAP of Gait Embeddings",
    dpi:          int = 150,
    label_names:  Optional[List[str]] = None,
) -> None:
    """Compute and plot UMAP projection of gait embeddings.

    Falls back gracefully to a warning if ``umap-learn`` is not installed.

    Args:
        embeddings:   (N, D) float32 embedding matrix.
        labels:       (N,)  integer identity labels.
        output_path:  Path for the output PNG.
        num_classes:  Subsample this many classes for readability.
        n_neighbors:  UMAP n_neighbors parameter.
        min_dist:     UMAP min_dist parameter.
        random_state: Reproducibility seed.
        point_size:   Scatter point size.
        alpha:        Point transparency.
        title:        Figure title.
        dpi:          Output resolution.
        label_names:  Optional human-readable class names.
    """
    if not _UMAP_AVAILABLE:
        print("[WARNING] umap-learn not installed. Skipping UMAP visualisation.")
        print("          Install with: pip install umap-learn")
        return

    unique_classes = np.unique(labels)
    n_total        = len(unique_classes)

    if n_total > num_classes:
        chosen_classes = np.random.default_rng(random_state).choice(
            unique_classes, num_classes, replace=False
        )
    else:
        chosen_classes = unique_classes

    mask = np.isin(labels, chosen_classes)
    emb  = embeddings[mask]
    lbl  = labels[mask]

    # ── UMAP reduction ─────────────────────────────────────────────────
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
        metric="euclidean",
    )
    coords = reducer.fit_transform(emb)            # (M, 2)

    # ── Plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 8))
    cmap    = plt.get_cmap("tab20", len(chosen_classes))

    for idx, cls in enumerate(sorted(chosen_classes)):
        m    = lbl == cls
        name = label_names[cls] if label_names else f"ID {cls:03d}"
        ax.scatter(
            coords[m, 0], coords[m, 1],
            s=point_size,
            color=cmap(idx),
            alpha=alpha,
            label=name,
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("UMAP dim 1", fontsize=10)
    ax.set_ylabel("UMAP dim 2", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if len(chosen_classes) <= 20:
        ax.legend(
            fontsize=7,
            loc="upper right",
            ncol=2,
            markerscale=2.0,
            framealpha=0.6,
        )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_umap_comparison(
    embeddings_a:  np.ndarray,
    labels_a:      np.ndarray,
    name_a:        str,
    embeddings_b:  np.ndarray,
    labels_b:      np.ndarray,
    name_b:        str,
    output_path:   str,
    num_classes:   int = 15,
    dpi:           int = 150,
) -> None:
    """Side-by-side UMAP: baseline vs. CHEGT."""
    if not _UMAP_AVAILABLE:
        print("[WARNING] umap-learn not installed. Skipping UMAP comparison.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    cmap = plt.get_cmap("tab20", num_classes)

    for ax, emb, lbl, name in [
        (axes[0], embeddings_a, labels_a, name_a),
        (axes[1], embeddings_b, labels_b, name_b),
    ]:
        unique_classes = np.unique(lbl)
        if len(unique_classes) > num_classes:
            chosen = np.random.choice(unique_classes, num_classes, replace=False)
        else:
            chosen = unique_classes

        mask    = np.isin(lbl, chosen)
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1,
                            n_components=2, random_state=42)
        coords  = reducer.fit_transform(emb[mask])
        sub_lbl = lbl[mask]

        for idx, cls in enumerate(sorted(chosen)):
            m = sub_lbl == cls
            ax.scatter(coords[m, 0], coords[m, 1],
                       s=8, color=cmap(idx), alpha=0.75)

        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("UMAP dim 1", fontsize=9)
        ax.set_ylabel("UMAP dim 2", fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("UMAP Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
