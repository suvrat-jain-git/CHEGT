"""Retrieval metrics: CMC (Rank-k) and mAP.

All computations are performed in NumPy for clarity and compatibility
with scikit-learn. Embeddings are expected as (N, D) float32 arrays.
"""

from typing import Dict, List, Optional

import numpy as np


def compute_distance_matrix(
    query: np.ndarray,
    gallery: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """Compute pairwise distance matrix between query and gallery sets.

    Args:
        query:   (Q, D) query embeddings.
        gallery: (G, D) gallery embeddings.
        metric:  ``"euclidean"`` or ``"cosine"``.

    Returns:
        dist_mat: (Q, G) distance matrix.
    """
    if metric == "cosine":
        # For L2-normalised embeddings: cosine distance = 1 - dot product
        query   = query   / (np.linalg.norm(query,   axis=1, keepdims=True) + 1e-12)
        gallery = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-12)
        return 1.0 - query @ gallery.T

    # Euclidean: ||q - g||^2 = ||q||^2 + ||g||^2 - 2*q@g.T
    q_sq = (query   ** 2).sum(axis=1, keepdims=True)
    g_sq = (gallery ** 2).sum(axis=1, keepdims=True)
    dist = q_sq + g_sq.T - 2.0 * (query @ gallery.T)
    return np.sqrt(np.clip(dist, a_min=0.0, a_max=None))


def compute_cmc_map(
    query_feats:   np.ndarray,
    query_labels:  np.ndarray,
    gallery_feats: np.ndarray,
    gallery_labels: np.ndarray,
    ranks:         List[int] = [1, 5, 10, 20],
    metric:        str = "euclidean",
    query_seqids:  Optional[np.ndarray] = None,
    gallery_seqids: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute CMC curve and mean Average Precision.

    Supports single-gallery-shot (one true match per query) and
    multi-gallery-shot (multiple correct matches per query).

    Args:
        query_feats:    (Q, D)
        query_labels:   (Q,)   integer identity labels
        gallery_feats:  (G, D)
        gallery_labels: (G,)   integer identity labels
        ranks:          List of k values for Rank-k accuracy.
        metric:         Distance metric.
        query_seqids:   Optional (Q,) sequence IDs to exclude identical
                        sequences from gallery (junk removal).
        gallery_seqids: Optional (G,) sequence IDs.

    Returns:
        Dict with keys ``rank_1``, ``rank_5``, …, ``mAP``.
    """
    dist_mat = compute_distance_matrix(query_feats, gallery_feats, metric)  # (Q, G)
    Q, G = dist_mat.shape

    all_ap:     List[float] = []
    rank_hits = {k: 0 for k in ranks}

    for q_idx in range(Q):
        q_label  = query_labels[q_idx]
        q_seqid  = query_seqids[q_idx] if query_seqids is not None else None

        dists = dist_mat[q_idx]                          # (G,)
        order = np.argsort(dists)

        # Build mask: good = same identity; junk = same sequence (if provided)
        gl = gallery_labels[order]
        good_mask = gl == q_label

        if q_seqid is not None and gallery_seqids is not None:
            gs        = gallery_seqids[order]
            junk_mask = (gs == q_seqid)
            good_mask = good_mask & (~junk_mask)         # exclude same sequence

        # Skip queries with no true match in gallery
        if good_mask.sum() == 0:
            continue

        # ── CMC ──────────────────────────────────────────────────────
        cmc = good_mask.cumsum()
        for k in ranks:
            if cmc[:k].max() >= 1:
                rank_hits[k] += 1

        # ── AP ───────────────────────────────────────────────────────
        n_rel = good_mask.sum()
        hit_positions = np.where(good_mask)[0] + 1.0   # 1-indexed
        precisions    = np.arange(1, n_rel + 1) / hit_positions
        ap            = precisions.mean()
        all_ap.append(float(ap))

    n_queries = len(all_ap) if all_ap else Q

    metrics: Dict[str, float] = {}
    for k in ranks:
        metrics[f"rank_{k}"] = 100.0 * rank_hits[k] / n_queries
    metrics["mAP"] = 100.0 * float(np.mean(all_ap)) if all_ap else 0.0

    return metrics


def compute_retrieval_metrics(
    query_feats:    np.ndarray,
    query_labels:   np.ndarray,
    gallery_feats:  np.ndarray,
    gallery_labels: np.ndarray,
    ranks:          List[int] = [1, 5, 10, 20],
    metric:         str = "euclidean",
) -> Dict[str, float]:
    """Convenience wrapper around compute_cmc_map."""
    return compute_cmc_map(
        query_feats, query_labels,
        gallery_feats, gallery_labels,
        ranks=ranks,
        metric=metric,
    )
