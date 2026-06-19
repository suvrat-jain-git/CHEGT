"""Verification metrics: Equal Error Rate (EER) and ROC-AUC.

Gait verification asks: "Are these two sequences from the same person?"
EER is the threshold where FAR == FRR — lower is better.
"""

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_eer(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
) -> Tuple[float, float]:
    """Compute Equal Error Rate.

    Args:
        genuine_scores:  Similarity scores for genuine (same-person) pairs.
        impostor_scores: Similarity scores for impostor (diff-person) pairs.

    Returns:
        eer:       Equal Error Rate (as a fraction, multiply by 100 for %).
        threshold: Decision threshold at EER.
    """
    n_genuine  = len(genuine_scores)
    n_impostor = len(impostor_scores)

    labels = np.concatenate([
        np.ones(n_genuine),
        np.zeros(n_impostor),
    ])
    scores = np.concatenate([genuine_scores, impostor_scores])

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    # Find the index where |FPR - FNR| is minimised
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])

    return eer, threshold


def compute_verification_metrics(
    embeddings:     np.ndarray,
    labels:         np.ndarray,
    distance_metric: str = "euclidean",
    max_pairs:       int = 100_000,
) -> Dict[str, float]:
    """Compute EER and ROC-AUC from an embedding matrix.

    Constructs genuine and impostor pairs, converts distances to
    similarity scores, then computes EER.

    Args:
        embeddings:      (N, D) embeddings.
        labels:          (N,)  integer identity labels.
        distance_metric: ``"euclidean"`` or ``"cosine"``.
        max_pairs:       Cap on number of pairs to keep evaluation tractable.

    Returns:
        Dict with keys ``eer``, ``eer_threshold``, ``auc``.
    """
    N = len(embeddings)

    # Build all pair indices (upper triangle)
    i_idx, j_idx = np.triu_indices(N, k=1)

    # Subsample if too many pairs
    n_pairs = len(i_idx)
    if n_pairs > max_pairs:
        chosen  = np.random.choice(n_pairs, max_pairs, replace=False)
        i_idx, j_idx = i_idx[chosen], j_idx[chosen]

    # Compute distances
    if distance_metric == "euclidean":
        diff = embeddings[i_idx] - embeddings[j_idx]
        dists = np.linalg.norm(diff, axis=1)
    else:
        e_i = embeddings[i_idx]
        e_j = embeddings[j_idx]
        e_i = e_i / (np.linalg.norm(e_i, axis=1, keepdims=True) + 1e-12)
        e_j = e_j / (np.linalg.norm(e_j, axis=1, keepdims=True) + 1e-12)
        dists = 1.0 - (e_i * e_j).sum(axis=1)

    # Convert distance → similarity (larger = more similar)
    max_d  = dists.max() + 1e-8
    sims   = 1.0 - (dists / max_d)

    # Labels: 1 = genuine pair, 0 = impostor
    same   = (labels[i_idx] == labels[j_idx]).astype(int)

    genuine_scores  = sims[same == 1]
    impostor_scores = sims[same == 0]

    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return {"eer": float("nan"), "eer_threshold": float("nan"), "auc": float("nan")}

    eer, threshold = compute_eer(genuine_scores, impostor_scores)

    # ROC-AUC
    auc = roc_auc_score(same, sims)

    return {
        "eer":           100.0 * eer,
        "eer_threshold": threshold,
        "auc":           100.0 * auc,
    }
