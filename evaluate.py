#!/usr/bin/env python3
"""CHEGT Evaluation Script.

Loads a trained checkpoint, extracts embeddings for gallery and probe
splits, computes full retrieval and verification metrics, and auto-generates
all visualisation figures.

Usage
-----
# Evaluate best checkpoint on CC-VID
python evaluate.py --checkpoint runs/chegt_ccvid/checkpoints/best.pth \
                   --dataset configs/ccvid.yaml

# Override split or use a specific epoch checkpoint
python evaluate.py --checkpoint runs/chegt_ccvid/checkpoints/epoch_0100.pth \
                   --dataset configs/fvgb.yaml \
                   --output_dir runs/eval_fvgb
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from src.utils.config     import load_config, print_config
from src.utils.seed       import set_seed
from src.utils.logger     import setup_logger
from src.utils.checkpoint import CheckpointManager

from src.models.chegt import CHEGT
from src.metrics.retrieval_metrics    import compute_retrieval_metrics, compute_distance_matrix
from src.metrics.verification_metrics import compute_verification_metrics

from datasets.loaders.ccvid_dataset import CCVIDDataset
from datasets.loaders.fvgb_dataset  import FVGBDataset

from src.visualization.attention_maps     import plot_attention_heatmap
from src.visualization.evidence_importance import plot_evidence_importance
from src.visualization.embedding_tsne     import plot_tsne
from src.visualization.embedding_umap     import plot_umap


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(cfg, split: str):
    name = cfg.dataset.name.lower()
    if name == "ccvid":
        return CCVIDDataset(cfg, split=split)
    elif name == "fvgb":
        return FVGBDataset(cfg, split=split)
    else:
        raise ValueError(f"Unknown dataset: {name!r}")


def build_loader(ds, cfg) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=32,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=False,
    )


@torch.no_grad()
def extract_embeddings(
    model: CHEGT,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> Dict[str, np.ndarray]:
    """Extract embeddings and auxiliary outputs for a full split.

    Returns:
        Dict with keys:
            embeddings        (N, D)
            labels            (N,)
            evidence_weights  (N, 3)
            graph_attention   (N, 3, 3)
            subject_ids       list[str]
            seq_ids           list[str]
    """
    model.eval()
    embs, lbls, ews, attns = [], [], [], []
    subj_ids, seq_ids = [], []

    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            out = model(frames)

        embs.append(out.embedding.cpu().float().numpy())
        lbls.append(batch["label"].numpy())
        ews.append(out.evidence_weights.cpu().float().numpy())
        attns.append(out.graph_attention.cpu().float().numpy())
        subj_ids.extend(batch.get("subject_id", [""] * len(batch["label"])))
        seq_ids.extend(batch.get("seq_id", [""] * len(batch["label"])))

    return {
        "embeddings":       np.concatenate(embs,  axis=0),
        "labels":           np.concatenate(lbls,  axis=0),
        "evidence_weights": np.concatenate(ews,   axis=0),
        "graph_attention":  np.concatenate(attns, axis=0),
        "subject_ids":      subj_ids,
        "seq_ids":          seq_ids,
    }


def print_metrics(metrics: Dict[str, float], logger) -> None:
    logger.info("=" * 55)
    logger.info("  Retrieval Metrics")
    logger.info("=" * 55)
    rank_keys = sorted([k for k in metrics if k.startswith("rank_")],
                       key=lambda x: int(x.split("_")[1]))
    for k in rank_keys:
        logger.info(f"  {k.replace('rank_', 'Rank-'):10s}  {metrics[k]:.2f}%")
    if "mAP" in metrics:
        logger.info(f"  {'mAP':10s}  {metrics['mAP']:.2f}%")
    logger.info("-" * 55)
    logger.info("  Verification Metrics")
    logger.info("-" * 55)
    if "eer" in metrics:
        logger.info(f"  {'EER':10s}  {metrics['eer']:.2f}%")
    if "auc" in metrics:
        logger.info(f"  {'AUC-ROC':10s}  {metrics['auc']:.2f}%")
    logger.info("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args) -> None:
    # ── Config ──────────────────────────────────────────────────────────
    cfg = load_config(dataset_cfg=args.dataset)

    # ── Output directory ─────────────────────────────────────────────
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ckpt_path = Path(args.checkpoint)
        out_dir   = ckpt_path.parent.parent / "eval"
    vis_dir   = out_dir / "vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # ── Logger ────────────────────────────────────────────────────────
    logger = setup_logger(str(out_dir), name="chegt_eval")
    logger.info(f"Checkpoint : {args.checkpoint}")
    logger.info(f"Dataset cfg: {args.dataset}")
    logger.info(f"Output dir : {out_dir}")

    # ── Device ────────────────────────────────────────────────────────
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Model ─────────────────────────────────────────────────────────
    model = CHEGT(cfg).to(device)
    epoch = CheckpointManager.load_model_only(args.checkpoint, model, device=str(device))
    logger.info(f"Loaded model from epoch {epoch}.")
    logger.info(model)

    use_amp = cfg.training.mixed_precision

    # ── Datasets ──────────────────────────────────────────────────────
    logger.info("Loading datasets …")
    gallery_ds = build_dataset(cfg, split="gallery")
    probe_ds   = build_dataset(cfg, split="probe")
    logger.info(f"  gallery={len(gallery_ds)}  probe={len(probe_ds)}")

    gallery_loader = build_loader(gallery_ds, cfg)
    probe_loader   = build_loader(probe_ds,   cfg)

    # ── Extract embeddings ────────────────────────────────────────────
    logger.info("Extracting gallery embeddings …")
    gallery = extract_embeddings(model, gallery_loader, device, use_amp)
    logger.info("Extracting probe embeddings …")
    probe   = extract_embeddings(model, probe_loader,   device, use_amp)

    # ── Metrics ───────────────────────────────────────────────────────
    logger.info("Computing retrieval metrics …")
    ret_metrics = compute_retrieval_metrics(
        query_feats=probe["embeddings"],
        query_labels=probe["labels"],
        gallery_feats=gallery["embeddings"],
        gallery_labels=gallery["labels"],
        ranks=list(cfg.evaluation.ranks),
    )

    logger.info("Computing verification metrics …")
    all_emb = np.concatenate([gallery["embeddings"], probe["embeddings"]], axis=0)
    all_lbl = np.concatenate([gallery["labels"],     probe["labels"]],     axis=0)
    ver_metrics = compute_verification_metrics(all_emb, all_lbl)

    metrics = {**ret_metrics, **ver_metrics}
    print_metrics(metrics, logger)

    # Save metrics as JSON
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"epoch": epoch, **metrics}, f, indent=2)
    logger.info(f"Metrics saved → {out_dir / 'metrics.json'}")

    # ── Visualisations ────────────────────────────────────────────────
    logger.info("Generating visualisations …")
    dpi = cfg.visualization.get("dpi", 150)
    node_labels = list(cfg.visualization.attention.node_labels)
    colors      = list(cfg.visualization.evidence_importance.colors)

    # Graph attention heatmap
    mean_attn = gallery["graph_attention"].mean(axis=0)    # (3, 3)
    plot_attention_heatmap(
        attn_matrix=mean_attn,
        output_path=str(vis_dir / "graph_attention.png"),
        node_labels=node_labels,
        title=f"Evidence Graph Attention (epoch {epoch})",
        dpi=dpi,
    )

    # Evidence importance
    plot_evidence_importance(
        weights=gallery["evidence_weights"],               # (N, 3)
        output_path=str(vis_dir / "evidence_importance.png"),
        node_labels=node_labels,
        colors=colors,
        title=f"Evidence Importance (epoch {epoch})",
        dpi=dpi,
    )

    # t-SNE
    n_vis   = min(cfg.visualization.get("num_vis_samples", 512), len(all_emb))
    vis_idx = np.random.choice(len(all_emb), n_vis, replace=False)
    plot_tsne(
        embeddings=all_emb[vis_idx],
        labels=all_lbl[vis_idx],
        output_path=str(vis_dir / "tsne.png"),
        num_classes=cfg.visualization.tsne.num_classes_to_show,
        perplexity=cfg.visualization.tsne.perplexity,
        n_iter=cfg.visualization.tsne.n_iter,
        title=f"t-SNE — Gait Embeddings (epoch {epoch})",
        dpi=dpi,
    )

    # UMAP
    plot_umap(
        embeddings=all_emb[vis_idx],
        labels=all_lbl[vis_idx],
        output_path=str(vis_dir / "umap.png"),
        num_classes=cfg.visualization.umap.num_classes_to_show,
        n_neighbors=cfg.visualization.umap.n_neighbors,
        min_dist=cfg.visualization.umap.min_dist,
        title=f"UMAP — Gait Embeddings (epoch {epoch})",
        dpi=dpi,
    )

    # Distance matrix heatmap (first 30 gallery vs probe for sanity check)
    _save_distance_heatmap(
        gallery["embeddings"][:30],
        gallery["labels"][:30],
        probe["embeddings"][:30],
        probe["labels"][:30],
        output_path=str(vis_dir / "distance_matrix.png"),
        dpi=dpi,
    )

    logger.info(f"All visualisations saved → {vis_dir}")
    logger.close()


def _save_distance_heatmap(
    g_emb: np.ndarray,
    g_lbl: np.ndarray,
    p_emb: np.ndarray,
    p_lbl: np.ndarray,
    output_path: str,
    dpi: int = 150,
) -> None:
    """Save a gallery × probe distance heatmap for sanity-checking."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    dist = compute_distance_matrix(p_emb, g_emb)   # (Q, G)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        dist, ax=ax, cmap="viridis_r",
        xticklabels=[f"{l}" for l in g_lbl],
        yticklabels=[f"{l}" for l in p_lbl],
        linewidths=0.0,
        cbar_kws={"label": "L2 Distance"},
    )
    ax.set_title("Probe × Gallery Distance Matrix (subset)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Gallery", fontsize=10)
    ax.set_ylabel("Probe",   fontsize=10)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a CHEGT checkpoint")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to .pth checkpoint file"
    )
    parser.add_argument(
        "--dataset", type=str, default="configs/ccvid.yaml",
        help="Dataset config YAML"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to save metrics and visualisations (default: next to checkpoint)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
