#!/usr/bin/env python3
"""CHEGT Standalone Visualisation Script.

Generate or regenerate any subset of visualisation figures from a saved
checkpoint without re-running evaluation.

Usage
-----
# All figures from best checkpoint
python visualize.py --checkpoint runs/chegt_ccvid/checkpoints/best.pth \
                    --dataset configs/ccvid.yaml

# Only t-SNE and UMAP
python visualize.py --checkpoint runs/.../best.pth \
                    --dataset configs/ccvid.yaml \
                    --figures tsne umap

# All figures with a custom output directory
python visualize.py --checkpoint runs/.../best.pth \
                    --dataset configs/ccvid.yaml \
                    --output_dir custom_vis/

Available figure names: tsne umap attention importance distance all
"""

import argparse
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from src.utils.config     import load_config
from src.utils.seed       import set_seed
from src.utils.logger     import setup_logger
from src.utils.checkpoint import CheckpointManager

from src.models.chegt import CHEGT

from datasets.loaders.ccvid_dataset import CCVIDDataset
from datasets.loaders.fvgb_dataset  import FVGBDataset

from src.visualization.attention_maps      import plot_attention_heatmap
from src.visualization.evidence_importance import plot_evidence_importance
from src.visualization.embedding_tsne      import plot_tsne
from src.visualization.embedding_umap      import plot_umap


AVAILABLE_FIGURES: Set[str] = {"tsne", "umap", "attention", "importance", "distance", "all"}


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
def extract(model: CHEGT, loader: DataLoader, device: torch.device, use_amp: bool):
    model.eval()
    embs, lbls, ews, attns = [], [], [], []
    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            out = model(frames)
        embs.append(out.embedding.cpu().float().numpy())
        lbls.append(batch["label"].numpy())
        ews.append(out.evidence_weights.cpu().float().numpy())
        attns.append(out.graph_attention.cpu().float().numpy())
    return (
        np.concatenate(embs,  axis=0),
        np.concatenate(lbls,  axis=0),
        np.concatenate(ews,   axis=0),
        np.concatenate(attns, axis=0),
    )


def main(args) -> None:
    cfg    = load_config(dataset_cfg=args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    out_dir = Path(args.output_dir) if args.output_dir else \
              Path(args.checkpoint).parent.parent / "standalone_vis"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(str(out_dir), name="chegt_vis")
    logger.info(f"Checkpoint : {args.checkpoint}")
    logger.info(f"Output dir : {out_dir}")
    logger.info(f"Figures    : {args.figures}")

    # ── Model ─────────────────────────────────────────────────────────
    model = CHEGT(cfg).to(device)
    epoch = CheckpointManager.load_model_only(args.checkpoint, model, device=str(device))
    logger.info(f"Loaded epoch {epoch}")

    use_amp = cfg.training.mixed_precision

    # ── Datasets & loaders ────────────────────────────────────────────
    gallery_ds = build_dataset(cfg, split="gallery")
    probe_ds   = build_dataset(cfg, split="probe")

    gallery_loader = build_loader(gallery_ds, cfg)
    probe_loader   = build_loader(probe_ds,   cfg)

    logger.info("Extracting embeddings …")
    g_emb, g_lbl, g_ew, g_attn = extract(model, gallery_loader, device, use_amp)
    p_emb, p_lbl, p_ew, p_attn = extract(model, probe_loader,   device, use_amp)

    all_emb = np.concatenate([g_emb, p_emb], axis=0)
    all_lbl = np.concatenate([g_lbl, p_lbl], axis=0)

    # ── Subsample for projections ──────────────────────────────────────
    n_vis   = min(cfg.visualization.get("num_vis_samples", 512), len(all_emb))
    rng     = np.random.default_rng(42)
    vis_idx = rng.choice(len(all_emb), n_vis, replace=False)
    vis_emb = all_emb[vis_idx]
    vis_lbl = all_lbl[vis_idx]

    dpi         = cfg.visualization.get("dpi", 150)
    node_labels = list(cfg.visualization.attention.node_labels)
    colors      = list(cfg.visualization.evidence_importance.colors)

    want = set(args.figures)
    if "all" in want:
        want = AVAILABLE_FIGURES - {"all"}

    # ── t-SNE ────────────────────────────────────────────────────────
    if "tsne" in want:
        path = str(out_dir / f"tsne_ep{epoch:04d}.png")
        logger.info(f"  → {path}")
        plot_tsne(
            embeddings=vis_emb,
            labels=vis_lbl,
            output_path=path,
            num_classes=cfg.visualization.tsne.num_classes_to_show,
            perplexity=cfg.visualization.tsne.perplexity,
            n_iter=cfg.visualization.tsne.n_iter,
            title=f"t-SNE — Gait Embeddings (ep {epoch})",
            dpi=dpi,
        )

    # ── UMAP ──────────────────────────────────────────────────────────
    if "umap" in want:
        path = str(out_dir / f"umap_ep{epoch:04d}.png")
        logger.info(f"  → {path}")
        plot_umap(
            embeddings=vis_emb,
            labels=vis_lbl,
            output_path=path,
            num_classes=cfg.visualization.umap.num_classes_to_show,
            n_neighbors=cfg.visualization.umap.n_neighbors,
            min_dist=cfg.visualization.umap.min_dist,
            title=f"UMAP — Gait Embeddings (ep {epoch})",
            dpi=dpi,
        )

    # ── Attention heatmap ─────────────────────────────────────────────
    if "attention" in want:
        path = str(out_dir / f"attention_ep{epoch:04d}.png")
        logger.info(f"  → {path}")
        mean_attn = g_attn.mean(axis=0)                # (3, 3)
        plot_attention_heatmap(
            attn_matrix=mean_attn,
            output_path=path,
            node_labels=node_labels,
            title=f"Evidence Graph Attention (ep {epoch})",
            dpi=dpi,
        )

    # ── Evidence importance ───────────────────────────────────────────
    if "importance" in want:
        path = str(out_dir / f"importance_ep{epoch:04d}.png")
        logger.info(f"  → {path}")
        plot_evidence_importance(
            weights=g_ew,                              # (N, 3)
            output_path=path,
            node_labels=node_labels,
            colors=colors,
            title=f"Evidence Importance (ep {epoch})",
            dpi=dpi,
        )

    # ── Distance matrix (subset) ──────────────────────────────────────
    if "distance" in want:
        from src.metrics.retrieval_metrics import compute_distance_matrix
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        n_show = min(30, len(g_emb), len(p_emb))
        dist   = compute_distance_matrix(p_emb[:n_show], g_emb[:n_show])

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            dist, ax=ax, cmap="viridis_r",
            xticklabels=[f"{l}" for l in g_lbl[:n_show]],
            yticklabels=[f"{l}" for l in p_lbl[:n_show]],
            linewidths=0.0,
            cbar_kws={"label": "L2 Distance"},
        )
        ax.set_title(f"Probe × Gallery Distance (subset, ep {epoch})",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Gallery")
        ax.set_ylabel("Probe")
        plt.tight_layout()
        path = str(out_dir / f"distance_ep{epoch:04d}.png")
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  → {path}")

    logger.info("Done.")
    logger.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CHEGT visualisations")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset",    type=str, default="configs/ccvid.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--figures", nargs="+", default=["all"],
        choices=sorted(AVAILABLE_FIGURES),
        help="Which figures to generate. Default: all",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
