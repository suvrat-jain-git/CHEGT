#!/usr/bin/env python3
"""CHEGT Training Script.

Usage
-----
# CC-VID dataset, default config
python train.py --dataset configs/ccvid.yaml

# FVGB with overrides
python train.py --dataset configs/fvgb.yaml training.epochs=200 optimizer.lr=1e-4

# Resume from checkpoint
python train.py --dataset configs/ccvid.yaml experiment.resume=runs/exp/ckpts/best.pth
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

# ── Project imports ───────────────────────────────────────────────────────────
from src.utils.config     import load_config, save_config, print_config
from src.utils.seed       import set_seed
from src.utils.logger     import setup_logger
from src.utils.checkpoint import CheckpointManager

from src.models.chegt   import CHEGT
from src.losses.combined_loss import CombinedLoss
from src.metrics.retrieval_metrics    import compute_retrieval_metrics
from src.metrics.verification_metrics import compute_verification_metrics

from datasets.loaders.ccvid_dataset import CCVIDDataset
from datasets.loaders.fvgb_dataset  import FVGBDataset
from datasets.loaders.pk_sampler    import PKSampler

from src.visualization.training_curves   import plot_training_curves
from src.visualization.attention_maps    import plot_attention_heatmap
from src.visualization.evidence_importance import plot_evidence_importance, plot_importance_over_epochs
from src.visualization.embedding_tsne   import plot_tsne
from src.visualization.embedding_umap   import plot_umap


# ─────────────────────────────────────────────────────────────────────────────
# Dataset factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(cfg, split: str):
    name = cfg.dataset.name.lower()
    if name == "ccvid":
        return CCVIDDataset(cfg, split=split)
    elif name == "fvgb":
        return FVGBDataset(cfg, split=split)
    else:
        raise ValueError(f"Unknown dataset: {name!r}")


def build_train_loader(cfg, train_ds) -> DataLoader:
    sampler = PKSampler(
        labels=train_ds.labels,
        P=cfg.sampler.P,
        K=cfg.sampler.K,
        num_iters_per_epoch=cfg.training.num_iters_per_epoch,
    )
    batch_size = cfg.sampler.P * cfg.sampler.K
    return DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.training.num_workers > 0,
    )


def build_eval_loader(cfg, ds) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=32,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer & scheduler
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model: CHEGT, cfg) -> torch.optim.Optimizer:
    param_groups = model.parameter_groups(
        base_lr=cfg.optimizer.lr,
        backbone_lr_mult=cfg.optimizer.backbone_lr_multiplier,
    )
    return torch.optim.AdamW(
        param_groups,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        betas=tuple(cfg.optimizer.betas),
        eps=cfg.optimizer.eps,
    )


def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    """Cosine decay with linear warm-up."""
    warmup_steps = cfg.scheduler.warmup_epochs * steps_per_epoch
    total_steps  = cfg.training.epochs * steps_per_epoch
    min_lr       = cfg.scheduler.min_lr
    base_lr      = cfg.optimizer.lr

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warm-up
            return float(step) / max(1, warmup_steps)
        # Cosine decay
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / base_lr, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    model: CHEGT,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
    """Extract embeddings and intermediate outputs for a whole split.

    Returns:
        embeddings:       (N, D)   — L2-normalised embeddings.
        labels:           (N,)     — integer identity labels.
        evidence_weights: list of (N, 3) arrays per batch → stacked (N, 3).
        attn_matrices:    (N, 3, 3) — graph attention matrices.
    """
    model.eval()
    all_emb, all_lbl, all_ew, all_attn = [], [], [], []

    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)   # (B, T, C, H, W)
        labels = batch["label"]

        with autocast(enabled=use_amp):
            out = model(frames)

        all_emb.append(out.embedding.cpu().float().numpy())
        all_lbl.append(labels.numpy())
        all_ew.append(out.evidence_weights.cpu().float().numpy())
        all_attn.append(out.graph_attention.cpu().float().numpy())

    return (
        np.concatenate(all_emb,  axis=0),
        np.concatenate(all_lbl,  axis=0),
        np.concatenate(all_ew,   axis=0),
        np.concatenate(all_attn, axis=0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model: CHEGT,
    gallery_loader: DataLoader,
    probe_loader: DataLoader,
    device: torch.device,
    cfg,
    use_amp: bool = False,
) -> Tuple[Dict, Dict]:
    """Run full evaluation: extract embeddings → compute metrics.

    Returns:
        metrics:         Dict of scalar metric values.
        evidence_weights:(N, 3) array for vis.
        attn_matrices:   (N, 3, 3) array for vis.
    """
    ranks = list(cfg.evaluation.ranks)

    g_emb, g_lbl, g_ew, g_attn = extract_embeddings(model, gallery_loader, device, use_amp)
    p_emb, p_lbl, p_ew, p_attn = extract_embeddings(model, probe_loader,   device, use_amp)

    ret_metrics = compute_retrieval_metrics(
        query_feats=p_emb,
        query_labels=p_lbl,
        gallery_feats=g_emb,
        gallery_labels=g_lbl,
        ranks=ranks,
    )

    ver_metrics = {}
    if cfg.evaluation.compute_eer:
        all_emb = np.concatenate([g_emb, p_emb], axis=0)
        all_lbl = np.concatenate([g_lbl, p_lbl], axis=0)
        ver_metrics = compute_verification_metrics(all_emb, all_lbl)

    metrics = {**ret_metrics, **ver_metrics}
    # FIXED CRITICAL-5: return embeddings dict so generate_visualizations can reuse them
    precomputed = dict(g_emb=g_emb, g_lbl=g_lbl, g_ew=g_ew, g_attn=g_attn,
                       p_emb=p_emb, p_lbl=p_lbl)
    return metrics, precomputed


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_visualizations(
    model: CHEGT,
    gallery_loader: DataLoader,
    probe_loader: DataLoader,
    device: torch.device,
    cfg,
    epoch: int,
    vis_dir: Path,
    use_amp: bool = False,
    history: Optional[Dict] = None,
    history_epochs: Optional[Dict] = None,
    importance_history: Optional[List] = None,
    importance_epochs: Optional[List[int]] = None,
    # FIXED CRITICAL-5: accept pre-computed embeddings to avoid redundant extraction
    precomputed: Optional[Dict] = None,
) -> None:
    """Generate and save all visualisation figures.

    Args:
        precomputed: Optional dict with keys g_emb, g_lbl, g_ew, g_attn, p_emb, p_lbl
                     from a prior evaluate() call. If provided, skips re-extraction.
    """
    vis_dir.mkdir(parents=True, exist_ok=True)

    # FIXED CRITICAL-5: reuse embeddings from evaluate() when available
    if precomputed is not None:
        g_emb  = precomputed["g_emb"]
        g_lbl  = precomputed["g_lbl"]
        g_ew   = precomputed["g_ew"]
        g_attn = precomputed["g_attn"]
        p_emb  = precomputed["p_emb"]
        p_lbl  = precomputed["p_lbl"]
    else:
        g_emb, g_lbl, g_ew, g_attn = extract_embeddings(model, gallery_loader, device, use_amp)
        p_emb, p_lbl, p_ew, p_attn = extract_embeddings(model, probe_loader,   device, use_amp)

    all_emb = np.concatenate([g_emb, p_emb], axis=0)
    all_lbl = np.concatenate([g_lbl, p_lbl], axis=0)

    dpi = cfg.visualization.get("dpi", 150)

    # 1. Training curves
    if history:
        plot_training_curves(
            history=history,
            output_path=str(vis_dir / "training_curves.png"),
            smooth_factor=cfg.visualization.training_curves.smooth_factor,
            dpi=dpi,
            history_epochs=history_epochs,   # FIXED CRITICAL-1
        )

    # 2. Graph attention heatmap (batch mean)
    mean_attn = g_attn.mean(axis=0)                     # (3, 3)
    plot_attention_heatmap(
        attn_matrix=mean_attn,
        output_path=str(vis_dir / f"attention_ep{epoch:04d}.png"),
        node_labels=list(cfg.visualization.attention.node_labels),
        dpi=dpi,
    )

    # 3. Evidence importance bar chart
    plot_evidence_importance(
        weights=g_ew,                                   # (N, 3)
        output_path=str(vis_dir / f"importance_ep{epoch:04d}.png"),
        node_labels=list(cfg.visualization.attention.node_labels),
        colors=list(cfg.visualization.evidence_importance.colors),
        dpi=dpi,
    )

    # 4. Evidence importance over training
    if importance_history and importance_epochs and len(importance_history) > 1:
        plot_importance_over_epochs(
            weights_list=importance_history,
            epochs=importance_epochs,
            output_path=str(vis_dir / "importance_over_epochs.png"),
            node_labels=list(cfg.visualization.attention.node_labels),
            colors=list(cfg.visualization.evidence_importance.colors),
            dpi=dpi,
        )

    # 5. t-SNE — subsample for speed
    n_vis = min(cfg.visualization.get("num_vis_samples", 512), len(all_emb))

    if n_vis < len(all_emb):
        rng     = np.random.default_rng(42)
        idx     = rng.choice(len(all_emb), n_vis, replace=False)
        vis_emb, vis_lbl = all_emb[idx], all_lbl[idx]
    else:
        vis_emb, vis_lbl = all_emb, all_lbl

    plot_tsne(
        embeddings=vis_emb,
        labels=vis_lbl,
        output_path=str(vis_dir / f"tsne_ep{epoch:04d}.png"),
        num_classes=cfg.visualization.tsne.num_classes_to_show,
        perplexity=cfg.visualization.tsne.perplexity,
        n_iter=cfg.visualization.tsne.n_iter,
        random_state=cfg.visualization.tsne.random_state,
        point_size=cfg.visualization.tsne.point_size,
        alpha=cfg.visualization.tsne.alpha,
        dpi=dpi,
    )

    # 6. UMAP
    plot_umap(
        embeddings=vis_emb,
        labels=vis_lbl,
        output_path=str(vis_dir / f"umap_ep{epoch:04d}.png"),
        num_classes=cfg.visualization.umap.num_classes_to_show,
        n_neighbors=cfg.visualization.umap.n_neighbors,
        min_dist=cfg.visualization.umap.min_dist,
        random_state=cfg.visualization.umap.random_state,
        point_size=cfg.visualization.umap.point_size,
        alpha=cfg.visualization.umap.alpha,
        dpi=dpi,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: CHEGT,
    loader: DataLoader,
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    cfg,
    logger,
) -> Dict[str, float]:
    """Train for one epoch. Returns dict of average metrics."""
    model.train()
    total_loss = 0.0
    total_trip = 0.0
    total_frac = 0.0
    n_batches  = 0
    t0         = time.time()

    use_amp = cfg.training.mixed_precision
    grad_clip = cfg.training.gradient_clip

    for batch_idx, batch in enumerate(loader):
        frames = batch["frames"].to(device, non_blocking=True)  # (B, T, C, H, W)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            out  = model(frames)
            loss_out = criterion(out.embedding, labels)

        if use_amp:
            scaler.scale(loss_out.total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_out.total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()

        total_loss += loss_out.total.item()
        total_trip += loss_out.triplet.item()
        total_frac += loss_out.frac_pos.item()
        n_batches  += 1

    elapsed  = time.time() - t0
    avg_loss = total_loss / max(n_batches, 1)
    avg_trip = total_trip / max(n_batches, 1)
    avg_frac = total_frac / max(n_batches, 1)
    cur_lr   = scheduler.get_last_lr()[-1]   # index -1 = non-backbone group

    metrics = {
        "loss":       avg_loss,
        "triplet":    avg_trip,
        "frac_pos":   avg_frac,
        "lr":         cur_lr,
        "epoch_time": elapsed,
    }

    logger.info(
        f"[Epoch {epoch:03d}] "
        f"loss={avg_loss:.4f}  triplet={avg_trip:.4f}  "
        f"frac_pos={avg_frac:.3f}  lr={cur_lr:.2e}  "
        f"time={elapsed:.1f}s"
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args) -> None:
    # ── Config ──────────────────────────────────────────────────────────
    overrides = args.overrides or []
    cfg = load_config(
        dataset_cfg=args.dataset,
        overrides=overrides,
    )
    print_config(cfg)

    # ── Output directories ────────────────────────────────────────────
    exp_name  = cfg.experiment.name
    out_dir   = Path(cfg.experiment.output_dir) / exp_name
    log_dir   = out_dir / "logs"
    ckpt_dir  = out_dir / "checkpoints"
    vis_dir   = out_dir / "vis"
    for d in [log_dir, ckpt_dir, vis_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Logger ────────────────────────────────────────────────────────
    logger = setup_logger(str(log_dir), name=exp_name)
    logger.info(f"Experiment: {exp_name}")
    save_config(cfg, str(out_dir / "config.yaml"))

    # ── Reproducibility ───────────────────────────────────────────────
    set_seed(cfg.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Datasets ──────────────────────────────────────────────────────
    logger.info("Building datasets …")
    train_ds   = build_dataset(cfg, split="train")
    gallery_ds = build_dataset(cfg, split="gallery")
    probe_ds   = build_dataset(cfg, split="probe")

    logger.info(
        f"  train={len(train_ds)} seqs  "
        f"gallery={len(gallery_ds)}  probe={len(probe_ds)}"
    )

    train_loader   = build_train_loader(cfg, train_ds)
    gallery_loader = build_eval_loader(cfg, gallery_ds)
    probe_loader   = build_eval_loader(cfg, probe_ds)

    # ── Model ─────────────────────────────────────────────────────────
    logger.info("Building model …")
    model = CHEGT(cfg).to(device)
    logger.info(model)

    # ── Loss / optimizer / scheduler ──────────────────────────────────
    criterion = CombinedLoss(cfg.loss)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler    = GradScaler(enabled=cfg.training.mixed_precision)

    # ── Checkpoint manager ────────────────────────────────────────────
    ckpt_manager = CheckpointManager(
        ckpt_dir=str(ckpt_dir),
        keep_last=cfg.checkpoint.keep_last,
        metric="rank_1",
        mode="max",
    )

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch = 1
    # FIXED CRITICAL-1: Track epochs per metric to handle different eval intervals
    history: Dict[str, List[float]] = {
        "loss": [], "rank_1": [], "mAP": [], "eer": []
    }
    history_epochs: Dict[str, List[int]] = {
        "loss": [], "rank_1": [], "mAP": [], "eer": []
    }
    importance_history: List[np.ndarray] = []
    importance_epochs:  List[int]        = []

    resume_path = cfg.experiment.get("resume")
    if resume_path and Path(resume_path).exists():
        state = CheckpointManager.load(
            path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=str(device),
        )
        start_epoch = state["epoch"] + 1
        # FIXED CRITICAL-2: restore history from checkpoint
        if "history" in state and state["history"] is not None:
            history = state["history"]
        if "history_epochs" in state and state["history_epochs"] is not None:
            history_epochs = state["history_epochs"]
        # FIXED CRITICAL-3: restore best metric so best.pth is not overwritten by worse model
        best_rank1 = state.get("metrics", {}).get("rank_1", 0.0)
        ckpt_manager._best_value = best_rank1
        logger.info(
            f"Resumed from {resume_path} (epoch {state['epoch']}, best_rank1={best_rank1:.2f}%)"
        )

    # ── Training loop ─────────────────────────────────────────────────
    best_rank1 = 0.0
    eval_interval = cfg.evaluation.interval
    vis_interval  = cfg.visualization.interval

    for epoch in range(start_epoch, cfg.training.epochs + 1):
        saved_this_epoch = False
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scheduler, scaler, device, epoch, cfg, logger,
        )
        history["loss"].append(train_metrics["loss"])
        history_epochs["loss"].append(epoch)
        logger.log_metrics({f"train/{k}": v for k, v in train_metrics.items()}, step=epoch)

        # Evaluate
        if epoch % eval_interval == 0 or epoch == cfg.training.epochs:
            logger.info(f"[Epoch {epoch:03d}] Evaluating …")
            metrics, precomputed = evaluate(   # FIXED CRITICAL-5
                model, gallery_loader, probe_loader,
                device, cfg,
                use_amp=cfg.training.mixed_precision,
            )
            g_ew   = precomputed["g_ew"]
            g_attn = precomputed["g_attn"]

            # Update history (FIXED CRITICAL-1: track epoch per metric)
            history["rank_1"].append(metrics.get("rank_1", 0.0))
            history["mAP"].append(metrics.get("mAP", 0.0))
            history["eer"].append(metrics.get("eer", 0.0))
            history_epochs["rank_1"].append(epoch)
            history_epochs["mAP"].append(epoch)
            history_epochs["eer"].append(epoch)

            mean_ew = g_ew.mean(axis=0)
            importance_history.append(mean_ew)
            importance_epochs.append(epoch)

            # Log
            logger.log_metrics({f"eval/{k}": v for k, v in metrics.items()}, step=epoch)
            rank1 = metrics.get("rank_1", 0.0)
            mAP   = metrics.get("mAP", 0.0)
            eer   = metrics.get("eer", float("nan"))
            logger.info(
                f"  Rank-1={rank1:.2f}%  mAP={mAP:.2f}%  EER={eer:.2f}%"
            )

            # Checkpoint (FIXED CRITICAL-2: save history; FIXED MINOR-2: set flag)
            saved_this_epoch = True
            ckpt_manager.save(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                cfg_dict=OmegaConf.to_container(cfg, resolve=True),
                history={"values": history, "epochs": history_epochs},
            )

            if rank1 > best_rank1:
                best_rank1 = rank1
                logger.info(f"  *** New best Rank-1: {best_rank1:.2f}% ***")

        # Visualise
        if epoch % vis_interval == 0 or epoch == cfg.training.epochs:
            logger.info(f"[Epoch {epoch:03d}] Generating visualisations …")
            generate_visualizations(
                model=model,
                gallery_loader=gallery_loader,
                probe_loader=probe_loader,
                device=device,
                cfg=cfg,
                epoch=epoch,
                vis_dir=vis_dir,
                use_amp=cfg.training.mixed_precision,
                history=history,
                history_epochs=history_epochs,
                importance_history=importance_history,
                importance_epochs=importance_epochs,
                precomputed=precomputed if saved_this_epoch else None,  # CRITICAL-5
            )

        # Checkpoint on interval — skip if already saved this epoch (FIXED MINOR-2)
        if epoch % cfg.checkpoint.save_interval == 0 and not saved_this_epoch:
            ckpt_manager.save(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={"rank_1": best_rank1},
                cfg_dict=OmegaConf.to_container(cfg, resolve=True),
                history={"values": history, "epochs": history_epochs},
            )

    logger.info(f"Training complete. Best Rank-1: {best_rank1:.2f}%")
    logger.close()


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train CHEGT")
    parser.add_argument(
        "--dataset", type=str, default="configs/ccvid.yaml",
        help="Dataset config YAML (e.g. configs/ccvid.yaml)"
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="OmegaConf dotlist overrides, e.g. training.epochs=200"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
