"""Training curve visualisation.

Reads scalar metrics logged during training and plots them with optional
exponential-moving-average smoothing.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless server-safe backend
import matplotlib.pyplot as plt
import numpy as np


def _ema_smooth(values: List[float], alpha: float = 0.6) -> List[float]:
    """Exponential moving average smoothing."""
    if not values:
        return values
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * smoothed[-1] + (1.0 - alpha) * v)
    return smoothed


def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: str,
    smooth_factor: float = 0.6,
    figsize=(14, 10),
    dpi: int = 150,
    history_epochs: Optional[Dict[str, List[int]]] = None,
) -> None:
    """Generate and save a 2×2 training dashboard.

    Expected keys in ``history``:
        - ``loss``    training loss per epoch
        - ``rank_1``  Rank-1 accuracy
        - ``mAP``     mean Average Precision
        - ``eer``     Equal Error Rate (optional)

    Args:
        history:        Dict mapping metric name → list of values.
        output_path:    Path to save the PNG figure.
        smooth_factor:  EMA smoothing factor (0 = no smoothing, 1 = maximal).
        figsize:        Matplotlib figure size.
        dpi:            Output resolution.
        history_epochs: Dict mapping metric name → list of epoch numbers.
                        FIXED CRITICAL-1: metrics have different sampling rates,
                        so each has its own epoch axis.
    """
    # Build per-metric epoch axis (FIXED CRITICAL-1)
    def _epochs_for(key, vals):
        if history_epochs and key in history_epochs and history_epochs[key]:
            return history_epochs[key]
        return list(range(1, len(vals) + 1))

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("CHEGT Training Curves", fontsize=16, fontweight="bold", y=1.01)

    panels = [
        ("loss",  "Training Loss",    "Loss",        axes[0, 0], "tab:blue",   False),
        ("rank1", "Rank-1 Accuracy",  "Rank-1 (%)", axes[0, 1], "tab:green",  True),
        ("mAP",   "Mean Avg. Precision", "mAP (%)", axes[1, 0], "tab:orange", True),
        ("eer",   "Equal Error Rate", "EER (%)",    axes[1, 1], "tab:red",    False),
    ]

    for key, title, ylabel, ax, color, higher_is_better in panels:
        if key not in history or not history[key]:
            ax.set_visible(False)
            continue

        raw      = history[key]
        ep_axis  = _epochs_for(key, raw)           # FIXED CRITICAL-1: per-metric epochs
        smoothed = _ema_smooth(raw, alpha=smooth_factor)

        ax.plot(ep_axis, raw, color=color, alpha=0.25, linewidth=1.0, label="raw")
        ax.plot(ep_axis, smoothed, color=color, linewidth=2.0, label="smoothed (EMA)")

        # Mark best epoch
        best_fn  = max if higher_is_better else min
        best_val = best_fn(raw)
        best_idx = raw.index(best_val)
        best_ep  = ep_axis[best_idx]
        ax.axvline(best_ep, color=color, linestyle="--", alpha=0.5, linewidth=1.0)
        ax.scatter([best_ep], [best_val], color=color, s=60, zorder=5,
                   label=f"best @ ep{best_ep}: {best_val:.2f}")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(
    histories: Dict[str, Dict[str, List[float]]],
    metric: str,
    output_path: str,
    ylabel: str = "",
    title: str = "",
    dpi: int = 150,
) -> None:
    """Overlay multiple runs for ablation comparison.

    Args:
        histories:   Dict mapping run_name → metric_history dict.
        metric:      Key to plot (e.g. ``"rank1"``).
        output_path: Output PNG path.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10.colors

    for idx, (run_name, hist) in enumerate(histories.items()):
        if metric not in hist:
            continue
        vals   = hist[metric]
        epochs = list(range(1, len(vals) + 1))
        color  = colors[idx % len(colors)]
        ax.plot(epochs, vals, color=color, linewidth=2.0, label=run_name)

    ax.set_title(title or f"Ablation: {metric}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(ylabel or metric, fontsize=11)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
