"""Checkpoint saving, loading, and management."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class CheckpointManager:
    """Saves, loads, and rotates model checkpoints.

    Maintains the N most recent checkpoints and always writes a
    ``best.pth`` when a new best metric is achieved.

    Args:
        ckpt_dir:   Directory to store checkpoints.
        keep_last:  How many recent checkpoints to retain.
        metric:     Name of the scalar to track for ``best.pth``
                    (e.g. ``"rank1"``).
        mode:       ``"max"`` (higher is better) or ``"min"``.
    """

    def __init__(
        self,
        ckpt_dir: str,
        keep_last: int = 5,
        metric: str = "rank1",
        mode: str = "max",
    ) -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.metric = metric
        self.mode = mode
        self._best_value: float = float("-inf") if mode == "max" else float("inf")
        self._history: List[Path] = []

    # ── Saving ───────────────────────────────────────────────────────

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        metrics: Dict[str, float],
        cfg_dict: Optional[Dict] = None,
        history: Optional[Dict] = None,           # FIXED CRITICAL-2: persist training history
    ) -> Path:
        """Persist a full training state and rotate old checkpoints.

        Returns:
            Path of the saved checkpoint file.
        """
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "cfg": cfg_dict,
            "history": history,                   # FIXED CRITICAL-2: always include history
        }

        ckpt_path = self.ckpt_dir / f"epoch_{epoch:04d}.pth"
        torch.save(state, ckpt_path)
        self._history.append(ckpt_path)

        # Rotate: delete oldest if over limit
        while len(self._history) > self.keep_last:
            old = self._history.pop(0)
            if old.exists():
                old.unlink()

        # Check for best
        current_val = metrics.get(self.metric, None)
        if current_val is not None and self._is_better(current_val):
            self._best_value = current_val
            best_path = self.ckpt_dir / "best.pth"
            torch.save(state, best_path)
            # Save a small JSON alongside for quick inspection
            with open(self.ckpt_dir / "best_metrics.json", "w") as f:
                json.dump({"epoch": epoch, **metrics}, f, indent=2)

        return ckpt_path

    def _is_better(self, value: float) -> bool:
        if self.mode == "max":
            return value > self._best_value
        return value < self._best_value

    # ── Loading ──────────────────────────────────────────────────────

    @staticmethod
    def load(
        path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu",
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load a checkpoint into model (and optionally optimizer/scheduler).

        Returns:
            The full state dict for inspection (epoch, metrics, etc.).
        """
        state = torch.load(path, map_location=device)
        model.load_state_dict(state["model_state_dict"], strict=strict)

        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])

        if scheduler is not None and state.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(state["scheduler_state_dict"])

        return state

    @staticmethod
    def load_model_only(path: str, model: nn.Module, device: str = "cpu") -> int:
        """Load only model weights; return the saved epoch number."""
        state = torch.load(path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        return state.get("epoch", 0)

    def best_path(self) -> Path:
        return self.ckpt_dir / "best.pth"
