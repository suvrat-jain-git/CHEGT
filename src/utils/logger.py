"""Experiment logger: console + TensorBoard."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Unified logger that writes to console and TensorBoard.

    Usage::

        logger = Logger(log_dir="runs/exp_001/logs", name="chegt")
        logger.info("Training started")
        logger.log_metrics({"loss": 0.5, "rank1": 85.3}, step=100)
        logger.close()
    """

    def __init__(self, log_dir: str, name: str = "chegt") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # ── Console / file handler ──────────────────────────────────
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()

        # stdout
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        self._logger.addHandler(ch)

        # file
        fh = logging.FileHandler(self.log_dir / "train.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        self._logger.addHandler(fh)

        # ── TensorBoard ─────────────────────────────────────────────
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    # ── Logging helpers ──────────────────────────────────────────────

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def debug(self, msg: str) -> None:
        self._logger.debug(msg)

    def warning(self, msg: str) -> None:
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        self._logger.error(msg)

    # ── TensorBoard helpers ──────────────────────────────────────────

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Write a dict of scalar metrics to TensorBoard."""
        for key, val in metrics.items():
            self.writer.add_scalar(key, val, global_step=step)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, global_step=step)

    def log_figure(self, tag: str, figure: Any, step: int) -> None:
        """Log a matplotlib figure to TensorBoard."""
        self.writer.add_figure(tag, figure, global_step=step)

    def log_hparams(self, hparams: Dict[str, Any], metrics: Dict[str, float]) -> None:
        self.writer.add_hparams(hparams, metrics)

    def close(self) -> None:
        self.writer.close()
        logging.shutdown()


def setup_logger(log_dir: str, name: str = "chegt") -> Logger:
    """Convenience constructor."""
    return Logger(log_dir=log_dir, name=name)
