"""CC-VID Dataset Loader.

Expected on-disk layout::

    datasets/ccvid/
        001/                  # identity (zero-padded string)
            nm-01/            # sequence  (type-index)
                00001.jpg
                00002.jpg
                …
            cl-01/
                …
        002/
            …

Sequence naming convention: ``<type>-<index>``
    - ``nm`` = normal walking
    - ``cl`` = clothes change
    - ``bg`` = carrying a bag
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.loaders.transforms import GaitTransform, get_transforms
from omegaconf import DictConfig


class CCVIDDataset(Dataset):
    """PyTorch Dataset for CC-VID.

    Each ``__getitem__`` returns a fixed-length clip of T frames uniformly
    sampled from a single pedestrian sequence.

    Args:
        cfg:      Full experiment config (uses ``cfg.dataset``).
        split:    ``"train"`` | ``"gallery"`` | ``"probe"``.
    """

    def __init__(self, cfg: DictConfig, split: str = "train") -> None:
        super().__init__()
        self.cfg = cfg
        self.split = split
        ds_cfg = cfg.dataset
        split_cfg = getattr(ds_cfg, split)

        self.root = Path(ds_cfg.root)
        self.num_frames: int = split_cfg.num_frames
        self.frame_stride: int = split_cfg.get("frame_stride", 1)
        self.transform: GaitTransform = get_transforms(ds_cfg, split=("train" if split == "train" else "val"))

        # Sequence type filter
        allowed_types: Optional[List[str]] = list(split_cfg.seq_types) \
            if split_cfg.get("seq_types") else None
        allowed_seqs: Optional[List[str]] = list(split_cfg.seq_ids) \
            if split_cfg.get("seq_ids") else None
        allowed_subjects: Optional[List[str]] = [str(s).zfill(3) for s in split_cfg.subjects] \
            if split_cfg.get("subjects") else None

        # ── Discover sequences ────────────────────────────────────────
        self.samples: List[Dict[str, Any]] = []
        self._label_to_int: Dict[str, int] = {}

        subject_dirs = sorted(self.root.iterdir())
        for subj_dir in subject_dirs:
            if not subj_dir.is_dir():
                continue
            subj_id = subj_dir.name
            if allowed_subjects and subj_id not in allowed_subjects:
                continue

            if subj_id not in self._label_to_int:
                self._label_to_int[subj_id] = len(self._label_to_int)
            label_int = self._label_to_int[subj_id]

            for seq_dir in sorted(subj_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                seq_id = seq_dir.name
                seq_type = seq_id.split("-")[0] if "-" in seq_id else seq_id

                if allowed_types and seq_type not in allowed_types:
                    continue
                if allowed_seqs and seq_id not in allowed_seqs:
                    continue

                frame_paths = sorted(
                    p for p in seq_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                )

                if len(frame_paths) < 2:
                    continue

                self.samples.append({
                    "frame_paths": frame_paths,
                    "label":       label_int,
                    "subject_id":  subj_id,
                    "seq_id":      seq_id,
                    "seq_type":    seq_type,
                })

        if not self.samples:
            raise RuntimeError(
                f"No sequences found in {self.root} for split='{split}'. "
                "Check your config paths and seq_types."
            )

    # ── Dataset interface ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        frames_tensor = self._load_frames(sample["frame_paths"])
        return {
            "frames":     frames_tensor,        # (T, C, H, W)
            "label":      sample["label"],
            "subject_id": sample["subject_id"],
            "seq_id":     sample["seq_id"],
            "seq_type":   sample["seq_type"],
        }

    # ── Helpers ───────────────────────────────────────────────────────

    def _load_frames(self, frame_paths: List[Path]) -> torch.Tensor:
        """Uniformly sample T frames from a sequence and apply transforms."""
        # Apply stride before uniform sampling
        strided = frame_paths[:: self.frame_stride]

        T = self.num_frames
        n = len(strided)

        if n >= T:
            # Uniform sampling: evenly-spaced indices
            step = n / T
            indices = [int(i * step) for i in range(T)]
        else:
            # Repeat frames cyclically to reach T
            indices = [i % n for i in range(T)]

        frames: List[Image.Image] = []
        for i in indices:
            img = Image.open(strided[i]).convert("RGB")
            frames.append(img)

        return self.transform(frames)  # (T, C, H, W)

    # ── Metadata helpers (used by PKSampler) ─────────────────────────

    def get_label(self, idx: int) -> int:
        return self.samples[idx]["label"]

    @property
    def labels(self) -> List[int]:
        return [s["label"] for s in self.samples]

    @property
    def num_classes(self) -> int:
        return len(self._label_to_int)

    def get_meta(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        return {"subject_id": s["subject_id"], "seq_id": s["seq_id"], "seq_type": s["seq_type"]}
