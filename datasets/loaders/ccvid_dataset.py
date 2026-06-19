"""CC-VID Dataset Loader — txt-file based splits.

Uses the official train.txt / gallery.txt / query.txt to define splits
exactly as published. Sessions are mixed across splits so filtering by
session type alone is insufficient; the txt files are authoritative.

Symlinked layout expected at cfg.dataset.root:
    data/ccvid/
        001/
            session1_01/   00001.jpg  00002.jpg ...
            session3_01/   ...
        002/
            ...
"""

from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from omegaconf import DictConfig

from datasets.loaders.transforms import GaitTransform, get_transforms


class CCVIDDataset(Dataset):
    def __init__(self, cfg: DictConfig, split: str = "train") -> None:
        super().__init__()
        self.split  = split
        ds_cfg      = cfg.dataset

        self.root       = Path(ds_cfg.root)
        self.num_frames = ds_cfg.get("num_frames", 16)
        self.transform  = get_transforms(
            ds_cfg, split="train" if split == "train" else "val"
        )

        txt_key = {"train": "train_txt", "gallery": "gallery_txt", "probe": "query_txt"}[split]
        txt_path = Path(ds_cfg[txt_key])
        if not txt_path.exists():
            raise FileNotFoundError(f"Split txt not found: {txt_path}")

        self.samples: List[Dict[str, Any]] = []
        self._label_to_int: Dict[str, int] = {}

        with open(txt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts      = line.split()
                video_path = parts[0]
                identity   = parts[1]
                clothes    = parts[2] if len(parts) > 2 else ""

                session, seq_folder = video_path.split("/")
                seq_num = seq_folder.split("_")[1]
                seq_id  = f"{session}_{seq_num}"

                seq_dir = self.root / identity / seq_id
                if not seq_dir.exists():
                    continue

                frame_paths = sorted(
                    p for p in seq_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                )
                if len(frame_paths) < 2:
                    continue

                if identity not in self._label_to_int:
                    self._label_to_int[identity] = len(self._label_to_int)

                self.samples.append({
                    "frame_paths": frame_paths,
                    "label":       self._label_to_int[identity],
                    "subject_id":  identity,
                    "seq_id":      seq_id,
                    "session":     session,
                    "clothes":     clothes,
                })

        if not self.samples:
            raise RuntimeError(
                f"No sequences loaded for split='{split}' from {txt_path}. "
                f"Did you run the symlink script? root={self.root}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        return {
            "frames":     self._load_frames(s["frame_paths"]),
            "label":      s["label"],
            "subject_id": s["subject_id"],
            "seq_id":     s["seq_id"],
            "session":    s["session"],
            "clothes":    s["clothes"],
        }

    def _load_frames(self, frame_paths: List[Path]) -> torch.Tensor:
        T, n = self.num_frames, len(frame_paths)
        if n >= T:
            indices = [int(i * n / T) for i in range(T)]
        else:
            indices = [i % n for i in range(T)]
        frames = [Image.open(frame_paths[i]).convert("RGB") for i in indices]
        return self.transform(frames)

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
        return {"subject_id": s["subject_id"], "seq_id": s["seq_id"], "clothes": s["clothes"]}
