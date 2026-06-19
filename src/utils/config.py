"""Configuration management via OmegaConf."""

from pathlib import Path
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf


def load_config(
    model_cfg: str = "configs/model.yaml",
    train_cfg: str = "configs/train.yaml",
    dataset_cfg: Optional[str] = None,
    vis_cfg: str = "configs/visualization.yaml",
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """Load and merge all configuration files into a single DictConfig.

    Priority (highest to lowest): overrides > dataset_cfg > train_cfg > model_cfg > vis_cfg.

    Args:
        model_cfg:   Path to model architecture YAML.
        train_cfg:   Path to training hyperparameter YAML.
        dataset_cfg: Path to dataset-specific YAML (e.g. configs/ccvid.yaml).
        vis_cfg:     Path to visualization YAML.
        overrides:   List of dotlist overrides, e.g. ["training.epochs=200"].

    Returns:
        Merged DictConfig.
    """
    cfgs: List[DictConfig] = []
    for path in [vis_cfg, model_cfg, train_cfg]:
        if path and Path(path).exists():
            cfgs.append(OmegaConf.load(path))
        elif path:
            raise FileNotFoundError(f"Config not found: {path}")

    if dataset_cfg:
        if not Path(dataset_cfg).exists():
            raise FileNotFoundError(f"Dataset config not found: {dataset_cfg}")
        cfgs.append(OmegaConf.load(dataset_cfg))

    cfg: DictConfig = OmegaConf.merge(*cfgs)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    return cfg


def save_config(cfg: DictConfig, path: str) -> None:
    """Persist a config to disk as YAML."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, str(out))


def print_config(cfg: DictConfig) -> None:
    """Pretty-print the full config."""
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
