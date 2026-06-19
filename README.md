# CHEGT: Cross-clothes Human Evidence Graph Transformer

A gait recognition model designed to be **robust to clothing changes** by reasoning over three disentangled evidence types — motion, appearance, and consistency — through a learned graph transformer.

---

## Key Novelty

Prior work shows HMR/pose geometry hurts gait recognition performance; **temporal information is the primary discriminative signal**. CHEGT operationalises this insight by introducing the **Human Evidence Graph (HEG)**:

| Evidence Node | What it captures | Key features |
|---|---|---|
| **Motion** | *How* the person moves | Inter-frame velocity, stride cadence (FFT), acceleration |
| **Appearance** | *What* they look like | DINOv2 semantics, temporally-attended frame selection |
| **Consistency** | *Stability* of their signature | Temporal variance, lag-1 autocorrelation, MAD |

A Graph Transformer reasons over all three nodes jointly, producing an attention weight matrix that is directly interpretable as evidence importances.

---

## Architecture

```
Input (B, T, C, H, W)
        │
   ┌────▼────────────┐
   │  DINOv2-B/14    │  CLS token per frame  → (B, T, 768)
   └────┬────────────┘
        │
   ┌────▼────────────┐
   │ Temporal        │  Pre-LN TransformerEncoder → (B, T, 768)
   │ Transformer     │  4 layers, 8 heads
   └────┬────────────┘
        │
   ┌────┼─────────────────────┐
   │    │                     │
   ▼    ▼                     ▼
Motion  Appearance      Consistency
 Node    Node              Node        Each → (B, 256)
   │    │                     │
   └────┼─────────────────────┘
        │  stack  (B, 3, 256)
   ┌────▼────────────┐
   │ Graph           │  Pre-LN, 2 layers, 4 heads
   │ Transformer     │  Saves (B, 3, 3) attention → vis
   └────┬────────────┘
        │
   ┌────▼────────────┐
   │ Evidence        │  Attention pooling → (B, 256)
   │ Pooling         │  Saves (B, 3) importance → vis
   └────┬────────────┘
        │
   ┌────▼────────────┐
   │ Gait Head       │  Linear → BN → GELU → Linear → BNNeck → L2-norm
   └────┬────────────┘
        │
   Embedding (B, 512)
```

---

## Installation

```bash
# 1. Clone
git clone <your-repo-url>
cd CHEGT

# 2. Create environment
conda create -n chegt python=3.10 -y
conda activate chegt

# 3. Install PyTorch (CUDA 11.8 example)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 4. Install remaining dependencies
pip install -r requirements.txt
```

---

## Dataset Preparation

### CC-VID
```
datasets/ccvid/
  001/
    nm-01/
      00001.jpg
      00002.jpg
      …
    cl-01/
      …
  002/
    …
```

### FVGB
```
datasets/fvgb/
  001/
    normal/
      seq_01/
        00001.jpg …
    bag/
      seq_01/ …
  002/ …
```

Update `configs/ccvid.yaml` or `configs/fvgb.yaml` with the correct `root` path.

---

## Training

```bash
# Train on CC-VID with default config
python train.py --dataset configs/ccvid.yaml

# Train on FVGB with overrides
python train.py --dataset configs/fvgb.yaml \
    training.epochs=150 \
    optimizer.lr=2e-4 \
    sampler.P=32 sampler.K=4

# Resume from checkpoint
python train.py --dataset configs/ccvid.yaml \
    experiment.resume=runs/chegt_ccvid/checkpoints/best.pth
```

### Monitor Training
```bash
tensorboard --logdir runs/
```

---

## Evaluation

```bash
# Full evaluation from best checkpoint
python evaluate.py \
    --checkpoint runs/chegt_ccvid/checkpoints/best.pth \
    --dataset configs/ccvid.yaml

# Cross-dataset evaluation
python evaluate.py \
    --checkpoint runs/chegt_ccvid/checkpoints/best.pth \
    --dataset configs/fvgb.yaml \
    --output_dir runs/cross_eval/
```

Outputs a `metrics.json`:
```json
{
  "epoch": 100,
  "rank_1": 87.3,
  "rank_5": 96.1,
  "rank_10": 97.8,
  "rank_20": 98.9,
  "mAP": 82.4,
  "eer": 4.2,
  "auc": 98.1
}
```

---

## Visualisation

```bash
# Generate all figures
python visualize.py \
    --checkpoint runs/chegt_ccvid/checkpoints/best.pth \
    --dataset configs/ccvid.yaml

# Specific figures only
python visualize.py \
    --checkpoint runs/chegt_ccvid/checkpoints/best.pth \
    --dataset configs/ccvid.yaml \
    --figures tsne umap attention importance
```

### Generated Figures

| File | Description |
|---|---|
| `tsne_epXXXX.png` | t-SNE of gait embeddings, coloured by identity |
| `umap_epXXXX.png` | UMAP projection |
| `attention_epXXXX.png` | 3×3 evidence graph attention heatmap |
| `importance_epXXXX.png` | Per-node importance bar chart |
| `distance_epXXXX.png` | Probe × gallery L2 distance heatmap |
| `training_curves.png` | Loss, Rank-1, mAP, EER over epochs |

---

## Configuration

All config is YAML-driven with OmegaConf merging and dotlist overrides.

```
configs/
  model.yaml       ← architecture dims, layer counts
  train.yaml       ← LR, epochs, sampler P/K, loss margin
  ccvid.yaml       ← dataset paths, split definitions
  fvgb.yaml        ← dataset paths, split definitions
  visualization.yaml ← figure sizes, colours, t-SNE params
```

Priority: `overrides > dataset > train > model > visualization`

---

## Project Structure

```
CHEGT/
├── configs/                  # YAML configuration files
├── datasets/
│   └── loaders/
│       ├── ccvid_dataset.py  # CC-VID dataset loader
│       ├── fvgb_dataset.py   # FVGB dataset loader
│       ├── pk_sampler.py     # PK batch sampler (P identities × K sequences)
│       └── transforms.py     # Synchronized video augmentation
├── src/
│   ├── models/
│   │   ├── backbone/         # DINOv2 wrapper
│   │   ├── temporal/         # Temporal Transformer
│   │   ├── evidence/         # Motion / Appearance / Consistency nodes
│   │   ├── graph/            # Graph Transformer + node embeddings
│   │   ├── pooling/          # Evidence Pooling (attention/mean/max)
│   │   ├── heads/            # GaitHead (BNNeck + L2 norm)
│   │   └── chegt.py          # Main model + CHEGTOutput dataclass
│   ├── losses/
│   │   ├── triplet_loss.py   # Batch-hard / semi-hard / all-pairs
│   │   └── combined_loss.py  # Weighted loss orchestrator
│   ├── metrics/
│   │   ├── retrieval_metrics.py    # CMC curve + mAP
│   │   └── verification_metrics.py # EER + AUC-ROC
│   ├── visualization/
│   │   ├── training_curves.py      # Loss/metric dashboards
│   │   ├── attention_maps.py       # 3×3 graph attention heatmap
│   │   ├── evidence_importance.py  # Per-node importance bar chart
│   │   ├── embedding_tsne.py       # t-SNE projections
│   │   └── embedding_umap.py       # UMAP projections
│   └── utils/
│       ├── checkpoint.py     # Save/load/rotate checkpoints
│       ├── config.py         # OmegaConf loader/merger
│       ├── logger.py         # Console + TensorBoard logger
│       └── seed.py           # Full reproducibility
├── train.py                  # Training entry point
├── evaluate.py               # Evaluation entry point
├── visualize.py              # Standalone visualisation entry point
└── requirements.txt
```

---

## Design Decisions

- **Pre-LN throughout** — LayerNorm before attention/FFN for training stability.
- **DINOv2-B/14** — strong frozen semantic features; only later blocks fine-tuned.
- **BNNeck** — standard ReID practice; batch norm before L2 normalisation.
- **Backbone at 0.1× LR** — prevents catastrophic forgetting of pretrained weights.
- **Batch-hard triplet mining** — most stable variant for small-N gait datasets.
- **Attention-stored intermediates** — graph attention and evidence weights are
  detached after each forward pass and exposed as properties for zero-overhead
  visualisation without extra forward passes.

---

## Citation

```bibtex
@article{chegt2025,
  title   = {CHEGT: Cross-clothes Human Evidence Graph Transformer for Gait Recognition},
  author  = {Your Name},
  year    = {2025},
}
```
