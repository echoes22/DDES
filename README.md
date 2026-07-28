# DDES: Dimensional Distribution Emotion State

Official PyTorch implementation of the paper:

> **Dimensional Distribution Emotion State: Leveraging Valence and Arousal as a Common Embedding Space for Visual Emotion Analysis**  
> Émile Bergeron et al., arXiv 2605.26262, 2026

The framework represents image emotion as a 2D probability heatmap over valence-arousal space (DDES) and provides bidirectional conversions between three complementary representations:
- **DDES** — 28×28 heatmap over V-A space
- **DES** — single (valence, arousal) point
- **CES** — N-class categorical emotion distribution

Three model variants are trained with unified supervision and shared conversion equations.

## Features

- **Three Model Architectures**
  - `convnext_unet`: Outputs 28×28 emotion heatmaps
  - `convnext_8`: Outputs 8-way emotion distributions
  - `convnext_2`: Outputs valence-arousal coordinates in [-1,1]

- **Multi-Dataset Support**
  - ArtEmis (KDE heatmaps + 8-class CES distributions)
  - D-ViSA (valence-arousal)
  - EmoSet-118k (8-class VA-mapped distributions)
  - WikiArt-Emotion (20-class distributions, benchmark)
  - EEMO-Bench (VA labels + 7-class emotion dict, benchmark)

- **Unified Training Loop**: Any model can train on any dataset combination
- **Automatic Metric Conversion**: VA ↔ emotion distribution conversions
- **Multi-GPU Training**: Distributed data parallel support
- **Mixed Precision**: Automatic mixed precision (AMP) support
- **Learning Rate Scheduling**: Cosine annealing with optional warmup
- **TensorBoard Logging**: Real-time training visualization

## Quick Start

### Training

**Single-dataset training** (Table 1 in paper):

```bash
# DDES-Net on ArtEmis
sbatch scripts/sbatch_train_ddes_rrg.sh

# CES-Net on ArtEmis
sbatch scripts/sbatch_train_ces_rrg.sh

# DES-Net on D-ViSA
sbatch scripts/sbatch_train_des_rrg.sh
```

**Combined training** (Table 3 in paper — ArtEmis + D-ViSA + EmoSet, evaluated on EEMO-Bench and WikiArt-Emotion):

```bash
sbatch scripts/sbatch_train_ddes_combined_rrg.sh
sbatch scripts/sbatch_train_ces_combined_rrg.sh
sbatch scripts/sbatch_train_des_combined_rrg.sh
```

Or use Python directly (recommended hyperparameters):

```bash
python -m public_emotion_pred.main \
    --model convnext_unet \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-4 \
    --warmup-epochs 2 \
    --scheduler cosine_warmup \
    --use-amp \
    --train-datasets artemis \
    --val-datasets artemis \
    --artemis-dir /path/to/artemis_kde \
    --wiki-root /path/to/wikiart \
    --checkpoint-dir ./ckpt/ddes
```

### Using Config Files

Train with a config file (CLI args override file values):

```bash
python -m public_emotion_pred.main --config config.yaml
python -m public_emotion_pred.main --config config.yaml --epochs 200 --batch-size 64
```

See [CONFIG.md](CONFIG.md) for detailed configuration documentation.

### Multi-GPU Training

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    -m public_emotion_pred.main \
    --model convnext_unet \
    --batch-size 32 \
    --epochs 100
```

### Inference on Image Folder

```bash
cd public_emotion_pred
./scripts/inference.sh --image-dir ./my_images --checkpoint ./ckpt/best/checkpoint_epoch_100.pt
```

## Installation

```bash
# Clone repository
git clone <repo-url>
cd <repo-dir>

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r public_emotion_pred/requirements.txt
```

## Documentation

- **[CONFIG.md](CONFIG.md)** - Configuration system documentation

## Package Structure

```
public_emotion_pred/
├── __init__.py                  # Package exports
├── main.py                      # Training entry point
├── train.py                     # Training & evaluation loops
├── models.py                    # Model architectures
├── dataset.py                   # Dataset loading
├── config.py                    # Configuration system
├── inference.py                 # Inference pipeline
├── inference_utils.py           # VA/emotion conversions
├── loss.py                      # Loss & metrics
├── kde.py                       # KDE sampling
├── lr_scheduler.py              # Learning rate schedulers
├── config.yaml                  # Sample YAML config
├── config.json                  # Sample JSON config
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── CONFIG.md                    # Configuration docs
├── ckpt/                        # Directory for checkpoints (user-provided)
└── scripts/
    ├── train.sh                 # Training bash script
    ├── evaluate.sh              # Evaluation bash script
    ├── inference.sh             # Inference bash script
    └── USAGE.md                 # Script usage guide
```

## Module Overview

### Core Modules

- **config.py** - Configuration management with file loading and CLI merging
- **models.py** - Three model variants (UNet, 8-way, 2D-VA)
- **dataset.py** - Multi-dataset unified loader
- **train.py** - Single training loop supporting any model/dataset combination
- **inference.py** - Batch inference on image folders

### Utilities

- **inference_utils.py** - VA/emotion conversions, heatmap generation
- **loss.py** - Metrics computation and TensorBoard logging
- **kde.py** - KDE-based heatmap to distribution conversion
- **lr_scheduler.py** - Learning rate scheduling options

## Command-Line Arguments

### Model & Checkpoint
- `--model`: Model architecture (convnext_unet | convnext_8 | convnext_2)
- `--checkpoint`: Path to pretrained backbone checkpoint
- `--freeze-backbone`: Freeze pretrained backbone weights

### Dataset
- `--train-datasets`: Training dataset names (e.g. `artemis dvisa emoset`)
- `--val-datasets`: Validation dataset names (default: same as train)
- `--benchmark-datasets`: Unseen benchmark datasets to eval after training (e.g. `eemo wikiart_emo`)
- `--data-root`: Path to data root directory (default: ./data)
- `--artemis-dir`: Directory containing ArtEmis `{train,val,test}_kde_df.pkl` files
- `--dvisa-dir`: Directory containing D-ViSA `{train,val,test}.pkl` files
- `--emoset-dir`: EmoSet-118k root directory
- `--wikiart-emo-dir`: WikiArt-Emotion directory (contains `WikiArt-Emotions-All-fixed-5.csv`)
- `--eemo-dir`: EEMO-Bench root directory (contains `EEmo-Bench.json` and `images/`)
- `--wiki-root`: WikiArt image root (used for image loading across all datasets)

### Training
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-3)
- `--weight-decay`: Weight decay (default: 0.0)
- `--optimizer`: Optimizer type (adam | adamw | sgd) (default: adamw)
- `--scheduler`: LR scheduler (cosine | cosine_warmup | cosine_restarts | none)
- `--warmup-epochs`: Number of warmup epochs (default: 5)
- `--gradient-accumulation`: Gradient accumulation steps (default: 1)
- `--use-amp`: Enable automatic mixed precision

### Checkpointing
- `--checkpoint-dir`: Directory for saving checkpoints (default: ./ckpt)
- `--resume`: Path to checkpoint to resume from
- `--save-freq`: Save checkpoint every N epochs (default: 1)

### Configuration
- `--config`: Path to config file (YAML or JSON)
- `--save-config`: Save resolved config to file after training starts

### Evaluation
- `--eval-only`: Skip training; load best checkpoint and run test evaluation
- `--num-workers`: Number of data loading workers (default: 4)
- `--seed`: Random seed (default: 42)
- `--logdir`: TensorBoard log directory (default: ./runs)

## Data Preparation

### ArtEmis (DDES / CES supervision)

Pre-process raw ArtEmis V2 CSV into per-painting KDE heatmaps:

```bash
python -m public_emotion_pred.scripts.prepare_artemis_kde \
    --input /path/to/artemis_v2.csv \
    --output-dir /path/to/artemis_kde \
    --grid-size 28
```

This produces `{train,val,test}_kde_df.pkl`, each with columns:
- `painting` — WikiArt artwork ID (e.g. `picasso_guernica`)
- `art_style` — WikiArt style name
- `kde_norm` — (28, 28) float32 array, normalised heatmap
- `mapped_dist` — (8,) float32 array, CES emotion distribution

### D-ViSA (DES supervision)

D-ViSA pickles are used as-is: `{train,val,test}.pkl` with columns `artwork`, `art_style`, `final_vad` (string `[v, a, d]`), `split`.

### WikiArt images

All datasets use artwork images from WikiArt. Point `--wiki-root` at the directory containing style subdirectories:

```
wikiart/
├── abstract_expressionism/
├── baroque/
└── ...
```

### Training example

```bash
python -m public_emotion_pred.main \
    --model convnext_unet \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-4 \
    --warmup-epochs 2 \
    --scheduler cosine_warmup \
    --use-amp \
    --train-datasets artemis \
    --val-datasets artemis \
    --artemis-dir /path/to/artemis_kde \
    --wiki-root /path/to/wikiart \
    --checkpoint-dir ./ckpt/ddes \
    --logdir ./runs/ddes
```

Or submit via SLURM:

```bash
sbatch scripts/sbatch_train_ddes_rrg.sh   # DDES-Net
sbatch scripts/sbatch_train_ces_rrg.sh    # CES-Net
sbatch scripts/sbatch_train_des_rrg.sh    # DES-Net
```

## Model Output Formats

- **convnext_unet** (DDES-Net): `(B, 1, 28, 28)` log-softmax heatmap
  - Loss: KL divergence vs. target KDE heatmap
  - Metrics: Converted to CES via bilinear interpolation at emotion coords (Eq. 4)

- **convnext_8** (CES-Net): `(B, 8)` log-softmax distribution over 8 emotions
  - Loss: KL divergence vs. target categorical distribution
  - Metrics: Top-1/2/3 accuracy, KL divergence

- **convnext_2** (DES-Net): `(B, 2)` raw valence-arousal in [-1, 1]²
  - Loss: MSE vs. target VA
  - Metrics: MSE per axis, 2D RMSE, Pearson correlation

## Emotion Coordinates

8 emotions in valence-arousal space:

| Emotion      | Valence | Arousal |
|--------------|---------|---------|
| Amusement    | 0.858   | 0.674   |
| Awe          | -0.062  | 0.480   |
| Contentment  | 0.750   | 0.220   |
| Excitement   | 0.792   | 0.368   |
| Anger        | -0.666  | 0.730   |
| Disgust      | -0.896  | 0.550   |
| Fear         | -0.854  | 0.680   |
| Sadness      | -0.896  | -0.424  |

## Performance Tips

### For Faster Training
1. Use mixed precision: `--use-amp`
2. Increase batch size (if GPU memory allows)
3. Use multi-GPU training with DDP
4. Use single gradient accumulation step

### For Better Convergence
1. Use warmup: `--scheduler cosine_warmup --warmup-epochs 10`
2. Adjust learning rate based on batch size
3. Train on multiple datasets for regularization
4. Use weight decay: `--weight-decay 1e-4` to `5e-4`

### For Best Accuracy
1. Train for more epochs: `--epochs 200`
2. Use smaller learning rate: `--lr 1e-4`
3. Fine-tune on target dataset after pre-training
4. Ensemble multiple models

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python -m public_emotion_pred.main --batch-size 16

# Use gradient accumulation
python -m public_emotion_pred.main --gradient-accumulation 2 --batch-size 64

# Enable mixed precision
python -m public_emotion_pred.main --use-amp --batch-size 128
```

### Poor Convergence
```bash
# Increase warmup epochs
python -m public_emotion_pred.main --scheduler cosine_warmup --warmup-epochs 20

# Reduce learning rate
python -m public_emotion_pred.main --lr 5e-5

# Add weight decay
python -m public_emotion_pred.main --weight-decay 1e-4
```

### Missing Data
- Ensure dataset files are in correct location
- Check `--data-root` path
- Verify CSV annotations have correct format
- Check image file extensions (supports JPEG, PNG, BMP, TIFF)

## Citation

If you use this code in your research, please cite:

```bibtex
@InProceedings{Bergeron_2026_CVPR,
    author    = {Bergeron, Emile and Dhossou, Tadagb\'e and Tremblay, S\'ebastien and Lalonde, Jean-Fran\c{c}ois},
    title     = {Dimensional Distribution Emotion State: Leveraging Valence and Arousal as a Common Embedding Space for Visual Emotion Analysis},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2026},
    pages     = {5234-5241}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contributing

Pull requests welcome! Please ensure:
1. Code follows existing style
2. New features include tests
3. Documentation is updated

## Support

For issues and questions, please open a GitHub issue or contact the authors.
