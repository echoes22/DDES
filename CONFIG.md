# Configuration System

The emotion prediction package includes a comprehensive configuration system that supports loading configuration from YAML/JSON files with CLI argument overrides.

## Quick Start

### Using Default Configuration
```bash
python -m public_emotion_pred.main
```

### Using Config File
```bash
python -m public_emotion_pred.main --config public_emotion_pred/config.yaml
python -m public_emotion_pred.main --config public_emotion_pred/config.json
```

### Override Config File with CLI Arguments
CLI arguments take precedence over config file values:
```bash
# Config file sets epochs=100, but this runs for 200 epochs
python -m public_emotion_pred.main --config public_emotion_pred/config.yaml --epochs 200

# Config file sets model=convnext_8, but this trains convnext_unet
python -m public_emotion_pred.main --config public_emotion_pred/config.yaml --model convnext_unet --batch-size 64
```

## Configuration Format

### YAML Format (`config.yaml`)
```yaml
data:
  image_path: ./data
  batch_size: 32
  num_workers: 4
  pin_memory: true
  
model:
  type: convnext_8
  resume: ''
  
train:
  epochs: 100
  base_lr: 0.001
  warmup_epochs: 5
  use_amp: false
  lr_scheduler:
    name: cosine_warmup
    decay_epochs: 5

output: ./ckpt
tensorboard_path: ./runs
seed: 42
```

### JSON Format (`config.json`)
```json
{
  "data": {
    "image_path": "./data",
    "batch_size": 32,
    "num_workers": 4,
    "pin_memory": true
  },
  "model": {
    "type": "convnext_8",
    "resume": ""
  },
  "train": {
    "epochs": 100,
    "base_lr": 0.001,
    "warmup_epochs": 5,
    "use_amp": false,
    "lr_scheduler": {
      "name": "cosine_warmup",
      "decay_epochs": 5
    }
  },
  "output": "./ckpt",
  "tensorboard_path": "./runs",
  "seed": 42
}
```

## Configuration Options

### Data Configuration

```yaml
data:
  image_path: ./data              # Path to data directory
  batch_size: 32                  # Batch size for training
  num_workers: 4                  # Number of data loading workers
  pin_memory: true                # Pin memory for faster loading
  dataset: artemis                # Default dataset name
  augmentation: false             # Enable/disable data augmentation
  use_mini_ds: false              # Use mini dataset for testing
```

### Model Configuration

```yaml
model:
  type: convnext_8                # Model architecture
                                  # Options: convnext_unet, convnext_8, convnext_2
  resume: ''                      # Path to pretrained checkpoint
                                  # Empty string = train from scratch
```

### Training Configuration

```yaml
train:
  epochs: 100                     # Number of training epochs
  base_lr: 0.001                  # Base learning rate
  warmup_lr: 0.0000005            # Warmup learning rate
  min_lr: 0.000005                # Minimum learning rate (for cosine annealing)
  warmup_epochs: 5                # Number of warmup epochs
  use_amp: false                  # Use automatic mixed precision
  from_scratch: false             # Train from scratch (ignore pretrained weights)
  
  lr_scheduler:
    name: cosine_warmup           # Scheduler type
                                  # Options: cosine, cosine_warmup, cosine_restarts, none
    decay_epochs: 5               # Decay epochs (for optional schedulers)
```

### Logging & Checkpointing

```yaml
save_freq: 1                      # Save checkpoint every N epochs
print_freq: 5                     # Print stats every N batches
output: ./ckpt                    # Checkpoint save directory
tensorboard_path: ./runs          # TensorBoard log directory
```

### Other Settings

```yaml
seed: 42                          # Random seed for reproducibility
local_rank: -1                    # Local rank for distributed training
                                  # -1 = no distributed training
```

## Using Configuration in Code

### Load Config File

```python
from public_emotion_pred import load_config_from_file

config = load_config_from_file('public_emotion_pred/config.yaml')
# or
config = load_config_from_file('public_emotion_pred/config.json')
```

### Load Config with CLI Override

```python
from public_emotion_pred import get_config_with_file
from argparse import Namespace

# Create arguments namespace
args = Namespace(
    epochs=200,
    batch_size=64,
    model='convnext_unet'
)

# Load config, CLI args take precedence
config = get_config_with_file(config_file='public_emotion_pred/config.yaml', args=args)
```

### Merge Config with Arguments

```python
from public_emotion_pred import load_config_from_file, merge_config_with_args

config = load_config_from_file('public_emotion_pred/config.yaml')
config = merge_config_with_args(config, args)  # args override config
```

### Save Config to File

```python
from public_emotion_pred import save_config_to_file

save_config_to_file(config, 'output_config.yaml')
# or
save_config_to_file(config, 'output_config.json')
```

### Convert Config to Dictionary

```python
from public_emotion_pred import config_to_dict

config_dict = config_to_dict(config)
print(config_dict)
```

## Workflow Examples

### Example 1: Train with Config File

```bash
python -m public_emotion_pred.main --config public_emotion_pred/config.yaml
```

Creates a trained model using all settings from `config.yaml`.

### Example 2: Train with Config File + CLI Overrides

```bash
python -m public_emotion_pred.main \
    --config public_emotion_pred/config.yaml \
    --epochs 200 \
    --batch-size 64 \
    --lr 5e-4 \
    --model convnext_unet
```

Uses `config.yaml` as base, then overrides:
- epochs: 100 → 200
- batch_size: 32 → 64
- base_lr: 0.001 → 0.0005
- model: convnext_8 → convnext_unet

### Example 3: Save Config After Training Starts

```bash
python -m public_emotion_pred.main \
    --config public_emotion_pred/config.yaml \
    --save-config ./runs/train_config.yaml
```

Saves the final resolved config (file + CLI overrides) to disk.

### Example 4: Multiple Training Runs with Different Configs

```bash
# Create separate config files
cp public_emotion_pred/config.yaml public_emotion_pred/config_small.yaml  # Edit for small model
cp public_emotion_pred/config.yaml public_emotion_pred/config_large.yaml  # Edit for large model

# Train both
python -m public_emotion_pred.main --config public_emotion_pred/config_small.yaml
python -m public_emotion_pred.main --config public_emotion_pred/config_large.yaml
```

### Example 5: Hyperparameter Sweep

Create `sweep_configs.py`:
```python
from public_emotion_pred import get_config_with_file, save_config_to_file
from argparse import Namespace
import os

for lr in [1e-3, 5e-4, 1e-4]:
    for batch_size in [32, 64, 128]:
        args = Namespace(
            lr=lr,
            batch_size=batch_size,
            epochs=100,
            checkpoint_dir=f'./ckpt/lr_{lr}_bs_{batch_size}',
            logdir=f'./runs/lr_{lr}_bs_{batch_size}'
        )
        
        config = get_config_with_file('public_emotion_pred/config.yaml', args)
        
        # Save config for reference
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        save_config_to_file(config, os.path.join(args.checkpoint_dir, 'config.yaml'))
```

Run:
```bash
python sweep_configs.py
```

## Configuration Precedence

When using `get_config_with_file()`:

1. **Default values** from Config dataclasses
2. **Config file values** (YAML/JSON) override defaults
3. **CLI arguments** override config file (if not None)

```
Defaults < Config File < CLI Args
```

Example:
```
# config.yaml sets:
epochs: 100
batch_size: 32
lr: 0.001

# Command:
python -m public_emotion_pred.main --config public_emotion_pred/config.yaml --epochs 200

# Result:
epochs: 200  (from CLI, overrides config file)
batch_size: 32  (from config file, no CLI override)
lr: 0.001  (from config file, no CLI override)
```

## Tips

1. **Version Control**: Commit config files to track experiment settings
2. **Reproducibility**: Use `--save-config` to save final config after training starts
3. **Defaults**: Keep minimal config files, use CLI for one-off changes
4. **Documentation**: Add comments to config files explaining each parameter

## Supported File Formats

- **YAML**: `.yaml`, `.yml`
- **JSON**: `.json`

Both formats are fully equivalent. Choose based on preference:
- **YAML**: More readable, better for humans
- **JSON**: Strictly structured, better for programmatic generation

## Error Handling

If config file doesn't exist:
```bash
python -m public_emotion_pred.main --config nonexistent.yaml
# Error: Config file not found: nonexistent.yaml
```

If config file has invalid format:
```bash
python -m public_emotion_pred.main --config invalid.yaml
# Error: Failed to parse config file
```

Invalid values will use defaults or raise validation errors.
