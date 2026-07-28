#!/bin/bash
# Evaluation script for testing on a dataset

set -e

# Set PYTHONPATH to include project root
export PYTHONPATH="$(cd "$(dirname "$0")/../../" && pwd):${PYTHONPATH:-}"

# Configuration
MODEL=${MODEL:-convnext_8}
CHECKPOINT=${CHECKPOINT:-./ckpt/best/checkpoint_epoch_000.pt}
DATA_ROOT=${DATA_ROOT:-./data}
TEST_DATASETS=${TEST_DATASETS:-"artemis"}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-4}
DEVICE=${DEVICE:-auto}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --test-datasets)
            shift
            TEST_DATASETS=""
            while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
                TEST_DATASETS="$TEST_DATASETS $1"
                shift
            done
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help)
            echo "Evaluation script for emotion prediction models"
            echo ""
            echo "Usage: ./evaluate.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL               Model type: convnext_unet, convnext_8, convnext_2 (default: convnext_8)"
            echo "  --checkpoint PATH           Path to model checkpoint (default: ./ckpt/best/checkpoint_epoch_000.pt)"
            echo "  --data-root DIR             Path to data directory (default: ./data)"
            echo "  --batch-size N              Batch size (default: 64)"
            echo "  --test-datasets NAMES...    Test datasets (default: artemis)"
            echo "  --device DEVICE             Device: auto, cuda, cpu (default: auto)"
            echo "  --help                      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "Evaluation Configuration:"
echo "  Model: $MODEL"
echo "  Checkpoint: $CHECKPOINT"
echo "  Data Root: $DATA_ROOT"
echo "  Test Datasets: $TEST_DATASETS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Device: $DEVICE"
echo ""

# Run evaluation using Python
python -c "
import torch
import sys
sys.path.insert(0, '.')

from public_emotion_pred import get_model, build_combined_emotion_dataloaders
from public_emotion_pred.train import evaluate_on_multi_dataset_any_model

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if '$DEVICE' == 'auto' else torch.device('$DEVICE')
print(f'Using device: {device}')

# Load model
print(f'Loading model: $MODEL')
model = get_model('$MODEL')
checkpoint = torch.load('$CHECKPOINT', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

# Build dataloader
print(f'Building test dataloader for: $TEST_DATASETS')
test_loader = build_combined_emotion_dataloaders(
    data_root='$DATA_ROOT',
    datasets=['test'] + '$TEST_DATASETS'.split(),
    batch_size=$BATCH_SIZE,
    num_workers=$NUM_WORKERS,
    shuffle=False
)

# Evaluate
print(f'Evaluating on {len(test_loader)} batches...')
results = evaluate_on_multi_dataset_any_model(
    test_loader,
    model,
    device,
    model_type='$MODEL',
    tag='test'
)

print()
print('Evaluation Results:')
for key, val in results.items():
    if isinstance(val, float):
        print(f'  {key}: {val:.6f}')
    else:
        print(f'  {key}: {val}')
"
