#!/bin/bash
# Training script with sensible defaults

set -e

# Set PYTHONPATH to include project root
export PYTHONPATH="$(cd "$(dirname "$0")/../../" && pwd):${PYTHONPATH:-}"

# Configuration with defaults
MODEL=${MODEL:-convnext_8}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-32}
LR=${LR:-1e-3}
# DATA_ROOT=${DATA_ROOT:-./data}
DATA_ROOTS=${DATA_ROOTS:-./data}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-./ckpt}
LOGDIR=${LOGDIR:-./runs}
NUM_WORKERS=${NUM_WORKERS:-4}
SCHEDULER=${SCHEDULER:-cosine_warmup}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
USE_AMP=${USE_AMP:-true}
GRAD_ACCUM=${GRAD_ACCUM:-1}
TRAIN_DATASETS=${TRAIN_DATASETS:-"artemis dvisa emoset"}
VAL_DATASETS=${VAL_DATASETS:-"artemis"}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --data-roots)
            DATA_ROOTS="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --train-datasets)
            shift
            TRAIN_DATASETS=""
            while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
                TRAIN_DATASETS="$TRAIN_DATASETS $1"
                shift
            done
            ;;
        --val-datasets)
            shift
            VAL_DATASETS=""
            while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
                VAL_DATASETS="$VAL_DATASETS $1"
                shift
            done
            ;;
        --help)
            echo "Training script for emotion prediction models"
            echo ""
            echo "Usage: ./train.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL               Model type: convnext_unet, convnext_8, convnext_2 (default: convnext_8)"
            echo "  --epochs N                  Number of training epochs (default: 100)"
            echo "  --batch-size N              Batch size (default: 32)"
            echo "  --lr LR                     Learning rate (default: 1e-3)"
            echo "  --data-root DIR             Path to data directory (default: ./data)"
            echo "  --data-roots DIRS...        Paths to data root directories (one per dataset)"
            echo "  --checkpoint-dir DIR        Checkpoint directory (default: ./ckpt)"
            echo "  --train-datasets NAMES...   Training datasets (default: artemis dvisa)"
            echo "  --val-datasets NAMES...     Validation datasets (default: artemis)"
            echo "  --resume CHECKPOINT         Resume from checkpoint"
            echo "  --help                      Show this help message"
            echo ""
            echo "Environment variables (for defaults in batch scripts):"
            echo "  MODEL, EPOCHS, BATCH_SIZE, LR, DATA_ROOT, USE_AMP, GRAD_ACCUM"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build command
CMD="python -m public_emotion_pred.main"
CMD="$CMD --model $MODEL"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --lr $LR"
CMD="$CMD --data-root $DATA_ROOT"
CMD="$CMD --checkpoint-dir $CHECKPOINT_DIR"
CMD="$CMD --logdir $LOGDIR"
CMD="$CMD --num-workers $NUM_WORKERS"
CMD="$CMD --scheduler $SCHEDULER"
CMD="$CMD --warmup-epochs $WARMUP_EPOCHS"
CMD="$CMD --gradient-accumulation $GRAD_ACCUM"
CMD="$CMD --train-datasets $TRAIN_DATASETS"
CMD="$CMD --val-datasets $VAL_DATASETS"

if [ "$USE_AMP" = true ]; then
    CMD="$CMD --use-amp"
fi

if [ -n "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
fi

echo "Starting training with command:"
echo "$CMD"
echo ""

eval $CMD
