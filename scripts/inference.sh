#!/bin/bash
# Inference script for running on a folder of images

set -e

# Set PYTHONPATH to include project root
export PYTHONPATH="$(cd "$(dirname "$0")/../../" && pwd):${PYTHONPATH:-}"

# Configuration
MODEL=${MODEL:-convnext_8}
CHECKPOINT=${CHECKPOINT:-./ckpt/best/checkpoint_epoch_000.pt}
IMAGE_DIR=${IMAGE_DIR:-.}
OUTPUT_DIR=${OUTPUT_DIR:-./inference_results}
BATCH_SIZE=${BATCH_SIZE:-32}
FORMAT=${FORMAT:-json}
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
        --image-dir)
            IMAGE_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help)
            echo "Inference script for emotion prediction on image folders"
            echo ""
            echo "Usage: ./inference.sh --image-dir PATH/TO/IMAGES [OPTIONS]"
            echo ""
            echo "Required:" 
            echo "  --image-dir DIR             Path to folder containing images"
            echo ""
            echo "Options:"
            echo "  --model MODEL               Model type: convnext_unet, convnext_8, convnext_2 (default: convnext_8)"
            echo "  --checkpoint PATH           Path to model checkpoint (default: ./ckpt/best/checkpoint_epoch_000.pt)"
            echo "  --output-dir DIR            Output directory for results (default: ./inference_results)"
            echo "  --batch-size N              Batch size for inference (default: 32)"
            echo "  --format FORMAT             Output format: json, csv, both (default: json)"
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

# Validate inputs
if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: Image directory not found: $IMAGE_DIR"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "Inference Configuration:"
echo "  Model: $MODEL"
echo "  Checkpoint: $CHECKPOINT"
echo "  Image Directory: $IMAGE_DIR"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Batch Size: $BATCH_SIZE"
echo "  Output Format: $FORMAT"
echo "  Device: $DEVICE"
echo ""

# Run inference
python -m public_emotion_pred.inference \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --image-dir "$IMAGE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --format "$FORMAT" \
    --device "$DEVICE"

echo ""
echo "Done! Results saved to: $OUTPUT_DIR"
