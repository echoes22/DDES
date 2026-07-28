#!/bin/bash
#SBATCH --job-name=ddes_des
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=rrg-jlalonde

# DES-Net training: ConvNeXt-Small → (valence, arousal) regression.
# Dataset: D-ViSA VA annotations.
# Uses rrg-jlalonde account + half-H100 MIG slice (3g.40gb, 40 GB VRAM).
#
# Usage:
#   sbatch scripts/sbatch_train_des_rrg.sh

set -e

PROJECT_ROOT="${PROJECT_ROOT:-/home/ember118/links/projects/def-jlalonde/ember118/public_emotion_pred}"
DVISA_DIR="${DVISA_DIR:-/home/ember118/links/scratch/emotion_pred/custom_ds/dvisa}"
WIKI_ROOT="${WIKI_ROOT:-/home/ember118/links/scratch/wiki_wrapper/wikiart}"
CKPT_DIR="${CKPT_DIR:-/home/ember118/links/scratch/emotion_pred/public_repl/ckpt/des_net}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/runs/des_net}"
RESUME="${RESUME:-/home/ember118/links/scratch/emotion_pred/public_repl/ckpt/des_net/best/checkpoint_epoch_009.pt}"

module load scipy-stack
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.9
module load python/3.10
source /home/ember118/ENV/emo/bin/activate

cd "$PROJECT_ROOT/.."
export PYTHONPATH="$PROJECT_ROOT/..:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== DES-Net Training (rrg-jlalonde, half-H100) ==="
echo "D-ViSA dir  : $DVISA_DIR"
echo "WikiArt root: $WIKI_ROOT"
echo "Checkpoint  : $CKPT_DIR"
echo ""

python -m public_emotion_pred.main \
    --model convnext_2 \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-4 \
    --optimizer adam \
    --scheduler cosine_warmup \
    --warmup-epochs 2 \
    --gradient-accumulation 1 \
    --use-amp \
    --num-workers 8 \
    --train-datasets dvisa \
    --val-datasets dvisa \
    --dvisa-dir "$DVISA_DIR" \
    --wiki-root "$WIKI_ROOT" \
    --checkpoint-dir "$CKPT_DIR" \
    --logdir "$LOG_DIR" \
    --resume "$RESUME" \
    --save-freq 50

echo "=== Done ==="
