#!/bin/bash
#SBATCH --job-name=ddes_unet
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=def-jlalonde

# DDES-Net training: ConvNeXt-Small + 2-stage decoder → 28×28 KL heatmap.
# Dataset: ArtEmis V2 KDE heatmaps (--train-datasets artemis).
#
# Usage:
#   sbatch scripts/sbatch_train_ddes.sh
#
# Override paths via environment variables before submitting:
#   ARTEMIS_DIR=/path/to/artemis sbatch scripts/sbatch_train_ddes.sh

set -e

PROJECT_ROOT="${PROJECT_ROOT:-/home/ember118/links/projects/def-jlalonde/ember118/public_emotion_pred}"
ARTEMIS_DIR="${ARTEMIS_DIR:-/home/ember118/links/scratch/emotion_pred/custom_ds/artemisV2/full_combined/path_fix_remove_prob/no-wikiart-emotion/emo_and_sent}"
WIKI_ROOT="${WIKI_ROOT:-/home/ember118/links/scratch/wiki_wrapper/wikiart}"
CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/ckpt/ddes_unet}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/runs/ddes_unet}"
RESUME_CKPT="${RESUME_CKPT:-}"

module load scipy-stack
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.9
module load python/3.10
source /home/ember118/ENV/emo/bin/activate

cd "$PROJECT_ROOT/.."
export PYTHONPATH="$PROJECT_ROOT/..:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== DDES-Net Training ==="
echo "ArtEmis dir : $ARTEMIS_DIR"
echo "WikiArt root: $WIKI_ROOT"
echo "Checkpoint  : $CKPT_DIR"
echo "Resume      : ${RESUME_CKPT:-none}"
echo ""

RESUME_ARG=""
if [ -n "$RESUME_CKPT" ]; then
    RESUME_ARG="--resume $RESUME_CKPT"
fi

python -m public_emotion_pred.main \
    --model convnext_unet \
    --epochs 50 \
    --batch-size 64 \
    --lr 5e-4 \
    --optimizer adam \
    --scheduler cosine_warmup \
    --warmup-epochs 5 \
    --gradient-accumulation 1 \
    --use-amp \
    --num-workers 8 \
    --train-datasets artemis \
    --val-datasets artemis \
    --artemis-dir "$ARTEMIS_DIR" \
    --wiki-root "$WIKI_ROOT" \
    --checkpoint-dir "$CKPT_DIR" \
    --logdir "$LOG_DIR" \
    --save-freq 5 \
    $RESUME_ARG

echo "=== Done ==="
