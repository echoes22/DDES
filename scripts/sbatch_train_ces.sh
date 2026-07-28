#!/bin/bash
#SBATCH --job-name=ddes_ces
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=def-jlalonde

# CES-Net training: ConvNeXt-Small → 8-class log-softmax emotion distribution.
# Dataset: ArtEmis V2 mapped_dist (8-class CES supervision).
#
# Usage:
#   sbatch scripts/sbatch_train_ces.sh

set -e

PROJECT_ROOT="${PROJECT_ROOT:-/home/ember118/links/projects/def-jlalonde/ember118/public_emotion_pred}"
ARTEMIS_DIR="${ARTEMIS_DIR:-/home/ember118/links/scratch/emotion_pred/custom_ds/artemisV2/full_combined/path_fix_remove_prob/no-wikiart-emotion/emo_and_sent}"
WIKI_ROOT="${WIKI_ROOT:-/home/ember118/links/scratch/wiki_wrapper/wikiart}"
CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/ckpt/ces_net}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/runs/ces_net}"

module load scipy-stack
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.9
module load python/3.10
source /home/ember118/ENV/emo/bin/activate

cd "$PROJECT_ROOT/.."
export PYTHONPATH="$PROJECT_ROOT/..:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== CES-Net Training ==="
echo "ArtEmis dir : $ARTEMIS_DIR"
echo "WikiArt root: $WIKI_ROOT"
echo "Checkpoint  : $CKPT_DIR"
echo ""

python -m public_emotion_pred.main \
    --model convnext_8 \
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
    --save-freq 5

echo "=== Done ==="
