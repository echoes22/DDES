#!/bin/bash
#SBATCH --job-name=ddes_smoke
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --account=def-jlalonde

# Quick smoke test: 1 epoch, all 3 models.
# Verifies the pipeline runs end-to-end without errors.
#
# Usage:
#   sbatch scripts/sbatch_smoke_test.sh

set -e

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-/home/ember118/links/projects/def-jlalonde/ember118/public_emotion_pred}"
ARTEMIS_DIR="${ARTEMIS_DIR:-/home/ember118/links/scratch/emotion_pred/custom_ds/artemisV2/full_combined/path_fix_remove_prob/no-wikiart-emotion/emo_and_sent}"
DVISA_DIR="${DVISA_DIR:-/home/ember118/links/scratch/emotion_pred/custom_ds/dvisa}"
WIKI_ROOT="${WIKI_ROOT:-/home/ember118/links/scratch/wiki_wrapper/wikiart}"
CKPT_BASE="${CKPT_BASE:-$PROJECT_ROOT/ckpt/smoke}"
LOG_BASE="${LOG_BASE:-$PROJECT_ROOT/runs/smoke}"

# --- Environment -------------------------------------------------------------
# Load scipy-stack first (no version) so LMOD can swap it to the Python 3.10
# compatible build when python/3.10 is loaded — mirroring activate.sh.
module load scipy-stack
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.9
module load python/3.10
source /home/ember118/ENV/emo/bin/activate

cd "$PROJECT_ROOT/.."
export PYTHONPATH="$PROJECT_ROOT/..:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== DDES Smoke Test ==="
echo "Project root : $PROJECT_ROOT"
echo "ArtEmis dir  : $ARTEMIS_DIR"
echo "D-ViSA dir   : $DVISA_DIR"
echo "WikiArt root : $WIKI_ROOT"
echo ""

BASE_ARGS=(
    --epochs 1
    --batch-size 64
    --lr 5e-4
    --optimizer adam
    --num-workers 2
    --scheduler none
    --artemis-dir "$ARTEMIS_DIR"
    --dvisa-dir "$DVISA_DIR"
    --wiki-root "$WIKI_ROOT"
)

# --- DDES-Net (convnext_unet on ArtEmis heatmaps) ---
echo ">>> Smoke: DDES-Net (convnext_unet)"
python -m public_emotion_pred.main \
    --model convnext_unet \
    --train-datasets artemis \
    --val-datasets artemis \
    --checkpoint-dir "$CKPT_BASE/convnext_unet" \
    --logdir "$LOG_BASE/convnext_unet" \
    "${BASE_ARGS[@]}"

# --- CES-Net (convnext_8 on ArtEmis mapped_dist) ---
echo ">>> Smoke: CES-Net (convnext_8)"
python -m public_emotion_pred.main \
    --model convnext_8 \
    --train-datasets artemis \
    --val-datasets artemis \
    --checkpoint-dir "$CKPT_BASE/convnext_8" \
    --logdir "$LOG_BASE/convnext_8" \
    "${BASE_ARGS[@]}"

# --- DES-Net (convnext_2 on D-ViSA VA) ---
echo ">>> Smoke: DES-Net (convnext_2)"
python -m public_emotion_pred.main \
    --model convnext_2 \
    --train-datasets dvisa \
    --val-datasets dvisa \
    --checkpoint-dir "$CKPT_BASE/convnext_2" \
    --logdir "$LOG_BASE/convnext_2" \
    "${BASE_ARGS[@]}"

echo "=== Smoke test complete ==="
