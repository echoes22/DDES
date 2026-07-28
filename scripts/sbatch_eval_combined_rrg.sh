#!/bin/bash
#SBATCH --job-name=eval_combined
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=rrg-jlalonde

# Eval-only: load best checkpoint from CKPT_DIR and run per-dataset + benchmark evaluation.
# Usage:
#   MODEL=convnext_unet CKPT_DIR=.../ddes_unet_combined sbatch scripts/sbatch_eval_combined_rrg.sh

set -e

PROJECT_ROOT="${PROJECT_ROOT:-/home/ember118/links/projects/def-jlalonde/ember118/public_emotion_pred}"
MODEL="${MODEL:-convnext_unet}"
ARTEMIS_DIR="${ARTEMIS_DIR:-/home/ember118/links/scratch/emotion_pred/custom_ds/artemisV2/full_combined/path_fix_remove_prob/no-wikiart-emotion/emo_and_sent}"
DVISA_DIR="${DVISA_DIR:-/home/ember118/links/scratch/emotion_pred/custom_ds/dvisa}"
EMOSET_DIR="${EMOSET_DIR:-/home/ember118/links/scratch/emoset}"
EEMO_DIR="${EEMO_DIR:-/home/ember118/links/scratch/eemo-bench}"
WIKIART_EMO_DIR="${WIKIART_EMO_DIR:-/home/ember118/links/scratch/wikiart_emotion/WikiArt-Emotions}"
WIKI_ROOT="${WIKI_ROOT:-/home/ember118/links/scratch/wiki_wrapper/wikiart}"
CKPT_DIR="${CKPT_DIR:-/home/ember118/links/scratch/emotion_pred/public_repl/ckpt/ddes_unet_combined}"

module load scipy-stack
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.9
module load python/3.10
source /home/ember118/ENV/emo/bin/activate

cd "$PROJECT_ROOT/.."
export PYTHONPATH="$PROJECT_ROOT/..:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== Eval-only: $MODEL ==="
echo "Checkpoint : $CKPT_DIR"

python -m public_emotion_pred.main \
    --model "$MODEL" \
    --eval-only \
    --train-datasets artemis dvisa emoset \
    --artemis-dir "$ARTEMIS_DIR" \
    --dvisa-dir "$DVISA_DIR" \
    --emoset-dir "$EMOSET_DIR" \
    --eemo-dir "$EEMO_DIR" \
    --wikiart-emo-dir "$WIKIART_EMO_DIR" \
    --wiki-root "$WIKI_ROOT" \
    --benchmark-datasets eemo wikiart_emo \
    --checkpoint-dir "$CKPT_DIR" \
    --batch-size 64 \
    --num-workers 8

echo "=== Done ==="
