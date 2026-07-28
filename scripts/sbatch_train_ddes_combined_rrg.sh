#!/bin/bash
#SBATCH --job-name=ddes_combined
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=rrg-jlalonde

# DDES-Net combined training: ArtEmis + D-ViSA + EmoSet → 28×28 KDE heatmap.
# LR fix: 1e-4 (was 5e-4) with 2-epoch warmup to prevent early overfitting.
# After training: evaluates on seen (artemis/dvisa/emoset) and unseen (eemo/wikiart_emo).

set -e

PROJECT_ROOT="${PROJECT_ROOT:-/home/ember118/links/projects/def-jlalonde/ember118/public_emotion_pred}"
ARTEMIS_DIR="${ARTEMIS_DIR:-/home/ember118/links/scratch/emotion_pred/custom_ds/artemisV2/full_combined/path_fix_remove_prob/no-wikiart-emotion/emo_and_sent}"
DVISA_DIR="${DVISA_DIR:-/home/ember118/links/scratch/emotion_pred/custom_ds/dvisa}"
EMOSET_DIR="${EMOSET_DIR:-/home/ember118/links/scratch/emoset}"
EEMO_DIR="${EEMO_DIR:-/home/ember118/links/scratch/eemo-bench}"
WIKIART_EMO_DIR="${WIKIART_EMO_DIR:-/home/ember118/links/scratch/wikiart_emotion/WikiArt-Emotions}"
WIKI_ROOT="${WIKI_ROOT:-/home/ember118/links/scratch/wiki_wrapper/wikiart}"
CKPT_DIR="${CKPT_DIR:-/home/ember118/links/scratch/emotion_pred/public_repl/ckpt/ddes_unet_combined}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/runs/ddes_unet_combined}"
RESUME="${RESUME:-}"

module load scipy-stack
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.9
module load python/3.10
source /home/ember118/ENV/emo/bin/activate

cd "$PROJECT_ROOT/.."
export PYTHONPATH="$PROJECT_ROOT/..:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== DDES-Net Combined Training (rrg-jlalonde, half-H100) ==="
echo "ArtEmis    : $ARTEMIS_DIR"
echo "D-ViSA     : $DVISA_DIR"
echo "EmoSet     : $EMOSET_DIR"
echo "EEMO       : $EEMO_DIR"
echo "WikiArt-Emo: $WIKIART_EMO_DIR"
echo "WikiArt img: $WIKI_ROOT"
echo "Checkpoint : $CKPT_DIR"
echo "Resume     : ${RESUME:-none}"
echo ""

RESUME_ARG=""
if [ -n "$RESUME" ]; then
    RESUME_ARG="--resume $RESUME"
fi

python -m public_emotion_pred.main \
    --model convnext_unet \
    --epochs 40 \
    --batch-size 64 \
    --lr 1e-4 \
    --optimizer adam \
    --scheduler cosine_warmup \
    --warmup-epochs 2 \
    --gradient-accumulation 1 \
    --use-amp \
    --num-workers 8 \
    --train-datasets artemis dvisa emoset \
    --artemis-dir "$ARTEMIS_DIR" \
    --dvisa-dir "$DVISA_DIR" \
    --emoset-dir "$EMOSET_DIR" \
    --eemo-dir "$EEMO_DIR" \
    --wikiart-emo-dir "$WIKIART_EMO_DIR" \
    --wiki-root "$WIKI_ROOT" \
    --benchmark-datasets eemo wikiart_emo \
    --checkpoint-dir "$CKPT_DIR" \
    --logdir "$LOG_DIR" \
    --save-freq 40 \
    $RESUME_ARG

echo "=== Done ==="
