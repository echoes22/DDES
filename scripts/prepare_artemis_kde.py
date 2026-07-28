#!/usr/bin/env python3
"""
Preprocess ArtEmis V2 annotations into per-painting KDE heatmaps.

For each painting:
1. Collect all annotator emotion labels.
2. Map each label to its V-A anchor coordinate using emotion_centers_no_se
   (excludes 'something else' — those annotations are dropped).
3. Add tiny Gaussian jitter (σ=1e-3) to avoid degenerate KDE when all points coincide.
4. Fit scipy.stats.gaussian_kde with Scott's bandwidth rule.
5. Evaluate on a 28×28 grid spanning [-1,1]×[-1,1].
6. Normalise to sum 1 → stored as numpy array in column 'kde_norm'.

Paintings with fewer than MIN_ANNOTATIONS valid labels are skipped.

Output: three pickle files (train/val/test) in --output-dir.

Usage:
    python scripts/prepare_artemis_kde.py \
        --input /path/to/artemis_v2.csv \
        --output-dir /path/to/output/artemis \
        [--grid-size 28] [--min-annotations 3] [--seed 42]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


# V-A anchors for 8 meaningful ArtEmis emotions (excludes 'something else')
EMOTION_CENTERS = {
    'amusement':   np.array([ 0.858,  0.674]),
    'awe':         np.array([-0.062,  0.480]),
    'contentment': np.array([ 0.750,  0.220]),
    'excitement':  np.array([ 0.792,  0.368]),
    'anger':       np.array([-0.666,  0.730]),
    'disgust':     np.array([-0.896,  0.550]),
    'fear':        np.array([-0.854,  0.680]),
    'sadness':     np.array([-0.896, -0.424]),
}

JITTER_SIGMA = 1e-3
MIN_ANNOTATIONS = 3


def build_kde_grid(va_points, grid_size=28):
    """
    Fit KDE to a set of V-A points and evaluate on an N×N grid.

    Args:
        va_points: (M, 2) array of [valence, arousal] values.
        grid_size: N for N×N output grid.

    Returns:
        (grid_size, grid_size) float32 array normalised to sum 1.
    """
    # Add jitter to avoid singular covariance matrix when all points are identical
    rng = np.random.default_rng(seed=0)
    va_jittered = va_points + rng.normal(0, JITTER_SIGMA, va_points.shape)

    kde = gaussian_kde(va_jittered.T)  # kde expects (2, N) input

    v_grid = np.linspace(-1, 1, grid_size)
    a_grid = np.linspace(-1, 1, grid_size)
    VV, AA = np.meshgrid(v_grid, a_grid)  # VV[row,col]=valence, AA[row,col]=arousal
    grid_pts = np.stack([VV.ravel(), AA.ravel()], axis=0)  # (2, grid_size²)

    density = kde(grid_pts).reshape(grid_size, grid_size)
    density = density.astype(np.float32)
    total = density.sum()
    if total > 0:
        density /= total
    return density


def process_artemis(df, grid_size, min_annotations, split_tag):
    """
    Process a split DataFrame of ArtEmis annotations into KDE heatmaps.

    Expected columns: 'painting', 'art_style', 'emotion' (one row per annotation).

    Returns a DataFrame with one row per painting and a 'kde_norm' column.
    """
    paintings = df['painting'].unique()
    records = []

    skipped = 0
    for painting in paintings:
        rows = df[df['painting'] == painting]
        art_style = rows['art_style'].iloc[0]

        # Map emotion labels to V-A coordinates
        va_list = []
        for emo in rows['emotion']:
            emo_key = str(emo).strip().lower()
            if emo_key in EMOTION_CENTERS:
                va_list.append(EMOTION_CENTERS[emo_key].copy())
            # 'something else' → skip

        if len(va_list) < min_annotations:
            skipped += 1
            continue

        va_pts = np.stack(va_list, axis=0)  # (M, 2)
        kde_grid = build_kde_grid(va_pts, grid_size=grid_size)

        records.append({
            'painting':  painting,
            'art_style': art_style,
            'kde_norm':  kde_grid,
            'n_valid':   len(va_list),
        })

    print(f"  [{split_tag}] {len(records)} paintings processed, {skipped} skipped (< {min_annotations} valid annotations)")
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description='Prepare ArtEmis V2 KDE heatmaps')
    parser.add_argument('--input', required=True,
                        help='Path to ArtEmis V2 CSV file')
    parser.add_argument('--output-dir', required=True,
                        help='Directory to write train/val/test pkl files')
    parser.add_argument('--grid-size', type=int, default=28,
                        help='Heatmap grid size N (default: 28)')
    parser.add_argument('--min-annotations', type=int, default=MIN_ANNOTATIONS,
                        help='Minimum valid annotations per painting (default: 3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--val-frac', type=float, default=0.1,
                        help='Fraction of paintings for validation (default: 0.1)')
    parser.add_argument('--test-frac', type=float, default=0.1,
                        help='Fraction of paintings for test (default: 0.1)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading ArtEmis from {args.input} ...")
    df = pd.read_csv(args.input)
    print(f"  {len(df)} annotations, {df['painting'].nunique()} unique paintings")

    # Reproducible train/val/test split on paintings
    rng = np.random.default_rng(args.seed)
    paintings = df['painting'].unique()
    rng.shuffle(paintings)
    n = len(paintings)
    n_val  = int(n * args.val_frac)
    n_test = int(n * args.test_frac)
    n_train = n - n_val - n_test

    train_paintings = set(paintings[:n_train])
    val_paintings   = set(paintings[n_train:n_train + n_val])
    test_paintings  = set(paintings[n_train + n_val:])

    splits = {
        'train': df[df['painting'].isin(train_paintings)],
        'val':   df[df['painting'].isin(val_paintings)],
        'test':  df[df['painting'].isin(test_paintings)],
    }

    for split_tag, split_df in splits.items():
        print(f"\nProcessing {split_tag} split ({len(split_df['painting'].unique())} paintings) ...")
        result_df = process_artemis(
            split_df,
            grid_size=args.grid_size,
            min_annotations=args.min_annotations,
            split_tag=split_tag,
        )
        out_path = os.path.join(args.output_dir, f'{split_tag}_kde_df.pkl')
        result_df.to_pickle(out_path)
        print(f"  Saved {len(result_df)} records → {out_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
