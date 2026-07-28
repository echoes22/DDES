import ast
import json
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

emotion_centers_no_se = {
    'contentment': [0.75, 0.22],
    'awe': [-0.062, 0.48],
    'excitement': [0.792, 0.368],
    'amusement': [0.858, 0.674],
    'anger': [-0.666, 0.73],
    'disgust': [-0.896, 0.55],
    'fear': [-0.854, 0.68],
    'sadness': [-0.896, -0.424],
    'something else': [0.0, 0.0]
}

# 8-emotion VA anchor coordinates (matches inference_utils.EMOTION_COORDS order)
_EMOTION_VA_ANCHORS = torch.tensor([
    [ 0.858,  0.674],   # amusement
    [-0.062,  0.480],   # awe
    [ 0.750,  0.220],   # contentment
    [ 0.792,  0.368],   # excitement
    [-0.666,  0.730],   # anger
    [-0.896,  0.550],   # disgust
    [-0.854,  0.680],   # fear
    [-0.896, -0.424],   # sadness
], dtype=torch.float32)


def _expected_va_from_dist(emo_dist: torch.Tensor) -> torch.Tensor:
    """Expected VA = weighted sum of emotion VA anchors. Returns (2,) tensor."""
    return (_EMOTION_VA_ANCHORS * emo_dist.unsqueeze(1)).sum(0)


class CombinedEmotionDataset(Dataset):
    """Unified dataset loader for multiple emotion datasets."""

    def __init__(
        self,
        artemis_df=None,
        dvisa_df=None,
        transform=None,
        wiki_root=None,
        first_n=None,
        emoset_root=None,
        emoset_phase=None,
        emoart_root=None,
        emoart_json=None,
        eemo_bench_data=None,
        eemo_bench_root=None,
        wikiart_emo_tsv=None,
        debug=False,
    ):
        self.samples = []
        self.transform = transform
        self.wiki_root = wiki_root
        self.emoset_root = emoset_root
        self.emoart_root = emoart_root
        self.eemo_bench_root = eemo_bench_root
        self.debug = debug

        if artemis_df is not None:
            from .kde import sample_kde_at_emotion_Z_differentiable

            for _, row in artemis_df.iterrows():
                kde_norm = row['kde_norm']

                # Use pre-computed mapped_dist if present and valid.
                md = row.get('mapped_dist', None)
                emo_dist = None
                if md is not None:
                    try:
                        t = md if isinstance(md, torch.Tensor) else torch.tensor(md, dtype=torch.float32)
                        if t.shape == (8,) and torch.isfinite(t).all() and t.sum() > 0:
                            emo_dist = t.float()
                    except Exception:
                        pass

                # Fall back to sampling the KDE on-the-fly (requires only kde_norm).
                if emo_dist is None and kde_norm is not None:
                    try:
                        kde_t = kde_norm if isinstance(kde_norm, torch.Tensor) else torch.tensor(kde_norm, dtype=torch.float32)
                        log_dist = sample_kde_at_emotion_Z_differentiable(kde_t.squeeze())
                        emo_dist = log_dist.exp()
                    except Exception:
                        pass

                self.samples.append({
                    'type': 'artemis',
                    'painting': row['painting'],
                    'styles': row['art_style'] if isinstance(row['art_style'], list) else [row['art_style']],
                    'heatmap': kde_norm,
                    'va': None,
                    'emo_dist': emo_dist,
                })

        if dvisa_df is not None:
            for _, row in dvisa_df.iterrows():
                vad_raw = row['final_vad']
                if isinstance(vad_raw, str):
                    vad_raw = ast.literal_eval(vad_raw)
                va_01 = np.array(vad_raw[:2], dtype=np.float32)
                va_norm = (va_01 - 0.5) * 2  # [0,1] → [-1,1]
                self.samples.append({
                    'type': 'dvisa',
                    'painting': row['artwork'],
                    'styles': row['art_style'] if isinstance(row['art_style'], list) else [row['art_style']],
                    'heatmap': None,
                    'va': torch.tensor(va_norm, dtype=torch.float32),
                    'emo_dist': None,
                })

        if wikiart_emo_tsv is not None:
            try:
                df_emo = pd.read_csv(wikiart_emo_tsv)
            except Exception:
                df_emo = pd.read_csv(wikiart_emo_tsv, sep='\t')

            emo_cols = [
                'ImageOnly: agreeableness', 'ImageOnly: anger', 'ImageOnly: anticipation',
                'ImageOnly: arrogance', 'ImageOnly: disagreeableness', 'ImageOnly: disgust',
                'ImageOnly: fear', 'ImageOnly: gratitude', 'ImageOnly: happiness',
                'ImageOnly: humility', 'ImageOnly: love', 'ImageOnly: optimism',
                'ImageOnly: pessimism', 'ImageOnly: regret', 'ImageOnly: sadness',
                'ImageOnly: shame', 'ImageOnly: shyness', 'ImageOnly: surprise',
                'ImageOnly: trust', 'ImageOnly: neutral',
            ]

            df_emo[emo_cols] = df_emo[emo_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)

            for _, row in df_emo.iterrows():
                rel_path = row.get('full_path')
                if pd.isna(rel_path) or str(rel_path).strip() == '':
                    continue

                probs = torch.tensor(row[emo_cols].values.astype(np.float32), dtype=torch.float32)
                if probs.sum() > 0:
                    probs = probs / probs.sum()

                self.samples.append({
                    'type': 'wikiart_emo',
                    'painting': str(rel_path),
                    'heatmap': None,
                    'va': None,
                    'emo_dist': probs,
                })

        if eemo_bench_data is not None:
            with open(eemo_bench_data, 'r') as f:
                eemo_bench_entries = json.load(f)

            eemo_bench_entries = self.remove_duplicate_entries(eemo_bench_entries)

            for entry in eemo_bench_entries:
                valence_norm = (entry['valence'] - 1) / 4 - 1
                arousal_norm = (entry['arousal'] - 1) / 4 - 1

                try:
                    emo_dict = ast.literal_eval(entry['emotion_dict'])
                    top_3_emotions = sorted(emo_dict.items(), key=lambda item: item[1], reverse=True)[:3]
                except (ValueError, SyntaxError):
                    top_3_emotions = []

                self.samples.append({
                    'type': 'eemo-bench',
                    'painting': entry['image_url'],
                    'styles': None,
                    'heatmap': None,
                    'va': torch.tensor([valence_norm, arousal_norm], dtype=torch.float32),
                    'top_3': top_3_emotions,
                    'emo_dist': None,
                })

        if emoset_root and emoset_phase:
            split_json = os.path.join(emoset_root, f'{emoset_phase}.json')
            with open(split_json, 'r') as f:
                entries = json.load(f)

            for emo_label, image_id, _ in entries:
                self.samples.append({
                    'type': 'emoset',
                    'painting': image_id,
                    'styles': None,
                    'heatmap': None,
                    'va': torch.tensor(emotion_centers_no_se[emo_label], dtype=torch.float32),
                    'emo_dist': None,
                })

        if emoart_root and emoart_json:
            self.add_emoart_samples(emoart_root=emoart_root, annotation_json=emoart_json)

        if first_n is not None:
            random.shuffle(self.samples)
            self.samples = self.samples[:first_n]

    def __len__(self):
        return len(self.samples)

    def _pil_placeholder(self, size=(224, 224), color=(128, 128, 128)) -> Image.Image:
        return Image.new('RGB', size, color)

    def _wikiart_path(self, painting: str, styles: List[str]) -> str:
        style = styles[0]
        stem = Path(painting).stem
        return os.path.join(self.wiki_root, style, stem + '.jpg')

    def _try_read_image(self, path: str):
        try:
            return Image.open(path).convert('RGB'), None
        except Exception as exc:
            err = f'PIL open failed: {exc}'

        root, _ = os.path.splitext(path)
        for alt in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
            alt_path = root + alt
            if alt_path == path:
                continue
            try:
                return Image.open(alt_path).convert('RGB'), None
            except Exception:
                continue

        return None, err

    def _safe_load(self, full_path: str):
        img, err = self._try_read_image(full_path)
        if img is None:
            img = self._pil_placeholder()
            return img, full_path, False, err or 'unknown read error'
        return img, full_path, True, ''

    def __getitem__(self, idx):
        s = self.samples[idx]

        if s['type'] in ('artemis', 'dvisa'):
            img_path = self._wikiart_path(s['painting'], s['styles'])
        elif s['type'] == 'emoset':
            img_path = os.path.join(self.emoset_root, s['painting'])
        elif s['type'] == 'emoart':
            img_path = os.path.join(self.emoart_root, 'Images', s['styles'][0], s['painting'])
        elif s['type'] == 'eemo-bench':
            img_path = os.path.join(self.eemo_bench_root, s['painting'])
        elif s['type'] == 'wikiart_emo':
            img_path = os.path.join(self.wiki_root, s['painting'])
        else:
            raise ValueError(f'Unknown sample type: {s["type"]}')

        img, img_path, load_ok, load_err = self._safe_load(img_path)
        if self.transform:
            img = self.transform(img)

        out = {'image': img, 'image_path': img_path, 'type': s['type']}

        if s['type'] == 'artemis':
            out['heatmap'] = torch.tensor(s['heatmap'], dtype=torch.float32).unsqueeze(0)
            if s['emo_dist'] is not None:
                out['emo_dist'] = s['emo_dist']
                # Compute expected VA from emotion distribution (used as GT for evaluation)
                out['va'] = _expected_va_from_dist(s['emo_dist'])
            else:
                out['va'] = None
        elif s['type'] == 'wikiart_emo':
            out['heatmap'] = None
            out['va'] = None
            out['emo_dist'] = s['emo_dist']
        else:
            out['heatmap'] = None
            out['va'] = s['va']

        if s['type'] == 'eemo-bench':
            out['top_3'] = s.get('top_3', [])

        out['load_ok'] = load_ok
        out['load_err'] = load_err

        return out

    def add_emoart_samples(
        self,
        emoart_root: str,
        annotation_json: str,
        quadrant_center=0.5,
        exclude_ids: set | None = None,
        store_emotion_label: bool = False,
    ):
        with open(annotation_json, 'r') as f:
            entries = json.load(f)

        for rec in entries:
            img_rel = rec.get('image_path')
            if img_rel is None:
                continue

            img_rel = img_rel.replace('\\', '/')
            art_style = img_rel.split('/')[1]
            img_path = os.path.join(emoart_root, art_style, img_rel)
            fname = os.path.basename(img_rel)

            if exclude_ids and (fname in exclude_ids or img_path in exclude_ids):
                continue

            emo_2_bin = {
                'High': 1,
                'Low': -1,
                'Positive': 1,
                'Negative': -1,
            }
            v_bin = emo_2_bin.get(rec['description']['third_section']['emotional_valence'], 0)
            a_bin = emo_2_bin.get(rec['description']['third_section']['emotional_arousal_level'], 0)

            v = v_bin * quadrant_center
            a = a_bin * quadrant_center

            sample = {
                'type': 'emoart',
                'painting': fname,
                'styles': [art_style],
                'heatmap': None,
                'va': torch.tensor([v, a], dtype=torch.float32),
            }

            if store_emotion_label:
                emo_lab = rec.get('third_section', {}).get('dominant_emotion')
                if emo_lab is not None:
                    sample['emotion_label'] = emo_lab

            self.samples.append(sample)

    @staticmethod
    def remove_duplicate_entries(data_list):
        seen_urls = set()
        unique_data = []

        for entry in data_list:
            url = entry.get('image_url')
            if url and url not in seen_urls:
                unique_data.append(entry)
                seen_urls.add(url)

        return unique_data


def get_dataset_weights(dataset):
    types = [s['type'] for s in dataset.samples]
    unique = sorted(set(types))
    type_freq = {t: types.count(t) for t in unique}
    weights = np.array([1.0 / type_freq[t] for t in types], dtype=np.float64)
    return weights


def collate_fn_2(batch):
    images = torch.stack([b['image'] for b in batch], dim=0)
    paths = [b.get('image_path', '') for b in batch]

    heatmaps = []
    vas = []
    hm_mask = []
    va_mask = []
    emo_dists = []
    dists_mask = []

    for b in batch:
        if b['type'] == 'artemis':
            heatmaps.append(b['heatmap'])
            has_emo = 'emo_dist' in b and b['emo_dist'] is not None
            emo_dists.append(b['emo_dist'] if has_emo else torch.zeros(8))
            vas.append(b['va'] if (has_emo and b.get('va') is not None) else torch.zeros(2))
            hm_mask.append(True)
            va_mask.append(has_emo and b.get('va') is not None)
            dists_mask.append(has_emo)
        elif b['type'] == 'wikiart_emo':
            heatmaps.append(torch.zeros(1, 28, 28))
            vas.append(torch.zeros(2))
            emo_dists.append(b['emo_dist'])
            hm_mask.append(False)
            va_mask.append(False)
            dists_mask.append(True)
        else:
            heatmaps.append(torch.zeros(1, 28, 28))
            vas.append(b['va'])
            emo_dists.append(torch.zeros(8))
            hm_mask.append(False)
            va_mask.append(True)
            dists_mask.append(False)

    return {
        'image': images,
        'heatmap': torch.stack(heatmaps, dim=0),
        'va': torch.stack(vas, dim=0),
        'emotion_dist': torch.stack(emo_dists, dim=0),
        'hm_mask': torch.tensor(hm_mask),
        'va_mask': torch.tensor(va_mask),
        'dists_mask': torch.tensor(dists_mask),
        'image_path': paths,
    }


def collate_fn_debug(batch):
    result = collate_fn_2(batch)
    result['load_ok'] = torch.tensor([bool(b.get('load_ok', True)) for b in batch], dtype=torch.bool)
    result['load_err'] = [b.get('load_err', '') for b in batch]
    return result

#    train_loader = build_combined_emotion_dataloaders(
    #     data_root=config.DATA.IMAGE_PATH,
    #     datasets=['train'] + args.train_datasets,
    #     batch_size=config.DATA.BATCH_SIZE,
    #     num_workers=config.DATA.NUM_WORKERS,
    #     shuffle=True,
    #     distributed=use_distributed
    # )

def build_combined_emotion_dataloaders(config, train, test, val, val_b_size=None, balanced_sampling=False, debug=False):
    num_tasks = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    global_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    if val_b_size is None:
        val_b_size = config.DATA.BATCH_SIZE

    if balanced_sampling:
        w_train = get_dataset_weights(train)
        len_train = config.TRAIN.SAMPLES_PER_EPOCH if 0 < config.TRAIN.SAMPLES_PER_EPOCH <= len(w_train) else len(w_train)
        sampler_train = WeightedRandomSampler(weights=w_train, num_samples=len_train, replacement=True)

        w_test = get_dataset_weights(test)
        sampler_test = WeightedRandomSampler(weights=w_test, num_samples=len(w_test), replacement=True)

        w_val = get_dataset_weights(val)
        sampler_val = WeightedRandomSampler(weights=w_val, num_samples=len(w_val), replacement=True)
    elif dist.is_available() and dist.is_initialized():
        sampler_train = torch.utils.data.DistributedSampler(train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        sampler_val = torch.utils.data.distributed.DistributedSampler(val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        sampler_test = torch.utils.data.distributed.DistributedSampler(test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(train)
        sampler_val = torch.utils.data.SequentialSampler(val)
        sampler_test = torch.utils.data.SequentialSampler(test)

    collate_fn = collate_fn_debug if debug else collate_fn_2

    data_loader_train = torch.utils.data.DataLoader(
        train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        collate_fn=collate_fn,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        val,
        sampler=sampler_val,
        batch_size=val_b_size,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        collate_fn=collate_fn,
        drop_last=False,
    )

    data_loader_test = torch.utils.data.DataLoader(
        test,
        sampler=sampler_test,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        collate_fn=collate_fn,
        drop_last=False,
    )

    return {'train': data_loader_train, 'val': data_loader_val, 'test': data_loader_test}


def build_combined_emo_test_loader(config, test, debug=False):
    if dist.is_available() and dist.is_initialized():
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        sampler_test = torch.utils.data.distributed.DistributedSampler(test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(test)
    data_loader_test = torch.utils.data.DataLoader(
        test,
        sampler=sampler_test,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        collate_fn=collate_fn_debug if debug else collate_fn_2,
        drop_last=True,
    )
    return data_loader_test


def build_test_loader(config, test):
    if dist.is_available() and dist.is_initialized():
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        sampler_test = torch.utils.data.distributed.DistributedSampler(test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(test)
    return torch.utils.data.DataLoader(
        test,
        sampler=sampler_test,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )


def build_test_loader_no_sampler(config, test):
    return torch.utils.data.DataLoader(
        test,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
        shuffle=False,
    )


def get_base_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])


def rescale_0_1_to_minus1_1(x):
    return np.clip(2 * x - 1, -1, 1)


def parse_predicted_vad(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return None
    return x

def load_dataset_for_split(dataset_name, data_root, split='train', transform=None, dataset_dir=None):
    """
    Load a single dataset for a single split (train/val/test).

    Args:
        dataset_name: 'artemis', 'dvisa', 'emoset', 'emoart', 'wikiart_emo'
        data_root: Root directory where datasets are stored (used as fallback)
        split: 'train', 'val', or 'test'
        transform: Image transform to apply
        dataset_dir: Explicit directory for this dataset (overrides data_root subdirectory)

    Returns:
        Dataset parameters as dict to pass to CombinedEmotionDataset
    """
    params = {}

    if dataset_name.lower() == 'artemis':
        # Explicit dir takes priority; fall back to data_root/artemis
        search_dir = dataset_dir or os.path.join(data_root, 'artemis')
        # Try standard name first, then alternate naming used by the preprocessing script
        candidate_names = [
            f'{split}_kde_df.pkl',
            f'{split}_kde_N-28_No-abs_df.pkl',
            f'{split}_kde_N-28_df.pkl',
        ]
        df_path = None
        for name in candidate_names:
            candidate = os.path.join(search_dir, name)
            if os.path.exists(candidate):
                df_path = candidate
                break
        if df_path is not None:
            df = torch.load(df_path, weights_only=False)
            params['artemis_df'] = df
        else:
            print(f"Warning: Artemis {split} data not found in {search_dir} "
                  f"(tried: {', '.join(candidate_names)})")

    elif dataset_name.lower() == 'dvisa':
        search_dir = dataset_dir or os.path.join(data_root, 'dvisa')
        df_path = os.path.join(search_dir, f'{split}.pkl')
        if os.path.exists(df_path):
            df = pd.read_pickle(df_path)
            params['dvisa_df'] = df
        else:
            print(f"Warning: D-ViSA {split} data not found at {df_path}")

    elif dataset_name.lower() == 'emoset':
        emoset_root = dataset_dir or os.path.join(data_root, 'emoset')
        if os.path.exists(emoset_root):
            params['emoset_root'] = emoset_root
            params['emoset_phase'] = split
        else:
            print(f"Warning: EmoSet data not found at {emoset_root}")

    elif dataset_name.lower() == 'emoart':
        emoart_root = dataset_dir or os.path.join(data_root, 'emoart')
        emoart_json = os.path.join(emoart_root, 'Annotation.json')
        if os.path.exists(emoart_json):
            params['emoart_root'] = emoart_root
            params['emoart_json'] = emoart_json
        else:
            print(f"Warning: EmoArt data not found at {emoart_root}")

    elif dataset_name.lower() == 'wikiart_emo':
        if dataset_dir:
            wikiart_emo_tsv = os.path.join(dataset_dir, 'WikiArt-Emotions-All-fixed-5.csv')
        else:
            wikiart_emo_tsv = os.path.join(data_root, 'wikiart_emotion/WikiArt-Emotions-All-fixed-5.csv')
        if os.path.exists(wikiart_emo_tsv):
            params['wikiart_emo_tsv'] = wikiart_emo_tsv
        else:
            print(f"Warning: WikiArt-Emotion data not found at {wikiart_emo_tsv}")

    elif dataset_name.lower() in ('eemo', 'eemo_bench'):
        # EEMO-Bench has no train/val/test split — always loads the full benchmark set
        eemo_dir = dataset_dir or os.path.join(data_root, 'eemo-bench')
        eemo_json = os.path.join(eemo_dir, 'EEmo-Bench.json')
        eemo_images = os.path.join(eemo_dir, 'images')
        if os.path.exists(eemo_json) and os.path.exists(eemo_images):
            params['eemo_bench_data'] = eemo_json
            params['eemo_bench_root'] = eemo_images
        else:
            print(f"Warning: EEMO benchmark not found at {eemo_dir} "
                  f"(expected EEmo-Bench.json and images/)")

    # Add common parameters
    params['transform'] = transform
    params['wiki_root'] = data_root  # Used as fallback root for image paths

    return params


def build_emotion_dataloaders(config, data_root, dataset_names, transforms_train,
                              transforms_val, balanced_sampling=False, debug=False,
                              dataset_dirs=None, wiki_root=None):
    """
    Build train/val/test dataloaders by loading and combining specified datasets.

    Args:
        config: Config with DATA.BATCH_SIZE, DATA.NUM_WORKERS, DATA.PIN_MEMORY
        data_root: Root directory where all datasets reside (used as fallback)
        dataset_names: List of dataset names to load (e.g., ['artemis', 'dvisa'])
        transforms_train: Transform for training images
        transforms_val: Transform for validation/test images
        balanced_sampling: Whether to use balanced sampling
        debug: Whether to use debug collate function
        dataset_dirs: Optional dict mapping dataset name → explicit directory
                      e.g. {'artemis': '/path/to/artemis_kde', 'dvisa': '/path/to/dvisa'}
        wiki_root: Root directory for WikiArt images (overrides data_root for image loading)

    Returns:
        Dict with 'train', 'val', 'test' DataLoader objects
    """
    print("Loading datasets...")
    dataset_dirs = dataset_dirs or {}
    image_root = wiki_root or data_root

    # Load and combine datasets for each split
    train_params, val_params, test_params = {}, {}, {}

    for dataset_name in dataset_names:
        print(f"  Loading {dataset_name}...")
        ds_dir = dataset_dirs.get(dataset_name)

        # Get parameters for each split
        train_p = load_dataset_for_split(dataset_name, data_root, 'train', transforms_train, dataset_dir=ds_dir)
        val_p = load_dataset_for_split(dataset_name, data_root, 'val', transforms_val, dataset_dir=ds_dir)
        test_p = load_dataset_for_split(dataset_name, data_root, 'test', transforms_val, dataset_dir=ds_dir)

        # Merge parameters (combine multiple datasets)
        train_params.update(train_p)
        val_params.update(val_p)
        test_params.update(test_p)

    # Override wiki_root if provided
    for params in (train_params, val_params, test_params):
        params['wiki_root'] = image_root
    
    # Create combined datasets
    print("Creating combined datasets...")
    train_dataset = CombinedEmotionDataset(**train_params)
    val_dataset = CombinedEmotionDataset(**val_params)
    test_dataset = CombinedEmotionDataset(**test_params)
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    
    # Build dataloaders using the existing function
    dataloaders = build_combined_emotion_dataloaders(
        config=config,
        train=train_dataset,
        test=test_dataset,
        val=val_dataset,
        val_b_size=None,
        balanced_sampling=balanced_sampling,
        debug=debug
    )
    
    return dataloaders