#!/usr/bin/env python3
"""
Main training script for emotion prediction models.

Supports:
- Models: convnext_unet, convnext_8, convnext_2
- Datasets: Artemis, D-ViSA, EmoSet, EmoArt, WikiArt-Emotion
- Multi-GPU training via torch.distributed

Usage:
    python -m public_emotion_pred.main --model convnext_unet --epochs 50 --batch-size 32
"""

import argparse
import glob
import json
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from datetime import datetime
from pathlib import Path

from .config import Config, get_config, update_config, get_config_with_file, save_config_to_file
from .models import get_model
from .dataset import CombinedEmotionDataset, build_emotion_dataloaders, collate_fn_2
from .train import single_epoch_train_multi_ds_any_model, evaluate_on_multi_dataset_any_model
from .lr_scheduler import build_scheduler


def setup_distributed():
    """Initialize distributed training."""
    if not dist.is_available():
        print("Distributed training not available")
        return False
    
    try:
        dist.init_process_group(backend='nccl')
        print(f"Distributed training initialized: rank={dist.get_rank()}, world_size={dist.get_world_size()}")
        return True
    except Exception as e:
        print(f"Failed to initialize distributed training: {e}")
        return False


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint(model, optimizer, scheduler, epoch, loss, checkpoint_dir='./ckpt', keep_only_latest=False):
    """Save model checkpoint. If keep_only_latest, delete previous checkpoints in the dir first."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    if keep_only_latest:
        for old in glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt')):
            os.remove(old)

    # Extract model from DDP wrapper if needed
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.__dict__ if scheduler else None,
        'loss': loss,
    }

    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:03d}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load model checkpoint. Returns (epoch, loss)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.__dict__.update(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    print(f"Checkpoint loaded from {checkpoint_path} (epoch {epoch}, loss={loss:.4f})")

    return epoch, loss


def main():
    parser = argparse.ArgumentParser(description='Train emotion prediction models')
    
    # Model and dataset
    parser.add_argument('--model', type=str, default='convnext_unet',
                        choices=['convnext_unet', 'convnext_8', 'convnext_2'],
                        help='Model architecture')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to pretrained checkpoint')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze backbone during training')
    
    # Dataset
    parser.add_argument('--train-datasets', type=str, nargs='+',
                        default=['artemis', 'dvisa', 'emoset', 'emoart'],
                        help='Training datasets')
    parser.add_argument('--val-datasets', type=str, nargs='+',
                        default=['artemis'],
                        help='Validation datasets')
    parser.add_argument('--test-datasets', type=str, nargs='+',
                        default=['artemis'],
                        help='Test datasets')
    parser.add_argument('--data-root', type=str, default='./data',
                        help='Path to data root directory (used for all datasets if explicit dirs not specified)')
    parser.add_argument('--artemis-dir', type=str, default=None,
                        help='Directory containing ArtEmis train/val/test_kde_df.pkl files')
    parser.add_argument('--dvisa-dir', type=str, default=None,
                        help='Directory containing D-ViSA train/val/test.pkl files')
    parser.add_argument('--emoset-dir', type=str, default=None,
                        help='EmoSet root directory')
    parser.add_argument('--emoart-dir', type=str, default=None,
                        help='EmoArt root directory containing Annotation.json')
    parser.add_argument('--wikiart-emo-dir', type=str, default=None,
                        help='WikiArt-Emotion directory containing WikiArt-Emotions-All-fixed-5.csv')
    parser.add_argument('--eemo-dir', type=str, default=None,
                        help='EEMO benchmark root directory (contains EEmo-Bench.json and images/)')
    parser.add_argument('--wiki-root', type=str, default=None,
                        help='WikiArt image root directory (for loading artwork images)')
    parser.add_argument('--benchmark-datasets', type=str, nargs='*', default=[],
                        help='Unseen benchmark datasets to evaluate after training (e.g. eemo wikiart_emo)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='cosine_warmup',
                        choices=['cosine', 'cosine_warmup', 'cosine_restarts', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--gradient-accumulation', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--use-amp', action='store_true',
                        help='Use automatic mixed precision')
    
    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='./ckpt',
                        help='Directory for saving checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip training; load best checkpoint from --checkpoint-dir and evaluate')
    parser.add_argument('--save-freq', type=int, default=1,
                        help='Save checkpoint every N epochs')
    
    # Other
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--logdir', type=str, default='./runs',
                        help='TensorBoard log directory')
    parser.add_argument('--local-rank', type=int, default=-1,
                        help='Local rank for distributed training')
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (YAML or JSON) - CLI args override config file')
    parser.add_argument('--save-config', type=str, default=None,
                        help='Save final config to file after training starts')
    
    args = parser.parse_args()
    
    # Load config from file and merge with CLI args
    config = get_config_with_file(config_file=args.config, args=args)
    
    # Setup random seeds
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Config loaded: {f'from {args.config}' if args.config else 'using defaults'}")
    
    # Setup distributed training
    use_distributed = setup_distributed() if torch.cuda.device_count() > 1 else False
    rank = dist.get_rank() if use_distributed else 0
    world_size = dist.get_world_size() if use_distributed else 1
    
    # Save config to file if requested (rank 0 only)
    if rank == 0 and args.save_config:
        os.makedirs(os.path.dirname(args.save_config) or '.', exist_ok=True)
        save_config_to_file(config, args.save_config)
        print(f"Config saved to: {args.save_config}")
    
    
    if rank != 0:
        # Only rank 0 prints
        sys.stdout = open(os.devnull, 'w')
    
    # Setup image transforms
    from torchvision import transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(0.1),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2),
                               saturation=(0.8, 1.2), hue=(-0.1, 0.1)),
        transforms.ToTensor(),
        normalize,
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Resolve data root: --data-root CLI arg > config default
    data_root = args.data_root
    if getattr(config.DATA, 'PATHS', None):
        # Per-dataset roots can be specified in config.DATA.PATHS
        print(f"Per-dataset path overrides available in config.DATA.PATHS")

    print(f"Data root: {data_root}")
    print(f"Training on: {args.train_datasets}")
    print(f"Validating on: {args.val_datasets}")

    # Build per-dataset directory overrides from explicit CLI args
    dataset_dirs = {}
    if args.artemis_dir:
        dataset_dirs['artemis'] = args.artemis_dir
    if args.dvisa_dir:
        dataset_dirs['dvisa'] = args.dvisa_dir
    if args.emoset_dir:
        dataset_dirs['emoset'] = args.emoset_dir
    if args.emoart_dir:
        dataset_dirs['emoart'] = args.emoart_dir
    if args.wikiart_emo_dir:
        dataset_dirs['wikiart_emo'] = args.wikiart_emo_dir
    if args.eemo_dir:
        dataset_dirs['eemo'] = args.eemo_dir

    data_loaders = build_emotion_dataloaders(
        config=config,
        data_root=data_root,
        dataset_names=args.train_datasets,
        transforms_train=transform_train,
        transforms_val=transform_val,
        balanced_sampling=False,
        debug=False,
        dataset_dirs=dataset_dirs,
        wiki_root=args.wiki_root,
    )

    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    test_loader = data_loaders['test']
    
    # Build model
    print(f"Building model: {config.MODEL.TYPE}")
    model = get_model(
        model_type=config.MODEL.TYPE,
        ckpt=config.MODEL.RESUME or None,
        freeze_backbone=getattr(args, 'freeze_backbone', False),
    )
    model = model.to(device)
    
    # Wrap with DDP if distributed
    if use_distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    
    # Build optimizer
    print(f"Building optimizer: {args.optimizer}")
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.TRAIN.BASE_LR, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.TRAIN.BASE_LR, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.TRAIN.BASE_LR, momentum=0.9, weight_decay=args.weight_decay)
    
    # Build scheduler (per-step, matching original training setup)
    scheduler = None
    if config.TRAIN.LR_SCHEDULER.NAME != 'none':
        print(f"Building scheduler: {config.TRAIN.LR_SCHEDULER.NAME}")
        scheduler = build_scheduler(
            optimizer,
            scheduler_type=config.TRAIN.LR_SCHEDULER.NAME,
            total_epochs=config.TRAIN.EPOCHS,
            warmup_epochs=config.TRAIN.WARMUP_EPOCHS,
            n_iter_per_epoch=len(train_loader),
            min_lr=config.TRAIN.MIN_LR,
            warmup_lr=config.TRAIN.WARMUP_LR,
        )
    
    # Setup tensorboard
    log_dir = os.path.join(config.TENSORBOARD_PATH, datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir) if rank == 0 else None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    best_epoch = -1
    if args.resume:
        resumed_epoch, resumed_loss = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = resumed_epoch + 1
        best_val_loss = resumed_loss
        best_epoch = resumed_epoch
        print(f"Resuming from epoch {resumed_epoch}, best_val_loss={resumed_loss:.4f}")

    # Training loop (skipped when --eval-only)
    metrics_log = []
    if args.eval_only:
        candidates = glob.glob(os.path.join(config.OUTPUT, 'best', 'checkpoint_epoch_*.pt'))
        if candidates:
            best_ckpt_path = sorted(candidates)[0]
            best_epoch, best_val_loss = load_checkpoint(best_ckpt_path, model)
            print(f"Eval-only: loaded {best_ckpt_path}")
        else:
            print("Eval-only: no best checkpoint found, evaluating with random weights")
            best_epoch = -1
    else:
        print(f"Starting training for {config.TRAIN.EPOCHS} epochs")

    for epoch in range(start_epoch, 0 if args.eval_only else config.TRAIN.EPOCHS):
        # Train
        train_loss = single_epoch_train_multi_ds_any_model(
            train_loader,
            model,
            optimizer,
            device,
            model_type=config.MODEL.TYPE,
            writer=writer,
            epoch=epoch,
            use_amp=getattr(args, 'use_amp', False),
            gradient_accumulation_steps=args.gradient_accumulation,
            lr_scheduler=scheduler,
        )

        # Validate
        val_results = evaluate_on_multi_dataset_any_model(
            val_loader,
            model,
            device,
            model_type=config.MODEL.TYPE,
            writer=writer,
            epoch=epoch,
            tag='val'
        )

        val_loss = val_results.get('val_loss', float('inf'))

        # Log metrics
        if writer:
            writer.add_scalar('Loss/train_epoch', train_loss, epoch)
            writer.add_scalar('Loss/val_loss', val_loss, epoch)

        # Save periodic checkpoint (keep only the latest to avoid disk bloat)
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, config.OUTPUT,
                            keep_only_latest=True)

        # Save best checkpoint by val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss,
                          os.path.join(config.OUTPUT, 'best'), keep_only_latest=True)

        epoch_record = {'epoch': epoch, 'train_loss': train_loss, **val_results}
        metrics_log.append(epoch_record)

        extra = ""
        if 'top1_acc' in val_results:
            extra += (f", top1_acc={val_results['top1_acc']:.4f}"
                      f", kendall_tau={val_results.get('kendall_tau', float('nan')):.4f}")
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"pearson_v={val_results.get('pearson_v', float('nan')):.4f}, "
              f"pearson_a={val_results.get('pearson_a', float('nan')):.4f}, "
              f"rmse2d={val_results.get('rmse2d', float('nan')):.4f}{extra}")

    if not args.eval_only:
        # Save per-epoch metrics log
        metrics_log_path = os.path.join(config.OUTPUT, 'metrics_log.json')
        os.makedirs(config.OUTPUT, exist_ok=True)
        with open(metrics_log_path, 'w') as f:
            json.dump(metrics_log, f, indent=2)
        print(f"Per-epoch metrics saved to {metrics_log_path}")
        print(f"Best epoch: {best_epoch} (val_loss={best_val_loss:.4f})")

        # Load best checkpoint before test evaluation
        best_ckpt = os.path.join(config.OUTPUT, 'best', f'checkpoint_epoch_{best_epoch:03d}.pt')
        if not os.path.exists(best_ckpt):
            candidates = glob.glob(os.path.join(config.OUTPUT, 'best', 'checkpoint_epoch_*.pt'))
            best_ckpt = candidates[0] if candidates else None
        if best_ckpt and os.path.exists(best_ckpt):
            print(f"Loading best checkpoint: {best_ckpt}")
            load_checkpoint(best_ckpt, model)
        else:
            print(f"Best checkpoint not found, evaluating with final weights")

    # Per-dataset test evaluation (seen training datasets + unseen benchmarks)
    from .dataset import load_dataset_for_split, CombinedEmotionDataset, collate_fn_2
    from torch.utils.data import DataLoader as _DataLoader

    def _build_test_loader(ds_name):
        ds_dir = dataset_dirs.get(ds_name)
        image_root = args.wiki_root or data_root
        params = load_dataset_for_split(ds_name, data_root, 'test', transform_val, dataset_dir=ds_dir)
        params['wiki_root'] = image_root
        ds = CombinedEmotionDataset(**params)
        if len(ds) == 0:
            return None
        return _DataLoader(ds, batch_size=config.DATA.BATCH_SIZE, shuffle=False,
                           num_workers=config.DATA.NUM_WORKERS, collate_fn=collate_fn_2,
                           pin_memory=True)

    datasets_to_eval = list(args.train_datasets) + list(getattr(args, 'benchmark_datasets', []))
    per_dataset_results = {}
    for ds_name in datasets_to_eval:
        print(f"Evaluating on {ds_name} test set...")
        loader = _build_test_loader(ds_name)
        if loader is None:
            print(f"  No test data found for {ds_name}, skipping")
            continue
        print(f"  {len(loader.dataset)} test samples")
        ds_results = evaluate_on_multi_dataset_any_model(
            loader, model, device,
            model_type=config.MODEL.TYPE,
            writer=writer,
            epoch=config.TRAIN.EPOCHS,
            tag=f'test_{ds_name}',
            dataset_type=ds_name,
        )
        per_dataset_results[ds_name] = ds_results
        print(f"  {ds_name}: {ds_results}")

    # Save test results
    test_results_path = os.path.join(config.OUTPUT, 'test_results.json')
    with open(test_results_path, 'w') as f:
        json.dump({'best_epoch': best_epoch, 'best_val_loss': best_val_loss,
                   'per_dataset': per_dataset_results}, f, indent=2)
    print(f"Test results saved to {test_results_path}")

    if writer:
        writer.close()

    cleanup_distributed()
    print("Training complete!")


if __name__ == '__main__':
    main()
