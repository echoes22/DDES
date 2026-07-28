import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from scipy import stats

from .loss import (
    initialize_meters, multi_evaluate, log_meters,
    compute_and_log_val_metrics, AverageMeter
)
from .inference_utils import (
    soft_argmax_2d, weighted_average_to_va, va_to_heatmap,
    va_to_emotion_distribution, EMOTION_COORDS,
    map_emotion_distribution_torch, _coords_tensor, _wikiart_emo_coords_tensor,
)
from .kde import sample_kde_batch


def _ccc(pred, target):
    """Concordance Correlation Coefficient."""
    pred_mean = np.mean(pred)
    target_mean = np.mean(target)
    cov = np.mean((pred - pred_mean) * (target - target_mean))
    return float(2 * cov / (np.var(pred) + np.var(target) + (pred_mean - target_mean) ** 2 + 1e-10))


def compute_va_metrics(pred_va, target_va):
    """Compute VA-space metrics. Both inputs are (N,2) CPU tensors."""
    pv = pred_va[:, 0].numpy()
    pa = pred_va[:, 1].numpy()
    tv = target_va[:, 0].numpy()
    ta = target_va[:, 1].numpy()

    mse_v = float(np.mean((pv - tv) ** 2))
    mse_a = float(np.mean((pa - ta) ** 2))

    return {
        'mse_v': mse_v,
        'mse_a': mse_a,
        'mae_v': float(np.mean(np.abs(pv - tv))),
        'mae_a': float(np.mean(np.abs(pa - ta))),
        'rmse2d': float(np.sqrt(np.mean((pv - tv) ** 2 + (pa - ta) ** 2))),
        'pearson_v': float(np.corrcoef(pv, tv)[0, 1]),
        'pearson_a': float(np.corrcoef(pa, ta)[0, 1]),
        'spearman_v': float(stats.spearmanr(pv, tv).correlation),
        'spearman_a': float(stats.spearmanr(pa, ta).correlation),
        'ccc_v': _ccc(pv, tv),
        'ccc_a': _ccc(pa, ta),
        'acc_within_0.1': float(np.mean(np.sqrt((pv - tv) ** 2 + (pa - ta) ** 2) <= 0.1)),
        'va_mse': mse_v + mse_a,
    }


def single_epoch_train_multi_ds_any_model(
    train_dataloader,
    model,
    optimizer,
    device,
    model_type='convnext_unet',
    writer=None,
    epoch=0,
    use_amp=False,
    gradient_accumulation_steps=1,
    epsilon=1e-10,
    lr_scheduler=None,
):
    """
    Train for one epoch supporting multiple datasets and model types.
    
    Args:
        train_dataloader: DataLoader with collate_fn_2 output
        model: Model instance (convnext_unet/8/2)
        optimizer: Optimizer instance
        device: torch.device
        model_type: 'convnext_unet' | 'convnext_8' | 'convnext_2'
        writer: TensorBoard SummaryWriter (optional)
        epoch: Current epoch number
        use_amp: Use automatic mixed precision
        gradient_accumulation_steps: Number of steps to accumulate gradients
        epsilon: Small value for numerical stability
    
    Returns:
        train_loss: Average loss for epoch
    """
    model.train()
    avg_meters = initialize_meters()
    loss_meter = AverageMeter()
    
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    optimizer.zero_grad()
    
    total_steps = 0

    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Train Epoch {epoch}")):
        images = batch['image'].to(device)
        heatmaps = batch.get('heatmap', None)
        vas = batch.get('va', None)
        emo_dists = batch.get('emotion_dist', None)
        hm_mask = batch.get('hm_mask', None)
        va_mask = batch.get('va_mask', None)
        dists_mask = batch.get('dists_mask', None)

        if heatmaps is not None:
            heatmaps = heatmaps.to(device)
        if vas is not None:
            vas = vas.to(device)
        if emo_dists is not None:
            emo_dists = emo_dists.to(device)
        if hm_mask is not None:
            hm_mask = hm_mask.to(device)
        if va_mask is not None:
            va_mask = va_mask.to(device)
        if dists_mask is not None:
            dists_mask = dists_mask.to(device)

        batch_size = images.shape[0]

        # Forward pass
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(images)

            loss = 0

            # Compute loss based on model type and available targets
            if model_type == 'convnext_unet' and hm_mask is not None and hm_mask.any():
                # outputs: (batch, 1, 28, 28) — already log-softmax from ConvNeXtDecoder
                # heatmaps: (batch, 1, 28, 28) — normalised probability maps (not log)
                valid_indices = hm_mask.nonzero(as_tuple=True)[0]
                if len(valid_indices) > 0:
                    valid_log_pred = outputs[valid_indices]   # log-probabilities
                    valid_targets = heatmaps[valid_indices]   # probabilities
                    # KL(target || pred) = Σ target * (log_target - log_pred)
                    B = valid_log_pred.shape[0]
                    log_pred_flat = valid_log_pred.view(B, -1)
                    target_flat = valid_targets.view(B, -1)
                    loss += F.kl_div(log_pred_flat, target_flat, reduction='batchmean')

            elif model_type == 'convnext_8' and dists_mask is not None and dists_mask.any():
                # outputs: (batch, 8) — already log-softmax from CustomConvNeXt
                # emo_dists: (batch, N) probability targets
                valid_indices = dists_mask.nonzero(as_tuple=True)[0]
                if len(valid_indices) > 0:
                    valid_outputs = outputs[valid_indices]
                    valid_targets = emo_dists[valid_indices]
                    # valid_outputs already log-softmax (CustomConvNeXt); valid_targets are probabilities
                    loss += F.kl_div(valid_outputs, valid_targets, reduction='batchmean')

            elif model_type == 'convnext_2' and va_mask is not None and va_mask.any():
                # outputs: (batch, 2) VA predictions
                # vas: (batch, 2) VA targets
                valid_indices = va_mask.nonzero(as_tuple=True)[0]
                if len(valid_indices) > 0:
                    valid_outputs = outputs[valid_indices]
                    valid_targets = vas[valid_indices]
                    loss += F.mse_loss(valid_outputs, valid_targets)

        if loss == 0:
            continue

        # Backward pass
        loss = loss / gradient_accumulation_steps
        
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                global_step = epoch * len(train_dataloader) + batch_idx
                lr_scheduler.step_update(global_step)

        loss_meter.update(loss.item() * gradient_accumulation_steps, batch_size)
        total_steps += 1

        # Log every 100 steps
        if total_steps % 100 == 0 and writer is not None:
            writer.add_scalar('Loss/train', loss_meter.avg, total_steps)

    if writer is not None:
        writer.add_scalar('Loss/train_epoch', loss_meter.avg, epoch)

    return loss_meter.avg


@torch.no_grad()
def evaluate_on_multi_dataset_any_model(
    val_dataloader,
    model,
    device,
    model_type='convnext_unet',
    writer=None,
    epoch=0,
    tag='val',
    epsilon=1e-10,
    dataset_type=None,
):
    """
    Evaluate on validation/test set supporting multiple datasets and model types.

    Returns:
        results: Dict with full VA metrics (pearson, CCC, RMSE2D, …) plus 'val_loss'
                 (dist_kl for unet/8, va_mse for convnext_2) for checkpoint selection.
    """
    model.eval()

    all_pred_va = []
    all_target_va = []
    all_pred_dist = []
    all_target_dist = []

    for batch in tqdm(val_dataloader, desc=f"Eval {tag.upper()} Epoch {epoch}"):
        images = batch['image'].to(device)
        vas = batch.get('va', None)
        emo_dists = batch.get('emotion_dist', None)
        va_mask = batch.get('va_mask', None)
        dists_mask = batch.get('dists_mask', None)

        if vas is not None:
            vas = vas.to(device)
        if emo_dists is not None:
            emo_dists = emo_dists.to(device)
        if va_mask is not None:
            va_mask = va_mask.to(device)
        if dists_mask is not None:
            dists_mask = dists_mask.to(device)

        outputs = model(images)

        is_wikiart = (dataset_type == 'wikiart_emo')

        if model_type == 'convnext_unet':
            hm_softmax = outputs.exp()  # (B,1,28,28)
            pred_va = soft_argmax_2d(hm_softmax.squeeze(1))  # (B,2)

            if va_mask is not None and va_mask.any():
                idx = va_mask.squeeze().nonzero(as_tuple=True)[0]
                all_pred_va.append(pred_va[idx].cpu())
                all_target_va.append(vas[idx].cpu())

            if dists_mask is not None and dists_mask.any():
                idx = dists_mask.squeeze().nonzero(as_tuple=True)[0]
                if is_wikiart:
                    # Sample heatmap at 20 WikiArt-Emo coords instead of 8 ArtEmis coords
                    w_coords = _wikiart_emo_coords_tensor(device, hm_softmax.dtype)
                    pred_dist_20 = sample_kde_batch(hm_softmax.squeeze(1), emotion_coords=w_coords, epsilon=epsilon)  # (B,20)
                    all_pred_dist.append(pred_dist_20[idx].cpu())
                else:
                    pred_dist = sample_kde_batch(hm_softmax.squeeze(1), epsilon=epsilon)  # (B,8)
                    all_pred_dist.append(pred_dist[idx].cpu())
                all_target_dist.append(emo_dists[idx].cpu())

        elif model_type == 'convnext_8':
            pred_dist_8 = outputs  # (B,8) — already log-softmax from CustomConvNeXt
            pred_va = weighted_average_to_va(pred_dist_8.exp())  # (B,2)

            if va_mask is not None and va_mask.any():
                idx = va_mask.squeeze().nonzero(as_tuple=True)[0]
                all_pred_va.append(pred_va[idx].cpu())
                all_target_va.append(vas[idx].cpu())

            if dists_mask is not None and dists_mask.any():
                idx = dists_mask.squeeze().nonzero(as_tuple=True)[0]
                if is_wikiart:
                    # IDW: map 8-class ArtEmis → 20-class WikiArt-Emo
                    a_coords = _coords_tensor(device, pred_dist_8.dtype)
                    w_coords = _wikiart_emo_coords_tensor(device, pred_dist_8.dtype)
                    probs_20 = map_emotion_distribution_torch(pred_dist_8[idx].exp(), a_coords, w_coords)
                    all_pred_dist.append(torch.log(probs_20 + epsilon).cpu())
                else:
                    all_pred_dist.append(pred_dist_8[idx].cpu())
                all_target_dist.append(emo_dists[idx].cpu())

        elif model_type == 'convnext_2':
            pred_va = outputs  # (B,2)

            if va_mask is not None and va_mask.any():
                idx = va_mask.squeeze().nonzero(as_tuple=True)[0]
                all_pred_va.append(pred_va[idx].cpu())
                all_target_va.append(vas[idx].cpu())

            if is_wikiart and dists_mask is not None and dists_mask.any():
                # WikiArt-Emo has 20-class dist labels but no VA labels.
                # Convert VA prediction → 20-class distribution for comparison.
                idx = dists_mask.squeeze().nonzero(as_tuple=True)[0]
                w_coords = _wikiart_emo_coords_tensor(device, pred_va.dtype)
                probs_20 = va_to_emotion_distribution(pred_va[idx], emotion_coords=w_coords)  # (M,20)
                all_pred_dist.append(torch.log(probs_20 + epsilon).cpu())
                all_target_dist.append(emo_dists[idx].cpu())

    results = {}

    # Full VA metrics
    if all_pred_va:
        pred_va_cat = torch.cat(all_pred_va, dim=0)
        target_va_cat = torch.cat(all_target_va, dim=0)
        va_metrics = compute_va_metrics(pred_va_cat, target_va_cat)
        results.update(va_metrics)
        if writer is not None:
            for k, v in va_metrics.items():
                writer.add_scalar(f'{tag}/{k}', v, epoch)

    # Distribution KL + categorical metrics (for unet/convnext_8)
    if all_pred_dist:
        pred_dist_cat = torch.cat(all_pred_dist, dim=0)    # (N, C_pred) log-probs
        target_dist_cat = torch.cat(all_target_dist, dim=0) # (N, C_target) probs
        if pred_dist_cat.shape[1] == target_dist_cat.shape[1]:
            dist_kl = F.kl_div(pred_dist_cat, target_dist_cat, reduction='batchmean').item()
            results['dist_kl'] = dist_kl
            if writer is not None:
                writer.add_scalar(f'{tag}/dist_kl', dist_kl, epoch)

            # Top-1 accuracy and Kendall's tau — paper ArtEmis metrics
            pred_probs = pred_dist_cat.exp().numpy()
            target_probs = target_dist_cat.numpy()
            pred_top1 = np.argmax(pred_probs, axis=1)
            target_top1 = np.argmax(target_probs, axis=1)
            top1_acc = float(np.mean(pred_top1 == target_top1))
            tau_scores = [float(stats.kendalltau(pred_probs[i], target_probs[i]).correlation)
                          for i in range(len(pred_probs))]
            tau_arr = np.array(tau_scores, dtype=np.float64)
            kendall_tau = float(np.nanmean(tau_arr))
            results['top1_acc'] = top1_acc
            results['kendall_tau'] = kendall_tau
            if writer is not None:
                writer.add_scalar(f'{tag}/top1_acc', top1_acc, epoch)
                writer.add_scalar(f'{tag}/kendall_tau', kendall_tau, epoch)
        # else: class-count mismatch (e.g. wikiart_emo has 20 classes, model outputs 8) — skip dist metrics

    # val_loss used for best-checkpoint selection
    if model_type == 'convnext_2':
        results['val_loss'] = results.get('va_mse', float('inf'))
    else:
        results['val_loss'] = results.get('dist_kl', results.get('va_mse', float('inf')))

    model.train()
    return results
