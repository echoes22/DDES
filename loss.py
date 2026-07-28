import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.val = self.avg = self.sum = self.count = 0

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_meters():
    return {
        'acc1': AverageMeter(),
        'acc2': AverageMeter(),
        'acc3': AverageMeter(),
        'mse':  AverageMeter(),
        'ce':   AverageMeter(),
        'kl':   AverageMeter(),
        'kendall': AverageMeter(),
    }


def top_k_accuracy(pred, target, k):
    """Top-k accuracy where target is a probability distribution."""
    _, top_pred = torch.topk(pred, k, dim=1)
    target_class = target.argmax(dim=1).unsqueeze(1).expand_as(top_pred)
    correct = torch.eq(top_pred, target_class).any(dim=1)
    return correct.float().mean().item()


def multi_evaluate(pred, target, avg_meters):
    """Update meters with batch predictions (log-probs) vs targets (probs)."""
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction='batchmean')

    pred_exp = pred.exp()

    with torch.no_grad():
        avg_meters['acc1'].update(top_k_accuracy(pred_exp, target, 1))
        avg_meters['acc2'].update(top_k_accuracy(pred_exp, target, 2))
        avg_meters['acc3'].update(top_k_accuracy(pred_exp, target, 3))
        avg_meters['mse'].update(mse_loss(pred_exp, target).item())
        avg_meters['ce'].update(ce_loss(pred_exp, target).item())
        avg_meters['kl'].update(kl_loss(pred, target).item())
        try:
            kt = stats.kendalltau(
                pred_exp.cpu().numpy().flatten(),
                target.cpu().numpy().flatten(),
            ).statistic
        except Exception:
            kt = 0.0
        avg_meters['kendall'].update(kt)

    return avg_meters


def class_accuracy(pred, target, num_classes):
    """Per-class top-1 accuracy and variance."""
    if isinstance(pred, list):
        pred = np.concatenate(pred)
    if isinstance(target, list):
        target = np.concatenate(target)

    class_acc, class_var = {}, {}
    for i in range(num_classes):
        mask = target == i
        if mask.sum() > 0:
            correct = pred[mask] == target[mask]
            class_acc[i] = float(correct.mean())
            class_var[i] = float(np.var(correct))
        else:
            class_acc[i] = class_var[i] = 0.0
    return class_acc, class_var


def log_meters(avg_meters, writer, dataset, step):
    if writer is None:
        return
    for k, v in avg_meters.items():
        if isinstance(v, AverageMeter):
            writer.add_scalar(f'{k}/{dataset}', v.avg, step)
        elif isinstance(v, dict):
            for emo, val in v.items():
                writer.add_scalar(f'{k}/{dataset}/{emo}', val, step)
        else:
            writer.add_scalar(f'{k}/{dataset}', v, step)


# ---------------------------------------------------------------------------
# Tensor-based metric computation (called with pre-stacked tensors)
# ---------------------------------------------------------------------------

def compute_and_log_val_metrics(preds, targets, writer, step, tag='val'):
    """
    Compute and log validation metrics.

    Args:
        preds:   Tensor (N, D) — predictions (log-probs for dist, raw VA for DES).
        targets: Tensor (N, D) — targets.
        writer:  TensorBoard SummaryWriter or None.
        step:    Global step for logging.
        tag:     Logging prefix.

    Returns:
        Dict of metric name → float.
    """
    if preds is None or (isinstance(preds, torch.Tensor) and preds.numel() == 0):
        return {}

    preds_np = preds.detach().cpu().float().numpy()
    targets_np = targets.detach().cpu().float().numpy()

    metrics = {}

    if preds_np.ndim == 2 and preds_np.shape[1] == 2:
        # VA predictions
        mse_v = float(np.mean((preds_np[:, 0] - targets_np[:, 0]) ** 2))
        mse_a = float(np.mean((preds_np[:, 1] - targets_np[:, 1]) ** 2))
        metrics['mse_v'] = mse_v
        metrics['mse_a'] = mse_a
        metrics['mse'] = (mse_v + mse_a) / 2

        try:
            metrics['corr_v'] = float(np.nan_to_num(np.corrcoef(preds_np[:, 0], targets_np[:, 0])[0, 1]))
            metrics['corr_a'] = float(np.nan_to_num(np.corrcoef(preds_np[:, 1], targets_np[:, 1])[0, 1]))
        except Exception:
            metrics['corr_v'] = metrics['corr_a'] = 0.0

        rmse_2d = float(np.sqrt(np.mean(
            (preds_np[:, 0] - targets_np[:, 0]) ** 2 +
            (preds_np[:, 1] - targets_np[:, 1]) ** 2
        )))
        metrics['rmse_2d'] = rmse_2d

    elif preds_np.ndim == 2:
        # Distribution predictions (log-probs vs probs)
        pred_probs = np.exp(preds_np)
        pred_class = pred_probs.argmax(axis=1)
        tgt_class  = targets_np.argmax(axis=1)
        metrics['acc1'] = float((pred_class == tgt_class).mean())

        top2 = np.argsort(pred_probs, axis=1)[:, -2:]
        metrics['acc2'] = float(np.mean([t in top2[i] for i, t in enumerate(tgt_class)]))

        kl = float(F.kl_div(
            torch.from_numpy(preds_np),
            torch.from_numpy(targets_np).clamp(min=0),
            reduction='batchmean',
        ).item())
        metrics['kl'] = kl

    if writer is not None:
        for k, v in metrics.items():
            writer.add_scalar(f'{k}/{tag}', v, step)

    return metrics


# ---------------------------------------------------------------------------
# KL divergence on heatmaps
# ---------------------------------------------------------------------------

def kl_heatmap_loss(log_pred, target_prob, scale=1.0):
    """
    KL divergence between predicted log-probability heatmap and target probability heatmap.

    Args:
        log_pred:    (B, 1, H, W) or (B, H, W) — log-softmax model output.
        target_prob: (B, 1, H, W) or (B, H, W) — normalised probability target.
        scale:       Loss scaling factor (paper uses 1e4 for numerical balance).

    Returns:
        Scalar loss.
    """
    B = log_pred.shape[0]
    log_pred_flat  = log_pred.view(B, -1)
    target_flat    = target_prob.view(B, -1).clamp(min=0)
    return scale * F.kl_div(log_pred_flat, target_flat, reduction='batchmean')


# ---------------------------------------------------------------------------
# Pearson correlation loss (differentiable, for VA supervision)
# ---------------------------------------------------------------------------

def pearson_va_loss(pred_va, target_va, eps=1e-8):
    """
    Negative Pearson correlation loss on V and A channels separately.

    Args:
        pred_va:   (B, 2) tensor.
        target_va: (B, 2) tensor.

    Returns:
        Scalar in [-1, 1].
    """
    loss = 0.0
    for i in range(2):
        p = pred_va[:, i] - pred_va[:, i].mean()
        t = target_va[:, i] - target_va[:, i].mean()
        corr = (p * t).sum() / (p.norm() * t.norm() + eps)
        loss = loss + (1.0 - corr)
    return loss / 2.0


# ---------------------------------------------------------------------------
# Auxiliary moment loss (from soft-argmax) — Eq. supplement
# ---------------------------------------------------------------------------

def aux_moment_loss_from_softargmax(pred_va, target_va, eps=1e-8):
    """
    Auxiliary loss encouraging soft-argmax VA to match target VA.

    Decomposes into position loss (L_mu), radial loss (L_r), angular loss (L_ang).

    Args:
        pred_va:   (B, 2) predicted VA from soft-argmax.
        target_va: (B, 2) target VA.

    Returns:
        Scalar total auxiliary loss.
    """
    # Position loss: L2
    l_mu = F.mse_loss(pred_va, target_va)

    pred_r = pred_va.norm(dim=1)
    tgt_r  = target_va.norm(dim=1)
    l_r = F.mse_loss(pred_r, tgt_r)

    pred_angle = torch.atan2(pred_va[:, 1], pred_va[:, 0])
    tgt_angle  = torch.atan2(target_va[:, 1], target_va[:, 0])
    angle_diff = pred_angle - tgt_angle
    l_ang = (1.0 - torch.cos(angle_diff)).mean()

    return l_mu + l_r + l_ang
