import torch
import torch.nn.functional as F
import numpy as np


def sample_kde_at_emotion_Z_differentiable(Z, emotion_coords=None, epsilon=1e-10):
    """
    Differentiably sample KDE at emotion coordinates using bilinear interpolation.
    
    Args:
        Z: KDE heatmap (ny, nx) with values in [-1, 1] space
        emotion_coords: Optional dict of emotion -> [valence, arousal] coordinates
        epsilon: Small value for numerical stability
    
    Returns:
        log_normalized: Log-space emotion distribution (num_emotions,)
    """
    if emotion_coords is None:
        emotion_coords = {
            'amusement': [0.858, 0.674],
            'awe': [-0.062, 0.48],
            'contentment': [0.75, 0.22],
            'excitement': [0.792, 0.368],
            'anger': [-0.666, 0.73],
            'disgust': [-0.896, 0.55],
            'fear': [-0.854, 0.68],
            'sadness': [-0.896, -0.424],
        }

    ny, nx = Z.shape
    device = Z.device
    dtype = Z.dtype

    # Prepare sampling grid for grid_sample
    if isinstance(emotion_coords, dict):
        coords_list = [torch.tensor(coord, dtype=dtype, device=device) for coord in emotion_coords.values()]
        coords_tensor = torch.stack(coords_list)
    else:
        coords_tensor = emotion_coords

    sampling_grid = coords_tensor.view(1, coords_tensor.shape[0], 1, 2)
    Z_reshaped = Z.unsqueeze(0).unsqueeze(0)

    sampled_values_raw = F.grid_sample(
        Z_reshaped,
        sampling_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )

    density_tensor = sampled_values_raw.squeeze()
    clamped_positive = torch.clamp(density_tensor, min=epsilon)
    log_clamped = torch.log(clamped_positive)
    log_total = torch.logsumexp(log_clamped, dim=0)
    log_normalized = log_clamped - log_total

    return log_normalized


def sample_kde_batch(Z_batch, emotion_coords=None, epsilon=1e-10):
    """
    Batch version of sample_kde_at_emotion_Z_differentiable.
    
    Args:
        Z_batch: Batch of heatmaps (batch_size, ny, nx)
        emotion_coords: Optional emotion coordinates
        epsilon: Numerical stability term
    
    Returns:
        log_dists: Batch of log-space distributions (batch_size, num_emotions)
    """
    if Z_batch.dim() == 2:
        return sample_kde_at_emotion_Z_differentiable(Z_batch, emotion_coords, epsilon)

    batch_size = Z_batch.shape[0]
    results = []

    for i in range(batch_size):
        log_dist = sample_kde_at_emotion_Z_differentiable(Z_batch[i], emotion_coords, epsilon)
        results.append(log_dist)

    return torch.stack(results, dim=0)
