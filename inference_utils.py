import torch
import torch.nn.functional as F
import numpy as np


# Eq. 1/3/4 — ArtEmis 8-class emotion anchors in V-A space (excludes 'something else')
EMOTION_COORDS = {
    'amusement':   [ 0.858,  0.674],
    'awe':         [-0.062,  0.480],
    'contentment': [ 0.750,  0.220],
    'excitement':  [ 0.792,  0.368],
    'anger':       [-0.666,  0.730],
    'disgust':     [-0.896,  0.550],
    'fear':        [-0.854,  0.680],
    'sadness':     [-0.896, -0.424],
}

# Full 9-class ArtEmis mapping including 'something else' (origin)
ARTEMIS_COORDS = {
    **EMOTION_COORDS,
    'something else': [0.000, 0.000],
}

EMOTION_NAMES = list(EMOTION_COORDS.keys())
NUM_EMOTIONS = len(EMOTION_NAMES)  # 8

# WikiArt-Emotion dataset: 20-class emotion anchors in V-A space
WIKIART_EMO_COORDS = {
    'agreeableness':    [ 0.812,  0.000],
    'anger':            [-0.666,  0.730],
    'anticipation':     [ 0.396,  0.078],
    'arrogance':        [-0.714,  0.500],
    'disagreeableness': [-0.916,  0.286],
    'disgust':          [-0.896,  0.550],
    'fear':             [-0.854,  0.680],
    'gratitude':        [ 0.770, -0.118],
    'happiness':        [ 0.920,  0.464],
    'humility':         [ 0.480, -0.764],
    'love':             [ 1.000,  0.038],
    'optimism':         [ 0.898,  0.130],
    'pessimism':        [-0.834, -0.032],
    'regret':           [-0.540,  0.246],
    'sadness':          [-0.896, -0.424],
    'shame':            [-0.880,  0.340],
    'shyness':          [-0.376, -0.470],
    'surprise':         [ 0.750,  0.750],
    'trust':            [ 0.776,  0.094],
    'neutral':          [-0.062, -0.632],
}

WIKIART_EMO_NAMES = list(WIKIART_EMO_COORDS.keys())
NUM_WIKIART_EMO = len(WIKIART_EMO_NAMES)  # 20


def _wikiart_emo_coords_tensor(device, dtype):
    """Return (20, 2) tensor of WikiArt-Emotion anchor coordinates."""
    return torch.tensor(
        [WIKIART_EMO_COORDS[n] for n in WIKIART_EMO_NAMES],
        dtype=dtype,
        device=device,
    )


def map_emotion_distribution_torch(source_probs, source_coords, target_coords, epsilon=1e-6):
    """
    Remap a probability distribution from source to target emotion space using IDW.

    For WikiArt-Emo evaluation: call with source_coords=8-class ArtEmis coords,
    target_coords=20-class WikiArt-Emo coords.

    Args:
        source_probs: (B, N_source) — probabilities summing to 1 per row.
        source_coords: (N_source, 2) — VA coords for source labels.
        target_coords: (N_target, 2) — VA coords for target labels.
        epsilon: Stability term for inverse-distance weighting.

    Returns:
        (B, N_target) — remapped probability distribution summing to 1.
    """
    dists = torch.cdist(target_coords, source_coords, p=2.0)   # (N_t, N_s)
    weights = 1.0 / (dists + epsilon)                           # (N_t, N_s)
    unnorm = torch.matmul(source_probs, weights.T)              # (B, N_t)
    return unnorm / unnorm.sum(dim=1, keepdim=True)


def _coords_tensor(device, dtype):
    """Return (8, 2) tensor of emotion anchor coordinates."""
    return torch.tensor(
        [EMOTION_COORDS[n] for n in EMOTION_NAMES],
        dtype=dtype,
        device=device,
    )


# ---------------------------------------------------------------------------
# Eq. 5 — DDES → DES: soft-argmax with temperature τ
# ---------------------------------------------------------------------------

def soft_argmax_2d(Z, temperature=0.05):
    """
    Differentiable soft-argmax over a 2D heatmap (Eq. 5).

    p̃ = softmax(log(Z) / τ),  v = Σ_{i,j} p̃(i,j) · z(i,j)

    Args:
        Z: (H, W) or (B, H, W) probability tensor (non-negative).
        temperature: τ. Smaller → sharper. Default 0.05.

    Returns:
        (2,) or (B, 2) [valence, arousal].
    """
    batch_mode = Z.dim() == 3
    if not batch_mode:
        Z = Z.unsqueeze(0)

    B, H, W = Z.shape
    device, dtype = Z.device, Z.dtype

    x_grid = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    y_grid = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y_grid, x_grid, indexing='ij')  # Y=arousal, X=valence

    log_Z = torch.log(Z.clamp(min=1e-10))
    weights = F.softmax((log_Z / temperature).view(B, -1), dim=1).view(B, H, W)

    valence = (weights * X.unsqueeze(0)).sum(dim=(1, 2))
    arousal = (weights * Y.unsqueeze(0)).sum(dim=(1, 2))
    va = torch.stack([valence, arousal], dim=1)

    return va if batch_mode else va.squeeze(0)


# ---------------------------------------------------------------------------
# Eq. 1 — CES → DES: weighted average of emotion coords
# ---------------------------------------------------------------------------

def weighted_average_to_va(probs, emotion_coords=None):
    """
    Convert emotion probability distribution to VA via weighted average (Eq. 1).

    Args:
        probs: (N,) or (B, N) float tensor.
        emotion_coords: Optional (N, 2) tensor.

    Returns:
        (2,) or (B, 2) VA tensor.
    """
    batch_mode = probs.dim() == 2
    if not batch_mode:
        probs = probs.unsqueeze(0)

    if emotion_coords is None:
        emotion_coords = _coords_tensor(probs.device, probs.dtype)

    va = probs @ emotion_coords
    return va if batch_mode else va.squeeze(0)


# ---------------------------------------------------------------------------
# Eq. 3 — DES → CES: Gaussian softmax with sharpness k=10
# ---------------------------------------------------------------------------

def va_to_emotion_distribution(va, emotion_coords=None, k=10.0):
    """
    Convert VA point to emotion probability distribution (Eq. 3).

    p_i = exp(-k · ||v − c_i||²) / Σ_j exp(-k · ||v − c_j||²)

    Args:
        va: (2,) or (B, 2) float tensor.
        emotion_coords: Optional (N, 2) tensor.
        k: Sharpness constant. Default 10 (per paper).

    Returns:
        (N,) or (B, N) probability tensor.
    """
    batch_mode = va.dim() == 2
    if not batch_mode:
        va = va.unsqueeze(0)

    if emotion_coords is None:
        emotion_coords = _coords_tensor(va.device, va.dtype)

    dists_sq = torch.cdist(va, emotion_coords).pow(2)
    probs = F.softmax(-k * dists_sq, dim=1)

    return probs if batch_mode else probs.squeeze(0)


# ---------------------------------------------------------------------------
# DES → DDES: Gaussian heatmap at VA point
# ---------------------------------------------------------------------------

def va_to_heatmap(va, grid_size=28, sigma=0.1):
    """
    Convert VA point to a normalised Gaussian heatmap.

    Args:
        va: (2,) tensor — [valence, arousal] in [-1, 1].
        grid_size: N for N×N output. Default 28.
        sigma: Gaussian bandwidth. Default 0.1.

    Returns:
        (grid_size, grid_size) tensor summing to 1.
    """
    device, dtype = va.device, va.dtype
    x_grid = torch.linspace(-1, 1, grid_size, device=device, dtype=dtype)
    y_grid = torch.linspace(-1, 1, grid_size, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y_grid, x_grid, indexing='ij')

    valence, arousal = va[0], va[1]
    heatmap = torch.exp(-((X - valence) ** 2 + (Y - arousal) ** 2) / (2 * sigma ** 2))
    return heatmap / (heatmap.sum() + 1e-10)


# ---------------------------------------------------------------------------
# CES → DDES: weighted sum of Gaussians (Eq. 2)
# ---------------------------------------------------------------------------

def ces_to_ddes(probs, emotion_coords=None, grid_size=28, sigma=None):
    """
    Build a DDES heatmap as a probability-weighted sum of Gaussians (Eq. 2).

    Z(i,j) = Σ_k p_k · exp(−||z(i,j) − c_k||² / (2σ²))

    Args:
        probs: (N,) float tensor.
        emotion_coords: Optional (N, 2) tensor.
        grid_size: Spatial resolution N.
        sigma: Gaussian bandwidth. If None, computed via Scott's Rule.

    Returns:
        (grid_size, grid_size) tensor summing to 1.
    """
    device, dtype = probs.device, probs.dtype
    if emotion_coords is None:
        emotion_coords = _coords_tensor(device, dtype)

    if sigma is None:
        n = emotion_coords.shape[0]
        std = emotion_coords.std(dim=0).mean()
        sigma = float(std) * (n ** (-1 / 5))

    x_grid = torch.linspace(-1, 1, grid_size, device=device, dtype=dtype)
    y_grid = torch.linspace(-1, 1, grid_size, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y_grid, x_grid, indexing='ij')

    grid_pts = torch.stack([X.flatten(), Y.flatten()], dim=1)  # (H*W, 2)
    dists_sq = torch.cdist(grid_pts, emotion_coords).pow(2)    # (H*W, N)
    gaussians = torch.exp(-dists_sq / (2 * sigma ** 2))        # (H*W, N)
    heatmap = (gaussians @ probs).view(grid_size, grid_size)
    return heatmap / (heatmap.sum() + 1e-10)


# ---------------------------------------------------------------------------
# Alias kept for backwards compatibility
# ---------------------------------------------------------------------------

def expectation_va_FT(Z, temperature=0.05):
    return soft_argmax_2d(Z, temperature=temperature)
