"""
Active Unit (AU) measurement and statistical truncation.

AU = number of latent dimensions with marginal KL > 0.01 nats.
If AU > AU_max_stat, truncate to the top AU_max_stat dimensions.

Reference: ISD Section MOD-006 — Sub-task 3.
"""

import math

import numpy as np
import torch

from src.vae.model import VAEModel


def measure_active_units(
    model: VAEModel,
    windows: torch.Tensor,
    batch_size: int = 512,
    au_threshold: float = 0.01,
    device: torch.device | None = None,
) -> tuple[int, np.ndarray, list[int]]:
    """
    Compute marginal KL per dimension and identify active units.

    KL_k = (1/N) Σ_i (1/2)(μ²_ik + exp(log_var_ik) - log_var_ik - 1)

    Active unit k ⟺ KL_k > au_threshold nats.

    :param model (VAEModel): Trained VAE model
    :param windows (torch.Tensor): All windows (N_windows, T, F)
    :param batch_size (int): Batch size
    :param au_threshold (float): KL threshold for AU detection (nats)
    :param device (torch.device | None): Device

    :return AU (int): Number of active dimensions
    :return kl_per_dim (np.ndarray): Marginal KL per dimension (K,)
    :return active_dims (list[int]): Indices of active dimensions, sorted by decreasing KL
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    N = windows.shape[0]
    K = model.K

    # Accumulate KL components across batches (float32 for numerical precision)
    sum_kl_components = torch.zeros(K, device=device, dtype=torch.float32)
    n_samples = 0

    # AMP autocast: 2-3x faster on CUDA/MPS Tensor Cores, no-op on CPU
    _use_amp = device.type in ("cuda", "mps")

    with torch.no_grad(), torch.amp.autocast(  # type: ignore[reportPrivateImportUsage]
        device_type=device.type,
        dtype=torch.float16,
        enabled=_use_amp,
    ):
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x = windows[start:end].to(device, non_blocking=True)

            # Get mu and log_var from encoder
            x_enc = x.transpose(1, 2)  # (B, F, T)
            mu, log_var = model.encoder(x_enc)

            # KL per dimension per sample: 0.5 * (μ² + exp(lv) - lv - 1)
            # Cast to float32 for accumulation precision
            kl_batch = 0.5 * (mu.float() ** 2 + torch.exp(log_var.float()) - log_var.float() - 1.0)
            sum_kl_components += kl_batch.sum(dim=0)
            n_samples += kl_batch.shape[0]

    # Marginal KL per dimension
    kl_per_dim = (sum_kl_components / max(1, n_samples)).cpu().numpy()

    # Active units: KL_k > threshold
    active_mask = kl_per_dim > au_threshold
    active_dims_unsorted = np.where(active_mask)[0]

    # Sort by decreasing KL
    sorted_order = np.argsort(-kl_per_dim[active_dims_unsorted])
    active_dims = active_dims_unsorted[sorted_order].tolist()

    AU = len(active_dims)

    return AU, kl_per_dim, active_dims


def compute_au_max_stat(
    n_obs: int,
    r_min: int = 2,
    ewma_half_life: int = 0,
) -> int:
    """
    Statistical upper bound on AU.

    AU_max_stat = floor(sqrt(2 × N_eff / r_min))

    When ewma_half_life > 0, uses the Kish (1965) effective sample size
    N_eff = 1 / sum(w_i^2) instead of raw N_obs.  This ensures AU is
    consistent with the effective data richness used by the DGJ spiked
    shrinker for Sigma_z estimation.

    :param n_obs (int): Number of historical days (observations)
    :param r_min (int): Minimum observations-per-parameter ratio
    :param ewma_half_life (int): EWMA half-life in days.  0 = no EWMA
        (uses raw n_obs).

    :return au_max (int): Statistical upper bound on AU
    """
    n_eff = n_obs
    if ewma_half_life > 0 and n_obs > ewma_half_life:
        n_eff = compute_ewma_n_eff(n_obs, ewma_half_life)
    return int(math.floor(math.sqrt(2.0 * n_eff / r_min)))


def compute_ewma_n_eff(
    n_obs: int,
    ewma_half_life: int,
) -> int:
    """
    Kish (1965) effective sample size for EWMA weights.

    N_eff = 1 / sum(w_i^2) where w_i = exp(-decay*(n-1-i)) / sum(w).

    :param n_obs (int): Number of raw observations
    :param ewma_half_life (int): EWMA half-life in days

    :return n_eff (int): Effective sample size (>= 2)
    """
    decay = math.log(2.0) / ewma_half_life
    raw_w = np.exp(-decay * np.arange(n_obs - 1, -1, -1, dtype=np.float64))
    raw_w /= raw_w.sum()
    return max(2, int(1.0 / float(np.sum(raw_w ** 2))))


def truncate_active_dims(
    AU: int,
    kl_per_dim: np.ndarray,
    active_dims: list[int],
    au_max_stat: int,
) -> tuple[int, list[int]]:
    """
    If AU > AU_max_stat, keep only the top AU_max_stat dimensions
    with the highest marginal KL.

    :param AU (int): Original number of active units
    :param kl_per_dim (np.ndarray): Marginal KL per dimension (K,)
    :param active_dims (list[int]): Active dimension indices (sorted by decreasing KL)
    :param au_max_stat (int): Statistical upper bound

    :return AU_truncated (int): Truncated AU
    :return active_dims_truncated (list[int]): Truncated active dims
    """
    if AU <= au_max_stat:
        return AU, active_dims

    # Keep top au_max_stat dimensions (already sorted by decreasing KL)
    truncated = active_dims[:au_max_stat]
    return len(truncated), truncated


def filter_exposure_matrix(
    B: np.ndarray,
    active_dims: list[int],
) -> np.ndarray:
    """
    Filter exposure matrix to active dimensions only.

    :param B (np.ndarray): Full exposure matrix (n_stocks, K)
    :param active_dims (list[int]): Active dimension indices

    :return B_A (np.ndarray): Filtered exposures (n_stocks, AU)
    """
    return B[:, active_dims]
