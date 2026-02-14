"""
Latent factor inference: stride-1 encoding and aggregation to exposure matrix B.

Passes all windows through the trained encoder (deterministic, no sampling),
then aggregates per-stock local latent vectors into composite profiles.

CRITICAL: Uses model.encode() (mu only), NOT model.forward() (which samples).

Reference: ISD Section MOD-006 — Sub-tasks 1-2.
"""

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.vae.model import VAEModel


def infer_latent_trajectories(
    model: VAEModel,
    windows: torch.Tensor,
    window_metadata: pd.DataFrame,
    batch_size: int = 512,
    device: torch.device | None = None,
    compute_kl: bool = False,
) -> tuple[dict[int, np.ndarray], np.ndarray | None]:
    """
    Forward pass (encode only, no sampling) for all windows.

    model.eval() + torch.no_grad() — inference only.
    Optionally computes marginal KL per dimension in the same pass
    (avoids a second forward pass for AU measurement).

    :param model (VAEModel): Trained VAE model
    :param windows (torch.Tensor): All windows (N_windows, T, F)
    :param window_metadata (pd.DataFrame): Must contain 'stock_id' column
    :param batch_size (int): Batch size for inference
    :param device (torch.device | None): Device for computation
    :param compute_kl (bool): Also compute KL per dimension (for AU measurement)

    :return trajectories (dict): stock_id (int) → ndarray (n_windows_for_stock, K)
    :return kl_per_dim (np.ndarray | None): Marginal KL per dimension (K,), or None
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    N = windows.shape[0]
    K = model.K
    all_mu: list[np.ndarray] = []
    n_batches = (N + batch_size - 1) // batch_size

    # KL accumulator (if requested)
    sum_kl = torch.zeros(K, device=device, dtype=torch.float32) if compute_kl else None
    n_samples = 0

    # AMP autocast: 2-3x faster on CUDA/MPS Tensor Cores, no-op on CPU
    _use_amp = device.type in ("cuda", "mps")

    with torch.no_grad(), torch.amp.autocast(  # type: ignore[reportPrivateImportUsage]
        device_type=device.type,
        dtype=torch.float16,
        enabled=_use_amp,
    ):
        batch_iter = tqdm(
            range(0, N, batch_size),
            total=n_batches,
            desc="    Inference",
            unit="batch",
        ) if n_batches > 1 else range(0, N, batch_size)

        for start in batch_iter:
            end = min(start + batch_size, N)
            x = windows[start:end].to(device, non_blocking=True)

            if compute_kl:
                # Use encoder directly to get both mu and log_var
                x_enc = x.transpose(1, 2)  # (B, F, T)
                mu, log_var = model.encoder(x_enc)
                all_mu.append(mu.float().cpu().numpy())
                # Accumulate KL: 0.5 * (μ² + exp(lv) - lv - 1)
                kl_batch = 0.5 * (mu.float() ** 2 + torch.exp(log_var.float()) - log_var.float() - 1.0)
                sum_kl += kl_batch.sum(dim=0)  # type: ignore[union-attr]
                n_samples += kl_batch.shape[0]
            else:
                mu = model.encode(x)  # (batch, K), deterministic
                all_mu.append(mu.float().cpu().numpy())

    mu_all = np.concatenate(all_mu, axis=0)  # (N_windows, K)

    # Group by stock_id
    stock_ids_arr = np.asarray(window_metadata["stock_id"].values)
    trajectories: dict[int, np.ndarray] = {}

    unique_stocks = np.unique(stock_ids_arr)
    for sid in unique_stocks:
        mask = stock_ids_arr == sid
        trajectories[int(sid)] = mu_all[mask]

    kl_per_dim: np.ndarray | None = None
    if sum_kl is not None and n_samples > 0:
        kl_per_dim = (sum_kl / n_samples).cpu().numpy()

    return trajectories, kl_per_dim


def aggregate_profiles(
    trajectories: dict[int, np.ndarray],
    method: str = "mean",
    half_life: int = 0,
) -> tuple[np.ndarray, list[int]]:
    """
    Aggregate local latent vectors into composite profiles.

    Default: mean (all windows contribute equally, preserving memory
    of all historical regimes).

    When half_life > 0, use exponential decay weighting: recent windows
    get more weight than older ones. Each stock's windows are assumed to
    be sorted chronologically (from create_windows). The weight of
    window i (0=oldest, n-1=newest) is w_i = exp(-ln(2) * (n-1-i) / half_life).

    This addresses the "B_A staleness" issue: with half_life=60 and
    stride=21, windows older than ~5 years have negligible weight,
    so B_A reflects the current factor structure rather than the
    average over the entire training history.

    :param trajectories (dict): stock_id (int) → (n_windows, K)
    :param method (str): Aggregation method ('mean', 'ewm')
    :param half_life (int): Exponential decay half-life in windows.
        0 = uniform mean (original behavior). Typical: 60 (1260 days
        at stride=21, i.e. ~5 years). Only used when method='mean'.

    :return B (np.ndarray): Exposure matrix (n_stocks, K)
    :return stock_ids (list[int]): Ordered stock identifiers (permnos)
    """
    stock_ids = sorted(trajectories.keys())
    profiles: list[np.ndarray] = []

    for sid in stock_ids:
        vectors = trajectories[sid]  # (n_windows_for_stock, K)

        if method == "mean":
            if half_life > 0 and vectors.shape[0] > 1:
                n_w = vectors.shape[0]
                decay = np.log(2.0) / half_life
                # i=0 oldest, i=n_w-1 newest → weight = exp(-decay * (n_w-1-i))
                indices = np.arange(n_w, dtype=np.float64)
                weights = np.exp(-decay * (n_w - 1 - indices))
                weights /= weights.sum()
                profile = weights @ vectors  # (K,)
            else:
                profile = np.mean(vectors, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        profiles.append(profile)

    B = np.stack(profiles, axis=0)  # (n_stocks, K)
    return B, stock_ids
