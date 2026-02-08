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
) -> dict[int, np.ndarray]:
    """
    Forward pass (encode only, no sampling) for all windows.

    model.eval() + torch.no_grad() — inference only.
    Uses model.encode(x) which returns mu directly.

    :param model (VAEModel): Trained VAE model
    :param windows (torch.Tensor): All windows (N_windows, T, F)
    :param window_metadata (pd.DataFrame): Must contain 'stock_id' column
    :param batch_size (int): Batch size for inference
    :param device (torch.device | None): Device for computation

    :return trajectories (dict): stock_id (int) → ndarray (n_windows_for_stock, K)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    N = windows.shape[0]
    all_mu: list[np.ndarray] = []
    n_batches = (N + batch_size - 1) // batch_size

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
            mu = model.encode(x)  # (batch, K), deterministic
            all_mu.append(mu.float().cpu().numpy())  # ensure float32 output

    mu_all = np.concatenate(all_mu, axis=0)  # (N_windows, K)

    # Group by stock_id
    stock_ids_arr = np.asarray(window_metadata["stock_id"].values)
    trajectories: dict[int, np.ndarray] = {}

    unique_stocks = np.unique(stock_ids_arr)
    for sid in unique_stocks:
        mask = stock_ids_arr == sid
        trajectories[int(sid)] = mu_all[mask]

    return trajectories


def aggregate_profiles(
    trajectories: dict[int, np.ndarray],
    method: str = "mean",
) -> tuple[np.ndarray, list[int]]:
    """
    Aggregate local latent vectors into composite profiles.

    Default: mean (all windows contribute equally, preserving memory
    of all historical regimes).

    :param trajectories (dict): stock_id (int) → (n_windows, K)
    :param method (str): Aggregation method ('mean')

    :return B (np.ndarray): Exposure matrix (n_stocks, K)
    :return stock_ids (list[int]): Ordered stock identifiers (permnos)
    """
    stock_ids = sorted(trajectories.keys())
    profiles: list[np.ndarray] = []

    for sid in stock_ids:
        vectors = trajectories[sid]  # (n_windows_for_stock, K)

        if method == "mean":
            profile = np.mean(vectors, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        profiles.append(profile)

    B = np.stack(profiles, axis=0)  # (n_stocks, K)
    return B, stock_ids
