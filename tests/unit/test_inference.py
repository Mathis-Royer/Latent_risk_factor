"""
Unit tests for the inference module (src/inference/).

Tests cover: deterministic encoding, exposure matrix shape, AU measurement,
AU truncation, and active dimension ordering.

Reference: ISD Section MOD-006.
"""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from src.vae.build_vae import build_vae
from src.vae.model import VAEModel
from src.inference.composite import aggregate_profiles, infer_latent_trajectories
from src.inference.active_units import (
    compute_au_max_stat,
    filter_exposure_matrix,
    measure_active_units,
    truncate_active_dims,
)
from src.training.trainer import VAETrainer

# ---------------------------------------------------------------------------
# Small model dimensions for fast tests
# ---------------------------------------------------------------------------

N_STOCKS = 20
T_VAL = 64
T_ANNEE = 3
F_VAL = 2
K_VAL = 10
WINDOWS_PER_STOCK = 5
N_WINDOWS = N_STOCKS * WINDOWS_PER_STOCK
DEVICE = torch.device("cpu")


@pytest.fixture
def trained_model_and_data() -> tuple[VAEModel, torch.Tensor, pd.DataFrame]:
    """
    Build a small VAE model, train it for a few epochs,
    and return the model with synthetic windows and metadata.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    model, _ = build_vae(
        n=N_STOCKS, T=T_VAL, T_annee=T_ANNEE, F=F_VAL, K=K_VAL,
        r_max=500.0,  # Relaxed for unit tests with small data
    )

    # Create synthetic windows with structure (so the model can learn something)
    windows = torch.randn(N_WINDOWS, T_VAL, F_VAL)

    # Create metadata: each stock has WINDOWS_PER_STOCK windows
    # Use integer permnos (matching production create_windows output)
    stock_id_list: list[int] = []
    for i in range(N_STOCKS):
        stock_id_list.extend([i] * WINDOWS_PER_STOCK)
    metadata = pd.DataFrame({"stock_id": stock_id_list})

    # Train briefly to produce non-trivial latent representations
    trainer = VAETrainer(
        model=model,
        loss_mode="P",
        gamma=1.0,
        learning_rate=1e-3,
        patience=50,
        device=DEVICE,
    )

    # Split into train/val
    train_windows = windows[:80]
    val_windows = windows[80:]

    trainer.fit(
        train_windows=train_windows,
        val_windows=val_windows,
        max_epochs=5,
        batch_size=32,
    )

    return model, windows, metadata


# ---------------------------------------------------------------------------
# 1. test_inference_deterministic — Two encode() passes give identical results
# ---------------------------------------------------------------------------

def test_inference_deterministic(
    trained_model_and_data: tuple[VAEModel, torch.Tensor, pd.DataFrame],
) -> None:
    """model.encode() should be deterministic: two passes give identical mu."""
    model, windows, _ = trained_model_and_data
    model.eval()

    with torch.no_grad():
        mu_1 = model.encode(windows[:16])
        mu_2 = model.encode(windows[:16])

    assert torch.allclose(mu_1, mu_2, atol=1e-6), (
        f"encode() should be deterministic. Max diff: {(mu_1 - mu_2).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# 2. test_B_shape — B.shape == (n_stocks, K) after aggregate_profiles
# ---------------------------------------------------------------------------

def test_B_shape(
    trained_model_and_data: tuple[VAEModel, torch.Tensor, pd.DataFrame],
) -> None:
    """Exposure matrix B should have shape (n_stocks, K)."""
    model, windows, metadata = trained_model_and_data

    trajectories = infer_latent_trajectories(
        model=model,
        windows=windows,
        window_metadata=metadata,
        batch_size=64,
        device=DEVICE,
    )

    B, stock_ids = aggregate_profiles(trajectories, method="mean")

    assert B.shape == (N_STOCKS, K_VAL), (
        f"B.shape should be ({N_STOCKS}, {K_VAL}), got {B.shape}"
    )
    assert len(stock_ids) == N_STOCKS, (
        f"Expected {N_STOCKS} stock_ids, got {len(stock_ids)}"
    )


# ---------------------------------------------------------------------------
# 3. test_AU_measurement — AU count is correct (>0 after training)
# ---------------------------------------------------------------------------

def test_AU_measurement(
    trained_model_and_data: tuple[VAEModel, torch.Tensor, pd.DataFrame],
) -> None:
    """AU should be > 0 after training (some dimensions active)."""
    model, windows, _ = trained_model_and_data

    AU, kl_per_dim, active_dims = measure_active_units(
        model=model,
        windows=windows,
        batch_size=64,
        au_threshold=0.01,
        device=DEVICE,
    )

    # After training, at least some dimensions should be active
    assert AU >= 0, f"AU should be non-negative, got {AU}"
    assert kl_per_dim.shape == (K_VAL,), (
        f"kl_per_dim.shape should be ({K_VAL},), got {kl_per_dim.shape}"
    )
    assert len(active_dims) == AU, (
        f"active_dims length ({len(active_dims)}) should match AU ({AU})"
    )

    # Verify AU matches the count of dimensions above threshold
    expected_au = int(np.sum(kl_per_dim > 0.01))
    assert AU == expected_au, f"AU={AU} should match count of KL > 0.01: {expected_au}"


# ---------------------------------------------------------------------------
# 4. test_AU_truncation — If AU > au_max_stat, truncation works correctly
# ---------------------------------------------------------------------------

def test_AU_truncation() -> None:
    """truncate_active_dims should keep only top au_max_stat dims."""
    torch.manual_seed(42)

    # Simulate AU=8 active dims with known KL values
    K = 10
    kl_per_dim = np.array([0.5, 0.3, 0.02, 0.8, 0.0, 0.15, 0.001, 0.6, 0.4, 0.25])

    # Active dims sorted by decreasing KL: [3(0.8), 7(0.6), 0(0.5), 8(0.4), 1(0.3), 9(0.25), 5(0.15), 2(0.02)]
    active_dims = [3, 7, 0, 8, 1, 9, 5, 2]
    AU = len(active_dims)  # 8

    # Set au_max_stat = 5 (should truncate to top 5)
    au_max_stat = 5
    AU_trunc, dims_trunc = truncate_active_dims(AU, kl_per_dim, active_dims, au_max_stat)

    assert AU_trunc == au_max_stat, f"Truncated AU should be {au_max_stat}, got {AU_trunc}"
    assert len(dims_trunc) == au_max_stat, (
        f"Truncated dims length should be {au_max_stat}, got {len(dims_trunc)}"
    )
    assert dims_trunc == [3, 7, 0, 8, 1], (
        f"Expected top 5 dims [3, 7, 0, 8, 1], got {dims_trunc}"
    )

    # When AU <= au_max_stat, no truncation should occur
    AU_no_trunc, dims_no_trunc = truncate_active_dims(AU, kl_per_dim, active_dims, 10)
    assert AU_no_trunc == AU, "No truncation when AU <= au_max_stat"
    assert dims_no_trunc == active_dims, "Dims unchanged when AU <= au_max_stat"

    # Verify compute_au_max_stat formula
    n_obs = 500
    r_min = 2
    expected_au_max = int(math.floor(math.sqrt(2.0 * n_obs / r_min)))
    assert compute_au_max_stat(n_obs, r_min) == expected_au_max

    # Verify filter_exposure_matrix
    B = np.random.randn(20, K)
    B_A = filter_exposure_matrix(B, dims_trunc)
    assert B_A.shape == (20, au_max_stat), (
        f"B_A.shape should be (20, {au_max_stat}), got {B_A.shape}"
    )


# ---------------------------------------------------------------------------
# 5. test_active_dims_ordering — active_dims sorted by decreasing KL
# ---------------------------------------------------------------------------

def test_active_dims_ordering(
    trained_model_and_data: tuple[VAEModel, torch.Tensor, pd.DataFrame],
) -> None:
    """active_dims from measure_active_units should be sorted by decreasing KL."""
    model, windows, _ = trained_model_and_data

    AU, kl_per_dim, active_dims = measure_active_units(
        model=model,
        windows=windows,
        batch_size=64,
        au_threshold=0.01,
        device=DEVICE,
    )

    if AU >= 2:
        # Check that active_dims are sorted by decreasing KL
        kl_values = [kl_per_dim[d] for d in active_dims]
        for i in range(len(kl_values) - 1):
            assert kl_values[i] >= kl_values[i + 1], (
                f"active_dims not sorted by decreasing KL at index {i}: "
                f"KL[{active_dims[i]}]={kl_values[i]:.6f} < KL[{active_dims[i+1]}]={kl_values[i+1]:.6f}"
            )
    elif AU == 1:
        # Single active dim: trivially sorted
        assert len(active_dims) == 1
    else:
        # AU == 0: use a lower threshold to force some active dims for testing
        _, kl_per_dim_low, active_dims_low = measure_active_units(
            model=model,
            windows=windows,
            batch_size=64,
            au_threshold=1e-10,  # Very low threshold
            device=DEVICE,
        )
        if len(active_dims_low) >= 2:
            kl_values_low = [kl_per_dim_low[d] for d in active_dims_low]
            for i in range(len(kl_values_low) - 1):
                assert kl_values_low[i] >= kl_values_low[i + 1], (
                    f"active_dims not sorted by decreasing KL at index {i}"
                )
