"""
Integration tests: verify data flows correctly between pipeline stages.

Tests cover stage-to-stage transitions:
  Data → Windows → VAE → Inference → Risk Model → Portfolio

Uses n=100, T_annee=16 to respect the capacity-data constraint r <= r_max=5.0
(ISD Section MOD-002, DVT Appendix A). With c_min=144 (small-universe setting),
K=5 gives r=4.77.

Reference: ISD pipeline flow.
"""

import math
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

from src.data_pipeline.data_loader import generate_synthetic_csv, load_stock_data
from src.data_pipeline.returns import compute_log_returns
from src.data_pipeline.features import compute_rolling_realized_vol
from src.data_pipeline.windowing import create_windows
from src.data_pipeline.universe import construct_universe
from src.vae.build_vae import build_vae
from src.inference.composite import aggregate_profiles, infer_latent_trajectories
from src.inference.active_units import (
    compute_au_max_stat,
    filter_exposure_matrix,
    measure_active_units,
    truncate_active_dims,
)
from src.risk_model.rescaling import rescale_estimation, rescale_portfolio
from src.risk_model.factor_regression import estimate_factor_returns, compute_residuals
from src.risk_model.covariance import assemble_risk_model, estimate_sigma_z, estimate_d_eps
from src.portfolio.sca_solver import multi_start_optimize


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_STOCKS = 100
T_WIN = 64
T_ANNEE = 16
F_VAL = 2
K_VAL = 5
C_MIN = 144
R_MAX = 5.0
DEVICE = torch.device("cpu")


@pytest.fixture(scope="module")
def synthetic_data() -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Generate synthetic stock data and returns."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name

    generate_synthetic_csv(
        output_path=path,
        n_stocks=N_STOCKS,
        start_date="1990-01-03",
        end_date="2006-12-31",
        n_delistings=0,
        seed=42,
    )
    stock_data = load_stock_data(path)
    returns_df = compute_log_returns(stock_data)
    stock_ids = list(returns_df.columns[:N_STOCKS])
    os.unlink(path)
    return returns_df, stock_data, stock_ids


# ---------------------------------------------------------------------------
# Test 36: Data → Windows shapes
# ---------------------------------------------------------------------------

def test_data_to_windows_shapes(
    synthetic_data: tuple[pd.DataFrame, pd.DataFrame, list[int]],
) -> None:
    """Raw data → create_windows → (N, T, 2) without NaN."""
    returns_df, _, stock_ids = synthetic_data

    vol_df = compute_rolling_realized_vol(returns_df, rolling_window=21)

    windows, metadata, raw_returns = create_windows(
        returns_df, vol_df, stock_ids, T=T_WIN, stride=T_WIN,
    )

    if windows.shape[0] == 0:
        pytest.skip("No windows generated (data too short for T)")

    # Shape checks
    assert windows.ndim == 3, f"Expected 3D tensor, got {windows.ndim}D"
    assert windows.shape[1] == T_WIN, f"T mismatch: {windows.shape[1]} != {T_WIN}"
    assert windows.shape[2] == 2, f"F mismatch: {windows.shape[2]} != 2"

    # No NaN
    assert not torch.isnan(windows).any(), "NaN in windows"
    assert not torch.isnan(raw_returns).any(), "NaN in raw_returns"

    # Metadata length matches
    assert len(metadata) == windows.shape[0]

    # FORMULA: CONV-02 — z-score per window per feature: mean≈0, std≈1
    for feat_idx in range(windows.shape[2]):
        feat = windows[:, :, feat_idx]  # (N, T)
        means = feat.mean(dim=1)
        stds = feat.std(dim=1)
        assert torch.all(torch.abs(means) < 1e-4), (
            f"Feature {feat_idx} not zero-mean: max|mean|={means.abs().max():.6f}"
        )
        assert torch.all(torch.abs(stds - 1.0) < 0.05), (
            f"Feature {feat_idx} not unit-std: range [{stds.min():.4f}, {stds.max():.4f}]"
        )


# ---------------------------------------------------------------------------
# Test 37: VAE forward+backward shapes
# ---------------------------------------------------------------------------

def test_vae_forward_backward_shapes(
    synthetic_data: tuple[pd.DataFrame, pd.DataFrame, list[int]],
) -> None:
    """Windows → VAE forward → x_hat (B,T,F), mu (B,K), loss scalar."""
    returns_df, _, stock_ids = synthetic_data

    vol_df = compute_rolling_realized_vol(returns_df, rolling_window=21)
    windows, _, _ = create_windows(
        returns_df, vol_df, stock_ids, T=T_WIN, stride=T_WIN,
    )

    if windows.shape[0] == 0:
        pytest.skip("No windows generated")

    torch.manual_seed(42)
    model, info = build_vae(
        n=N_STOCKS, T=T_WIN, T_annee=T_ANNEE, F=F_VAL, K=K_VAL,
        r_max=R_MAX, c_min=C_MIN,
    )
    model.train()

    batch = windows[:4]
    x_hat, mu, log_var = model(batch)

    # Shape checks
    assert x_hat.shape == batch.shape, (
        f"x_hat shape {x_hat.shape} != input shape {batch.shape}"
    )
    assert mu.shape == (4, K_VAL)
    assert log_var.shape == (4, K_VAL)

    # Loss is scalar
    loss = torch.mean((batch - x_hat) ** 2) + torch.mean(mu ** 2)
    assert loss.ndim == 0, f"Loss should be scalar, got ndim={loss.ndim}"

    # Backward works
    loss.backward()
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "No gradients after backward"

    # FORMULA: INV-002 — log_sigma_sq scalar in [ln(1e-4), ln(10)]
    lss = model.log_sigma_sq.detach()
    assert lss.ndim == 0, f"log_sigma_sq not scalar: ndim={lss.ndim}"
    sigma_sq = torch.exp(lss).item()
    assert 1e-4 <= sigma_sq <= 10.0, (
        f"σ²={sigma_sq} out of clamped range [1e-4, 10]"
    )


# ---------------------------------------------------------------------------
# Test 38: Inference → B matrix
# ---------------------------------------------------------------------------

def test_inference_to_B_matrix(
    synthetic_data: tuple[pd.DataFrame, pd.DataFrame, list[int]],
) -> None:
    """Encoder → infer → B (n, K), AU int, active_dims list."""
    returns_df, _, stock_ids = synthetic_data

    vol_df = compute_rolling_realized_vol(returns_df, rolling_window=21)
    windows, metadata, _ = create_windows(
        returns_df, vol_df, stock_ids, T=T_WIN, stride=T_WIN,
    )

    if windows.shape[0] == 0:
        pytest.skip("No windows generated")

    torch.manual_seed(42)
    model, _ = build_vae(
        n=N_STOCKS, T=T_WIN, T_annee=T_ANNEE, F=F_VAL, K=K_VAL,
        r_max=R_MAX, c_min=C_MIN,
    )
    model.eval()

    trajectories, _ = infer_latent_trajectories(
        model=model, windows=windows, window_metadata=metadata,
        batch_size=64, device=DEVICE,
    )

    B, inferred_ids = aggregate_profiles(trajectories, method="mean")

    assert B.ndim == 2, f"B should be 2D, got {B.ndim}D"
    assert B.shape[1] == K_VAL, f"B columns={B.shape[1]} != K={K_VAL}"
    assert len(inferred_ids) == B.shape[0]
    assert np.all(np.isfinite(B)), "B contains non-finite values"

    # AU measurement
    AU, kl_per_dim, active_dims = measure_active_units(
        model=model, windows=windows, batch_size=64,
        au_threshold=0.01, device=DEVICE,
    )
    assert isinstance(AU, int)
    assert AU >= 0
    assert len(active_dims) == AU


# ---------------------------------------------------------------------------
# Test 39: B → Risk model → Sigma_assets PSD
# ---------------------------------------------------------------------------

def test_B_to_risk_model_to_Sigma() -> None:
    """B → rescale → factor regression → Sigma_assets positive semi-definite."""
    np.random.seed(42)
    rng = np.random.RandomState(42)

    n = 15
    au = 5
    n_dates = 60

    stock_ids = list(range(n))
    dates = pd.bdate_range(start="2020-01-02", periods=n_dates, freq="B")
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]

    # Synthetic B, vol, returns
    B_A = rng.randn(n, au).astype(np.float64) * 0.3
    vol_values = rng.uniform(0.10, 0.50, size=(n_dates, n)).astype(np.float64)
    trailing_vol = pd.DataFrame(vol_values, index=date_strs, columns=stock_ids)
    universe_snapshots = {d: stock_ids[:] for d in date_strs}
    ret_values = rng.randn(n_dates, n).astype(np.float64) * 0.01
    returns = pd.DataFrame(ret_values, index=date_strs, columns=stock_ids)

    # Rescaling
    B_A_by_date = rescale_estimation(B_A, trailing_vol, universe_snapshots, stock_ids)
    B_A_port = rescale_portfolio(
        B_A, trailing_vol, date_strs[-1], stock_ids, stock_ids,
    )

    # Factor regression
    z_hat, valid_dates = estimate_factor_returns(
        B_A_by_date, returns, universe_snapshots,
    )
    assert z_hat.shape[0] > 0, "No valid dates for factor regression"

    # Covariance
    Sigma_z, _ = estimate_sigma_z(z_hat)
    eigenvalues = np.linalg.eigvalsh(Sigma_z)
    assert np.all(eigenvalues >= -1e-10), (
        f"Sigma_z not PSD: min eigenvalue = {eigenvalues.min():.2e}"
    )

    # Residuals
    residuals = compute_residuals(
        B_A_by_date, z_hat, returns, universe_snapshots, valid_dates, stock_ids,
    )
    D_eps = estimate_d_eps(residuals, stock_ids)

    # Assemble
    risk_model = assemble_risk_model(B_A_port, Sigma_z, D_eps)
    Sigma_assets = risk_model["Sigma_assets"]

    assert Sigma_assets.shape == (n, n)
    eig_assets = np.linalg.eigvalsh(Sigma_assets)
    assert np.all(eig_assets >= -1e-8), (
        f"Sigma_assets not PSD: min eigenvalue = {eig_assets.min():.2e}"
    )

    # FORMULA: Σ_assets = BΣ_zB^T + diag(D_ε) — verify assembly numerically
    Sigma_expected = B_A_port @ Sigma_z @ B_A_port.T + np.diag(D_eps)
    assert np.allclose(Sigma_assets, Sigma_expected, atol=1e-10), (
        f"Covariance assembly mismatch: max diff="
        f"{np.abs(Sigma_assets - Sigma_expected).max():.2e}"
    )

    # FORMULA: D_ε ≥ 1e-6 (floor)
    assert np.all(D_eps >= 1e-6 - 1e-12), (
        f"D_eps floor violated: min={D_eps.min():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 40: Sigma → Portfolio weights
# ---------------------------------------------------------------------------

def test_Sigma_to_portfolio_weights() -> None:
    """Sigma_assets → multi_start_optimize → valid weights."""
    np.random.seed(42)
    rng = np.random.RandomState(42)

    n = 15
    au = 5

    B_A_port = rng.randn(n, au).astype(np.float64) * 0.3
    raw = rng.randn(au, au).astype(np.float64) * 0.05
    Sigma_z = raw @ raw.T + np.eye(au) * 0.01

    eigvals, V = np.linalg.eigh(Sigma_z)
    idx = np.argsort(-eigvals)
    eigenvalues = np.maximum(eigvals[idx], 0.0)
    V = V[:, idx]
    B_prime = B_A_port @ V

    D_eps = rng.uniform(0.001, 0.01, size=n).astype(np.float64)
    Sigma_assets = B_A_port @ Sigma_z @ B_A_port.T + np.diag(D_eps)

    w_opt, f_opt, H_opt = multi_start_optimize(
        Sigma_assets=Sigma_assets,
        B_prime=B_prime,
        eigenvalues=eigenvalues,
        D_eps=D_eps,
        alpha=1.0,
        n_starts=3,
        seed=42,
        lambda_risk=1.0,
        phi=25.0,
        w_bar=0.03,
        w_max=0.10,
        is_first=True,
        max_iter=30,
    )

    # Valid weights
    assert w_opt.shape == (n,), f"Weights shape {w_opt.shape} != ({n},)"
    assert np.all(w_opt >= -1e-8), f"Negative weights: min={np.min(w_opt)}"
    assert abs(np.sum(w_opt) - 1.0) < 1e-4, (
        f"Weights sum to {np.sum(w_opt):.6f}, expected 1.0"
    )
    assert np.all(np.isfinite(w_opt)), "Non-finite weights"
    assert np.isfinite(H_opt), "Non-finite entropy"

    # FORMULA: H ∈ [0, max_H] and constraints satisfied
    # Must pass D_eps to match two-layer entropy used by the optimizer
    from src.portfolio.entropy import compute_entropy_and_gradient
    H_verify, grad_H = compute_entropy_and_gradient(
        w_opt, B_prime, eigenvalues, D_eps=D_eps,
    )
    assert abs(H_opt - H_verify) < 1e-6, (
        f"Returned H={H_opt:.6f} != recomputed H={H_verify:.6f}"
    )
    assert H_opt >= -1e-10, f"Entropy H={H_opt} should be ≥ 0"
    # Two-layer max: (1-w_idio)*ln(AU) + w_idio*ln(n)
    max_H = 0.8 * np.log(au) + 0.2 * np.log(n)
    assert H_opt <= max_H + 0.01, (
        f"H={H_opt:.4f} exceeds max_H={max_H:.4f}"
    )

    # FORMULA: INV-012 constraints — w_max check
    assert np.all(w_opt <= 0.10 + 1e-6), (
        f"Weight exceeds w_max=0.10: max w={w_opt.max():.4f}"
    )


# ---------------------------------------------------------------------------
# Test 41: Data → Windows → VAE → Inference → B chain
# ---------------------------------------------------------------------------

def test_data_windows_to_vae_to_inference_chain(
    synthetic_data: tuple[pd.DataFrame, pd.DataFrame, list[int]],
) -> None:
    """Full chain: Data -> Windows -> VAE -> Inference -> B without NaN."""
    returns_df, _, stock_ids = synthetic_data
    vol_df = compute_rolling_realized_vol(returns_df, rolling_window=21)
    windows, metadata, raw_returns = create_windows(
        returns_df, vol_df, stock_ids, T=T_WIN, stride=T_WIN,
    )
    if windows.shape[0] < 5:
        pytest.skip("Not enough windows")
    # No NaN in windows
    assert not torch.isnan(windows).any(), "NaN in windows"
    # Build and run VAE
    torch.manual_seed(42)
    model, _ = build_vae(n=N_STOCKS, T=T_WIN, T_annee=T_ANNEE, F=F_VAL, K=K_VAL, r_max=R_MAX, c_min=C_MIN)
    model.eval()
    # Inference
    trajectories, kl = infer_latent_trajectories(
        model=model, windows=windows, window_metadata=metadata,
        batch_size=64, device=DEVICE,
    )
    B, ids = aggregate_profiles(trajectories, method="mean")
    # No NaN
    assert np.all(np.isfinite(B)), "B contains non-finite values"
    assert B.ndim == 2
    assert B.shape[1] == K_VAL


# ---------------------------------------------------------------------------
# Test 42: Risk model → Portfolio chain
# ---------------------------------------------------------------------------

def test_risk_model_to_portfolio_chain() -> None:
    """B -> rescale -> factor regression -> Sigma -> portfolio weights."""
    rng = np.random.RandomState(42)
    n, au = 20, 5
    stock_ids = list(range(n))
    n_dates = 60
    dates = pd.bdate_range("2020-01-02", periods=n_dates, freq="B")
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]

    B_A = rng.randn(n, au) * 0.3
    vol_values = rng.uniform(0.10, 0.40, (n_dates, n))
    trailing_vol = pd.DataFrame(vol_values, index=date_strs, columns=stock_ids)
    universe_snapshots = {d: stock_ids[:] for d in date_strs}
    ret_values = rng.randn(n_dates, n) * 0.01
    returns = pd.DataFrame(ret_values, index=date_strs, columns=stock_ids)

    # Rescaling
    B_A_by_date = rescale_estimation(B_A, trailing_vol, universe_snapshots, stock_ids)
    B_A_port = rescale_portfolio(B_A, trailing_vol, date_strs[-1], stock_ids, stock_ids)

    # Factor regression
    z_hat, valid_dates = estimate_factor_returns(B_A_by_date, returns, universe_snapshots)
    assert z_hat.shape[0] > 0, "No valid factor returns"

    # Covariance
    Sigma_z, _ = estimate_sigma_z(z_hat)
    residuals = compute_residuals(B_A_by_date, z_hat, returns, universe_snapshots, valid_dates, stock_ids)
    D_eps = estimate_d_eps(residuals, stock_ids)
    risk_model = assemble_risk_model(B_A_port, Sigma_z, D_eps)

    # Portfolio
    w_opt, f_opt, H_opt = multi_start_optimize(
        Sigma_assets=risk_model["Sigma_assets"],
        B_prime=risk_model["B_prime_port"],
        eigenvalues=risk_model["eigenvalues"],
        D_eps=D_eps, alpha=1.0, n_starts=3, seed=42,
        lambda_risk=1.0, phi=25.0, w_bar=0.03, w_max=0.10,
        is_first=True, max_iter=30,
    )
    assert w_opt.shape == (n,)
    assert np.all(w_opt >= -1e-8)
    assert abs(np.sum(w_opt) - 1.0) < 1e-4
    assert np.all(np.isfinite(w_opt))

    # FORMULA: Σ_assets = BΣ_zB^T + D_ε — verify in chain
    Sigma_expected = B_A_port @ Sigma_z @ B_A_port.T + np.diag(D_eps)
    assert np.allclose(
        risk_model["Sigma_assets"], Sigma_expected, atol=1e-10,
    ), "Covariance assembly mismatch in full chain"

    # FORMULA: rotation preserves factor covariance
    B_prime = risk_model["B_prime_port"]
    eigenvalues = risk_model["eigenvalues"]
    cov_original = B_A_port @ Sigma_z @ B_A_port.T
    cov_rotated = B_prime @ np.diag(eigenvalues) @ B_prime.T
    assert np.allclose(cov_original, cov_rotated, atol=1e-10), (
        "Rotation should preserve factor covariance structure"
    )
