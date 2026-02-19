"""
End-to-end integration test: synthetic 100-stock, 16-year pipeline.

Runs the full pipeline on synthetic data to verify no NaN at any stage,
correct types, and basic invariant satisfaction.

Uses n=100, T_annee=16 to respect the capacity-data constraint r <= r_max=5.0
(ISD Section MOD-002, DVT Appendix A). With c_min=144 (small-universe setting
from _adapt_vae_params), K=5 gives r=4.77.

Reference: ISD INV-001 to INV-012.
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
from src.vae.build_vae import build_vae
from src.vae.loss import compute_loss, compute_validation_elbo
from src.training.trainer import VAETrainer
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
from src.portfolio.entropy import compute_entropy_and_gradient
from src.portfolio.sca_solver import multi_start_optimize
from src.portfolio.cardinality import enforce_cardinality


# ---------------------------------------------------------------------------
# Test 41: E2E synthetic pipeline
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_e2e_synthetic_100stocks_16years() -> None:
    """
    Full pipeline on synthetic data: 100 stocks, 16 years.
    Respects capacity-data constraint r <= 5.0 (ISD/DVT).
    Verifies no NaN at any stage, correct types, and key invariants.
    """
    N_STOCKS = 100
    T_ANNEE = 16

    # ---------------------------------------------------------------
    # Stage 1: Data pipeline
    # ---------------------------------------------------------------
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = f.name

    try:
        generate_synthetic_csv(
            output_path=csv_path,
            n_stocks=N_STOCKS,
            start_date="1990-01-03",
            end_date="2006-12-31",
            n_delistings=0,
            seed=42,
        )
        stock_data = load_stock_data(csv_path)
    finally:
        os.unlink(csv_path)

    returns_df = compute_log_returns(stock_data)
    assert not returns_df.empty, "Returns DataFrame is empty"
    assert returns_df.dtypes.apply(
        lambda d: np.issubdtype(d, np.floating)
    ).all(), "Returns not float"

    stock_ids = list(returns_df.columns[:N_STOCKS])

    vol_df = compute_rolling_realized_vol(returns_df, rolling_window=21)

    # ---------------------------------------------------------------
    # Stage 2: Windowing
    # ---------------------------------------------------------------
    T = 64  # Short window for test speed
    windows, metadata, raw_returns = create_windows(
        returns_df, vol_df, stock_ids, T=T, stride=T,
    )

    if windows.shape[0] < 10:
        pytest.skip(f"Only {windows.shape[0]} windows generated, need >= 10")

    assert windows.ndim == 3
    assert windows.shape[1] == T
    assert windows.shape[2] == 2
    assert not torch.isnan(windows).any(), "NaN in windows"
    assert not torch.isnan(raw_returns).any(), "NaN in raw_returns"

    # Z-score check: per-window mean ≈ 0, std ≈ 1 (CONV-02)
    for i in range(min(5, windows.shape[0])):
        for f in range(2):
            feat = windows[i, :, f].numpy()
            assert abs(feat.mean()) < 1e-3, (
                f"CONV-02: Window {i}, feat {f}: mean={feat.mean():.6f} "
                f"(expected < 1e-3)"
            )
            assert abs(feat.std() - 1.0) < 0.05, (
                f"CONV-02: Window {i}, feat {f}: std={feat.std():.6f} "
                f"(expected within 0.05 of 1.0)"
            )

    # ---------------------------------------------------------------
    # Stage 3: VAE build + train
    # ---------------------------------------------------------------
    K = 5
    F_val = 2
    device = torch.device("cpu")

    torch.manual_seed(42)
    model, info = build_vae(
        n=len(stock_ids), T=T, T_annee=T_ANNEE, F=F_val, K=K,
        r_max=5.0, c_min=144,
    )

    # Verify capacity constraint is respected
    assert info["r"] <= 5.0, (
        f"Capacity constraint violated: r={info['r']:.2f} > 5.0"
    )

    # INV-002: sigma_sq is scalar
    assert model.log_sigma_sq.ndim == 0, "INV-002: log_sigma_sq must be scalar"

    n_windows = windows.shape[0]
    split = int(0.8 * n_windows)
    train_windows = windows[:split]
    val_windows = windows[split:]

    trainer = VAETrainer(
        model=model,
        loss_mode="P",
        gamma=1.0,
        learning_rate=1e-3,
        patience=5,
        device=device,
    )

    result = trainer.fit(
        train_windows=train_windows,
        val_windows=val_windows,
        max_epochs=10,
        batch_size=32,
    )

    assert "history" in result
    assert len(result["history"]) > 0

    # sigma_sq in bounds after training
    sigma_sq = model.obs_var.item()
    assert 1e-4 - 1e-8 <= sigma_sq <= 10.0 + 1e-8, (
        f"INV-002: sigma_sq={sigma_sq} outside [1e-4, 10]"
    )

    # ---------------------------------------------------------------
    # Stage 4: Inference
    # ---------------------------------------------------------------
    model.eval()

    trajectories, _ = infer_latent_trajectories(
        model=model, windows=windows, window_metadata=metadata,
        batch_size=64, device=device,
    )

    B, inferred_ids = aggregate_profiles(trajectories, method="mean")
    assert B.ndim == 2
    assert B.shape[1] == K
    assert np.all(np.isfinite(B)), "B contains non-finite values"

    # AU measurement
    AU, kl_per_dim, active_dims = measure_active_units(
        model=model, windows=windows, batch_size=64,
        au_threshold=0.01, device=device,
    )
    assert AU >= 0

    # Truncation
    n_obs = len(metadata)
    au_max = compute_au_max_stat(n_obs, r_min=2)
    AU_final, active_dims_final = truncate_active_dims(
        AU, kl_per_dim, active_dims, au_max,
    )

    if AU_final == 0:
        pytest.fail(
            f"AU=0 after truncation: the VAE produced no active factors "
            f"(AU_raw={AU}, au_max={au_max}, kl_per_dim={kl_per_dim}). "
            f"This indicates the model failed to learn useful latent "
            f"representations on synthetic data. Check training config, "
            f"synthetic data quality, or increase max_epochs."
        )

    B_A = filter_exposure_matrix(B, active_dims_final)
    assert B_A.shape == (len(inferred_ids), AU_final)

    # ---------------------------------------------------------------
    # Stage 5: Risk model
    # ---------------------------------------------------------------
    n_dates = min(60, len(returns_df))
    dates_idx = returns_df.index[-n_dates:]
    date_strs = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime")
                 else str(d)[:10] for d in dates_idx]

    # Build trailing vol and universe snapshots
    vol_values = np.abs(returns_df.iloc[-n_dates:][inferred_ids].values) * 16 + 0.1
    trailing_vol = pd.DataFrame(
        vol_values, index=date_strs, columns=inferred_ids,
    )
    universe_snapshots = {d: list(inferred_ids) for d in date_strs}

    # Rescaling
    B_A_by_date = rescale_estimation(
        B_A, trailing_vol, universe_snapshots, list(inferred_ids),
    )
    B_A_port = rescale_portfolio(
        B_A, trailing_vol, date_strs[-1],
        list(inferred_ids), list(inferred_ids),
    )

    # Factor regression
    ret_sub = returns_df.iloc[-n_dates:][inferred_ids].copy()
    ret_sub.index = date_strs
    z_hat, valid_dates = estimate_factor_returns(
        B_A_by_date, ret_sub, universe_snapshots,
    )

    if z_hat.shape[0] == 0:
        pytest.skip("No valid dates for factor regression")

    Sigma_z, _, _ = estimate_sigma_z(z_hat)
    eigenvalues_z = np.linalg.eigvalsh(Sigma_z)
    assert np.all(eigenvalues_z >= -1e-10), (
        f"INV-007: Sigma_z not PSD: min eigenvalue = {eigenvalues_z.min():.2e}"
    )

    residuals = compute_residuals(
        B_A_by_date, z_hat, ret_sub, universe_snapshots, valid_dates, list(inferred_ids),
    )
    D_eps = estimate_d_eps(residuals, list(inferred_ids))

    risk_model = assemble_risk_model(B_A_port, Sigma_z, D_eps)
    Sigma_assets = risk_model["Sigma_assets"]
    eigenvalues = risk_model["eigenvalues"]
    B_prime = risk_model["B_prime_port"]

    # INV-007: eigenvalues non-negative
    assert np.all(eigenvalues >= -1e-10), (
        f"INV-007: eigenvalues not non-negative: min={eigenvalues.min():.2e}"
    )

    # Sigma_assets PSD
    eig_assets = np.linalg.eigvalsh(Sigma_assets)
    assert np.all(eig_assets >= -1e-8), (
        f"Sigma_assets not PSD: min eigenvalue = {eig_assets.min():.2e}"
    )

    # ---------------------------------------------------------------
    # Stage 6: Portfolio optimization
    # ---------------------------------------------------------------
    n_active = len(inferred_ids)

    w_opt, f_opt, H_opt, _ = multi_start_optimize(
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

    # Valid portfolio
    assert w_opt.shape == (n_active,), f"Weights shape mismatch: {w_opt.shape}"
    assert np.all(w_opt >= -1e-8), f"Negative weight: min={np.min(w_opt)}"
    assert abs(np.sum(w_opt) - 1.0) < 1e-4, (
        f"Weights sum to {np.sum(w_opt):.6f}"
    )
    assert np.all(np.isfinite(w_opt)), "Non-finite weights"

    # Entropy in valid range
    H_check, _ = compute_entropy_and_gradient(w_opt, B_prime, eigenvalues)
    assert H_check >= 0.0, f"Entropy negative: {H_check}"
    if AU_final > 1:
        assert H_check <= np.log(AU_final) + 1e-6, (
            f"Entropy {H_check} > ln(AU)={np.log(AU_final)}"
        )

    # ---------------------------------------------------------------
    # Summary: all stages passed without NaN or type errors
    # ---------------------------------------------------------------
    assert True, "E2E pipeline completed successfully"


# ---------------------------------------------------------------------------
# Test 42: E2E benchmarks complete
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_e2e_benchmarks_complete() -> None:
    """All 6 benchmarks produce valid weights on synthetic data (INV-012)."""
    from src.benchmarks.equal_weight import EqualWeight
    from src.benchmarks.inverse_vol import InverseVolatility
    from src.benchmarks.min_variance import MinimumVariance
    from src.benchmarks.erc import EqualRiskContribution
    from src.benchmarks.pca_factor_rp import PCAFactorRiskParity
    from src.benchmarks.pca_vol import PCAVolRiskParity

    rng = np.random.RandomState(42)
    n = 30
    dates = pd.bdate_range("2020-01-01", periods=252, freq="B")
    returns = pd.DataFrame(
        rng.normal(0.0005, 0.02, (252, n)),
        index=dates, columns=list(range(n)),
    )
    stock_ids = list(range(n))
    trailing_vol = pd.DataFrame(
        np.abs(rng.randn(252, n)) * 0.02 + 0.15,
        index=dates, columns=stock_ids,
    )
    current_date = str(dates[-1].date())  # type: ignore[union-attr]

    w_max = 0.05
    w_min = 0.001
    constraint_params: dict[str, float] = {
        "w_max": w_max,
        "w_min": w_min,
        "phi": 25.0,
        "kappa_1": 0.1,
        "kappa_2": 7.5,
        "delta_bar": 0.01,
        "tau_max": 0.30,
        "lambda_risk": 1.0,
    }

    all_benchmarks = [
        EqualWeight(constraint_params=constraint_params),
        InverseVolatility(constraint_params=constraint_params),
        MinimumVariance(constraint_params=constraint_params),
        EqualRiskContribution(constraint_params=constraint_params),
        PCAFactorRiskParity(constraint_params=constraint_params),
        PCAVolRiskParity(constraint_params=constraint_params),
    ]

    for bench in all_benchmarks:
        name = bench.__class__.__name__
        bench.fit(returns, stock_ids, trailing_vol=trailing_vol,
                  current_date=current_date)
        w = bench.optimize(is_first=True)

        # Shape and finiteness
        assert w.shape == (n,), f"{name}: shape {w.shape}"
        assert np.all(np.isfinite(w)), f"{name}: non-finite weights"

        # Fully invested
        assert abs(np.sum(w) - 1.0) < 1e-4, (
            f"{name}: sum={np.sum(w):.6f}, expected 1.0"
        )
        # Long-only
        assert np.all(w >= -1e-8), f"{name}: negative weights, min={np.min(w)}"

        # INV-012: w_max constraint
        assert np.max(w) <= w_max + 1e-6, (
            f"{name}: max weight {np.max(w):.6f} exceeds w_max={w_max}"
        )

        # INV-012: Semi-continuous constraint (w_i = 0 or w_i >= w_min)
        for i, wi in enumerate(w):
            assert wi < 1e-6 or wi >= w_min - 1e-6, (
                f"{name}: w[{i}]={wi:.8f} violates semi-continuous "
                f"(must be 0 or >= {w_min})"
            )
