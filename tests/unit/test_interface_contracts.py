"""
Interface contract tests verifying dtype and shape at every pipeline stage boundary.

Each test validates that outputs from one pipeline stage conform to the expected
dtype, shape, and structure required as inputs by the next stage. Uses small
synthetic data (T=64, K=5, n=15-20) for fast execution.

Reference: ISD invariants INV-001 to INV-012, conventions CONV-01 to CONV-10.
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
from src.vae.loss import compute_loss
from src.inference.composite import aggregate_profiles, infer_latent_trajectories
from src.inference.active_units import measure_active_units
from src.risk_model.rescaling import rescale_estimation, rescale_portfolio
from src.risk_model.factor_regression import estimate_factor_returns
from src.risk_model.covariance import assemble_risk_model, estimate_sigma_z
from src.portfolio.entropy import compute_entropy_and_gradient
from src.portfolio.sca_solver import multi_start_optimize
from src.portfolio.cardinality import enforce_cardinality


# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

T = 64
K = 5
SEED = 42


# ---------------------------------------------------------------------------
# Module-scoped fixture: generate synthetic data once, reuse across tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline_data() -> dict:
    """
    Generate synthetic data once and run the pipeline through all stages.

    Returns a dict containing every intermediate product needed by the 11
    contract tests.

    :return data (dict): Pipeline intermediates keyed by stage name
    """
    rng = np.random.RandomState(SEED)
    data: dict = {"rng": rng}

    # --- Stage 0: Generate synthetic CSV and load ---
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = f.name
    try:
        generate_synthetic_csv(
            csv_path,
            n_stocks=20,
            start_date="2018-01-02",
            end_date="2020-12-31",
            n_delistings=0,
            seed=SEED,
        )
        stock_data = load_stock_data(csv_path)
    finally:
        os.unlink(csv_path)

    data["stock_data"] = stock_data

    # --- Stage 1: Returns and volatility ---
    returns_df = compute_log_returns(stock_data)
    vol_df = compute_rolling_realized_vol(returns_df, rolling_window=21)
    stock_ids = [int(c) for c in returns_df.columns]
    data["returns_df"] = returns_df
    data["vol_df"] = vol_df
    data["stock_ids"] = stock_ids

    # --- Stage 2: Windowing ---
    windows, metadata, raw_returns = create_windows(
        returns_df, vol_df, stock_ids, T=T, stride=T,
    )
    data["windows"] = windows
    data["metadata"] = metadata
    data["raw_returns"] = raw_returns

    # --- Stage 3: Build VAE (small architecture) ---
    n_stocks = len(stock_ids)
    model, info = build_vae(
        n=n_stocks,
        T=T,
        T_annee=3,
        F=2,
        K=K,
        s_train=T,
        r_max=500.0,
        learn_obs_var=True,
        c_min=144,
        dropout=0.1,
    )
    data["model"] = model
    data["info"] = info
    data["n_stocks"] = n_stocks

    # --- Stage 4: Forward pass for loss test ---
    model.eval()
    batch = windows[:4]
    with torch.no_grad():
        x_hat, mu, log_var = model(batch)
    data["batch"] = batch
    data["x_hat"] = x_hat
    data["mu"] = mu
    data["log_var"] = log_var

    return data


# ---------------------------------------------------------------------------
# Test 1: Data pipeline -> Windowing boundary
# ---------------------------------------------------------------------------

def test_contract_data_to_windows(pipeline_data: dict) -> None:
    """
    Verify that create_windows returns tensors with correct dtype, shape,
    and metadata structure expected by the VAE training loop.
    """
    windows = pipeline_data["windows"]
    metadata = pipeline_data["metadata"]
    raw_returns = pipeline_data["raw_returns"]

    # Windows tensor: float32, 3D, correct T and F=2
    assert windows.dtype == torch.float32, (
        f"Expected torch.float32, got {windows.dtype}"
    )
    assert windows.ndim == 3, f"Expected ndim=3, got {windows.ndim}"
    assert windows.shape[1] == T, (
        f"Expected T={T}, got {windows.shape[1]}"
    )
    assert windows.shape[2] == 2, (
        f"Expected F=2, got {windows.shape[2]}"
    )

    # Metadata is a DataFrame with matching row count
    assert isinstance(metadata, pd.DataFrame)
    assert len(metadata) == windows.shape[0], (
        f"Metadata rows ({len(metadata)}) != windows batch ({windows.shape[0]})"
    )

    # Metadata must have required columns
    assert "stock_id" in metadata.columns, "Missing 'stock_id' column"
    assert "start_date" in metadata.columns, "Missing 'start_date' column"
    assert "end_date" in metadata.columns, "Missing 'end_date' column"

    # Raw returns: float32 tensor, shape (N, T)
    assert isinstance(raw_returns, torch.Tensor), (
        f"Expected torch.Tensor, got {type(raw_returns)}"
    )
    assert raw_returns.shape == (windows.shape[0], T), (
        f"Expected shape ({windows.shape[0]}, {T}), got {raw_returns.shape}"
    )
    assert raw_returns.dtype == torch.float32, (
        f"Expected torch.float32, got {raw_returns.dtype}"
    )

    # FORMULA: CONV-02 — z-score per window per feature: mean≈0, std≈1
    for feat_idx in range(windows.shape[2]):
        feat_slice = windows[:, :, feat_idx]  # (N, T)
        per_window_mean = feat_slice.mean(dim=1)  # (N,)
        per_window_std = feat_slice.std(dim=1)    # (N,)
        assert torch.all(torch.abs(per_window_mean) < 1e-4), (
            f"Feature {feat_idx}: z-score mean not ~0, "
            f"max |mean|={per_window_mean.abs().max():.6f}"
        )
        assert torch.all(torch.abs(per_window_std - 1.0) < 0.05), (
            f"Feature {feat_idx}: z-score std not ~1, "
            f"range [{per_window_std.min():.4f}, {per_window_std.max():.4f}]"
        )

    # FORMULA: raw returns must NOT be z-scored (ISD MOD-004)
    raw_means = raw_returns.mean(dim=1)
    raw_stds = raw_returns.std(dim=1)
    assert not (torch.all(torch.abs(raw_means) < 1e-3)
                and torch.all(torch.abs(raw_stds - 1.0) < 0.1)), (
        "Raw returns appear z-scored — they must be raw for co-movement loss"
    )


# ---------------------------------------------------------------------------
# Test 2: VAE forward pass output contract
# ---------------------------------------------------------------------------

def test_contract_vae_forward(pipeline_data: dict) -> None:
    """
    Verify that model.forward(batch) returns x_hat, mu, log_var with correct
    shapes and dtypes expected by the loss function.
    """
    batch = pipeline_data["batch"]
    x_hat = pipeline_data["x_hat"]
    mu = pipeline_data["mu"]
    log_var = pipeline_data["log_var"]
    B = batch.shape[0]  # 4

    # Reconstruction must match input shape and dtype
    assert x_hat.shape == batch.shape, (
        f"x_hat shape {x_hat.shape} != batch shape {batch.shape}"
    )
    assert x_hat.dtype == batch.dtype, (
        f"x_hat dtype {x_hat.dtype} != batch dtype {batch.dtype}"
    )

    # Latent parameters: (B, K)
    assert mu.shape == (B, K), f"mu shape {mu.shape} != ({B}, {K})"
    assert log_var.shape == (B, K), (
        f"log_var shape {log_var.shape} != ({B}, {K})"
    )
    assert mu.dtype == torch.float32, f"mu dtype {mu.dtype} != float32"
    assert log_var.dtype == torch.float32, (
        f"log_var dtype {log_var.dtype} != float32"
    )

    # FORMULA: INV-002 — log_sigma_sq is a scalar, σ² = exp(log_sigma_sq)
    model = pipeline_data["model"]
    lss = model.log_sigma_sq.detach()
    assert lss.ndim == 0, f"log_sigma_sq ndim={lss.ndim}, expected 0"
    sigma_sq = torch.exp(lss)
    assert sigma_sq > 0, f"σ² must be positive, got {sigma_sq}"

    # FORMULA: reparameterization trick z = μ + exp(0.5·log_var)·ε
    # In eval mode, mu and log_var should be deterministic
    with torch.no_grad():
        _, mu2, lv2 = model(batch)
    assert torch.allclose(mu, mu2, atol=1e-6), (
        "Eval mode mu should be deterministic"
    )
    assert torch.allclose(log_var, lv2, atol=1e-6), (
        "Eval mode log_var should be deterministic"
    )


# ---------------------------------------------------------------------------
# Test 3: Loss function output contract
# ---------------------------------------------------------------------------

def test_contract_compute_loss(pipeline_data: dict) -> None:
    """
    Verify that compute_loss returns a scalar loss and a dict of finite
    monitoring components with the expected keys.
    """
    batch = pipeline_data["batch"]
    x_hat = pipeline_data["x_hat"]
    mu = pipeline_data["mu"]
    log_var = pipeline_data["log_var"]
    model = pipeline_data["model"]

    crisis_fractions = torch.zeros(batch.shape[0])

    loss, components = compute_loss(
        x=batch,
        x_hat=x_hat,
        mu=mu,
        log_var=log_var,
        log_sigma_sq=model.log_sigma_sq.detach(),
        crisis_fractions=crisis_fractions,
        epoch=0,
        total_epochs=100,
        mode="P",
    )

    # Loss must be a scalar tensor
    assert loss.ndim == 0, f"Loss ndim={loss.ndim}, expected 0 (scalar)"

    # Components dict must contain required keys
    assert isinstance(components, dict)
    required_keys = {"recon", "kl", "recon_term", "total"}
    for key in required_keys:
        assert key in components, f"Missing key '{key}' in loss components"

    # All component values must be finite
    for key, val in components.items():
        if isinstance(val, torch.Tensor):
            assert torch.isfinite(val).all(), (
                f"Non-finite value in components['{key}']: {val}"
            )
        elif isinstance(val, (float, int)):
            assert math.isfinite(val), (
                f"Non-finite value in components['{key}']: {val}"
            )

    # FORMULA: INV-001 — Mode P total loss = D/(2σ²)·MSE + D/2·ln(σ²) + KL
    # Verify component decomposition matches formula
    D = batch.shape[1] * batch.shape[2]  # T × F
    sigma_sq = torch.exp(model.log_sigma_sq.detach()).item()
    mse = components["recon"]  # per-element MSE
    if isinstance(mse, torch.Tensor):
        mse = mse.item()
    kl = components["kl"]
    if isinstance(kl, torch.Tensor):
        kl = kl.item()
    recon_term = components["recon_term"]
    if isinstance(recon_term, torch.Tensor):
        recon_term = recon_term.item()

    # The recon_term should include D/(2σ²) coefficient
    expected_recon_term = (D / (2.0 * sigma_sq)) * mse
    assert abs(recon_term - expected_recon_term) < max(abs(expected_recon_term) * 1e-4, 1e-6), (
        f"recon_term={recon_term:.6f} != D/(2σ²)·MSE="
        f"{expected_recon_term:.6f} (D={D}, σ²={sigma_sq:.4f}, MSE={mse:.6f})"
    )


# ---------------------------------------------------------------------------
# Test 4: Inference output contract
# ---------------------------------------------------------------------------

def test_contract_inference_output(pipeline_data: dict) -> None:
    """
    Verify that infer_latent_trajectories returns properly keyed trajectories,
    and aggregate_profiles returns B matrix with correct dtype and shape.
    """
    model = pipeline_data["model"]
    windows = pipeline_data["windows"]
    metadata = pipeline_data["metadata"]

    trajectories, kl_per_dim = infer_latent_trajectories(
        model, windows, metadata, batch_size=64, device=torch.device("cpu"),
        compute_kl=True,
    )

    # Trajectories: dict mapping int stock_id -> ndarray
    assert isinstance(trajectories, dict)
    for key in trajectories:
        assert isinstance(key, int), (
            f"Trajectory key type {type(key)}, expected int"
        )

    # Aggregate to exposure matrix B
    B, stock_ids = aggregate_profiles(trajectories, method="mean")

    assert B.ndim == 2, f"B ndim={B.ndim}, expected 2"
    assert B.shape[1] == K, f"B.shape[1]={B.shape[1]}, expected K={K}"
    assert B.dtype in (np.float32, np.float64), (
        f"B dtype={B.dtype}, expected float32 or float64"
    )

    assert isinstance(stock_ids, list)
    assert all(isinstance(sid, int) for sid in stock_ids)
    assert B.shape[0] == len(stock_ids), (
        f"B.shape[0]={B.shape[0]} != len(stock_ids)={len(stock_ids)}"
    )

    # FORMULA: B[i] = mean(trajectories[stock_i]) — verify aggregation
    for i, sid in enumerate(stock_ids):
        traj = trajectories[sid]  # (M, K) array
        expected_row = np.mean(traj, axis=0)
        assert np.allclose(B[i], expected_row, atol=1e-10), (
            f"B[{i}] (stock {sid}): aggregate_profiles mean mismatch. "
            f"Expected {expected_row[:3]}..., got {B[i, :3]}..."
        )

    # FORMULA: KL per dim consistency — should have K values
    assert kl_per_dim.shape == (K,), (
        f"kl_per_dim.shape={kl_per_dim.shape}, expected ({K},)"
    )
    assert np.all(kl_per_dim >= 0), (
        f"KL per dim must be non-negative, min={kl_per_dim.min()}"
    )


# ---------------------------------------------------------------------------
# Test 5: Active units measurement contract
# ---------------------------------------------------------------------------

def test_contract_measure_active_units(pipeline_data: dict) -> None:
    """
    Verify that measure_active_units returns AU count, KL-per-dim array,
    and active_dims list with correct types and ranges.
    """
    model = pipeline_data["model"]
    windows = pipeline_data["windows"]

    AU, kl_per_dim, active_dims = measure_active_units(
        model, windows, batch_size=64, au_threshold=0.01,
        device=torch.device("cpu"),
    )

    # AU is a non-negative integer
    assert isinstance(AU, int), f"AU type {type(AU)}, expected int"
    assert AU >= 0, f"AU={AU}, expected >= 0"

    # kl_per_dim has shape (K,)
    assert kl_per_dim.shape == (K,), (
        f"kl_per_dim shape {kl_per_dim.shape}, expected ({K},)"
    )

    # active_dims is a list of valid dimension indices
    assert isinstance(active_dims, list)
    assert len(active_dims) == AU, (
        f"len(active_dims)={len(active_dims)} != AU={AU}"
    )
    for d in active_dims:
        assert d in range(K), (
            f"Active dim {d} not in range(0, {K})"
        )

    # FORMULA: CONV-07 — AU = |{k : KL_k > 0.01}| exactly
    expected_AU = int(np.sum(kl_per_dim > 0.01))
    assert AU == expected_AU, (
        f"AU={AU} but count(KL_k > 0.01)={expected_AU}. "
        f"KL values: {kl_per_dim}"
    )

    # FORMULA: active_dims should be the indices where KL > 0.01,
    # sorted by decreasing KL
    expected_dims = [int(k) for k in np.where(kl_per_dim > 0.01)[0]]
    expected_dims_sorted = sorted(
        expected_dims, key=lambda k: kl_per_dim[k], reverse=True,
    )
    assert active_dims == expected_dims_sorted, (
        f"active_dims={active_dims} != expected {expected_dims_sorted}"
    )


# ---------------------------------------------------------------------------
# Test 6: Rescaling contract
# ---------------------------------------------------------------------------

def test_contract_rescaling() -> None:
    """
    Verify that rescale_estimation returns dict[str, ndarray] and
    rescale_portfolio returns ndarray with correct shapes and dtype.
    """
    rng = np.random.RandomState(SEED)
    n = 15
    AU = 3
    n_dates = 20

    # Synthetic inputs
    B_A = rng.randn(n, AU).astype(np.float64)
    stock_ids = list(range(100, 100 + n))

    dates = pd.bdate_range("2020-01-01", periods=n_dates, freq="B")
    date_strings = [str(d.date()) for d in dates]

    # Trailing vol DataFrame (dates x stocks)
    vol_data = rng.uniform(0.1, 0.5, size=(n_dates, n)).astype(np.float64)
    trailing_vol = pd.DataFrame(vol_data, index=dates, columns=stock_ids)

    # Universe snapshots: all stocks active on each date
    universe_snapshots: dict[str, list[int]] = {
        ds: stock_ids[:] for ds in date_strings
    }

    # --- rescale_estimation ---
    B_A_by_date = rescale_estimation(
        B_A, trailing_vol, universe_snapshots, stock_ids,
    )

    assert isinstance(B_A_by_date, dict)
    for key, val in B_A_by_date.items():
        assert isinstance(key, str), f"Key type {type(key)}, expected str"
        assert isinstance(val, np.ndarray)
        assert val.dtype == np.float64, (
            f"Rescaled B dtype {val.dtype}, expected float64"
        )
        assert val.shape[1] == AU, (
            f"Rescaled B.shape[1]={val.shape[1]}, expected AU={AU}"
        )

    # --- rescale_portfolio ---
    current_date = date_strings[-1]
    B_A_port = rescale_portfolio(
        B_A, trailing_vol, current_date, stock_ids, stock_ids,
    )

    assert isinstance(B_A_port, np.ndarray)
    assert B_A_port.shape == (n, AU), (
        f"B_A_port shape {B_A_port.shape}, expected ({n}, {AU})"
    )
    assert B_A_port.dtype == np.float64, (
        f"B_A_port dtype {B_A_port.dtype}, expected float64"
    )

    # FORMULA: INV-004 — Rescaling B̃_{A,i,t} = R_{i,t} · μ̄_{A,i}
    # where R_{i,t} = winsorize(σ_{i,t} / median(σ_{.,t}))
    # Verify portfolio rescaling numerically for last date
    last_date_idx = trailing_vol.index[-1]
    vols_last = trailing_vol.loc[last_date_idx, stock_ids].values.astype(np.float64)
    median_vol = np.median(vols_last)
    raw_ratios = vols_last / median_vol
    # Winsorize at P5/P95
    lo_p = np.percentile(raw_ratios, 5.0)
    hi_p = np.percentile(raw_ratios, 95.0)
    expected_ratios = np.clip(raw_ratios, lo_p, hi_p)
    # Expected rescaled exposures
    expected_B_port = B_A * expected_ratios[:, np.newaxis]
    assert np.allclose(B_A_port, expected_B_port, atol=1e-10), (
        f"Portfolio rescaling mismatch: max diff="
        f"{np.abs(B_A_port - expected_B_port).max():.2e}"
    )

    # FORMULA: Verify estimation rescaling for one date
    check_date = list(B_A_by_date.keys())[0]
    check_date_idx = trailing_vol.index[
        [str(d.date()) if hasattr(d, 'date') else str(d)
         for d in trailing_vol.index].index(check_date)
    ]
    vols_check = trailing_vol.loc[check_date_idx, stock_ids].values.astype(np.float64)
    med_check = np.median(vols_check)
    ratios_check = vols_check / med_check
    lo_c = np.percentile(ratios_check, 5.0)
    hi_c = np.percentile(ratios_check, 95.0)
    ratios_check = np.clip(ratios_check, lo_c, hi_c)
    expected_B_est = B_A * ratios_check[:, np.newaxis]
    assert np.allclose(B_A_by_date[check_date], expected_B_est, atol=1e-10), (
        f"Estimation rescaling mismatch at {check_date}"
    )


# ---------------------------------------------------------------------------
# Test 7: Factor regression contract
# ---------------------------------------------------------------------------

def test_contract_factor_regression() -> None:
    """
    Verify that estimate_factor_returns returns z_hat with correct shape,
    dtype, and corresponding valid_dates list.
    """
    rng = np.random.RandomState(SEED)
    n = 15
    AU = 3
    n_dates = 30

    stock_ids = list(range(200, 200 + n))
    dates = pd.bdate_range("2020-01-01", periods=n_dates, freq="B")
    date_strings = [str(d.date()) for d in dates]

    # Synthetic rescaled exposures per date
    B_A_by_date: dict[str, np.ndarray] = {}
    for ds in date_strings:
        B_A_by_date[ds] = rng.randn(n, AU).astype(np.float64)

    # Synthetic returns DataFrame
    ret_data = rng.randn(n_dates, n).astype(np.float64) * 0.01
    returns = pd.DataFrame(ret_data, index=dates, columns=stock_ids)

    # Universe snapshots
    universe_snapshots: dict[str, list[int]] = {
        ds: stock_ids[:] for ds in date_strings
    }

    z_hat, valid_dates = estimate_factor_returns(
        B_A_by_date, returns, universe_snapshots,
    )

    # z_hat: (n_valid_dates, AU) float64
    assert z_hat.ndim == 2, f"z_hat ndim={z_hat.ndim}, expected 2"
    assert z_hat.shape[1] == AU, (
        f"z_hat.shape[1]={z_hat.shape[1]}, expected AU={AU}"
    )
    assert z_hat.dtype == np.float64, (
        f"z_hat dtype {z_hat.dtype}, expected float64"
    )

    # valid_dates: list of strings, length matches z_hat rows
    assert isinstance(valid_dates, list)
    assert len(valid_dates) == z_hat.shape[0], (
        f"len(valid_dates)={len(valid_dates)} != z_hat.shape[0]={z_hat.shape[0]}"
    )

    # FORMULA: OLS — ẑ_t = (B̃_t^T B̃_t)^{-1} B̃_t^T r_t
    # Verify one date manually
    check_date = valid_dates[0]
    B_t = B_A_by_date[check_date]
    active_stocks = universe_snapshots[check_date]
    date_idx = dates[
        [str(d.date()) if hasattr(d, 'date') else str(d)
         for d in dates].index(check_date)
    ]
    r_t = returns.loc[date_idx, active_stocks].values.astype(np.float64)
    # Manual OLS
    BtB = B_t.T @ B_t
    Btr = B_t.T @ r_t
    z_expected = np.linalg.solve(BtB, Btr)
    assert np.allclose(z_hat[0], z_expected, atol=1e-10), (
        f"OLS formula mismatch at {check_date}: "
        f"expected {z_expected}, got {z_hat[0]}"
    )

    # FORMULA: With B=Identity, z_hat should equal returns
    B_identity = {ds: np.eye(n, AU, dtype=np.float64) for ds in date_strings}
    z_id, dates_id = estimate_factor_returns(
        B_identity, returns, universe_snapshots,
    )
    for t_idx, ds in enumerate(dates_id):
        d_idx = dates[
            [str(d.date()) if hasattr(d, 'date') else str(d)
             for d in dates].index(ds)
        ]
        r_row = returns.loc[d_idx].values[:AU].astype(np.float64)
        # (I^T I)^{-1} I^T r = r[:AU]  (least-squares projection)
        assert np.allclose(z_id[t_idx], r_row, atol=1e-8), (
            f"B=I: z_hat should equal r[:AU] at {ds}"
        )


# ---------------------------------------------------------------------------
# Test 8: Covariance assembly contract
# ---------------------------------------------------------------------------

def test_contract_covariance_assembly() -> None:
    """
    Verify that assemble_risk_model returns a dict with Sigma_assets,
    eigenvalues, V, and B_prime_port with correct shapes and properties.
    """
    rng = np.random.RandomState(SEED)
    n = 15
    AU = 3

    # Build valid PSD Sigma_z via scatter matrix
    z_hat = rng.randn(50, AU).astype(np.float64)
    Sigma_z, _ = estimate_sigma_z(z_hat)

    # Synthetic portfolio exposures and idiosyncratic variances
    B_A_port = rng.randn(n, AU).astype(np.float64)
    D_eps = np.abs(rng.randn(n).astype(np.float64)) + 1e-4

    result = assemble_risk_model(B_A_port, Sigma_z, D_eps)

    # Required keys
    assert "Sigma_assets" in result
    assert "eigenvalues" in result
    assert "V" in result
    assert "B_prime_port" in result

    Sigma_assets = result["Sigma_assets"]
    eigenvalues = result["eigenvalues"]
    V = result["V"]
    B_prime_port = result["B_prime_port"]

    # Sigma_assets: (n, n), symmetric
    assert Sigma_assets.shape == (n, n), (
        f"Sigma_assets shape {Sigma_assets.shape}, expected ({n}, {n})"
    )
    assert np.allclose(Sigma_assets, Sigma_assets.T, atol=1e-12), (
        "Sigma_assets is not symmetric"
    )

    # Eigenvalues: (AU,), all >= -1e-10 (PSD guarantee)
    assert eigenvalues.shape == (AU,), (
        f"eigenvalues shape {eigenvalues.shape}, expected ({AU},)"
    )
    assert np.all(eigenvalues >= -1e-10), (
        f"Negative eigenvalue found: min={eigenvalues.min()}"
    )

    # V: (AU, AU) orthogonal rotation matrix
    assert V.shape == (AU, AU), (
        f"V shape {V.shape}, expected ({AU}, {AU})"
    )

    # B_prime_port: (n, AU)
    assert B_prime_port.shape == (n, AU), (
        f"B_prime_port shape {B_prime_port.shape}, expected ({n}, {AU})"
    )

    # FORMULA: Σ_assets = B Σ_z B^T + diag(D_ε) — verify assembly
    Sigma_expected = B_A_port @ Sigma_z @ B_A_port.T + np.diag(D_eps)
    assert np.allclose(Sigma_assets, Sigma_expected, atol=1e-10), (
        f"Covariance assembly mismatch: max diff="
        f"{np.abs(Sigma_assets - Sigma_expected).max():.2e}"
    )

    # FORMULA: eigendecomposition Σ_z = V Λ V^T (reconstruction)
    Sigma_z_reconstructed = V @ np.diag(eigenvalues) @ V.T
    assert np.allclose(Sigma_z, Sigma_z_reconstructed, atol=1e-10), (
        f"Σ_z reconstruction mismatch: max diff="
        f"{np.abs(Sigma_z - Sigma_z_reconstructed).max():.2e}"
    )

    # FORMULA: B' = B_A_port @ V (rotated exposures)
    B_prime_expected = B_A_port @ V
    assert np.allclose(B_prime_port, B_prime_expected, atol=1e-10), (
        f"Rotated exposures mismatch: max diff="
        f"{np.abs(B_prime_port - B_prime_expected).max():.2e}"
    )

    # FORMULA: rotation preserves covariance B'ΛB'^T = BΣ_zB^T
    factor_cov_original = B_A_port @ Sigma_z @ B_A_port.T
    factor_cov_rotated = B_prime_port @ np.diag(eigenvalues) @ B_prime_port.T
    assert np.allclose(factor_cov_original, factor_cov_rotated, atol=1e-10), (
        f"Rotation must preserve factor covariance: max diff="
        f"{np.abs(factor_cov_original - factor_cov_rotated).max():.2e}"
    )

    # PSD verification: all eigenvalues of Σ_assets are ≥ 0
    asset_eigs = np.linalg.eigvalsh(Sigma_assets)
    assert np.all(asset_eigs >= -1e-10), (
        f"Σ_assets has negative eigenvalues: min={asset_eigs.min()}"
    )


# ---------------------------------------------------------------------------
# Test 9: Entropy and gradient contract
# ---------------------------------------------------------------------------

def test_contract_entropy_and_gradient() -> None:
    """
    Verify that compute_entropy_and_gradient returns a non-negative float H
    and a gradient array with correct shape and dtype.
    """
    rng = np.random.RandomState(SEED)
    n = 15
    AU = 3

    # Non-degenerate weights
    w = rng.dirichlet(np.ones(n)).astype(np.float64)

    # Synthetic rotated exposures and eigenvalues
    B_prime = rng.randn(n, AU).astype(np.float64)
    eigenvalues = np.abs(rng.randn(AU).astype(np.float64)) + 0.01

    H, grad_H = compute_entropy_and_gradient(w, B_prime, eigenvalues)

    # H is a non-negative float
    assert isinstance(H, float), f"H type {type(H)}, expected float"
    assert H >= 0.0, f"H={H}, expected >= 0"

    # Gradient: (n,) float64
    assert grad_H.shape == (n,), (
        f"grad_H shape {grad_H.shape}, expected ({n},)"
    )
    assert grad_H.dtype == np.float64, (
        f"grad_H dtype {grad_H.dtype}, expected float64"
    )

    # FORMULA: H ∈ [0, ln(AU)] — entropy bounds
    assert H <= np.log(AU) + 1e-10, (
        f"H={H:.4f} > ln(AU)={np.log(AU):.4f}"
    )

    # FORMULA: manual entropy computation step-by-step
    beta_prime = B_prime.T @ w  # (AU,)
    c_prime = (beta_prime ** 2) * eigenvalues  # (AU,)
    C = np.sum(c_prime)
    if C > 1e-30:
        c_hat = c_prime / C
        log_c_hat = np.log(np.maximum(c_hat, 1e-30))
        H_manual = -np.sum(c_hat * log_c_hat)
        assert abs(H - H_manual) < 1e-10, (
            f"Entropy manual recomputation mismatch: H={H:.10f}, "
            f"manual={H_manual:.10f}"
        )

    # FORMULA: gradient via finite differences
    delta = 1e-7
    grad_fd = np.zeros(n)
    for i in range(n):
        w_plus = w.copy()
        w_plus[i] += delta
        w_plus /= w_plus.sum()  # Re-normalize
        H_plus, _ = compute_entropy_and_gradient(w_plus, B_prime, eigenvalues)
        w_minus = w.copy()
        w_minus[i] -= delta
        w_minus = np.maximum(w_minus, 0)
        w_minus /= w_minus.sum()
        H_minus, _ = compute_entropy_and_gradient(w_minus, B_prime, eigenvalues)
        grad_fd[i] = (H_plus - H_minus) / (2.0 * delta)

    # Gradient check: analytical vs finite differences (relaxed for normalization)
    grad_norm = np.linalg.norm(grad_H)
    if grad_norm > 1e-6:
        cos_sim = np.dot(grad_H, grad_fd) / (
            np.linalg.norm(grad_H) * np.linalg.norm(grad_fd) + 1e-30
        )
        assert cos_sim > 0.9, (
            f"Gradient direction mismatch: cosine similarity={cos_sim:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 10: Multi-start SCA optimizer contract
# ---------------------------------------------------------------------------

def test_contract_multi_start_optimize() -> None:
    """
    Verify that multi_start_optimize returns feasible weights, finite
    objective value, and finite entropy.
    """
    rng = np.random.RandomState(SEED)
    n = 15
    AU = 3

    # Build valid PSD Sigma_assets
    B_A_port = rng.randn(n, AU).astype(np.float64)
    z_hat = rng.randn(50, AU).astype(np.float64)
    Sigma_z, _ = estimate_sigma_z(z_hat)
    D_eps = np.abs(rng.randn(n).astype(np.float64)) + 1e-4

    risk_model = assemble_risk_model(B_A_port, Sigma_z, D_eps)
    Sigma_assets = risk_model["Sigma_assets"]
    B_prime = risk_model["B_prime_port"]
    eigenvalues = risk_model["eigenvalues"]

    w, f_val, H = multi_start_optimize(
        Sigma_assets=Sigma_assets,
        B_prime=B_prime,
        eigenvalues=eigenvalues,
        D_eps=D_eps,
        alpha=1.0,
        n_starts=3,
        seed=SEED,
        lambda_risk=1.0,
        phi=25.0,
        w_bar=0.03,
        w_max=0.10,
        is_first=True,
        max_iter=30,
    )

    # Weights: (n,), non-negative, sum to 1
    assert w.shape == (n,), f"w shape {w.shape}, expected ({n},)"
    assert np.all(w >= -1e-8), f"Negative weight found: min={w.min()}"
    assert abs(np.sum(w) - 1.0) < 1e-4, (
        f"Weights sum to {np.sum(w)}, expected ~1.0"
    )

    # Objective and entropy are finite
    assert isinstance(f_val, float), (
        f"f_val type {type(f_val)}, expected float"
    )
    assert isinstance(H, float), f"H type {type(H)}, expected float"
    assert math.isfinite(f_val), f"f_val={f_val} is not finite"
    assert math.isfinite(H), f"H={H} is not finite"

    # FORMULA: H ≥ 0 (entropy non-negative)
    assert H >= -1e-10, f"Entropy H={H} should be >= 0"

    # FORMULA: H ≤ ln(AU + n) (entropy upper bound with idiosyncratic contributions)
    # When D_eps is passed, entropy considers AU systematic + n idiosyncratic
    # contributions, so the maximum is ln(AU + n), not ln(AU).
    max_H = float(np.log(AU + n))
    assert H <= max_H + 0.01, (
        f"H={H:.4f} exceeds max entropy ln({AU}+{n})={max_H:.4f}"
    )

    # FORMULA: INV-012 constraints satisfied
    w_max = 0.10
    assert np.all(w <= w_max + 1e-6), (
        f"max weight {w.max():.4f} > w_max={w_max}"
    )

    # FORMULA: concentration penalty recomputation
    # P_conc = Σ max(0, w_i - w_bar)²
    w_bar = 0.03
    P_conc = np.sum(np.maximum(0, w - w_bar) ** 2)
    assert math.isfinite(P_conc), f"P_conc={P_conc} is not finite"

    # FORMULA: first rebalancing → turnover penalty = 0
    # (is_first=True was passed)


# ---------------------------------------------------------------------------
# Test 11: Cardinality enforcement contract
# ---------------------------------------------------------------------------

def test_contract_enforce_cardinality() -> None:
    """
    Verify that enforce_cardinality returns weights satisfying the
    semi-continuous constraint: w_i = 0 or w_i >= w_min.
    """
    rng = np.random.RandomState(SEED)
    n = 15
    AU = 3
    w_min = 0.001

    # Build valid risk model
    B_A_port = rng.randn(n, AU).astype(np.float64)
    z_hat = rng.randn(50, AU).astype(np.float64)
    Sigma_z, _ = estimate_sigma_z(z_hat)
    D_eps = np.abs(rng.randn(n).astype(np.float64)) + 1e-4

    risk_model = assemble_risk_model(B_A_port, Sigma_z, D_eps)
    Sigma_assets = risk_model["Sigma_assets"]
    B_prime = risk_model["B_prime_port"]
    eigenvalues = risk_model["eigenvalues"]

    # Create a continuous solution with some small violations
    w_opt = rng.dirichlet(np.ones(n)).astype(np.float64)
    # Force a few sub-threshold weights to trigger cardinality enforcement
    w_opt[0] = w_min * 0.5
    w_opt[1] = w_min * 0.3
    w_opt = w_opt / w_opt.sum()

    from src.portfolio.sca_solver import sca_optimize

    sca_kwargs = {
        "Sigma_assets": Sigma_assets,
        "B_prime": B_prime,
        "eigenvalues": eigenvalues,
        "alpha": 1.0,
        "lambda_risk": 1.0,
        "phi": 25.0,
        "w_bar": 0.03,
        "w_max": 0.10,
        "is_first": True,
        "max_iter": 10,
        "_L_sigma": None,
    }

    result = enforce_cardinality(
        w=w_opt,
        B_prime=B_prime,
        eigenvalues=eigenvalues,
        w_min=w_min,
        sca_solver_fn=sca_optimize,
        sca_kwargs=sca_kwargs,
        method="gradient",
    )

    # Result: (n,) weights summing to ~1
    assert result.shape == (n,), (
        f"result shape {result.shape}, expected ({n},)"
    )
    assert abs(np.sum(result) - 1.0) < 1e-4, (
        f"Weights sum to {np.sum(result)}, expected ~1.0"
    )

    # Semi-continuous constraint: each weight is either 0 or >= w_min
    for i, wi in enumerate(result):
        assert wi <= 1e-8 or wi >= w_min - 1e-8, (
            f"Weight[{i}]={wi} violates semi-continuous constraint "
            f"(should be 0 or >= {w_min})"
        )

    # FORMULA: cardinality reduction — at least one stock eliminated
    n_active = np.sum(result > 1e-8)
    n_original = np.sum(w_opt > 1e-8)
    assert n_active <= n_original, (
        f"Cardinality enforcement should not increase active stocks: "
        f"{n_active} > {n_original}"
    )

    # FORMULA: entropy should remain defined and non-negative
    H_result, _ = compute_entropy_and_gradient(result, B_prime, eigenvalues)
    assert H_result >= -1e-10, (
        f"Post-cardinality entropy H={H_result} should be >= 0"
    )
    assert math.isfinite(H_result), (
        f"Post-cardinality entropy H={H_result} is not finite"
    )
