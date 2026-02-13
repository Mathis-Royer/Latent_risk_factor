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
    Sigma_z = estimate_sigma_z(z_hat)

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
    Sigma_z = estimate_sigma_z(z_hat)
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
    Sigma_z = estimate_sigma_z(z_hat)
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
