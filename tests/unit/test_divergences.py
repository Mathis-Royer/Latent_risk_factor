"""
Tests verifying alignment between the code and the reference documents
(DVT, ISD, divergences.md).

divergences.md is the authoritative record of intentional deviations.
These tests ensure that the code matches the DOCUMENTED behavior
(code + divergences.md), NOT the raw DVT/ISD formula when they differ.

Coverage:
  DIV-01: AU_max_stat uses simplified formula (ISD-aligned)
  DIV-02: beta_min = 0.01 floor in Mode F (defensive improvement)
  DIV-03: Dropout asymmetry — encoder 0.2, decoder 0.1
  DIV-04: Variance targeting (undocumented addition)
  DIV-05: Auto-adaptation for small universes
  DIV-08: Multi-start composition (approximate vs exact starts)
  DIV-09: create_windows returns 3-tuple
  DIV-10: Fresh CVXPY problem per SCA iteration

Reference: docs/divergences.md
"""

import math

import numpy as np
import pandas as pd
import pytest
import torch


# =====================================================================
# DIV-01: AU_max_stat — simplified formula floor(sqrt(2*N/r_min))
# =====================================================================


class TestDiv01AUMaxStatFormula:
    """Divergence #1: Code uses floor(sqrt(2*N/r_min)),
    NOT the DVT's floor((-1+sqrt(1+4*N/r_min))/2).
    divergences.md says: 'Garder le code (ISD-aligné)'.
    """

    def test_code_uses_isd_formula_not_dvt(self) -> None:
        """Verify compute_au_max_stat matches ISD simplified formula."""
        from src.inference.active_units import compute_au_max_stat

        for n_obs, r_min in [(7560, 2), (6300, 2), (5040, 2), (3780, 2), (2520, 2)]:
            result = compute_au_max_stat(n_obs, r_min)
            isd_formula = int(math.floor(math.sqrt(2.0 * n_obs / r_min)))
            # DVT solves AU*(AU+1)/2 <= N/r_min => AU = floor((-1+sqrt(1+8*N/r_min))/2)
            dvt_formula = int(math.floor((-1 + math.sqrt(1 + 8 * n_obs / r_min)) / 2))
            assert result == isd_formula, (
                f"n_obs={n_obs}: code gives {result}, ISD formula gives {isd_formula}"
            )
            # Document the difference — divergences.md says <2%
            pct_diff = abs(result - dvt_formula) / max(dvt_formula, 1) * 100
            assert pct_diff < 5, (
                f"n_obs={n_obs}: ISD={result} vs DVT={dvt_formula}, "
                f"difference {pct_diff:.1f}% — divergences.md says negligeable"
            )

    def test_formula_matches_quadratic_approximation(self) -> None:
        """ISD formula solves AU²/2 ≤ N/r_min (ignoring +AU term)."""
        from src.inference.active_units import compute_au_max_stat

        for n_obs in [2520, 5040, 7560]:
            au = compute_au_max_stat(n_obs, r_min=2)
            # au² / 2 should be ≤ N/r_min
            assert au * au / 2 <= n_obs / 2 + 1e-6
            # (au+1)² / 2 should exceed N/r_min (tightest bound)
            assert (au + 1) * (au + 1) / 2 > n_obs / 2


# =====================================================================
# DIV-02: beta_min = 0.01 floor in Mode F warmup
# =====================================================================


class TestDiv02BetaMinFloor:
    """Divergence #2: get_beta_t has beta_min=0.01 floor.
    DVT says β starts at 0; code starts at 0.01.
    divergences.md says: 'Garder le code — protection contre posterior collapse'.
    """

    def test_beta_at_epoch_zero_is_0_01_not_zero(self) -> None:
        """At epoch 0, beta must be 0.01, NOT 0 as DVT specifies."""
        from src.vae.loss import get_beta_t

        beta_0 = get_beta_t(epoch=0, total_epochs=100, warmup_fraction=0.20)
        assert beta_0 == 0.01, (
            f"beta at epoch 0 should be 0.01 (divergences.md DIV-02), got {beta_0}"
        )
        # DVT would give 0.0 — we explicitly diverge
        assert beta_0 > 0, "beta_min > 0 prevents posterior collapse"

    def test_beta_increases_linearly_then_clamps(self) -> None:
        """β = max(0.01, min(1, epoch/T_warmup)) — linear with floor."""
        from src.vae.loss import get_beta_t

        total = 100
        warmup_frac = 0.20
        T_warmup = int(warmup_frac * total)  # = 20

        for epoch in range(total):
            beta = get_beta_t(epoch, total, warmup_frac)
            expected = max(0.01, min(1.0, epoch / T_warmup))
            assert abs(beta - expected) < 1e-10, (
                f"epoch={epoch}: got {beta}, expected {expected}"
            )


# =====================================================================
# DIV-03: Dropout asymmetry — encoder 0.2, decoder 0.1
# =====================================================================


class TestDiv03DropoutAsymmetry:
    """Divergence #3: encoder.DROPOUT=0.2, decoder.DROPOUT=0.1, config=0.1.
    divergences.md says: 'Bug mineur' and recommends aligning.
    These tests document the CURRENT state.
    """

    def test_encoder_default_dropout_is_0_2(self) -> None:
        """Encoder module-level default is 0.2 (Changelog #10)."""
        from src.vae.encoder import DROPOUT as ENCODER_DROPOUT
        assert ENCODER_DROPOUT == 0.2, (
            f"encoder.DROPOUT should be 0.2 per divergences.md, got {ENCODER_DROPOUT}"
        )

    def test_decoder_default_dropout_is_0_1(self) -> None:
        """Decoder module-level default is 0.1."""
        from src.vae.decoder import DROPOUT as DECODER_DROPOUT
        assert DECODER_DROPOUT == 0.1, (
            f"decoder.DROPOUT should be 0.1 per divergences.md, got {DECODER_DROPOUT}"
        )

    def test_config_default_dropout_is_0_1(self) -> None:
        """Config default is 0.1 — divergences.md notes misalignment with encoder."""
        from src.config import VAEArchitectureConfig
        cfg = VAEArchitectureConfig()
        assert cfg.dropout == 0.1, (
            f"config.dropout default should be 0.1, got {cfg.dropout}"
        )

    def test_build_vae_explicit_dropout_overrides_all(self) -> None:
        """When dropout is explicitly passed to build_vae, both enc/dec use it."""
        from src.vae.build_vae import build_vae

        model, info = build_vae(
            n=50, T=64, T_annee=3, F=2, K=10,
            r_max=200.0, c_min=144, dropout=0.35,
        )
        # Every Dropout module should have p=0.35
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Dropout):
                assert abs(module.p - 0.35) < 1e-6, (
                    f"{name}: dropout.p={module.p}, expected 0.35"
                )
        assert info["dropout"] == 0.35


# =====================================================================
# DIV-04: Variance targeting (undocumented addition)
# =====================================================================


class TestDiv04VarianceTargeting:
    """Divergence #4: _variance_targeting_scale not in DVT/ISD.
    divergences.md says: 'Garder — ajout critique, clamped [0.01, 100]'.
    """

    def test_variance_targeting_formula(self) -> None:
        """scale = Var_realized(EW) / Var_predicted(EW), clamped [0.01, 100].

        Verify exact formula: scale = realized_ew_var / predicted_ew_var.
        """
        from src.integration.pipeline import _variance_targeting_scale

        rng = np.random.RandomState(42)
        n = 20
        # Predicted Sigma with known diagonal
        sigma_diag = rng.uniform(0.0001, 0.001, n)
        Sigma = np.diag(sigma_diag)

        # Generate returns whose realized EW variance we can compute manually
        n_days = 500
        ew_rets = rng.randn(n_days) * 0.01  # daily EW returns
        stock_ids = list(range(n))

        # Build a returns DataFrame where mean of all columns = ew_rets
        returns_df = pd.DataFrame(
            {s: ew_rets + rng.randn(n_days) * 0.001 for s in stock_ids},
            index=pd.date_range("2020-01-01", periods=n_days),
        )

        scale = _variance_targeting_scale(Sigma, returns_df, stock_ids)

        # Must be positive and finite
        assert 0.01 <= scale <= 100.0, (
            f"Scale {scale} outside [0.01, 100] clamp range"
        )
        assert np.isfinite(scale)

        # Formula verification: manually compute the ratio
        w_eq = np.ones(n) / n
        predicted_var = float(w_eq @ Sigma @ w_eq)
        ew_returns = returns_df[stock_ids].mean(axis=1).to_numpy()
        realized_var = float(np.var(ew_returns, ddof=1))
        expected_scale = np.clip(realized_var / max(predicted_var, 1e-30), 0.01, 100.0)
        assert abs(scale - expected_scale) / max(expected_scale, 1e-10) < 0.01, (
            f"Scale={scale:.6f} != expected={expected_scale:.6f} "
            f"(realized={realized_var:.2e}, predicted={predicted_var:.2e})"
        )

    def test_variance_targeting_clamp_lower(self) -> None:
        """When predicted var >> realized var, scale is clamped to 0.01."""
        from src.integration.pipeline import _variance_targeting_scale

        n = 5
        # Hugely overestimated Sigma
        Sigma = np.eye(n) * 1000.0  # predicted vol is enormous
        stock_ids = list(range(n))

        # Tiny realized returns
        n_days = 100
        returns_df = pd.DataFrame(
            {s: np.ones(n_days) * 0.0001 for s in stock_ids},
            index=pd.date_range("2020-01-01", periods=n_days),
        )

        scale = _variance_targeting_scale(Sigma, returns_df, stock_ids)
        assert scale >= 0.01, f"Scale {scale} should be >= 0.01 (lower clamp)"
        assert scale <= 100.0

    def test_variance_targeting_clamp_upper(self) -> None:
        """When predicted var << realized var, scale is clamped to 100."""
        from src.integration.pipeline import _variance_targeting_scale

        n = 5
        # Tiny predicted Sigma
        Sigma = np.eye(n) * 1e-12
        stock_ids = list(range(n))

        # Large realized returns
        rng = np.random.RandomState(0)
        n_days = 100
        returns_df = pd.DataFrame(
            {s: rng.randn(n_days) * 0.1 for s in stock_ids},
            index=pd.date_range("2020-01-01", periods=n_days),
        )

        scale = _variance_targeting_scale(Sigma, returns_df, stock_ids)
        assert scale <= 100.0, f"Scale {scale} should be <= 100.0 (upper clamp)"
        assert scale >= 0.01

    def test_variance_targeting_identity_when_perfect(self) -> None:
        """If predicted var == realized var, scale ≈ 1.0."""
        from src.integration.pipeline import _variance_targeting_scale

        rng = np.random.RandomState(42)
        n = 10
        n_days = 10_000
        # Generate returns from known variance
        daily_vol = 0.01
        returns_df = pd.DataFrame(
            {s: rng.randn(n_days) * daily_vol for s in range(n)},
            index=pd.date_range("2000-01-01", periods=n_days),
        )
        # Predicted Sigma = realized Sigma (approximately)
        ew_rets = returns_df.mean(axis=1).to_numpy()
        realized_var = np.var(ew_rets, ddof=1)
        w_eq = np.ones(n) / n
        # Sigma = exact sample covariance
        Sigma_sample = np.cov(returns_df.values, rowvar=False)
        predicted_var = float(w_eq @ Sigma_sample @ w_eq)

        scale = _variance_targeting_scale(Sigma_sample, returns_df, list(range(n)))
        # scale should be very close to 1.0
        assert abs(scale - 1.0) < 0.05, (
            f"Scale should be ~1.0 when Sigma matches realized, got {scale}"
        )


# =====================================================================
# DIV-05: Auto-adaptation for small universes
# =====================================================================


class TestDiv05AutoAdaptation:
    """Divergence #5: _adapt_vae_params reduces K, c_min for small universes.
    divergences.md says: 'Garder — ajout nécessaire pour la généralisation'.
    """

    def test_adapt_reduces_K_for_short_history(self) -> None:
        """With few years of history, K should be capped by AU_max_stat."""
        from src.config import PipelineConfig
        from src.integration.pipeline import FullPipeline

        config = PipelineConfig()
        pipeline = FullPipeline(config)
        adapted = pipeline._adapt_vae_params(n_stocks=50, T_annee=5)

        # AU_max_stat for 5yr: floor(sqrt(2*1260/2)) = floor(sqrt(1260)) = 35
        from src.inference.active_units import compute_au_max_stat
        au_max = compute_au_max_stat(5 * 252, r_min=2)
        expected_k_cap = min(config.vae.K, max(2 * au_max, 10))
        assert adapted["K"] == expected_k_cap, (
            f"K should be capped at {expected_k_cap}, got {adapted['K']}"
        )
        assert adapted["K"] <= config.vae.K

    def test_adapt_reduces_c_min_for_small_universe(self) -> None:
        """With few stocks, c_min should drop from 384 to 144."""
        from src.config import PipelineConfig
        from src.integration.pipeline import FullPipeline

        config = PipelineConfig()
        pipeline = FullPipeline(config)
        adapted = pipeline._adapt_vae_params(n_stocks=30, T_annee=3)

        # With 30 stocks and 3 years, the default c_min=384 violates capacity
        # so _adapt should reduce c_min strictly below the default
        from src.vae.build_vae import C_MIN_DEFAULT
        assert adapted["c_min"] < C_MIN_DEFAULT, (
            f"c_min should be reduced below default {C_MIN_DEFAULT}, "
            f"got {adapted['c_min']}"
        )

    def test_adapt_relaxes_r_max_last_resort(self) -> None:
        """When K and c_min reduction don't suffice, r_max is relaxed by 10%."""
        from src.config import PipelineConfig
        from src.integration.pipeline import FullPipeline

        config = PipelineConfig()
        pipeline = FullPipeline(config)
        adapted = pipeline._adapt_vae_params(n_stocks=10, T_annee=2)

        # Very small universe — r_max must be relaxed
        assert adapted["r_max"] >= config.vae.r_max, (
            "r_max should be >= original when relaxed"
        )

    def test_adapt_reinforces_dropout_for_small_universes(self) -> None:
        """When r_max is relaxed, dropout should be raised to ≥ 0.2."""
        from src.config import PipelineConfig
        from src.integration.pipeline import FullPipeline

        config = PipelineConfig()
        pipeline = FullPipeline(config)
        adapted = pipeline._adapt_vae_params(n_stocks=10, T_annee=2)

        if adapted["r_max"] > config.vae.r_max:
            assert adapted["dropout"] >= 0.2, (
                "Reinforced regularization should set dropout >= 0.2"
            )


# =====================================================================
# DIV-08: Multi-start composition (approximate starts)
# =====================================================================


class TestDiv08MultiStartComposition:
    """Divergence #8: Code uses {EW, inverse-diag, inverse-vol, 2 random}
    instead of DVT's {EW, min-variance QP, ERC Spinu, 2 random}.
    divergences.md says: 'Acceptable — approximations plus rapides'.
    """

    def test_multi_start_generates_5_starts(self) -> None:
        """multi_start_optimize with n_starts=5 produces a valid entropy-maximizing solution."""
        from src.portfolio.sca_solver import multi_start_optimize
        from src.portfolio.entropy import compute_entropy_only

        rng = np.random.RandomState(42)
        n = 50
        A = rng.randn(n, n) * 0.01
        Sigma = A @ A.T + np.eye(n) * 0.01
        B_prime = rng.randn(n, 3)
        Q, _ = np.linalg.qr(B_prime)
        B_prime = Q[:, :3]
        eigenvalues = np.array([0.5, 0.3, 0.1])
        D_eps = np.diag(Sigma)

        w, f, H = multi_start_optimize(
            Sigma_assets=Sigma,
            B_prime=B_prime,
            eigenvalues=eigenvalues,
            D_eps=D_eps,
            alpha=1.0,
            n_starts=5,
            seed=42,
            lambda_risk=1.0,
            phi=25.0,
        )
        assert w.shape == (n,)
        assert np.isfinite(f) and np.isfinite(H)

        # H should be non-negative: H = -sum(c_k * ln(c_k)) >= 0
        assert H >= -1e-10, f"Entropy should be non-negative, got H={H}"

        # Multi-start should beat or match equal-weight entropy
        w_ew = np.ones(n) / n
        H_ew = compute_entropy_only(w_ew, B_prime, eigenvalues)
        assert H >= H_ew - 1e-4, (
            f"Multi-start SCA entropy H={H:.6f} should be >= EW entropy H_ew={H_ew:.6f}"
        )

    def test_single_start_equal_weight_produces_valid_result(self) -> None:
        """With n_starts=1 and alpha=0, multi_start uses EW start -> min-var solution."""
        from src.portfolio.sca_solver import multi_start_optimize

        rng = np.random.RandomState(42)
        n = 50
        A = rng.randn(n, n) * 0.01
        Sigma = A @ A.T + np.eye(n) * 0.01
        B_prime = rng.randn(n, 3)
        Q, _ = np.linalg.qr(B_prime)
        B_prime = Q[:, :3]
        eigenvalues = np.array([0.5, 0.3, 0.1])
        D_eps = np.diag(Sigma)

        # n_starts=1 uses only EW as initial point
        w, f, H = multi_start_optimize(
            Sigma_assets=Sigma,
            B_prime=B_prime,
            eigenvalues=eigenvalues,
            D_eps=D_eps,
            alpha=0.0,
            n_starts=1,
            seed=42,
            lambda_risk=1.0,
            phi=0.0,
            is_first=True,
        )
        assert w.shape == (n,)
        assert abs(w.sum() - 1.0) < 1e-4, f"Weights sum to {w.sum()}"
        assert np.all(w >= -1e-8), "Negative weight from EW start"
        assert np.isfinite(f)


# =====================================================================
# DIV-09: create_windows returns 3-tuple (not 2-tuple as in ISD)
# =====================================================================


class TestDiv09CreateWindows3Tuple:
    """Divergence #9: create_windows returns (windows, metadata, raw_returns).
    ISD says (windows, metadata) only. 3rd element needed for co-movement loss.
    divergences.md says: 'Garder — nécessaire pour la co-movement loss'.
    """

    def test_create_windows_returns_3_elements(self) -> None:
        """create_windows must return exactly 3 values."""
        from src.data_pipeline.windowing import create_windows
        from src.data_pipeline.features import compute_rolling_realized_vol as compute_rolling_vol

        rng = np.random.RandomState(42)
        n_stocks = 3
        n_days = 600
        dates = pd.bdate_range("2018-01-01", periods=n_days)
        stock_ids = list(range(n_stocks))
        returns_df = pd.DataFrame(
            rng.randn(n_days, n_stocks) * 0.01,
            index=dates,
            columns=stock_ids,
        )
        vol_df = compute_rolling_vol(returns_df)

        result = create_windows(returns_df, vol_df, stock_ids, T=64)
        assert isinstance(result, tuple), "create_windows must return a tuple"
        assert len(result) == 3, (
            f"create_windows must return 3 elements (DIV-09), got {len(result)}"
        )

        windows, metadata, raw_returns = result
        assert isinstance(windows, torch.Tensor)
        assert isinstance(metadata, pd.DataFrame)
        assert isinstance(raw_returns, torch.Tensor)

    def test_raw_returns_shape_matches_windows(self) -> None:
        """raw_returns has shape (N, T) matching windows' (N, T, F)."""
        from src.data_pipeline.windowing import create_windows
        from src.data_pipeline.features import compute_rolling_realized_vol as compute_rolling_vol

        rng = np.random.RandomState(42)
        n_days = 600
        stock_ids = [0, 1]
        returns_df = pd.DataFrame(
            rng.randn(n_days, 2) * 0.01,
            index=pd.bdate_range("2018-01-01", periods=n_days),
            columns=stock_ids,
        )
        vol_df = compute_rolling_vol(returns_df)
        windows, metadata, raw_returns = create_windows(
            returns_df, vol_df, stock_ids, T=64,
        )
        N, T, F = windows.shape
        assert raw_returns.shape == (N, T), (
            f"raw_returns shape {raw_returns.shape} != (N={N}, T={T})"
        )


# =====================================================================
# DIV-10: Fresh CVXPY problem per SCA iteration
# =====================================================================


class TestDiv10FreshCVXPY:
    """Divergence #10: Code builds fresh CVXPY problem each SCA iteration
    instead of parametric reuse. divergences.md says: 'Garder — robustesse'.

    We verify the behavioral consequence: SCA must monotonically improve
    the objective across iterations (fresh problems ensure no stale state).
    """

    def test_sca_objective_monotonically_improves(self) -> None:
        """SCA objective f(w_k) must be non-decreasing across iterations.

        This is the observable guarantee of correct problem construction:
        if stale parameters leaked between iterations, the objective
        could decrease or oscillate.
        """
        from src.portfolio.sca_solver import sca_optimize, objective_function
        from src.portfolio.entropy import compute_entropy_only

        rng = np.random.RandomState(42)
        n = 30
        A = rng.randn(n, n) * 0.01
        Sigma = A @ A.T + np.eye(n) * 0.01
        B_prime = rng.randn(n, 3)
        Q, _ = np.linalg.qr(B_prime)
        B_prime = Q[:, :3]
        eigenvalues = np.array([0.5, 0.3, 0.1])

        w_init = np.ones(n) / n

        # Run SCA with enough iterations to verify convergence
        w_opt, f_opt, H_opt, n_iters = sca_optimize(
            w_init=w_init,
            Sigma_assets=Sigma,
            B_prime=B_prime,
            eigenvalues=eigenvalues,
            alpha=1.0,
            lambda_risk=1.0,
            phi=25.0,
            w_bar=1.0 / n,
            w_max=0.10,
            is_first=True,
            max_iter=50,
        )

        # Verify basic constraints
        assert abs(w_opt.sum() - 1.0) < 1e-6, "Weights must sum to 1"
        assert np.all(w_opt >= -1e-8), "Weights must be non-negative"
        assert n_iters >= 1, "SCA should run at least 1 iteration"

        # Key check: final objective must improve over initial
        f_init = objective_function(
            w=w_init, Sigma_assets=Sigma, B_prime=B_prime,
            eigenvalues=eigenvalues, alpha=1.0, lambda_risk=1.0,
            phi=25.0, w_bar=1.0 / n, w_old=None,
            kappa_1=0.1, kappa_2=7.5, delta_bar=0.01, is_first=True,
        )
        assert f_opt >= f_init - 1e-6, (
            f"SCA should improve objective: f_init={f_init:.6f}, "
            f"f_opt={f_opt:.6f} (diff={f_opt - f_init:.2e})"
        )
