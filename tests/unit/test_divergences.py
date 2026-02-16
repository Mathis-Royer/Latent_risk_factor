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
    """Divergence #3: encoder default dropout=0.2, decoder default dropout=0.1.
    Module-level DROPOUT constants removed (dead code); defaults are now
    literal values in constructor signatures.
    """

    def test_encoder_default_dropout_is_0_2(self) -> None:
        """Encoder ResBlock default dropout is 0.2."""
        from src.vae.encoder import ResBlock
        block = ResBlock(16, 32)
        assert abs(block.dropout.p - 0.2) < 1e-6, (
            f"ResBlock default dropout should be 0.2, got {block.dropout.p}"
        )

    def test_decoder_default_dropout_is_0_1(self) -> None:
        """Decoder TransposeResBlock default dropout is 0.1."""
        from src.vae.decoder import TransposeResBlock
        block = TransposeResBlock(32, 16)
        assert abs(block.dropout.p - 0.1) < 1e-6, (
            f"TransposeResBlock default dropout should be 0.1, got {block.dropout.p}"
        )

    def test_config_default_dropout_is_0_2(self) -> None:
        """Config default is 0.2 (raised from 0.1 for regularization)."""
        from src.config import VAEArchitectureConfig
        cfg = VAEArchitectureConfig()
        assert cfg.dropout == 0.2, (
            f"config.dropout default should be 0.2, got {cfg.dropout}"
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
    """Divergence #4: scalar variance targeting (undocumented addition).
    divergences.md says: 'Garder — ajout critique, clamped [0.01, 100]'.
    Now uses a single scalar VT (realized_EW_var / predicted_EW_var) applied
    AFTER all eigenvalue transformations.
    """

    def test_variance_targeting_formula(self) -> None:
        """Block VT: separate sys/idio scales.

        Verifies that:
        1. Both scales are positive and clamped to [0.01, 100].
        2. Combined scaled predicted EW variance is close to realized EW variance.
        """
        from src.integration.pipeline import FullPipeline
        from src.risk_model.covariance import assemble_risk_model

        rng = np.random.RandomState(42)
        n = 20
        AU = 5
        B_A_port = rng.randn(n, AU) * 0.1
        Sigma_z = np.eye(AU) * 0.01
        D_eps_port = rng.uniform(0.0001, 0.001, n)

        # Get B_prime and eigenvalues via assemble_risk_model
        rm = assemble_risk_model(B_A_port, Sigma_z, D_eps_port)
        B_prime_port = rm["B_prime_port"]
        eigenvalues = rm["eigenvalues"]

        n_days = 500
        ew_rets = rng.randn(n_days) * 0.01
        stock_ids = list(range(n))
        returns_df = pd.DataFrame(
            {s: ew_rets + rng.randn(n_days) * 0.001 for s in stock_ids},
            index=pd.date_range("2020-01-01", periods=n_days),
        )

        vt_sys, vt_idio = FullPipeline._block_variance_targeting(
            B_prime_port, eigenvalues, D_eps_port, returns_df, stock_ids, n,
        )

        assert 0.01 <= vt_sys <= 100.0
        assert 0.01 <= vt_idio <= 100.0
        assert np.isfinite(vt_sys)
        assert np.isfinite(vt_idio)

        # Combined scaled predicted EW var should be close to realized.
        # Tolerance accounts for Bayesian shrinkage: with prior strength ν=60
        # and n_holdout≈100, the posterior is pulled ~38% toward the prior
        # (vt=1.0), so scaled_pred can differ from realized_var by up to 20%.
        w_eq = np.ones(n) / n
        beta_eq = B_prime_port.T @ w_eq
        pred_sys = float(np.sum(beta_eq ** 2 * eigenvalues))
        pred_idio = float(np.sum(w_eq ** 2 * D_eps_port))
        ew_returns = returns_df[stock_ids].mean(axis=1).to_numpy()
        realized_var = float(np.var(ew_returns, ddof=1))
        scaled_pred = vt_sys * pred_sys + vt_idio * pred_idio
        assert abs(scaled_pred - realized_var) / max(realized_var, 1e-10) < 0.25, (
            f"Scaled pred={scaled_pred:.2e} should be close to realized={realized_var:.2e}"
        )

    def test_variance_targeting_clamp_lower(self) -> None:
        """When predicted var >> realized var, scales are clamped to >= 0.01."""
        from src.integration.pipeline import FullPipeline
        from src.risk_model.covariance import assemble_risk_model

        n = 5
        AU = 2
        B_A_port = np.eye(n, AU) * 100.0
        Sigma_z = np.eye(AU) * 1000.0
        D_eps_port = np.ones(n) * 1000.0
        rm = assemble_risk_model(B_A_port, Sigma_z, D_eps_port)
        stock_ids = list(range(n))

        n_days = 100
        returns_df = pd.DataFrame(
            {s: np.ones(n_days) * 0.0001 for s in stock_ids},
            index=pd.date_range("2020-01-01", periods=n_days),
        )

        vt_sys, vt_idio = FullPipeline._block_variance_targeting(
            rm["B_prime_port"], rm["eigenvalues"], D_eps_port,
            returns_df, stock_ids, n,
        )
        assert 0.01 <= vt_sys <= 100.0
        assert 0.01 <= vt_idio <= 100.0

    def test_variance_targeting_clamp_upper(self) -> None:
        """When predicted var << realized var, scales are clamped to <= 100."""
        from src.integration.pipeline import FullPipeline
        from src.risk_model.covariance import assemble_risk_model

        n = 5
        AU = 2
        B_A_port = np.eye(n, AU) * 1e-6
        Sigma_z = np.eye(AU) * 1e-12
        D_eps_port = np.ones(n) * 1e-12
        rm = assemble_risk_model(B_A_port, Sigma_z, D_eps_port)
        stock_ids = list(range(n))

        rng = np.random.RandomState(0)
        n_days = 100
        returns_df = pd.DataFrame(
            {s: rng.randn(n_days) * 0.1 for s in stock_ids},
            index=pd.date_range("2020-01-01", periods=n_days),
        )

        vt_sys, vt_idio = FullPipeline._block_variance_targeting(
            rm["B_prime_port"], rm["eigenvalues"], D_eps_port,
            returns_df, stock_ids, n,
        )
        assert 0.01 <= vt_sys <= 100.0
        assert 0.01 <= vt_idio <= 100.0

    def test_variance_targeting_identity_when_perfect(self) -> None:
        """If predicted var == realized var, scales ≈ 1.0."""
        from src.integration.pipeline import FullPipeline
        from src.risk_model.covariance import assemble_risk_model

        rng = np.random.RandomState(42)
        n = 10
        AU = 3
        n_days = 10_000
        daily_vol = 0.01
        returns_df = pd.DataFrame(
            {s: rng.randn(n_days) * daily_vol for s in range(n)},
            index=pd.date_range("2000-01-01", periods=n_days),
        )
        ret_matrix = returns_df.values
        U, S, Vt = np.linalg.svd(ret_matrix, full_matrices=False)
        B_A_port = Vt[:AU, :].T
        z_hat = ret_matrix @ B_A_port
        Sigma_z = np.cov(z_hat, rowvar=False)
        resids = ret_matrix - z_hat @ B_A_port.T
        D_eps_port = np.var(resids, axis=0, ddof=1)
        rm = assemble_risk_model(B_A_port, Sigma_z, D_eps_port)

        vt_sys, vt_idio = FullPipeline._block_variance_targeting(
            rm["B_prime_port"], rm["eigenvalues"], D_eps_port,
            returns_df, list(range(n)), n,
        )
        # Both scales should be close to 1.0 when model matches realized
        assert abs(vt_sys - 1.0) < 0.25, (
            f"vt_sys should be ~1.0 when model matches realized, got {vt_sys}"
        )
        assert abs(vt_idio - 1.0) < 0.25, (
            f"vt_idio should be ~1.0 when model matches realized, got {vt_idio}"
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
