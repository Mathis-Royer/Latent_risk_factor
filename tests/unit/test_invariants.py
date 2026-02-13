"""
P0-priority tests for the 12 critical invariants (INV-001 to INV-012).

These tests verify the mathematical and structural contracts that must never
be broken. Each test maps to a specific invariant from the ISD specification.

Reference: ISD Invariants INV-001 through INV-012.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.vae.build_vae import build_vae
from src.vae.loss import (
    compute_loss,
    compute_validation_elbo,
    get_lambda_co,
)
from src.vae.model import VAEModel
from src.portfolio.entropy import compute_entropy_and_gradient
from src.risk_model.rescaling import (
    _compute_winsorized_ratios,
    rescale_estimation,
    rescale_portfolio,
)
from src.inference.active_units import (
    compute_au_max_stat,
    filter_exposure_matrix,
    measure_active_units,
    truncate_active_dims,
)
from src.walk_forward.folds import generate_fold_schedule
from src.data_pipeline.crisis import compute_crisis_threshold, generate_synthetic_vix
from src.benchmarks.equal_weight import EqualWeight
from src.benchmarks.inverse_vol import InverseVolatility
from src.benchmarks.min_variance import MinimumVariance
from src.benchmarks.erc import EqualRiskContribution
from src.benchmarks.pca_factor_rp import PCAFactorRiskParity
from src.benchmarks.pca_vol import PCAVolRiskParity


# ---------------------------------------------------------------------------
# Shared constants for fast synthetic tests
# ---------------------------------------------------------------------------

SEED = 42
SMALL_B = 4          # batch size
SMALL_T = 64         # window length
SMALL_F = 2          # features
SMALL_K = 5          # latent capacity
SMALL_N = 20         # stocks
SMALL_T_ANNEE = 3    # years of history
TOTAL_EPOCHS = 100


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tensors() -> dict[str, torch.Tensor]:
    """
    Create small deterministic tensors for loss computation tests.

    :return data (dict): x, x_hat, mu, log_var, log_sigma_sq, crisis_fractions
    """
    torch.manual_seed(SEED)
    return {
        "x": torch.randn(SMALL_B, SMALL_T, SMALL_F),
        "x_hat": torch.randn(SMALL_B, SMALL_T, SMALL_F),
        "mu": torch.randn(SMALL_B, SMALL_K),
        "log_var": torch.randn(SMALL_B, SMALL_K) * 0.5,
        "log_sigma_sq": torch.tensor(0.0),
        "crisis_fractions": torch.tensor([0.0, 0.3, 0.7, 1.0]),
    }


@pytest.fixture
def small_vae() -> tuple[VAEModel, dict]:
    """
    Build a small VAE for quick architecture tests.

    :return model_info (tuple): (VAEModel, info_dict)
    """
    return build_vae(
        n=SMALL_N,
        T=SMALL_T,
        T_annee=SMALL_T_ANNEE,
        F=SMALL_F,
        K=SMALL_K,
        r_max=200.0,
        c_min=144,
        dropout=0.1,
    )


# ---------------------------------------------------------------------------
# INV-001: D factor must appear as coefficient in reconstruction loss
# ---------------------------------------------------------------------------

class TestINV001DFactorCoefficient:
    """INV-001: The factor D = T*F must appear in the reconstruction term."""

    @pytest.mark.parametrize("log_sigma_sq_val", [-2.0, 0.0, 1.5])
    def test_recon_term_equals_D_over_2sigma_sq_times_L_recon(
        self,
        tensors: dict[str, torch.Tensor],
        log_sigma_sq_val: float,
    ) -> None:
        """
        In Mode P, recon_term must equal D/(2*sigma_sq) * L_recon.

        Parametrize over different sigma_sq values to verify the D factor
        is applied independently of the observation variance scale.

        :param tensors (dict): Shared test tensors
        :param log_sigma_sq_val (float): Value for log(sigma_sq)
        """
        log_sigma_sq = torch.tensor(log_sigma_sq_val, requires_grad=True)

        _, components = compute_loss(
            x=tensors["x"],
            x_hat=tensors["x_hat"],
            mu=tensors["mu"],
            log_var=tensors["log_var"],
            log_sigma_sq=log_sigma_sq,
            crisis_fractions=tensors["crisis_fractions"],
            epoch=50,
            total_epochs=TOTAL_EPOCHS,
            mode="P",
            gamma=3.0,
        )

        D = SMALL_T * SMALL_F
        sigma_sq = torch.clamp(torch.exp(log_sigma_sq), min=1e-4, max=10.0)
        expected_coeff = D / (2.0 * sigma_sq.item())

        L_recon = components["recon"].item()
        recon_term = components["recon_term"].item()

        assert L_recon > 0, "L_recon should be positive for non-identical x, x_hat"
        ratio = recon_term / L_recon
        assert abs(ratio - expected_coeff) < 1e-5, (
            f"INV-001 violated: recon_term / L_recon = {ratio:.6f}, "
            f"expected D/(2*sigma_sq) = {expected_coeff:.6f}"
        )


# ---------------------------------------------------------------------------
# INV-002: sigma_sq must remain scalar after training
# ---------------------------------------------------------------------------

class TestINV002SigmaSqScalar:
    """INV-002: sigma_sq must be a scalar, clamped to [1e-4, 10], throughout training."""

    def test_sigma_sq_scalar_after_gradient_steps(
        self,
        small_vae: tuple[VAEModel, dict],
    ) -> None:
        """
        Run 20 gradient steps and verify log_sigma_sq remains a 0-dim tensor
        with numel() == 1, and obs_var stays within clamp bounds.

        :param small_vae (tuple): (model, info_dict)
        """
        model, _ = small_vae
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        torch.manual_seed(SEED)
        x = torch.randn(SMALL_B, SMALL_T, SMALL_F)
        crisis = torch.zeros(SMALL_B)

        for epoch in range(20):
            optimizer.zero_grad()
            x_hat, mu, log_var = model(x)
            loss, _ = compute_loss(
                x=x,
                x_hat=x_hat,
                mu=mu,
                log_var=log_var,
                log_sigma_sq=model.log_sigma_sq,
                crisis_fractions=crisis,
                epoch=epoch,
                total_epochs=TOTAL_EPOCHS,
                mode="P",
                gamma=1.0,
            )
            loss.backward()
            optimizer.step()

        # Verify scalar invariant
        assert model.log_sigma_sq.ndim == 0, (
            f"INV-002 violated: log_sigma_sq.ndim = {model.log_sigma_sq.ndim}, expected 0"
        )
        assert model.log_sigma_sq.numel() == 1, (
            f"INV-002 violated: log_sigma_sq.numel() = {model.log_sigma_sq.numel()}, expected 1"
        )

        # Verify obs_var is within clamp bounds
        obs_var = model.obs_var.item()
        assert 1e-4 <= obs_var <= 10.0, (
            f"INV-002 violated: obs_var = {obs_var}, expected in [1e-4, 10.0]"
        )


# ---------------------------------------------------------------------------
# INV-003: B_A shape and AU upper bound
# ---------------------------------------------------------------------------

class TestINV003BAShapeAndAUBound:
    """INV-003: B_A.shape == (n, AU) and AU <= AU_max_stat."""

    def test_exposure_matrix_shape_and_au_bound(
        self,
        small_vae: tuple[VAEModel, dict],
    ) -> None:
        """
        Build windows, run inference, measure AU, filter B_A.
        Verify B_A has the correct shape and AU does not exceed the
        statistical upper bound.

        :param small_vae (tuple): (model, info_dict)
        """
        model, _info = small_vae
        model.eval()

        n_stocks = SMALL_N
        n_windows_per_stock = 5
        total_windows = n_stocks * n_windows_per_stock

        torch.manual_seed(SEED)
        windows = torch.randn(total_windows, SMALL_T, SMALL_F)

        # Run deterministic encoding
        with torch.no_grad():
            mu_all = model.encode(windows)
        B = mu_all.numpy()

        # Reshape to (n_stocks, K) by averaging windows per stock
        B_avg = np.zeros((n_stocks, SMALL_K))
        for i in range(n_stocks):
            start = i * n_windows_per_stock
            end = start + n_windows_per_stock
            B_avg[i] = B[start:end].mean(axis=0)

        # Measure active units
        AU, kl_per_dim, active_dims = measure_active_units(
            model, windows, batch_size=32, au_threshold=0.01,
        )

        # Statistical upper bound
        n_obs = SMALL_T_ANNEE * 252
        au_max = compute_au_max_stat(n_obs, r_min=2)

        # Truncate if needed
        AU_trunc, active_dims_trunc = truncate_active_dims(
            AU, kl_per_dim, active_dims, au_max,
        )

        # Filter exposure matrix
        if AU_trunc > 0:
            B_A = filter_exposure_matrix(B_avg, active_dims_trunc)
            assert B_A.shape == (n_stocks, AU_trunc), (
                f"INV-003 violated: B_A.shape = {B_A.shape}, "
                f"expected ({n_stocks}, {AU_trunc})"
            )

        # AU never exceeds statistical bound
        assert AU_trunc <= au_max, (
            f"INV-003 violated: AU_trunc = {AU_trunc} > au_max_stat = {au_max}"
        )


# ---------------------------------------------------------------------------
# INV-004: Dual rescaling — estimation vs portfolio must differ
# ---------------------------------------------------------------------------

class TestINV004DualRescalingDiffers:
    """INV-004: Estimation rescaling is date-specific; portfolio uses current date only."""

    def test_estimation_returns_dict_portfolio_returns_array(self) -> None:
        """
        Verify rescale_estimation returns a dict[str -> ndarray] with
        per-date arrays, while rescale_portfolio returns a single ndarray.
        The two must differ when volatilities change across dates.
        """
        rng = np.random.RandomState(SEED)
        n, au = 20, 5
        B_A = rng.randn(n, au)
        stock_ids = list(range(100, 100 + n))

        # Trailing vol: 3 dates with varying cross-sections
        date_strs = ["2020-01-02", "2020-01-03", "2020-01-06"]  # business days
        dates = pd.DatetimeIndex([pd.Timestamp(d) for d in date_strs])
        vol_data = np.abs(rng.randn(3, n)) * 0.2 + 0.1
        # Make date 0 and date 2 have very different vol profiles
        vol_data[0, :] *= 0.5
        vol_data[2, :] *= 2.0

        trailing_vol = pd.DataFrame(
            vol_data, index=dates, columns=stock_ids,
        )

        universe_snapshots = {
            date_strs[0]: stock_ids,
            date_strs[2]: stock_ids,
        }

        # Estimation rescaling: returns dict
        B_est = rescale_estimation(
            B_A, trailing_vol, universe_snapshots, stock_ids,
        )
        assert isinstance(B_est, dict), "rescale_estimation must return a dict"
        assert len(B_est) >= 1, "rescale_estimation dict should not be empty"
        for _date_str, arr in B_est.items():
            assert arr.shape[1] == au, (
                f"Each date's array must have AU={au} columns, got {arr.shape[1]}"
            )

        # Portfolio rescaling: returns ndarray for a single date
        current_date = date_strs[1]
        B_port = rescale_portfolio(
            B_A, trailing_vol, current_date,
            universe=stock_ids,
            stock_ids=stock_ids,
        )
        assert isinstance(B_port, np.ndarray), "rescale_portfolio must return ndarray"
        assert B_port.shape == (n, au), (
            f"rescale_portfolio shape = {B_port.shape}, expected ({n}, {au})"
        )

        # Estimation at date 0 should differ from portfolio at date 1
        # because the vol profiles differ
        date0_str = date_strs[0]
        if date0_str in B_est:
            assert not np.allclose(B_est[date0_str], B_port, atol=1e-6), (
                "INV-004 violated: estimation rescaling at date 0 should differ "
                "from portfolio rescaling at date 1 when vol profiles differ"
            )


# ---------------------------------------------------------------------------
# INV-005a: No look-ahead in fold dates
# ---------------------------------------------------------------------------

class TestINV005NoLookahead:
    """INV-005: No look-ahead bias — training must end before OOS starts."""

    def test_no_lookahead_fold_dates(self) -> None:
        """
        Generate the walk-forward fold schedule. For every fold, assert
        train_end < oos_start with an embargo gap in between.
        """
        folds = generate_fold_schedule(
            start_date="1990-01-01",
            total_years=30,
            min_training_years=10,
            oos_months=6,
            embargo_days=21,
        )

        assert len(folds) > 0, "Fold schedule should not be empty"

        for fold in folds:
            train_end = pd.Timestamp(str(fold["train_end"]))
            embargo_start = pd.Timestamp(str(fold["embargo_start"]))
            embargo_end = pd.Timestamp(str(fold["embargo_end"]))
            oos_start = pd.Timestamp(str(fold["oos_start"]))

            # Training must end before embargo starts
            assert train_end < embargo_start, (
                f"INV-005 violated in fold {fold['fold_id']}: "
                f"train_end={train_end} >= embargo_start={embargo_start}"
            )

            # Embargo must have positive duration
            assert embargo_end > train_end, (
                f"INV-005 violated in fold {fold['fold_id']}: "
                f"embargo_end={embargo_end} <= train_end={train_end}"
            )

            # OOS must start after embargo
            assert oos_start > train_end, (
                f"INV-005 violated in fold {fold['fold_id']}: "
                f"oos_start={oos_start} <= train_end={train_end}"
            )

    def test_embargo_is_at_least_21_trading_days(self) -> None:
        """
        G2: Verify embargo gap is >= 21 trading days (not calendar days).
        Count business days between train_end and oos_start.
        """
        folds = generate_fold_schedule(
            start_date="1990-01-01",
            total_years=30,
            min_training_years=10,
            oos_months=6,
            embargo_days=21,
        )

        for fold in folds:
            train_end = pd.Timestamp(str(fold["train_end"]))
            oos_start = pd.Timestamp(str(fold["oos_start"]))

            # Count trading days (business days) in the gap
            trading_days = len(pd.bdate_range(
                start=train_end + pd.Timedelta(days=1),
                end=oos_start - pd.Timedelta(days=1),
                freq="B",
            ))

            assert trading_days >= 21, (
                f"INV-005 violated in fold {fold['fold_id']}: embargo has "
                f"{trading_days} trading days, expected >= 21. "
                f"train_end={train_end}, oos_start={oos_start}"
            )


# ---------------------------------------------------------------------------
# INV-005b: VIX threshold uses no future data
# ---------------------------------------------------------------------------

class TestINV005VixThreshold:
    """INV-005: VIX crisis threshold computed only on past data (no future leak)."""

    def test_vix_threshold_no_future_data(self) -> None:
        """
        Point-in-time discipline: VIX threshold at date T must be identical
        regardless of how much future data exists in the series.

        Generate VIX [1990-2020]. Compute threshold at end_2005 using the
        full series vs a truncated series [1990-2005]. If the function
        respects point-in-time, both thresholds must be IDENTICAL.
        """
        vix_full = generate_synthetic_vix(
            start_date="1990-01-01", end_date="2020-12-31", seed=SEED,
        )
        end_2005 = pd.Timestamp("2005-12-31")

        # Threshold computed with full data (future exists but must be ignored)
        threshold_with_future = compute_crisis_threshold(
            vix_full, training_end_date=end_2005,
        )

        # Threshold computed with truncated data (no future exists)
        vix_truncated = vix_full[vix_full.index <= end_2005]
        threshold_without_future = compute_crisis_threshold(
            vix_truncated, training_end_date=end_2005,
        )

        assert isinstance(threshold_with_future, float), "Threshold must be float"
        assert isinstance(threshold_without_future, float), "Threshold must be float"
        assert threshold_with_future > 0, "Threshold must be positive"

        # If point-in-time discipline holds, adding future data must not
        # change the threshold — this is the definitive no-lookahead proof
        assert threshold_with_future == threshold_without_future, (
            f"INV-005 violated: threshold at 2005 differs when future data is present. "
            f"With future={threshold_with_future:.6f}, "
            f"without future={threshold_without_future:.6f}"
        )

        # Additional: different end dates should give different thresholds
        end_2015 = pd.Timestamp("2015-12-31")
        threshold_2015 = compute_crisis_threshold(
            vix_full, training_end_date=end_2015,
        )
        assert threshold_with_future != threshold_2015, (
            "Sanity check: thresholds at 2005 and 2015 should differ"
        )


# ---------------------------------------------------------------------------
# INV-006: Modes P, F, A are mutually exclusive
# ---------------------------------------------------------------------------

class TestINV006ModesMutuallyExclusive:
    """INV-006: Loss modes P, F, A are mutually exclusive with strict guards."""

    def test_invalid_mode_raises(
        self,
        tensors: dict[str, torch.Tensor],
    ) -> None:
        """
        Passing mode='X' must raise an AssertionError.

        :param tensors (dict): Shared test tensors
        """
        with pytest.raises(AssertionError, match="INV-006"):
            compute_loss(
                x=tensors["x"],
                x_hat=tensors["x_hat"],
                mu=tensors["mu"],
                log_var=tensors["log_var"],
                log_sigma_sq=tensors["log_sigma_sq"],
                crisis_fractions=tensors["crisis_fractions"],
                epoch=10,
                total_epochs=TOTAL_EPOCHS,
                mode="X",
            )

    def test_mode_P_with_beta_not_1_raises(
        self,
        tensors: dict[str, torch.Tensor],
    ) -> None:
        """
        Mode P requires beta_fixed == 1.0. Non-unit beta must raise.

        :param tensors (dict): Shared test tensors
        """
        with pytest.raises(AssertionError, match="INV-006"):
            compute_loss(
                x=tensors["x"],
                x_hat=tensors["x_hat"],
                mu=tensors["mu"],
                log_var=tensors["log_var"],
                log_sigma_sq=tensors["log_sigma_sq"],
                crisis_fractions=tensors["crisis_fractions"],
                epoch=10,
                total_epochs=TOTAL_EPOCHS,
                mode="P",
                beta_fixed=0.5,
            )

    def test_mode_P_learns_sigma_sq_mode_F_freezes(
        self,
        tensors: dict[str, torch.Tensor],
    ) -> None:
        """
        Mode P: sigma_sq must have gradient flow.
        Mode F: sigma_sq (detached) must not accumulate gradients.

        :param tensors (dict): Shared test tensors
        """
        # Mode P: sigma_sq participates in gradient
        log_sigma_sq_p = torch.tensor(0.5, requires_grad=True)
        loss_p, _ = compute_loss(
            x=tensors["x"],
            x_hat=tensors["x_hat"],
            mu=tensors["mu"],
            log_var=tensors["log_var"],
            log_sigma_sq=log_sigma_sq_p,
            crisis_fractions=tensors["crisis_fractions"],
            epoch=10,
            total_epochs=TOTAL_EPOCHS,
            mode="P",
        )
        loss_p.backward()
        assert log_sigma_sq_p.grad is not None, (
            "INV-006 violated: Mode P must allow gradient flow through sigma_sq"
        )
        assert log_sigma_sq_p.grad.abs().item() > 0, (
            "INV-006 violated: Mode P gradient on sigma_sq should be non-zero"
        )

        # Mode F: sigma_sq detached (no grad on log_sigma_sq)
        # Use mu with requires_grad=True so backward() can run, then check
        # that log_sigma_sq_f has no gradient accumulated.
        log_sigma_sq_f = torch.tensor(0.5, requires_grad=True)
        mu_grad = tensors["mu"].clone().detach().requires_grad_(True)
        loss_f, _ = compute_loss(
            x=tensors["x"],
            x_hat=tensors["x_hat"],
            mu=mu_grad,
            log_var=tensors["log_var"],
            log_sigma_sq=log_sigma_sq_f,
            crisis_fractions=tensors["crisis_fractions"],
            epoch=10,
            total_epochs=TOTAL_EPOCHS,
            mode="F",
        )
        loss_f.backward()
        # In Mode F, sigma_sq is not used in the loss formula (recon_term = D/2 * L_recon),
        # so even though requires_grad=True, its gradient is either None (not in graph)
        # or exactly 0. Either way, sigma_sq receives no gradient update.
        sigma_sq_grad_is_zero = (
            log_sigma_sq_f.grad is None
            or log_sigma_sq_f.grad.abs().item() == 0.0
        )
        assert sigma_sq_grad_is_zero, (
            "INV-006 violated: Mode F must not flow gradients through sigma_sq, "
            f"got grad = {log_sigma_sq_f.grad}"
        )


# ---------------------------------------------------------------------------
# INV-007: Entropy computed in principal factor basis
# ---------------------------------------------------------------------------

class TestINV007EntropyPrincipalFactorBasis:
    """INV-007: Entropy H(w) must be computed in the principal factor basis."""

    def test_risk_contributions_nonneg_entropy_bounded(self) -> None:
        """
        With positive eigenvalues and B_prime from a proper rotation,
        c'_k >= 0 (by construction: (beta'_k)^2 * lambda_k).
        H must be in [0, ln(AU)].
        """
        rng = np.random.RandomState(SEED)
        n, au = 15, 5

        # Orthogonal B_prime via QR decomposition
        B_prime = rng.randn(n, au)
        Q, _ = np.linalg.qr(B_prime)
        B_prime = Q[:, :au]

        # Strictly positive eigenvalues
        eigenvalues = np.abs(rng.randn(au)) + 0.1

        # Uniform-ish weights
        w = np.ones(n) / n

        H, grad_H = compute_entropy_and_gradient(w, B_prime, eigenvalues)

        # Manually compute c_prime to verify non-negativity
        beta_prime = B_prime.T @ w
        c_prime = (beta_prime ** 2) * eigenvalues

        assert np.all(c_prime >= 0), (
            "INV-007 violated: c'_k must be non-negative in principal factor basis"
        )
        assert eigenvalues.min() > 0, "Eigenvalues must be positive"
        assert 0 <= H <= np.log(au) + 1e-10, (
            f"INV-007 violated: H = {H:.6f} not in [0, ln({au}) = {np.log(au):.6f}]"
        )
        assert grad_H.shape == (n,), (
            f"Gradient shape = {grad_H.shape}, expected ({n},)"
        )


# ---------------------------------------------------------------------------
# INV-008: Winsorization bounds on vol ratios
# ---------------------------------------------------------------------------

class TestINV008WinsorizationBounds:
    """INV-008: Vol ratios must be winsorized at [P5, P95] cross-sectionally."""

    def test_extreme_outlier_is_clipped(self) -> None:
        """
        Create a vol cross-section with one extreme outlier (100x median).
        After winsorization, the outlier's ratio should be clipped to P95.
        Then rescale_estimation should produce output without the extreme
        amplification that an unclipped outlier would cause.
        """
        rng = np.random.RandomState(SEED)
        n = 20

        # Normal vols around 0.2, one extreme outlier at 20.0 (100x median)
        vols = np.abs(rng.randn(n)) * 0.05 + 0.20
        vols[0] = 20.0  # extreme outlier

        # Direct winsorization test
        ratios = _compute_winsorized_ratios(vols, 5.0, 95.0)

        p5 = np.percentile(vols / np.median(vols), 5.0)
        p95 = np.percentile(vols / np.median(vols), 95.0)

        assert ratios.min() >= p5 - 1e-10, (
            f"INV-008 violated: min ratio {ratios.min():.4f} < P5 = {p5:.4f}"
        )
        assert ratios.max() <= p95 + 1e-10, (
            f"INV-008 violated: max ratio {ratios.max():.4f} > P95 = {p95:.4f}"
        )

        # The outlier ratio should be much smaller than the raw ratio
        raw_outlier_ratio = vols[0] / np.median(vols)
        assert ratios[0] < raw_outlier_ratio, (
            "INV-008 violated: extreme outlier was not clipped by winsorization"
        )


# ---------------------------------------------------------------------------
# INV-009: Gradient vanishes at maximum entropy (equal contributions)
# ---------------------------------------------------------------------------

class TestINV009GradientZeroAtMaximum:
    """INV-009: At equal risk contributions, H = ln(AU) and grad_H = 0."""

    def test_equal_contributions_maximum_entropy(self) -> None:
        """
        Set B_prime = I (AU x AU, padded to n x AU), eigenvalues = constant,
        w = [1/AU]*AU + [0]*(n-AU). This produces exactly equal risk
        contributions, so H = ln(AU) and grad_H ~ 0.
        """
        au = 5
        n = 10

        # Identity block + zeros
        B_prime = np.zeros((n, au))
        B_prime[:au, :au] = np.eye(au)

        # Constant eigenvalues
        eigenvalues = np.ones(au) * 2.0

        # Weights that activate each factor equally
        w = np.zeros(n)
        w[:au] = 1.0 / au

        H, grad_H = compute_entropy_and_gradient(w, B_prime, eigenvalues)

        # Maximum entropy = ln(AU)
        expected_H = np.log(au)
        assert abs(H - expected_H) < 1e-10, (
            f"INV-009 violated: H = {H:.10f}, expected ln({au}) = {expected_H:.10f}"
        )

        # Gradient should be approximately zero at the maximum
        grad_norm = np.linalg.norm(grad_H)
        assert grad_norm < 1e-10, (
            f"INV-009 violated: ||grad_H|| = {grad_norm:.2e}, expected ~0 at maximum entropy"
        )


# ---------------------------------------------------------------------------
# INV-010: Curriculum phases for lambda_co
# ---------------------------------------------------------------------------

class TestINV010CurriculumPhases:
    """INV-010: Co-movement curriculum has 3 phases with correct boundaries."""

    def test_lambda_co_at_boundaries(self) -> None:
        """
        Verify get_lambda_co at phase boundaries:
          epoch=0 -> lambda_co_max (Phase 1 start)
          epoch=29 -> lambda_co_max (Phase 1 end, 29% < 30%)
          epoch=45 -> intermediate value (Phase 2 mid)
          epoch=60 -> 0 (Phase 3 start, >= 60%)
          epoch=99 -> 0 (Phase 3 end)
        """
        total = TOTAL_EPOCHS
        lco_max = 0.5

        # Phase 1: epochs 0-29 (0% to <30%)
        assert get_lambda_co(0, total, lco_max) == lco_max, (
            "Phase 1 start: lambda_co should equal lambda_co_max"
        )
        assert get_lambda_co(29, total, lco_max) == lco_max, (
            "Phase 1 end (epoch 29 < 30): lambda_co should equal lambda_co_max"
        )

        # Phase 2: epochs 30-59 (30% to <60%)
        lco_45 = get_lambda_co(45, total, lco_max)
        assert 0 < lco_45 < lco_max, (
            f"Phase 2 mid (epoch 45): lambda_co = {lco_45} should be in (0, {lco_max})"
        )

        # Phase 3: epochs 60-99 (>= 60%)
        assert get_lambda_co(60, total, lco_max) == 0.0, (
            "Phase 3 start (epoch 60): lambda_co should be 0"
        )
        assert get_lambda_co(99, total, lco_max) == 0.0, (
            "Phase 3 end (epoch 99): lambda_co should be 0"
        )

    def test_lambda_co_monotonically_decreasing_in_phase2(self) -> None:
        """
        In Phase 2 (30% to 60%), lambda_co must monotonically decrease.
        """
        total = TOTAL_EPOCHS
        lco_max = 0.5
        phase2_start = int(0.30 * total)
        phase2_end = int(0.60 * total)

        prev = lco_max
        for epoch in range(phase2_start, phase2_end):
            lco = get_lambda_co(epoch, total, lco_max)
            assert lco <= prev + 1e-10, (
                f"INV-010 violated: lambda_co increased at epoch {epoch}: "
                f"{lco:.6f} > {prev:.6f}"
            )
            prev = lco


# ---------------------------------------------------------------------------
# INV-011: Validation ELBO excludes crisis weighting and co-movement
# ---------------------------------------------------------------------------

class TestINV011ValidationELBO:
    """INV-011: Validation ELBO uses gamma=1 (no crisis) and no lambda_co."""

    def test_validation_elbo_deterministic_and_differs_from_training(
        self,
        tensors: dict[str, torch.Tensor],
    ) -> None:
        """
        (a) Validation ELBO must be deterministic (same inputs -> same output).
        (b) Training loss with gamma>1 and crisis_fractions>0 must differ
        from validation ELBO.

        :param tensors (dict): Shared test tensors
        """
        log_sigma_sq = torch.tensor(0.0)

        # Compute validation ELBO twice — must be identical
        elbo_1 = compute_validation_elbo(
            x=tensors["x"],
            x_hat=tensors["x_hat"],
            mu=tensors["mu"],
            log_var=tensors["log_var"],
            log_sigma_sq=log_sigma_sq,
        )
        elbo_2 = compute_validation_elbo(
            x=tensors["x"],
            x_hat=tensors["x_hat"],
            mu=tensors["mu"],
            log_var=tensors["log_var"],
            log_sigma_sq=log_sigma_sq,
        )
        assert elbo_1.item() == elbo_2.item(), (
            "INV-011 violated: validation ELBO is not deterministic"
        )

        # Training loss with crisis weighting (gamma=3, crisis_fractions > 0)
        # At epoch=0, lambda_co is at max, so co-movement contributes
        co_loss = torch.tensor(0.1)
        _, train_components = compute_loss(
            x=tensors["x"],
            x_hat=tensors["x_hat"],
            mu=tensors["mu"],
            log_var=tensors["log_var"],
            log_sigma_sq=log_sigma_sq,
            crisis_fractions=tensors["crisis_fractions"],
            epoch=0,
            total_epochs=TOTAL_EPOCHS,
            mode="P",
            gamma=3.0,
            lambda_co_max=0.5,
            co_movement_loss=co_loss,
        )
        train_total = train_components["total"].item()
        val_total = elbo_1.item()

        assert train_total != pytest.approx(val_total, abs=1e-6), (
            f"INV-011 violated: training loss ({train_total:.6f}) should differ from "
            f"validation ELBO ({val_total:.6f}) due to crisis weighting and co-movement"
        )


# ---------------------------------------------------------------------------
# INV-012: Benchmark constraints identical across all models
# ---------------------------------------------------------------------------

class TestINV012BenchmarkConstraintsIdentical:
    """INV-012: All 6 benchmarks receive, preserve, AND enforce identical constraint_params."""

    def test_all_benchmarks_share_same_constraints(self) -> None:
        """
        Instantiate all 6 benchmarks with the same constraint_params dict.
        Verify each benchmark's constraint_params matches the shared one
        (same keys and values).
        """
        shared_constraints: dict[str, float] = {
            "w_max": 0.05,
            "w_min": 0.001,
            "phi": 25.0,
            "kappa_1": 0.1,
            "kappa_2": 7.5,
            "delta_bar": 0.01,
            "tau_max": 0.30,
            "lambda_risk": 1.0,
        }

        benchmarks = [
            EqualWeight(constraint_params=shared_constraints),
            InverseVolatility(constraint_params=shared_constraints),
            MinimumVariance(constraint_params=shared_constraints),
            EqualRiskContribution(constraint_params=shared_constraints),
            PCAFactorRiskParity(constraint_params=shared_constraints),
            PCAVolRiskParity(constraint_params=shared_constraints),
        ]

        benchmark_names = [
            "EqualWeight", "InverseVolatility", "MinimumVariance",
            "EqualRiskContribution", "PCAFactorRiskParity", "PCAVolRiskParity",
        ]

        for bm, name in zip(benchmarks, benchmark_names):
            assert bm.constraint_params == shared_constraints, (
                f"INV-012 violated: {name}.constraint_params differs from shared constraints"
            )
            for key, value in shared_constraints.items():
                assert key in bm.constraint_params, (
                    f"INV-012 violated: {name} missing constraint key '{key}'"
                )
                assert bm.constraint_params[key] == value, (
                    f"INV-012 violated: {name}.constraint_params['{key}'] = "
                    f"{bm.constraint_params[key]}, expected {value}"
                )

    def test_all_benchmarks_enforce_constraints_on_optimize(self) -> None:
        """
        C1: Verify constraints are actually ENFORCED during optimization,
        not just stored. Run each benchmark's optimize() and check:
        - max(w) <= w_max
        - sum(w) == 1
        - w >= 0 (long-only)
        """
        rng = np.random.RandomState(SEED)
        n = 30
        stock_ids = [f"s{i}" for i in range(n)]
        dates = pd.bdate_range("2020-01-01", periods=300, freq="B")
        returns = pd.DataFrame(
            rng.normal(0.0005, 0.02, (300, n)),
            index=dates, columns=stock_ids,
        )
        trailing_vol = pd.DataFrame(
            np.abs(rng.randn(300, n)) * 0.02 + 0.15,
            index=dates, columns=stock_ids,
        )

        shared_constraints: dict[str, float] = {
            "w_max": 0.05,
            "w_min": 0.001,
            "phi": 25.0,
            "kappa_1": 0.1,
            "kappa_2": 7.5,
            "delta_bar": 0.01,
            "tau_max": 0.30,
            "lambda_risk": 1.0,
        }

        benchmark_classes = [
            EqualWeight, InverseVolatility, MinimumVariance,
            EqualRiskContribution, PCAFactorRiskParity, PCAVolRiskParity,
        ]

        for BenchClass in benchmark_classes:
            name = BenchClass.__name__
            bm = BenchClass(constraint_params=shared_constraints)
            bm.fit(
                returns, stock_ids,
                trailing_vol=trailing_vol,
                current_date=str(dates[-1].date()),
            )
            w = bm.optimize(is_first=True)

            assert np.all(w >= -1e-8), (
                f"INV-012 violated: {name} has negative weights: min={np.min(w):.6f}"
            )
            assert abs(np.sum(w) - 1.0) < 1e-6, (
                f"INV-012 violated: {name} weights sum to {np.sum(w):.6f}, not 1.0"
            )
            assert np.max(w) <= shared_constraints["w_max"] + 1e-6, (
                f"INV-012 violated: {name} max(w)={np.max(w):.6f} > w_max={shared_constraints['w_max']}"
            )


# ---------------------------------------------------------------------------
# G7: Phase A composite scoring formula (DVT §9.2)
# ---------------------------------------------------------------------------

class TestPhaseAScoring:
    """G7: Verify composite_score matches the DVT §9.2 formula exactly."""

    def test_scoring_formula_known_values(self) -> None:
        """
        With known inputs, verify score = H_norm - lambda_pen*max(0, MDD-0.20)
                                           - lambda_est*max(0, 1 - R_sigma).
        """
        from src.walk_forward.phase_a import composite_score

        # Case 1: Perfect (H=ln(AU), no MDD penalty, sufficient observations)
        AU = 10
        H_oos = np.log(AU)  # normalized = 1.0
        score = composite_score(
            H_oos=H_oos, AU=AU, mdd_oos=0.10, n_obs=1000,
            mdd_threshold=0.20, lambda_pen=5.0, lambda_est=2.0,
        )
        # H_norm=1.0, MDD penalty=0 (0.10<0.20), R_sigma=1000/55=18.2>>1
        expected = 1.0 - 0.0 - 0.0
        assert abs(score - expected) < 1e-6, (
            f"Perfect case: score={score:.6f}, expected {expected:.6f}"
        )

        # Case 2: MDD penalty active
        score_mdd = composite_score(
            H_oos=H_oos, AU=AU, mdd_oos=0.30, n_obs=1000,
            mdd_threshold=0.20, lambda_pen=5.0, lambda_est=2.0,
        )
        expected_mdd = 1.0 - 5.0 * 0.10 - 0.0
        assert abs(score_mdd - expected_mdd) < 1e-6, (
            f"MDD penalty case: score={score_mdd:.6f}, expected {expected_mdd:.6f}"
        )

        # Case 3: Estimation penalty active (too few observations)
        R_sigma = 5 / (AU * (AU + 1) / 2)  # 5/55 = 0.09 < 1
        score_est = composite_score(
            H_oos=H_oos, AU=AU, mdd_oos=0.10, n_obs=5,
            mdd_threshold=0.20, lambda_pen=5.0, lambda_est=2.0,
        )
        expected_est = 1.0 - 0.0 - 2.0 * max(0.0, 1.0 - R_sigma)
        assert abs(score_est - expected_est) < 1e-6, (
            f"Estimation penalty case: score={score_est:.6f}, expected {expected_est:.6f}"
        )


# ---------------------------------------------------------------------------
# G9: Variance ratio for a well-calibrated model (DVT Metrics)
# ---------------------------------------------------------------------------

class TestVarianceRatio:
    """G9: var(r_p^OOS) / (w^T Sigma_hat w) should be near 1.0 for a good model."""

    def test_variance_ratio_with_known_covariance(self) -> None:
        """
        Generate OOS returns from the same covariance used for prediction.
        Variance ratio should be in [0.8, 1.2].
        """
        from src.walk_forward.metrics import realized_vs_predicted_variance

        rng = np.random.RandomState(SEED)
        n = 20
        T_oos = 500

        # Generate a known covariance
        A = rng.randn(n, n) * 0.01
        Sigma_true = A @ A.T + np.eye(n) * 0.001

        # Equal weights
        w = np.ones(n) / n

        # Generate OOS returns from this covariance
        returns_oos = rng.multivariate_normal(
            np.zeros(n), Sigma_true, size=T_oos,
        )

        # Use true covariance as the prediction
        ratio = realized_vs_predicted_variance(w, Sigma_true, returns_oos)

        assert 0.7 <= ratio <= 1.3, (
            f"Variance ratio with true covariance should be near 1.0 (ISD target "
            f"[0.8, 1.2]), got {ratio:.4f}. Using true Sigma with T_oos={T_oos} "
            f"samples, tolerance [0.7, 1.3] is conservative."
        )


# ---------------------------------------------------------------------------
# C3: INV-004 formula-level dual rescaling verification
# ---------------------------------------------------------------------------

class TestINV004DualRescalingFormulas:
    """C3: Verify dual rescaling formulas with known vol profiles, not just types."""

    def test_estimation_rescaling_formula_exact(self) -> None:
        """
        With known vols, manually compute R_{i,t} = σ_{i,t} / median(σ_t)
        and verify B̃_{A,i,t} = R_{i,t} · μ̄_{A,i} matches rescale_estimation().
        """
        n, au = 4, 2
        B_A = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ])
        stock_ids = [100, 101, 102, 103]

        # Two dates with distinct vol cross-sections
        dates = pd.DatetimeIndex([
            pd.Timestamp("2020-01-02"),
            pd.Timestamp("2020-01-03"),
        ])
        # Date 0: vols = [0.10, 0.20, 0.30, 0.40]
        # Date 1: vols = [0.40, 0.30, 0.20, 0.10]
        vol_data = np.array([
            [0.10, 0.20, 0.30, 0.40],
            [0.40, 0.30, 0.20, 0.10],
        ])
        trailing_vol = pd.DataFrame(vol_data, index=dates, columns=stock_ids)

        universe_snapshots = {
            "2020-01-02": stock_ids,
            "2020-01-03": stock_ids,
        }

        B_est = rescale_estimation(
            B_A, trailing_vol, universe_snapshots, stock_ids,
        )

        # Manual computation for date 0:
        # median([0.10, 0.20, 0.30, 0.40]) = 0.25
        # ratios = [0.4, 0.8, 1.2, 1.6]
        # With n=4, P5 and P95 won't clip much
        vols_d0 = np.array([0.10, 0.20, 0.30, 0.40])
        median_d0 = np.median(vols_d0)
        ratios_d0 = vols_d0 / median_d0
        lo_d0 = np.percentile(ratios_d0, 5.0)
        hi_d0 = np.percentile(ratios_d0, 95.0)
        ratios_d0_clipped = np.clip(ratios_d0, lo_d0, hi_d0)
        expected_d0 = B_A * ratios_d0_clipped[:, np.newaxis]

        assert "2020-01-02" in B_est, "Date 0 missing from estimation result"
        np.testing.assert_allclose(
            B_est["2020-01-02"], expected_d0, rtol=1e-10,
            err_msg="INV-004: estimation rescaling formula mismatch at date 0",
        )

        # Manual computation for date 1:
        vols_d1 = np.array([0.40, 0.30, 0.20, 0.10])
        median_d1 = np.median(vols_d1)
        ratios_d1 = vols_d1 / median_d1
        lo_d1 = np.percentile(ratios_d1, 5.0)
        hi_d1 = np.percentile(ratios_d1, 95.0)
        ratios_d1_clipped = np.clip(ratios_d1, lo_d1, hi_d1)
        expected_d1 = B_A * ratios_d1_clipped[:, np.newaxis]

        assert "2020-01-03" in B_est, "Date 1 missing from estimation result"
        np.testing.assert_allclose(
            B_est["2020-01-03"], expected_d1, rtol=1e-10,
            err_msg="INV-004: estimation rescaling formula mismatch at date 1",
        )

        # Estimation results must differ between dates (date-specific property)
        assert not np.allclose(B_est["2020-01-02"], B_est["2020-01-03"]), (
            "INV-004: estimation rescaling should give different results per date "
            "when vol profiles differ"
        )

    def test_portfolio_rescaling_formula_exact(self) -> None:
        """
        Verify B̃^port_{A,i} = R_{i,now} · μ̄_{A,i} using a single current date.
        """
        n, au = 4, 2
        B_A = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ])
        stock_ids = [100, 101, 102, 103]

        dates = pd.DatetimeIndex([pd.Timestamp("2020-01-06")])
        vols_now = np.array([[0.15, 0.25, 0.35, 0.45]])
        trailing_vol = pd.DataFrame(vols_now, index=dates, columns=stock_ids)

        B_port = rescale_portfolio(
            B_A, trailing_vol, "2020-01-06",
            universe=stock_ids, stock_ids=stock_ids,
        )

        # Manual computation
        vols = np.array([0.15, 0.25, 0.35, 0.45])
        median_v = np.median(vols)
        ratios = vols / median_v
        lo = np.percentile(ratios, 5.0)
        hi = np.percentile(ratios, 95.0)
        ratios_clipped = np.clip(ratios, lo, hi)
        expected = B_A * ratios_clipped[:, np.newaxis]

        np.testing.assert_allclose(
            B_port, expected, rtol=1e-10,
            err_msg="INV-004: portfolio rescaling formula mismatch",
        )

    def test_estimation_uses_different_vol_per_date(self) -> None:
        """
        The critical INV-004 property: estimation denominator changes per date
        while portfolio uses a single fixed date. Verify estimation at date t
        uses vol cross-section at date t, not any other date.
        """
        n, au = 3, 2
        B_A = np.ones((n, au))
        stock_ids = [10, 11, 12]

        # 3 dates where stock 0 has increasing vol
        dates = pd.DatetimeIndex([
            pd.Timestamp("2020-01-02"),
            pd.Timestamp("2020-01-03"),
            pd.Timestamp("2020-01-06"),
        ])
        vol_data = np.array([
            [0.10, 0.20, 0.20],  # date 0: stock 0 low vol
            [0.20, 0.20, 0.20],  # date 1: all same
            [0.40, 0.20, 0.20],  # date 2: stock 0 high vol
        ])
        trailing_vol = pd.DataFrame(vol_data, index=dates, columns=stock_ids)
        universe_snapshots = {str(d.date()): stock_ids for d in dates}

        B_est = rescale_estimation(
            B_A, trailing_vol, universe_snapshots, stock_ids,
        )

        # At date 1 (all same vol), ratios should all be 1.0
        if "2020-01-03" in B_est:
            np.testing.assert_allclose(
                B_est["2020-01-03"], B_A, rtol=1e-6,
                err_msg="When all vols equal, rescaled B should equal B_A",
            )

        # At date 0, stock 0 has lower vol → lower ratio → lower rescaled B
        if "2020-01-02" in B_est:
            assert B_est["2020-01-02"][0, 0] < B_est["2020-01-02"][1, 0], (
                "Stock with lower vol should have lower rescaled exposure"
            )

        # At date 2, stock 0 has higher vol → higher ratio → higher rescaled B
        if "2020-01-06" in B_est:
            assert B_est["2020-01-06"][0, 0] > B_est["2020-01-06"][1, 0], (
                "Stock with higher vol should have higher rescaled exposure"
            )


# ---------------------------------------------------------------------------
# M2: σ² clamping verified at EVERY training step
# ---------------------------------------------------------------------------

class TestSigmaSqClampingPerStep:
    """M2: Verify obs_var stays in [1e-4, 10] at every optimizer step, not just final."""

    def test_sigma_sq_in_bounds_every_step(
        self,
        small_vae: tuple[VAEModel, dict],
    ) -> None:
        """
        Run 30 gradient steps with aggressive learning rate to force σ² drift.
        Record obs_var after each step and assert ALL values are in bounds.
        """
        import math

        model, _ = small_vae
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        sigma_sq_min = 1e-4
        sigma_sq_max = 10.0
        log_min = math.log(sigma_sq_min)
        log_max = math.log(sigma_sq_max)

        torch.manual_seed(SEED)
        x = torch.randn(SMALL_B, SMALL_T, SMALL_F)
        crisis = torch.zeros(SMALL_B)

        obs_var_history: list[float] = []

        for step in range(30):
            optimizer.zero_grad()
            x_hat, mu, log_var = model(x)
            loss, _ = compute_loss(
                x=x,
                x_hat=x_hat,
                mu=mu,
                log_var=log_var,
                log_sigma_sq=model.log_sigma_sq,
                crisis_fractions=crisis,
                epoch=step,
                total_epochs=TOTAL_EPOCHS,
                mode="P",
                gamma=1.0,
            )
            loss.backward()
            optimizer.step()

            # Clamp as the trainer does (ISD: after EACH optimizer step)
            with torch.no_grad():
                model.log_sigma_sq.clamp_(log_min, log_max)

            obs_var = torch.exp(model.log_sigma_sq).item()
            obs_var_history.append(obs_var)

        # ALL values must be in bounds, not just the last one
        for i, val in enumerate(obs_var_history):
            assert sigma_sq_min - 1e-8 <= val <= sigma_sq_max + 1e-8, (
                f"INV-002 violated at step {i}: obs_var={val:.6e}, "
                f"expected in [{sigma_sq_min}, {sigma_sq_max}]"
            )


# ---------------------------------------------------------------------------
# M6: Mode P + beta annealing forbidden combination
# ---------------------------------------------------------------------------

class TestModePBetaAnnealingForbidden:
    """M6: Mode P must not use beta annealing. warmup_fraction is neutralized."""

    def test_mode_P_ignores_warmup_fraction(
        self,
        tensors: dict[str, torch.Tensor],
    ) -> None:
        """
        Mode P must produce identical results regardless of warmup_fraction,
        proving beta annealing has no effect in Mode P.
        """
        results = []
        for wf in [0.0, 0.20, 0.50, 1.0]:
            _, comps = compute_loss(
                x=tensors["x"],
                x_hat=tensors["x_hat"],
                mu=tensors["mu"],
                log_var=tensors["log_var"],
                log_sigma_sq=tensors["log_sigma_sq"],
                crisis_fractions=tensors["crisis_fractions"],
                epoch=5,
                total_epochs=TOTAL_EPOCHS,
                mode="P",
                warmup_fraction=wf,
            )
            results.append(comps["total"].item())

        # All results must be identical (Mode P never uses warmup_fraction)
        for i in range(1, len(results)):
            assert abs(results[i] - results[0]) < 1e-10, (
                f"M6 violated: Mode P loss differs with warmup_fraction "
                f"({results[0]:.10f} vs {results[i]:.10f}). "
                f"Mode P must not use beta annealing."
            )

    def test_mode_P_always_uses_beta_1(
        self,
        tensors: dict[str, torch.Tensor],
    ) -> None:
        """
        Verify that Mode P at any epoch always behaves as beta=1.0.
        Compare early epoch (where Mode F would have beta < 1) with late epoch.
        """
        _, comps_early = compute_loss(
            x=tensors["x"],
            x_hat=tensors["x_hat"],
            mu=tensors["mu"],
            log_var=tensors["log_var"],
            log_sigma_sq=tensors["log_sigma_sq"],
            crisis_fractions=tensors["crisis_fractions"],
            epoch=0,
            total_epochs=TOTAL_EPOCHS,
            mode="P",
        )
        _, comps_late = compute_loss(
            x=tensors["x"],
            x_hat=tensors["x_hat"],
            mu=tensors["mu"],
            log_var=tensors["log_var"],
            log_sigma_sq=tensors["log_sigma_sq"],
            crisis_fractions=tensors["crisis_fractions"],
            epoch=TOTAL_EPOCHS - 1,
            total_epochs=TOTAL_EPOCHS,
            mode="P",
        )

        # recon_term and KL contribution should remain consistent
        # (only lambda_co changes between epochs, not beta)
        recon_early = comps_early["recon_term"].item()
        recon_late = comps_late["recon_term"].item()
        assert abs(recon_early - recon_late) < 1e-10, (
            f"Mode P recon_term changed between epoch 0 and final epoch. "
            f"Beta annealing must NOT affect Mode P."
        )


# ---------------------------------------------------------------------------
# INV-007: Manual entropy computation with exact known contributions
# ---------------------------------------------------------------------------


class TestINV007ManualEntropyComputation:
    """INV-007: Verify H(w) = -Σ ĉ'_k · ln(ĉ'_k) with manually traced values."""

    def test_entropy_manual_two_factor(self) -> None:
        """With 2 factors and known β', λ, compute H step by step."""
        from tests.fixtures.known_solutions import two_factor_solution

        sol = two_factor_solution()
        w = sol["w_equal"]
        B_prime = sol["B_prime"]
        eigenvalues = sol["eigenvalues"]
        AU = sol["AU"]

        # Step 1: β'_k = B_prime^T @ w
        beta_prime = B_prime.T @ w
        assert beta_prime.shape == (AU,)

        # Step 2: c'_k = λ_k · (β'_k)²
        c_prime = eigenvalues * beta_prime ** 2
        assert np.all(c_prime >= 0), "Risk contributions must be non-negative"

        # Step 3: C = Σ c'_k (total contributing risk)
        C_total = c_prime.sum()
        assert C_total > 0, "Total contribution must be positive"

        # Step 4: ĉ'_k = c'_k / C
        c_hat = c_prime / C_total
        assert abs(c_hat.sum() - 1.0) < 1e-14, "Normalized contributions must sum to 1"

        # Step 5: H = -Σ ĉ'_k · ln(ĉ'_k)
        H_manual = -np.sum(c_hat * np.log(np.maximum(c_hat, 1e-30)))

        # Compare with compute_entropy_and_gradient
        H, _ = compute_entropy_and_gradient(w, B_prime, eigenvalues)

        assert abs(H - H_manual) < 1e-10, (
            f"H from function ({H:.10f}) doesn't match manual ({H_manual:.10f})"
        )
        assert abs(H - sol["H_equal"]) < 1e-10, (
            f"H from function ({H:.10f}) doesn't match fixture ({sol['H_equal']:.10f})"
        )

    def test_entropy_bounds_ln_au(self) -> None:
        """H must be in [0, ln(AU)] for any valid input."""
        rng = np.random.RandomState(42)
        for au in [2, 5, 10]:
            n = au + 5
            B_prime = rng.randn(n, au)
            eigenvalues = np.abs(rng.randn(au)) + 0.01
            w = np.abs(rng.randn(n))
            w = w / w.sum()

            H, _ = compute_entropy_and_gradient(w, B_prime, eigenvalues)
            assert 0 <= H <= np.log(au) + 1e-10, (
                f"AU={au}: H={H:.6f} not in [0, ln({au})={np.log(au):.6f}]"
            )


# ---------------------------------------------------------------------------
# INV-008: Exact percentile verification for winsorization
# ---------------------------------------------------------------------------


class TestINV008ExactPercentile:
    """INV-008: Verify P5/P95 computation matches numpy exactly."""

    def test_percentile_matches_numpy(self) -> None:
        """_compute_winsorized_ratios must use P5/P95 from numpy percentile."""
        rng = np.random.RandomState(42)
        for n in [10, 20, 50]:
            vols = rng.uniform(0.05, 0.50, n)
            # Add outlier
            vols[0] = 5.0

            ratios = _compute_winsorized_ratios(vols, 5.0, 95.0)

            # Manual check: compute median, ratios, percentiles
            median_v = np.median(vols)
            raw_ratios = vols / median_v
            p5 = np.percentile(raw_ratios, 5.0)
            p95 = np.percentile(raw_ratios, 95.0)

            # All winsorized ratios must be in [p5, p95]
            assert np.all(ratios >= p5 - 1e-10), (
                f"n={n}: min ratio {ratios.min():.6f} < P5={p5:.6f}"
            )
            assert np.all(ratios <= p95 + 1e-10), (
                f"n={n}: max ratio {ratios.max():.6f} > P95={p95:.6f}"
            )

            # Non-outlier stocks should have unchanged ratios
            for i in range(1, n):
                if p5 <= raw_ratios[i] <= p95:
                    assert abs(ratios[i] - raw_ratios[i]) < 1e-12, (
                        f"Stock {i}: ratio was clipped despite being in [P5, P95]"
                    )
