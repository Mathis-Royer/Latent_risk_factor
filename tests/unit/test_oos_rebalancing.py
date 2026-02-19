"""
Unit tests for OOS periodic rebalancing simulation.

Tests cover:
- Buy-and-hold mode (frequency=0)
- Scheduled rebalancing frequency
- Exceptional rebalancing trigger
- Delisting handling with Shumway imputation
- Turnover constraint respect
- Solver failure fallback
"""

import numpy as np
import pandas as pd
import pytest

from src.walk_forward.oos_rebalancing import (
    OOSRebalancingResult,
    RebalancingEvent,
    simulate_oos_rebalancing,
    _align_weights_to_new_universe,
    _align_matrix_to_universe,
    _align_vector_to_universe,
    _handle_delistings_at_date,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_setup():
    """Create a simple test setup with 10 stocks, 5 AU, 60 days."""
    np.random.seed(42)
    n = 10
    AU = 5
    T = 60

    # Stock IDs
    stock_ids = list(range(1001, 1001 + n))

    # Random exposures and eigenvalues
    B_prime = np.random.randn(n, AU) * 0.1
    eigenvalues = np.array([0.1, 0.05, 0.02, 0.01, 0.005])

    # Covariance
    Sigma = B_prime @ np.diag(eigenvalues) @ B_prime.T + np.eye(n) * 0.01

    # D_eps
    D_eps = np.diag(Sigma) * 0.5

    # Returns DataFrame
    dates = pd.date_range("2020-01-01", periods=T, freq="B")
    returns_data = np.random.randn(T, n) * 0.01  # Daily returns ~1% std
    returns = pd.DataFrame(returns_data, index=dates, columns=stock_ids)

    # Trailing vol
    trailing_vol = pd.DataFrame(
        np.full((T, n), 0.20),  # 20% annual vol
        index=dates,
        columns=stock_ids,
    )

    # Initial weights (equal weight)
    w_initial = np.ones(n) / n

    # Exchange codes (all NYSE)
    exchange_codes = {sid: 1 for sid in stock_ids}

    return {
        "B_prime": B_prime,
        "eigenvalues": eigenvalues,
        "B_A_raw": B_prime.copy(),
        "stock_ids": stock_ids,
        "Sigma": Sigma,
        "D_eps": D_eps,
        "returns": returns,
        "trailing_vol": trailing_vol,
        "w_initial": w_initial,
        "exchange_codes": exchange_codes,
        "n": n,
        "AU": AU,
        "T": T,
    }


# ---------------------------------------------------------------------------
# Alignment Helper Tests
# ---------------------------------------------------------------------------

def test_align_weights_to_new_universe():
    """Test weight alignment when universe changes."""
    w_old = np.array([0.25, 0.25, 0.25, 0.25])
    old_ids = [101, 102, 103, 104]
    new_ids = [102, 103, 105]  # 101 exits, 105 enters

    w_aligned = _align_weights_to_new_universe(w_old, old_ids, new_ids)

    assert len(w_aligned) == 3
    assert w_aligned[0] == 0.25  # 102 keeps weight
    assert w_aligned[1] == 0.25  # 103 keeps weight
    assert w_aligned[2] == 0.0   # 105 is new, gets 0


def test_align_matrix_to_universe():
    """Test matrix alignment when universe changes."""
    M = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    old_ids = [101, 102, 103, 104]
    new_ids = [102, 104, 105]

    M_aligned = _align_matrix_to_universe(M, old_ids, new_ids)

    assert M_aligned.shape == (3, 2)
    np.testing.assert_array_equal(M_aligned[0], [3, 4])  # 102
    np.testing.assert_array_equal(M_aligned[1], [7, 8])  # 104
    np.testing.assert_array_equal(M_aligned[2], [0, 0])  # 105 (new)


def test_align_vector_to_universe():
    """Test vector alignment when universe changes."""
    v = np.array([1.0, 2.0, 3.0, 4.0])
    old_ids = [101, 102, 103, 104]
    new_ids = [102, 104, 105]

    v_aligned = _align_vector_to_universe(v, old_ids, new_ids, default=0.5)

    assert len(v_aligned) == 3
    assert v_aligned[0] == 2.0  # 102
    assert v_aligned[1] == 4.0  # 104
    assert v_aligned[2] == 0.5  # 105 (new, default)


# ---------------------------------------------------------------------------
# Delisting Handling Tests
# ---------------------------------------------------------------------------

def test_handle_delistings_nyse():
    """Test Shumway imputation for NYSE delisting (-30%)."""
    w = np.array([0.25, 0.25, 0.25, 0.25])
    stock_ids = [101, 102, 103, 104]
    delisted = {102}  # Stock 102 delists
    exchange_codes = {101: 1, 102: 1, 103: 1, 104: 1}  # All NYSE

    w_new, imputed, total_ret = _handle_delistings_at_date(
        w, stock_ids, delisted, exchange_codes,
        delisting_return_nyse_amex=-0.30,
        delisting_return_nasdaq=-0.55,
    )

    # Delisted stock should have 0 weight
    assert w_new[1] == 0.0

    # Shumway return should be -30%
    assert imputed[1] == -0.30

    # Total return contribution = 0.25 * (-0.30) = -0.075
    assert np.isclose(total_ret, -0.075)

    # Remaining weights should sum to 1
    assert np.isclose(np.sum(w_new), 1.0)


def test_handle_delistings_nasdaq():
    """Test Shumway imputation for NASDAQ delisting (-55%)."""
    w = np.array([0.5, 0.5])
    stock_ids = [101, 102]
    delisted = {101}
    exchange_codes = {101: 3, 102: 1}  # 101=NASDAQ, 102=NYSE

    w_new, imputed, total_ret = _handle_delistings_at_date(
        w, stock_ids, delisted, exchange_codes,
        delisting_return_nyse_amex=-0.30,
        delisting_return_nasdaq=-0.55,
    )

    # NASDAQ delisting uses -55%
    assert imputed[0] == -0.55
    assert np.isclose(total_ret, -0.275)  # 0.5 * (-0.55)


def test_handle_delistings_no_delistings():
    """Test that no change occurs when there are no delistings."""
    w = np.array([0.25, 0.25, 0.25, 0.25])
    stock_ids = [101, 102, 103, 104]
    delisted: set[int] = set()
    exchange_codes = {101: 1, 102: 1, 103: 1, 104: 1}

    w_new, imputed, total_ret = _handle_delistings_at_date(
        w, stock_ids, delisted, exchange_codes,
        delisting_return_nyse_amex=-0.30,
        delisting_return_nasdaq=-0.55,
    )

    np.testing.assert_array_equal(w_new, w)
    assert np.sum(imputed) == 0.0
    assert total_ret == 0.0


# ---------------------------------------------------------------------------
# Buy-and-Hold Mode Tests
# ---------------------------------------------------------------------------

def test_buy_and_hold_mode(simple_setup):
    """Test that frequency=0 produces buy-and-hold behavior (no rebalancing)."""
    setup = simple_setup

    result = simulate_oos_rebalancing(
        B_prime=setup["B_prime"],
        eigenvalues=setup["eigenvalues"],
        B_A_raw=setup["B_A_raw"],
        inferred_stock_ids=setup["stock_ids"],
        Sigma_assets_initial=setup["Sigma"],
        D_eps_initial=setup["D_eps"],
        returns_oos=setup["returns"],
        trailing_vol=setup["trailing_vol"],
        exchange_codes=setup["exchange_codes"],
        w_initial=setup["w_initial"],
        H_initial=1.0,
        alpha_opt=0.5,
        rebalancing_frequency_days=0,  # Buy-and-hold
        entropy_trigger_alpha=0.90,
        tc_bps=10.0,
    )

    # Should have daily returns for all OOS days
    assert len(result.daily_returns) == setup["T"]

    # No rebalancing should occur
    assert result.n_scheduled_rebalances == 0
    assert result.n_exceptional_rebalances == 0

    # Only initial event recorded
    assert len(result.rebalancing_events) == 1
    assert result.rebalancing_events[0].trigger == "initial"

    # Zero turnover and TC
    assert result.cumulative_turnover == 0.0
    assert result.total_transaction_cost == 0.0


# ---------------------------------------------------------------------------
# Scheduled Rebalancing Tests
# ---------------------------------------------------------------------------

def test_scheduled_rebalancing_frequency(simple_setup):
    """Test that scheduled rebalancing occurs at the specified frequency."""
    setup = simple_setup

    # Frequency = 21 days, 60 total days -> expect ~2 scheduled rebalances
    result = simulate_oos_rebalancing(
        B_prime=setup["B_prime"],
        eigenvalues=setup["eigenvalues"],
        B_A_raw=setup["B_A_raw"],
        inferred_stock_ids=setup["stock_ids"],
        Sigma_assets_initial=setup["Sigma"],
        D_eps_initial=setup["D_eps"],
        returns_oos=setup["returns"],
        trailing_vol=setup["trailing_vol"],
        exchange_codes=setup["exchange_codes"],
        w_initial=setup["w_initial"],
        H_initial=1.0,
        alpha_opt=0.5,
        rebalancing_frequency_days=21,
        entropy_trigger_alpha=0.90,
        tc_bps=10.0,
    )

    # Should have 2-3 scheduled rebalances (depends on exact timing)
    assert result.n_scheduled_rebalances >= 1
    assert result.n_scheduled_rebalances <= 3

    # Verify scheduled events exist
    scheduled_events = [e for e in result.rebalancing_events if e.trigger == "scheduled"]
    assert len(scheduled_events) == result.n_scheduled_rebalances


def test_short_frequency_more_rebalances(simple_setup):
    """Test that shorter frequency produces more rebalances."""
    setup = simple_setup

    result_21 = simulate_oos_rebalancing(
        B_prime=setup["B_prime"],
        eigenvalues=setup["eigenvalues"],
        B_A_raw=setup["B_A_raw"],
        inferred_stock_ids=setup["stock_ids"],
        Sigma_assets_initial=setup["Sigma"],
        D_eps_initial=setup["D_eps"],
        returns_oos=setup["returns"],
        trailing_vol=setup["trailing_vol"],
        exchange_codes=setup["exchange_codes"],
        w_initial=setup["w_initial"],
        H_initial=1.0,
        alpha_opt=0.5,
        rebalancing_frequency_days=21,
        tc_bps=10.0,
    )

    result_10 = simulate_oos_rebalancing(
        B_prime=setup["B_prime"],
        eigenvalues=setup["eigenvalues"],
        B_A_raw=setup["B_A_raw"],
        inferred_stock_ids=setup["stock_ids"],
        Sigma_assets_initial=setup["Sigma"],
        D_eps_initial=setup["D_eps"],
        returns_oos=setup["returns"],
        trailing_vol=setup["trailing_vol"],
        exchange_codes=setup["exchange_codes"],
        w_initial=setup["w_initial"],
        H_initial=1.0,
        alpha_opt=0.5,
        rebalancing_frequency_days=10,
        tc_bps=10.0,
    )

    # 10-day frequency should have more rebalances than 21-day
    assert result_10.n_scheduled_rebalances >= result_21.n_scheduled_rebalances


# ---------------------------------------------------------------------------
# Transaction Cost Tests
# ---------------------------------------------------------------------------

def test_transaction_costs_accumulate(simple_setup):
    """Test that transaction costs accumulate correctly."""
    setup = simple_setup

    result = simulate_oos_rebalancing(
        B_prime=setup["B_prime"],
        eigenvalues=setup["eigenvalues"],
        B_A_raw=setup["B_A_raw"],
        inferred_stock_ids=setup["stock_ids"],
        Sigma_assets_initial=setup["Sigma"],
        D_eps_initial=setup["D_eps"],
        returns_oos=setup["returns"],
        trailing_vol=setup["trailing_vol"],
        exchange_codes=setup["exchange_codes"],
        w_initial=setup["w_initial"],
        H_initial=1.0,
        alpha_opt=0.5,
        rebalancing_frequency_days=15,
        tc_bps=50.0,  # 50 bps = 0.5%
    )

    # If rebalancing occurred, TC should be positive
    if result.n_scheduled_rebalances > 0:
        assert result.total_transaction_cost > 0

        # TC should be sum of individual events
        event_tc = sum(e.transaction_cost for e in result.rebalancing_events)
        assert np.isclose(result.total_transaction_cost, event_tc)


# ---------------------------------------------------------------------------
# Result Structure Tests
# ---------------------------------------------------------------------------

def test_result_structure(simple_setup):
    """Test that OOSRebalancingResult has correct structure."""
    setup = simple_setup

    result = simulate_oos_rebalancing(
        B_prime=setup["B_prime"],
        eigenvalues=setup["eigenvalues"],
        B_A_raw=setup["B_A_raw"],
        inferred_stock_ids=setup["stock_ids"],
        Sigma_assets_initial=setup["Sigma"],
        D_eps_initial=setup["D_eps"],
        returns_oos=setup["returns"],
        trailing_vol=setup["trailing_vol"],
        exchange_codes=setup["exchange_codes"],
        w_initial=setup["w_initial"],
        H_initial=1.5,
        alpha_opt=0.5,
        rebalancing_frequency_days=21,
    )

    # Check all fields are present
    assert isinstance(result, OOSRebalancingResult)
    assert isinstance(result.daily_returns, np.ndarray)
    assert isinstance(result.cumulative_return, float)
    assert isinstance(result.cumulative_turnover, float)
    assert isinstance(result.total_transaction_cost, float)
    assert isinstance(result.n_scheduled_rebalances, int)
    assert isinstance(result.n_exceptional_rebalances, int)
    assert isinstance(result.entropy_trajectory, list)
    assert isinstance(result.rebalancing_events, list)
    assert isinstance(result.final_weights, np.ndarray)
    assert isinstance(result.final_universe, list)

    # Entropy trajectory should start with H_initial
    assert result.entropy_trajectory[0] == 1.5


def test_rebalancing_event_structure(simple_setup):
    """Test that RebalancingEvent has correct structure."""
    setup = simple_setup

    result = simulate_oos_rebalancing(
        B_prime=setup["B_prime"],
        eigenvalues=setup["eigenvalues"],
        B_A_raw=setup["B_A_raw"],
        inferred_stock_ids=setup["stock_ids"],
        Sigma_assets_initial=setup["Sigma"],
        D_eps_initial=setup["D_eps"],
        returns_oos=setup["returns"],
        trailing_vol=setup["trailing_vol"],
        exchange_codes=setup["exchange_codes"],
        w_initial=setup["w_initial"],
        H_initial=1.0,
        alpha_opt=0.5,
        rebalancing_frequency_days=21,
    )

    # Check initial event
    initial = result.rebalancing_events[0]
    assert isinstance(initial, RebalancingEvent)
    assert initial.trigger == "initial"
    assert initial.turnover == 0.0
    assert initial.transaction_cost == 0.0

    # Check scheduled events (if any)
    for event in result.rebalancing_events[1:]:
        assert event.trigger in ("scheduled", "exceptional")
        assert event.turnover >= 0.0
        assert event.transaction_cost >= 0.0
        assert event.n_active > 0


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

def test_empty_returns():
    """Test handling of empty OOS returns."""
    n = 5
    AU = 3
    stock_ids = list(range(101, 101 + n))
    B_prime = np.random.randn(n, AU)
    eigenvalues = np.array([0.1, 0.05, 0.02])
    Sigma = np.eye(n)
    D_eps = np.ones(n) * 0.01

    # Empty returns DataFrame
    returns = pd.DataFrame(columns=stock_ids)
    trailing_vol = pd.DataFrame(columns=stock_ids)

    result = simulate_oos_rebalancing(
        B_prime=B_prime,
        eigenvalues=eigenvalues,
        B_A_raw=B_prime.copy(),
        inferred_stock_ids=stock_ids,
        Sigma_assets_initial=Sigma,
        D_eps_initial=D_eps,
        returns_oos=returns,
        trailing_vol=trailing_vol,
        exchange_codes={},
        w_initial=np.ones(n) / n,
        H_initial=1.0,
        alpha_opt=0.5,
        rebalancing_frequency_days=21,
    )

    # Should handle gracefully
    assert len(result.daily_returns) == 0
    assert result.cumulative_return == 0.0


def test_single_stock():
    """Test with a single stock (edge case for diversification)."""
    n = 1
    AU = 1
    T = 30
    stock_ids = [101]
    B_prime = np.array([[0.1]])
    eigenvalues = np.array([0.05])
    Sigma = np.array([[0.01]])
    D_eps = np.array([0.005])

    dates = pd.date_range("2020-01-01", periods=T, freq="B")
    returns = pd.DataFrame(np.random.randn(T, 1) * 0.01, index=dates, columns=stock_ids)
    trailing_vol = pd.DataFrame(np.full((T, 1), 0.20), index=dates, columns=stock_ids)

    result = simulate_oos_rebalancing(
        B_prime=B_prime,
        eigenvalues=eigenvalues,
        B_A_raw=B_prime.copy(),
        inferred_stock_ids=stock_ids,
        Sigma_assets_initial=Sigma,
        D_eps_initial=D_eps,
        returns_oos=returns,
        trailing_vol=trailing_vol,
        exchange_codes={101: 1},
        w_initial=np.array([1.0]),
        H_initial=0.0,  # Single stock has 0 entropy
        alpha_opt=0.5,
        rebalancing_frequency_days=10,
    )

    # Single stock: final weight should be 1.0
    assert len(result.final_weights) == 1
    assert np.isclose(result.final_weights[0], 1.0)


# ---------------------------------------------------------------------------
# Tests for simulate_benchmark_oos_rebalancing
# ---------------------------------------------------------------------------

from src.walk_forward.oos_rebalancing import simulate_benchmark_oos_rebalancing
from src.benchmarks.equal_weight import EqualWeight
from src.benchmarks.inverse_vol import InverseVolatility
from src.benchmarks.min_variance import MinimumVariance
from src.benchmarks.erc import EqualRiskContribution


@pytest.fixture
def benchmark_setup():
    """Create test data for benchmark OOS rebalancing tests."""
    np.random.seed(42)
    n = 20
    T = 60

    # Stock IDs as strings of integers (benchmark convention: "1001", "1002", ...)
    stock_ids_int = list(range(1001, 1001 + n))
    stock_ids = [str(sid) for sid in stock_ids_int]

    # Returns DataFrame with string columns
    dates = pd.date_range("2020-01-01", periods=T, freq="B")
    returns_data = np.random.randn(T, n) * 0.015
    returns = pd.DataFrame(returns_data, index=dates, columns=stock_ids)

    # Trailing vol with int columns (for exchange lookup compatibility)
    trailing_vol = pd.DataFrame(
        np.full((T, n), 0.20),
        index=dates,
        columns=stock_ids_int,
    )

    # Initial weights
    w_initial = np.ones(n) / n

    # Exchange codes
    exchange_codes = {sid: 1 for sid in stock_ids_int}  # All NYSE

    # Constraint params
    constraint_params = {
        "w_max": 0.10,
        "w_min": 0.001,
        "phi": 0.0,
        "tau_max": 0.30,
    }

    return {
        "stock_ids": stock_ids,
        "stock_ids_int": stock_ids_int,
        "returns": returns,
        "trailing_vol": trailing_vol,
        "w_initial": w_initial,
        "exchange_codes": exchange_codes,
        "constraint_params": constraint_params,
        "n": n,
        "T": T,
    }


class TestBenchmarkOOSRebalancing:
    """Tests for simulate_benchmark_oos_rebalancing()."""

    def test_benchmark_buy_and_hold(self, benchmark_setup):
        """Test buy-and-hold mode (frequency=0) for benchmark."""
        setup = benchmark_setup

        ew = EqualWeight(constraint_params=setup["constraint_params"])
        ew.fit(setup["returns"], setup["stock_ids"])
        w_init = ew.optimize(is_first=True)

        result = simulate_benchmark_oos_rebalancing(
            benchmark=ew,
            returns_oos=setup["returns"],
            trailing_vol=setup["trailing_vol"],
            w_initial=w_init,
            universe=setup["stock_ids"],
            exchange_codes=setup["exchange_codes"],
            rebalancing_frequency_days=0,  # Buy-and-hold
            tc_bps=10.0,
        )

        # Should have daily returns for all OOS days
        assert len(result.daily_returns) == setup["T"]

        # No rebalancing
        assert result.n_scheduled_rebalances == 0
        assert result.cumulative_turnover == 0.0
        assert result.total_transaction_cost == 0.0

        # Only initial event
        assert len(result.rebalancing_events) == 1
        assert result.rebalancing_events[0].trigger == "initial"

    def test_benchmark_scheduled_rebalancing(self, benchmark_setup):
        """Test scheduled rebalancing for EqualWeight benchmark."""
        setup = benchmark_setup

        ew = EqualWeight(constraint_params=setup["constraint_params"])
        ew.fit(setup["returns"], setup["stock_ids"])
        w_init = ew.optimize(is_first=True)

        result = simulate_benchmark_oos_rebalancing(
            benchmark=ew,
            returns_oos=setup["returns"],
            trailing_vol=setup["trailing_vol"],
            w_initial=w_init,
            universe=setup["stock_ids"],
            exchange_codes=setup["exchange_codes"],
            rebalancing_frequency_days=21,
            tc_bps=10.0,
        )

        # With 60 days and 21-day frequency, expect 2-3 rebalances
        assert result.n_scheduled_rebalances >= 1
        assert result.n_scheduled_rebalances <= 3

        # Verify scheduled events exist
        scheduled = [e for e in result.rebalancing_events if e.trigger == "scheduled"]
        assert len(scheduled) == result.n_scheduled_rebalances

    def test_benchmark_min_variance_rebalancing(self, benchmark_setup):
        """Test MinimumVariance benchmark rebalancing."""
        setup = benchmark_setup

        mv = MinimumVariance(constraint_params=setup["constraint_params"])
        mv.fit(setup["returns"], setup["stock_ids"])
        w_init = mv.optimize(is_first=True)

        result = simulate_benchmark_oos_rebalancing(
            benchmark=mv,
            returns_oos=setup["returns"],
            trailing_vol=setup["trailing_vol"],
            w_initial=w_init,
            universe=setup["stock_ids"],
            exchange_codes=setup["exchange_codes"],
            rebalancing_frequency_days=15,
            tc_bps=10.0,
        )

        # Should produce valid results
        assert len(result.daily_returns) == setup["T"]
        assert result.n_scheduled_rebalances >= 1

        # Final weights should be valid
        assert len(result.final_weights) > 0
        assert np.isclose(np.sum(result.final_weights), 1.0, atol=0.01)

    def test_benchmark_erc_rebalancing(self, benchmark_setup):
        """Test ERC benchmark rebalancing."""
        setup = benchmark_setup

        erc = EqualRiskContribution(constraint_params=setup["constraint_params"])
        erc.fit(setup["returns"], setup["stock_ids"])
        w_init = erc.optimize(is_first=True)

        result = simulate_benchmark_oos_rebalancing(
            benchmark=erc,
            returns_oos=setup["returns"],
            trailing_vol=setup["trailing_vol"],
            w_initial=w_init,
            universe=setup["stock_ids"],
            exchange_codes=setup["exchange_codes"],
            rebalancing_frequency_days=20,
            tc_bps=10.0,
        )

        assert len(result.daily_returns) == setup["T"]
        assert result.n_scheduled_rebalances >= 1

    def test_benchmark_transaction_costs(self, benchmark_setup):
        """Test that benchmark rebalancing accumulates transaction costs."""
        setup = benchmark_setup

        mv = MinimumVariance(constraint_params=setup["constraint_params"])
        mv.fit(setup["returns"], setup["stock_ids"])
        w_init = mv.optimize(is_first=True)

        result = simulate_benchmark_oos_rebalancing(
            benchmark=mv,
            returns_oos=setup["returns"],
            trailing_vol=setup["trailing_vol"],
            w_initial=w_init,
            universe=setup["stock_ids"],
            exchange_codes=setup["exchange_codes"],
            rebalancing_frequency_days=10,
            tc_bps=50.0,  # High TC to make effect visible
        )

        # If rebalancing occurred, TC should be positive
        if result.n_scheduled_rebalances > 0:
            assert result.total_transaction_cost > 0
            assert result.cumulative_turnover > 0

            # Verify TC = sum of event TCs
            event_tc = sum(e.transaction_cost for e in result.rebalancing_events)
            assert np.isclose(result.total_transaction_cost, event_tc, atol=1e-8)

    def test_benchmark_turnover_tracking(self, benchmark_setup):
        """Test that benchmark rebalancing tracks turnover correctly."""
        setup = benchmark_setup

        # Use MinVariance which doesn't require trailing_vol
        mv = MinimumVariance(constraint_params=setup["constraint_params"])
        mv.fit(setup["returns"], setup["stock_ids"])
        w_init = mv.optimize(is_first=True)

        result = simulate_benchmark_oos_rebalancing(
            benchmark=mv,
            returns_oos=setup["returns"],
            trailing_vol=setup["trailing_vol"],
            w_initial=w_init,
            universe=setup["stock_ids"],
            exchange_codes=setup["exchange_codes"],
            rebalancing_frequency_days=15,
            tc_bps=10.0,
        )

        # Turnover should be two-way (sum of event turnovers × 2)
        event_turnover = sum(e.turnover for e in result.rebalancing_events)
        assert np.isclose(result.cumulative_turnover, event_turnover * 2, atol=1e-8)

    def test_benchmark_result_structure(self, benchmark_setup):
        """Test OOSRebalancingResult structure for benchmark."""
        setup = benchmark_setup

        ew = EqualWeight(constraint_params=setup["constraint_params"])
        ew.fit(setup["returns"], setup["stock_ids"])
        w_init = ew.optimize(is_first=True)

        result = simulate_benchmark_oos_rebalancing(
            benchmark=ew,
            returns_oos=setup["returns"],
            trailing_vol=setup["trailing_vol"],
            w_initial=w_init,
            universe=setup["stock_ids"],
            exchange_codes=setup["exchange_codes"],
            rebalancing_frequency_days=21,
        )

        # Check structure
        assert isinstance(result.daily_returns, np.ndarray)
        assert isinstance(result.cumulative_return, float)
        assert isinstance(result.cumulative_turnover, float)
        assert isinstance(result.total_transaction_cost, float)
        assert isinstance(result.n_scheduled_rebalances, int)
        assert isinstance(result.n_exceptional_rebalances, int)
        assert isinstance(result.rebalancing_events, list)
        assert isinstance(result.final_weights, np.ndarray)
        assert isinstance(result.final_universe, list)

        # Benchmark doesn't track entropy
        assert result.n_exceptional_rebalances == 0  # No exceptional triggers
        assert len(result.entropy_trajectory) == 0

    def test_benchmark_shorter_frequency_more_rebalances(self, benchmark_setup):
        """Test that shorter frequency produces more rebalances."""
        setup = benchmark_setup

        ew = EqualWeight(constraint_params=setup["constraint_params"])
        ew.fit(setup["returns"], setup["stock_ids"])
        w_init = ew.optimize(is_first=True)

        result_long = simulate_benchmark_oos_rebalancing(
            benchmark=ew,
            returns_oos=setup["returns"],
            trailing_vol=setup["trailing_vol"],
            w_initial=w_init,
            universe=setup["stock_ids"],
            exchange_codes=setup["exchange_codes"],
            rebalancing_frequency_days=30,
        )

        result_short = simulate_benchmark_oos_rebalancing(
            benchmark=ew,
            returns_oos=setup["returns"],
            trailing_vol=setup["trailing_vol"],
            w_initial=w_init,
            universe=setup["stock_ids"],
            exchange_codes=setup["exchange_codes"],
            rebalancing_frequency_days=10,
        )

        # Shorter frequency should produce more rebalances
        assert result_short.n_scheduled_rebalances >= result_long.n_scheduled_rebalances

    def test_benchmark_handles_nan_returns_gracefully(self, benchmark_setup):
        """Test benchmark handles NaN returns (missing data) without crashing."""
        setup = benchmark_setup

        # Create returns with some NaN (simulating missing data or delisting)
        returns = setup["returns"].copy()
        # Stock at index 5 has NaN from day 30 onwards
        returns.iloc[30:, 5] = np.nan

        ew = EqualWeight(constraint_params=setup["constraint_params"])
        ew.fit(returns.iloc[:30], setup["stock_ids"])
        w_init = ew.optimize(is_first=True)

        result = simulate_benchmark_oos_rebalancing(
            benchmark=ew,
            returns_oos=returns,
            trailing_vol=setup["trailing_vol"],
            w_initial=w_init,
            universe=setup["stock_ids"],
            exchange_codes=setup["exchange_codes"],
            rebalancing_frequency_days=15,
            delisting_return_nyse_amex=-0.30,
        )

        # Should produce valid results without crashing
        assert len(result.daily_returns) == setup["T"]

        # Daily returns should all be finite
        assert np.all(np.isfinite(result.daily_returns))

        # Final weights should sum to 1 (or close to it)
        assert np.isclose(np.sum(result.final_weights), 1.0, atol=0.01)

    def test_benchmark_cumulative_return_computation(self, benchmark_setup):
        """Test cumulative return is correctly computed from daily returns."""
        setup = benchmark_setup

        ew = EqualWeight(constraint_params=setup["constraint_params"])
        ew.fit(setup["returns"], setup["stock_ids"])
        w_init = ew.optimize(is_first=True)

        result = simulate_benchmark_oos_rebalancing(
            benchmark=ew,
            returns_oos=setup["returns"],
            trailing_vol=setup["trailing_vol"],
            w_initial=w_init,
            universe=setup["stock_ids"],
            exchange_codes=setup["exchange_codes"],
            rebalancing_frequency_days=0,  # Buy-and-hold for simpler test
        )

        # Cumulative return = exp(sum(log_returns)) - 1
        expected_cumulative = np.exp(np.sum(result.daily_returns)) - 1.0
        assert np.isclose(result.cumulative_return, expected_cumulative, atol=1e-10)
