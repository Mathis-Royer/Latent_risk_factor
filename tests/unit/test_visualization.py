"""
Unit tests for visualization helper functions.

Tests the externalized plotting functions from src/integration/visualization.py
and table generation functions from src/integration/diagnostic_report.py.
"""

import numpy as np
import pytest

from src.integration.visualization import (
    plot_causal_chain_diagram,
    plot_factor_exposure_comparison,
    plot_kl_per_dim_heatmap,
    plot_model_inspection,
    plot_pca_eigenvalue_spectrum,
    plot_training_history_panels,
    plot_vae_pca_correlation,
)
from src.integration.diagnostic_report import (
    build_decision_rules_table,
    build_literature_comparison_table,
    build_portfolio_holdings_table,
    build_recommendations_table,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_analysis():
    """Sample root cause analysis dict."""
    return {
        "weakest_component": "solver",
        "weakest_score": 45.0,
        "causal_analysis": {
            "metric": "solver_score",
            "upstream_causes": ["batch_size", "learning_rate", "model_complexity"],
            "downstream_effects": ["convergence", "portfolio_quality", "returns"],
        },
        "matching_rules": [
            {
                "rule_id": "SOLVER_ISSUE",
                "diagnosis": "Solver not converging",
                "confidence": 0.85,
                "severity": "high",
                "root_causes": ["learning_rate", "batch_size"],
                "actions": ["reduce learning rate", "increase epochs"],
            },
        ],
        "detected_patterns": [
            {
                "name": "gradient_explosion",
                "interpretation": "Gradients growing unbounded",
                "recommendation": "Add gradient clipping",
            },
        ],
    }


@pytest.fixture
def sample_exposure_matrix():
    """Sample exposure matrix B_A (n_stocks, AU)."""
    np.random.seed(42)
    n_stocks = 100
    AU = 20
    return np.random.randn(n_stocks, AU) * 0.5


@pytest.fixture
def sample_weights():
    """Sample portfolio weights."""
    np.random.seed(42)
    n_stocks = 100
    # Most weights are zero, a few are positive
    weights = np.zeros(n_stocks)
    active_idx = np.random.choice(n_stocks, size=30, replace=False)
    weights[active_idx] = np.random.dirichlet(np.ones(30))
    return weights


@pytest.fixture
def sample_fit_result():
    """Sample training fit result."""
    n_epochs = 50
    return {
        "train_losses": list(np.linspace(10, 2, n_epochs) + np.random.randn(n_epochs) * 0.1),
        "val_elbos": list(np.linspace(9, 2.5, n_epochs) + np.random.randn(n_epochs) * 0.05),
        "recon_losses": list(np.linspace(8, 1.5, n_epochs)),
        "kl_losses": list(np.linspace(2, 0.5, n_epochs)),
        "co_losses": list(np.linspace(0.5, 0.1, n_epochs)),
        "au_history": list(np.linspace(5, 20, n_epochs).astype(int)),
        "sigma_sq_history": list(np.linspace(1.0, 0.5, n_epochs)),
        "lr_history": list(np.full(n_epochs, 1e-3)),
    }


@pytest.fixture
def sample_kl_per_dim():
    """Sample KL per dimension array."""
    np.random.seed(42)
    K = 50
    # Most dimensions have low KL, a few are active
    kl = np.random.exponential(0.001, size=K)
    kl[:15] = np.random.uniform(0.02, 0.5, size=15)  # Active units
    return kl


@pytest.fixture
def sample_kl_history():
    """Sample KL per dimension history (epochs, K)."""
    np.random.seed(42)
    n_epochs = 50
    K = 30
    # KL increases during training for active dims
    kl_history = np.zeros((n_epochs, K))
    for i in range(10):  # 10 active dims
        kl_history[:, i] = np.linspace(0.001, 0.1 + i * 0.02, n_epochs)
    return kl_history


@pytest.fixture
def sample_eigenvalues():
    """Sample PCA eigenvalues."""
    np.random.seed(42)
    n_dims = 100
    # Exponentially decaying eigenvalues
    return np.exp(-np.arange(n_dims) * 0.1)


@pytest.fixture
def sample_literature_comparison():
    """Sample literature comparison dict."""
    return {
        "vae_au": 15,
        "eigenvalues_above_mp": 12,
        "marchenko_pastur_edge": 0.05,
        "bai_ng_k": 10,
        "onatski_k": 8,
    }


# ---------------------------------------------------------------------------
# Visualization function tests
# ---------------------------------------------------------------------------


class TestPlotCausalChainDiagram:
    """Tests for plot_causal_chain_diagram()."""

    def test_returns_figure(self, sample_analysis):
        """Should return a matplotlib figure."""
        import matplotlib.pyplot as plt
        fig = plot_causal_chain_diagram(sample_analysis)
        assert fig is not None
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_handles_empty_causes(self):
        """Should handle empty upstream/downstream causes."""
        import matplotlib.pyplot as plt
        analysis = {
            "weakest_component": "unknown",
            "weakest_score": 50.0,
            "causal_analysis": {
                "metric": "test",
                "upstream_causes": [],
                "downstream_effects": [],
            },
        }
        fig = plot_causal_chain_diagram(analysis)
        assert fig is not None
        plt.close("all")

    def test_custom_figsize(self, sample_analysis):
        """Should respect custom figsize."""
        import matplotlib.pyplot as plt
        fig = plot_causal_chain_diagram(sample_analysis, figsize=(10, 4))
        assert fig is not None
        plt.close("all")


class TestPlotFactorExposureComparison:
    """Tests for plot_factor_exposure_comparison()."""

    def test_returns_figure(self, sample_exposure_matrix, sample_weights):
        """Should return a matplotlib figure."""
        import matplotlib.pyplot as plt
        fig = plot_factor_exposure_comparison(
            B_A=sample_exposure_matrix,
            weights=sample_weights,
            AU=20,
        )
        assert fig is not None
        plt.close("all")

    def test_handles_all_zero_weights(self, sample_exposure_matrix):
        """Should handle all-zero weights."""
        import matplotlib.pyplot as plt
        weights = np.zeros(sample_exposure_matrix.shape[0])
        fig = plot_factor_exposure_comparison(
            B_A=sample_exposure_matrix,
            weights=weights,
            AU=20,
        )
        assert fig is not None
        plt.close("all")


class TestPlotTrainingHistoryPanels:
    """Tests for plot_training_history_panels()."""

    def test_returns_figure(self, sample_fit_result):
        """Should return a matplotlib figure."""
        import matplotlib.pyplot as plt
        fig = plot_training_history_panels(sample_fit_result)
        assert fig is not None
        plt.close("all")

    def test_handles_empty_history(self):
        """Should handle empty training history."""
        import matplotlib.pyplot as plt
        fig = plot_training_history_panels({})
        assert fig is not None
        plt.close("all")


class TestPlotModelInspection:
    """Tests for plot_model_inspection()."""

    def test_returns_figure(self, sample_exposure_matrix, sample_kl_per_dim):
        """Should return a matplotlib figure."""
        import matplotlib.pyplot as plt
        fig = plot_model_inspection(
            B_A=sample_exposure_matrix,
            kl_per_dim=sample_kl_per_dim,
            AU=15,
        )
        assert fig is not None
        plt.close("all")


class TestPlotKlPerDimHeatmap:
    """Tests for plot_kl_per_dim_heatmap()."""

    def test_returns_figure(self, sample_kl_history):
        """Should return a matplotlib figure."""
        import matplotlib.pyplot as plt
        fig = plot_kl_per_dim_heatmap(sample_kl_history)
        assert fig is not None
        plt.close("all")

    def test_handles_no_active_units(self):
        """Should handle case with no active units."""
        import matplotlib.pyplot as plt
        kl_history = np.zeros((50, 30))  # All zeros
        fig = plot_kl_per_dim_heatmap(kl_history)
        assert fig is not None
        plt.close("all")


class TestPlotPcaEigenvalueSpectrum:
    """Tests for plot_pca_eigenvalue_spectrum()."""

    def test_returns_figure(self, sample_eigenvalues, sample_literature_comparison):
        """Should return a matplotlib figure."""
        import matplotlib.pyplot as plt
        fig = plot_pca_eigenvalue_spectrum(sample_eigenvalues, sample_literature_comparison)
        assert fig is not None
        plt.close("all")

    def test_handles_missing_mp_edge(self, sample_eigenvalues):
        """Should handle missing Marchenko-Pastur edge."""
        import matplotlib.pyplot as plt
        lit_comp = {"vae_au": 10, "bai_ng_k": 8}
        fig = plot_pca_eigenvalue_spectrum(sample_eigenvalues, lit_comp)
        assert fig is not None
        plt.close("all")


class TestPlotVaePcaCorrelation:
    """Tests for plot_vae_pca_correlation()."""

    def test_returns_figure(self, sample_exposure_matrix):
        """Should return a matplotlib figure."""
        import matplotlib.pyplot as plt
        np.random.seed(42)
        B_pca = np.random.randn(100, 25) * 0.5
        fig = plot_vae_pca_correlation(sample_exposure_matrix, B_pca, k_compare=15)
        assert fig is not None
        plt.close("all")


# ---------------------------------------------------------------------------
# Table generation function tests
# ---------------------------------------------------------------------------


class TestBuildPortfolioHoldingsTable:
    """Tests for build_portfolio_holdings_table()."""

    def test_returns_dataframe(self, sample_weights):
        """Should return a pandas DataFrame."""
        import pandas as pd
        stock_ids = list(range(len(sample_weights)))
        df = build_portfolio_holdings_table(sample_weights, stock_ids)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_filters_low_weights(self, sample_weights):
        """Should filter out positions below min_weight."""
        stock_ids = list(range(len(sample_weights)))
        df = build_portfolio_holdings_table(sample_weights, stock_ids, min_weight=0.01)
        # All weights in table should be >= 1%
        assert (df["Weight (%)"] >= 1.0).all()

    def test_sorted_by_weight_descending(self, sample_weights):
        """Should sort by weight descending."""
        stock_ids = list(range(len(sample_weights)))
        df = build_portfolio_holdings_table(sample_weights, stock_ids)
        weights = df["Weight (%)"].values
        assert all(weights[i] >= weights[i + 1] for i in range(len(weights) - 1))

    def test_uses_ticker_mapping(self, sample_weights):
        """Should use permno_to_ticker mapping."""
        stock_ids = list(range(len(sample_weights)))
        permno_to_ticker = {0: "AAPL", 1: "GOOGL", 2: "MSFT"}
        df = build_portfolio_holdings_table(
            sample_weights, stock_ids, permno_to_ticker=permno_to_ticker
        )
        # Tickers that have mappings should appear
        assert "AAPL" in df["Ticker"].values or df.empty


class TestBuildLiteratureComparisonTable:
    """Tests for build_literature_comparison_table()."""

    def test_returns_markdown(self, sample_literature_comparison):
        """Should return markdown table string."""
        result = build_literature_comparison_table(sample_literature_comparison)
        assert isinstance(result, str)
        assert "|" in result
        assert "VAE Active Units" in result
        assert "Marchenko-Pastur" in result

    def test_handles_missing_values(self):
        """Should handle missing values in comparison dict."""
        lit_comp = {"vae_au": 10}
        result = build_literature_comparison_table(lit_comp)
        assert isinstance(result, str)
        assert "N/A" in result


class TestBuildDecisionRulesTable:
    """Tests for build_decision_rules_table()."""

    def test_returns_markdown(self, sample_analysis):
        """Should return markdown table string."""
        rules = sample_analysis["matching_rules"]
        result = build_decision_rules_table(rules)
        assert isinstance(result, str)
        assert "|" in result
        assert "SOLVER_ISSUE" in result

    def test_handles_empty_rules(self):
        """Should handle empty rules list."""
        result = build_decision_rules_table([])
        assert "No issues detected" in result

    def test_includes_all_columns(self, sample_analysis):
        """Should include all required columns."""
        rules = sample_analysis["matching_rules"]
        result = build_decision_rules_table(rules)
        assert "Rule ID" in result
        assert "Diagnosis" in result
        assert "Confidence" in result
        assert "Severity" in result


class TestBuildRecommendationsTable:
    """Tests for build_recommendations_table()."""

    def test_returns_markdown_for_recognized_actions(self):
        """Should return markdown table for recognized actions."""
        actions = [
            {
                "recognized": True,
                "component": "solver",
                "config_key": "max_iter",
                "suggested_value": 200,
                "rationale": "Increase iterations",
            },
        ]
        result = build_recommendations_table(actions)
        assert isinstance(result, str)
        assert "|" in result
        assert "max_iter" in result

    def test_handles_unrecognized_actions(self):
        """Should handle unrecognized actions."""
        actions = [
            {
                "recognized": False,
                "component": "unknown",
                "original_action": "manual review needed",
            },
        ]
        result = build_recommendations_table(actions)
        assert "manual review" in result

    def test_handles_empty_actions(self):
        """Should handle empty actions list."""
        result = build_recommendations_table([])
        assert "No configuration changes recommended" in result


# ---------------------------------------------------------------------------
# Consolidated notebook helper functions
# ---------------------------------------------------------------------------


class TestDisplayDiagnosticResults:
    """Tests for display_diagnostic_results() consolidation function."""

    @pytest.fixture
    def mock_diagnostics(self):
        """Minimal diagnostics dict for testing."""
        return {
            "_raw_weights": np.array([0.05, 0.03, 0.02, 0.0, 0.0]),
            "_raw_stock_ids": [100, 101, 102, 103, 104],
            "state_bag": {
                "B_A": np.random.randn(5, 10),
                "AU": 10,
            },
            "composite_scores": {},
        }

    @pytest.fixture
    def mock_run_data(self):
        """Minimal run_data dict for testing."""
        return {
            "kl_per_dim_history": np.random.rand(20, 30),
            "pca_eigenvalues": np.sort(np.random.rand(30))[::-1],
            "literature_comparison": {
                "vae_au": 10,
                "eigenvalues_above_mp": 8,
                "bai_ng_k": 12,
                "marchenko_pastur_edge": 0.05,
            },
            "B_A": np.random.randn(50, 10),
            "pca_loadings": np.random.randn(50, 15),
        }

    def test_function_exists(self):
        """Should be importable."""
        from src.integration.notebook_helpers import display_diagnostic_results
        assert callable(display_diagnostic_results)

    def test_returns_summary_dict(self, tmp_path, mock_diagnostics, mock_run_data):
        """Should return a summary dict with display status."""
        from src.integration.notebook_helpers import display_diagnostic_results

        # Create minimal output directory structure
        output_dir = tmp_path / "diagnostic"
        output_dir.mkdir()
        (output_dir / "diagnostic_report.md").write_text("# Test Report\n")

        # Run with all displays disabled (no IPython available)
        result = display_diagnostic_results(
            diagnostics=mock_diagnostics,
            run_data=mock_run_data,
            output_dir=output_dir,
            show_plots=False,
            show_report=False,
            show_holdings=False,
            show_exposures=False,
            show_ml_diagnostics=False,
            export_zip=False,
        )

        assert isinstance(result, dict)
        assert "plots_displayed" in result
        assert "report_displayed" in result
        assert "holdings_displayed" in result
        assert "exposures_displayed" in result
        assert "ml_diagnostics_displayed" in result
        assert "zip_created" in result

    def test_export_zip_creates_archive(self, tmp_path, mock_diagnostics, mock_run_data):
        """Should create ZIP archive when export_zip=True."""
        from src.integration.notebook_helpers import display_diagnostic_results

        output_dir = tmp_path / "diagnostic"
        output_dir.mkdir()
        (output_dir / "test_file.txt").write_text("test content")

        result = display_diagnostic_results(
            diagnostics=mock_diagnostics,
            run_data=mock_run_data,
            output_dir=output_dir,
            show_plots=False,
            show_report=False,
            show_holdings=False,
            show_exposures=False,
            show_ml_diagnostics=False,
            export_zip=True,
        )

        assert result["zip_created"] is True
        assert "zip_path" in result


class TestRunDecisionSynthesis:
    """Tests for run_decision_synthesis() consolidation function."""

    @pytest.fixture
    def mock_diagnostics(self):
        """Diagnostics dict with composite_scores for testing."""
        return {
            "composite_scores": {
                "solver": {"score": 75.0, "grade": "C"},
                "constraint": {"score": 80.0, "grade": "B"},
                "covariance": {"score": 65.0, "grade": "D", "details": {}},
                "reconstruction": {"score": 70.0, "grade": "C"},
                "vae_health": {"score": 85.0, "grade": "B"},
                "factor_model": {"score": 60.0, "grade": "D"},
                "overall": {"score": 72.5, "priority_actions": []},
            },
            "state_bag": {
                "AU": 15,
                "latent_stability_rho": 0.85,
            },
            "training_summary": {},
        }

    def test_function_exists(self):
        """Should be importable."""
        from src.integration.notebook_helpers import run_decision_synthesis
        assert callable(run_decision_synthesis)

    def test_returns_synthesis_dict(self, tmp_path, mock_diagnostics):
        """Should return synthesis dict with analysis results."""
        from src.integration.notebook_helpers import run_decision_synthesis

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = run_decision_synthesis(
            diagnostics=mock_diagnostics,
            output_dir=output_dir,
            show_analysis=False,
            show_rules_table=False,
            show_causal_diagram=False,
            show_recommendations=False,
            export_json=False,
        )

        assert isinstance(result, dict)
        assert "analysis" in result
        assert "matched_rules" in result
        assert "detected_patterns" in result
        assert "exec_actions" in result
        assert "json_valid" in result
        assert "json_path" in result

    def test_exports_valid_json(self, tmp_path, mock_diagnostics):
        """Should export valid JSON when export_json=True."""
        from src.integration.notebook_helpers import run_decision_synthesis
        import json

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = run_decision_synthesis(
            diagnostics=mock_diagnostics,
            output_dir=output_dir,
            show_analysis=False,
            show_rules_table=False,
            show_causal_diagram=False,
            show_recommendations=False,
            export_json=True,
        )

        assert result["json_valid"] is True
        json_path = output_dir / "decision_synthesis.json"
        assert json_path.exists()

        # Verify JSON is parseable
        with open(json_path) as f:
            data = json.load(f)
        assert "overall_score" in data
        assert "severity" in data

    def test_analysis_contains_root_cause(self, tmp_path, mock_diagnostics):
        """Should perform root cause analysis."""
        from src.integration.notebook_helpers import run_decision_synthesis

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = run_decision_synthesis(
            diagnostics=mock_diagnostics,
            output_dir=output_dir,
            show_analysis=False,
            show_rules_table=False,
            show_causal_diagram=False,
            show_recommendations=False,
            export_json=False,
        )

        analysis = result["analysis"]
        assert analysis is not None
        assert "weakest_component" in analysis
        assert "weakest_score" in analysis
