"""
Diagnostic plot generation for the full pipeline.

All functions return matplotlib Figure objects and optionally save to PNG.
Used by scripts/run_diagnostic.py and notebooks/dashboard.ipynb.
"""

import logging
import os
from typing import Any

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Consistent styling
_COLORS = {
    "vae": "#2563eb",
    "loss": "#2563eb",
    "recon": "#16a34a",
    "kl": "#dc2626",
    "co": "#f59e0b",
    "val": "#7c3aed",
    "sigma": "#06b6d4",
    "au": "#ec4899",
    "best": "#dc2626",
    "frontier": "#2563eb",
    "alpha_opt": "#dc2626",
}

_BENCH_COLORS = [
    "#94a3b8", "#f97316", "#14b8a6", "#a855f7", "#f43f5e", "#84cc16",
]


# ---------------------------------------------------------------------------
# Training plots
# ---------------------------------------------------------------------------

def plot_training_convergence(
    training_diag: dict[str, Any],
    figsize: tuple[float, float] = (16, 12),
) -> object:
    """
    Plot 4-panel training convergence with overfitting diagnostics.

    Panel layout:
      [0,0] Train Loss + Val ELBO (overlaid) — check convergence & gap
      [0,1] Reconstruction + KL (loss components)
      [1,0] Co-movement loss (curriculum phases visible)
      [1,1] Overfit ratio (val/train) with healthy-band shading

    :param training_diag (dict): Training diagnostics
    :param figsize (tuple): Figure size

    :return fig (object): Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    best_epoch = training_diag.get("best_epoch", 0)

    train_loss = training_diag.get("train_loss", [])
    val_elbo = training_diag.get("val_elbo", [])
    train_recon = training_diag.get("train_recon", [])
    train_kl = training_diag.get("train_kl", [])
    train_co = training_diag.get("train_co", [])

    n_epochs = max(len(train_loss), len(val_elbo), 1)
    epochs = list(range(1, n_epochs + 1))

    def _add_best_line(ax: Any) -> None:
        if 0 < best_epoch <= n_epochs:
            ax.axvline(
                best_epoch, color=_COLORS["best"], linestyle="--",
                alpha=0.6, linewidth=1,
            )

    # --- [0,0] Train Loss + Val ELBO (overlaid) ---
    ax = axes[0][0]
    if train_loss:
        ax.plot(
            epochs[:len(train_loss)], train_loss,
            color=_COLORS["loss"], linewidth=1.5, label="Train Loss",
        )
    if val_elbo:
        ax.plot(
            epochs[:len(val_elbo)], val_elbo,
            color=_COLORS["val"], linewidth=1.5, linestyle="--",
            label="Val ELBO",
        )
    _add_best_line(ax)
    if 0 < best_epoch <= n_epochs:
        ax.plot([], [], color=_COLORS["best"], linestyle="--",
                alpha=0.6, label=f"Best = {best_epoch}")
    ax.set_title("Loss Curves (Train + Val)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- [0,1] Reconstruction + KL ---
    ax = axes[0][1]
    if train_recon:
        ax.plot(
            epochs[:len(train_recon)], train_recon,
            color=_COLORS["recon"], linewidth=1.5, label="Reconstruction",
        )
    if train_kl:
        ax.plot(
            epochs[:len(train_kl)], train_kl,
            color=_COLORS["kl"], linewidth=1.5, label="KL Divergence",
        )
    _add_best_line(ax)
    ax.set_title("Loss Components", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- [1,0] Co-movement loss ---
    ax = axes[1][0]
    if train_co:
        ax.plot(
            epochs[:len(train_co)], train_co,
            color=_COLORS["co"], linewidth=1.5,
        )
    _add_best_line(ax)
    ax.set_title("Co-movement Loss", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # --- [1,1] Overfit ratio (val / train) ---
    ax = axes[1][1]
    if train_loss and val_elbo:
        min_len = min(len(train_loss), len(val_elbo))
        ratio = [
            v / max(t, 1e-10)
            for t, v in zip(train_loss[:min_len], val_elbo[:min_len])
        ]
        ax.plot(
            epochs[:min_len], ratio,
            color="#6366f1", linewidth=1.5,
        )
        ax.axhspan(0.85, 1.5, color="#22c55e", alpha=0.08, label="Healthy [0.85, 1.5]")
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    _add_best_line(ax)
    ax.set_title("Overfit Ratio (Val / Train)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlabel("Epoch", fontsize=9)

    fig.suptitle("Training Convergence", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_sigma_and_au(
    training_diag: dict[str, Any],
    figsize: tuple[float, float] = (14, 5),
) -> object:
    """
    Plot sigma_sq evolution and AU count over training.

    :param training_diag (dict): Training diagnostics
    :param figsize (tuple): Figure size

    :return fig (object): Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Sigma_sq
    sigma = training_diag.get("sigma_sq_series", [])
    if sigma:
        epochs = list(range(1, len(sigma) + 1))
        ax1.plot(epochs, sigma, color=_COLORS["sigma"], linewidth=1.5)
        ax1.set_yscale("log")
    ax1.set_title("Observation Noise (sigma_sq)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=9)
    ax1.set_ylabel("sigma_sq (log scale)", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # AU
    au = training_diag.get("au_series", [])
    if au:
        epochs = list(range(1, len(au) + 1))
        ax2.plot(epochs, au, color=_COLORS["au"], linewidth=1.5, marker="o", markersize=2)
    ax2.set_title("Active Units (AU)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=9)
    ax2.set_ylabel("AU count", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Latent space plots
# ---------------------------------------------------------------------------

def plot_kl_spectrum(
    latent_diag: dict[str, Any],
    figsize: tuple[float, float] = (12, 5),
) -> object:
    """
    Plot KL divergence per latent dimension (sorted, log scale).

    :param latent_diag (dict): Latent space diagnostics
    :param figsize (tuple): Figure size

    :return fig (object): Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    kl_sorted = latent_diag.get("kl_per_dim_sorted", [])
    au = latent_diag.get("AU", 0)

    if kl_sorted:
        x = list(range(len(kl_sorted)))
        colors = [
            _COLORS["vae"] if i < au else "#d1d5db"
            for i in range(len(kl_sorted))
        ]
        ax.bar(x, kl_sorted, color=colors, width=0.8)
        ax.set_yscale("log")
        ax.axhline(0.01, color=_COLORS["best"], linestyle="--", alpha=0.5,
                    label="AU threshold (0.01 nats)")
        if au > 0:
            ax.axvline(au - 0.5, color=_COLORS["best"], linestyle=":",
                       alpha=0.7, label=f"AU = {au}")
        ax.legend(fontsize=9)

    ax.set_title("KL Divergence per Latent Dimension", fontsize=12, fontweight="bold")
    ax.set_xlabel("Dimension (sorted by KL)", fontsize=10)
    ax.set_ylabel("KL (nats, log scale)", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Risk model plots
# ---------------------------------------------------------------------------

def plot_eigenvalue_spectrum(
    risk_diag: dict[str, Any],
    figsize: tuple[float, float] = (12, 5),
) -> object:
    """
    Plot eigenvalue scree plot and cumulative explained variance.

    :param risk_diag (dict): Risk model diagnostics
    :param figsize (tuple): Figure size

    :return fig (object): Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    eigenvalues = risk_diag.get("eigenvalues", [])

    if eigenvalues:
        eig = np.array(eigenvalues)
        x = list(range(1, len(eig) + 1))

        ax1.bar(x, eig, color=_COLORS["vae"], alpha=0.8)
        ax1.set_yscale("log")
        ax1.set_title("Eigenvalue Scree Plot", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Principal Factor", fontsize=9)
        ax1.set_ylabel("Eigenvalue (log)", fontsize=9)
        ax1.grid(True, alpha=0.3, axis="y")

        cumulative = np.cumsum(eig) / np.sum(eig)
        ax2.plot(x, cumulative, color=_COLORS["vae"], linewidth=2, marker="o", markersize=3)
        ax2.axhline(0.9, color="#94a3b8", linestyle="--", alpha=0.5, label="90%")
        ax2.axhline(0.95, color="#94a3b8", linestyle=":", alpha=0.5, label="95%")
        ax2.set_title("Cumulative Explained Variance", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Number of Factors", fontsize=9)
        ax2.set_ylabel("Cumulative Fraction", fontsize=9)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Portfolio plots
# ---------------------------------------------------------------------------

def plot_weight_distribution(
    w_vae: "np.ndarray",
    figsize: tuple[float, float] = (10, 5),
) -> object:
    """
    Plot histogram of portfolio weight distribution.

    :param w_vae (np.ndarray): VAE portfolio weights
    :param figsize (tuple): Figure size

    :return fig (object): Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    w_pos = w_vae[w_vae > 1e-8]
    n_zero = np.sum(w_vae <= 1e-8)

    if len(w_pos) > 0:
        ax1.hist(w_pos, bins=min(50, len(w_pos)), color=_COLORS["vae"],
                 alpha=0.8, edgecolor="white")
        ax1.axvline(np.mean(w_pos), color=_COLORS["best"], linestyle="--",
                    label=f"Mean = {np.mean(w_pos):.4f}")
        ax1.set_title("Active Weight Distribution", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Weight", fontsize=9)
        ax1.set_ylabel("Count", fontsize=9)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

    # Sorted weights
    sorted_w = np.sort(w_vae)[::-1]
    ax2.bar(range(len(sorted_w)), sorted_w, color=_COLORS["vae"], width=1.0, alpha=0.8)
    if n_zero > 0:
        ax2.axvline(len(w_pos) - 0.5, color=_COLORS["best"], linestyle="--",
                     alpha=0.7, label=f"{int(n_zero)} zero-weight stocks")
        ax2.legend(fontsize=8)
    ax2.set_title("Sorted Portfolio Weights", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Stock Rank", fontsize=9)
    ax2.set_ylabel("Weight", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return fig


def plot_frontier(
    portfolio_diag: dict[str, Any],
    figsize: tuple[float, float] = (10, 6),
) -> object:
    """
    Plot variance-entropy frontier with alpha_opt marker.

    :param portfolio_diag (dict): Portfolio diagnostics
    :param figsize (tuple): Figure size

    :return fig (object): Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    frontier_data = portfolio_diag.get("frontier", {})
    alpha_opt = portfolio_diag.get("alpha_opt", 0)

    if frontier_data.get("available", False):
        h_values = frontier_data.get("H_values", [])
        var_values = frontier_data.get("variance_values", [])
        alphas = frontier_data.get("alpha_grid", [])

        if h_values and var_values:
            ax.plot(var_values, h_values, "o-", color=_COLORS["frontier"],
                    linewidth=2, markersize=6)

            # Annotate alpha values
            for i, alpha in enumerate(alphas):
                if i < len(h_values) and i < len(var_values):
                    ax.annotate(
                        f"a={alpha}",
                        (var_values[i], h_values[i]),
                        fontsize=7, textcoords="offset points",
                        xytext=(5, 5), alpha=0.7,
                    )

            # Mark alpha_opt
            h_opt = frontier_data.get("H_at_alpha_opt", 0)
            if h_opt > 0:
                opt_var = None
                for i, a in enumerate(alphas):
                    if a == alpha_opt and i < len(var_values):
                        opt_var = var_values[i]
                        break
                if opt_var is not None:
                    ax.plot(opt_var, h_opt, "*", color=_COLORS["alpha_opt"],
                            markersize=15, zorder=5,
                            label=f"alpha* = {alpha_opt}")
                    ax.legend(fontsize=10)

    ax.set_title("Variance-Entropy Frontier", fontsize=12, fontweight="bold")
    ax.set_xlabel("Portfolio Variance", fontsize=10)
    ax.set_ylabel("Factor Entropy H(w)", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_risk_decomposition(
    portfolio_diag: dict[str, Any],
    figsize: tuple[float, float] = (10, 6),
) -> object:
    """
    Plot factor risk contribution breakdown.

    :param portfolio_diag (dict): Portfolio diagnostics
    :param figsize (tuple): Figure size

    :return fig (object): Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    risk_decomp = portfolio_diag.get("risk_decomposition", {})
    if not risk_decomp.get("available", False):
        ax1.text(0.5, 0.5, "Risk decomposition not available",
                 ha="center", va="center", fontsize=11)
        ax1.axis("off")
        ax2.axis("off")
        fig.tight_layout()
        return fig

    fractions = risk_decomp.get("fractions", [])
    if not fractions:
        ax1.text(0.5, 0.5, "No factor data", ha="center", va="center")
        ax1.axis("off")
        ax2.axis("off")
        fig.tight_layout()
        return fig

    fracs = np.array(fractions)
    n_factors = len(fracs)

    # Top 10 factors bar chart
    n_show = min(10, n_factors)
    ax1.barh(range(n_show), fracs[:n_show], color=_COLORS["vae"], alpha=0.8)
    ax1.set_yticks(range(n_show))
    ax1.set_yticklabels([f"Factor {i+1}" for i in range(n_show)])
    ax1.set_xlabel("Risk Contribution Fraction", fontsize=9)
    ax1.set_title("Top Factor Risk Contributions", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="x")
    ax1.invert_yaxis()

    # Cumulative bar
    cumulative = np.cumsum(fracs)
    ax2.plot(range(1, n_factors + 1), cumulative, "o-", color=_COLORS["vae"],
             linewidth=1.5, markersize=3)
    ax2.axhline(0.9, color="#94a3b8", linestyle="--", alpha=0.5, label="90%")
    ax2.set_title("Cumulative Risk Explained", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Number of Factors", fontsize=9)
    ax2.set_ylabel("Cumulative Fraction", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Benchmark comparison plot
# ---------------------------------------------------------------------------

def plot_benchmark_comparison(
    bench_comparison: dict[str, Any],
    figsize: tuple[float, float] = (14, 7),
) -> object:
    """
    Plot multi-metric bar chart comparing VAE vs benchmarks.

    :param bench_comparison (dict): Benchmark comparison diagnostics
    :param figsize (tuple): Figure size

    :return fig (object): Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    per_bench = bench_comparison.get("per_benchmark", {})
    vae_m = bench_comparison.get("vae_metrics", {})

    if not per_bench:
        ax.text(0.5, 0.5, "No benchmark data", ha="center", va="center", fontsize=11)
        ax.axis("off")
        fig.tight_layout()
        return fig

    # Negate max_drawdown_oos for display: lower MDD = better, shown as
    # positive bar ("neg_max_drawdown") so all bars point "up = good".
    metrics_to_plot = ["sharpe", "ann_return", "H_norm_oos", "sortino", "neg_max_drawdown"]
    strategies = ["VAE"] + list(per_bench.keys())
    n_metrics = len(metrics_to_plot)
    n_strategies = len(strategies)

    x = np.arange(n_metrics)
    width = 0.8 / n_strategies

    def _get_metric(src: dict[str, Any], key: str) -> float:
        """Retrieve metric, negating MDD for display."""
        if key == "neg_max_drawdown":
            return -float(src.get("max_drawdown_oos", 0))
        return float(src.get(key, 0))

    for i, strategy in enumerate(strategies):
        if strategy == "VAE":
            values = [_get_metric(vae_m, m) for m in metrics_to_plot]
            color = _COLORS["vae"]
        else:
            bm = per_bench[strategy].get("bench_metrics", {})
            values = [_get_metric(bm, m) for m in metrics_to_plot]
            color = _BENCH_COLORS[min(i - 1, len(_BENCH_COLORS) - 1)]

        offset = (i - n_strategies / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=strategy, color=color, alpha=0.85)

        # Value labels
        for bar, val in zip(bars, values):
            if abs(val) > 0.001:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=6, rotation=45,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, fontsize=10)
    ax.set_title("Strategy Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Health check summary plot
# ---------------------------------------------------------------------------

def plot_health_summary(
    health_checks: list[dict[str, str]],
    figsize: tuple[float, float] = (10, 4),
) -> object:
    """
    Plot health check summary as a colored status grid.

    :param health_checks (list[dict]): Health checks from diagnostics
    :param figsize (tuple): Figure size

    :return fig (object): Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if not health_checks:
        ax.text(0.5, 0.5, "No health checks", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig

    status_colors = {"OK": "#22c55e", "WARNING": "#f59e0b", "CRITICAL": "#ef4444"}

    n = len(health_checks)
    for i, check in enumerate(health_checks):
        color = status_colors.get(check["status"], "#94a3b8")
        ax.barh(n - 1 - i, 1, color=color, alpha=0.8, edgecolor="white")
        label = f"{check['category']}: {check['check']}"
        ax.text(0.02, n - 1 - i, label, va="center", fontsize=8, fontweight="bold")
        ax.text(0.5, n - 1 - i, check["message"], va="center", fontsize=7, alpha=0.8)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Health Check Summary", fontsize=12, fontweight="bold")

    # Legend
    for status, color in status_colors.items():
        ax.barh([], 0, color=color, label=status)
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Stock count per year
# ---------------------------------------------------------------------------

def plot_stock_count_per_year(
    data_quality: dict[str, Any],
    train_end: str = "",
    oos_start: str = "",
    figsize: tuple[float, float] = (14, 5),
) -> object:
    """
    Plot number of listed stocks per year with train/holdout split.

    :param data_quality (dict): Data quality diagnostics (with stocks_per_year)
    :param train_end (str): Last date of the training period (YYYY-MM-DD)
    :param oos_start (str): First date of the holdout period (YYYY-MM-DD)
    :param figsize (tuple): Figure size

    :return fig (object): Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    spy = data_quality.get("stocks_per_year", {})
    years = spy.get("years", [])
    mean_counts = spy.get("mean", [])
    min_counts = spy.get("min", [])
    max_counts = spy.get("max", [])

    if not years:
        ax.text(0.5, 0.5, "No stock count data", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig

    # Colour bars by training vs holdout period
    split_year = int(oos_start[:4]) if oos_start else 9999
    colors = [
        _COLORS["vae"] if y < split_year else "#ef4444"
        for y in years
    ]

    ax.bar(years, mean_counts, color=colors, alpha=0.7, edgecolor="white")

    # Min-max error bars
    if min_counts and max_counts:
        yerr_lo = [m - mn for m, mn in zip(mean_counts, min_counts)]
        yerr_hi = [mx - m for m, mx in zip(mean_counts, max_counts)]
        ax.errorbar(
            years, mean_counts, yerr=[yerr_lo, yerr_hi],
            fmt="none", color="gray", alpha=0.4, capsize=2,
        )

    # Vertical split line
    if oos_start:
        ax.axvline(
            split_year - 0.5, color=_COLORS["best"], linestyle="--",
            linewidth=2, label=f"Holdout start ({oos_start[:10]})",
        )
        ax.legend(fontsize=9)

    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Number of Stocks (mean per day)", fontsize=10)
    ax.set_title("Listed Stocks per Year", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    if len(years) > 15:
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Cumulative returns on holdout
# ---------------------------------------------------------------------------

def _compute_portfolio_returns(
    returns_oos: pd.DataFrame,
    weights: "np.ndarray",
    stock_ids: list[int],
) -> "np.ndarray":
    """
    Compute daily portfolio log-returns from weights and OOS returns.

    :param returns_oos (pd.DataFrame): OOS returns (dates x stocks)
    :param weights (np.ndarray): Portfolio weights (n,)
    :param stock_ids (list[int]): Stock identifiers matching weights

    :return daily_returns (np.ndarray): Daily portfolio log-returns (T,)
    """
    n_w = len(weights)
    ids_used = stock_ids[:n_w]
    avail_mask = np.array([sid in returns_oos.columns for sid in ids_used])
    avail_idx = np.where(avail_mask)[0]
    if len(avail_idx) == 0:
        return np.zeros(len(returns_oos))
    avail_ids = [ids_used[i] for i in avail_idx]
    w_sub = weights[avail_idx].copy()
    w_sum = float(w_sub.sum())
    if w_sum > 1e-10:
        w_sub = w_sub / w_sum
    r_mat = np.asarray(returns_oos[avail_ids].fillna(0.0).values, dtype=np.float64)
    return r_mat @ w_sub


def _sharpe_from_daily(daily_returns: "np.ndarray") -> float:
    """
    Compute annualised Sharpe from daily log-returns.

    :param daily_returns (np.ndarray): Daily portfolio log-returns

    :return sharpe (float): Annualised Sharpe ratio
    """
    if len(daily_returns) < 2:
        return 0.0
    ann_ret = float(np.mean(daily_returns) * 252)
    ann_vol = float(np.std(daily_returns, ddof=1) * np.sqrt(252))
    return ann_ret / max(ann_vol, 1e-10)


# Extended colour palette for benchmarks (enough for 7 benchmarks)
_BENCH_COLORS_EXT = [
    "#94a3b8", "#f97316", "#14b8a6", "#a855f7", "#f43f5e", "#84cc16",
    "#0ea5e9",
]

_SP500_COLOR = "#000000"


def plot_cumulative_returns(
    returns_oos: pd.DataFrame,
    w_vae: "np.ndarray",
    inferred_stock_ids: list[int],
    benchmark_weights: dict[str, dict[str, Any]],
    benchmark_results: dict[str, list[dict[str, float]]],
    figsize: tuple[float, float] = (14, 7),
) -> object:
    """
    Plot cumulative returns on holdout for VAE, all benchmarks, and S&P 500.

    :param returns_oos (pd.DataFrame): OOS returns (dates x stocks)
    :param w_vae (np.ndarray): VAE portfolio weights
    :param inferred_stock_ids (list[int]): Stock IDs matching w_vae
    :param benchmark_weights (dict): {name: {weights, universe, daily_returns}}
    :param benchmark_results (dict): {name: [fold_metrics]}
    :param figsize (tuple): Figure size

    :return fig (object): Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    dates = returns_oos.index

    # --- VAE ---
    vae_daily = _compute_portfolio_returns(returns_oos, w_vae, inferred_stock_ids)
    vae_cum = (np.exp(np.cumsum(vae_daily)) - 1) * 100
    vae_sharpe = _sharpe_from_daily(vae_daily)
    ax.plot(
        dates, vae_cum, color=_COLORS["vae"], linewidth=2.5,
        label=f"VAE (Sharpe={vae_sharpe:.2f})",
    )

    # --- Benchmarks ---
    bench_color_idx = 0
    for bench_name, bw in benchmark_weights.items():
        is_sp500 = bench_name == "sp500_index"

        # Get daily returns
        if bw.get("daily_returns") is not None:
            daily = np.asarray(bw["daily_returns"], dtype=np.float64)
        elif bw.get("weights") is not None and bw.get("universe") is not None:
            w_bench = np.asarray(bw["weights"], dtype=np.float64)
            u_int = [int(s) for s in bw["universe"]]
            daily = _compute_portfolio_returns(returns_oos, w_bench, u_int)
        else:
            continue

        if len(daily) != len(returns_oos):
            continue

        cum = (np.exp(np.cumsum(daily)) - 1) * 100
        sharpe = _sharpe_from_daily(daily)
        display_name = "S&P 500" if is_sp500 else bench_name.replace("_", " ").title()

        if is_sp500:
            ax.plot(
                dates, cum, color=_SP500_COLOR, linewidth=2,
                linestyle="--", label=f"{display_name} (Sharpe={sharpe:.2f})",
            )
        else:
            color = _BENCH_COLORS_EXT[bench_color_idx % len(_BENCH_COLORS_EXT)]
            bench_color_idx += 1
            ax.plot(
                dates, cum, color=color, linewidth=1.2, alpha=0.8,
                label=f"{display_name} (Sharpe={sharpe:.2f})",
            )

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Cumulative Return (%)", fontsize=10)
    ax.set_title(
        "Holdout Cumulative Returns — All Strategies",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Save all plots
# ---------------------------------------------------------------------------

def save_all_plots(
    diagnostics: dict[str, Any],
    w_vae: "np.ndarray",
    output_dir: str = "results/diagnostic/plots",
    *,
    returns_oos: Any = None,
    benchmark_weights: dict[str, dict[str, Any]] | None = None,
    benchmark_results: dict[str, list[dict[str, float]]] | None = None,
    inferred_stock_ids: list[int] | None = None,
    train_end: str = "",
    oos_start: str = "",
) -> list[str]:
    """
    Generate and save all diagnostic plots as PNG files.

    :param diagnostics (dict): Full diagnostics from collect_diagnostics()
    :param w_vae (np.ndarray): VAE portfolio weights
    :param output_dir (str): Output directory for PNG files
    :param returns_oos (pd.DataFrame | None): OOS returns for cumulative plot
    :param benchmark_weights (dict | None): {name: {weights, universe, daily_returns}}
    :param benchmark_results (dict | None): {name: [fold_metrics]}
    :param inferred_stock_ids (list[int] | None): Stock IDs matching w_vae
    :param train_end (str): Last training date (YYYY-MM-DD)
    :param oos_start (str): First OOS date (YYYY-MM-DD)

    :return files (list[str]): List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved: list[str] = []

    plot_specs: list[tuple[str, object]] = []

    training = diagnostics.get("training", {})
    latent = diagnostics.get("latent", {})
    risk = diagnostics.get("risk_model", {})
    portfolio = diagnostics.get("portfolio", {})
    bench = diagnostics.get("benchmark_comparison", {})
    checks = diagnostics.get("health_checks", [])
    data_qual = diagnostics.get("data_quality", {})

    # Training plots
    if training.get("available", False):
        plot_specs.append(("training_convergence", plot_training_convergence(training)))
        plot_specs.append(("sigma_and_au", plot_sigma_and_au(training)))

    # Latent space
    if latent.get("kl_per_dim_sorted"):
        plot_specs.append(("kl_spectrum", plot_kl_spectrum(latent)))

    # Risk model
    if risk.get("eigenvalues"):
        plot_specs.append(("eigenvalue_spectrum", plot_eigenvalue_spectrum(risk)))

    # Portfolio
    if len(w_vae) > 0:
        plot_specs.append(("weight_distribution", plot_weight_distribution(w_vae)))

    if portfolio.get("frontier", {}).get("available", False):
        plot_specs.append(("frontier", plot_frontier(portfolio)))

    if portfolio.get("risk_decomposition", {}).get("available", False):
        plot_specs.append(("risk_decomposition", plot_risk_decomposition(portfolio)))

    # Benchmark comparison
    if bench.get("per_benchmark"):
        plot_specs.append(("benchmark_comparison", plot_benchmark_comparison(bench)))

    # Health summary
    if checks:
        plot_specs.append(("health_summary", plot_health_summary(checks)))

    # Stock count per year
    if data_qual.get("stocks_per_year", {}).get("years"):
        plot_specs.append((
            "stock_count_per_year",
            plot_stock_count_per_year(data_qual, train_end=train_end, oos_start=oos_start),
        ))

    # Cumulative returns on holdout
    if (
        returns_oos is not None
        and len(w_vae) > 0
        and benchmark_weights
        and inferred_stock_ids
    ):
        plot_specs.append((
            "cumulative_returns",
            plot_cumulative_returns(
                returns_oos, w_vae, inferred_stock_ids,
                benchmark_weights, benchmark_results or {},
            ),
        ))

    # Save all
    for name, fig in plot_specs:
        path = os.path.join(output_dir, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")  # type: ignore[union-attr]
        plt.close(fig)  # type: ignore[arg-type]
        saved.append(path)
        logger.info("Plot saved: %s", path)

    return saved
