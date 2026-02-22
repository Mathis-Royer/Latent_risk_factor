"""
Visualization helpers for notebook dashboard display.

Provides plotting and formatting functions for walk-forward results.
All functions return matplotlib figures or styled DataFrames for
inline notebook rendering.

Reference: ISD Section MOD-016 — Sub-task 4.
"""

from typing import Any

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Fold metrics comparison
# ---------------------------------------------------------------------------


def plot_fold_metrics(
    vae_results: list[dict[str, float]],
    benchmark_results: dict[str, list[dict[str, float]]],
    metrics: list[str] | None = None,
    figsize: tuple[float, float] = (16, 10),
) -> object:
    """
    Plot per-fold metrics: VAE vs all benchmarks.

    :param vae_results (list[dict]): Per-fold VAE metrics
    :param benchmark_results (dict): Benchmark name -> per-fold metrics
    :param metrics (list[str] | None): Metrics to plot
    :param figsize (tuple): Figure size

    :return fig (object): Matplotlib figure
    """
    if metrics is None:
        metrics = ["sharpe", "ann_vol_oos", "max_drawdown_oos", "H_norm_oos"]

    vae_df = pd.DataFrame(vae_results)
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)

    if n_metrics == 1:
        axes = [axes]

    fold_ids = vae_df["fold_id"].values if "fold_id" in vae_df.columns else np.arange(len(vae_df))

    for ax, metric in zip(axes, metrics):
        if metric in vae_df.columns:
            ax.plot(fold_ids, vae_df[metric].values, "o-", label="VAE", linewidth=2, markersize=5, color="#2563eb")

        for bench_name, bench_metrics in benchmark_results.items():
            bench_df = pd.DataFrame(bench_metrics)
            if metric in bench_df.columns:
                ax.plot(fold_ids, bench_df[metric].values, "s--", label=bench_name, alpha=0.7, markersize=3)

        ax.set_ylabel(metric, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best", ncol=3)

    axes[-1].set_xlabel("Fold ID", fontsize=11)
    fig.suptitle("Walk-Forward Fold Metrics: VAE vs Benchmarks", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# E* distribution
# ---------------------------------------------------------------------------


def plot_e_star_distribution(
    e_stars: list[int],
    figsize: tuple[float, float] = (8, 4),
) -> object:
    """
    Plot E* (optimal epochs) distribution across folds.

    :param e_stars (list[int]): E* values per fold
    :param figsize (tuple): Figure size

    :return fig (object): Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    e_arr = np.array(e_stars)

    ax.bar(range(len(e_arr)), e_arr, color="#2563eb", alpha=0.8, edgecolor="white")
    ax.axhline(float(np.mean(e_arr)), color="#dc2626", linestyle="--", linewidth=1.5,
               label=f"Mean = {np.mean(e_arr):.1f}")

    ax.set_xlabel("Fold", fontsize=11)
    ax.set_ylabel("E* (epochs)", fontsize=11)
    ax.set_title("Optimal Training Epochs per Fold", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Statistical tests heatmap
# ---------------------------------------------------------------------------


def plot_pairwise_heatmap(
    report: dict[str, Any],
    figsize: tuple[float, float] = (10, 6),
) -> object:
    """
    Plot heatmap of pairwise test results (VAE vs benchmarks).

    :param report (dict): Full report from generate_report()
    :param figsize (tuple): Figure size

    :return fig (object): Matplotlib figure
    """
    tests = report.get("statistical_tests", {})
    pairwise = tests.get("pairwise", {})

    if not pairwise:
        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        ax.text(0.5, 0.5, "No pairwise test results available",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    benchmarks = list(pairwise.keys())
    all_metrics: list[str] = []
    for bench_data in pairwise.values():
        all_metrics.extend(bench_data.keys())
    metrics = sorted(set(all_metrics))

    delta_matrix = np.full((len(benchmarks), len(metrics)), np.nan)
    sig_matrix = np.zeros((len(benchmarks), len(metrics)), dtype=bool)

    for i, bench in enumerate(benchmarks):
        for j, metric in enumerate(metrics):
            result = pairwise.get(bench, {}).get(metric, {})
            if result.get("skipped", False):
                continue
            delta_matrix[i, j] = result.get("median_delta", np.nan)
            sig_matrix[i, j] = result.get("significant_corrected", False)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(delta_matrix, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(benchmarks)))
    ax.set_yticklabels(benchmarks, fontsize=9)

    for i in range(len(benchmarks)):
        for j in range(len(metrics)):
            val = delta_matrix[i, j]
            if np.isnan(val):
                continue
            marker = "*" if sig_matrix[i, j] else ""
            ax.text(j, i, f"{val:+.3f}{marker}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Median effect (VAE - Benchmark)")
    ax.set_title("Pairwise Tests: VAE vs Benchmarks (* = significant after Holm-Bonferroni)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary styling
# ---------------------------------------------------------------------------


def style_summary_table(
    summary_df: pd.DataFrame,
) -> object:
    """
    Apply conditional styling to a summary statistics DataFrame.

    :param summary_df (pd.DataFrame): From summary_statistics()

    :return styled (Styler): Styled DataFrame for notebook display
    """
    numeric_cols = [c for c in summary_df.columns if c != "metric"]
    styler = summary_df.style.format(
        {c: "{:.4f}" for c in numeric_cols},
    )
    return styler


def style_fold_table(
    fold_df: pd.DataFrame,
    highlight_cols: list[str] | None = None,
) -> object:
    """
    Apply conditional coloring to per-fold metrics DataFrame.

    Green = better, Red = worse (relative to column median).

    :param fold_df (pd.DataFrame): Per-fold metrics
    :param highlight_cols (list[str] | None): Columns to highlight

    :return styled (Styler): Styled DataFrame for notebook display
    """
    if highlight_cols is None:
        highlight_cols = ["sharpe", "H_norm_oos", "ann_vol_oos", "max_drawdown_oos"]

    numeric_cols = fold_df.select_dtypes(include=[np.number]).columns.tolist()
    styler = fold_df.style.format({c: "{:.4f}" for c in numeric_cols})
    return styler


# ---------------------------------------------------------------------------
# Causal chain visualization
# ---------------------------------------------------------------------------


def plot_causal_chain_diagram(
    analysis: dict[str, Any],
    figsize: tuple[float, float] = (14, 5),
) -> object:
    """
    Plot causal chain flow diagram from root cause analysis.

    Visualizes upstream causes -> weakest component -> downstream effects
    as a horizontal flow diagram with colored boxes and arrows.

    :param analysis (dict): Root cause analysis dict from get_root_cause_analysis()
        Expected keys: weakest_component, weakest_score, causal_analysis
    :param figsize (tuple): Figure size (width, height)

    :return fig (object): Matplotlib figure
    """
    import matplotlib.patches as mpatches

    causal = analysis.get("causal_analysis", {})
    metric = causal.get("metric", "unknown")
    upstream = causal.get("upstream_causes", [])
    downstream = causal.get("downstream_effects", [])

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    weakest = analysis.get("weakest_component", "unknown")
    weakest_score = analysis.get("weakest_score", 0)
    ax.set_title(
        f"Causal Chain for Weakest Component: {weakest} (score: {weakest_score:.1f})",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    def draw_box(x: float, y: float, text: str, color: str, width: float = 1.8) -> None:
        """Draw a rounded rectangle with text."""
        rect = mpatches.FancyBboxPatch(
            (x - width / 2, y - 0.35),
            width,
            0.7,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            facecolor=color,
            edgecolor="#333333",
            linewidth=1.5,
        )
        ax.add_patch(rect)
        display_text = text.replace("_", "\n") if len(text) > 12 else text
        ax.text(x, y, display_text, ha="center", va="center", fontsize=9, fontweight="medium", wrap=True)

    # Central metric (weakest component)
    draw_box(6, 2, metric.replace("_", "\n"), "#ff9999", width=2.2)

    # Upstream causes (left side)
    n_up = min(4, len(upstream))
    for i, cause in enumerate(upstream[:n_up]):
        x = 1.2 + i * 1.3
        draw_box(x, 3.2, cause, "#99ccff", width=1.5)
        ax.annotate(
            "",
            xy=(6 - 1.1, 2.35),
            xytext=(x + 0.6, 2.85),
            arrowprops={"arrowstyle": "->", "color": "#3366cc", "lw": 1.5},
        )

    # Downstream effects (right side)
    n_down = min(4, len(downstream))
    for i, effect in enumerate(downstream[:n_down]):
        x = 7 + i * 1.3
        draw_box(x, 0.8, effect, "#99ff99", width=1.5)
        ax.annotate(
            "",
            xy=(x - 0.4, 1.15),
            xytext=(6 + 1.1, 1.65),
            arrowprops={"arrowstyle": "->", "color": "#339933", "lw": 1.5},
        )

    # Legend
    legend_items = [
        mpatches.Patch(facecolor="#99ccff", edgecolor="#333", label="Upstream Causes"),
        mpatches.Patch(facecolor="#ff9999", edgecolor="#333", label="Weakest Component"),
        mpatches.Patch(facecolor="#99ff99", edgecolor="#333", label="Downstream Effects"),
    ]
    ax.legend(handles=legend_items, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Factor exposure comparison
# ---------------------------------------------------------------------------


def plot_factor_exposure_comparison(
    B_A: np.ndarray,
    weights: np.ndarray,
    AU: int = 0,
    n_factors_show: int = 30,
    n_stocks_show: int = 50,
    figsize: tuple[float, float] | None = None,
) -> object:
    """
    Side-by-side heatmaps of factor exposures before/after optimization.

    Left panel: All stocks (equal-weight universe order)
    Right panel: Selected stocks sorted by portfolio weight descending

    :param B_A (np.ndarray): Exposure matrix (n_stocks, AU)
    :param weights (np.ndarray): Portfolio weights (n_stocks,)
    :param AU (int): Number of active units (for title)
    :param n_factors_show (int): Max factors to display (columns)
    :param n_stocks_show (int): Max stocks to display per panel (rows)
    :param figsize (tuple | None): Figure size; auto-calculated if None

    :return fig (object): Matplotlib figure
    """
    weights = np.asarray(weights)
    selected_mask = weights > 1e-8
    n_selected = int(np.sum(selected_mask))

    n_factors_show = min(n_factors_show, AU if AU > 0 else B_A.shape[1])

    # BEFORE: Top N stocks by universe order
    n_before = min(n_stocks_show, B_A.shape[0])
    B_before = B_A[:n_before, :n_factors_show]

    # AFTER: Selected stocks sorted by weight descending
    if n_selected == 0:
        # No stocks selected - show empty panel
        B_after = np.zeros((1, n_factors_show))
        w_sorted = np.array([0.0])
        n_after_show = 0
    else:
        B_after_full = B_A[selected_mask, :]
        w_selected = weights[selected_mask]

        sort_idx = np.argsort(w_selected)[::-1]
        B_after = B_after_full[sort_idx, :n_factors_show]
        w_sorted = w_selected[sort_idx]

        n_after_show = min(n_stocks_show, len(sort_idx))
        B_after = B_after[:n_after_show, :]
        w_sorted = w_sorted[:n_after_show]

    # Figure size
    if figsize is None:
        figsize = (16, max(8, max(1, n_after_show) * 0.2))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Color limits for consistent comparison
    vmax = float(np.abs(B_before).max())
    if n_selected > 0 and B_after.size > 0:
        vmax = max(vmax, float(np.abs(B_after).max()))
    vmin = -vmax

    # LEFT: Before optimization
    axes[0].imshow(B_before, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[0].set_xlabel(f"Latent Factor (1 to {n_factors_show})")
    axes[0].set_ylabel("Stock (universe order)")
    axes[0].set_title(f"BEFORE: All Stocks ({n_before}/{B_A.shape[0]})\nEqual-weight exposure")

    # RIGHT: After optimization
    im2 = axes[1].imshow(B_after, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[1].set_xlabel(f"Latent Factor (1 to {n_factors_show})")
    axes[1].set_ylabel("Stock (sorted by weight)")
    axes[1].set_title(f"AFTER: Selected Stocks ({n_after_show}/{n_selected})\nSorted by entropy-optimal weight")

    # Weight annotations on y-axis for "after" plot
    y_labels = [f"{w_sorted[i] * 100:.1f}%" for i in range(len(w_sorted))]
    axes[1].set_yticks(range(len(y_labels)))
    axes[1].set_yticklabels(y_labels, fontsize=7)
    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.tick_right()

    # Shared colorbar
    fig.colorbar(im2, ax=axes, orientation="vertical", fraction=0.02, pad=0.04, label="Exposure (z-scored)")

    fig.suptitle(
        f"Factor Exposure Heatmap — Before vs After Shannon Entropy Optimization\n" f"AU = {AU} active dimensions",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Training history panels
# ---------------------------------------------------------------------------


def plot_training_history_panels(
    fit_result: dict[str, Any],
    figsize: tuple[float, float] = (14, 10),
) -> object:
    """
    4-panel training history visualization.

    Panel 1: Total loss (train) and ELBO (validation) over epochs
    Panel 2: Loss components (reconstruction, KL, co-movement)
    Panel 3: Active units (AU) evolution
    Panel 4: Observation noise (sigma_sq) and learning rate

    :param fit_result (dict): Training result dict from trainer.fit()
        Expected keys: train_losses, val_elbos, recon_losses, kl_losses,
        co_losses, au_history, sigma_sq_history, lr_history
    :param figsize (tuple): Figure size

    :return fig (object): Matplotlib figure
    """
    train_losses = fit_result.get("train_losses", [])
    val_elbos = fit_result.get("val_elbos", [])
    recon_losses = fit_result.get("recon_losses", [])
    kl_losses = fit_result.get("kl_losses", [])
    co_losses = fit_result.get("co_losses", [])
    au_history = fit_result.get("au_history", [])
    sigma_sq_history = fit_result.get("sigma_sq_history", [])
    lr_history = fit_result.get("lr_history", [])

    n_epochs = len(train_losses)
    if n_epochs == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No training history available", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    epochs = np.arange(1, n_epochs + 1)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Panel 1: Total loss and ELBO
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=1.5)
    if val_elbos:
        ax1.plot(epochs[: len(val_elbos)], val_elbos, "r--", label="Val ELBO", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss and Validation ELBO")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Loss components
    ax2 = axes[0, 1]
    if recon_losses:
        ax2.plot(epochs[: len(recon_losses)], recon_losses, "g-", label="Reconstruction", linewidth=1.5)
    if kl_losses:
        ax2.plot(epochs[: len(kl_losses)], kl_losses, "m-", label="KL Divergence", linewidth=1.5)
    if co_losses:
        ax2.plot(epochs[: len(co_losses)], co_losses, "c-", label="Co-movement", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss Component")
    ax2.set_title("Loss Decomposition")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Active units
    ax3 = axes[1, 0]
    if au_history:
        ax3.plot(epochs[: len(au_history)], au_history, "o-", color="#2563eb", markersize=3, linewidth=1.5)
        ax3.axhline(au_history[-1], color="#dc2626", linestyle="--", label=f"Final AU = {au_history[-1]}")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Active Units (AU)")
    ax3.set_title("Active Units Evolution")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: Sigma_sq and learning rate
    ax4 = axes[1, 1]
    if sigma_sq_history:
        ax4.plot(epochs[: len(sigma_sq_history)], sigma_sq_history, "b-", label="sigma_sq", linewidth=1.5)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("sigma_sq", color="blue")
    ax4.tick_params(axis="y", labelcolor="blue")

    if lr_history:
        ax4_twin = ax4.twinx()
        ax4_twin.plot(epochs[: len(lr_history)], lr_history, "r--", label="Learning Rate", linewidth=1.5)
        ax4_twin.set_ylabel("Learning Rate", color="red")
        ax4_twin.tick_params(axis="y", labelcolor="red")
        ax4_twin.set_yscale("log")

    ax4.set_title("Observation Noise and Learning Rate")
    ax4.grid(True, alpha=0.3)

    fig.suptitle("VAE Training History", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Model inspection (exposure matrix + KL)
# ---------------------------------------------------------------------------


def plot_model_inspection(
    B_A: np.ndarray,
    kl_per_dim: np.ndarray,
    AU: int = 0,
    n_dims_show: int = 20,
    figsize: tuple[float, float] = (14, 5),
) -> object:
    """
    Exposure matrix heatmap + KL bar chart.

    Left: Exposure matrix B_A (stocks x latent dims)
    Right: Marginal KL per dimension with AU threshold line

    :param B_A (np.ndarray): Exposure matrix (n_stocks, K or AU)
    :param kl_per_dim (np.ndarray): KL divergence per latent dimension
    :param AU (int): Number of active units
    :param n_dims_show (int): Max dimensions to show in heatmap
    :param figsize (tuple): Figure size

    :return fig (object): Matplotlib figure
    """
    n_stocks, n_dims = B_A.shape
    n_show = min(n_dims_show, n_dims)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Exposure matrix heatmap
    ax1 = axes[0]
    im = ax1.imshow(B_A[:, :n_show], aspect="auto", cmap="RdBu_r")
    ax1.set_xlabel(f"Active Latent Dimension (showing {n_show}/{AU if AU > 0 else n_dims})")
    ax1.set_ylabel("Stock")
    ax1.set_title(f"Exposure Matrix B_A ({n_stocks} stocks x {AU if AU > 0 else n_dims} active dims)")
    fig.colorbar(im, ax=ax1, shrink=0.8)

    # Right: KL bar chart
    ax2 = axes[1]
    sorted_kl = np.sort(kl_per_dim)[::-1]
    ax2.bar(range(len(sorted_kl)), sorted_kl, color="#2563eb", alpha=0.7)
    ax2.axhline(0.01, color="#dc2626", linestyle="--", linewidth=1.5, label="AU threshold (0.01)")
    ax2.set_xlabel("Dimension (sorted by KL)")
    ax2.set_ylabel("Marginal KL (nats)")
    ax2.set_title(f"Latent Dimension Usage — AU={AU}/{len(kl_per_dim)}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# KL per-dimension heatmap (training dynamics)
# ---------------------------------------------------------------------------


def plot_kl_per_dim_heatmap(
    kl_per_dim_history: np.ndarray,
    au_threshold: float = 0.01,
    figsize: tuple[float, float] = (14, 5),
) -> object:
    """
    Heatmap of KL per latent dimension over training epochs.

    Left: Full heatmap (all K dimensions)
    Right: Active units only (sorted by final KL)

    :param kl_per_dim_history (np.ndarray): Shape (n_epochs, K)
    :param au_threshold (float): KL threshold for active units
    :param figsize (tuple): Figure size

    :return fig (object): Matplotlib figure
    """
    E, K = kl_per_dim_history.shape

    # Identify active units at final epoch
    final_kl = kl_per_dim_history[-1, :]
    active_mask = final_kl > au_threshold
    n_active = int(active_mask.sum())

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Full heatmap
    im1 = axes[0].imshow(kl_per_dim_history.T, aspect="auto", cmap="viridis", extent=[0, E, K, 0])
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Latent Dimension")
    axes[0].set_title(f"KL per Dimension over Training (K={K})")
    plt.colorbar(im1, ax=axes[0], label="KL (nats)")

    # Right: Active units only (sorted by final KL)
    if n_active > 0:
        active_idx = np.where(active_mask)[0]
        sorted_idx = active_idx[np.argsort(-final_kl[active_idx])]
        kl_active = kl_per_dim_history[:, sorted_idx]

        im2 = axes[1].imshow(kl_active.T, aspect="auto", cmap="plasma", extent=[0, E, n_active, 0])
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Active Unit (sorted by final KL)")
        axes[1].set_title(f"Active Units Only (AU={n_active}, KL>{au_threshold})")
        plt.colorbar(im2, ax=axes[1], label="KL (nats)")
    else:
        axes[1].text(0.5, 0.5, "No active units found", ha="center", va="center")
        axes[1].axis("off")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# PCA eigenvalue spectrum with Marchenko-Pastur edge
# ---------------------------------------------------------------------------


def plot_pca_eigenvalue_spectrum(
    eigenvalues: np.ndarray,
    literature_comparison: dict[str, Any],
    n_show: int = 100,
    figsize: tuple[float, float] = (10, 5),
) -> object:
    """
    PCA eigenvalue spectrum with Marchenko-Pastur edge and factor count markers.

    :param eigenvalues (np.ndarray): PCA eigenvalues (sorted descending)
    :param literature_comparison (dict): Comparison metrics
        Expected keys: marchenko_pastur_edge, eigenvalues_above_mp, bai_ng_k, vae_au
    :param n_show (int): Number of eigenvalues to display
    :param figsize (tuple): Figure size

    :return fig (object): Matplotlib figure
    """
    mp_edge = literature_comparison.get("marchenko_pastur_edge")
    mp_signal_count = literature_comparison.get("eigenvalues_above_mp", 0)
    bai_ng_k = literature_comparison.get("bai_ng_k", 0)
    au = literature_comparison.get("vae_au", 0)

    fig, ax = plt.subplots(figsize=figsize)

    n_show = min(n_show, len(eigenvalues))
    ax.bar(range(n_show), eigenvalues[:n_show], alpha=0.7, label="Eigenvalues")

    # Marchenko-Pastur edge
    if mp_edge is not None:
        ax.axhline(y=mp_edge, color="red", linestyle="--", linewidth=2, label=f"MP Edge = {mp_edge:.4f}")

    # Mark different thresholds
    if mp_signal_count > 0:
        ax.axvline(x=mp_signal_count - 0.5, color="red", linestyle=":", alpha=0.7, label=f"MP signal count = {mp_signal_count}")
    if bai_ng_k > 0:
        ax.axvline(x=bai_ng_k - 0.5, color="green", linestyle=":", alpha=0.7, label=f"Bai-Ng IC2 k = {bai_ng_k}")
    if au > 0:
        ax.axvline(x=au - 0.5, color="purple", linestyle=":", alpha=0.7, label=f"VAE AU = {au}")

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("PCA Eigenvalue Spectrum vs Random Matrix Theory")
    ax.legend(loc="upper right")
    ax.set_yscale("log")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# VAE vs PCA loading correlation
# ---------------------------------------------------------------------------


def plot_vae_pca_correlation(
    B_vae: np.ndarray,
    B_pca: np.ndarray,
    k_compare: int = 20,
    figsize: tuple[float, float] = (12, 5),
) -> object:
    """
    Correlation analysis between VAE and PCA factor loadings.

    Left: Correlation heatmap (|Spearman rho|)
    Right: Best PCA match per VAE factor

    :param B_vae (np.ndarray): VAE exposure matrix (n_stocks, AU)
    :param B_pca (np.ndarray): PCA loadings (n_stocks, K_pca)
    :param k_compare (int): Number of factors to compare
    :param figsize (tuple): Figure size

    :return fig (object): Matplotlib figure
    """
    from scipy import stats

    _, au = B_vae.shape
    k_pca = B_pca.shape[1]
    k_compare = min(k_compare, au, k_pca)

    # Compute correlation matrix using Spearman rank correlation
    corr_matrix = np.zeros((k_compare, k_compare))
    for i in range(k_compare):
        for j in range(k_compare):
            # Compute rank correlation manually to avoid scipy type issues
            ranks_vae = stats.rankdata(B_vae[:, i])
            ranks_pca = stats.rankdata(B_pca[:, j])
            rho = np.corrcoef(ranks_vae, ranks_pca)[0, 1]
            corr_matrix[i, j] = abs(rho) if np.isfinite(rho) else 0.0

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Correlation heatmap
    im = axes[0].imshow(corr_matrix, cmap="YlOrRd", vmin=0, vmax=1)
    axes[0].set_xlabel("PCA Factor")
    axes[0].set_ylabel("VAE Factor")
    axes[0].set_title(f"VAE vs PCA Factor Correlation (top {k_compare})")
    plt.colorbar(im, ax=axes[0], label="|Spearman rho|")

    # Right: Best match per VAE factor
    best_match = corr_matrix.max(axis=1)
    axes[1].bar(range(k_compare), best_match, color="steelblue", alpha=0.8)
    axes[1].axhline(y=0.5, color="red", linestyle="--", label="rho=0.5")
    axes[1].set_xlabel("VAE Factor")
    axes[1].set_ylabel("Best PCA Match |rho|")
    axes[1].set_title("Best PCA Match per VAE Factor")
    axes[1].legend()

    fig.tight_layout()
    return fig
