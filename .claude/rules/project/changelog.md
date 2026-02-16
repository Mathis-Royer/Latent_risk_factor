# Changelog

> **Purpose:** Track recent significant changes. Update after each major modification.
> Keep ~10 entries. Merge similar entries rather than adding duplicates.

## Recent Changes

| # | Date | Modification |
|---|------|--------------|
| 1 | 2026-02-16 | **Deep audit phase 11: Alpha selection + PCA benchmark ENB fix — 2 findings across 4 files**: (1) **IMPORTANT: Min-variance among qualifying alphas** (Meucci 2009, DeMiguel et al. 2009) — `select_operating_alpha()` now picks the alpha with **minimum variance** among all qualifying points (ENB_mono ≥ target), instead of the smallest qualifying alpha. On U-shaped variance-entropy frontiers, the old logic selected alpha=0.005 (Var=6.22e-5) while alpha=2.0 was Pareto-dominant (-16.7% variance, +37.6% entropy). At low alpha entropy is too weak to diversify; at high alpha entropy forces factor diversification which reduces variance through correlation structure. (2) **BUG: PCA benchmark naive ENB target (INV-012 violation)** — PCA used `max(2.0, k/2)` = 15.0 for k=30, but ENB_spectrum=1.71 making target 8.8x unreachable → forced to extreme alpha → negative Sharpe (-0.006/-0.009). Extracted shared `compute_adaptive_enb_target()` function into frontier.py, used by both pipeline.py and pca_factor_rp.py. Formula: `max(2.0, min(n_signal/2, ENB_spectrum * 0.7))`. PCA benchmarks now produce positive Sharpe. 4 files modified (frontier.py, pipeline.py, pca_factor_rp.py, test_portfolio_optimization.py), pyright 0 errors, 468 tests pass (5 new). |
| 2 | 2026-02-16 | **Phase 10: Cardinality + momentum activation — 3 changes across 3 files**: (1) `_MAX_BINARY_VARS` 80→150 in cardinality.py — lifts MIQP ceiling to avoid fallback to greedy elimination for larger universes. (2) `momentum_enabled` False→True, `momentum_weight` 0.02→0.05 in config.py — activates momentum signal by default. (3) `np.asarray`→`np.array(copy=True)` in momentum.py — fixes read-only array error from pandas Series. |
| 3 | 2026-02-16 | **Phase 9 audit: 5 fixes across 5 files** (post-VAE portfolio + benchmarks): (1) **CRITICAL: Adaptive ENB target** (Meucci 2009, Roncalli 2013 Ch. 7) — replaced naive `n_signal/2` target with `min(n_signal/2, ENB_spectrum * 0.7)` where `ENB_spectrum = exp(H(eigenvalue_spectrum))` is the maximum achievable ENB for the current eigenvalue structure. (2) **BUG: Fallback argmax(enb_mono)** — `select_operating_alpha` fallback used raw `enb` instead of `enb_mono` (monotone envelope). (3) **INV-012: constraint_params complete** — added missing `w_bar` and `alpha_grid`. (4) **INV-012: effective_cap = min(w_bar, w_max)** in base.py and min_variance.py. (5) **DOC: ERC near-uniform warning**. 5+2 files modified, pyright 0 errors, 440 unit tests pass. |
| 4 | 2026-02-16 | **Deep audit phase 8: 7 literature-backed fixes across 5 files**: (1) w_bar 0.01→0.03 (Brodie 2009). (2) Market PC1 excluded from entropy (Meucci 2009). (3) B_A_SHRINKAGE_ALPHA 0.15→0.0. (4) PCA benchmark frontier (INV-012). (5) ERC NaN handling. (6) var_ratio fillna(0)@w. (7) PHI 20→0.0. 5 files modified, pyright 0 errors, 440 unit tests pass. |
| 5 | 2026-02-15 | **Deep audit phase 7: 5 fixes across 5 files**: (1) w_old alignment by stock ID. (2) Bayesian VT shrinkage (Barra USE4). (3) entropy_idio_weight 0.2→0.05. (4) lstsq rank diagnostic. (5) Diversified frontier seeds. 5 files modified, pyright 0 errors, 77 unit tests pass. |
| 6 | 2026-02-15 | **Deep audit phase 6: 5 fixes across 5 files**: (1) ENB target on H_factor only (Meucci 2009). (2) DGJ/EWMA decoupled. (3) Per-factor VT (Barra USE4). (4) INV-012 turnover symmetry. (5) w_max/w_bar doc. 5 files modified, pyright 0 errors, 198 unit + 22 integration tests pass. |
| 7 | 2026-02-15 | **Deep audit phases 3-5: 19 fixes across 9 files**: Highlights: full Sigma_assets for risk (DGJ 2018), frontier returns weights, turnover tracking + net Sharpe (DeMiguel 2009), ENB monotone envelope, WLS regression (Fama-MacBeth 1973), disjoint VT holdout (Shephard-Sheppard 2010), hard concentration constraint (Jagannathan-Ma 2003), D_eps kurtosis correction, parametric CVXPY, factor regression lstsq, D_eps James-Stein corrected (Ledoit-Wolf 2004), B_A normalization ddof, H_norm_signal, Sigma_assets symmetrization. |
| 8 | 2026-02-15 | **Deep audit phases 1-2: 15 fixes across 15 files**: Highlights: lambda_risk 1.0→252.0 (CRITICAL scale mismatch), two-layer entropy (Roncalli 2013/Meucci 2009), ERC budget_scale c=n→c=1, block-level VT (Barra USE4), EWMA half-life 0→252, momentum lost in cardinality fix, B_A scale normalization, alpha grid expansion, n_starts 3→5, D_eps James-Stein shrinkage. |
| 9 | 2026-02-14 | **Comprehensive diagnostic metrics audit — 7 fixes**: Benchmark return/Sharpe aligned to geometric, dual VT non-degenerate (Barra USE4), MDD in percentage, var ratio NaN bias, convergence NaN guard, KL top-3 fraction, overfit thresholds aligned. 6 files modified, pyright 0 errors, 436 tests pass. |
| 10 | 2026-02-10 | **E2E diagnostic pipeline + covariance overestimation fixes**: Created diagnostics.py, diagnostic_report.py, diagnostic_plots.py, run_diagnostic.py (health checks, MD/JSON/CSV/PNG reports). Exponential decay aggregation (half_life) + eigenvalue truncation (eigenvalue_pct) to reduce B·Σ_z·B^T overestimation. |


---

## Current State

- **Status**: Development — All 4 phases + tests + dashboard + performance optimizations + TensorBoard + diagnostic pipeline complete. 48 source files + 13 test files + 1 notebook. SP500 priority download + penny stock/min history filters active.
- **Main features**: Full data pipeline, 1D-CNN VAE, 3-mode loss with co-movement curriculum, training loop with AMP + TensorBoard + curriculum batching + CUDA T4 optimizations (TF32, fused AdamW, gradient accumulation/checkpointing), inference + AU with AMP autocast, dual-rescaled factor risk model, portfolio optimization (SCA+Armijo, parametric CVXPY, Cholesky, parallel multi-starts, 4-strategy cardinality enforcement with MIQP pre-screening + two-stage decomposition), early training checkpoint, 6 benchmarks, walk-forward (34 folds) + direct training mode, statistical tests, reporting, **E2E diagnostic pipeline** (health checks, MD/JSON/CSV/PNG reports, quick/full profiles), 3 CLI entry points, dashboard notebook, SP500-first download with data quality filters
- **Next**: Full Tiingo SP500 diagnostic re-run to verify Phase 11 alpha selection improvement (expect alpha_opt shift to ~2.0, PCA benchmarks positive Sharpe)

---

## Update Protocol

After each significant modification:

1. Add new entry at position 1, shift others down
2. If similar entry exists, update it instead of adding
3. If > 10 entries, remove entry #10
4. Update "Current State" if project status changed
