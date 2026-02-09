# Changelog

> **Purpose:** Track recent significant changes. Update after each major modification.
> Keep ~10 entries. Merge similar entries rather than adding duplicates.

## Recent Changes

| # | Date | Modification |
|---|------|--------------|
| 10 | 2026-02-09 | **Cardinality enforcement: 3 optimized strategies + MIQP pre-screening + DCP fix + early checkpoint**: Rewrote `cardinality.py` with 4 selectable methods (`auto`/`sequential`/`gradient`/`miqp`/`two_stage`). (1) **Gradient**: first-order Taylor ΔH_i ≈ \|∂H/∂w_i\| × w_i. (2) **MIQP**: single MOSEK MIQP with binary z_i. (3) **Two-stage** (DVT §4.7): analytical y* in factor space + MIQP tracking QP. **Pre-screening v2**: w_min-aware threshold (`w >= w_min*0.5`) + hard cap (`_MAX_BINARY_VARS=80`); e.g. 194→~60 binary vars. **DCP compliance**: replaced `sum_squares(maximum(0,...))` with `sum(square(pos(...)))` in `sca_solver.py` and `cardinality.py` — `sum_squares` requires affine arg, `pos(convex)` is convex+nonneg so `square(pos(...))` uses nondecreasing composition. **Early checkpoint**: `_run_single_fold()` saves after training, before portfolio opt. Fallback chain: two_stage→miqp→gradient→sequential. 5 files modified, 7 new tests, pyright clean, 233 tests pass. |
| 9 | 2026-02-09 | **SCA solver performance optimizations (DVT-aligned)**: 4 optimizations in `sca_solver.py`: (1) Parametric CVXPY — build problem once with `cp.Parameter`, reuse across SCA iterations (avoids re-compilation). (2) Cholesky pre-factorization — `_safe_cholesky(Sigma)` computed once, shared via `_L_sigma`; `sum_squares(L.T @ w)` replaces `quad_form(w, Sigma)`. (3) Solver chain MOSEK→CLARABEL→ECOS→SCS per DVT "MOSEK/ECOS recommended". (4) Parallel multi-starts via `ThreadPoolExecutor` (solvers release GIL). Removed `solve_sca_subproblem` (replaced by `_ParametricSCAProblem`). 1 file modified, pyright clean, 227 tests pass. |
| 8 | 2026-02-08 | **Direct training mode (skip walk-forward)**: Added `FullPipeline.run_direct()` with same signature as `run()` plus `holdout_start`/`holdout_fraction`. Trains VAE on full period minus holdout with early stopping, evaluates on holdout, runs benchmarks. Added `_state_bag` to `_run_single_fold()` for exposing intermediates (B matrix, risk model, fit history). 5 new notebook cells (Section 4b) with training history + model inspection plots. 2 files modified (`pipeline.py`, `dashboard.ipynb`), pyright clean. |
| 8 | 2026-02-08 | **Wire co-movement loss into trainer (ISD-compliant)**: Fixed bug where `compute_co_movement_loss()` was never called — `Step/co_movement` was always 0. `create_windows()` now returns raw returns (3rd tensor) for Spearman (ISD MOD-004: NOT z-scored). `trainer.fit()` uses `CurriculumBatchSampler` with synchronous+stratified batching during phases 1-2, random phase 3 (INV-010). Strata computed via k-means in pipeline and passed through. 5 files modified (`windowing.py`, `trainer.py`, `pipeline.py`, 2 test files). Pyright clean, 33 tests pass. |
| 7 | 2026-02-07 | **VAE auto-adaptation for small universes**: Added `_adapt_vae_params()` to FullPipeline — 3-lever adaptation (K scaling via AU_max_stat, C_MIN reduction 384→144, r_max relaxation with reinforced reg). Fixed Phase A hardcoded `T_annee=10` bug. Made dropout configurable throughout encoder/decoder/model/build_vae chain. 5 files modified, all 216 unit tests pass, pyright clean. |
| 6 | 2026-02-07 | **SP500 priority + data quality filters**: Added `--sp500-first` to `download_tiingo.py` (fetches SP500 constituents from Wikipedia, downloads them first, then remaining tickers). Added penny stock filter (`--min-price`, default $1.00) and minimum history filter (`--min-history-days`, default 504) to both `phase_merge()` and `load_tiingo_data()`. Also added `--sp500-file` for offline use with local CSV. |
| 5 | 2026-02-07 | **Performance optimizations (auto-adaptive)**: Created `src/utils.py` with 3 hardware adaptation helpers. Auto-detect MPS/CUDA/CPU. AMP mixed precision + DataLoader workers. Vectorized windowing via `sliding_window_view`. 13 new tests in `test_utils.py`. |
| 4 | 2026-02-07 | **Actionable constraint validation**: Added `__post_init__` validation to all 9 config dataclasses (3 shared helpers: `_validate_range`, `_validate_in`, `_validate_pair`). Restored `r_max` capacity check as hard `ValueError` with 4 computed remediation values. Added `_find_max_K()` to `build_vae.py`. 37 new tests in `test_config_validation.py`. |
| 3 | 2026-02-07 | **Dashboard notebook (verified)**: Created `notebooks/dashboard.ipynb` — all 26 code cells execute end-to-end on synthetic data. Fixed benchmark column type mismatch (str vs int permno). Auto-overrides for synthetic mode. Created `src/integration/visualization.py` (5 display helpers). |
| 1 | 2026-02-06 | **Phase 2-4 implementation complete**: 31 source files for MOD-004 through MOD-016 (loss, training, inference, risk model, portfolio, benchmarks, walk-forward, integration). All pyright clean. |


---

## Current State

- **Status**: Development — All 4 phases + tests + dashboard + performance optimizations + TensorBoard complete. 45 source files + 13 test files + 1 notebook. SP500 priority download + penny stock/min history filters active.
- **Main features**: Full data pipeline, 1D-CNN VAE, 3-mode loss with co-movement curriculum, training loop with AMP + TensorBoard + curriculum batching + CUDA T4 optimizations (TF32, fused AdamW, gradient accumulation/checkpointing), inference + AU with AMP autocast, dual-rescaled factor risk model, portfolio optimization (SCA+Armijo, parametric CVXPY, Cholesky, parallel multi-starts, 4-strategy cardinality enforcement with MIQP pre-screening + two-stage decomposition), early training checkpoint, 6 benchmarks, walk-forward (34 folds) + direct training mode, statistical tests, reporting, 2 CLI entry points, dashboard notebook, SP500-first download with data quality filters
- **Next**: End-to-end run on Tiingo SP500 data (VAE auto-adaptation now handles small universes)

---

## Update Protocol

After each significant modification:

1. Add new entry at position 1, shift others down
2. If similar entry exists, update it instead of adding
3. If > 10 entries, remove entry #10
4. Update "Current State" if project status changed
