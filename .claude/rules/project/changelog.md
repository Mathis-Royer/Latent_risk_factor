# Changelog

> **Purpose:** Track recent significant changes. Update after each major modification.
> Keep ~10 entries. Merge similar entries rather than adding duplicates.

## Recent Changes

| # | Date | Modification |
|---|------|--------------|
| 10 | 2026-02-08 | **CUDA T4 optimizations (MPS-compatible, ISD-aligned)**: Added TF32 backend flags, fused AdamW, gradient accumulation, gradient checkpointing to trainer. Extended `torch.compile` to MPS. AMP autocast for inference (`composite.py`, `active_units.py`). Centralized `configure_backend()` and `clear_device_cache()` in `utils.py`. GPU peak memory logging + inter-fold cleanup in `pipeline.py`. 2 new `TrainingConfig` fields (`gradient_accumulation_steps`, `gradient_checkpointing`). 6 files modified, pyright clean, 184 tests pass. |
| 9 | 2026-02-08 | **Direct training mode (skip walk-forward)**: Added `FullPipeline.run_direct()` with same signature as `run()` plus `holdout_start`/`holdout_fraction`. Trains VAE on full period minus holdout with early stopping, evaluates on holdout, runs benchmarks. Added `_state_bag` to `_run_single_fold()` for exposing intermediates (B matrix, risk model, fit history). 5 new notebook cells (Section 4b) with training history + model inspection plots. 2 files modified (`pipeline.py`, `dashboard.ipynb`), pyright clean. |
| 8 | 2026-02-08 | **Wire co-movement loss into trainer (ISD-compliant)**: Fixed bug where `compute_co_movement_loss()` was never called — `Step/co_movement` was always 0. `create_windows()` now returns raw returns (3rd tensor) for Spearman (ISD MOD-004: NOT z-scored). `trainer.fit()` uses `CurriculumBatchSampler` with synchronous+stratified batching during phases 1-2, random phase 3 (INV-010). Strata computed via k-means in pipeline and passed through. 5 files modified (`windowing.py`, `trainer.py`, `pipeline.py`, 2 test files). Pyright clean, 33 tests pass. |
| 7 | 2026-02-07 | **VAE auto-adaptation for small universes**: Added `_adapt_vae_params()` to FullPipeline — 3-lever adaptation (K scaling via AU_max_stat, C_MIN reduction 384→144, r_max relaxation with reinforced reg). Fixed Phase A hardcoded `T_annee=10` bug. Made dropout configurable throughout encoder/decoder/model/build_vae chain. 5 files modified, all 216 unit tests pass, pyright clean. |
| 6 | 2026-02-07 | **SP500 priority + data quality filters**: Added `--sp500-first` to `download_tiingo.py` (fetches SP500 constituents from Wikipedia, downloads them first, then remaining tickers). Added penny stock filter (`--min-price`, default $1.00) and minimum history filter (`--min-history-days`, default 504) to both `phase_merge()` and `load_tiingo_data()`. Also added `--sp500-file` for offline use with local CSV. |
| 5 | 2026-02-07 | **Performance optimizations (auto-adaptive)**: Created `src/utils.py` with 3 hardware adaptation helpers. Auto-detect MPS/CUDA/CPU. AMP mixed precision + DataLoader workers. Vectorized windowing via `sliding_window_view`. 13 new tests in `test_utils.py`. |
| 4 | 2026-02-07 | **Actionable constraint validation**: Added `__post_init__` validation to all 9 config dataclasses (3 shared helpers: `_validate_range`, `_validate_in`, `_validate_pair`). Restored `r_max` capacity check as hard `ValueError` with 4 computed remediation values. Added `_find_max_K()` to `build_vae.py`. 37 new tests in `test_config_validation.py`. |
| 3 | 2026-02-07 | **Dashboard notebook (verified)**: Created `notebooks/dashboard.ipynb` — all 26 code cells execute end-to-end on synthetic data. Fixed benchmark column type mismatch (str vs int permno). Auto-overrides for synthetic mode. Created `src/integration/visualization.py` (5 display helpers). |
| 2 | 2026-02-06 | **Phase 3+4 implementation complete**: 15 source files — MOD-010 to MOD-015 (6 benchmarks), MOD-009 (5 walk-forward files), MOD-016 (pipeline.py + statistical_tests.py + reporting.py). All pyright clean. |
| 1 | 2026-02-06 | **Phase 2 implementation complete**: 16 source files for MOD-004 through MOD-008. All pyright clean. |


---

## Current State

- **Status**: Development — All 4 phases + tests + dashboard + performance optimizations + TensorBoard complete. 45 source files + 13 test files + 1 notebook. SP500 priority download + penny stock/min history filters active.
- **Main features**: Full data pipeline, 1D-CNN VAE, 3-mode loss with co-movement curriculum, training loop with AMP + TensorBoard + curriculum batching + CUDA T4 optimizations (TF32, fused AdamW, gradient accumulation/checkpointing), inference + AU with AMP autocast, dual-rescaled factor risk model, portfolio optimization (SCA+Armijo), 6 benchmarks, walk-forward (34 folds) + direct training mode, statistical tests, reporting, 2 CLI entry points, dashboard notebook, SP500-first download with data quality filters
- **Next**: End-to-end run on Tiingo SP500 data (VAE auto-adaptation now handles small universes)

---

## Update Protocol

After each significant modification:

1. Add new entry at position 1, shift others down
2. If similar entry exists, update it instead of adding
3. If > 10 entries, remove entry #10
4. Update "Current State" if project status changed
