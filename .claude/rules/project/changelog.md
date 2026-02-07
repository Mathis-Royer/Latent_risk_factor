# Changelog

> **Purpose:** Track recent significant changes. Update after each major modification.
> Keep ~10 entries. Merge similar entries rather than adding duplicates.

## Recent Changes

| # | Date | Modification |
|---|------|--------------|
| 10 | 2026-02-07 | **Code-vs-spec audit + 4 corrections**: Full audit of 44 source files against ISD/DVT. 0 divergences. 4 missing implementations fixed: `crisis_period_return()` (ISD L3 primary), `realized_vs_predicted_correlation()` (ISD L2), `--config PATH` CLI arg, `--device` arg in benchmarks CLI. Updated `implementation_decisions.md` with Corrections + Divergences sections. |
| 9 | 2026-02-07 | **Actionable constraint validation**: Added `__post_init__` validation to all 9 config dataclasses (3 shared helpers: `_validate_range`, `_validate_in`, `_validate_pair`). Restored `r_max` capacity check as hard `ValueError` with 4 computed remediation values (n_min, T_annee_min, K_max via binary search, r_max_needed). Added `_find_max_K()` to `build_vae.py`. 37 new tests in `test_config_validation.py`. Total: 144 tests (144 pass). |
| 8 | 2026-02-07 | **Dashboard notebook (verified)**: Created `notebooks/dashboard.ipynb` — all 26 code cells execute end-to-end on synthetic data. Fixed benchmark column type mismatch (str vs int permno). Auto-overrides for synthetic mode (max_epochs=2, HP_GRID=1 config). Created `src/integration/visualization.py` (5 display helpers). Externalized `serialize_for_json()` to reporting.py. |
| 7 | 2026-02-06 | **Phase 3+4 implementation complete**: 15 source files — MOD-010 to MOD-015 (6 benchmarks), MOD-009 (5 walk-forward files), MOD-016 (pipeline.py + statistical_tests.py + reporting.py). All pyright clean. |
| 6 | 2026-02-06 | **Phase 2 implementation complete**: 16 source files for MOD-004 through MOD-008. All pyright clean. |
| 5 | 2026-02-06 | **Phase 1 implementation complete**: 12 source files for MOD-001 (data_pipeline), MOD-002 (vae_architecture), MOD-003 (test_infrastructure). All pyright clean. |
| 4 | 2026-02-06 | **Volume promoted to core schema**: Moved `volume` from extended-only (F>2) to core columns in ISD for ADV filter; updated 6 ISD sections; fixed config.py defaults to match DVT baselines |
| 3 | 2026-02-06 | **Implementation decisions log**: Added `docs/implementation_decisions.md` to track spec gap decisions; added ISD Section 00 guideline for mandatory logging |
| 2 | 2026-02-06 | **ISD data schema update**: Separated F=2 vs F>2 data requirements; added EODHD as production data source (2000-2025) |
| 1 | 2026-02-06 | **Initial setup**: Created `.claude/` configuration, translated ISD and DVT documents |

---

## Current State

- **Status**: Development — All 4 phases + tests + dashboard complete. 44 source files + 11 test files + 1 notebook. 144 tests (144 pass, 0 skipped). Pyright 0 errors.
- **Main features**: Full data pipeline, 1D-CNN VAE, 3-mode loss, training loop, inference + AU, dual-rescaled factor risk model, portfolio optimization (SCA+Armijo), 6 benchmarks, walk-forward (34 folds), statistical tests, reporting, 2 CLI entry points, dashboard notebook
- **Next**: End-to-end run on real data (CRSP/EODHD)

---

## Update Protocol

After each significant modification:

1. Add new entry at position 1, shift others down
2. If similar entry exists, update it instead of adding
3. If > 10 entries, remove entry #10
4. Update "Current State" if project status changed
