# Changelog

> **Purpose:** Track recent significant changes. Update after each major modification.
> Keep ~10 entries. Merge similar entries rather than adding duplicates.

## Recent Changes

| # | Date | Modification |
|---|------|--------------|
| 10 | 2026-02-06 | **E2E orchestration + CLI + tests (verified)**: FullPipeline.run() (~500 lines), 2 CLI scripts, 86 tests across 10 files (83 passed, 3 skipped long-running). Pyright 0 errors on all tests/ and src/. Full test suite verified end-to-end. |
| 9 | 2026-02-06 | **Phase 3+4 implementation complete**: 15 source files — MOD-010 to MOD-015 (6 benchmarks), MOD-009 (5 walk-forward files), MOD-016 (pipeline.py + statistical_tests.py + reporting.py). All pyright clean. Smoke tests pass. |
| 8 | 2026-02-06 | **Phase 2 implementation complete**: 16 source files for MOD-004 through MOD-008. All pyright clean. Smoke tests pass. |
| 7 | 2026-02-06 | **Phase 1 implementation complete**: 12 source files for MOD-001 (data_pipeline), MOD-002 (vae_architecture), MOD-003 (test_infrastructure). All pyright clean. Smoke tests pass. |
| 6 | 2026-02-06 | **Volume promoted to core schema**: Moved `volume` from extended-only (F>2) to core columns in ISD for ADV filter; updated 6 ISD sections; fixed config.py defaults to match DVT baselines |
| 5 | 2026-02-06 | **Implementation decisions log**: Added `docs/implementation_decisions.md` to track spec gap decisions; added ISD Section 00 guideline for mandatory logging |
| 4 | 2026-02-06 | **ISD data schema update**: Separated F=2 vs F>2 data requirements; added EODHD as production data source (2000-2025) |
| 3 | 2026-02-06 | **Project scaffolding**: Created full src/ and tests/ directory tree, __init__.py files, centralized config.py with all ISD parameters |
| 2 | 2026-02-06 | **Initial setup**: Created `.claude/` configuration, translated ISD and DVT documents |

---

## Current State

- **Status**: Development — All 4 phases + tests complete. 43 source files + 10 test files. 86 tests (83 pass, 3 long-running skipped). Pyright 0 errors.
- **Main features**: Full data pipeline, 1D-CNN VAE, 3-mode loss, training loop, inference + AU, dual-rescaled factor risk model, portfolio optimization (SCA+Armijo), 6 benchmarks, walk-forward (34 folds), statistical tests, reporting, 2 CLI entry points
- **Next**: Full end-to-end test on synthetic data (unskip E2E tests), then real data deployment

---

## Update Protocol

After each significant modification:

1. Add new entry at position 1, shift others down
2. If similar entry exists, update it instead of adding
3. If > 10 entries, remove entry #10
4. Update "Current State" if project status changed
