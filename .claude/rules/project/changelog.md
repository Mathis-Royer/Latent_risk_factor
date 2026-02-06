# Changelog

> **Purpose:** Track recent significant changes. Update after each major modification.
> Keep ~10 entries. Merge similar entries rather than adding duplicates.

## Recent Changes

| # | Date | Modification |
|---|------|--------------|
| 8 | 2026-02-06 | **Phase 2 implementation complete**: Implemented 16 source files for MOD-004 (loss.py: 3 modes P/F/A, crisis weighting, co-movement loss, curriculum scheduling, validation ELBO), MOD-005 (trainer.py, batching.py, early_stopping.py, scheduler.py: training loop, curriculum batching, early stopping, LR scheduling), MOD-006 (composite.py, active_units.py: stride-1 inference, AU measurement, truncation), MOD-007 (rescaling.py, factor_regression.py, covariance.py, conditioning.py: dual rescaling, OLS, Ledoit-Wolf Σ_z, D_ε, conditioning guard), MOD-008 (entropy.py, sca_solver.py, constraints.py, cardinality.py, frontier.py: entropy+gradient, SCA+Armijo, multi-start, cardinality enforcement, variance-entropy frontier). All pyright clean (0 errors). Smoke tests pass: crisis weighting 3.0x, curriculum phases correct, gradient matches numerical (5.8e-10 error), training loop trains+stops, inference deterministic, dual rescaling differs, rotation preserves Σ_assets, SCA converges. |
| 7 | 2026-02-06 | **Phase 1 implementation complete**: Implemented all 12 source files for MOD-001 (data_pipeline), MOD-002 (vae_architecture), MOD-003 (test_infrastructure). All pyright clean. Smoke tests pass. |
| 6 | 2026-02-06 | **Volume promoted to core schema**: Moved `volume` from extended-only (F>2) to core columns in ISD for ADV filter; updated 6 ISD sections; fixed config.py defaults to match DVT baselines |
| 5 | 2026-02-06 | **Implementation decisions log**: Added `docs/implementation_decisions.md` to track spec gap decisions; added ISD Section 00 guideline for mandatory logging |
| 4 | 2026-02-06 | **ISD data schema update**: Separated F=2 vs F>2 data requirements; added EODHD as production data source (2000-2025) |
| 3 | 2026-02-06 | **ISD completeness**: Added Phase 1 ISD sections for MOD-001, MOD-002, MOD-003; translated French Appendix A to English in DVT |
| 2 | 2026-02-06 | **Project scaffolding**: Created full src/ and tests/ directory tree, __init__.py files, centralized config.py with all ISD parameters |
| 1 | 2026-02-06 | **Initial setup**: Created `.claude/` configuration, translated ISD and DVT documents |

---

## Current State

- **Status**: Development — Phase 1 + Phase 2 complete, ready for Phase 3 (benchmarks) and Phase 4 (walk-forward + integration)
- **Main features**: Full data pipeline, 1D-CNN VAE architecture, 3-mode loss function, training loop with curriculum batching, inference + AU measurement, dual-rescaled factor risk model (Ledoit-Wolf), portfolio optimization (SCA+Armijo+multi-start+cardinality+frontier)
- **Next**: Phase 3 — MOD-010 to MOD-015 (benchmarks), Phase 4 — MOD-009 (walk_forward), MOD-016 (integration)

---

## Update Protocol

After each significant modification:

1. Add new entry at position 1, shift others down
2. If similar entry exists, update it instead of adding
3. If > 10 entries, remove entry #10
4. Update "Current State" if project status changed
