# Changelog

> **Purpose:** Track recent significant changes. Update after each major modification.
> Keep ~10 entries. Merge similar entries rather than adding duplicates.

## Recent Changes

| # | Date | Modification |
|---|------|--------------|
| 10 | 2026-02-06 | **Phase 3+4 implementation complete**: Implemented 15 source files — MOD-010 to MOD-015 (base.py, equal_weight.py, inverse_vol.py, min_variance.py, erc.py, pca_factor_rp.py, pca_vol.py: all 6 benchmarks with shared constraints), MOD-009 (folds.py: 34+1 folds, phase_a.py: composite scoring + elimination, phase_b.py: E* determination + sanity, metrics.py: 3-layer evaluation, selection.py: deployment criteria A/B/C/D), MOD-016 (pipeline.py: FullPipeline orchestrator, statistical_tests.py: Wilcoxon + bootstrap + Holm-Bonferroni + regime decomposition, reporting.py: full report generation + text formatting). All pyright clean (0 errors). Smoke tests pass: EW/IV/MV/ERC/PCA-RP/PCA-Vol all sum to 1.0, fold schedule 34+1 validated, Wilcoxon p=0.005, Holm-Bonferroni rejects correctly, report generates with Scenario A. |
| 9 | 2026-02-06 | **Phase 2 implementation complete**: Implemented 16 source files for MOD-004 through MOD-008. All pyright clean. Smoke tests pass. |
| 8 | 2026-02-06 | **Phase 1 implementation complete**: Implemented all 12 source files for MOD-001 (data_pipeline), MOD-002 (vae_architecture), MOD-003 (test_infrastructure). All pyright clean. Smoke tests pass. |
| 7 | 2026-02-06 | **Volume promoted to core schema**: Moved `volume` from extended-only (F>2) to core columns in ISD for ADV filter; updated 6 ISD sections; fixed config.py defaults to match DVT baselines |
| 6 | 2026-02-06 | **Implementation decisions log**: Added `docs/implementation_decisions.md` to track spec gap decisions; added ISD Section 00 guideline for mandatory logging |
| 5 | 2026-02-06 | **ISD data schema update**: Separated F=2 vs F>2 data requirements; added EODHD as production data source (2000-2025) |
| 4 | 2026-02-06 | **ISD completeness**: Added Phase 1 ISD sections for MOD-001, MOD-002, MOD-003; translated French Appendix A to English in DVT |
| 3 | 2026-02-06 | **Project scaffolding**: Created full src/ and tests/ directory tree, __init__.py files, centralized config.py with all ISD parameters |
| 2 | 2026-02-06 | **Initial setup**: Created `.claude/` configuration, translated ISD and DVT documents |

---

## Current State

- **Status**: Development — All 4 phases complete (43 source files across MOD-001 to MOD-016). Full pipeline implementation done.
- **Main features**: Full data pipeline, 1D-CNN VAE, 3-mode loss, training loop, inference + AU, dual-rescaled factor risk model, portfolio optimization (SCA+Armijo), 6 benchmarks, walk-forward (34 folds), statistical tests, reporting
- **Next**: Integration tests, full end-to-end test on synthetic data, then real data deployment

---

## Update Protocol

After each significant modification:

1. Add new entry at position 1, shift others down
2. If similar entry exists, update it instead of adding
3. If > 10 entries, remove entry #10
4. Update "Current State" if project status changed
