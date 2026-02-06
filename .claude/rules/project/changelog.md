# Changelog

> **Purpose:** Track recent significant changes. Update after each major modification.
> Keep ~10 entries. Merge similar entries rather than adding duplicates.

## Recent Changes

| # | Date | Modification |
|---|------|--------------|
| 7 | 2026-02-06 | **Phase 1 implementation complete**: Implemented all 12 source files for MOD-001 (data_pipeline: data_loader, returns, features, universe, windowing, crisis), MOD-002 (vae_architecture: build_vae, encoder, decoder, model), MOD-003 (test_infrastructure: synthetic_data, known_solutions). All pyright clean (0 errors). Smoke tests pass: VAE forward/encode shapes correct, sizing rules match DVT table, data pipeline end-to-end (synthetic CSV → returns → vol → windows → crisis labels), z-scoring CONV-02 verified, log-returns CONV-01 verified. |
| 6 | 2026-02-06 | **Volume promoted to core schema**: Moved `volume` from extended-only (F>2) to core columns in ISD for ADV filter; updated 6 ISD sections; fixed config.py defaults to match DVT baselines |
| 5 | 2026-02-06 | **Implementation decisions log**: Added `docs/implementation_decisions.md` to track spec gap decisions; added ISD Section 00 guideline for mandatory logging |
| 4 | 2026-02-06 | **ISD data schema update**: Separated F=2 vs F>2 data requirements; added EODHD as production data source (2000-2025) |
| 3 | 2026-02-06 | **ISD completeness**: Added Phase 1 ISD sections for MOD-001, MOD-002, MOD-003; translated French Appendix A to English in DVT |
| 2 | 2026-02-06 | **Project scaffolding**: Created full src/ and tests/ directory tree, __init__.py files, centralized config.py with all ISD parameters |
| 1 | 2026-02-06 | **Initial setup**: Created `.claude/` configuration, translated ISD and DVT documents |

---

## Current State

- **Status**: Development — Phase 1 complete, ready for Phase 2
- **Main features**: Full data pipeline (synthetic + loader + returns + vol + universe + windows + crisis), 1D-CNN VAE architecture (InceptionHead + ResBlock encoder/decoder, VAEModel with scalar σ²), test infrastructure (synthetic generators + known solutions)
- **Next**: Phase 2 — MOD-004 (loss_function), MOD-005 (training), MOD-006 (inference), MOD-007 (risk_model), MOD-008 (portfolio_optimization)

---

## Update Protocol

After each significant modification:

1. Add new entry at position 1, shift others down
2. If similar entry exists, update it instead of adding
3. If > 10 entries, remove entry #10
4. Update "Current State" if project status changed
