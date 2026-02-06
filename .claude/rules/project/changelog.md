# Changelog

> **Purpose:** Track recent significant changes. Update after each major modification.
> Keep ~10 entries. Merge similar entries rather than adding duplicates.

## Recent Changes

| # | Date | Modification |
|---|------|--------------|
| 4 | 2026-02-06 | **ISD data schema update**: Separated F=2 (returns+vol) vs F>2 (volume, intraday range, sector deviation, skewness) data requirements; removed delisting_code; added EODHD as production data source (2000-2025); added EODHD column mapping |
| 3 | 2026-02-06 | **ISD completeness**: Added Phase 1 ISD sections for MOD-001 (data_pipeline), MOD-002 (vae_architecture), MOD-003 (test_infrastructure); translated French Appendix A to English in DVT |
| 2 | 2026-02-06 | **Project scaffolding**: Created full src/ and tests/ directory tree, __init__.py files, centralized config.py with all ISD parameters, filled project documentation |
| 1 | 2026-02-06 | **Initial setup**: Created `.claude/` configuration, translated ISD and DVT documents |

---

## Current State

- **Status**: Development — scaffolding complete, implementation not started
- **Main features**: Project structure, centralized config (frozen dataclasses), full documentation
- **In progress**: Phase 1 infrastructure (MOD-001 data_pipeline, MOD-002 vae_architecture, MOD-003 test_infrastructure)

---

## Update Protocol

After each significant modification:

1. Add new entry at position 1, shift others down
2. If similar entry exists, update it instead of adding
3. If > 10 entries, remove entry #10
4. Update "Current State" if project status changed
