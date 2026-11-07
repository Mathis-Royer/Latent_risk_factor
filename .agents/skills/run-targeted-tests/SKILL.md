---
name: run-targeted-tests
description: Run only the tests that cover the Python files changed in LatentRiskFactor, report pass/fail counts and failure summaries. Use after editing src/, scripts/ or tests/, or when asked to test a change. Never runs the whole suite.
---

# Run targeted tests

The full suite takes ~13 minutes and is a heavy job: run it only when the user asks or before a merge, under
`heavy_lock.py` (see `.agents/project/instructions.md`).

## 1. Find what changed
```bash
git diff --name-only HEAD -- '*.py'; git ls-files --others --exclude-standard -- '*.py'
```

## 2. Find the tests that import it (always current, unlike a fixed table)
```bash
# module path src/risk_model/covariance.py -> src.risk_model.covariance
grep -rlE "from src\.risk_model\.covariance import|import src\.risk_model\.covariance" tests/
```
Main unit test per package, as a starting point:

| Source | Unit tests |
|---|---|
| `src/data_pipeline/` | `test_data_pipeline.py`, `test_data_quality.py` |
| `src/vae/` | `test_vae_architecture.py`, `test_loss_function.py` |
| `src/training/` | `test_training.py` |
| `src/inference/` | `test_inference.py` |
| `src/risk_model/` | `test_risk_model.py`, `test_factor_quality.py` |
| `src/portfolio/` | `test_portfolio_optimization.py` |
| `src/benchmarks/` | `test_benchmarks.py` |
| `src/walk_forward/` | `test_walk_forward_detailed.py`, `test_oos_rebalancing.py` |
| `src/diagnostics/`, `src/integration/` | `test_diagnostics.py`, `test_composite_scores.py`, `test_decision_rules.py`, `test_visualization.py`, `test_pipeline_state.py`, `test_diagnostic_experiments.py` |
| `src/config.py` | `test_config_validation.py` + the tests of the modules using the changed field |

Cross-cutting files (`test_invariants.py`, `test_conventions.py`, `test_interface_contracts.py`,
`test_edge_cases.py`) import many modules: include them when the grep finds them, they guard the spec invariants.

## 3. Run
```bash
.venv/bin/python -m pytest tests/unit/test_risk_model.py tests/unit/test_invariants.py -x -q --tb=short
# narrow further: file::Class, file::Class::test, or -k "expression"
```
Integration tests (`tests/integration/`, 10-60 s each) only when a change spans several modules; one file at a time.

## 4. Report
Number passed/failed; for each failure the test id and a one-line cause; the exact command used; what was not run.
