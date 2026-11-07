# Project overview

What the project is for and how its pipelines fit together (the "what" and "why"). How each part works is in the
specification documents under `docs/`; the code layout is in [structure.md](structure.md); the data in
[data.md](data.md).

## Overview

End-to-end portfolio construction based on latent risk factors learned by a Variational Autoencoder. A 1D-CNN VAE is
trained on windows of daily stock returns and volatility; its encoder gives each stock an exposure vector, from which
a factor risk model is estimated. Portfolios maximize the Shannon entropy of risk contributions in the principal-factor
basis (factor diversification) under turnover and concentration constraints. A walk-forward protocol compares the
strategy with six benchmarks.

**Goal.** Diversify across factors and show a measurable gain of the non-linear VAE over linear alternatives (PCA) and
classical allocations (1/N, inverse volatility, minimum variance, ERC).

**Status (2026-11-07).** The goal is not met: on the last real run the VAE portfolio had a Sharpe ratio of 0.38 versus
0.49 for 1/N, with realized variance 2.74 times the model forecast and the first principal factor carrying 81% of the
risk. The re-audit `docs/audit_v5/` traces this to seven shared root causes and proposes a redesign in which the VAE
becomes one candidate risk model under a kill criterion. Nothing of the redesign is implemented yet.

**Users.** Quantitative researchers and portfolio managers evaluating non-linear factor models for risk budgeting.

**Normative documents.** `docs/DVT_strategie_facteurs_risque_latents_v4_1.md` (vision, French),
`docs/ISD_vae_latent_risk_factors.md` (implementation spec with invariants and conventions),
`docs/Benchmark_latent_risk_factor.md` (benchmarks and tests), `docs/implementation_decisions.md` and
`docs/divergences.md` (where the code departs from the spec). The other files in `docs/` are dated analyses.

## Pipelines

### Data
- **Goal:** a point-in-time panel of US equities in the pipeline schema.
- **Main files:** `scripts/download_tiingo.py`, `src/data_pipeline/` (`data_loader.py`, `returns.py`,
  `universe.py`, `windowing.py`, `features.py`, `crisis.py`).
- **Input:** Tiingo API (download on explicit request only). **Output:** `data/tiingo_us_equities.parquet`, windows.

### VAE training and inference
- **Goal:** learn latent risk factors and the exposure matrix B.
- **Main files:** `src/vae/` (model, encoder, decoder, loss), `src/training/`, `src/inference/` (composite profiles,
  active units).
- **Input:** z-scored windows (T=504 days, F=2 features). **Output:** trained encoder, B, active units.

### Risk model and portfolio
- **Goal:** turn exposures into a covariance forecast and an entropy-maximizing portfolio.
- **Main files:** `src/risk_model/` (rescaling, cross-sectional regression, covariance, conditioning),
  `src/portfolio/` (entropy, SCA solver, constraints, cardinality, frontier).
- **Input:** B, returns, trailing volatilities. **Output:** factor risk model, weights.

### Walk-forward evaluation and benchmarks
- **Goal:** out-of-sample comparison with 1/N, inverse volatility, minimum variance, ERC, PCA factor risk parity and
  PCA + volatility under the same constraints, with statistical tests.
- **Main files:** `src/walk_forward/`, `src/benchmarks/`, `src/integration/pipeline.py`,
  `src/integration/statistical_tests.py`, `scripts/run_walk_forward.py`, `scripts/run_benchmarks.py`.
- **Output:** per-fold metrics, tests, report.

### Diagnostics
- **Goal:** explain where the pipeline loses value (factor quality, solver, risk model, benchmarks) on one run.
- **Main files:** `scripts/run_diagnostic.py`, `src/diagnostics/`, `src/integration/diagnostic*.py`,
  `notebooks/dashboard.ipynb`, `notebooks/diagnostic_experiments.ipynb`.
- **Output:** `results/diagnostic_runs/<timestamp>/` (report Markdown/JSON, plots, checkpoints).

## Platforms

Python 3.11+, PyTorch ≥ 2.1 (MPS on Apple Silicon, CUDA on Colab), CVXPY with MOSEK (ECOS fallback). Runs locally and
on Google Colab (Drive persistence in `src/integration/colab_drive.py`).
