# Project Overview

> **Note:** This file describes project goals by pipeline (WHAT and WHY, not HOW). See `datasets.md` for dataset details and `structure.md` for file organization.

## Overview

End-to-end portfolio construction pipeline based on latent risk factor discovery using a Variational Autoencoder (VAE). The system trains a 1D-CNN VAE on financial time series to uncover non-linear latent risk factors, then builds portfolios that maximize Shannon entropy of factor risk contributions (factor diversification). Walk-forward validation over 30 years of CRSP history with 6 benchmarks provides rigorous out-of-sample evaluation.

**Primary Goal:** Maximize factor diversification via entropy on principal factor risk contributions, while demonstrating measurable added value of the VAE over linear alternatives (PCA) and classical portfolio methods (1/N, min-var, ERC).

**Model/Technology:** 1D-CNN Variational Autoencoder (encoder-decoder), PyTorch. K=200 latent capacity with automatic pruning (AU active units). Three loss modes (P/F/A) with crisis weighting and co-movement curriculum.

**Target Users:** Quantitative portfolio managers and researchers evaluating non-linear factor models for equity risk budgeting.

## Pipelines

The project contains multiple pipelines that can use the same files for different purposes.

---

### Pipeline: VAE Training & Inference

**Goal:** Train the VAE on historical equity returns to discover latent risk factors, extract composite risk profiles for each stock.

**Main files:**
- `src/vae/model.py` — VAEModel (forward, encode, reparameterize)
- `src/vae/loss.py` — Loss computation (3 modes, crisis weighting, co-movement)
- `src/training/trainer.py` — Training loop with curriculum batching
- `src/inference/composite.py` — Sliding inference and aggregation to exposure matrix B

**Input:** Z-scored windows (N, T=504, F=2) from data pipeline, crisis labels

**Output:** Trained encoder, exposure matrix B (n x K), AU active dimensions

---

### Pipeline: Risk Model & Portfolio Optimization

**Goal:** Transform latent exposures into a full factor risk model and construct optimal portfolios maximizing factor entropy.

**Main files:**
- `src/risk_model/rescaling.py` — Dual rescaling (estimation vs portfolio)
- `src/risk_model/covariance.py` — Sigma_z (Ledoit-Wolf), D_eps, Sigma_assets
- `src/portfolio/entropy.py` — H(w) and gradient in principal factor basis
- `src/portfolio/sca_solver.py` — SCA + Armijo + multi-start optimization

**Input:** Exposure matrix B_A, trailing volatilities, historical returns

**Output:** Optimal weights w* (n,), factor risk model components

---

### Pipeline: Walk-Forward Validation

**Goal:** Validate the full pipeline out-of-sample across ~34 folds (30 years), compare against 6 benchmarks with statistical tests.

**Main files:**
- `src/walk_forward/folds.py` — Fold scheduling
- `src/walk_forward/phase_a.py` — Hyperparameter selection
- `src/walk_forward/phase_b.py` — Deployment run
- `src/integration/pipeline.py` — E2E orchestrator
- `scripts/run_walk_forward.py` — CLI entry point

**Input:** Full CRSP dataset, pipeline configuration

**Output:** Per-fold metrics (3 layers), statistical test results, final report

**Commands:**
```bash
python scripts/run_walk_forward.py
python scripts/run_benchmarks.py
```

---

### Pipeline: Benchmarks

**Goal:** Provide 6 reference strategies under identical constraints for fair comparison.

**Main files:**
- `src/benchmarks/base.py` — Abstract benchmark class
- `src/benchmarks/equal_weight.py` — 1/N
- `src/benchmarks/inverse_vol.py` — Inverse volatility
- `src/benchmarks/min_variance.py` — Minimum variance (Ledoit-Wolf)
- `src/benchmarks/erc.py` — Equal Risk Contribution (Spinu)
- `src/benchmarks/pca_factor_rp.py` — PCA factor risk parity (Bai-Ng IC2 + SCA)
- `src/benchmarks/pca_vol.py` — PCA + realized vol variant

**Input:** Returns, universe, trailing volatilities (same as VAE)

**Output:** Benchmark weights and OOS metrics under identical constraints

---

## Supported Languages/Platforms

| Language/Platform | Version | Status |
|-------------------|---------|--------|
| Python | 3.11+ | Active |
| PyTorch | >= 2.1 | Active |
| CVXPY + MOSEK | >= 1.4 | Active (ECOS fallback) |

---

## Target Use Case

| Task | Description | Data Type |
|------|-------------|-----------|
| Factor discovery | Uncover non-linear latent risk factors from equity returns | Time series (N x T x F) |
| Risk budgeting | Build portfolios with maximum factor entropy | Covariance matrices, exposures |
| Backtesting | Walk-forward validation with embargo, 6 benchmarks | 30 years CRSP history |

---

## Update History

| Date | Section | Change |
|------|---------|--------|
| 2026-02-06 | Initial | Created from ISD specification |
