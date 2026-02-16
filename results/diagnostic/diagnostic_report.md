# VAE Latent Risk Factor — Diagnostic Report

---

## Executive Summary

**Overall Status: ISSUES DETECTED** — 1 critical, 2 warnings, 7 OK

### Health Checks

| Status | Category | Check | Details |
|--------|----------|-------|---------|
| [OK] | Latent | Active units | AU = 26 / K = 52 (50.0% utilization) |
| [CRIT] | Risk Model | Variance ratio | var_ratio = 2.742 — outside [0.5, 2.0] |
| [OK] | Risk Model | Condition number | cond(Sigma) = 2e+04 |
| [OK] | Risk Model | Explanatory power | EP_oos = 0.3563, EP_is = 0.2190 |
| [OK] | Portfolio | Sharpe ratio | Sharpe = 0.382 |
| [OK] | Portfolio | Factor entropy | H_norm_signal = 0.399 (n_signal=12), ENB = 2.70, H_norm_eff = 0.566 (n_eff=5.8) |
| [WARN] | Portfolio | Max drawdown | MDD = 35.3% (EW benchmark: 37.8%) |
| [OK] | Portfolio | Selection non-trivial | Mean rank = 272/639 (ratio=0.85), first quintile = 31% of active |
| [WARN] | Data | Missing data | 35.3% missing values |
| [OK] | Data | Universe size | 788 stocks |

## 1. Data Quality

- **Universe size**: 788 stocks
- **Date range**: 1995-01-03 to 2026-02-13
- **Trading days**: 7874
- **Years of data**: 31.2
- **Missing data**: 35.31%
- **Stocks > 20% missing**: 410

## 2. Training Convergence

*Training diagnostics not available: pretrained model (no training)*

## 3. Latent Space Analysis

- **K (latent capacity)**: 52
- **AU (active units)**: 26
- **Utilization ratio**: 50.0%
- **Effective latent dims**: 40.1
- **KL total**: 81.7802
- **KL top-3 fraction**: 9.6%

### Exposure Matrix B

- **Shape**: [639, 52]
- **Sparsity**: 0.0%
- **Max absolute entry**: 2.1580
- **Mean dim norm**: 4.2069
- **Mean stock norm**: 1.5622

## 4. Risk Model Quality

- **Variance targeting**: sys=1.0916, idio=0.5270
- **Variance ratio (OOS)**: 2.7422 (target: [0.5, 2.0])
- **Rank correlation (OOS)**: 0.3477
- **Explanatory power (OOS)**: 0.3563
- **Explanatory power (IS)**: 0.2190
- **Avg cross-sectional R² (OOS)**: 0.1300

### Exposure Matrix (B_A) Scale

- **Mean |B_A|**: 0.7281
- **Std B_A**: 0.9992
- **Max |B_A|**: 16.2257
- **Column norm (mean)**: 25.2587
- **Column norm (max)**: 25.2587
- **Condition number**: 2.40e+04

### Eigenvalue Spectrum

- **Number of eigenvalues**: 27
- **Top eigenvalue**: 5.83e-05
- **Top 3 explained**: 72.9%
- **Top 10 explained**: 84.7%
- **Ratio #1/#2**: 10.08

## 5. Portfolio Optimization

- **Alpha (risk aversion)**: 1.00e-03
- **Active positions**: 80 / 639
- **Effective N**: 50.5
- **HHI**: 0.0198
- **Gini coefficient**: 0.4164
- **Max weight**: 0.0300
- **Min active weight**: 2.49e-03

### Factor Risk Decomposition

- **Top 1 factor contribution**: 81.1%
- **Top 3 factor contribution**: 84.2%
- **Risk entropy (H)**: 0.9924
- **Max possible entropy**: 3.2958

## 6. Out-of-Sample Performance

### VAE Portfolio

- **Annualized return**: 6.26%
- **Annualized volatility**: 16.40%
- **Sharpe ratio**: 0.382
- **Sortino ratio**: 0.419
- **Calmar ratio**: 0.177
- **Max drawdown**: 35.34% (EW benchmark: 37.80%)
- **Normalized entropy (H_norm)**: 0.5337
- **H_norm_signal (vs n_signal)**: 0.3994 (n_signal = 12, ENB = 2.70)
- **H_norm_eff (vs effective dims)**: 0.5657 (n_eff = 5.8)

### VAE vs Benchmarks

| Metric | VAE | Equal Weight | Inverse Vol | Min Variance | Erc | Pca Factor Rp | Pca Vol | Sp500 Index |
|--------|-----|------|------|------|------|------|------|------|
| sharpe | 0.3816 | 0.4916 | 0.5018 | 0.1996 | 0.4916 | 0.3016 | 0.3678 | 0.6605 |
| ann_return | 0.0626 | 0.0968 | 0.0958 | 0.0284 | 0.0968 | 0.0455 | 0.0550 | 0.1363 |
| ann_vol_oos | 0.1640 | 0.1970 | 0.1909 | 0.1425 | 0.1970 | 0.1508 | 0.1495 | 0.2063 |
| max_drawdown_oos | 0.3534 | 0.3780 | 0.3745 | 0.2634 | 0.3780 | 0.2727 | 0.2850 | 0.3392 |
| H_norm_oos | 0.5337 | nan | nan | nan | nan | nan | nan | nan |
| eff_n_positions | 50.4733 | 477.0000 | 420.4286 | 41.7076 | 477.0000 | 41.2262 | 40.9714 | 500.0000 |

### Win/Loss Summary

| Benchmark | VAE Wins | VAE Losses |
|-----------|----------|------------|
| equal_weight | 2 | 4 |
| inverse_vol | 2 | 4 |
| min_variance | 4 | 2 |
| erc | 2 | 4 |
| pca_factor_rp | 4 | 2 |
| pca_vol | 3 | 3 |
| sp500_index | 1 | 5 |

**Total: 18 wins, 24 losses**

## 7. Diagnosis & Recommendations

- **Covariance underestimation** (var_ratio = 2.742): the model predicts less risk than observed. The risk model may be poorly calibrated.

## Appendix: Configuration

```json
{
  "data": {
    "n_stocks": 2000,
    "window_length": 504,
    "n_features": 2,
    "vol_window": 252,
    "vix_lookback_percentile": 80.0,
    "min_valid_fraction": 0.8,
    "data_source": "synthetic",
    "data_dir": "data/",
    "training_stride": 21
  },
  "vae": {
    "K": 75,
    "sigma_sq_init": 1.0,
    "sigma_sq_min": 0.0001,
    "sigma_sq_max": 10.0,
    "window_length": 504,
    "n_features": 2,
    "r_max": 5.0,
    "dropout": 0.3
  },
  "loss": {
    "mode": "P",
    "gamma": 3.0,
    "lambda_co_max": 0.5,
    "beta_fixed": 1.0,
    "warmup_fraction": 0.2,
    "max_pairs": 2048,
    "delta_sync": 21
  },
  "training": {
    "max_epochs": 500,
    "batch_size": 512,
    "learning_rate": 0.005,
    "weight_decay": 0.001,
    "adam_betas": [
      0.9,
      0.999
    ],
    "adam_eps": 1e-08,
    "patience": 50,
    "es_min_delta": 1,
    "lr_patience": 30,
    "lr_factor": 0.75,
    "n_strata": 15,
    "curriculum_phase1_frac": 0.3,
    "curriculum_phase2_frac": 0.3,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": false,
    "compile_model": true
  },
  "inference": {
    "batch_size": 512,
    "au_threshold": 0.01,
    "r_min": 2,
    "aggregation_method": "mean",
    "aggregation_half_life": 60
  },
  "risk_model": {
    "winsorize_lo": 5.0,
    "winsorize_hi": 95.0,
    "d_eps_floor": 1e-06,
    "conditioning_threshold": 1000000.0,
    "ridge_scale": 1e-06,
    "sigma_z_shrinkage": "spiked",
    "sigma_z_eigenvalue_pct": 0.95,
    "sigma_z_ewma_half_life": 252,
    "b_a_shrinkage_alpha": 0.0,
    "use_wls": true,
    "b_a_normalize": true,
    "market_intercept": true
  },
  "portfolio": {
    "lambda_risk": 252.0,
    "w_max": 0.03,
    "w_min": 0.001,
    "w_bar": 0.03,
    "phi": 0.0,
    "kappa_1": 0.1,
    "kappa_2": 7.5,
    "delta_bar": 0.01,
    "tau_max": 0.3,
    "n_starts": 3,
    "sca_max_iter": 100,
    "sca_tol": 1e-08,
    "armijo_c": 0.0001,
    "armijo_rho": 0.5,
    "armijo_max_iter": 20,
    "max_cardinality_elim": 100,
    "entropy_eps": 1e-30,
    "cardinality_method": "auto",
    "alpha_grid": [
      0,
      0.001,
      0.005,
      0.01,
      0.02,
      0.05,
      0.1,
      0.2,
      0.5,
      1.0,
      2.0,
      5.0
    ],
    "momentum_enabled": false,
    "momentum_lookback": 252,
    "momentum_skip": 21,
    "momentum_weight": 0.5,
    "entropy_idio_weight": 0.05,
    "target_enb": 0.0,
    "transaction_cost_bps": 10.0
  },
  "walk_forward": {
    "total_years": 35,
    "min_training_years": 10,
    "oos_months": 6,
    "embargo_days": 21,
    "holdout_years": 3,
    "val_years": 2,
    "score_lambda_pen": 5.0,
    "score_lambda_est": 2.0,
    "score_mdd_threshold": 0.2
  },
  "seed": 42,
  "_diagnostic": {
    "profile": "full",
    "data_source": "tiingo",
    "n_stocks_actual": 788,
    "n_dates_actual": 7874,
    "date_range": "1995-01-03 to 2026-02-13",
    "pipeline_time_seconds": 1509.201466507,
    "holdout_fraction": 0.2,
    "loss_mode": "P"
  }
}
```
