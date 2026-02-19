# Guide d'Interprétation des Diagnostics

This guide provides interpretation for the composite diagnostic scores in the VAE Latent Risk Factor pipeline.

---

## 1. Strategy Overview

**Objective**: Maximize Shannon entropy of principal factor risk contributions (factor diversification), subject to portfolio constraints.

**Key Parameters** (from `PortfolioConfig`):
- `K=200`: Latent capacity ceiling
- `w_max=0.05`: Maximum weight per position (hard cap)
- `w_min=0.001`: Minimum active weight (semi-continuous)
- `tau_max=0.30`: Maximum one-way turnover
- `lambda_risk=252`: Risk aversion coefficient (annualized)

**Expected Behavior**:
- Solver converges in <50 iterations with gradient norm < 1e-5
- Constraint binding: 10-20% positions at w_max is normal
- Variance ratio (realized/predicted): target [0.8, 1.2], acceptable [0.5, 2.0]
- Shrinkage intensity: 0.15-0.65 indicates adequate sample size

---

## 2. Composite Scores (0-100)

All scores follow a unified grading scale:

| Score | Grade | Status | Meaning |
|-------|-------|--------|---------|
| 90-100 | A | Excellent | No action needed |
| 75-89 | B | Good | Minor optimization opportunities |
| 60-74 | C | Acceptable | Review recommended |
| 40-59 | D | Marginal | Action required |
| 0-39 | F | Critical | Immediate investigation needed |

---

### 2.1 Solver Health Score

**What it measures**: Quality of SCA (Successive Convex Approximation) optimizer convergence.

**Formula**:
```
grad_score = 50 × (1 - clip(log10(grad_norm) + 10, 0, 8) / 8)
conv_score = 30 × converged_ratio
iter_score = 20 × (1 - iterations/max_iter) if not saturated, else 5

solver_health = grad_score + conv_score + iter_score
```

**Interpretation**:
- **90-100 (A)**: Optimum reached with high confidence. Gradient norm < 1e-7.
- **75-89 (B)**: Reliable solution. Gradient norm < 1e-4.
- **60-74 (C)**: Acceptable but verify constraints aren't causing difficulty.
- **<60 (D/F)**: Optimization likely failed. Check max_iter, step size, or objective conditioning.

**Diagnostic signals**:
- `best_final_grad_norm > 1e-3`: Solver did not converge
- `converged_ratio < 0.5`: Many restarts failed (objective may have local optima)
- `iterations = max_iter`: Hit iteration limit (increase max_iter or loosen tolerance)

---

### 2.2 Constraint Pressure Score

**What it measures**: How much the constraints are restricting the optimizer's freedom.

**Formula**:
```
wmax_score = 40 × (1 - binding_fraction)^0.5
wmin_score = 20 × (1 - min(n_at_wmin/n_active × 2, 1))
turn_score = 25 × (1 - actual_turnover/tau_max × 0.8) if not binding
conc_score = 15 × (eff_n / n_active)

constraint_pressure = wmax_score + wmin_score + turn_score + conc_score
```

**Interpretation**:
- **>80**: Optimizer is free — solution is in the interior
- **65-80**: Normal pressure — some constraints are active
- **50-64**: Moderate pressure — consider relaxing constraints
- **<50**: Strong pressure — constraints may be too restrictive

**What high/low means**:
- **High score** (>80): The optimizer found a solution without hitting constraint boundaries. This is ideal — the entropy-optimal portfolio is naturally diversified.
- **Low score** (<50): Many positions are at the w_max cap. The optimizer "wants" to concentrate but is constrained. This could indicate:
  - w_max is too restrictive
  - Factor model has poor diversification opportunities
  - Risk estimates favor a few stocks

---

### 2.3 Covariance Quality Score

**What it measures**: Quality of the factor risk model calibration.

**Formula**:
```
cond_score = 30 × (1 - clip(log10(cond_num) - 2, 0, 8) / 8)
var_score = 35 × f(|var_ratio - 1|)  # penalty outside [0.8, 1.2]
ep_score = 25 × min(explanatory_power × 5, 1)
shrink_score = 10 if shrinkage ∈ [0.15, 0.65], else decreasing

covariance_quality = cond_score + var_score + ep_score + shrink_score
```

**Interpretation**:
- **>80**: Risk model is well-calibrated
- **60-80**: Acceptable calibration, monitor var_ratio
- **<60**: Calibration problem — investigate data or shrinkage

**Key metrics**:
- `var_ratio` (realized/predicted variance): Target 1.0
  - < 0.5: Model overestimates risk (too conservative)
  - > 2.0: Model underestimates risk (dangerous)
- `shrinkage_intensity`:
  - 0.15-0.35: Adequate data, minimal regularization
  - 0.35-0.65: Normal range
  - > 0.70: Insufficient data — consider more history
- `condition_number`:
  - < 1e4: Well-conditioned
  - 1e4-1e6: Acceptable
  - > 1e6: Ill-conditioned — numerical instability risk

---

### 2.4 Reconstruction Balance Score

**What it measures**: How well the VAE learns both features (returns and volatility).

**Formula**:
```
recon_score = 40 × (1 - clip(best_recon × 5, 0, 1))
balance_score = 35 if ratio ∈ [0.8, 2.5], else decreasing
stability_score = 25 if overfit_ratio ∈ [0.9, 1.15], else decreasing

reconstruction_balance = recon_score + balance_score + stability_score
```

**Interpretation**:
- **>80**: VAE learns both features well
- **60-80**: Acceptable learning
- **<60**: Imbalance — one feature dominates

**Expected behavior**:
- Returns reconstruction is typically 1.5-2.5× harder than volatility (returns are noisier)
- `overfit_ratio` (train_loss / val_loss) should be 0.9-1.15:
  - < 0.85: Underfitting (increase capacity or epochs)
  - > 1.3: Overfitting (increase dropout or reduce K)

---

### 2.5 Overall Score

**Formula**:
```
overall = 0.25 × solver + 0.15 × constraint + 0.35 × covariance + 0.25 × reconstruction
```

Weights reflect importance:
- **35% Covariance**: Risk model accuracy is critical for portfolio quality
- **25% Solver**: Must find the actual optimum
- **25% Reconstruction**: VAE must learn meaningful factors
- **15% Constraints**: Lower weight — some constraint pressure is acceptable

**Missing component penalty**: -2.5 points per unavailable component.

---

## 3. Diagnostic Scenarios

### Scenario A: Overall Score > 80
**Status**: Pipeline healthy, performance as expected.

**Action**: None required. Monitor for degradation across folds.

---

### Scenario B: Solver Low + Constraints High
**Symptom**: Solver score < 60, Constraint score > 80

**Diagnosis**: Pure optimization problem, not constraint-related.

**Possible causes**:
- Objective function ill-conditioned (check eigenvalue spectrum)
- Step size too aggressive (reduce initial step)
- Tolerance too tight (increase sca_tol)

**Actions**:
1. Increase `sca_max_iter` from 100 to 200
2. Reduce `armijo_rho` from 0.5 to 0.3
3. Check `condition_number` of Sigma_assets

---

### Scenario C: Solver OK + Constraints Low
**Symptom**: Solver score > 75, Constraint score < 50

**Diagnosis**: Constraints are too restrictive for the factor structure.

**Possible causes**:
- w_max too low for the number of attractive stocks
- Factor model identifies few diversification opportunities
- Market regime favors concentration

**Actions**:
1. Increase `w_max` from 0.05 to 0.07-0.10
2. Review factor quality (AU vs Bai-Ng)
3. Check if this pattern is consistent across folds (structural) or transient

---

### Scenario D: Covariance Score Low
**Symptom**: Covariance score < 60

**Diagnosis**: Risk model miscalibration.

**Sub-scenarios**:

**D1: var_ratio < 0.5 (overestimation)**
- Factor model predicts more risk than realized
- Actions: Check variance targeting, reduce shrinkage

**D2: var_ratio > 2.0 (underestimation)**
- Factor model predicts less risk than realized — dangerous
- Actions: Increase shrinkage, add more history, check for data errors

**D3: shrinkage > 0.70**
- Insufficient data for reliable covariance estimation
- Actions: Extend training window, reduce factor count

**D4: condition_number > 1e6**
- Near-singular covariance matrix
- Actions: Increase ridge regularization, reduce AU

---

### Scenario E: Reconstruction Imbalance
**Symptom**: Reconstruction score < 60, ratio outside [0.8, 2.5]

**Diagnosis**: VAE not learning both features equally.

**If ratio < 0.8** (volatility harder):
- Volatility patterns may have higher-frequency structure
- Consider increasing kernel sizes

**If ratio > 2.5** (returns much harder):
- Returns may be too noisy relative to signal
- Consider increasing window length or filtering outliers

---

## 4. Reference Values by Configuration

| Parameter | Default | Impact on Scores |
|-----------|---------|------------------|
| `K=200, AU~50-80` | Standard | Solver converges in <50 iter, EP>0.05 |
| `w_max=0.05` | Restrictive | 10-20% at cap is normal |
| `w_max=0.10` | Relaxed | <5% at cap expected |
| `tau_max=0.30` | Moderate | Binding only during major reallocation |
| `tau_max=0.15` | Restrictive | May bind frequently |
| `sca_max_iter=100` | Default | Sufficient for most cases |
| `sca_max_iter=200` | Conservative | For difficult landscapes |

---

## 5. Cross-Fold Comparison

When running walk-forward validation, compare scores across folds:

### Stability Indicators
- **Score std < 10**: Stable strategy — consistent behavior
- **Score std 10-20**: Moderate variation — investigate outliers
- **Score std > 20**: Unstable — significant regime dependency

### Trend Analysis
- **Stable**: Scores fluctuate but no trend
- **Improving**: Scores increase over time (more data helps)
- **Degrading**: Scores decrease over time — possible structural drift

### Anomaly Detection
Folds with scores > 2 standard deviations from mean warrant investigation:
- Check market regime (crisis period?)
- Verify data quality (missing data spike?)
- Review factor stability (AU changed significantly?)

---

## 6. Quick Reference Card

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| `solver_health` | >80 | 60-75 | <60 |
| `constraint_pressure` | >65 | 50-65 | <50 |
| `covariance_quality` | >75 | 60-75 | <60 |
| `reconstruction_balance` | >70 | 55-70 | <55 |
| `overall` | >75 | 60-75 | <60 |
| `grad_norm` | <1e-5 | 1e-5 to 1e-3 | >1e-3 |
| `var_ratio` | 0.8-1.2 | 0.5-2.0 | outside |
| `shrinkage` | 0.15-0.65 | 0.65-0.75 | >0.75 |
| `binding_fraction` | <0.20 | 0.20-0.50 | >0.50 |

---

## 7. Changelog

| Date | Change |
|------|--------|
| 2026-02-19 | Initial version |
