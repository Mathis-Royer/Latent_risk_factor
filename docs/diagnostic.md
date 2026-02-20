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

**Formula** (10 components):
```
overall = 0.20 × covariance + 0.15 × solver + 0.12 × reconstruction
        + 0.10 × vae_health + 0.10 × factor_model + 0.08 × constraint
        + 0.08 × training_convergence + 0.07 × active_unit
        + 0.05 × portfolio_diversification + 0.05 × factor_stability
```

Weights reflect importance:
- **20% Covariance**: Risk model accuracy is critical for portfolio quality
- **15% Solver**: Must find the actual optimum
- **12% Reconstruction**: VAE must learn meaningful factors
- **10% VAE Health**: Posterior quality matters for stable factors
- **10% Factor Model**: Factor model specification quality
- **8% Constraints**: Some pressure is acceptable
- **8% Training Convergence**: VAE training quality
- **7% Active Units**: Latent space utilization
- **5% Portfolio Diversification**: Diversification quality check
- **5% Factor Stability**: Factor temporal stability

**Missing component penalty**: -2.5 points per unavailable component.

---

### 2.6 Training Convergence Score

**What it measures**: Quality of VAE training convergence.

**Formula**:
```
timing_score = 30 × f(best_epoch_fraction)
  - [0.3, 0.85] → 30 pts (optimal)
  - < 0.15 → 10 pts (too early = underfitting)
  - > 0.95 → 15 pts (at end = may still converge)

stability_score = 25 if not still_decreasing else 15
  - Penalty if val ELBO still decreasing at end

lr_score = 25 × f(n_lr_reductions)
  - 2-5 reductions → 25 pts (ReduceLROnPlateau working)
  - 0-1 → 15 pts (no scheduling needed)
  - > 10 → 10 pts (instability)

sigma_score = 20 - 10×(min_hit) - 10×(max_hit)
  - Penalty if sigma_sq hits bounds

training_convergence = timing + stability + lr + sigma
```

**Interpretation**:
- **>85 (A)**: Training converged optimally
- **70-85 (B)**: Acceptable convergence
- **55-70 (C)**: Minor issues — review training parameters
- **<55 (D/F)**: Convergence problems — action required

**Diagnostic signals**:
- `best_epoch_fraction < 0.15`: Model may be underfitting
- `still_decreasing_at_end = True`: Need more epochs
- `n_lr_reductions > 10`: Training unstable
- `sigma_sq_min/max_hit`: Observation noise at bounds

---

### 2.7 Active Unit Score

**What it measures**: Efficient utilization of latent space capacity.

**Formula**:
```
util_score = 35 × f(AU / K)
  - [0.15, 0.60] → 35 pts (optimal utilization)
  - < 0.05 → 5 pts (severe collapse)
  - > 0.80 → 20 pts (near saturation)

stability_score = 30 × f(au_final / au_max_during_training)
  - [0.8, 1.0] → 30 pts (stable AU)
  - < 0.5 → 10 pts (excessive pruning)

spectrum_score = 35 × f(eff_latent_dims / AU)
  - [0.5, 1.0] → 35 pts (balanced distribution)
  - < 0.3 → 15 pts (few dims dominate)

active_unit = util + stability + spectrum
```

**Interpretation**:
- **>80 (A)**: Latent space well-utilized
- **60-80 (B)**: Acceptable utilization
- **<60 (C/D/F)**: Issues with latent space

**Diagnostic signals**:
- `utilization_ratio < 0.05`: Posterior collapse
- `utilization_ratio > 0.80`: Consider increasing K
- `au_retention < 0.5`: AU dropped significantly during training
- `spectrum_ratio < 0.3`: Few dimensions dominate (mode collapse risk)

---

### 2.8 Portfolio Diversification Score

**What it measures**: Quality of factor diversification in the portfolio.

**Formula**:
```
entropy_score = 40 × min(H_norm_signal / 0.5, 1)
  - H_norm ≥ 0.5 → 40 pts (excellent)
  - H_norm < 0.3 → reduced proportionally

enb_score = 25 × min(ENB / n_signal, 1)
  - ENB close to n_signal = perfectly diversified

position_score = 25 × f(eff_n / n_active)
  - > 0.7 → 25 pts (balanced weights)
  - < 0.3 → 10 pts (concentration)

gini_score = 10 × (1 - gini)
  - Gini = 0 (equal) → 10 pts

portfolio_diversification = entropy + enb + position + gini
```

**Interpretation**:
- **>80 (A)**: Excellent factor diversification
- **60-80 (B)**: Acceptable diversification
- **<60 (C/D/F)**: Concentration risk

**Key metrics**:
- `H_norm_signal`: Normalized entropy vs n_signal factors
- `ENB`: Effective Number of Bets (exp of entropy)
- `eff_n_positions / n_active`: Weight balance ratio
- `gini_coefficient`: Inequality of position sizes

**Actions**:
- Low H_norm: Review alpha/entropy tradeoff in frontier
- Low position_ratio: Check if w_max is too restrictive
- High Gini: Consider tighter position limits

---

### 2.9 Factor Stability Score

**What it measures**: Temporal stability of factor structure across folds.

**Formula**:
```
stability_score = 50 × f(latent_stability_rho)
  - rho ≥ 0.85 → 50 pts (very stable)
  - rho < 0.70 → 20 pts (unstable)
  - N/A → 35 pts (neutral for single fold)

composition_score = 30 × f(pct_structural)
  - > 50% structural → 30 pts (persistent factors)
  - < 20% structural → 15 pts (episodic dominated)

consistency_score = 20 × f(|AU - k_bai_ng|)
  - < 5 diff → 20 pts (consistent)
  - > 20 diff → 5 pts (divergent)

factor_stability = stability + composition + consistency
```

**Interpretation**:
- **>80 (A)**: Factor structure robust across folds
- **60-80 (B)**: Acceptable stability
- **<60 (C/D/F)**: Factor instability risk

**Key metrics**:
- `latent_stability_rho`: Spearman correlation of inter-stock distances between B_A matrices across folds
- `pct_structural`: Percentage of factors classified as structural (half-life > 126 days)
- `au_bai_ng_diff`: Difference between VAE AU and Bai-Ng IC2 optimal k

**What high/low means**:
- **High stability** (rho > 0.85): Factor loadings are consistent — suitable for longer holding periods
- **Low stability** (rho < 0.70): Factor structure changes between folds — may degrade OOS
- **Low pct_structural**: Most factors are episodic — higher OOS risk

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

### Composite Scores

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| `solver_health` | >80 | 60-75 | <60 |
| `constraint_pressure` | >65 | 50-65 | <50 |
| `covariance_quality` | >75 | 60-75 | <60 |
| `reconstruction_balance` | >70 | 55-70 | <55 |
| `vae_health` | >75 | 55-75 | <55 |
| `factor_model` | >75 | 55-75 | <55 |
| `training_convergence` | >85 | 55-70 | <55 |
| `active_unit` | >80 | 60-80 | <60 |
| `portfolio_diversification` | >80 | 60-80 | <60 |
| `factor_stability` | >80 | 60-80 | <60 |
| `overall` | >75 | 60-75 | <60 |

### Raw Metrics

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| `grad_norm` | <1e-5 | 1e-5 to 1e-3 | >1e-3 |
| `var_ratio` | 0.8-1.2 | 0.5-2.0 | outside |
| `shrinkage` | 0.15-0.65 | 0.65-0.75 | >0.75 |
| `binding_fraction` | <0.20 | 0.20-0.50 | >0.50 |
| `best_epoch_fraction` | 0.30-0.85 | 0.15-0.30 | <0.15 or >0.95 |
| `utilization_ratio` | 0.15-0.60 | 0.05-0.15 or 0.60-0.80 | <0.05 or >0.80 |
| `H_norm_signal` | >0.50 | 0.30-0.50 | <0.30 |
| `latent_stability_rho` | >0.85 | 0.70-0.85 | <0.70 |

---

## 7. Advanced Diagnostics (New)

This section documents advanced diagnostic metrics introduced for deeper pipeline analysis.

### 7.1 Training KL Trajectory

**What it measures**: KL divergence evolution during VAE training, indicating latent space utilization patterns.

**Key Metrics**:
- `kl_trajectory`: Per-epoch KL divergence values
- `kl_slope`: Trend direction (positive = growing, negative = decreasing)
- `kl_final`: Final KL value at best epoch
- `kl_stability`: Coefficient of variation in later epochs

**Interpretation**:
- **Healthy trajectory**: KL increases gradually, then stabilizes (beta annealing working)
- **KL explosion** (slope > 0.5, final > 10): Posterior explosion risk — reduce beta or increase capacity
- **KL collapse** (final < 0.1 per dimension): Posterior collapse — reduce beta, slow annealing

**Thresholds**:
| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| kl_final / AU | 0.2-2.0 nats | 0.1-0.2 or 2-5 | <0.1 or >5 |
| kl_slope | <0.2 | 0.2-0.5 | >0.5 |
| kl_stability | <0.15 | 0.15-0.30 | >0.30 |

---

### 7.2 Eigenvector Rotation Stability

**What it measures**: Consistency of principal factor directions across estimation windows.

**Key Metrics**:
- `rotation_angle`: Average angle between corresponding eigenvectors (degrees)
- `subspace_stability`: Overlap of top-K principal subspaces
- `rotation_rate`: Rate of rotation per trading day

**Interpretation**:
- **Stable factors** (angle < 15°): Factor structure is persistent — suitable for longer holding periods
- **Moderate rotation** (15-30°): Normal dynamics — monthly rebalancing appropriate
- **High rotation** (> 30°): Factor structure unstable — consider more frequent updates or longer windows

**Actions**:
- High rotation + good performance: Factors capture regime changes (acceptable)
- High rotation + poor performance: Noise dominates signal — extend estimation window

---

### 7.3 AU Retention Tracking

**What it measures**: How many active units (AU) are retained across consecutive folds.

**Key Metrics**:
- `au_retention_rate`: Fraction of AUs present in both consecutive folds
- `au_churn`: Number of AUs entering/exiting per fold
- `au_core_set`: AUs consistently active across all folds

**Interpretation**:
- **High retention** (> 0.80): Stable latent structure — robust factor model
- **Moderate retention** (0.60-0.80): Some adaptation to regime changes
- **Low retention** (< 0.60): Unstable factors — may indicate overfitting or regime sensitivity

**Relationship to latent_stability_rho**:
- Both measure factor consistency, but AU retention is binary (active/inactive)
- `latent_stability_rho` measures continuous loading similarity

---

### 7.4 Deflated Sharpe Ratio (DSR)

**What it measures**: Sharpe ratio adjusted for multiple testing and overfitting risk.

**Formula** (Bailey & López de Prado, 2014):
```
DSR = SR × sqrt(1 - skew × SR / 3 + (kurt - 1) × SR² / 4)
     × (1 - 1 / (4 × N_trials))
```

**Key Metrics**:
- `raw_sharpe`: Unadjusted Sharpe ratio
- `deflated_sharpe`: Adjusted for trials and distribution
- `haircut`: DSR / raw_sharpe ratio
- `n_trials`: Number of alpha configurations tested

**Interpretation**:
- **haircut > 0.80**: Minimal overfitting risk
- **haircut 0.60-0.80**: Moderate adjustment — some overfitting present
- **haircut < 0.60**: Significant deflation — strategy may be overfit

**When DSR < 0.5 × raw_sharpe**:
- Too many hyperparameters tested
- Consider fixing more parameters a priori
- Increase out-of-sample test period

---

### 7.5 Reconstruction Temporal Structure

**What it measures**: Whether VAE reconstruction errors show temporal patterns.

**Key Metrics**:
- `recon_autocorr_lag1`: Lag-1 autocorrelation of reconstruction error
- `recon_seasonality`: Periodic patterns in error (monthly, quarterly)
- `recon_regime_dependency`: Error variation by VIX regime

**Interpretation**:
- **autocorr < 0.1**: Errors are white noise — good
- **autocorr > 0.3**: Temporal structure not captured — increase window or add recurrence
- **regime_dependency > 0.2**: Performance differs by regime — consider regime conditioning

---

### 7.6 Mutual Information I(X;Z)

**What it measures**: Information preserved in the latent representation.

**Formula** (MINE estimator or binned approximation):
```
I(X;Z) ≥ E[T(x,z)] - log(E[exp(T(x,z'))])
```

**Key Metrics**:
- `mi_estimate`: Estimated mutual information (nats)
- `mi_normalized`: I(X;Z) / H(X) — fraction of input information retained
- `compression_ratio`: H(X) / H(Z) — how much the latent compresses

**Interpretation**:
- **mi_normalized > 0.3**: Good information retention
- **mi_normalized 0.1-0.3**: Moderate compression — may lose signal
- **mi_normalized < 0.1**: Over-compression — increase K or reduce beta

---

### 7.7 DGJ Shrinkage Validation

**What it measures**: Quality of Donoho-Gavish-Johnstone nonlinear shrinkage.

**Key Metrics** (from `validate_dgj_recovery()`):
- `n_signal`: Number of eigenvalues above BBP threshold (signal count)
- `noise_cv`: Coefficient of variation in noise eigenvalues
- `bbp_threshold`: Baik-Ben Arous-Péché phase transition threshold
- `dgj_quality`: "good" (CV < 0.1), "fair" (< 0.2), "poor" (≥ 0.2)

**Formula**:
```
bbp_threshold = sigma² × (1 + √γ)²
where sigma² = median(eigenvalues), γ = n_samples / n_features
```

**Interpretation**:
- **noise_cv < 0.10**: DGJ correctly flattened noise bulk — excellent
- **noise_cv 0.10-0.20**: Partial noise separation — acceptable
- **noise_cv > 0.20**: Poor separation — consider linear shrinkage or more data

**When DGJ is ineffective**:
- gamma too unfavorable (< 0.1 or > 10)
- True eigenvalue structure doesn't match spiked model assumptions
- Consider switching to `sklearn.covariance.LedoitWolf`

---

### 7.8 PBO (Probability of Backtest Overfitting)

**What it measures**: Probability that strategy selection overfit to in-sample data.

**Formula** (Bailey et al., 2015):
```
PBO = Pr[rank_OOS(best_IS) > N/2]
```

**Key Metrics**:
- `pbo`: Probability of overfitting [0, 1]
- `is_oos_correlation`: Spearman correlation of IS/OOS ranks
- `best_is_oos_rank`: OOS rank of best in-sample strategy

**Interpretation**:
- **PBO < 0.25**: Low overfitting risk — selection is robust
- **PBO 0.25-0.50**: Moderate risk — consider reducing hyperparameter space
- **PBO > 0.50**: High risk — strategy may not generalize

**Actions for high PBO**:
1. Reduce number of tested configurations
2. Use longer out-of-sample periods
3. Apply combinatorial purged cross-validation

---

### 7.9 Shadow Prices (Constraint Economics)

**What it measures**: Economic value of relaxing each constraint.

**Source**: Dual values from CVXPY optimization in `sca_solver.py`.

**Key Metrics**:
- `budget`: Shadow price of sum(w) = 1 constraint
- `w_max_binding_prices`: Per-asset prices for upper bound (n values)
- `w_min_binding_prices`: Per-asset prices for lower bound (n values)
- `turnover`: Shadow price of turnover constraint

**Interpretation**:
- **High w_max shadow price** (> 0.01): Asset wants higher weight — consider increasing w_max
- **Many high w_max prices**: Widespread binding — constraint too tight
- **High turnover shadow price**: Rebalancing constrained — consider higher tau_max
- **budget ≠ 0**: Budget constraint active (always true for fully invested)

**Use cases**:
- Identify most constrained assets
- Quantify opportunity cost of constraints
- Guide constraint relaxation decisions

---

## 8. Decision Rules Engine

The diagnostic pipeline includes a decision rules engine for automated interpretation.

### 8.1 Rule Structure

Each rule has:
- `condition`: Score thresholds that trigger the rule
- `diagnosis`: What the pattern indicates
- `root_causes`: Possible underlying causes
- `actions`: Recommended interventions
- `confidence`: Rule reliability [0, 1]
- `severity`: critical | high | medium | low | none

### 8.2 Key Rules

| Rule ID | Condition | Diagnosis |
|---------|-----------|-----------|
| PURE_OPTIMIZATION | solver < 60, constraint > 80 | Optimization problem, not constraints |
| CONSTRAINT_DOMINATED | solver > 70, constraint < 50 | Constraints too restrictive |
| COVARIANCE_DEGRADATION | covariance < 55 | Risk model miscalibrated |
| VAE_COLLAPSE | recon < 50, vae_health < 50 | Posterior collapse |
| OVERALL_DEGRADATION | all scores < 60 | Systemic pipeline issue |

### 8.3 Causal Graph

The engine traces causal relationships:
```
AU → entropy → factor_diversification
shrinkage_intensity → var_ratio → risk_forecast
condition_number → solver_convergence → portfolio_optimality
```

Use `trace_causal_chain(metric, "upstream")` to identify root causes.

---

## 9. AI Integration

### 9.1 JSON Schema

All diagnostic outputs conform to:
```json
{
  "fold": "2025-01",
  "scores": {
    "solver_score": 85.2,
    "constraint_score": 72.1,
    "covariance_score": 78.5,
    "reconstruction_score": 81.3,
    "overall_score": 79.3
  },
  "interpretations": {
    "solver_score": "Good convergence (B grade)",
    "overall_score": "Pipeline healthy"
  },
  "actions": [
    "Monitor var_ratio trend",
    "Consider increasing w_max if concentration persists"
  ],
  "metadata": {
    "timestamp": "2025-01-15T14:30:00Z",
    "pipeline_version": "1.2.0"
  }
}
```

### 9.2 AI-Ready Output

For LLM integration, use `format_diagnosis_summary()`:

```text
Diagnostic Summary [OK]
========================================

Weakest Component: constraint (score: 72.1)

Diagnosis (1 rules matched):
  - Normal constraint pressure (95% confidence)

Priority Actions:
  1. Continue monitoring
  2. Review constraint settings if concentration increases
```

---

## 10. Changelog

| Date | Change |
|------|--------|
| 2026-02-20 | Added Sections 7-9: Advanced diagnostics, decision rules, AI integration |
| 2026-02-19 | Initial version |
