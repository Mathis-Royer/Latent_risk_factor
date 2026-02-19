# Justification: sigma_z_ewma_half_life = 0

## Summary

The parameter `sigma_z_ewma_half_life` controls temporal weighting for estimating the factor covariance matrix (Sigma_z). Setting it to **0 (equal weights)** rather than **126 (Barra USE4 standard)** is strongly aligned with the strategy's anti-cyclical philosophy.

**Configuration change:** `sigma_z_ewma_half_life: 126 → 0` in `src/config.py:451`

---

## Strategic Alignment

### Principle 4 — Anti-Cyclical Memory (strategy_philosophy.md:44-58)

> "This is the **anti-cyclical principle**: the training window is **expanding** (not rolling), the composite profile is a **mean over all windows**, and the factor covariance is estimated on **full history**. **Crisis correlations persist permanently** in the model."

**Direct implication:** Sigma_z must use equal weights (half-life=0), not EWMA which "forgets" historical correlations.

### DVT v4.1 — Iteration Hierarchy

| Document | Section | Recommendation |
|----------|---------|----------------|
| DVT §6.2 | Iteration 1 — EWMA aggregation | **Optional** iteration, not baseline. Half-life: **3-10 years** (756-2520 days), not 6 months |
| DVT §6.3 | Iteration 2 — Regime-aware blending | **Optional** with diagnostic trigger. Uses blended estimator, not pure EWMA |
| DVT §6.1 | Baseline | "Full-history" estimation = equal weights |

**Key citation (DVT:1109):**
> "The simple mean gives equal weight to all historical windows"

The DVT presents EWMA as an **optional improvement** when the following diagnostic triggers:
> "latent stability metric (Spearman ρ on inter-stock distances) drops below 0.80 between consecutive retrainings"

---

## Barra USE4 vs. Anti-Cyclical Strategy

| Criterion | Barra USE4 (half-life=126) | Strategy (half-life=0) |
|-----------|---------------------------|-------------------------|
| **Objective** | Track current regimes | Preserve crisis memory |
| **Horizon** | Short-term (risk management) | Long-term (structural diversification) |
| **Crisis correlations** | Forgotten after ~3 years | Preserved **permanently** |
| **Dormant factors** | Disappear | Remain in model |

---

## Academic Literature

### 1. DeMiguel et al. (2009) — Estimation Error

> "The estimation window needed for mean-variance to outperform 1/N is ~3000 months for 25 assets"

**Implication:** With limited samples, reducing N_eff via EWMA **amplifies** estimation error. Equal weights maximize N_eff.

### 2. Ledoit & Wolf (2004, 2020) — Covariance Shrinkage

Ledoit-Wolf estimators are designed for sample covariances with **equal weights**. EWMA introduces complexity not modeled by these shrinkages.

### 3. Meucci (2010) — Managing Diversification

The "Effective Number of Bets" (ENB) is maximized when all factors contribute equally to risk. This requires **stable** Sigma_z estimation, not reactive estimation.

### 4. Tail Risk Diversification Principle

> "A system that diversifies only against **currently observable** correlations is structurally blind to episodic risks." (strategy_philosophy.md:51-52)

EWMA with 126 days makes the system blind to crisis correlations observed >3 years ago.

---

## Technical Impact

With `sigma_z_ewma_half_life=0`:

1. **K adaptation:** `compute_au_max_stat()` uses N_eff = n_obs (not reduced)
   - Result: Higher AU_max_stat → more latent capacity
   - Previous fix (changelog #7): "K 38→150, AU_max_stat 19→79"

2. **Sigma_z estimation:** `estimate_sigma_z()` uses equal weights
   - All historical correlations contribute equally
   - Consistent with "crisis correlations persist permanently"

3. **Coherence:** Both uses (K adaptation and Sigma_z) now use the same config value

---

## When to Activate EWMA (Future)

Only activate EWMA when the diagnostic trigger is observed:
- "latent stability metric drops below 0.80"

If activated:
- Use half-life of **3-10 years** (756-2520 days), not 6 months
- Consider the blended estimator (DVT §6.3) rather than pure EWMA
- Document the trigger condition in changelog

---

## Files Modified

| File | Line | Change |
|------|------|--------|
| `src/config.py` | 451 | `sigma_z_ewma_half_life: int = 126` → `0` |
| `src/integration/pipeline.py` | 289 | `ewma_half_life=0` → `self.config.risk_model.sigma_z_ewma_half_life` |

---

## Conclusion

Setting `sigma_z_ewma_half_life=0` is **strongly aligned** with the strategy:

1. **Anti-cyclical:** Preserves crisis correlations permanently
2. **Expanding window:** Uses full history with equal weights
3. **DVT baseline:** EWMA is presented as optional iteration, not baseline
4. **Dormant factors:** Model retains memory of episodic risks
5. **Stable estimation:** Maximizes N_eff, minimizes estimation error

---

## References

- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal versus naive diversification: How inefficient is the 1/N portfolio strategy? *Review of Financial Studies*, 22(5), 1915-1953.
- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365-411.
- Ledoit, O., & Wolf, M. (2020). Analytical nonlinear shrinkage of large-dimensional covariance matrices. *Annals of Statistics*, 48(5), 3043-3065.
- Meucci, A. (2010). Managing diversification. *Risk*, 22(5), 74-79.
- Roncalli, T. (2013). *Introduction to Risk Parity and Budgeting*. Chapman & Hall/CRC.
