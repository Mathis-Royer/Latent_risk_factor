# Strategy Comparison: Uniform Entropy vs Sortino-Budgeted Factor Risk Parity

> **Date:** 2026-02-18
> **Context:** Evaluation of the current DVT portfolio construction approach and a proposed alternative that addresses its structural limitations.

---

## 1. Strategies Compared

### Strategy A — Uniform Entropy Maximization (Current DVT)

Maximize Shannon entropy of **factor risk contributions** with uniform budgets:

```
max  alpha * H(w)  -  lambda * w'Sigma*w  +  mu'w  -  phi * P_conc  -  P_turn
        |                   |                  |           |             |
   diversification     risk penalty      momentum tilt  concentration  turnover
   (primary obj.)                        (optional add)
```

Where `H(w) = -sum_k  p_k * ln(p_k)` with `p_k = (beta'_k)^2 * lambda_k / C_total` and all factors targeted equally (uniform budget = 1/K).

**Core idea:** Equalize risk contributions across all latent factors discovered by the VAE, regardless of their profitability.

### Strategy B — Sortino-Budgeted Factor Risk Parity (Proposed)

Replace uniform entropy with **KL-divergence from Sortino-proportional budgets**:

```
max  alpha * H_KL(w || b)  -  lambda * w'Sigma*w  -  phi * P_conc  -  P_turn

where  b_k  proportional to  Sortino_k = mu_k / sigma_down_k
```

Where `H_KL(w || b) = -sum_k  p_k * ln(p_k / b_k)` targets risk contributions proportional to each factor's **downside-risk-adjusted return** (Sortino ratio).

**Core idea:** Allocate more risk budget to factors that deliver higher return per unit of downside risk; less to factors that are volatile without reward.

---

## 2. Advantages and Disadvantages

### Strategy A — Uniform Entropy (Current DVT)

| # | Advantages | Detail |
|---|-----------|--------|
| A1 | **No return estimation needed** | Entropy is computed purely from covariance structure. No mu estimation = no estimation error from expected returns. (DeMiguel, Garlappi & Uppal 2009: estimation error in mu dominates any mean-variance optimality.) |
| A2 | **Maximum factor diversification** | Equal risk contributions across all active factors = maximum effective number of bets (ENB). Portfolio is the most "spread" across independent risk sources. (Meucci 2009) |
| A3 | **Robust to factor model misspecification** | If the VAE discovers spurious factors, equal weighting limits damage — no single factor dominates. Agnostic strategy = robust strategy. |
| A4 | **Theoretically grounded** | Entropy maximization is equivalent to the least informative prior (MaxEnt principle, Jaynes 1957). Equivalent to risk parity on uncorrelated factors (Roncalli 2013, Ch. 7). |
| A5 | **Low turnover** | Without a noisy return signal driving rebalancing, the strategy is more stable across periods. Entropy targets are structural, not directional. |
| A6 | **Simple calibration** | Only one key parameter (alpha) controls the risk-diversification tradeoff. No need to estimate K Sortino ratios. |

| # | Disadvantages | Detail |
|---|-------------|--------|
| D1 | **Sacrifices risk premium** | Tilts away from high-variance factors (market, momentum) that carry risk premia. A factor with lambda_k = 100 and SR_k = 0.8 gets the same budget as a factor with lambda_k = 1 and SR_k = 0.0. Return is sacrificed for diversification. (Clarke, De Silva & Thorley 2013: in equities, min-var beats ERC on Sharpe.) |
| D2 | **Momentum is a band-aid, not optimal** | The mu'w term is a linear tilt injected heuristically. It does not optimize risk-adjusted return. The momentum_weight is a tuned hyperparameter, not derived from any optimality condition. The optimizer doesn't know mu'w exists when computing entropy gradients. |
| D3 | **Sharpe ratio is a diagnostic, not an objective** | The pipeline evaluates Sharpe ex post on OOS returns but never optimizes for it during portfolio construction. There is no feedback loop between realized performance and the construction rule. |
| D4 | **No within-factor diversification guarantee** | Nothing prevents the optimizer from concentrating a factor's exposure in 2-3 stocks instead of spreading it across 50. The w_max constraint and P_conc help, but they don't reason about factor exposure distribution. |
| D5 | **Penalizes all volatility equally** | Standard variance w'Sigma*w penalizes upside and downside symmetrically. A factor that delivers positive skewness (occasional large gains) is penalized identically to one with negative skewness (crashes). |
| D6 | **Calm factors may be noise** | By equalizing risk across factors, the strategy allocates significant risk budget to low-lambda factors that may represent estimation noise rather than genuine risk sources. (Clarke et al. 2013: risk parity includes all assets, including uninformative ones.) |

---

### Strategy B — Sortino-Budgeted Factor Risk Parity (Proposed)

| # | Advantages | Detail |
|---|-----------|--------|
| B1 | **Preserves risk premium proportionally** | Factors with high Sortino ratio (high return per downside risk) receive larger risk budgets. The portfolio captures factor premia instead of diversifying them away. (Roncalli 2013, Ch. 8: non-uniform budgets optimally balance risk and return.) |
| B2 | **Penalizes downside only** | Sortino ratio uses sigma_down, not sigma_total. Factors with positive skewness (upside volatility) are rewarded, not penalized. Better aligned with investor loss aversion. (Sortino & van der Meer 1991; Rom & Ferguson 1993.) |
| B3 | **Eliminates momentum band-aid** | Return information enters directly through the budget structure, not as a separate heuristic term. No need for a separate mu'w term with an arbitrary momentum_weight parameter. |
| B4 | **Still diversified** | KL-divergence H_KL(p || b) still acts as an entropy regularizer — it prevents any factor from receiving 0% or 100% of risk. Minimum budget (floor) ensures all factors keep some allocation. Diversification is maintained, but not forced to be uniform. |
| B5 | **Adapts to regime changes** | Factor Sortino ratios are computed on trailing data (e.g., 252 days). If a factor's risk premium disappears or inverts, its budget naturally decreases at the next rebalancing. |
| B6 | **Reduces noise factor allocation** | Low-Sortino factors (noise) get small budgets; high-Sortino factors (signal) get large budgets. Natural screening without explicit pruning. |

| # | Disadvantages | Detail |
|---|-------------|--------|
| B7 | **Requires factor return estimation** | Need trailing factor returns z_hat_t to compute mu_k and sigma_down_k. Introduces estimation error that Strategy A completely avoids. Factor returns from cross-sectional regression carry residual noise. |
| B8 | **Downside deviation needs sufficient data** | sigma_down uses only negative-return days (~50% of observations). With 252 trailing days and K=20 factors, each factor has ~120-130 negative observations. Adequate for point estimates but unreliable for confidence intervals. |
| B9 | **Sensitive to lookback window** | Sortino ratios computed on 6-month vs 12-month vs 24-month windows can give different budgets. Adds a hyperparameter (lookback) that needs calibration. |
| B10 | **May chase recent performance** | If a factor had a good trailing period, its Sortino ratio is high, and it receives a large budget. This can be a form of factor momentum — beneficial on average, but vulnerable to factor reversal. |
| B11 | **Higher turnover** | Budgets change each rebalancing as Sortino ratios evolve. This creates additional rebalancing vs uniform entropy, increasing transaction costs. |
| B12 | **Loss of maximum diversification guarantee** | Non-uniform budgets mean ENB < K by design. The portfolio is intentionally less diversified across risk sources. In periods where factor premia are unstable, this concentration can hurt. |

---

## 3. How the Proposed Solution Addresses DVT Disadvantages

| DVT Disadvantage | How Strategy B Addresses It | Residual Limitation |
|---|---|---|
| **D1: Sacrifices risk premium** | Sortino-proportional budgets allocate more risk to profitable factors → premium captured. | Estimation noise in Sortino ratios may misidentify which factors are truly profitable. |
| **D2: Momentum is a band-aid** | Return information enters through budgets, not a separate linear term. No arbitrary momentum_weight. | Still relies on trailing return estimates — just structured better. |
| **D3: Sharpe is diagnostic only** | Sortino ratio is explicitly part of the construction rule via budgets. Portfolio is built to favor high-Sortino factors. | Sortino is computed on in-sample trailing data, not forward-looking OOS Sharpe. |
| **D4: No within-factor guarantee** | Not directly addressed. Same w_max and P_conc constraints. | Remains unaddressed — would need explicit within-factor diversification constraint. |
| **D5: Penalizes all volatility** | Sortino uses downside deviation only. Upside volatility does not reduce a factor's budget. | The risk penalty (w'Sigma*w) in the objective still uses full variance, not semi-variance. Only the budget is Sortino-aware. |
| **D6: Calm factors may be noise** | Low-Sortino factors naturally get small budgets → noise factors are de-emphasized. | If a noise factor has a lucky positive trailing period, it may receive an undeserved high budget. |

---

## 4. What is Lost from Strategy A

| Strategy A Advantage | Impact of Switching to Strategy B | Severity |
|---|---|---|
| **A1: No return estimation needed** | **Lost.** Strategy B requires estimating K factor Sortino ratios from trailing data. Introduces estimation error and a lookback hyperparameter. | **Moderate.** Factor-level estimation (K=10-30) is much more stable than stock-level (n=500+), but still noisy. |
| **A2: Maximum diversification** | **Partially lost.** ENB will be lower than maximum by design. Non-uniform budgets intentionally concentrate risk in high-Sortino factors. | **Low-Moderate.** KL-divergence still prevents extreme concentration. Diversification is reduced, not eliminated. |
| **A3: Robust to misspecification** | **Partially lost.** If the VAE discovers spurious factors with high trailing Sortino (lucky noise), Strategy B allocates significant risk to them. Strategy A's agnosticism was protective. | **Low.** AU pruning already removes inactive factors. A Sortino floor parameter can limit damage. |
| **A4: Theoretically grounded** | **Preserved.** KL-divergence from non-uniform budgets is well-founded (Roncalli 2013, Ch. 7-8; Bruder & Roncalli 2012). Sortino budgets have literature support (Rom & Ferguson 1993, Sortino & van der Meer 1991). | **None.** |
| **A5: Low turnover** | **Partially lost.** Budgets change across rebalancing dates as trailing Sortino ratios evolve. This increases turnover vs static uniform budgets. | **Low-Moderate.** EWMA smoothing of Sortino ratios and turnover constraints (kappa_1, kappa_2, tau_max) mitigate this. |
| **A6: Simple calibration** | **Partially lost.** Adds hyperparameters: Sortino lookback window, floor parameter, EWMA half-life for budget smoothing. | **Low.** These are second-order tuning parameters with reasonable defaults (lookback=252, floor=0.05, EWMA=63). |

---

## 5. Remaining Disadvantages After the Proposed Solution

| # | Remaining Issue | Source | Mitigation Available |
|---|----------------|--------|---------------------|
| R1 | **Within-factor concentration** | Both strategies | Add explicit constraint: `max_i(B_{i,k} * w_i) / sum_i(B_{i,k} * w_i) <= threshold` per factor. Computationally expensive (K additional constraints per SCA iteration). |
| R2 | **Full variance in risk penalty** | Both strategies | Replace w'Sigma*w with downside semi-variance. Requires semi-covariance matrix estimation (O(n^2) with half the data). Literature support: Estrada (2006), Ang, Chen & Xing (2006). Significant implementation effort. |
| R3 | **Backward-looking budgets** | Strategy B | Factor Sortino ratios reflect the past, not the future. Factor premia can reverse. Mitigation: shrink Sortino toward cross-sectional mean (Bayesian shrinkage), or use regime-conditional estimates. |
| R4 | **No explicit Sharpe optimization** | Both strategies | Neither strategy directly maximizes portfolio Sharpe. Strategy B approximates it through budgets, but the mapping from factor-level Sortino to portfolio-level Sharpe is indirect. Full Sharpe optimization requires stock-level return forecasts, which are unreliable. |
| R5 | **Estimation error in factor covariance** | Both strategies | Sigma_z estimation affects both risk contributions (p_k) and Sortino denominators. Ledoit-Wolf shrinkage helps, but K eigenvalues still carry sampling noise. |
| R6 | **Cardinality constraints interaction** | Both strategies | After cardinality enforcement (MIQP/two-stage), the Sortino-budgeted risk contributions may be distorted. The removed stocks may carry exposure to high-budget factors, undermining the budget design. |

---

## 6. Consolidated Comparison Table

| Criterion | Strategy A (Uniform Entropy) | Strategy B (Sortino-Budgeted) | Winner |
|---|---|---|---|
| **Expected return** | Low (premium sacrificed) | Higher (premium preserved) | **B** |
| **Volatility** | Lower (risk-spread) | Moderately higher (concentrated in rewarded factors) | **A** |
| **Sharpe ratio** | Moderate (high diversification, low return) | Expected higher (return/risk balanced by Sortino) | **B** (expected) |
| **Sortino ratio** | Moderate | Expected higher (downside-aware by construction) | **B** (expected) |
| **Maximum drawdown** | Lower (less exposure to crash-prone factors) | Moderately higher (more exposure to rewarded but risky factors) | **A** |
| **Turnover** | Lower (static budgets) | Higher (budgets evolve with Sortino) | **A** |
| **Estimation risk** | None (covariance only) | Moderate (K Sortino ratios) | **A** |
| **Robustness to noise factors** | Moderate (equal budget = equal noise exposure) | Higher (noise → low Sortino → low budget) | **B** |
| **Robustness to regime change** | Low (ignores regime) | Moderate (trailing Sortino adapts) | **B** |
| **Calibration complexity** | Low (1 parameter: alpha) | Moderate (alpha + lookback + floor + smoothing) | **A** |
| **Theoretical foundation** | MaxEnt (Jaynes 1957), Risk Parity (Meucci 2009) | Budgeted RP (Roncalli 2013 Ch.8), Post-Modern PT (Rom & Ferguson 1993) | **Tie** |
| **Within-factor diversification** | Not addressed | Not addressed | **Tie** |

---

## 7. Recommendation

**Implement Strategy B as an additional benchmark** alongside the current Strategy A. The walk-forward validation framework (34 folds, 30 years) will provide empirical evidence on which approach dominates for the VAE's latent factor structure.

Key implementation requirements:
1. Compute factor-level trailing Sortino ratios from cross-sectional regression residuals z_hat_t
2. Map Sortino ratios to budgets: `b_k = max(So_k, floor) / sum(max(So_j, floor))`
3. Pass budgets to existing `H_KL(p || b)` in entropy.py (tilted entropy already implemented)
4. EWMA smoothing (half-life = 63 days) on Sortino ratios to reduce turnover

The existing codebase already supports:
- Tilted entropy with non-uniform budgets (entropy.py, `budget` parameter)
- KL-divergence gradient computation (entropy.py, `_factor_entropy_and_gradient`)
- Budget propagation through the pipeline (config.py `entropy_budget_mode`, pipeline.py)

The main new code is: (1) Sortino ratio estimation at factor level, (2) budget smoothing, (3) a new benchmark class or pipeline mode.

---

## References

- Asness, Frazzini & Pedersen (2012). *Leverage Aversion and Risk Parity.* Financial Analysts Journal, 68(1), 47-59.
- Bruder & Roncalli (2012). *Managing Risk Exposures Using the Risk Budgeting Approach.* SSRN 2009778.
- Choueifaty, Froidure & Reynier (2013). *Properties of the Most Diversified Portfolio.* Journal of Investment Strategies, 2(2), 49-70.
- Clarke, De Silva & Thorley (2013). *Risk Parity, Maximum Diversification, and Minimum Variance.* Journal of Portfolio Management, 39(3), 39-53.
- DeMiguel, Garlappi & Uppal (2009). *Optimal Versus Naive Diversification.* Review of Financial Studies, 22(5), 1915-1953.
- Meucci (2009). *Managing Diversification.* Risk, May 2009, 74-79.
- Roncalli (2013). *Introduction to Risk Parity and Budgeting.* Chapman & Hall/CRC.
- Roncalli & Weisang (2016). *Risk Parity Portfolios with Risk Factors.* Quantitative Finance, 16(3), 377-388.
- Rom & Ferguson (1993). *Post-Modern Portfolio Theory Comes of Age.* Journal of Investing, 2(4), 27-33.
- Sortino & van der Meer (1991). *Downside Risk.* Journal of Portfolio Management, 17(4), 27-31.
- Xiong & Idzorek (2011). *The Impact of Skewness and Fat Tails on the Asset Allocation Decision.* Journal of Asset Management, 12, 126-141.
