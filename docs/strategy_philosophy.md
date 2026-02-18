# Strategy Philosophy — Latent Risk Factor Discovery

*Synthesized from DVT v4.1 — Methodological Synthesis Document*

---

## 1. Diversify risk, do not predict returns

The goal is not to maximize returns through prediction — a problem where even the best quantitative funds achieve only marginal edges. The goal is to **minimize risk through optimal diversification**: identify the truly independent sources of risk, then build a portfolio where no exposure is unintentionally duplicated.

The distinction is epistemological. Predicting returns requires being *right about the future*. Identifying risk requires being *right about the present structure* of dependencies between assets — a fundamentally more tractable problem. The market's dependency structure is not static, but it evolves slowly enough to be measured, modeled, and acted upon between periodic retrainings.

**Key consequence:** the strategy produces no return forecast by default (mu = 0). Return views, if any, are injected as a separate, removable layer (see Principle 9).

---

## 2. Let the data reveal risk factors — no a priori bias

Traditional diversification (by sector, geography, asset class) relies on human-defined categories. But an investor "diversified" across 10 sectors may in reality be massively concentrated on a single latent risk factor — say, global growth sensitivity — that cuts across all categories. When that factor turns, the entire portfolio moves together.

**Core principle:** invert the classical workflow. Instead of starting from economic theory ("these are the factors that matter") and then measuring exposures, start from observed price behavior ("these assets move together") and ask *afterward* what economic reality might explain the discovered structure.

The model is free to discover risk factors that no analyst would have thought to define — factors that may not correspond to any standard classification but that genuinely drive portfolio risk. Factor interpretation is optional and always *a posteriori*.

---

## 3. The exposure heatmap: every stock, every factor, varying intensity

Forget tree-like classifications where each stock belongs to one sector, one geography, one style box. The real structure of risk is a **continuous heatmap**: a matrix of n stocks by K factors where each cell represents a degree of exposure — not binary membership, but a continuous intensity.

```
              Factor 1   Factor 2   ...   Factor K
Stock A    [   0.82       0.15      ...    -0.03   ]
Stock B    [   0.04       0.71      ...     0.45   ]
Stock C    [   0.67       0.33      ...     0.12   ]
```

A pharmaceutical company is simultaneously exposed to drug regulation risk, the business cycle, the EUR/USD rate, and dozens of other factors — with different intensities for each. The exposure matrix B (n x K) captures this full picture: multiple memberships, continuous intensities, and no forced categorization.

This is what the VAE encoder produces: each stock's position in a continuous K-dimensional space where proximity means similar risk profiles and distance means different ones. The exposures are not assigned by an analyst — they are measured from the data.

---

## 4. Memory of dormant risks — anti-cyclical by construction

Not all risk factors are active at all times. Three temporal categories:

- **Structural factors** (industry, geography, supply chain): permanent, continuously visible in prices. Stability above 0.90 across periods.
- **Style factors** (leverage, size, momentum): slowly evolving, half-lives exceeding 25 months.
- **Episodic factors** (crisis contagion, tail dependence): dormant for years, then suddenly dominant. These drive the most destructive drawdowns precisely because they activate when correlations spike across assets that appeared uncorrelated.

A system that diversifies only against *currently observable* correlations is structurally blind to episodic risks. It will react *after* the damage is done — selling correlated assets after a shock rather than being protected before it.

**The model's answer:** build a **complete historical risk profile** for each stock by aggregating its latent vectors across its entire available history. A stock that was exposed to interbank contagion in 2008 retains that exposure in its composite profile, even if the factor has been dormant since. When it reactivates, the portfolio is already diversified against it.

This is the **anti-cyclical principle**: the training window is expanding (not rolling), the composite profile is a mean over all windows, and the factor covariance is estimated on full history. Crisis correlations persist permanently in the model.

---

## 5. Prices for discovery, fundamentals for interpretation

Price data is chosen for latent factor discovery because:

- **High frequency** (daily): maximizes training volume, captures crisis dynamics where diversification matters most.
- **No a priori bias**: using fundamentals as input would bias the model toward structures we inject (sectors, leverage ratios), at the expense of truly latent factors.
- **Captures risks as they manifest**: investment flows, analyst herding, sentiment, market microstructure — all visible in prices, invisible in balance sheets.
- **No publication delay**: balance sheets are quarterly, macro data monthly. Prices are daily.

Fundamental data is reserved for *a posteriori* interpretation of discovered factors: once the model identifies a latent dimension, it can be correlated with fundamental characteristics to understand what it represents economically. But the discovery itself is unbiased.

---

## 6. Why a VAE: the model decides how many risk factors exist

The number of risk factors is not chosen in advance. The Variational Autoencoder architecture provides a mechanism for automatic factor count determination:

- **K is a capacity ceiling** (default: 200), not the number of factors the model will use.
- The **KL divergence** in the loss function pushes unused latent dimensions toward the prior N(0,1), effectively deactivating them. Only dimensions that capture genuine shared structure resist this pressure.
- Dimensions with marginal KL > 0.01 nats are **active units (AU)**: the effective factor count discovered from data. A VAE with K=200 on data supporting 80 factors converges to AU ≈ 80 with ~120 inactive dimensions.
- Unlike PCA (where every component captures additional variance), **superfluous VAE dimensions deactivate themselves**. K should therefore be set generously — the effective factor count is read as AU after training.

This automatic pruning means the strategy does not require a human decision about how many risk factors exist in the market. The model discovers the answer from the data, constrained only by a statistical guard that ensures the downstream covariance estimation remains reliable (AU_max_stat ≈ 85 for 30 years of daily data).

---

## 7. Equalize factor risk contributions — do not minimize exposures

This is the most important and most subtle principle. The strategy does **not** minimize the portfolio's exposure to each risk factor. It **equalizes the contribution of each factor to total portfolio risk**.

The distinction is critical. Given the principal factor basis (after diagonalizing the factor covariance Sigma_z = V Lambda V^T), each factor k contributes:

```
c'_k = (beta'_k)^2 * lambda_k
```

where beta'_k is the portfolio's exposure to principal factor k and lambda_k is the factor's variance (eigenvalue). The entropy of these contributions:

```
H(w) = -sum_k  chat'_k * ln(chat'_k)     where  chat'_k = c'_k / sum_j c'_j
```

is maximized when all contributions are equal: c'_1 = c'_2 = ... = c'_AU.

**This implies:**

```
(beta'_k)^2 * lambda_k = constant    for all k
```

Therefore:

```
|beta'_k| proportional to 1 / sqrt(lambda_k)
```

The portfolio's exposure to each factor is **inversely proportional to the square root of the factor's variance**. For a strong factor (large lambda_k, like the market), the portfolio has a **small** exposure. For a weak factor (small lambda_k, like dormant crisis contagion), the portfolio has a **large** exposure.

| Factor | Variance (lambda) | Exposure (beta') | Risk contribution (c') |
|--------|-------------------|------------------|------------------------|
| Market (strong) | 0.050 | small (4.5) | 1/AU |
| Momentum (medium) | 0.010 | medium (10.0) | 1/AU |
| Contagion (weak) | 0.001 | large (31.6) | 1/AU |

**This is not "being exposed to everything equally."** It is being exposed to each risk *in the exact proportion that makes its contribution to total portfolio risk identical to all others*. The portfolio "bets big" on weak factors because their risk contribution remains small even with large exposure — and "bets small" on strong factors because even a small exposure generates significant risk.

The complete objective function balances this entropy maximization against total risk minimization:

```
max_w  -lambda * w^T Sigma w  +  alpha * H(w)
```

The ratio alpha/lambda is calibrated at the elbow of the variance-entropy frontier: the point where the marginal gain in diversification no longer justifies the additional variance cost.

---

## 8. Dual diversification: across factors and within each factor

The strategy diversifies at two levels simultaneously:

**Between factors (inter-factor):** the entropy H(w) ensures no single principal factor dominates the portfolio's risk. This is Principle 7.

**Within each factor (intra-factor):** portfolio constraints prevent the portfolio from "betting" on a single stock per factor:

- **Concentration penalty**: soft threshold at 3% per stock, hard cap at 5%.
- **Minimum position size**: 0.10% or zero (no dust positions).
- **Entropy-aware cardinality enforcement**: when eliminating sub-threshold positions, the optimizer removes the stock whose elimination costs the *least* entropy — preserving positions that are factorially important.

Why does this matter? If the portfolio achieved factor diversification by holding one stock per factor, a single stock-specific event (fraud, product failure) would wipe out the portfolio's exposure to that factor entirely. Holding multiple stocks per factor makes the portfolio robust to idiosyncratic shocks while maintaining the same factor risk profile.

The concentration penalty and minimum position constraints work *with* the entropy objective: they force the optimizer to achieve factor diversification through many small, distributed positions rather than a few concentrated bets.

---

## 9. Modular separation: risk model and return views are independent

The VAE produces a **pure risk model** — a factor covariance structure with no opinion about which direction any factor will move. The default mode sets mu = 0: no return forecast.

Return-seeking behavior is a separate, removable layer:

- **Default mode** (mu = 0): max -lambda w^T Sigma w + alpha H(w). Pure risk diversification.
- **Directional mode** (mu != 0): max w^T mu - lambda w^T Sigma w + alpha H(w). External return views (momentum, valuation, analyst consensus) are injected, but the risk infrastructure is unchanged.

This separation is a deliberate design choice:

1. **Risk structure is intrinsic** to the market and measurable from data. Return forecasts are external opinions that may or may not be correct.
2. **Testability**: the risk model can be validated independently (covariance forecast accuracy, factor explanatory power) regardless of whether return views are correct.
3. **Upgradability**: return signals can be improved, replaced, or removed without touching the factor discovery or risk estimation pipeline.
4. **Intellectual honesty**: if no reliable return signal exists, the strategy gracefully degrades to maximum-diversification rather than injecting noise.

---

## 10. Rigorous walk-forward validation — intellectual honesty as a principle

The strategy is validated through walk-forward simulation over ~34 folds spanning 30 years of history, not through random train/test splits (which leak future information for financial time series).

**Key properties:**

- **Expanding training window**: each fold uses all available history, preserving crisis memory (consistent with Principle 4).
- **Embargo gap** (21 days): prevents residual temporal dependence from contaminating the out-of-sample evaluation.
- **Final holdout** (~3 years): permanently reserved, never touched during hyperparameter selection — the only uncontaminated evaluation.
- **6 benchmarks under identical constraints**: equal-weight, inverse volatility, minimum variance, equal risk contribution, PCA factor risk parity, PCA + realized volatility. All share the same universe, constraints, and walk-forward protocol.

**The terminal principle:** if after all iterative improvements the VAE strategy does not statistically outperform the PCA factor risk parity baseline across walk-forward folds, the intellectual elegance of the architecture does not justify its operational complexity. The VAE is abandoned in favor of the simpler model.

Complexity must earn its place through measurable out-of-sample benefit. No exceptions.
