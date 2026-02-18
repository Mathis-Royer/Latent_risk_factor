# Literature Review: Strategy Analysis

> **Last Updated:** 2026-02-18
> **Status:** Comprehensive review of academic literature addressing three fundamental questions about the latent risk factor strategy.

---

## Executive Summary

This document synthesizes academic research (2015-2025) on three key questions:

1. **Downside Risk**: Should we use semi-variance/Sortino instead of symmetric volatility?
2. **Portfolio Diversification**: How to maximize stock count and integrate diversification into optimization?
3. **Factor Characterization**: What dimensions describe a risk factor beyond eigenvalue magnitude?

**Key Finding**: The current symmetric covariance approach with crisis weighting is defensible for a diversification-focused (mu=0) strategy. Recommended enhancements focus on cardinality constraints, factor quality metrics, and MDP diversification ratio rather than switching to downside risk measures.

---

## Question 1: Downside Risk vs. Symmetric Volatility

### Current Strategy Approach

The strategy uses **symmetric covariance** with crisis weighting (gamma=3.0) in VAE training as the only asymmetric element. The philosophy states: "What matters is the effective correlation of assets as it manifests in price co-movements."

### Literature Review

#### The Downside Beta Risk Premium Debate

**Original Finding — Ang, Chen & Xing (2006)**: [Downside Risk, RFS](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=641843)
- Stocks with high downside beta (covariance with market during downturns) earn ~6% annual premium over CAPM predictions
- Rationale: Loss-averse investors require compensation for holding assets that fall when markets fall

**Subsequent Critiques — Atilgan, Demirtas & Gunaydin (2020)**: [European Financial Management](https://onlinelibrary.wiley.com/doi/abs/10.1111/eufm.12258)
- The downside beta premium **disappears** with:
  - Value-weighted returns (vs. equal-weighted)
  - Extended sample periods (post-2006)
  - Ex-ante measurement (vs. ex-post)
  - Control for other return determinants
- Conclusion: "The positive association between down-betas and returns disappears when down betas are measured ex ante rather than ex post"

**Recent Evidence — Barroso & Maio (2020)**: [RFS Symmetric and Asymmetric Market Betas](https://academic.oup.com/rfs/article/33/6/2772/5571734)
- Mixed results; the premium is sensitive to methodology and sample selection

#### Semi-Variance/Sortino Optimization

**Core Theory — Fishburn (1977), Bawa (1975)**:
- Lower Partial Moment (LPM) generalizes semi-variance: LPM_n(tau) = E[max(0, tau - R)^n]
- LPM_2 (semi-variance) = Sortino-denominator when tau = 0 or risk-free rate
- Advantage: Focuses on "bad" volatility, ignoring upside deviations

**Implementation Challenges — [Annals of Operations Research 2024](https://link.springer.com/article/10.1007/s10479-024-06043-z)**:
- "Downside risk measures are particularly affected by parameter uncertainty because the estimates of the necessary inputs are less reliable than the estimates of the full covariance matrix"
- Semi-covariance requires 2x the observations for equivalent estimation precision
- Non-convex optimization (semi-covariance is not positive semi-definite in general)

### Synthesis: Should the Strategy Use Downside Risk?

| Criterion | Symmetric Volatility | Semi-Variance/Sortino |
|-----------|---------------------|----------------------|
| **Risk Premium Capture** | Neutral (no return forecast) | Theoretically better (targets downside beta premium) |
| **Estimation Reliability** | **Better** (N observations) | Worse (requires ~2N for equivalent precision) |
| **Convex Optimization** | **Yes** (SCA works) | No (semi-covariance non-PSD; requires SOCP reformulation) |
| **Factor Covariance** | Direct (Ledoit-Wolf) | Non-trivial (asymmetric factor semi-covariance unclear) |
| **Anti-Cyclical Memory** | Preserves all crisis correlations | Preserves only downside crisis correlations |
| **Empirical Evidence for Premium** | N/A (diversification focus) | **Disputed** (Atilgan 2020: premium disappears ex-ante) |

### Recommendation

**The current symmetric approach is defensible** for a diversification-focused (mu=0) strategy because:

1. **No return forecast = no premium capture regardless**: If mu=0, the strategy doesn't seek the downside beta premium anyway.

2. **Estimation precision matters**: With 50 years of data and 20,000 stocks, semi-covariance estimation requires conditioning on market states, cutting effective sample size roughly in half.

3. **The crisis weighting (gamma=3.0) in VAE training already provides asymmetry**: The encoder allocates capacity to tail co-movement patterns, capturing downside correlations implicitly.

**Alternative (Tier 3, High Risk)**: Consider **Sortino-budgeted factor risk parity** — use symmetric covariance for factor discovery (VAE), but replace the portfolio objective with H_downside over downside factor risk contributions and Sigma_down semi-covariance.

---

## Question 2: Portfolio Diversification & Cardinality

### Current Strategy Approach

- Hard cap w_max = 5%, soft threshold w_bar = 3%
- Minimum position w_min = 0.1% (semi-continuous)
- **Result**: Typically 20-50 positions (dozens, not hundreds)
- Concentration penalty phi = 0.0 (disabled)
- Diversification enters only as **entropy** (factor-level) not stock-count

### Literature Review

#### How Many Stocks for Idiosyncratic Risk Reduction?

**Classic View**: 8-10 stocks for "adequate" diversification (Evans & Archer 1968)

**Modern Evidence — [JRFM 2021](https://www.mdpi.com/1911-8074/14/11/551)**:
- "Unsystematic risk has increased relative to overall stock market variability over the past thirty years"
- "Correlations among stocks have declined correspondingly, underscoring the need for larger portfolios"
- 18 different performance measures yield vastly different optimal stock counts
- "A significant amount of idiosyncratic risk remains, even for portfolios with large numbers of stocks"

**Key Finding — DeMiguel et al. (2009)**: [1/N vs. Optimal Portfolios](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1376199)
- "The estimation window needed for mean-variance to outperform 1/N is ~3000 months for 25 assets, ~6000 months for 50 assets"
- With 50 years (600 months), mean-variance cannot reliably outperform 1/N for portfolios of 50+ stocks
- **Implication**: For very large portfolios, simple rules (equal-weight, inverse-vol) dominate optimized strategies

#### Cardinality Constraints in Optimization

**Academic Framework — [Operations Research 2014](https://pubsonline.informs.org/doi/10.1287/opre.2013.1170)**:
- Cardinality constraints (max n_stocks) are NP-hard (MIQP)
- Two approaches: (1) Exact MIQP (small n), (2) Heuristics (large n)

**Practical Implementation — [Quantitative Finance 2021](https://www.tandfonline.com/doi/full/10.1080/14697688.2021.1879392)**:
- "Double cardinality constraints capture the trade-off between investment scale limits and diversified industry coverage"
- Industry-level cardinality + stock-level cardinality = two-dimensional diversification

#### Maximum Diversification Portfolio (MDP) — Choueifaty

**Original Paper — [Choueifaty & Coignard 2008](https://www.tobam.fr/wp-content/uploads/2014/12/TOBAM-JoPM-Maximum-Div-2008.pdf)**:
- Maximize Diversification Ratio: DR = (Sum w_i sigma_i) / sigma_portfolio
- Interpretation: Ratio of weighted average volatility to portfolio volatility
- Higher DR = more diversification benefit from correlations

**Comparison with Entropy — [Quantitative Finance 2018](https://www.tandfonline.com/doi/full/10.1080/14697688.2017.1383625)**:
- Rao's Quadratic Entropy (RQE) generalizes both MDP and Shannon entropy
- RQE = Sum_ij w_i w_j d(i,j) where d(i,j) is a distance metric
- With d(i,j) = 1 - rho_ij: RQE ~ MDP
- With d(i,j) = indicator(i != j): RQE ~ Shannon weight entropy

#### Hierarchical Risk Parity (HRP) — Lopez de Prado

**Key Innovation — [Lopez de Prado 2016](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678)**:
- Cluster assets hierarchically using correlation distance
- Allocate risk by recursive bisection (no covariance inversion required)
- "HRP portfolios address instability, concentration, and underperformance of quadratic optimizers"
- "HRP can compute a portfolio on an ill-degenerated or even singular covariance matrix"

**Relevance**: HRP naturally produces portfolios with **hundreds of positions** because it doesn't require covariance inversion.

### The "Breadth vs. Depth" Problem

**Question**: Should exposure to a factor increase by adding more correlated stocks, not by increasing weight on existing stocks?

| Approach | Advantages | Disadvantages |
|----------|------------|---------------|
| **Depth** (increase weight on existing stocks) | Simple, lower turnover | Concentration risk, idiosyncratic exposure |
| **Breadth** (increase number of stocks per factor) | Lower idiosyncratic risk | Higher turnover, more complexity |

**Fundamental Law of Active Management** (Grinold 1989):
- IR ~ IC x sqrt(Breadth)
- More independent bets (breadth) improve information ratio
- **But**: Adding correlated stocks != independent bets

### Recommendation

**For 20,000 stocks and 50 years of data**, implement a **tiered diversification approach**:

1. **Tier 1: Minimum stock count** (Validated): n_stocks >= 100
   - Rationale: DeMiguel shows estimation error dominates for n < 100 with 600-month history

2. **Tier 1: Factor-level breadth**: For each of the top 20 factors (by eigenvalue), require >= 5 stocks with |B_ik| > 0.3
   - Rationale: Prevents single-stock-per-factor vulnerability (Principle 8)

3. **Tier 2: MDP component**: Replace concentration penalty with diversification ratio
   - Current: max H(w) - lambda w^T Sigma w - phi P_conc(w)
   - Proposed: max H(w) + beta * DR(w) - lambda w^T Sigma w

4. **Tier 3: HRP fallback**: For n > 500, consider HRP due to numerical stability

---

## Question 3: Factor Characterization Beyond Eigenvalue Magnitude

### Current Strategy Approach

Factors are characterized by:
- **Active Units (AU)**: KL divergence > 0.01 nats (binary: active/inactive)
- **AU_max_stat**: Statistical upper bound on factor count
- **Implicit eigenvalue ranking**: Principal factors by lambda_k

### Literature Review

#### Factor Quality Dimensions

**1. Persistence (Half-Life)**

[Lettau & Pelger 2020, JFE](https://www.sciencedirect.com/science/article/abs/pii/S0304407620300051): "Estimating Latent Asset-Pricing Factors"
- Factors with longer half-lives (slower mean-reversion) are more investable
- Measurement: Autocorrelation of factor returns; half-life = ln(2) / ln(|rho_1|)
- **Relevance**: A factor with 3-day half-life cannot be exploited with monthly rebalancing

**2. Breadth (Number of Affected Stocks)**

[Giglio et al. 2021](https://finance.business.uconn.edu/wp-content/uploads/sites/723/2021/04/wf_22.pdf): "Test Assets and Weak Factors"
- **Strong factors**: Affect essentially all assets (market factor)
- **Weak factors**: Affect only a subset (industry-specific, tail-risk)
- "Many asset-pricing factors are weak" — harder to detect but may be more profitable
- **Relevance**: VAE may discover both strong and weak factors; weak factors need explicit detection

**3. Signal-to-Noise Ratio (Eigenvalue Gap)**

[Onatski 2010](https://academic.oup.com/jfec/article/23/1/nbad024/7271793): Eigenvalue Tests for Latent Factors
- Gap between adjacent eigenvalues indicates signal vs. noise
- Large gap after factor k implies k factors are "real"
- **Relevance**: Validates AU count; complements KL-based pruning

**4. Crowding (Investor Overlap)**

- Not directly measurable from prices alone
- Proxy: Factor return correlation with known crowded strategies (momentum, value)
- **Relevance**: Crowded factors may have compressed premiums and crash risk

**5. Tail Dependence (Non-Linear Correlations)**

- Factors may appear uncorrelated normally but correlate strongly in tails
- Measurement: Tail dependence coefficient, copula analysis
- **Relevance**: Crisis weighting (gamma=3.0) partially captures this, but not explicitly measured

#### Factor Count Selection: Bai-Ng IC2 vs. VAE AU

**Bai-Ng IC2 — [Econometrica 2002](https://sebkrantz.github.io/dfms/reference/ICr.html)**:
- Penalized least-squares: IC_2 = ln(MSE) + k * (n+T)/(nT) * ln(min(n,T))
- Requires: n, T tend to infinity jointly
- With 20,000 stocks x 12,500 days: well within asymptotic regime

**Onatski Eigenvalue Test — [JFEc 2025](https://academic.oup.com/jfec/article/23/1/nbad024/7271793)**:
- Uses Tracy-Widom law for eigenvalue gaps
- Provides p-values (statistical significance)
- Works for short panels (small T)

**VAE Active Units (AU)**:
- Non-parametric: No asymptotic assumptions
- Captures non-linear factor structure (unlike Bai-Ng/Onatski which assume linear)
- **But**: No formal statistical test; 0.01 nats threshold is heuristic

### What Determines the K Factors?

**Answer: All three dimensions, but weighted differently**

The VAE encoder discovers factors by minimizing reconstruction error + KL regularization:
- **Recurrence**: Factors must explain variance across many windows (or KL pushes them to prior)
- **Strength**: High-variance factors are prioritized (reconstruction error dominates)
- **Scope**: Factors affecting many stocks contribute more to reconstruction error

**Empirically**, the top factors by eigenvalue tend to be:
1. Market factor (highest lambda, universal scope, permanent recurrence)
2. Sector factors (moderate lambda, medium scope, permanent)
3. Style factors (moderate lambda, broad scope, slowly evolving)
4. Crisis factors (low lambda normally, broad scope in crises, episodic recurrence)

### Recommendation: Extended Factor Quality Dashboard

For each factor k, compute and track:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Eigenvalue** | lambda_k from Sigma_z | Variance contribution (strength) |
| **Scope** | count(|B_ik| > 0.3) | Number of affected stocks |
| **Persistence** | ln(2) / ln(|autocorr(z_k)|) | Half-life in days |
| **Stability** | corr(B^(t), B^(t-1)) | Exposure stability across folds |
| **KL Activity** | KL_k from VAE | Active vs. dormant classification |
| **Eigenvalue Gap** | lambda_k - lambda_{k+1} | Signal-to-noise separation |

**Use these metrics to**:
1. Weight factors in entropy calculation (more persistent = more weight)
2. Flag unstable factors for potential exclusion
3. Distinguish structural, style, and episodic factors programmatically

---

## Summary: Recommended Strategy Modifications

### Tier 1: Validated Changes (Minimal Risk)

1. **Factor quality dashboard**: Track persistence, scope, stability, eigenvalue gap per factor
2. **Minimum stock count constraint**: n_stocks >= 100 (reduces idiosyncratic risk)
3. **Per-factor breadth constraint**: >= 5 stocks per top-20 factor

### Tier 2: Research-Backed Changes (Moderate Risk)

4. **MDP diversification ratio component**: Explicitly reward diversification ratio
5. **Persistence-weighted entropy**: Weight factor contributions by half-life
6. **Bai-Ng/Onatski validation**: Cross-check AU count with formal statistical tests

### Tier 3: Fundamental Redesign (High Risk, High Potential)

7. **Sortino-budgeted factor risk parity**: Replace symmetric Sigma with semi-covariance in portfolio construction (not VAE training)
8. **HRP fallback for large n**: Use hierarchical risk parity for n > 500 stocks
9. **Factor-specific cardinality**: Different n_min per factor type (structural > style > episodic)

---

## Verification Plan

After implementation, validate via walk-forward:
1. **Idiosyncratic risk reduction**: Compare realized idiosyncratic contribution (new vs. current)
2. **Factor diversification**: Compare H(w) and ENB across strategies
3. **Drawdown analysis**: Compare maximum drawdown and Sortino ratio
4. **Statistical tests**: Wilcoxon signed-rank for paired fold comparisons

---

## Key Academic Sources

- [Ang, Chen, Xing 2006 - Downside Risk](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=641843)
- [Atilgan et al. 2020 - Downside Beta Replication](https://onlinelibrary.wiley.com/doi/abs/10.1111/eufm.12258)
- [DeMiguel et al. 2009 - 1/N vs. Optimal Portfolios](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1376199)
- [Choueifaty 2008 - Maximum Diversification](https://www.tobam.fr/wp-content/uploads/2014/12/TOBAM-JoPM-Maximum-Div-2008.pdf)
- [Lopez de Prado 2016 - Hierarchical Risk Parity](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678)
- [Lettau & Pelger 2020 - Latent Asset-Pricing Factors](https://www.sciencedirect.com/science/article/abs/pii/S0304407620300051)
- [Onatski 2010 - Eigenvalue Tests](https://academic.oup.com/jfec/article/23/1/nbad024/7271793)
- [Bai & Ng 2002 - Factor Number Selection](https://sebkrantz.github.io/dfms/reference/ICr.html)
- [JRFM 2021 - How Many Stocks for Diversification](https://www.mdpi.com/1911-8074/14/11/551)
- [Quantitative Finance 2018 - Rao's Quadratic Entropy vs MDP](https://www.tandfonline.com/doi/full/10.1080/14697688.2017.1383625)
