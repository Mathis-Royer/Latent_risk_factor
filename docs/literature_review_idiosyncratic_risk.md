# Idiosyncratic Risk: Literature Review

> **Purpose**: Synthesize academic research on idiosyncratic risk magnitude, optimal portfolio cardinality, and endogenous diversification methods. Justifies reactivating the two-layer entropy with `entropy_idio_weight = 0.20`.

---

## 1. Severity Assessment: How Much Does Idiosyncratic Risk Matter?

### 1.1 Magnitude of Idiosyncratic Risk

**Finding: Idiosyncratic risk dominates total stock risk.**

> "Idiosyncratic risk constitutes almost 90 percent of total risk."
> — [Goyal & Santa-Clara (2001)](http://www.econ.yale.edu/~shiller/behfin/2001-05-11/goyal-santa-clara.pdf)

In a factor model framework:
- **Total Risk** = Systematic Risk (factor exposure) + Idiosyncratic Risk (residual)
- A well-diversified portfolio eliminates idiosyncratic risk
- **But**: With only 30-50 stocks, substantial residual variance remains

### 1.2 Has Idiosyncratic Risk Increased?

**Finding: Yes, over the last 30 years.**

> "Non-systematic risk has increased relatively to the total variability of the market over the last 30 years. Correlations between stocks have decreased, emphasizing the need for larger portfolios."
> — [JRFM (2021)](https://www.mdpi.com/1911-8074/14/11/551)

**Implication**: Historical rules of thumb (8-10 stocks) are outdated. Modern markets require more stocks for equivalent diversification.

### 1.3 The DeMiguel Challenge

**Finding: Estimation error often outweighs optimization benefits.**

> "The estimation window needed for the sample-based mean-variance strategy and its extensions to outperform the 1/N benchmark is around **3000 months for 25 assets** and about **6000 months for 50 assets**."
> — [DeMiguel, Garlappi & Uppal (2009)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1376199), Review of Financial Studies

With 50 years (600 months) of data, we are far from the required 3000-6000 months. This means:
- **For large N**: Simple methods (1/N, inverse volatility) often dominate in out-of-sample performance
- **Our VAE approach** must demonstrate clear value-add vs. 1/N benchmark
- **More stocks** = more estimation error, but also less idiosyncratic risk

### 1.4 Terminal Wealth Perspective

**Finding: 100 stocks may still not be enough.**

> "Shortfall risk reduction continues as portfolio size is increased, even above 100 stocks."
> — [Domian et al. (2007)](https://www.researchgate.net/publication/4744485_Diversification_in_Portfolios_of_Individual_Stocks_100_Stocks_Are_Not_Enough)

Traditional variance-based analysis underestimates the benefits of more stocks. When measuring **terminal wealth dispersion** (important for long-term investors), even 100+ stocks show marginal benefits.

---

## 2. How Many Stocks Are Needed? Academic Consensus

### 2.1 Historical Evolution of Recommendations

| Period | Authors | Recommended # Stocks | Metric |
|--------|---------|---------------------|--------|
| 1960s | Evans & Archer (1968) | 8-10 | Variance reduction |
| 1970s | Elton & Gruber (1977) | 15+ | Marginal risk reduction |
| 1980s | Statman (1987) | 30-40 | Utility-based |
| 2000s | Campbell et al. (2001) | 50 | Increased idio risk |
| 2007 | Domian et al. | 100+ | Terminal wealth |
| 2021 | JRFM Review | "No consensus" | Multiple metrics |

### 2.2 Dependence on Portfolio Style

> "For large-cap portfolios, there's little to be gained by diversifying beyond 15 stocks. For small-cap portfolios, peak diversification is achieved with around 26 stocks."
> — [CFA Institute (2021)](https://blogs.cfainstitute.org/investor/2021/05/06/peak-diversification-how-many-stocks-best-diversify-an-equity-portfolio/)

**Our case**: Universe of ~500-2000 stocks (SP500 + extensions), mixed cap → **Target: 100+ stocks** is reasonable.

### 2.3 Factor Model Perspective

> "Idiosyncratic return residuals can be diversified away... Under a factor model, idiosyncratic risk can be largely eliminated if we hold a sufficiently large number of assets."
> — [CFA Institute Factor Models](https://www.cfainstitute.org/en/membership/professional-development/refresher-readings/using-multifactor-models)

**Key insight**: In our VAE factor model:
- Systematic risk is handled by factor entropy (equalizing factor contributions)
- **Idiosyncratic risk requires stock-level diversification**
- The two are complementary, not substitutes

---

## 3. Endogenous Methods vs Fixed Constraints

### 3.0 The Problem with Fixed Constraints

Fixed constraints (n ≥ 100, ≥5 stocks/factor) are **arbitrary** and potentially suboptimal:
- The optimal number depends on the current covariance structure
- A constraint too strict forces unwanted positions
- A constraint too loose leaves risk on the table

**Key question**: Can we **optimize endogenously** the diversification level rather than fixing it?

### 3.1 Regularization Approaches (Endogenous)

#### 3.1.1 L1 Regularization (LASSO) — Brodie et al. (2009)

**Source**: [Brodie et al. (2009)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2169062)

**Objective**:
```
min_w  w'Σw - λ·μ'w + γ·||w||₁
```

**Effect**: L1 penalty produces **sparse** portfolios — some weights exactly zero.

**Problem for our case**: L1 **reduces** the number of stocks, which is the **opposite** of what we want. L1 is useful for reducing transaction costs, not for diversifying.

#### 3.1.2 L2 Regularization (Ridge) — Anti-concentration

**Source**: [MOSEK Cookbook](https://docs.mosek.com/portfolio-cookbook/regression.html)

**Objective**:
```
min_w  w'Σw + γ·||w||₂²
```

**Effect**: Pushes weights toward equal-weighting, **encourages dispersion**.

**Problem**: L2 does not explicitly penalize low stock counts — a portfolio of 30 equally-weighted stocks has a low ||w||₂².

#### 3.1.3 L1/L2 Ratio — Endogenous Cardinality (2024)

**Source**: [Sparse Portfolio via ℓ1/ℓ2 (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0377221724005599)

**Objective**:
```
min_w  w'Σw + γ · (||w||₁ / ||w||₂)
```

**Advantages**:
- **No need to specify cardinality a priori**
- Optimal cardinality emerges from optimization
- Scale-invariant

**Limitation for our case**: Still oriented toward **sparsity**, not **dispersion**.

### 3.2 Concentration Index Approaches (HHI)

#### 3.2.1 Minimizing the Herfindahl-Hirschman Index

**Source**: [Portfolio Optimization Using Modified Herfindahl Constraint](https://link.springer.com/chapter/10.1007/978-3-319-61320-8_10)

**Herfindahl Index**:
```
HHI = Σ w_i²
```

**Effective Number of Stocks**:
```
ENS = 1 / HHI
```

**Modified Objective**:
```
min_w  w'Σw + γ·HHI
```

**Advantages**:
- ENS is **endogenous** — optimized, not fixed
- HHI is convex → stable optimization
- Clear interpretation: maximize effective number of positions

### 3.3 Maximum Diversification Portfolio (MDP)

**Source**: [Choueifaty & Coignard (2008)](https://www.tobam.fr/wp-content/uploads/2014/12/TOBAM-JoPM-Maximum-Div-2008.pdf)

**Diversification Ratio**:
```
DR = Σ w_i σ_i / σ_portfolio
```

**Properties**:
- DR ≥ 1 always (equality when all assets perfectly correlated)
- Maximizing DR encourages **many positions with low correlations**
- Under constant correlations: MDP = Risk Parity

**Comparison with our entropy approach**:

| Aspect | Factor Entropy H(w) | Diversification Ratio DR |
|--------|---------------------|-------------------------|
| Diversifies | Between factors | Between assets |
| Metric | Shannon entropy on factor contributions | Ratio of avg vol to portfolio vol |
| Encourages | Equal factor risk contributions | Many uncorrelated positions |
| Weakness | Ignores stock count | Ignores factor structure |

### 3.4 Effective Number of Bets (ENB)

**Source**: [Meucci (2009)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1358533)

**Already used in our strategy** as a diagnostic metric.

**Definition**:
```
ENB = exp(H(risk contributions))
```

**Interpretation**: ENB ranges from 1 (concentrated) to N (perfectly diversified). A portfolio with ENB = 20 has "20 effective independent bets."

---

## 4. Optimal Method: Factor + Idiosyncratic Risk Budgeting (Roncalli)

### 4.1 The Roncalli Approach

**Source**: [Roncalli - Risk Parity with Risk Factors](http://www.thierry-roncalli.com/download/risk-factor-parity.pdf)

**Risk decomposition**:
```
σ²_portfolio = w'(BΣ_zB' + D_ε)w
             = (Factor Risk) + (Idiosyncratic Risk)
```

**Key idea**: Treat idiosyncratic risk as a **"pseudo-factor"** in risk budgeting.

**Extended risk budget**:
```
RC_factor_k + RC_idio = σ²_portfolio

where:
- RC_factor_k = contribution of factor k
- RC_idio = w'D_ε w = idiosyncratic contribution
```

### 4.2 Two-Layer Entropy Formulation

**Current entropy**:
```
H_factor(w) = -Σ_k ĉ_k · log(ĉ_k)   where ĉ_k = normalized factor k contribution
```

**Extended entropy (implemented in entropy.py)**:
```
H_total(w) = (1 - w_idio) · H_factor(w) + w_idio · H_idio(w)

where:
- H_factor = -Σ_k ĉ'_k ln(ĉ'_k)   (entropy over AU systematic contributions)
- H_idio  = -Σ_i ĉ^ε_i ln(ĉ^ε_i)  (entropy over n idiosyncratic contributions)
```

**Two-layer rationale** (from entropy.py docstring):
> "Two-layer formulation prevents the n >> AU idiosyncratic terms from drowning out the systematic factor diversification signal. With a flat (single-layer) entropy over AU + n bins, maximizing H is approximately equivalent to a 1/N objective because the idiosyncratic block has ~12× more terms than the factor block."

### 4.3 Effect on Optimization

- If idiosyncratic contribution is too large (few stocks) → H_idio decreases → optimizer adds stocks
- If idiosyncratic contribution is too small → no problem, rare in practice
- The optimal number of stocks **emerges** from the optimization

### 4.4 Advantages of the Two-Layer Approach

1. **Endogenous**: No arbitrary threshold n ≥ 100
2. **Coherent**: Integrates naturally into the existing framework
3. **Interpretable**: "Idiosyncratic risk is one factor among others"
4. **Flexible**: The `entropy_idio_weight` parameter can weight relative importance

---

## 5. Empirical Calibration of `entropy_idio_weight`

### 5.1 Risk Decomposition Data

**For an individual stock** ([Goyal & Santa-Clara 2001](http://www.econ.yale.edu/~shiller/behfin/2001-05-11/goyal-santa-clara.pdf)):
- 90% idiosyncratic / 10% systematic

**For a diversified portfolio**:

| Diversification Level | % Idio Risk | % Factor Risk |
|-----------------------|-------------|---------------|
| 10 stocks | ~25% | ~75% |
| 20 stocks | ~20% | ~80% |
| 100+ stocks | ~10-15% | ~85-90% |
| FF 3-factor R² | ~10% residual | **>90%** explained |

Source: [Wikipedia - Fama-French](https://en.wikipedia.org/wiki/Fama%E2%80%93French_three-factor_model), [ICFS](https://icfs.com/financial-knowledge-center/systematic-and-unsystematic-risk)

### 5.2 Empirical Conclusion

In a **well-diversified** portfolio (100+ stocks), idiosyncratic risk represents approximately **10-20%** of total risk.

**→ `entropy_idio_weight ≈ 0.15-0.20` reflects this empirical reality.**

The default in `entropy.py` is `idio_weight: float = 0.2` — **exactly** what the literature suggests.

### 5.3 Weight Calibration Table

| Value | H_factor | H_idio | Empirical Justification |
|-------|----------|--------|------------------------|
| 0.0 | 100% | 0% | Ignores idio ❌ |
| 0.05 | 95% | 5% | Under-weights idio |
| 0.10 | 90% | 10% | Very diversified (200+) |
| **0.15** | 85% | 15% | **Typical diversified portfolio** ⭐ |
| **0.20** | 80% | 20% | **Function default** ⭐ |
| 0.30 | 70% | 30% | CAPM level (R²=70%) |
| 0.50+ | <50% | >50% | Not justified |

### 5.4 Final Recommendation

**`entropy_idio_weight = 0.20`** (function default)

**Justification**:
1. **Empirically correct**: ~20% idio risk in diversified portfolios
2. **Function default**: Already calibrated by the original developer
3. **Conservative**: Slightly above minimum (~10%) to encourage diversification

---

## 6. Implementation Status

### 6.1 Implementation Already Exists

The two-layer entropy with idiosyncratic component is **already implemented** in `src/portfolio/entropy.py`.

**Current formulation** (docstring lines 4-16):
```
H(w) = (1 - idio_weight) · H_factor(w) + idio_weight · H_idio(w)

Where:
- H_factor = -Σ_k ĉ'_k ln(ĉ'_k)   (entropy over AU systematic contributions)
- H_idio  = -Σ_i ĉ^ε_i ln(ĉ^ε_i)  (entropy over n idiosyncratic contributions)
```

### 6.2 Current State: Disabled

**Problem**: `entropy_idio_weight = 0.0` in `src/config.py:597`

According to changelog (entry #4, Phase 14):
> `entropy_idio_weight 0.05→0.0`

**Effect**: The idiosyncratic entropy component is **disabled** → only H_factor is used → no incentive to diversify idiosyncratic risk.

### 6.3 Solution: Reactivate

**Single modification required** in `src/config.py`:

```python
# BEFORE (line 597)
entropy_idio_weight: float = 0.0

# AFTER
entropy_idio_weight: float = 0.20  # Function default, empirically justified
```

**No new code needed** — all infrastructure already exists:
- ✅ `compute_entropy_and_gradient()` supports `idio_weight`
- ✅ `compute_entropy_only()` supports `idio_weight`
- ✅ `sca_solver.py` passes `idio_weight` to functions
- ✅ `pipeline.py` passes `entropy_idio_weight` from config
- ✅ `D_eps` (residual variances) is already computed and passed

---

## 7. Comparison: Endogenous vs Fixed Constraints

| Approach | Type | Advantages | Disadvantages | Suited? |
|----------|------|------------|---------------|---------|
| **n ≥ 100** | Fixed constraint | Simple | Arbitrary | ⚠️ Suboptimal |
| **L1/L2 regularization** | Penalty | Endogenous | Encourages sparsity (inverse) | ❌ No |
| **-HHI penalty** | Penalty | Endogenous, convex | Doesn't integrate factors | ✅ Complement |
| **Idio in entropy** | Objective modification | Integrated in framework | Modifies interpretation | ⭐ **RECOMMENDED** |
| **w'D_ε w penalty** | Penalty | Simple | Doesn't force # stocks | ⚠️ Partial |
| **DR (MDP)** | Objective modification | Well-established | Ignores factor structure | ✅ Complement |

---

## Sources

### Diversification & Portfolio Size
- [JRFM (2021) - How Many Stocks Are Sufficient?](https://www.mdpi.com/1911-8074/14/11/551)
- [DeMiguel, Garlappi & Uppal (2009) - Optimal vs Naive Diversification](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1376199)
- [Domian et al. (2007) - 100 Stocks Are Not Enough](https://www.researchgate.net/publication/4744485_Diversification_in_Portfolios_of_Individual_Stocks_100_Stocks_Are_Not_Enough)
- [CFA Institute - Peak Diversification](https://blogs.cfainstitute.org/investor/2021/05/06/peak-diversification-how-many-stocks-best-diversify-an-equity-portfolio/)

### Portfolio Construction Methods
- [Choueifaty & Coignard (2008) - Maximum Diversification](https://www.tobam.fr/wp-content/uploads/2014/12/TOBAM-JoPM-Maximum-Div-2008.pdf)
- [Lopez de Prado (2016) - Hierarchical Risk Parity](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678)
- [Meucci (2009) - Effective Number of Bets](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1358533)

### Regularization & Sparse Portfolios
- [Brodie et al. (2009) - Sparse Portfolios via L1 Regularization](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2169062)
- [Sparse Portfolio via ℓ1/ℓ2 (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0377221724005599)
- [Roncalli - Portfolio Regularization](http://www.thierry-roncalli.com/download/Portfolio_Regularization.pdf)
- [MOSEK Portfolio Cookbook - Factor Models](https://docs.mosek.com/portfolio-cookbook/factormodels.html)

### Concentration & Diversification Indices
- [Portfolio Optimization Using Modified Herfindahl Constraint](https://link.springer.com/chapter/10.1007/978-3-319-61320-8_10)
- [Vermorken, Medda & Schröder (2012) - Diversification Delta](https://digitalcommons.sacredheart.edu/wcob_fac/203/)
- [Effective Number of Stocks - Definition](https://diversification.com/term/effective-number-of-stocks)

### Factor Risk Parity & Idiosyncratic Risk
- [Roncalli - Risk Parity with Risk Factors](http://www.thierry-roncalli.com/download/risk-factor-parity.pdf) — **Key reference for recommended approach**
- [Roncalli (2013) - Introduction to Risk Parity and Budgeting](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2272973)
- [MATLAB - Factor Model Portfolio Optimization](https://www.mathworks.com/help/finance/portfolio-optimization-using-factor-models.html)

### Factor Models
- [Bai & Ng (2002) - Determining Number of Factors](https://sebkrantz.github.io/dfms/reference/ICr.html)
- [CFA Institute - Using Multifactor Models](https://www.cfainstitute.org/en/membership/professional-development/refresher-readings/using-multifactor-models)

### Idiosyncratic Risk
- [Goyal & Santa-Clara (2001) - Idiosyncratic Risk Matters](http://www.econ.yale.edu/~shiller/behfin/2001-05-11/goyal-santa-clara.pdf)

---

## Executive Summary

### The Problem
- Idiosyncratic risk represents **~90% of total risk** for an individual stock
- Current portfolio (30-50 stocks) leaves significant residual variance
- Fixed constraints (n ≥ 100) are **arbitrary and suboptimal**

### The Key Discovery
**The implementation already exists!** `src/portfolio/entropy.py` contains a two-layer formulation (Roncalli) but it is **disabled** (`entropy_idio_weight = 0.0`).

### The Solution
**Reactivate the idiosyncratic component** by changing a single parameter:

```python
# src/config.py:597
entropy_idio_weight: float = 0.20  # was 0.0, function default = 0.2
```

### Implementation Effort
| Aspect | Detail |
|--------|--------|
| Lines of code | **1** (config change) |
| New code | **None** — everything exists |
| Risk | **Low** — existing functionality |

### Advantages of the Existing Approach
1. **Two layers**: Avoids the n >> AU (dimensions) problem
2. **Endogenous**: Optimal stock count emerges from optimization
3. **Validated**: Roncalli (2013) reference in code
4. **Testable**: Test infrastructure already in place
