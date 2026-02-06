# Latent Risk Factor Discovery Strategy

## Methodological Synthesis Document — v4.1

---

## Executive Summary

This document describes a portfolio construction strategy based on **latent risk factor discovery**. The core idea: instead of diversifying across predefined categories (sectors, geographies), we use a neural network to discover the true sources of co-movement between stocks directly from price data, then build a portfolio that is diversified across all of them — including those that are currently dormant.

The model is a **Variational Autoencoder (VAE)** trained on daily return series enriched with realized volatility (two features per time step: z-scored returns and z-scored 21-day rolling volatility). The encoder compresses windows of ~2 years into a low-dimensional latent vector (K≤200 capacity ceiling; the effective number of factors is determined automatically by the VAE's auto-pruning mechanism), capturing each stock's exposure to latent risk factors. A **co-movement loss** with curriculum scheduling (scaffolding → annealing → free refinement) explicitly guides the latent space to reflect empirical correlation structure during early training, then releases the encoder to discover non-linear dependencies. A **crisis weighting** term in the loss function prioritizes the capture of tail-risk patterns that standard training would ignore.

Each stock's complete risk profile is built by aggregating its latent vectors across its full available history — preserving the memory of all risk exposures, including episodic ones (e.g., crisis contagion). The resulting exposure matrix ($n$ stocks × $K$ factors) enables a **factor-based covariance estimation** that is far more stable than direct estimation in asset space.

Portfolio optimization minimizes risk through factorial diversification: the default mode maximizes a combined objective of risk minimization and entropy of principal factor risk contributions ($\max -\lambda w^T\Sigma w + \alpha H(w)$, with no return forecast — $\mu = 0$). The ratio $\alpha/\lambda$ is calibrated at the elbow of the variance-entropy frontier, interpolating between minimum-variance and maximum-diversification. An optional directional mode injects return views from external sources while maintaining factorial diversification. The risk model is retrained monthly to quarterly; between retrainings, the portfolio is stable by design.

**Key properties:** no a priori bias in factor discovery, anti-cyclical by construction (no reactive selling after shocks), modular separation between risk model and return views, low turnover. **Main limitations:** slow adaptation to persistent regime shifts, computational cost at retraining, and latent factor interpretability.

---

## 1. Vision and Objective

### 1.1 Fundamental objective

The goal of this strategy is not to maximize returns through a predictive algorithm, but to **minimize risk through optimal diversification**. This diversification relies on identifying truly independent risk sources, enabling the construction of a portfolio where no exposure is unintentionally duplicated.

This distinction is critical. Return prediction requires the model to be *right about the future* — a task where even the best quantitative hedge funds achieve only marginal edges (Renaissance Technologies reportedly has a ~50.75% win rate). Moreover, many of these return-seeking strategies depend on execution speed that is unattainable for most participants: arbitrage opportunities evaporate within milliseconds to seconds.

Risk identification, by contrast, requires the model to be *right about the present structure* of dependencies between assets. This is a fundamentally more tractable problem: we are not forecasting what will happen, but understanding what is currently connected and why. That said, the risk structure of the market is not static — it evolves constantly as macroeconomic conditions shift, new sectors emerge, and market regimes change. The strategy addresses this through **periodic retraining** of the model to incorporate new data and **historical inference** that builds a complete risk profile for each stock by aggregating its entire observable history.

### 1.2 Starting observation

Traditional diversification methods (by sector, geography, asset class) rely on categories defined *a priori* by humans. However, these categories do not necessarily capture the entire sources of asset co-movement. The consequence is that an investor who believes they are diversified — because they hold stocks across 10 sectors and 5 geographies — may in reality be heavily concentrated on a single latent risk factor (e.g., global growth sensitivity) that cuts across all these categories. When that factor turns, the entire portfolio moves together, and the supposed diversification provides no protection.

### 1.3 Ambition

Let the data reveal the underlying risk structures themselves, without imposing prior categorization. The discovered risk factors will not be named *a priori* but identified by the model, then optionally interpreted *a posteriori* if desired.

This approach inverts the traditional workflow. Instead of starting from economic theory ("these are the risk factors that matter") and then measuring exposures, we start from observed price behavior ("these assets move together") and then ask what economic reality might explain the discovered structure. The model is free to discover risk factors that no analyst would have thought to define — factors that may not correspond to any standard classification but that genuinely drive portfolio risk.

---

## 2. The Nature of Risk

Before designing the model, we need to clarify what it must capture. This section examines the structure of financial risk — its multiple dimensions, its graph-like topology, and its temporal dynamics — to establish the requirements that the architecture must satisfy.

### 2.1 Multiplicity of risk sources

Risk is not a one-dimensional concept. There potentially exists an infinity of risk sources operating at different scales and with different impacts:

| Dimension | Examples |
|-----------|----------|
| **Geographic** | Exposure to China, emerging markets, eurozone |
| **Sectoral** | Cyclicals vs defensives, tech vs utilities |
| **Temporal** | Short-term risks (liquidity) vs long-term (demographics) |
| **Systemic** | Exposure to global market, economic cycle |
| **Non-linear** | Tail risks, asymmetric behavior during stress periods |
| **Structural** | Financial leverage, short-term funding dependency |

### 2.2 Risk as a graph

A natural first intuition is to picture this structure as a **tree**: global economic risk at the root, splitting into broad categories (cyclical vs defensive), then into finer sub-risks (sector, sub-sector, supply chain). But a tree forces each stock into **exactly one branch** at each level. In reality, a company like LVMH is simultaneously exposed to the Chinese consumer, the business cycle, and the EUR/USD rate — exposures that cut across any tree.

The true structure is closer to a **graph**: each stock sits at the intersection of multiple risk dimensions, with **continuous** degrees of exposure rather than binary membership. This implies that the appropriate representation must allow multiple simultaneous memberships with varying intensities — which is what a continuous latent space provides.

### 2.3 Temporality of risk factors

Not all risks are active at all times. They can be classified into three categories with distinct temporal behaviors:

1. **Structural factors** (industry, geography, supply chain): persistent by nature. A pharmaceutical company's exposure to drug regulation risk is a permanent feature of the business. Barra/MSCI risk models report factor stability coefficients above 0.90 for industry exposures (USE4 methodology). These factors are continuously visible in prices.

2. **Style factors** (leverage, size, momentum, volatility): evolve as company characteristics change, but slowly. Academic studies show half-lives exceeding 25 months for value exposures. A company may deleverage or grow in market cap, but the transition unfolds over years, not days.

3. **Episodic factors** (crisis dynamics, tail contagion): dormant for extended periods, then suddenly dominant. Interbank contagion (2008), sovereign debt spirals (2011), pandemic supply chain disruption (2020). These factors drive the most destructive portfolio drawdowns precisely because they activate when correlations spike across assets that appeared uncorrelated.

The boundary between active and dormant is not binary. Semi-dormant factors may leave residual traces in price behavior (tail co-movements, similar micro-reactions to policy announcements). In a continuous latent space, these produce small but non-zero exposures. Slow-moving risks (demographics, secular trends) operate on cycles of 10-20 years and are not actionable at portfolio rebalancing frequency.

The existence of dormant factors has a direct consequence for the model design: a system that diversifies only against currently observable correlations is structurally blind to episodic risks, and will react to their reactivation only after the damage is done — selling correlated assets after a shock rather than being protected before it. The model therefore builds a **complete historical risk profile** for each stock, covering all factors ever observed — structural, stylistic, and episodic alike.

### 2.4 Risk definitions

Several risk definitions were considered:

1. **Classical volatility**: standard deviation of returns (symmetric, penalizes gains as well as losses)
2. **Downside measures**: semi-variance, VaR (Value at Risk), CVaR (Conditional Value at Risk) — focused on losses
3. **Factor exposure**: sensitivity to systematic factors (multidimensional view)
4. **Conditional behavior**: correlations during stress periods, tail dependence

For our diversification objective, what matters is the **effective correlation of assets**, regardless of its origin. The relevant risk is that which materializes in price co-movements.

---

## 3. Price Data vs Fundamental Data

Given these requirements — continuous multi-dimensional exposures, memory of dormant factors, sensitivity to tail co-movements — what data should the model consume? The choice between price data and fundamental data has deep consequences for what the model can discover.

### 3.1 The argument for price data

If one adheres to a form of market efficiency, prices already aggregate all relevant information. Balance sheets, macro data, announcements: everything is "digested" by the market and reflected in prices.

**Advantages:**
- High-frequency availability (daily, even intraday)
- Standardization and comparability across assets
- No publication delay or revision issues
- Captures risks as they actually manifest

**What prices capture that fundamentals don't see:**
- Investment flows (sector ETFs — Exchange-Traded Funds — creating mechanical correlations)
- Analyst behavior (correlated views by sector)
- Sentiment and narrative effects
- Market structure itself (market making, arbitrage)

### 3.2 The argument for fundamental data

Past correlations do not guarantee future correlations. Fundamental data can identify common vulnerabilities *before* they materialize.

**Example:** Two companies with no historical correlation but with high leverage and short-term funding dependency will be hit simultaneously during a credit tightening.

**Advantages:**
- *Ex ante* risk identification
- Characteristic stability (less noisy than correlations)
- Economic interpretability

### 3.3 The limitations of fundamental data

**Frequency problem:** Balance sheets are quarterly, macro data monthly at best. Over 20 years, this represents only 80 quarterly observations, insufficient to train a complex model.

**Accessibility problem:** Quality fundamental data is expensive and less standardized than price data.

**Bias problem:** Using fundamentals as a *prior* risks biasing the model toward structures we injected (sectors, geography, leverage...), at the expense of truly latent factors.

**Coverage problem:** Two tech companies with radically different balance sheets will nonetheless be correlated because the market perceives them as "tech." This real correlation does not appear in balance sheets but is perfectly visible in prices.

### 3.4 Resolution: prices for discovery, fundamentals for interpretation

> **Price data is used for latent factor discovery.**
> **Fundamental data is used only for *a posteriori* interpretation.**

This approach:
- Preserves discovery capability (no *a priori* bias)
- Captures risks as they actually manifest (including non-fundamental "market" risks)
- Remains data-parsimonious
- Subsequently allows understanding what the factors represent economically

---


## 4. Architecture

### 4.1 Model and pipeline overview

#### Latent factor model

Observed price movements (high-dimensional: one series per stock) are driven by a much smaller number of underlying, non-directly observable sources of variation: the **latent factors**. The return of stock *i* at time *t*:

$$r_{i,t} = f(z_{1,t}, z_{2,t}, ..., z_{K,t}, \beta_{i,1}, \beta_{i,2}, ..., \beta_{i,K}) + \varepsilon_{i,t}$$

Where $z_{k,t}$ is the realization of latent factor *k* at time *t*, $\beta_{i,k}$ the exposure of stock *i* to factor *k*, $\varepsilon_{i,t}$ the idiosyncratic noise, and $f$ a potentially non-linear function. The model simultaneously estimates the K latent factors, each stock's exposures, and the dependence structure induced by these common exposures.

#### Why a Variational Autoencoder (VAE)

The strategy requires estimating each stock's exposure vector $\beta_i$ to latent risk factors. A **Variational Autoencoder (VAE)** solves this naturally.

Unlike a standard autoencoder (which maps inputs to fixed points), a VAE maps each input to a **probability distribution** (mean and variance) in latent space, then samples during training. This forces the latent space to be **continuous and smooth** — nearby points correspond to similar return dynamics, with no empty gaps. Distances between stocks are therefore meaningful and comparable, which is exactly what portfolio construction requires.

The VAE has three components:

- **The encoder** compresses a window of T daily returns (enriched with realized volatility — 2 features per time step) into parameters of a distribution in K-dimensional latent space. The bottleneck (K ≪ T) forces discarding of idiosyncratic noise, retaining only dominant structures of variation.

- **The decoder** reconstructs the original series from a sampled latent vector, serving as a training signal. It is discarded after training — only the encoder is used at inference (producing a deterministic mean vector).

- **The co-movement loss** (default, with curriculum scheduling) adds an explicit relational constraint during early training: stocks with correlated returns should be mapped to nearby latent points. This scaffolding is progressively removed, allowing the encoder to discover non-linear dependencies beyond what correlation-based guidance can capture (Section 4.4).

Once trained, the encoder maps return windows to latent vectors — these are the estimated exposures $\beta_i$.

#### End-to-end pipeline

```
For each stock:
  Full price history
    → daily log returns + 21-day realized volatility (2 features per time step)
    → sliding windows of T days (tensor T × 2, z-scored per feature per window)
    → each window through the encoder → one local latent vector per window
    → aggregation across all windows → one composite risk profile per stock

For the portfolio:
  1. Stack the n composite profiles into the exposure matrix B (n × K)
  2. Filter to active dimensions: B_A = B[:, A] where A = {k : KL_k > 0.01}
  3. For factor estimation (historical): rescale per date:
       B̃_A,i,t = (σ_i,t / σ̄_t) · B_A,i  (trailing vol at date t / cross-sectional median at t)
  4. Recover factor realizations: ẑ_t = (B̃_A,t^T B̃_A,t)^{-1} B̃_A,t^T r_t
  5. Estimate factor covariance Σ_z (AU × AU) and idiosyncratic variances D_ε
  6. For portfolio construction: rescale at current date:
       B̃_A,i^port = (σ_i,now / σ̄_now) · B_A,i
  7. Diagonalize: Σ_z = VΛV^T, rotate B̃'_A = B̃_A^port·V (principal factor basis)
  8. Derive the asset covariance: Σ = B̃'_A·Λ·B̃'_A^T + D_ε
  9. Optimize portfolio weights (entropy on principal factors)
```

The following sections detail each component: data (4.2), encoder-decoder (4.3), loss function (4.4), training (4.5), inference and aggregation (4.6), and portfolio construction (4.7).

### 4.2 Data

#### Data type and frequency

**Daily logarithmic return time series** for each stock:

$$r_{i,t} = \ln\left(\frac{P_{i,t}}{P_{i,t-1}}\right)$$

Where $P_{i,t}$ is the adjusted price (dividends and splits) of stock *i* on day *t*.

Data frequency acts as a filter determining which risks the model can see. Daily data is chosen because it maximizes training volume (critical for K up to 200), captures crisis dynamics where diversification matters most, and provides granularity to distinguish risk regimes. The Epps effect (correlations increase at lower frequencies) means daily noise masks underlying co-movements — a neural encoder can learn to look through this noise given sufficient data.

| Frequency | Obs / year | Obs / 30 years | Captures | Limitation |
|-----------|-----------|-----------------|----------|------------|
| **Daily** | ~252 | ~7,560 | Short-term dynamics, crisis contagion, liquidity | More microstructure noise |
| **Weekly** | ~52 | ~1,560 | Cleaner correlations, medium-term | Loses intra-week crisis dynamics, 5× fewer obs |
| **Monthly** | ~12 | ~360 | Macro exposures | Far too few for K≤200, no crisis granularity |

Additional noise is managed through per-window z-score normalization and latent space regularization — idiosyncratic noise does not produce correlated behavior across stocks and is therefore not captured as a shared pattern.

Raw time series are used rather than pre-calculated correlations, which would discard temporal information and limit the model to average pairwise correlation. Raw series let the model discover *which* statistics matter — conditional correlations, volatility synchronization, tail dependence.

#### Preprocessing

1. **Per-stock normalization**: z-score standardization (mean = 0, std = 1) for each series, computed **per-window**. For a given stock *i* and window $[t - T + 1, \; t]$:

$$\tilde{r}_{i,\tau} = \frac{r_{i,\tau} - \hat{\mu}_i^{(t)}}{\hat{\sigma}_i^{(t)}}, \qquad \tau \in [t - T + 1, \; t]$$

where $\hat{\mu}_i^{(t)}$ and $\hat{\sigma}_i^{(t)}$ are the sample mean and standard deviation of stock *i*'s raw log-returns *within that window only*.

**Why per-window rather than expanding or full-history?**

| Method | Description | Look-ahead? | Issue for this model |
|--------|-------------|-------------|---------------------|
| **Per-window** (chosen) | μ, σ from the T days of the window | No | — |
| Expanding (point-in-time) | μ, σ from all data up to date *t* | No | Early windows have unstable statistics (few observations); late windows dilute recent regime changes. A stock whose volatility tripled recently will still show moderate normalized returns — the encoder cannot distinguish a calm stock from a newly volatile one. |
| Full history | μ, σ from the stock's entire available history | **Yes** | Uses future data. Unacceptable for backtesting. |

Per-window normalization is the natural choice for an encoder that processes each window independently. It forces the model to learn *structural patterns* (volatility clustering, mean-reversion shapes, co-movement signatures) rather than absolute return levels — which is precisely what a risk factor model requires. Two windows with identical shapes but different raw volatilities will produce identical latent vectors, which is desirable: they represent the same risk *structure*. However, per-window normalization erases **volatility dynamics** — the encoder cannot distinguish a stock whose volatility doubled from one that remained calm, nor can it detect cross-sectional co-movements in variance (volatility clustering across stocks), which constitute a systematic risk factor in their own right. This information is restored by enriching the input with realized volatility (see below).

**Edge case: $\hat{\sigma}_i^{(t)} \approx 0$.** Extremely low-volatility windows (e.g., a stock suspended for most of the window) produce near-zero denominators. Clamp: $\hat{\sigma}_i^{(t)} \geq \sigma_{\min} = 10^{-8}$. If more than 20% of days in a window have zero return (suspension), exclude the window entirely.

2. **Realized volatility feature**: the second input feature is the rolling 21-day standard deviation of raw returns, z-scored per window:

$$v_{i,\tau} = \text{std}\!\left(r_{i,\tau-20}, \ldots, r_{i,\tau}\right), \qquad \tilde{v}_{i,\tau} = \frac{v_{i,\tau} - \hat{\mu}_{v,i}^{(t)}}{\hat{\sigma}_{v,i}^{(t)}}$$

where $\hat{\mu}_{v,i}^{(t)}$ and $\hat{\sigma}_{v,i}^{(t)}$ are the mean and standard deviation of $v_{i,\tau}$ *within the window* $[t-T+1, t]$. The z-scoring is separate from the return z-scoring — each feature is normalized independently.

**Why realized volatility is not redundant with returns.** The z-scored return preserves *temporal shape* (autocorrelation, asymmetry, clustering patterns) but erases the absolute volatility scale. The 21-day rolling volatility captures *volatility dynamics* — clustering, cross-stock co-movements in variance, and local regime transitions. These are complementary: a stock with identical return shapes but transitioning from 10% to 40% annualized volatility carries fundamentally different risk, invisible to the return channel alone.

**Warm-up requirement.** The 21-day rolling window for $v_{i,\tau}$ must be computed from returns *preceding* the start of the encoder window to avoid look-ahead. In practice, the first 21 days of each encoder window use a volatility estimate based partly on pre-window returns — this is handled by extending the price history requirement by 21 days.

The input tensor per window is therefore $(T, 2)$: $[\tilde{r}_{i,\tau}, \; \tilde{v}_{i,\tau}]$.

**Architectural impact.** The encoder is parameterized by $F$ (in_channels = $F$). Passing from $F = 1$ to $F = 2$ modifies only the first convolutional layer (Inception head) and the decoder's output layer — approximately 0.1% of total parameters. The capacity-data constraint, the sizing rules, and the entire downstream pipeline (latent vector $\mu_i \in \mathbb{R}^K$, aggregation, $B$, $\Sigma_z$, optimization) are strictly unchanged.

3. **Missing value handling**: rare on the target universe (large-cap US equities on CRSP), but when encountered: forward-fill the price (equivalent to imputing a 0 return) for isolated gaps ≤ 5 consecutive days; exclude the window if a gap exceeds 5 consecutive days (temporal continuity broken over the shortest Inception kernel scale).
4. **Windowing**: splitting into fixed-length T temporal windows

#### Investment universe

The universe defines *which stocks* enter the pipeline. An incorrectly constructed universe silently invalidates the backtest through **survivorship bias** — including only stocks that exist today, excluding those that disappeared (bankruptcy, delisting, acquisition). The effect is systematic: excluded stocks are overwhelmingly losers. Shumway (1997) documented missing delisting returns averaging -30% for NYSE/AMEX and -55% for Nasdaq. Boynton & Oppenheimer (2006) estimate over 40% of the reported size premium is attributable to this bias. For this strategy, the bias also affects **encoder training**: excluding failed stocks deprives the encoder of tail-risk patterns that crisis weighting γ (Section 4.4) is designed to capture.

**Point-in-time principle.** The universe is reconstructed as it existed at each historical date. At reconstitution date $t$, $\mathcal{U}_t$ contains exactly the stocks satisfying eligibility criteria *at that date*. If delisted without an available return, a corrective return is imputed (Shumway convention: -30% NYSE/AMEX, -55% Nasdaq).

**Eligibility criteria** (drawing on MSCI GIMI, CRSP, Russell, Fama-French, Gu-Kelly-Xiu methodologies):

1. **Minimum market capitalization.** Entry: float-adjusted cap ≥ \$500M. Exit: \$400M (20% buffer for stability). Excludes micro-caps where transaction costs exceed model assumptions (spreads 50–200 bps vs ~10 bps for large-caps). For a broader universe, threshold can be lowered to ~\$200M.

2. **Minimum liquidity.** ADV ≥ \$2M over trailing 3 months. A filter, not a ranking criterion — ADV is too volatile and procyclical to use for selection.

3. **Minimum listing history.** T = 504 trading days (~2 years) of continuous listing. This matches the encoder's window length — no position is taken on a stock whose risk profile is extrapolated from incomplete data. Cost: ~30–60 recent IPOs temporarily excluded (~2–3% of eligible stocks).

4. **Security type.** Common equities only. Exclusions: ETFs, ADRs (unless primary market inaccessible), REITs (optional), preferred shares, warrants, SPACs pre-merger. *Multi-asset extension (future phase):* the VAE architecture is asset-agnostic and can extend to bonds, commodities, etc.

5. **Listing market.** NYSE + NASDAQ + AMEX (US equities). *Geographic expansion planned* with challenges: asynchronous trading hours, currency effects, variable data quality (Ince & Porter, 2006).

**Selection when universe exceeds target size.** If more than $n$ stocks qualify, select the **top $n$ by float-adjusted market capitalization** — stable, economically representative, and consistent with index methodologies.

**Training universe vs investment universe.** The **investment universe** at date $t$ is $\mathcal{U}_t$. The **training universe** for a walk-forward fold is the union of all stocks ever in $\mathcal{U}_{t'}$ for any $t'$ in the training period — including since-delisted stocks. A stock that failed in 2008 contributes pre-failure windows. If training windows exceed capacity, **windows** are subsampled (stride 5 or 21), not stocks.

**Reconstitution and delisting handling.** Reconstitution at each rebalancing (monthly to quarterly). Between reconstitutions:
- Delisted positions are liquidated at the last available price (or imputed return). Freed capital held in cash.
- No full rebalancing triggered by a single delisting — **except** if factorial entropy drops below threshold:

$$H(w^{\text{post-delisting}}) < \alpha_{\text{trigger}} \times H(w^{\text{last\;rebalancing}})$$

with $\alpha_{\text{trigger}} = 0.90$. This captures the case where the delisted stock provided a unique factor exposure. The exceptional rebalancing is subject to all standard constraints ($P_{\text{turn}}$, $\tau_{\max}^{\text{hard}}$, $P_{\text{conc}}$).

**Data source.** CRSP is the academic gold standard for survivorship-bias-free US equity data. Alternatives: EODHD (~\$80/month, 26,000+ delisted tickers since 2000), Norgate Data (~\$150–300/month). Free sources (Yahoo Finance) do not reliably include delisted stocks.

### 4.3 Encoder-Decoder

#### Window length T

T is the length of the fixed-length sequence the encoder processes — an **architectural constraint**, not a choice about how much history to consider. A stock with 30 years of data is divided into overlapping windows of T days, each producing one local latent vector, then aggregated (Section 4.6).

- **T too small (e.g., 20 days):** captures only one market context; cannot distinguish behavior from noise.
- **T too large (e.g., 5 years):** spans structural regime changes; blends distinct realities.
- **T = 504 days (~2 years):** long enough for several market conditions (calm, crisis, recovery), short enough for a coherent risk regime.

If the base model doesn't capture enough structure, the input can be further enriched with additional features (normalized volume, sector deviation), producing a tensor of dimension (T, F) with F > 2. The baseline uses F = 2 (returns + realized volatility — see Section 4.2).

#### Encoder architecture type

The encoder maps a T-day return sequence to a K-dimensional latent vector, learning pattern detectors (volatility signatures, mean-reversion profiles, market sensitivity). Because it operates solely on return dynamics, it **generalizes across stocks** — a new stock can be processed without retraining.

**1D CNN with multi-scale kernels and stride (retained).** Three properties make this well-suited:

- **Translation invariance**: the same kernel recognizes a volatility cluster whether it occurs at day 10 or day 200 — the risk profile should reflect *what* patterns occur, not *when*.
- **Multi-scale extraction**: different kernel sizes (InceptionTime head: ~1 week, ~1 month, ~1 quarter) capture temporal granularities simultaneously, mirroring the multi-scale nature of risk factors.
- **Parallelization**: all positions processed simultaneously. CNN encoders converge in ~60% fewer epochs than CNN-LSTM baselines (VAR-VAE, 2025).

**Temporal bias note.** Strided convolutions and MSE loss both favor low-frequency patterns (Frequency Principle, Rahaman et al., 2019). For this application, the bias is **functionally aligned**: relevant risk factors are medium- to long-duration. Crisis weighting γ (Section 4.4) captures short-lived systemic events separately.

**Alternatives rejected.** (1) **WaveNet** (dilated convolutions + skip connections): does not eliminate the MSE spectral bias (a property of gradient descent, not topology); substantial overhead for unguaranteed gain. (2) **LSTM/GRU**: optimized for sequential prediction, not static profiling; non-parallelizable. (3) **Transformer**: requires positional encoding (reintroduces position dependence), higher parameter count increasing overfitting risk.

#### Encoder specification

The architecture is fully determined by **5 variable parameters** (n, T, T_année, F, K); all other hyperparameters are fixed. The baseline uses $F = 2$ (returns + realized volatility — Section 4.2).

**Fixed hyperparameters:**

| Symbol | Value | Component | Justification |
|--------|-------|-----------|---------------|
| $k_{\text{head}}$ | (5, 21, 63) | Inception head kernels | InceptionTime [7]: ~1 week / ~1 month / ~1 quarter |
| $C_{\text{branch}}$ | 48 | Filters per branch | InceptionTime default; $C_{\text{head}} = 3 \times 48 = 144$ |
| $k_{\text{body}}$ | 7 | Body kernel size | Standard 1D ResNet [3] |
| $\text{stride}_{\text{conv}}$ | 2 | Per-block downsampling | Temporal dimension halved per block |
| $\alpha_{\text{proj}}$ | 1.3 | Projection ratio | Ensures $C_L > 2K$ (compression, not expansion) |
| $C_{\min}$ | 384 | Minimum final width | Avoids capacity starvation for small K |
| Activation | GELU | All layers | Smoother than ReLU [9] |
| Normalization | BatchNorm1d | After every conv | Stabilizes training across input scales |
| Dropout | 0.1 | After each residual block | Moderate regularization [8] |
| $\beta$ | 1.0 (fixed) | KL weight | Fixed at 1.0 when $\sigma^2$ is learned (primary mode) — the recon/KL balance is handled by $\sigma^2$. See Section 4.4 for alternative modes. |
| $\sigma^2_0$ | 1.0 (learned) | Observation noise init | Learned scalar; clamped to $[10^{-4}, 10]$ [12, 13]. Mutually exclusive with KL annealing (see Section 4.4). |
| Weight decay | $10^{-5}$ | L2 on all weights | Standard for CNN training |

**Sizing rule 1 — Depth L (from T).** Ensures the receptive field covers the full input:

$$L(T) = \max\!\left(3,\; \left\lceil \log_2\!\left(\frac{T}{k_{\max}}\right) \right\rceil + 2 \right)$$

Concrete: $T = 252 \Rightarrow L = 4$, $T = 504 \Rightarrow L = 5$, $T = 756 \Rightarrow L = 6$.

**Sizing rule 2 — Width $C_L$ (from K).** Final layer before Global Average Pooling must project in compression to $2K$ outputs ($\mu$ and $\log\sigma^2$):

$$C_L(K) = \max\!\left(C_{\min},\; \left\lceil \alpha_{\text{proj}} \times 2K \right\rceil \right)$$

For $K \leq 147$, the floor $C_{\min} = 384$ binds. Beyond that, $C_L$ scales linearly with K.

**Sizing rule 3 — Channel progression.** Intermediate blocks interpolate geometrically:

$$C_l = \text{round}_{16}\!\left( C_{\text{head}} \times \left(\frac{C_L}{C_{\text{head}}}\right)^{l/L} \right), \quad l = 1, \ldots, L$$

Example for $K = 100$, $L = 5$: $144 \to 176 \to 208 \to 256 \to 320 \to 384$.

**Parameter count.** Each residual block: two convolutions, batch normalizations, 1×1 skip projection (always active since stride = 2):

$$P_{\text{enc}} = P_{\text{head}} + \sum_{l=1}^{L} P_{\text{block}}(C_{l-1}, C_l) + P_{\text{proj}}(C_L, K)$$

Full VAE: $P_{\text{total}} \approx 2 \times P_{\text{enc}}$ (decoder slightly larger — see below). Dominant scaling: $O(K^2 \times L)$ with $L \sim O(\log T)$.

#### Capacity-data constraint

$$\frac{P_{\text{total}}(T, K)}{N(n, T, T_{\text{hist}})} \leq r_{\max}$$

where $N = n \times (T_{\text{hist}} - T + 1)$ is the **maximum** number of windows (stride $s = 1$), $T_{\text{hist}} = T_{\text{année}} \times 252$. The ratio $r$ is always computed at $s = 1$ regardless of the actual training stride $s_{\text{train}}$, because stride subsamples redundant windows (consecutive windows share $T-1$ days) without reducing the dataset's informational content — the same risk regimes and co-movement patterns are present. Overfitting protection comes from the KL (auto-pruning), dropout, weight decay, early stopping, and walk-forward OOS evaluation, not from window count. The actual training dataset size is $N_{\text{train}} = n \times \lfloor(T_{\text{hist}} - T) / s_{\text{train}}\rfloor + n$ — reported by `build_vae` for computational planning.

**Literature context for $r_{\max}$.** Observed values: $r \approx 0.05$ for compute-optimal LLMs [1]; $r \approx 0.001$ for small FC networks on financial features [2]; $r \approx 1$ effective for ResNets on ImageNet (after augmentation/weight sharing) [3]. Double descent [4] warns $r \in [0.5, 2]$ is dangerous. The VAE's KL term [5, 6] reduces $P_{\text{eff}} \ll P$ by collapsing unused dimensions.

**Threshold: $r_{\max} = 5$ (baseline), extensible to 10 with reinforced regularization** (dropout ≥ 0.2 + weight decay + $\beta > 1$). Above the double descent zone; conservative vs ImageNet; justified vs Gu et al. by convolutional weight sharing.

**Feasibility table** ($n = 1000$, $T_{\text{année}} = 30$, $F = 2$, $s = 1$):

| K | T | L | Channels | $P_{\text{total}}$ | N | $r$ | $r \leq 5$? |
|---|---|---|----------|-------------------|---|-----|-------------|
| 100 | 504 | 5 | 144→176→208→256→320→384 | 10.4M | 7.06M | 1.48 | ✓ |
| 150 | 504 | 5 | 144→176→208→256→320→384 | 10.8M | 7.06M | 1.52 | ✓ |
| 200 | 504 | 5 | 144→192→240→304→400→512 | 16.3M | 7.06M | 2.31 | ✓ |
| 100 | 756 | 6 | 144→176→192→240→272→320→384 | 12.1M | 6.80M | 1.77 | ✓ |
| 200 | 756 | 6 | 144→176→224→272→336→416→512 | 18.7M | 6.80M | 2.76 | ✓ |

**Levers when violated:** expand $n$ (doubles $N$), extend $T_{\text{année}}$, cap $C_L$ at 384, raise $r_{\max}$ to 10 with reinforced regularization.

The complete construction is implemented in `build_vae.py` (Appendix A).

#### Decoder

The decoder has the **same block structure** as the encoder (channel progression reversed, transposed convolutions for upsampling) but its intermediate temporal dimensions do not match the encoder's exactly:

1. **Initial projection.** Linear layer: $z \to (C_L, T_{\text{comp}})$, where $T_{\text{comp}}$ is the encoder's last residual block output temporal dimension. For $T = 504$, $L = 5$: $T_{\text{comp}} = 16$ (encoder sequence: $504 \to 252 \to 126 \to 63 \to 32 \to 16$). Contains $K \times C_L \times T_{\text{comp}}$ parameters — the main source of encoder/decoder asymmetry.

2. **Transposed residual body.** $L$ blocks with `ConvTranspose1d` (stride 2, `output_padding=1`), channel progression reversed: $C_L \to C_{L-1} \to \ldots \to 144$. Same block structure (two convolutions + skip + BatchNorm + GELU + Dropout). Each transposed block doubles the temporal dimension exactly ($L_{\text{out}} = 2 \times L_{\text{in}}$), whereas the encoder's stride-2 convolutions use $\lfloor(L_{\text{in}}-1)/2\rfloor + 1$, which rounds up on odd inputs. Result for $T = 504$: decoder produces $16 \to 32 \to 64 \to 128 \to 256 \to 512$ vs encoder's $16 \to 32 \to 63 \to 126 \to 252 \to 504$. Discrepancies of 1–8 elements appear at intermediate levels but are harmless: without skip connections, each block operates independently on its input tensor. Final trim/pad to exact target length $T$.

3. **Output head.** $1 \times 1$ convolution from $C_{\text{head}} = 144$ to $F$ features.

**Note on skip connections.** If encoder→decoder skip connections (U-Net style) were added to improve reconstruction quality, the intermediate size mismatches would cause dimension errors at concatenation. The fix: replace `output_padding=1` with a per-block adaptive `output_padding` computed from the encoder's recorded temporal sizes, or pad/trim after each transposed block to match. This is not needed for the current architecture.

**Parameter count:** $P_{\text{dec}} \approx P_{\text{enc}} + K \times C_L \times (T_{\text{comp}} - 2)$. Exact counts in `build_vae.py`.

#### Latent space dimension (K)

K is a **capacity ceiling**, not the number of factors the model will use. This stems from the VAE's **auto-pruning** property: the KL divergence pushes each unused dimension toward the prior $\mathcal{N}(0, 1)$, effectively deactivating it. Dimensions with marginal KL > 0.01 nats are **active units (AU)**. This is confirmed by TempVAE on financial time series (Sicks et al., 2023 [26]).

A VAE with $K = 200$ on data supporting 80 factors converges to $\text{AU} \approx 80$ with ~120 inactive dimensions. Unlike PCA (where every component captures additional variance), superfluous VAE dimensions deactivate themselves. K should therefore be set **generously** — the effective factor count is read as AU after training.

**K selection — two phases:**

**Phase 1 — Calibration (single exploratory run).** Train with $K_{\max} = 200$ on the first walk-forward fold. Observe:

1. **Active units (AU).** If AU ≈ 180–200: increase $K_{\max}$. If AU ≈ 60–100: $K_{\max}$ is adequate.
2. **KL profile.** Sort marginal KL per dimension. A sharp elbow indicates well-defined factor count; gradual decline suggests a continuous spectrum.
3. **Factor explanatory power.** $1 - \text{tr}(D_\varepsilon) / \text{tr}(\Sigma_{\text{assets}})$. Target > 0.60. If < 0.50 with AU ≈ $K_{\max}$, the encoder lacks capacity.

**PCA sanity check.** Bai & Ng (2002) [27] information criteria on the return matrix provide a floor: the VAE must capture at least as many factors as PCA (typically 5–15 for ~1000 stocks monthly).

**Phase 2 — Walk-forward validation.** Test 2–3 values informed by Phase 1: $K \in \{\text{AU}, 1.5 \times \text{AU}, 2 \times \text{AU}\}$. Selection by OOS score (Section 4.8). Elimination: AU < 0.5K on most folds (excessive K) or explanatory power < 0.50 (insufficient K).

**Baseline:** $K = 200$. Effective factors = AU ≤ K, observed post-training. K serves three functions: (1) **exploration space** for auto-pruning (the VAE needs K ≥ 1.5–2× the intrinsic dimensionality to auto-prune reliably — Burda et al. 2015, Sicks et al. 2023); (2) **noise absorption buffer** (dimensions beyond AU collapse to the prior and absorb non-reproducible patterns, preventing contamination of active dimensions — with K = 200 and AU ≈ 85, ~115 dimensions serve as buffer); (3) **diagnostic signal** (if AU converges near K, the true dimensionality exceeds capacity — invisible with a tight K).

**Statistical constraint on AU.** The downstream factor covariance $\Sigma_z$ ($\text{AU} \times \text{AU}$) has $\text{AU}(\text{AU}+1)/2$ free parameters. Reliable estimation requires a minimum ratio of observations to parameters. Define:

$$\text{AU}_{\max}^{\text{stat}} = \left\lfloor \frac{-1 + \sqrt{1 + 4 \cdot N_{\text{obs}} / r_{\min}}}{2} \right\rfloor$$

where $N_{\text{obs}} = T_{\text{année}} \times f_{\text{freq}} - T_{\text{vol}} + 1$ (number of daily dates available for cross-sectional OLS), $f_{\text{freq}} = 252$ for daily data, $T_{\text{vol}} = 252$ (trailing volatility warm-up), and $r_{\min} = 2.0$ (minimum observations-to-parameters ratio for Ledoit-Wolf shrinkage to remain moderate). For the baseline (30 years, daily): $N_{\text{obs}} \approx 7{,}309$, yielding $\text{AU}_{\max}^{\text{stat}} \approx 85$.

| Historical depth | Frequency | $N_{\text{obs}}$ | $\text{AU}_{\max}^{\text{stat}}$ ($r_{\min}=2$) |
|---|---|---|---|
| 30 years | Daily | ~7,309 | **85** |
| 25 years | Daily | ~6,049 | 77 |
| 20 years | Daily | ~4,789 | 69 |
| 15 years | Daily | ~3,529 | 59 |
| 30 years | Weekly | ~1,309 | 36 |

If $\text{AU} > \text{AU}_{\max}^{\text{stat}}$ after training, truncate to the $\text{AU}_{\max}^{\text{stat}}$ most active dimensions (ranked by descending marginal KL). The truncated dimensions still contributed to learning a better latent space — the constraint applies only to the downstream covariance estimation pipeline. This guard is the **binding constraint** in practice, not K itself.

### 4.4 Loss function

The loss function determines what the latent space captures. It is built in layers: a minimal baseline, then targeted extensions.

#### Baseline: reconstruction + regularization

The **reconstruction loss** (MSE between original and reconstructed series) is the foundation. Explicitly:

$$\mathcal{L}_{\text{recon}} = \frac{1}{T \cdot F} \sum_{t=1}^{T} \sum_{f=1}^{F} \left(\tilde{r}_{t,f} - \hat{r}_{t,f}\right)^2$$

where $\tilde{r}_{t,f}$ are the per-window normalized returns (Section 4.2) and $\hat{r}_{t,f}$ the decoder's reconstruction. This is a **mean** (not a sum) over the $D = T \times F$ elements of each window — so $\sigma^2$ (below) represents the per-element residual variance, independent of $T$ and $F$, directly interpretable on normalized data (variance ~1), and comparable across configurations with different window lengths or feature counts.

The bottleneck (K ≪ T) forces the encoder to discard idiosyncratic noise and retain dominant variation structures. Because the encoder is the same network applied to every stock, similar return dynamics naturally produce similar latent vectors — an **implicit correlation capture** mechanism.

**KL divergence regularization** prevents degenerate encodings. For each stock *i*, the encoder outputs $\mu_i$ and $\sigma_i^2$ in each of K dimensions. The KL measures deviation from $\mathcal{N}(0, 1)$:

$$\mathcal{L}_{\text{KL}} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{2} \sum_{k=1}^{K} \left(\mu_{i,k}^2 + \sigma_{i,k}^2 - \ln \sigma_{i,k}^2 - 1\right)$$

This creates a fundamental **tension**: KL pushes all stocks toward the center; reconstruction pulls them apart. The balance determines latent space quality — too much reconstruction → arbitrary distances; too much KL → posterior collapse (all stocks indistinguishable).

**Balancing via learned observation noise $\sigma^2$.** Under a Gaussian observation model $p_\theta(x|z) = \mathcal{N}(x; \hat{x}, \sigma^2 I)$:

$$-\log p_\theta(x|z) = \frac{D}{2\sigma^2} \cdot \text{MSE} + \frac{D}{2}\ln\sigma^2 + \text{const}$$

where $\text{MSE} = \frac{1}{D}\|x - \hat{x}\|^2 = \mathcal{L}_{\text{recon}}$ and $D = T \times F$. The factor $D$ is critical: it scales the reconstruction term relative to the KL, ensuring that the per-dimension cost of encoding information in latent space is compared against the per-dimension cost of reconstruction error. Without it, the KL dominates by a factor of $D$ and all latent dimensions collapse to the prior (posterior collapse).

The key insight (Dai & Wipf, 2019 [12]; Lucas et al., 2019 [13]) is that $\sigma^2$ can be **learned as a scalar parameter**. Setting $\partial \mathcal{L}/\partial \sigma^2 = 0$ yields $\sigma^2 = \text{MSE}$ — the average per-element residual variance the decoder cannot explain. The model invests capacity in reconstructing shared patterns and stops trying to reconstruct stock-specific noise. A latent dimension remains active only if the reconstruction gain it provides (reduction in $D \cdot \text{MSE}/(2\sigma^2)$) exceeds its KL cost — which, for $D = 504$ and $\sigma^2 \approx 0.5$, requires explaining at least ~0.1% of signal variance per dimension. This is the mechanism that yields a data-dependent number of active factors (AU) out of K = 200 — typically 60–120 for 30 years of daily data on ~1000 stocks, constrained downstream by $\text{AU}_{\max}^{\text{stat}}$ (Section 4.3).

$$\mathcal{L}_{\text{baseline}} = \frac{D}{2\sigma^2} \cdot \mathcal{L}_{\text{recon}} + \frac{D}{2}\ln\sigma^2 + \mathcal{L}_{\text{KL}}$$

$\sigma^2 = \exp(\log\sigma^2)$, initialized at 1.0, clamped to $[10^{-4}, 10]$.

**Monitoring.** Log $\sigma^2$ and **active units (AU)** (dimensions with marginal KL > 0.01 nats) every epoch. Healthy ranges are **adaptive**, not fixed — they depend on the data configuration:

- **$\sigma^2$**: calibrate by running a **linear VAE** (single linear encoder/decoder layer, same K) on the same data as a reference. The linear VAE's converged $\sigma^2_{\text{lin}}$ represents the residual variance unexplainable by linear factors. The non-linear VAE should achieve $\sigma^2 < \sigma^2_{\text{lin}}$; a ratio $\sigma^2 / \sigma^2_{\text{lin}} > 1.2$ signals that the non-linear model is not extracting additional structure. $\sigma^2$ hitting the lower clamp ($10^{-4}$) → the $D$ scaling factor may be missing or the model is overfitting reconstruction.
- **AU — posterior collapse**: AU < max(0.15·K, $\text{AU}_{\text{PCA}}$), where $\text{AU}_{\text{PCA}}$ is the number of factors estimated by the Bai & Ng (2002) IC₂ criterion on the same return panel. This ensures the VAE captures at least as many factors as a linear baseline.
- **AU — capacity exhaustion**: AU > 0.85·K → increase K.
- **Factor explanatory power**: $1 - \text{tr}(D_\varepsilon)/\text{tr}(\Sigma_{\text{assets}})$ should exceed max(0.50, $\text{EP}_{\text{PCA}} + 0.10$), where $\text{EP}_{\text{PCA}}$ is the explanatory power of the PCA baseline. The VAE must add value beyond linear factor decomposition.
- **AU vs $\text{AU}_{\max}^{\text{stat}}$**: if AU exceeds the statistical guard (Section 4.3), log a warning — the downstream pipeline will truncate, but persistent exceedance across folds signals insufficient historical depth for the model's discovered complexity.

**Recon/KL balancing — three modes (mutually exclusive).**

The reconstruction and KL terms compete for the same latent capacity. Two mechanisms can balance them — learned $\sigma^2$ and $\beta$ weighting — but they address the same degree of freedom. Combining them without care leads to unpredictable interactions: if $\beta > 1$ is applied on top of learned $\sigma^2$, the model faces double KL pressure, and $\sigma^2$ may compensate by decreasing (partially absorbing the $\beta$ effect), yielding a net result that is neither the intended $\sigma^2$-optimal balance nor the intended $\beta$-disentanglement. Similarly, KL annealing ($\beta_t$ ramping from 0 to 1) combined with learned $\sigma^2$ causes $\sigma^2$ to collapse during the warmup phase (no KL pressure → model over-reconstructs → $\sigma^2 \to 10^{-4}$), then must recover abruptly when $\beta_t$ approaches 1.

The three modes, in order of preference:

**Mode P (primary) — Learned $\sigma^2$, $\beta = 1$ fixed.** The baseline. $\sigma^2$ automatically finds the optimal recon/KL balance via gradient descent — no schedule, no hyperparameter. $\beta$ is fixed at 1.0 and never modified. This is the loss described above ($\mathcal{L}_{\text{baseline}}$).

**Mode F (fallback) — Fixed $\sigma^2 = 1$, KL annealing $\beta_t$.** If Mode P is unstable (diagnosed by $\sigma^2$ oscillating or hitting clamps persistently), disable $\sigma^2$ learning (freeze at 1.0) and use a gradually increasing KL weight:

$$\mathcal{L}_t = \frac{D}{2} \cdot \mathcal{L}_{\text{recon}} + \beta_t \cdot \mathcal{L}_{\text{KL}}, \quad \beta_t = \min\!\left(1, \; \frac{t}{T_{\text{warmup}}}\right)$$

Note: $\sigma^2$ is frozen at 1.0; the $D/2$ scaling factor is retained (since $D/(2\sigma^2)\big|_{\sigma^2=1} = D/2$). The manual schedule $\beta_t$ replaces only the adaptive role of $\sigma^2$, not the dimensional scaling. Without $D/2$, the MSE (a per-element mean ≈ 0.3–0.7) would be dwarfed by the KL (a sum over AU dimensions ≈ 60–120 nats), causing immediate posterior collapse at $\beta_t = 1$. With $D/2 = 252$ for $T = 504, F = 1$: effective reconstruction ≈ $252 \times 0.5 = 126$ vs KL ≈ 80 — a healthy balance. This lets the model first learn representations (as a standard autoencoder), then progressively introduces regularization. A cyclical variant (Fu et al., 2019 [15]) can help escape poor local minima.

**Mode A (advanced) — Learned $\sigma^2$, $\beta > 1$ fixed.** For explicit disentanglement pressure beyond what $\sigma^2$ alone produces (β-VAE [6]). Valid but requires monitoring: track $\sigma^2$ and AU as $\beta$ increases. If $\sigma^2$ decreases proportionally to $\beta$ (absorbing the extra KL pressure), the net effect is null — increase $\beta$ further or switch to Mode F. **Range:** $\beta \in [1.0, 4.0]$; explore only after Mode P is validated.

#### Co-movement scaffolding (default)

The encoder optimizes a per-sample MSE: each stock is reconstructed independently. While the shared encoder architecture creates an implicit tendency for similar inputs to produce similar latent vectors, this mechanism is **not guaranteed** to preserve empirical correlation structure — particularly with a wide bottleneck (AU ≥ 80), where the encoder has enough capacity to use different latent dimensions for correlated stocks without penalty (Locatello et al., 2019 show that unsupervised disentanglement without inductive bias is theoretically impossible). A **co-movement loss** $\mathcal{L}_{\text{co-mov}}$ explicitly guides the latent space to reflect pairwise dependence structure:

$$\mathcal{L} = \frac{D}{2\sigma^2} \cdot \mathcal{L}_{\text{recon}} + \frac{D}{2}\ln\sigma^2 + \mathcal{L}_{\text{KL}} + \lambda_{\text{co}}(t) \cdot \mathcal{L}_{\text{co-mov}}$$

**Computing $\mathcal{L}_{\text{co-mov}}$.** For each pair (i, j) in a batch: compute Spearman rank correlation $\rho_{ij}$ of their returns; compute cosine distance $d(z_i, z_j)$ in latent space; penalize the gap:

$$\mathcal{L}_{\text{co-mov}} = \frac{1}{|\mathcal{P}|} \sum_{(i,j)} \left( d(z_i, z_j) - g(\rho_{ij}) \right)^2$$

with $g(\rho) = 1 - \rho$ (high correlation → small target distance).

**Curriculum schedule for $\lambda_{\text{co}}$ (three phases).**

The co-movement loss is a temporary scaffolding, not a permanent term — it guides the encoder toward a relational structure that the reconstruction loss alone may not discover, then releases it to capture non-linear dependencies beyond what Spearman correlation can express.

- **Phase 1 — Scaffolding:** $\lambda_{\text{co}} = \lambda_{\text{co}}^{\max}$. The encoder is strongly guided toward a latent space reflecting empirical correlation structure. Duration: ~30% of total epochs.
- **Phase 2 — Annealing:** $\lambda_{\text{co}}$ decreases linearly from $\lambda_{\text{co}}^{\max}$ to 0. The encoder progressively refines beyond linear co-movement — tail dependence, conditional correlations, non-linear interactions emerge. Duration: ~30% of total epochs.
- **Phase 3 — Free refinement:** $\lambda_{\text{co}} = 0$. The encoder operates on reconstruction + KL alone, free to discover structures invisible to Spearman scaffolding. Duration: ~40% of remaining epochs.

$\lambda_{\text{co}}^{\max}$ and phase durations are hyperparameters validated within the walk-forward protocol (Section 4.8). **Baseline:** $\lambda_{\text{co}}^{\max} = 0.5$.

**Pair specification (Phases 1–2).** A batch contains *windows*, not stocks. Eligible pairs must satisfy three conditions: (1) **distinct stocks** ($i \neq j$) — same-stock pairs are trivial ($\rho = 1$, $d = 0$) and excluded; (2) **temporal synchronization** — the two windows' end dates must be within $\delta_{\text{sync}} = 21$ days, ensuring ≥ 96% temporal overlap on $T = 504$; (3) **sufficient valid data** — both stocks must have ≥ 80% non-missing returns in the overlapping period. The correlation $\rho_{ij}$ is Spearman rank correlation on **raw returns** (not z-scored) over the common temporal segment. To cap computational cost, pairs are randomly subsampled to $|\mathcal{P}| \leq 2048$ per batch.

**Batching protocol — co-movement phases vs free refinement.** The co-movement loss imposes a technical constraint: valid Spearman correlations between pairs require temporally synchronized windows. This forces **synchronous batching** during Phases 1–2 (all windows in a batch share a common date block of $\delta_{\text{sync}}$ days).

However, synchronous batching increases gradient variance: windows from the same temporal block share market context, producing correlated gradients ($\text{Cov}(g_i, g_j) > 0$) that reduce the effective batch size (Zhao & Zhang, 2014). For a batch of size $B$ with average intra-batch gradient correlation $\bar{\rho}$, the effective batch size is:

$$B_{\text{eff}} = \frac{B}{1 + \bar{\rho}(B-1)}$$

**Mitigation: stratified sampling within each temporal block.** Pre-cluster the ~1000 stocks into $S$ strates (10–20 groups via k-means on trailing 63-day returns, or GICS sectors as a zero-cost proxy). For each batch during Phases 1–2:

1. Select a random temporal block (date range of $\delta_{\text{sync}}$ days).
2. Sample $B/S$ windows from each strate, ensuring cross-sectional diversity while maintaining temporal synchronization.
3. Recalculate the clustering at each walk-forward fold (the strates themselves are not exposed to the encoder — they affect only batch composition).

This satisfies the synchronization constraint (co-movement loss computable) while maximizing intra-batch diversity (reduced gradient variance). Stratified sampling with Neyman-like allocation accelerates SGD convergence 2–5× vs uniform sampling (Zhao & Zhang, 2014; Zhang et al., 2017 — DPP mini-batch diversification).

During **Phase 3** ($\lambda_{\text{co}} = 0$), synchronous batching provides no benefit and only increases gradient variance. The batching strategy reverts to **standard random shuffling** — windows drawn uniformly across all stocks and time periods.

| Phase | $\lambda_{\text{co}}$ | Batching strategy | Rationale |
|-------|----------------------|-------------------|-----------|
| 1 (scaffolding) | $\lambda_{\text{co}}^{\max}$ | Synchronous + stratified sectoral | Co-movement loss requires synchronization; stratification mitigates gradient variance |
| 2 (annealing) | Linear decay → 0 | Synchronous + stratified sectoral | Same constraint, progressive release |
| 3 (free refinement) | 0 | Random shuffling | No inter-sample term → optimal variance via diversity |

**Inference impact: none.** The co-movement loss and synchronous batching affect training only. At inference, the encoder produces $\mu_i$ by individual forward pass — the downstream pipeline (aggregation, $B$, $\Sigma_z$, optimization) is strictly unchanged.

**Architectural alternative (rejected).** A cross-attention encoder processing all $n$ stocks simultaneously could learn correlations directly, but ties the model to a fixed universe — adding/removing a stock requires retraining. Universe flexibility is a core requirement.

#### Crisis weighting

Stress periods represent ~2–5% of historical data, so the gradient is dominated by normal-regime patterns. The solution: a **per-window weight** γ so crisis windows contribute γ× more to the gradient, incentivizing the encoder to allocate capacity to tail co-movements.

**Window labeling.** A window is labeled by the fraction $f_c^{(w)}$ of its days where VIX exceeds a threshold. The effective weight:

$$\gamma_{\text{eff}}^{(w)} = 1 + f_c^{(w)} \cdot (\gamma - 1)$$

Calm windows ($f_c = 0$) receive weight 1; full-crisis windows ($f_c = 1$) receive γ; mixed windows receive proportional weight. This continuous formulation avoids arbitrary binary discontinuities.

**VIX threshold.** Computed on an **expanding window** over the training period only (point-in-time: no look-ahead):

$$\tau_{\text{VIX}}^{(k)} = \text{Percentile}_{80}\left(\text{VIX}_{t_0:t_{\text{train}}}\right)$$

Recalculated once per fold. The 80th percentile aligns with Bansal & Stivers (2023) [28], who identify this zone as where the equity risk premium steps up sharply. On US VIX since 1990, this yields ~21–24. The expanding window preserves memory of all past crises. **Pre-1990 data:** the CBOE VIX is available from January 1990. For training periods extending before this date, use the VXO (CBOE's original volatility index, available from 1986) or, prior to 1986, a realized volatility proxy: annualized 21-day rolling standard deviation of S&P 500 daily returns, percentile-matched to VIX over their common history to calibrate the threshold.

**Calibrating γ via effective contribution.** Under the assumption that windows are predominantly fully calm or fully crisis (reasonable given VIX persistence and 504-day windows):

$$\eta_{\text{binary}} \approx \frac{p_c \cdot \gamma}{p_c \cdot \gamma + (1 - p_c)}$$

For $p_c \approx 0.20$ (80th percentile):

| Target $\eta$ | γ | Interpretation |
|----------|---|----------------|
| 20% | 1.0 | No weighting |
| **30%** | **1.7** | Mild |
| **40%** | **2.7** | Moderate (≈ √inverse freq.) |
| **43%** | **3.0** | **Baseline** |
| 50% | 4.0 | Strong |

The exact effective $\eta$ is computed from $\mathbb{E}[f_c]$ and $\mathbb{E}[f_c^2]$ on training data:

$$\eta_{\text{eff}} = \frac{\mathbb{E}\left[f_c + f_c^2(\gamma - 1)\right]}{1 + \mathbb{E}[f_c](\gamma - 1)}$$

The expectations are empirical averages over the **constructed training windows** (not individual days): $\mathbb{E}[f_c] = \frac{1}{|\mathcal{W}|}\sum_{w \in \mathcal{W}} f_c^{(w)}$, and similarly for $\mathbb{E}[f_c^2]$. Because VIX is autocorrelated (crisis periods cluster), $f_c$ is bimodal (most windows are near 0 or near 1 for $T = 504$), so $\mathbb{E}[f_c^2]$ is close to $\mathbb{E}[f_c]$ — making the continuous $\eta_{\text{eff}}$ close to the binary approximation but slightly lower. Baseline $\gamma = 3.0$ produces ~35–43% effective crisis contribution.

**Interaction with $\sigma^2$.** γ is per-window; $\sigma^2$ is global. They operate on orthogonal axes, but very large γ may push $\sigma^2$ upward, diluting crisis emphasis. Monitor: if $\sigma^2$ increases after raising γ, the model is relaxing globally rather than learning crisis structure.

**Monitoring.** Track $\overline{\text{MSE}}_{\text{crisis}}$ (windows with $f_c > 0.5$) / $\overline{\text{MSE}}_{\text{normal}}$ (with $f_c < 0.2$). Ratio > 2: crisis under-reconstructed, increase γ. Ratio < 0.5: over-allocated to crisis. Verify empirical $\eta_{\text{eff}} \in [0.25, 0.45]$. **Range to explore:** γ ∈ [1.5, 5.0].

**Complete loss function (Mode P — primary):**

$$\mathcal{L} = \frac{D}{2\sigma^2} \cdot \mathcal{L}_{\text{recon, weighted}} + \frac{D}{2}\ln\sigma^2 + \mathcal{L}_{\text{KL}} + \lambda_{\text{co}}(t) \cdot \mathcal{L}_{\text{co-mov}}$$

$$\mathcal{L}_{\text{recon, weighted}} = \frac{1}{|\mathcal{B}|} \sum_{w \in \mathcal{B}} \gamma_{\text{eff}}^{(w)} \cdot \text{MSE}(w), \quad \gamma_{\text{eff}}^{(w)} = 1 + f_c^{(w)} \cdot (\gamma - 1)$$

Baseline: $\lambda_{\text{co}}(t)$ follows the curriculum schedule (Phases 1→2→3), $\sigma^2$ learned, $\beta = 1$. For Mode F, replace $\frac{D}{2\sigma^2} \cdot \mathcal{L}_{\text{recon, weighted}} + \frac{D}{2}\ln\sigma^2 + \mathcal{L}_{\text{KL}}$ with $\frac{D}{2} \cdot \mathcal{L}_{\text{recon, weighted}} + \beta_t \cdot \mathcal{L}_{\text{KL}}$ ($\sigma^2$ frozen at 1.0, $D/2$ retained). For Mode A, multiply $\mathcal{L}_{\text{KL}}$ by $\beta > 1$.

### 4.5 Training

#### Uniform across all available history

The encoder learns a broad vocabulary of risk patterns across **all** historical regimes, not weighted toward the present. Current relevance is handled at inference (Section 4.6), not training. Batch composition follows the **curriculum batching protocol** defined in Section 4.4: during co-movement scaffolding phases (1–2), synchronous temporal blocks with stratified sectoral sampling; during free refinement (Phase 3), standard random shuffling across all stocks and time periods.

**Batch size.** Default **256 windows**. Lower bound ~32 (BatchNorm reliability). Upper bound constrained by GPU memory (~500 MB for 256 samples). Range 64–512 has modest effect; 256 is standard for comparable architectures.

#### Optimizer

**Adam** (Kingma & Ba, 2015 [29]) with default momentum parameters $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$. Adam is the de facto standard for training VAEs — every comparable financial VAE in the literature uses it (FactorVAE [30], RVRAE [31], Diffusion-VAE [32], VAR-VAE [33]). Its adaptive per-parameter learning rates are well suited to the mixed loss landscape of VAEs, where the reconstruction and KL terms operate at different gradient scales. SGD with momentum is a viable alternative for fine-tuning but offers no advantage for initial training of this architecture.

#### Learning rate

**Baseline: $\eta_0 = 10^{-4}$.** Conservative relative to the Adam default ($10^{-3}$), justified by the low signal-to-noise ratio of financial return data — a lower learning rate reduces the risk of the encoder memorizing noise patterns during early training. The literature on financial VAEs reports learning rates in the range $[5 \times 10^{-4}, 5 \times 10^{-3}]$ ([30]–[33]), but these typically involve smaller architectures or shorter windows. For this model ($K = 200$, $T = 504$), a more conservative starting point is appropriate.

**Search range:** $[5 \times 10^{-5}, 10^{-3}]$ (log-scale), selected via the nested temporal cross-validation described in Section 4.8.

**Scheduler: ReduceLROnPlateau** — reduce $\eta$ by a factor of 0.5 when the validation ELBO has not improved for 5 consecutive epochs. This is simpler and more robust than cosine annealing for models of this size. **Interaction with KL annealing (Mode F only):** during the $T_{\text{warmup}}$ phase (Section 4.4), the loss composition changes mechanically as $\beta_t$ ramps up, which can trigger spurious plateau detections. **Default: disable the scheduler during warmup** — activate it only once $\beta_t$ reaches its target value (1.0). During warmup the loss landscape is non-stationary by design; plateau detection is meaningful only when the loss composition is stable. Not applicable to Mode P/A (no warmup).

#### Epochs and early stopping

**Maximum epochs: 100.** This serves as a computational ceiling, not a convergence target. The architecture (1D CNN encoder/decoder with $K = 200$) is modest compared to image-generation VAEs; convergence typically occurs well before 100 epochs on datasets of this size ($n \times$ windows per stock $\approx 10^6$+ samples). For reference, comparable financial VAEs train for 20–100 epochs: Diffusion-VAE uses 20 epochs per stock [32], Stockformer 100 epochs [34], agent-based VAE backbones 30 epochs [35].

**Early stopping** is the actual convergence criterion. Monitor the **validation ELBO** on the temporal validation set (last 2 years of training data, with embargo):

$$\mathcal{L}_{\text{val}} = \frac{D}{2\sigma^2} \cdot \mathcal{L}_{\text{recon}}^{(\gamma=1)} + \frac{D}{2}\ln\sigma^2 + \mathcal{L}_{\text{KL}}$$

**Shared validation set (intentional).** This is the same split used for nested hyperparameter selection within each fold (Section 4.8). The sharing introduces a mild optimistic bias on the checkpoint's validation ELBO — the selected checkpoint is, by construction, the one with the best score on this set. However, the bias is contained for two reasons: (1) early stopping selects among a *temporally ordered* sequence of ~50–100 checkpoints (not arbitrary models), with the ELBO plateau spanning only 5–10 effectively distinct candidates; (2) hyperparameter selection does *not* use this validation ELBO — it uses the OOS fold score ($\bar{\hat{H}}_{\text{OOS}}$ minus drawdown penalty), which is computed on data entirely unseen during training and early stopping. The validation set only determines *when to stop training* for a given HP config; the *comparison between configs* happens OOS. Splitting the 2 years further (e.g., 18 months + 6 months) would degrade both signals without meaningful debiasing. **Overfit diagnostic:** if $\mathcal{L}_{\text{val}}^{\text{best}} / \mathcal{L}_{\text{train}}^{\text{best}} < 0.85$ or $> 1.5$, flag the fold for inspection — the former suggests the validation set is unrepresentative, the latter confirms overfitting.

Crisis weighting is **excluded** ($\gamma = 1$): it is a training-time gradient redistribution choice, not a property of the data. Applying γ on validation would bias model selection toward folds with crisis-rich validation periods. The learned $\sigma^2$ is **included**: it is a model parameter (part of the likelihood $p_\theta(x|z)$), not a training trick — a model that inflates $\sigma^2$ is correctly penalized. The co-movement loss ($\lambda_{\text{co}}$) is **excluded**: it is auxiliary scaffolding, not part of the generative model's ELBO.

- **Patience:** 10 epochs without improvement.
- **Restore best weights:** yes — revert to the checkpoint with the lowest validation ELBO. The epoch of this checkpoint is denoted $E^*$ and serves as the fixed stopping point for the Phase B deployment run (Section 4.8), where no validation set is available.
- **Warmup exclusion (Mode F only):** early stopping is **disabled** during the KL annealing warmup phase ($T_{\text{warmup}}$). During this phase, the loss changes composition (the KL term ramps from 0 to its target weight), so apparent degradation is expected and does not signal overfitting. Not applicable to Mode P/A (no warmup phase).

The reparameterization trick ($z = \mu + \sigma \cdot \epsilon$, $\epsilon \sim \mathcal{N}(0,1)$) is used to make latent sampling differentiable during training — standard for all VAEs since [5], implemented in the `forward` pass of the encoder.

#### Retraining frequency

The model is **fully retrained** when the training dataset changes meaningfully:

| Trigger | Frequency | Rationale |
|---------|-----------|-----------|
| Accumulated new data | Monthly to quarterly | ~60–180 new trading days |
| Universe changes | As needed | Delistings or reconstitutions |
| Structural market shifts | As detected | New asset types, major regulatory changes |

Daily fine-tuning rejected: negligible new information, destabilizes latent space, reintroduces temporal bias.

### 4.6 Inference: composite risk profiles

For each stock *i* and window position *t*, the encoder produces a local latent vector $\mu_i^{(t)} \in \mathbb{R}^K$. The stock's **composite risk profile** aggregates all local vectors across its full history:

$$\bar{\mu}_i = \text{Agg}\left(\mu_i^{(t_1)}, \mu_i^{(t_2)}, ..., \mu_i^{(t_M)}\right)$$

**Inference stride $s_{\text{inf}} = 1$ (daily).** The sliding window is advanced one day at a time, producing the maximum number of local vectors per stock (~7,057 for 30 years of history with $T = 504$). The computational cost is negligible at inference (forward pass only, no backpropagation — ~7M passes across 1,000 stocks complete in minutes on GPU; storage: ~5.6 GB at $K = 200$, float32). This choice is justified on three grounds:

1. **No arbitrary degree of freedom.** A stride $s_{\text{inf}} > 1$ introduces a calendar alignment dependency (the result changes with the starting day). $s_{\text{inf}} = 1$ eliminates this.
2. **Full compatibility with alternative aggregations.** While the effective sample size (ESS) framework shows that for mean aggregation, the result is quasi-identical for any $s_{\text{inf}} \leq 21$ (the $N_{\text{eff}} \approx T_{\text{hist}} / T \approx 15$ independent windows are determined by window overlap, not stride), alternative aggregation methods (max-pool, weighted mean, learned attention — Section 6) exploit the full distribution of local vectors. A dense trajectory preserves this option.
3. **Diagnostic value.** The daily-resolution trajectory $\mu_i^{(t)}$ enables visualization of latent space transitions and regime detection, independently of the composite profile.

Note: $s_{\text{inf}}$ is distinct from the training stride $s_{\text{train}}$ (Section 5), which controls the diversity of the training dataset. The training stride affects gradient computation and dataset size; the inference stride affects only the resolution of composite profiles. They are decoupled and need not be equal.

Default aggregation: **simple mean** (all windows contribute equally, preserving memory of all historical regimes). Alternatives discussed in Section 6.

#### Exposure matrix B

Stacking the $n$ composite vectors:

```
Exposure matrix B (n × K):

              Factor 1   Factor 2   ...   Factor K
Stock 1    [  β_{1,1}    β_{1,2}    ...   β_{1,K}  ]
Stock 2    [  β_{2,1}    β_{2,2}    ...   β_{2,K}  ]
  ...             ...        ...    ...       ...
Stock n    [  β_{n,1}    β_{n,2}    ...   β_{n,K}  ]
```

This matrix realizes the graph structure from Section 2.2: multiple membership, continuous intensities, memory of dormant exposures. B contains **shape-only** exposures (from per-window z-scored data). Before downstream use, B is filtered to active dimensions $B_{\mathcal{A}}$ (Section 4.7), then rescaled with **date-specific** trailing volatility for factor estimation ($\tilde{B}_{\mathcal{A},t}$) or **current** trailing volatility for portfolio construction ($\tilde{B}_{\mathcal{A}}^{\text{port}}$), and rotated to the principal factor basis ($\tilde{B}'^{\text{port}}_{\mathcal{A}}$).

**New stock integration.** Pass all available T-day windows through the encoder and aggregate. Short histories produce less reliable composites — can be reflected in a confidence weight during optimization.

**Stability.** Between retrainings, B is **fixed** — factor dimensions have consistent meaning.

### 4.7 Portfolio construction

The encoder produces a **pure risk model** with no return views. Return-seeking behavior is injected at the optimizer level as a separate, removable layer.

#### Diversification metric

**Entropy of principal factor risk contributions.** The risk contributions $c_k = \beta_{p,k} \cdot (\Sigma_z \beta_p)_k$ can be **negative** when $\Sigma_z$ has non-zero off-diagonal terms (correlated factors), making Shannon entropy undefined. The solution (Meucci [39]): diagonalize $\Sigma_z = V \Lambda V^T$ and work in the **principal factor basis** $\tilde{B}'^{\text{port}}_{\mathcal{A}} = \tilde{B}_{\mathcal{A}}^{\text{port}} \cdot V$, where $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_{\text{AU}})$. In this basis, the portfolio's factor exposure is $\beta'_p = \tilde{B}'^{\text{port},T}_{\mathcal{A}} w$, and each factor's risk contribution becomes:

$$c'_k = (\beta'_{p,k})^2 \cdot \lambda_k \geq 0 \quad \forall k$$

since $\lambda_k \geq 0$ (eigenvalues of a covariance matrix) and the square is non-negative. The entropy is then always well-defined:

$$H(w) = -\sum_{k=1}^{\text{AU}} \hat{c}'_k \ln \hat{c}'_k \quad \text{where } \hat{c}'_k = \frac{c'_k}{\sum_j c'_j}$$

The rotation is orthogonal: $\tilde{B}'^{\text{port}}_{\mathcal{A}} \Lambda \tilde{B}'^{\text{port},T}_{\mathcal{A}} = \tilde{B}^{\text{port}}_{\mathcal{A}} \Sigma_z (\tilde{B}^{\text{port}}_{\mathcal{A}})^T$, so the asset covariance is unchanged. This rotation also resolves the rotational indeterminacy of the VAE latent space: the principal factors of $\Sigma_z$ are unique (up to sign).

**Gradient of $H$ w.r.t. $w$.** Denote $B' = \tilde{B}'^{\text{port}}_{\mathcal{A}}$, $\beta' = B'^T w$ (portfolio exposure in the principal factor basis), $C = \sum_k \lambda_k (\beta'_k)^2$ (total systematic risk). The gradient is derived by composing three layers:

1. $\partial H / \partial \hat{c}'_k = -(\ln \hat{c}'_k + 1)$
2. $\partial \hat{c}'_k / \partial c'_m = (\delta_{km} - \hat{c}'_k) / C$ (Jacobian of the normalization $\hat{c}'_k = c'_k / C$)
3. $\partial c'_k / \partial w_j = 2\lambda_k \beta'_k B'_{j,k}$

Chaining and using $\sum_k \hat{c}'_k (\ln \hat{c}'_k + 1) = 1 - H$:

$$\boxed{\nabla_w H = -\frac{2}{C}\; B'\; \phi, \qquad \phi_k = \lambda_k \, \beta'_k \,(\ln \hat{c}'_k + H)}$$

**Verification:** at the entropy maximum ($\hat{c}'_k = 1/\text{AU}$ for all $k$), $\ln \hat{c}'_k = -\ln(\text{AU}) = -H$, so $\phi_k = 0$ and $\nabla_w H = 0$ — the gradient vanishes as expected. **Cost:** one matrix-vector product $O(n \cdot \text{AU})$ per SCA iteration. At each SCA step, $H(w)$ is linearized as $H(w^{(t)}) + \nabla_w H(w^{(t)})^T (w - w^{(t)})$, and the subproblem maximizes this linear term plus the convex penalties.

Maximum entropy = equal risk from each principal factor = no dominant risk concentration. The diversification ratio (Choueifaty & Coignard 2008) is a complementary asset-level diagnostic but cannot detect latent factor concentration.

#### Covariance estimation

The $n \times n$ asset covariance decomposes as $\Sigma_{\text{assets}} = \tilde{B}_{\mathcal{A}}^{\text{port}} \cdot \Sigma_z \cdot (\tilde{B}_{\mathcal{A}}^{\text{port}})^T + D_\varepsilon$, where all quantities operate in the **active dimension subspace** $\mathcal{A}$, and $\tilde{B}_{\mathcal{A}}^{\text{port}}$ is rescaled at the current rebalancing date (see below).

**Step 0 — Active dimension filtering.** The auto-pruning mechanism (Section 4.3) deactivates superfluous latent dimensions by collapsing them toward the prior $\mathcal{N}(0,1)$. For these dimensions, the encoder outputs $\mu_{i,k} \approx 0$ for all stocks $i$ — the corresponding columns of $B$ are quasi-null. A regression on the full $K$-dimensional $\tilde{B}$ would produce a singular (or near-singular) $\tilde{B}^T \tilde{B}$, since rank$(\tilde{B}) \approx \text{AU} \ll K$.

The solution is to use the information the VAE already provides: filter $B$ to the set of active units $\mathcal{A} = \{k : \text{KL}_k > 0.01 \text{ nats}\}$, with $|\mathcal{A}| = \text{AU}$:

$$B_{\mathcal{A}} \in \mathbb{R}^{n \times \text{AU}}$$

This set is determined once per retraining (when AU is measured) and propagated through the entire downstream pipeline. **Statistical guard:** if $\text{AU} > \text{AU}_{\max}^{\text{stat}}$ (Section 4.3), truncate $\mathcal{A}$ to the $\text{AU}_{\max}^{\text{stat}}$ dimensions with highest marginal KL. The truncated dimensions contributed to learning a richer latent space but cannot be reliably estimated in $\Sigma_z$ given the available observation count. With $n = 1000$ and effective AU $\approx$ 80 (after potential truncation), all subsequent linear systems are well-conditioned (ratio $n / \text{AU} \approx 12.5$).

**Why not ridge regression or SVD pseudo-inverse?** Ridge ($(\tilde{B}^T \tilde{B} + \lambda I)^{-1}$) regularizes all dimensions including active ones, biasing factor estimates toward zero and introducing a hyperparameter $\lambda$ without clear calibration target. SVD truncation rediscovers the active/inactive boundary that the KL has already identified, at higher cost and with its own threshold to tune. AU filtering is the only approach that *uses* the auto-pruning signal rather than working around it.

The remaining components, all operating on the filtered $B_{\mathcal{A}}$:

- **Exposure rescaling — estimation vs portfolio.** The encoder produces composite profiles $\bar{\mu}_i$ from per-window z-scored data — pure shape, no absolute scale. Two stocks with identical return dynamics but volatilities of 5% and 50% produce the same $\bar{\mu}_i$. Restoring scale compatibility with raw returns $r_t$ requires rescaling by realized volatility. However, **a single volatility snapshot is insufficient**: using today's $\sigma_i$ to rescale loadings applied to returns from 2008 (when the same stock may have had 3× higher volatility) misattributes systematic risk to idiosyncratic residuals — inflating $D_\varepsilon$ and depressing $\Sigma_z$, exactly the anti-cyclical failure mode the strategy is designed to avoid. The solution: **time-varying rescaling for estimation, current-date rescaling for portfolio construction.**

  - **For historical factor estimation** (recovering $\hat{z}_t$ across all dates): the rescaled exposure of stock $i$ at date $t$ is:

    $$\tilde{B}_{\mathcal{A},i,t} = \frac{\sigma_{i,t}}{\bar{\sigma}_t} \cdot \bar{\mu}_{\mathcal{A},i}$$

    where $\sigma_{i,t}$ is the trailing annualized volatility (252-day window ending at $t$) and $\bar{\sigma}_t$ the cross-sectional median at date $t$. The composite profile $\bar{\mu}_{\mathcal{A},i}$ (shape) is fixed between retrainings; only the volatility scale varies across dates. The ratio $\sigma_{i,t} / \bar{\sigma}_t$ captures each stock's relative risk magnitude at the time the returns were actually realized, so the cross-sectional regression correctly attributes co-movement to factors rather than residuals.

    **Ratio bounding.** The raw ratio $R_{i,t} = \sigma_{i,t} / \bar{\sigma}_t$ is **winsorized cross-sectionally at each date** to $[P_5(R_{\cdot,t}),\; P_{95}(R_{\cdot,t})]$ before rescaling. Without bounding, a stock at $R = 15$ (pre-delisting spike, biotech event volatility) receives $225\times$ the regression weight of a median stock in the OLS for $\hat{z}_t$, letting a single idiosyncratic return dominate the recovered factor realizations and propagate noise into $\Sigma_z$. The 5th/95th percentile bounds are adaptive (tighter in calm periods, wider in crises) and affect ~50 stocks per tail on $n = 1000$, preserving legitimate volatility differences while neutralizing outliers. The same winsorized ratio is applied to $\tilde{B}_{\mathcal{A}}^{\text{port}}$. **Diagnostic:** log the count and identity of clipped stocks per date; persistent clipping of the same stock across dates signals a poor factor fit for that stock. **Range to explore:** $[P_1, P_{99}]$ (permissive) to $[P_{10}, P_{90}]$ (aggressive).

    **Why 252 days.** The window length controls the bias-variance trade-off of the volatility estimate. A shorter window (63–126 days) reacts faster to regime changes but produces noisier $\tilde{B}_{\mathcal{A},i,t}$, amplifying estimation error in $\hat{z}_t$ and inflating $\Sigma_z$. A longer window (504 days) oversmooths and blends distinct regimes. 252 days (~1 year) is the industry standard (Barra USE4, Axioma) because it captures characteristic volatility without reacting to transient spikes. During crisis onset (e.g., COVID Feb–Mar 2020), the 252-day estimate lags — but $\Sigma_z$ on full history already preserves crisis-era factor correlations, and by the next rebalancing (monthly to quarterly) the window has absorbed the shock. Sensitivity: the walk-forward protocol (Section 4.8) can test 126 vs 252 if needed; the choice is not expected to rank among the top sensitivity drivers.

    **$\bar{\sigma}_t$ survivorship note.** $\bar{\sigma}_t$ is a cross-sectional median over stocks in the point-in-time universe $\mathcal{U}_t$. On CRSP (the specified data source), delisted stocks are included with their full pre-delisting history, so the composition of $\mathcal{U}_t$ is unbiased. Even with minor data gaps (e.g., alternative providers), $\bar{\sigma}_t$ is a robust statistic (median, not mean), and any residual bias is a common scaling factor across all loadings — absorbed into $\hat{z}_t$ magnitude without affecting relative risk contributions or portfolio weights.

    Pre-computation: rolling 252-day std over each stock's return series — $O(n \times T_{\text{hist}})$, vectorizable. Storage: $n \times T_{\text{hist}} \approx 7.5\text{M}$ floats (~30 MB).

  - **For portfolio construction** (projecting factor covariance into asset space): the rescaled exposure uses the current snapshot:

    $$\tilde{B}_{\mathcal{A},i}^{\text{port}} = \frac{\sigma_{i,\text{now}}}{\bar{\sigma}_{\text{now}}} \cdot \bar{\mu}_{\mathcal{A},i}$$

    This produces: "what would the covariance of my portfolio be if factors behaved as they have historically, given today's volatility structure." The ratio $\sigma_{i,\text{now}} / \bar{\sigma}_{\text{now}}$ is recalculated at each portfolio reconstitution.

- **$\hat{z}_t$** ($\text{AU} \times 1$): factor realizations recovered by cross-sectional OLS with the **date-specific** rescaled loadings:

  $$\hat{z}_t = (\tilde{B}_{\mathcal{A},t}^T \tilde{B}_{\mathcal{A},t})^{-1} \tilde{B}_{\mathcal{A},t}^T r_t$$

  At each date $t$, the regression uses only stocks active at $t$ (listed with available returns), so $\tilde{B}_{\mathcal{A},t}$ is the corresponding submatrix (rows for active stocks, columns for AU dimensions). The system remains heavily overdetermined as long as the number of active stocks $\gg$ AU. The recovered $z_t$ is in units of "return per unit of normalized exposure at median cross-sectional volatility" — a stable unit across time because $\bar{\sigma}_t$ absorbs global volatility inflation/deflation.

- **$\Sigma_z$** ($\text{AU} \times \text{AU}$): factor covariance. Empirical covariance of $\{\hat{z}_t\}$ over **full history** (crisis correlations persist permanently). Ledoit-Wolf shrinkage [18]: $\hat{\Sigma}_z = (1 - \delta^*) S_{\text{emp}} + \delta^* \frac{\text{tr}(S_{\text{emp}})}{\text{AU}} I_{\text{AU}}$, with $\delta^*$ computed analytically. The observations/free-parameters ratio is $\approx 7560 / \binom{\text{AU}+1}{2}$: for AU = 80 this gives $\approx 2.3$ (moderate — $\delta^*$ typically $0.05$–$0.15$); for AU = 120, $\approx 1.0$ (shrinkage becomes substantial). The analytical formula adapts automatically — no manual tuning required.
- **$D_\varepsilon$** ($n$ diagonal): idiosyncratic variances from residuals $\varepsilon_{i,t} = r_{i,t} - \tilde{B}_{\mathcal{A},i,t} \hat{z}_t$ (note: the **date-specific** $\tilde{B}_{\mathcal{A},i,t}$ is used, so residuals are correctly scaled at each date). Variance computed over all dates where stock $i$ is listed with available returns. Floor: $D_{\varepsilon,i} \geq 10^{-6}$ (prevents degenerate zero-idiosyncratic positions). **Short-history stocks:** a stock near the minimum listing requirement ($T = 504$ days) has fewer windows and a composite profile $\bar{\mu}_i$ that may miss episodic exposures. The regression then attributes unmodeled systematic risk to residuals, **inflating** $D_{\varepsilon,i}$. No correction is applied: $D_\varepsilon$ enters the risk penalty $w^T \Sigma w$ in the default objective, so the inflated $D_{\varepsilon,i}$ increases the stock's perceived risk and naturally under-weights it — the conservative behavior appropriate for a stock whose risk profile is incomplete. As the stock accumulates history and is re-processed at the next retraining, its composite profile improves and $D_{\varepsilon,i}$ decreases toward its true value.

**Asset covariance assembly.** Uses the **portfolio** rescaling:

$$\Sigma_{\text{assets}} = \tilde{B}_{\mathcal{A}}^{\text{port}} \cdot \Sigma_z \cdot (\tilde{B}_{\mathcal{A}}^{\text{port}})^T + D_\varepsilon$$

**Conditioning guard.** Verify $\kappa(\tilde{B}_{\mathcal{A},t}^T \tilde{B}_{\mathcal{A},t}) < 10^6$ at representative dates (e.g., first, last, and worst-conditioned). If violated (quasi-collinear active dimensions — unlikely given KL pressure toward distinctness), apply a minimal ridge: $\lambda = 10^{-6} \cdot \text{tr}(\tilde{B}_{\mathcal{A},t}^T \tilde{B}_{\mathcal{A},t}) / \text{AU}$.

All inter-stock correlation passes through the factors.

#### Optimization modes

The strategy's declared objective — "minimize risk through optimal diversification" — is a **two-level hierarchy**: risk minimization is the primary goal, factorial diversification is the means. This hierarchy is directly captured by combining a risk penalty with an entropy reward in a single objective.

**Default mode — Risk minimization via factorial diversification ($\mu = 0$).** No return forecast.

$$\max_w \; -\lambda \, w^T \Sigma w + \alpha \cdot H(w)$$

The parameter $\lambda$ penalizes portfolio variance; $\alpha$ rewards factorial entropy. Only the ratio $\tilde{\alpha} = \alpha / \lambda$ affects optimal weights. This mode interpolates continuously:
- $\alpha = 0$ ($\tilde{\alpha} = 0$): pure minimum-variance.
- $\alpha / \lambda \to \infty$: entropy dominates, converges to maximum diversification (the former Mode 1).
- Intermediate $\tilde{\alpha}$: the optimizer finds the portfolio that best trades off total risk against factorial concentration. **This is the operating point** — calibrated at the elbow of the variance-entropy frontier (see below).

Setting $\mu = 0$ is not an omission but a design choice: the strategy produces risk structure, not return forecasts. If no external return views are available, injecting $\mu = 0$ is the correct neutral position.

**Directional mode — Mean-variance with entropy reward.** Adds return views $\mu$ from external sources (valuation, momentum, analyst consensus, discretionary):

$$\max_w \; w^T \mu - \lambda \, w^T \Sigma w + \alpha \cdot H(w)$$

This is the full formulation. The default mode is the special case $\mu = 0$. The solver, constraints, and calibration procedures are identical.

**Exploratory mode — Pure entropy.** $\max_w \; H(w)$. Equivalent to $\lambda = 0$ in the default mode. Useful for diagnostic: what is the maximum achievable factorial diversification, unconstrained by risk? The gap between this and the default mode quantifies the cost of the risk penalty.

**Baseline: $\lambda = 1.0$.** Calibrated via the implied risk aversion of the market portfolio: $\lambda_{\text{mkt}} = \bar{\mu}_{\text{mkt}} / (2\sigma_{\text{mkt}}^2)$. With an equity risk premium of ~7% and market variance of ~3.5% ($\sigma \approx 18.7\%$), $\lambda_{\text{mkt}} \approx 1.0$. **Range:** $\lambda \in [0.5, 5.0]$.

**$\alpha$-$\lambda$ interaction.** Only the ratio $\tilde{\alpha} = \alpha / \lambda$ affects the optimal weights — multiplying both by the same constant leaves $w^*$ unchanged. The variance-entropy frontier (below) is therefore constructed at fixed $\lambda$, sweeping $\alpha$ alone. If $\lambda$ is subsequently changed, the $\alpha$ grid must be rescaled proportionally. At $\lambda = 1.0$, the $\alpha$ grid $\{0, 0.01, 0.05, 0.1, 0.5, 1, 5\}$ spans the range from pure minimum-variance to entropy-dominated.

#### Portfolio constraints

Design principle: **internalize costs into the objective**; hard caps as safety limits only.

**Concentration penalty.** $P_{\text{conc}}(w) = \sum_{i} \max(0, w_i - \bar{w})^2$ with soft threshold $\bar{w} = 3\%$, hard cap $w_{\max}^{\text{hard}} = 5\%$.

**Minimum position size.** Binary: $w_i \geq w_{\min} = 0.10\%$ or $w_i = 0$.

**Turnover penalty** (Almgren & Chriss [19]): spread (linear) + market impact (quadratic):

$$P_{\text{turn}}(w, w^{\text{old}}) = \kappa_1 \cdot \frac{1}{2}\sum_{i} |w_i - w_i^{\text{old}}| \;+\; \kappa_2 \cdot \sum_{i} \max\!\left(0,\; |w_i - w_i^{\text{old}}| - \bar{\delta}\right)^2$$

The quadratic term forces turnover distribution across many small trades. Hard cap $\tau_{\max}^{\text{hard}} = 30\%$ one-way. First rebalancing: $\kappa_1 = \kappa_2 = 0$.

**Calibration baselines** (refined by walk-forward): $\phi_0 \approx 25$ [5–100], $\kappa_{1,0} \approx 0.1$ [0.01–1.0], $\bar{\delta}_0 = 1\%$ [0.5%–3%], $\kappa_{2,0} \approx 7.5$ [1–50].

#### Complete formulations

Default ($\mu = 0$): $\;\max_w \; -\lambda \, w^T \Sigma w + \alpha \cdot H(w) - \phi \cdot P_{\text{conc}}(w) - P_{\text{turn}}(w, w^{\text{old}})$

Directional: $\;\max_w \; w^T \mu - \lambda \, w^T \Sigma w + \alpha \cdot H(w) - \phi \cdot P_{\text{conc}}(w) - P_{\text{turn}}(w, w^{\text{old}})$

Both subject to: $w_i = 0$ or $w_{\min} \leq w_i \leq w_{\max}^{\text{hard}}$, $\mathbf{1}^T w = 1$, $\frac{1}{2}\|w - w^{\text{old}}\|_1 \leq \tau_{\max}^{\text{hard}}$. The exploratory mode ($\max H(w)$) is the limit case $\lambda = 0$. The default mode at $\alpha = 0$ is pure minimum-variance. Both formulations share the same SCA solver — the $-\lambda w^T\Sigma w$ term is concave quadratic, entering each SCA subproblem unchanged.

**Calibrating $\alpha$ — variance-entropy frontier.** Solve the default formulation for a grid of $\alpha$ values at fixed $\lambda = 1.0$ (e.g. $\{0, 0.01, 0.05, 0.1, 0.5, 1, 5\}$). Each solution yields a point $(\text{Var}^*(\alpha), \; H^*(\alpha))$. The resulting frontier is the analogue, in variance-entropy space, of the classical mean-variance efficient frontier. The elbow — where the marginal entropy gain per unit of additional variance starts to diminish — is the natural operating point. For automated selection, use the $\alpha$ value where $\Delta H / \Delta \text{Var}$ drops below a threshold. If $\lambda \neq 1.0$, rescale the grid proportionally ($\alpha \propto \lambda$) since only $\tilde{\alpha} = \alpha / \lambda$ affects the solution.

#### Solver: non-convexity and resolution

$H(w)$ is **not concave in $w$**: the risk contributions $\hat{c}'_k(w) = (\tilde{B}'^{\text{port},T}_{\mathcal{A}} w)_k^2 \cdot \lambda_k$ are quadratic in $w$, and the composition with Shannon entropy breaks concavity. This is well-known for factor risk parity [36, 37].

**Resolution: Successive Convex Approximation (SCA)** per Feng & Palomar [36]:

1. **Multi-start** ($M = 5$): equal-weight, minimum variance, approximate ERC via Spinu [38], + 2 random feasible portfolios. **Random portfolio generation:** (a) draw a random cardinality $n_r \sim \text{Uniform}(30, 300)$; (b) sample $n_r$ stocks uniformly without replacement; (c) draw raw weights from $\text{Dirichlet}(\mathbf{1})$; (d) project to feasibility: clip to $w_{\max}^{\text{hard}}$, renormalize, zero out positions below $w_{\min}$, renormalize — iterate until no new violations (converges in 2–3 passes). The 3 deterministic starts cover the dispersed-to-concentrated spectrum; the 2 random starts explore different stock subsets as insurance against shared basins of attraction.
2. **Iterate**: linearize $H(w)$ around $w^{(t)}$ → convex surrogate $\hat{f}(w; w^{(t)})$ → solve via conic solver (MOSEK/ECOS) → obtain surrogate maximizer $w^*$ → update $w^{(t+1)} = w^{(t)} + \eta^{(t)}_{\text{step}}(w^* - w^{(t)})$ with step size $\eta^{(t)}_{\text{step}}$ determined by **Armijo backtracking line search** (see below). In both the default and directional modes, the subproblem at each iteration is a concave QP: the linearized $\alpha \cdot \nabla H^T w$ plus the concave $-\lambda w^T \Sigma w$ (and $w^T \mu$ if directional) — globally solvable.
3. **Step size — Armijo backtracking.** The linearization of $H$ is exact at first order around $w^{(t)}$ but may be a poor approximation at $w^*$ (especially in early iterations). A full step ($\eta_{\text{step}} = 1$) can decrease the *real* objective even though it maximizes the *surrogate*. The line search finds the largest step that guarantees sufficient improvement of the real objective $f$:

   $$\eta^{(t)}_{\text{step}} = \max\{\rho^j : j = 0, 1, \ldots, j_{\max}\} \quad \text{s.t.} \quad f(w_{\text{trial}}) \geq f(w^{(t)}) + c \cdot \eta_{\text{step}} \cdot \Delta_{\text{surr}}$$

   where $w_{\text{trial}} = w^{(t)} + \eta_{\text{step}}(w^* - w^{(t)})$, $f$ is the complete objective (default: $-\lambda w^T\Sigma w + \alpha H - \phi P_{\text{conc}} - P_{\text{turn}}$; directional: $w^T\mu - \lambda w^T\Sigma w + \alpha H - \phi P_{\text{conc}} - P_{\text{turn}}$), and $\Delta_{\text{surr}} = \hat{f}(w^*; w^{(t)}) - \hat{f}(w^{(t)}; w^{(t)}) \geq 0$ is the gain predicted by the surrogate ($\geq 0$ by construction since $w^*$ is the surrogate maximizer). Fixed parameters: $c = 10^{-4}$ (standard Armijo constant), $\rho = 0.5$ (contraction factor), $j_{\max} = 20$. Cost per trial: one evaluation of $f(w_{\text{trial}})$ — a few matrix products $O(n \cdot \text{AU})$, negligible. In practice, $\eta_{\text{step}} = 1$ passes the condition in the majority of iterations; backtracking activates mainly in the first 3–5 iterations. This sufficient-decrease condition is exactly what Feng & Palomar [36] require to guarantee convergence to a KKT point. **Notation:** $\eta_{\text{step}}$ denotes the Armijo step size, distinct from $\alpha$ (entropy reward weight), $\alpha_{\text{proj}}$ (projection ratio, Section 4.3), and $\alpha_{\text{trigger}}$ (delisting entropy threshold, Section 4.2).

4. **Converge** when $|f(w^{(t+1)}) - f(w^{(t)})| < 10^{-8}$. Typical: 20–50 iterations.
5. **Select** the solution with highest objective value across starts (default: highest $-\lambda w^T\Sigma w + \alpha H$; directional: highest $w^T\mu - \lambda w^T\Sigma w + \alpha \cdot H$).
6. **Enforce cardinality** on each start's solution via sequential entropy-aware elimination (see below), then select the best $H(w^{\text{card}}_m)$.

Cardinality constraint handled by **sequential entropy-aware elimination** after each SCA convergence. The semi-continuous constraint ($w_i = 0$ or $w_i \geq w_{\min}$) is NP-hard; the following greedy procedure provides a tractable approximation that preserves factorially important positions:

1. **Identify** all sub-threshold positions: $S_{\text{sub}} = \{i : 0 < w_i < w_{\min}\}$. If empty, stop.
2. **Evaluate** for each $i \in S_{\text{sub}}$ the entropy cost $\Delta H_i = H(w) - H(w^{(-i)})$, where $w^{(-i)}$ sets $w_i = 0$ and renormalizes.
3. **Eliminate** the single stock $i^* = \arg\min_{i \in S_{\text{sub}}} \Delta H_i$ (least costly).
4. **Re-optimize** via SCA on the reduced active set $S_{\text{active}} \setminus \{i^*\}$.
5. **Repeat** from step 1.

Convergence is guaranteed: the active set decreases strictly by one at each iteration and stocks are never reintroduced. In practice, the redistribution of capital from eliminated stocks often pushes remaining sub-threshold positions above $w_{\min}$, so the loop terminates in 5–15 iterations for $n = 1000$. The per-start enforcement runs after each of the $M = 5$ SCA multi-start solutions; final selection is on $H(w^{\text{card}}_m)$.

Implementation: CVXPY + MOSEK/ECOS (recommended) or SciPy SLSQP. The analytical gradients concern $H(w)$ only (the term being linearized in the SCA). The penalty terms ($P_{\text{conc}}$, $P_{\text{turn}}$) are convex and enter each SCA subproblem unchanged — L1 norms and piecewise-quadratic terms are natively conic-representable and handled automatically by CVXPY (`cp.norm1`, `cp.pos`). For $n \leq 2000$, full procedure completes in under a minute.

**Alternative — two-stage decomposition**: (1) optimize factor exposures $y^* = \arg\max_y H(y)$ in $\mathbb{R}^{\text{AU}}$ (small, reliable — the entropy of a vector in $\mathbb{R}^{\text{AU}}$ is a tractable low-dimensional problem); (2) project into asset space: $w^* = \arg\min_w \|\tilde{B}'^T_{\mathcal{A}} w - y^*\|^2 + \phi \cdot P_{\text{conc}}(w) + P_{\text{turn}}(w, w^{\text{old}})$ s.t. $w_i = 0$ or $w_{\min} \leq w_i \leq w_{\max}^{\text{hard}}$, $\mathbf{1}^T w = 1$, turnover cap. This remains a convex QP (tracking quadratic + convex penalties), globally solvable. With $n \gg \text{AU}$, $\tilde{B}'^T_{\mathcal{A}} w^* \approx y^*$ is tight — the penalty terms have minimal impact on factor fidelity. Simpler to implement than full SCA but does not support the directional mode (return views cannot be injected in step 1).

#### Update frequencies

| Component | Frequency | Mechanism |
|-----------|-----------|-----------|
| Encoder weights | Monthly to quarterly | Full retraining |
| Active dimensions $\mathcal{A}$ | At each retraining | KL > 0.01 nats; determines AU; fixed until next retraining |
| Composite profiles, $B_{\mathcal{A}}$ | At each retraining | Full history re-processed; filtered to $\mathcal{A}$ |
| $\sigma_{i,t}$ series (trailing 252d vol) | At each retraining | Rolling std over each stock's full return history; pre-computed once, stored (~30 MB) |
| $\tilde{B}_{\mathcal{A},t}^{\text{est}}$, $\hat{z}_t$, $\Sigma_z$, $D_\varepsilon$ | At each retraining | Full-history recomputation with date-specific rescaling $\sigma_{i,t}/\bar{\sigma}_t$ |
| $\tilde{B}_{\mathcal{A}}^{\text{port}}$, $\tilde{B}'^{\text{port}}_{\mathcal{A}}$ (rescaling + rotation) | At each retraining + each rebalancing (vol refresh) | $\sigma_{i,\text{now}}$ trailing 252d; $\Sigma_z$ eigendecomposition |
| Portfolio weights | At retraining, scheduled rebalancing, or alpha updates. Exceptional on delisting if $H < \alpha_{\text{trigger}} \times H^{\text{last}}$ | Re-optimization |

### 4.8 Validation protocol

The validation protocol answers whether the strategy *actually* works without hindsight. Without it, all metrics are suspect. Walk-forward metrics alone are insufficient: the strategy must also be evaluated against benchmark models to determine whether its complexity is justified. The benchmarking procedure is specified in `Latent_risk_factor_benchmark.md`.

#### Why walk-forward

Standard random train/test splits leak future information for financial time series. **Walk-forward validation** simulates real deployment: train on the past, deploy on the next period, observe, advance. A single split yields one data point; walk-forward produces a **distribution** of OOS results across ~34 folds covering crisis and calm periods.

#### Fold structure

Each fold has three contiguous, non-overlapping segments:

1. **Available data (expanding window):** all data from history start to $t_k$. Expanding (not rolling) to preserve all past crisis data — consistent with the anti-cyclical principle. Internally split into:
   - **Training subset:** $[\text{start}, \; t_k - 2\text{yr}]$ — used for gradient descent.
   - **Validation subset:** $[t_k - 2\text{yr}, \; t_k]$ — used for early stopping and LR scheduling (Section 4.5). Not used for HP comparison (that uses OOS).

2. **Embargo gap ($\Delta_{\text{embargo}} = 21$ days):** addresses residual temporal dependence (volatility clustering, momentum), not data overlap — zero shared daily returns exist even without embargo.

3. **OOS test period:** $t_k + \Delta_{\text{embargo}} + 1$ to $t_{k+1}$ (~6 months). Portfolio held fixed.

**Minimum training set:** ~10 years. **Holdout:** last ~3 years permanently reserved; the only uncontaminated evaluation of the final configuration. The holdout model is retrained on all data up to $t_{\text{holdout}}$ using $E^*$ epochs (median across walk-forward folds), with no validation set — same protocol as Phase B.

**Fold schedule** ($T_{\text{hist}} = 30$ years, 6-month OOS, 21-day embargo, 3-year holdout):

| Fold | Training end | Embargo | OOS period | OOS length |
|------|-------------|---------|------------|------------|
| 1 | Year 10.0 | 21 days | Year 10.1 – 10.6 | 6 months |
| 2 | Year 10.5 | 21 days | Year 10.6 – 11.1 | 6 months |
| ... | ... | ... | ... | ... |
| ~34 | Year 26.5 | 21 days | Year 26.6 – 27.0 | 6 months |
| **Holdout** | **Year 27.0** | **21 days** | **Year 27.1 – 30.0** | **~3 years** |

**Per-fold procedure — two phases.**

*Phase A — HP selection (all candidate configs).* For each HP config: (1) Train encoder on $[\text{start}, t_k - 2\text{yr}]$ with early stopping on validation $[t_k - 2\text{yr}, t_k]$; record best-checkpoint epoch $E^*_{\text{config}}$. (2) Build downstream pipeline (AU, $B_{\mathcal{A}}$, $\Sigma_z$, portfolio). (3) Evaluate on OOS → fold score. Select best config by OOS score (Section below).

*Phase B — Deployment run (best config only).* (4) Retrain encoder on **all** data $[\text{start}, t_k]$ — no validation set — for $E^*$ epochs (the best-checkpoint epoch from Phase A for the selected config). (5) Measure AU; define $\mathcal{A}$. (6) Build composite profiles; filter to $B_{\mathcal{A}}$. (7) Pre-compute trailing $\sigma_{i,t}$ series; rescale $\tilde{B}_{\mathcal{A},t}$ per date; estimate $\hat{z}_t$; compute $\Sigma_z$ and $D_\varepsilon$. (8) Rescale $\tilde{B}_{\mathcal{A}}^{\text{port}}$ at OOS start date; optimize weights. (9) Evaluate on OOS period.

Phase B reflects actual deployment: in production, no validation holdout is maintained — the model trains on all available data. The OOS metrics reported across folds come from Phase B models, ensuring the backtest faithfully represents production behavior. Phase A exists solely to select *which* config to deploy.

**Stopping criterion for Phase B.** The fixed epoch count $E^*$ replaces early stopping. This is justified because: (a) the VAE's primary overfitting protection comes from the KL term (auto-pruning), dropout, and weight decay — not early stopping, which is a secondary safety net; (b) more training data implies convergence at least as fast, so $E^*$ is a conservative upper bound; (c) it is deterministic — no new degree of freedom. For robustness, $E^*$ can be taken as the **median** of $E^*_{\text{config}}$ across folds rather than the single fold's value. **Sanity check:** if the training loss at epoch $E^*$ in Phase B is more than 20% lower than in Phase A, flag the fold for inspection — the encoder may be exploiting the fresh data aberrantly.

**Computational cost.** Phase B adds one training run per fold (the selected config only). For 34 folds testing 4 configs each: 34 additional runs on top of 136, i.e., +25% total compute.

#### Metrics

Organized in three layers, all computed out-of-sample:

**Layer 1 — VAE quality.** OOS reconstruction error per regime (target: OOS/train MSE < 1.5; crisis/normal ratio in [0.5, 2.0]). Active units AU (target: AU ≥ max(0.15K, AU_PCA); AU ≤ min(0.85K, AU_max_stat)). Latent space stability across retrainings (target: Spearman ρ > 0.85). **Measurement:** since AU may differ between retrainings (rendering direct vector comparison invalid), stability is computed as Spearman rank correlation of **pairwise inter-stock distances**: for the $n$ stocks present in both retrainings, compute the $\binom{n}{2}$ pairwise Euclidean distances in each active latent subspace, then rank-correlate the two distance vectors. This metric is invariant to dimension changes, rotations, and sign flips.

**Layer 2 — Risk model quality.** Covariance forecast error vs Ledoit-Wolf baseline. Minimum variance portfolio variance ratio (target: [0.8, 1.2]). Factor explanatory power (target: > max(0.50, EP_PCA + 0.10)).

**Layer 3 — Portfolio quality.**

| Metric | What it catches | Primary / diagnostic |
|--------|-----------------|---------------------|
| OOS factorial entropy | Strategy's own objective OOS | **Primary** |
| Annualized volatility | Overall risk level | Primary |
| Maximum drawdown | Tail risk exposure | Primary |
| Crisis-period return | Diversification failure during stress | Primary |
| Annualized return | Sanity — shouldn't destroy value | Diagnostic |
| Sharpe ratio | Risk-adjusted performance | Diagnostic |
| Turnover at rebalancing | Risk model instability (target < 30%) | Diagnostic |
| Diversification ratio (DR) | Asset-level complement to entropy | Diagnostic |

#### Hyperparameter evaluation (Phase A)

**Structural hyperparameters** ($K$, $T$, architecture): require full retraining per fold — coarse grid across full walk-forward.

**Training hyperparameters** ($\gamma$, $\sigma^2_0$, $\lambda_{\text{co}}$, $\eta_0$): selected within each fold via the nested temporal split (training subset $[\text{start}, t_k - 2\text{yr}]$ + validation subset $[t_k - 2\text{yr}, t_k]$). The validation set is shared with early stopping (Section 4.5) — it determines *when* to stop training for each config and records $E^*_{\text{config}}$. The *comparison between configs* uses the OOS fold score below, computed on data unseen during both training and early stopping. Once the best config is selected, Phase B retrains on all data up to $t_k$ (Section above). The optimizer (Adam) and its momentum parameters ($\beta_1$, $\beta_2$) are fixed; the learning rate $\eta_0$ is the primary tunable optimization parameter.

**Selection criterion:**

$$\text{Score} = \bar{\hat{H}}_{\text{OOS}} - \lambda_{\text{pen}} \cdot \max(0, \; \text{MDD}_{\text{OOS}} - \text{MDD}_{\text{threshold}}) - \lambda_{\text{est}} \cdot \max(0, \; 1 - R_{\Sigma})$$

where $\hat{H} = H(w) / \ln(\text{AU})$ is the **normalized entropy** (diversification utilization rate), $\in [0, 1]$, and MDD is expressed as a **fraction** $\in [0, 1]$ (e.g., a 25% drawdown is 0.25, not 25). $\bar{\hat{H}}_{\text{OOS}}$ is the median across OOS folds. The term $R_\Sigma = N_{\text{obs}} / (\text{AU}(\text{AU}+1)/2)$ is the observations-to-parameters ratio of $\Sigma_z$; the penalty activates when $R_\Sigma < 1$ (severely underestimated covariance). **Baseline:** $\lambda_{\text{est}} = 2$.

**Why normalize H.** Raw entropy $H$ has a maximum of $\ln(\text{AU})$, which varies across configurations: AU = 40 → $H_{\max}$ = 3.69; AU = 120 → $H_{\max}$ = 4.79. Without normalization, the score structurally favors higher-AU configurations regardless of diversification quality. The normalized $\hat{H}$ measures how well the portfolio uses its *available* factor capacity — 1.0 = perfect equi-contribution across all principal factors. Typical range for a well-diversified portfolio: $\hat{H} \approx 0.80$–$0.95$.

Both terms now live in $[0, 1]$, so $\lambda_{\text{pen}}$ has a direct interpretation: points of diversification utilization sacrificed per unit of excess drawdown. $\lambda_{\text{pen}} = 5$ means 1% of excess MDD (0.01) costs 5% of diversification capacity (0.05) — aggressive but consistent with a risk-first mandate.

**Worked example** ($\text{MDD}_{\text{threshold}} = 0.20$, $\lambda_{\text{pen}} = 5$):

| | Config A | Config B |
|---|---|---|
| AU | 80 | 120 |
| $\bar{H}_{\text{OOS}}$ | 3.80 | 4.20 |
| $\bar{\hat{H}}_{\text{OOS}}$ | 3.80 / ln(80) = 0.867 | 4.20 / ln(120) = 0.877 |
| $\text{MDD}_{\text{OOS}}$ | 0.18 | 0.28 |
| Penalty | 0 | 5 × 0.08 = 0.40 |
| **Score (raw H)** | **3.80** | **3.80** (tie) |
| **Score (normalized)** | **0.867** | **0.477** (A wins) |

Config A uses its capacity almost as well and stays within the drawdown budget. Config B has more factors but the excess drawdown destroys its score — the desired behavior for a risk-minimization strategy.

Configurations failing hard constraints (AU < max(0.15K, AU_PCA), explanatory power < max(0.40, EP_PCA), OOS/train MSE > 3.0) are eliminated before scoring.

#### Robustness checks

**Fold sensitivity.** Report all metrics per fold. If performance is concentrated in a few folds, the result is not robust.

**Regime decomposition.** Aggregate metrics separately for crisis and calm folds. The strategy must protect during crises without sacrificing calm-period performance.

---

## 5. Tunable Parameters

This section consolidates all parameters that define a concrete instance of the strategy. They constitute the axes of analysis during implementation and backtesting.

### 5.1 Parameter inventory

◆ = concerns an alternative to the default choice — not a priority to resolve.

**Data**

| Parameter | Description | Ref. | Baseline | Range to explore |
|-----------|-------------|------|----------|-----------------|
| **Frequency** | Sampling frequency of return series | 4.2 | Daily | Daily, weekly |
| **$n$** | Number of stocks in the universe | 4.2 | 1,000 | 200 – 2,000 |
| **$T_{\text{année}}$** | Historical depth available for training (years) | 4.2 | 30 | 10 – 30+ |
| **$F$** | Features per time step | 4.2 | 2 (return + realized volatility) | 2 – 5 (+ volume, sector deviation…) |
| **Normalization** | Z-score computation scope | 4.2 | Per-window (μ, σ from the T days of each window) | — (resolved) |

**Universe**

| Parameter | Description | Ref. | Baseline | Range to explore |
|-----------|-------------|------|----------|-----------------|
| **Capi min (entry)** | Minimum float-adjusted market cap for universe entry | 4.2 | \$500M | \$200M – \$1B |
| **Capi min (exit)** | Exit threshold (buffer) | 4.2 | \$400M | 80% of entry threshold |
| **ADV min** | Minimum average daily dollar volume (trailing 3 months) | 4.2 | \$2M | \$500K – \$5M |
| **Listing history min** | Minimum continuous listing history for eligibility | 4.2 | 504 days (= T) | — (resolved, tied to T) |
| **Delisting return** | Imputed return for delistings without available data | 4.2 | -30% (NYSE/AMEX), -55% (Nasdaq) | Shumway convention |
| **$\alpha_{\text{trigger}}$** | Entropy threshold for exceptional rebalancing on delisting (fraction of $H$ at last rebalancing) | 4.2 | 0.90 | 0.80 – 0.95 |

**Encoder**

| Parameter | Description | Ref. | Baseline | Range to explore |
|-----------|-------------|------|----------|-----------------|
| **$T$** | Window length (days) | 4.3 | 504 | — (resolved) |
| **$K$** | Latent capacity ceiling (effective factor count is AU ≤ K) | 4.3 | 200 | 50 – 300 |
| **Architecture** | Encoder type | 4.3 | 1D CNN (multi-scale kernels, stride) | — (resolved) |
| **$s_{\text{train}}$** | Stride between consecutive windows (training dataset) | 4.5 | 1 | 1 – 21 (1 = daily, 21 = monthly) |
| **$s_{\text{inf}}$** | Stride between consecutive windows (inference / composite profiles) | 4.6 | 1 | — (resolved: fixed at 1) |

**Loss function**

| Parameter | Description | Ref. | Baseline | Range to explore |
|-----------|-------------|------|----------|-----------------|
| **$\sigma^2$** | Observation noise variance (learned scalar) | 4.4 | 1.0 (learned) | Clamped to $[10^{-4}, 10]$. **Mode P** (primary): learned. **Mode F**: frozen at 1.0, $D/2$ scaling retained. |
| **$\beta$** | KL weight | 4.4 | 1.0 (fixed) | **Mode P**: fixed at 1.0 (balance via $\sigma^2$). **Mode F**: replaced by $\beta_t$ annealing. **Mode A**: $\beta \in [1.0, 4.0]$ with learned $\sigma^2$, monitor $\sigma^2$/AU interaction. |
| **$T_{\text{warmup}}$** ◆ | KL annealing warmup period (**Mode F** only, mutually exclusive with learned $\sigma^2$) | 4.4 | — | 10–30% of total epochs |
| **$\lambda_{\text{co}}$** | Weight of co-movement loss (curriculum schedule) | 4.4 | $\lambda_{\text{co}}^{\max} = 0.5$ (Phases 1→2→3) | $\lambda_{\text{co}}^{\max} \in [0.1, 2.0]$ |
| **Dependence measure** ◆ | Co-movement target (if $\lambda_{\text{co}} > 0$) | 4.4 | Spearman | Spearman, Pearson, tail dependence, copula |
| **Distance metric $d$** ◆ | Distance in latent space (if $\lambda_{\text{co}} > 0$) | 4.4 | Cosine | Cosine, Euclidean |
| **Target mapping $g$** ◆ | Correlation → target distance (if $\lambda_{\text{co}} > 0$) | 4.4 | $g(\rho) = 1 - \rho$ | Linear, non-linear |
| **γ** | Crisis peak overweighting factor | 4.4 | 3.0 | 1.5 – 5.0 |
| **Crisis threshold** | VIX percentile for crisis labeling (expanding, per-fold) | 4.4 | 80th percentile | 75th – 90th percentile |
| **Crisis labeling** | How windows receive crisis weight | 4.4 | Continuous ($\gamma_{\text{eff}} = 1 + f_c(\gamma - 1)$) | Continuous (default), binary ($f_c > 0.20$) |
| **Crisis definition** | Source of stress labeling | 4.4 | VIX threshold | VIX-based, drawdown-based, regime model |

**Training**

| Parameter | Description | Ref. | Baseline | Range to explore |
|-----------|-------------|------|----------|-----------------|
| **Batch size** | Number of windows per gradient step | 4.5 | 256 | 64 – 512 |
| **Optimizer** | Gradient descent algorithm | 4.5 | Adam ($\beta_1$=0.9, $\beta_2$=0.999, $\varepsilon$=10⁻⁸) | — (resolved) |
| **$\eta_0$** | Initial learning rate | 4.5 | $10^{-4}$ | $5 \times 10^{-5}$ – $10^{-3}$ (log-scale) |
| **LR scheduler** | Learning rate decay strategy | 4.5 | ReduceLROnPlateau (factor=0.5, patience=5 epochs) | — |
| **Max epochs** | Computational ceiling for training | 4.5 | 100 | 50 – 200 |
| **Early stopping** | Convergence criterion (on validation ELBO) | 4.5 | Patience = 10 epochs, restore best weights | Patience 5 – 15 |
| **Batch composition** | How windows are sampled per training phase | 4.4, 4.5 | Curriculum: synchronous + stratified (Phases 1–2), random (Phase 3) | — (resolved, tied to $\lambda_{\text{co}}$ curriculum) |
| **Retraining frequency** | How often encoder weights are updated | 4.5 | Quarterly | Monthly – semi-annually |

**Aggregation and portfolio**

| Parameter | Description | Ref. | Baseline | Range to explore |
|-----------|-------------|------|----------|-----------------|
| **Aggregation function** | How local latent vectors become a composite profile | 4.6 | Mean | Mean, max-pool, weighted mean, learned |
| **Ratio winsorizing** | Percentile bounds on $\sigma_{i,t}/\bar{\sigma}_t$ before rescaling | 4.7 | [P5, P95] cross-sectional per date | [P1, P99] – [P10, P90] |
| **$\Sigma_z$ estimation window** | History used for factor covariance | 4.7 | Full available history | Full history (default) |
| **$\Sigma_z$ shrinkage** | Regularization of factor covariance | 4.7 | Ledoit-Wolf (analytical) | Ledoit-Wolf, Oracle Approximating Shrinkage |
| **Diversification metric** | Objective function for portfolio diversification | 4.7 | Factorial entropy | Entropy, diversification ratio (DR) |
| **Optimization mode** | Portfolio construction objective | 4.7 | Default ($-\lambda w^T\Sigma w + \alpha H$, $\mu = 0$) | Default, directional ($+w^T\mu$), exploratory ($\lambda=0$) |
| **$\bar{w}$** | Soft concentration threshold | 4.7 | 3% | 1% – 5% |
| **$\phi$** | Concentration penalty weight | 4.7 | 25 | 5 – 100 |
| **$w_{\max}^{\text{hard}}$** | Hard position cap | 4.7 | 5% | 3% – 10% |
| **$w_{\min}$** | Minimum position size if non-zero | 4.7 | 0.10% | 0.05% – 0.25% |
| **$\kappa_1$** | Turnover penalty weight (spread component) | 4.7 | 0.1 | 0.01 – 1.0 |
| **$\kappa_2$** | Turnover penalty weight (market impact component) | 4.7 | 7.5 | 1 – 50 |
| **$\bar{\delta}$** | Per-stock market impact threshold | 4.7 | 1% | 0.5% – 3% |
| **$\tau_{\max}^{\text{hard}}$** | Hard turnover cap — circuit breaker (one-way per rebalancing) | 4.7 | 30% | 20% – 50% |
| **λ** | Risk aversion (Mode 2) — only ratio α/λ matters | 4.7 | 1.0 | 0.5 – 5.0 (implied market: ~1.0) |
| **α** | Entropy reward weight (Mode 2) — marginal price of diversification in return units | 4.7 | Elbow of return-entropy frontier at $\lambda = 1.0$ | Sweep: {0, 0.01, 0.05, 0.1, 0.5, 1, 5}; rescale if λ ≠ 1 |

**Validation**

| Parameter | Description | Ref. | Baseline | Range to explore |
|-----------|-------------|------|----------|-----------------|
| **$\Delta_{\text{embargo}}$** | Embargo gap between training end and OOS start (trading days) | 4.8 | 21 (~1 month) | 21 – 126 (1 month – 6 months) |
| **OOS period length** | Duration of each out-of-sample fold | 4.8 | 126 days (~6 months) | 63 – 252 (quarterly – annually) |
| **Min training set** | Minimum history required for the first fold | 4.8 | 10 years (~2,520 days) | — (resolved) |
| **Holdout length** | Final reserved period for uncontaminated evaluation | 4.8 | ~3 years | 2 – 5 years |
| **$\text{MDD}_{\text{threshold}}$** | Maximum drawdown threshold for selection criterion penalty (fraction, not %) | 4.8 | 0.20 | 0.15 – 0.30 |
| **$\lambda_{\text{pen}}$** | Penalty scaling: points of $\hat{H}$ lost per unit of excess MDD | 4.8 | 5 | 1 – 10 |
| **$\lambda_{\text{est}}$** | Penalty scaling: points of $\hat{H}$ lost per unit of $\Sigma_z$ estimation deficit | 4.8 | 2 | 1 – 5 |

### 5.2 Key relationships between parameters

**Training volume.** The number of training windows per stock is $T_{\text{année}} \times 252 - T + 1$. With daily data: ~7,560 − T + 1 for 30 years. Increasing T reduces training volume (modestly for large $T_{\text{année}}$), while increasing stride $s$ reduces it proportionally (stride 21 ≈ 1/21 of stride 1). The total training set size scales as $n \times$ (windows per stock).

**K vs n.** K (the capacity ceiling) must be substantially smaller than $n$ for the factor decomposition to achieve dimensionality reduction. The effective factor count AU (determined by auto-pruning) is typically much smaller than K. $K \approx n$ would reproduce the original asset-space covariance with no compression benefit. Conversely, K too small forces the encoder to merge distinct risk structures into shared dimensions — detectable when AU ≈ K and factor explanatory power is below target.

**K vs $T_{\text{année}}$ × frequency.** Higher K provides more capacity for the auto-pruning mechanism to discover structure. $K = 200$ with daily data and $T_{\text{année}} = 30$ years provides ~7,560 observations per stock across varied regimes — sufficient for the encoder to populate ~60–120 active dimensions. However, the downstream $\Sigma_z$ estimation imposes a binding statistical constraint: $\text{AU}_{\max}^{\text{stat}} \approx 85$ for 30 years of daily data ($r_{\min} = 2$). Dimensions beyond $\text{AU}_{\max}^{\text{stat}}$ contribute to learning but are truncated before covariance estimation. Weekly data would reduce $N_{\text{obs}}$ to ~1,309, yielding $\text{AU}_{\max}^{\text{stat}} \approx 36$ — a much tighter constraint.

**T vs information content.** T controls what the encoder can observe within a single window. Short T (< 126 days) risks capturing only one market regime per window, producing context-specific rather than characteristic representations. Long T (> 504 days) increases the probability that a window spans a structural break, blurring the representation.

**$s$ vs training volume and redundancy.** Stride $s = 1$ maximizes the number of windows but introduces heavy overlap between consecutive samples ($T-1$ shared days). Larger strides reduce redundancy and training time, at the cost of fewer examples. The effective number of independent training samples is closer to $T_{\text{hist}} / T$ than to $T_{\text{hist}} - T + 1$.

**$n^2$ scaling (if co-movement loss is used).** The co-movement loss evaluates pairwise relationships. For $n = 1000$ in a batch, this is 499,500 pairs. Batch composition strategy (random subsampling, stratified by sector, rotating pairs) directly affects both training cost and the quality of the learned correlation structure.

**$\sigma^2$ / γ trade-off (and $\lambda_{\text{co}}$ if applicable).** The learned observation noise $\sigma^2$ and crisis weight γ operate on orthogonal axes — $\sigma^2$ controls the global reconstruction-KL balance, γ redistributes gradient attention between regimes within the reconstruction term via the continuous weighting $\gamma_{\text{eff}}^{(w)} = 1 + f_c^{(w)}(\gamma - 1)$. However, they interact indirectly: a large γ increases the average reconstruction loss (especially for windows with high $f_c$), which may push $\sigma^2$ upward, partially absorbing the intended crisis emphasis. The effective contribution framework (Section 4.4) monitors this via the empirical $\eta_{\text{eff}}$ and the per-regime MSE ratio ($\overline{\text{MSE}}_{\text{crisis}} / \overline{\text{MSE}}_{\text{normal}}$), which is the primary diagnostic for this interaction. If co-movement loss is used, high $\lambda_{\text{co}}$ relative to the reconstruction term strengthens correlation structure in latent space but may introduce linear bias.

---

## 6. Iterative Alternatives — Contingency Roadmap

If the base system (VAE with default configuration as described in Sections 4–5) underperforms benchmarks or exhibits specific failure modes during walk-forward validation, the following alternatives should be applied iteratively in order of diagnostic priority. Each alternative targets a specific failure mode and can be tested independently. The ordering reflects a progression from low-cost, high-impact adjustments to more fundamental architectural changes.

### 6.1 Phase 0 — Establish benchmark baselines (prerequisite)

**What:** Before any VAE iteration, implement four benchmark strategies on the same universe with the same walk-forward protocol:

- **1/N** (equal-weight): naive diversification — often difficult to beat OOS (DeMiguel, Garlappi & Uppal, 2009 [44]).
- **Minimum variance** (Ledoit-Wolf shrinkage on sample covariance): requires no factor model at all.
- **Classical risk parity** (inverse-volatility or equal risk contribution on shrunk covariance): same diversification philosophy, far lower complexity.
- **PCA factor risk parity** (Bai & Ng IC₂ to select $k$ factors, then entropy maximization on PCA factor contributions): same objective function as the VAE strategy but with linear factors.

**Problem addressed:** absence of credible performance reference. Without these baselines, it is impossible to determine whether the VAE's added complexity produces measurable OOS benefit. DeMiguel et al. (2009) demonstrated that with typical estimation windows, 1/N beats most sophisticated optimization methods — the benefit of complexity is often absorbed by estimation error. If the VAE strategy does not surpass these benchmarks, its additional complexity is not justified.

**Cost:** 2–4 weeks. All four share the same data pipeline and walk-forward infrastructure. Full specification in `Latent_risk_factor_benchmark.md`.

### 6.2 Iteration 1 — Exponentially-weighted composite aggregation

**What:** Replace the simple mean aggregation (Section 4.6) with an exponentially-weighted mean:

$$\bar{\mu}_i = \frac{\sum_{t} \lambda^{M-t} \cdot \mu_i^{(t)}}{\sum_{t} \lambda^{M-t}}$$

where $\lambda \in (0, 1)$ is a decay factor controlling the half-life, and $M$ is the index of the most recent window. The half-life $h = -\ln 2 / \ln \lambda$ becomes a tunable hyperparameter (baseline range: 3–10 years).

**Problem addressed:** stationarity of factor structure. The simple mean gives equal weight to all historical windows, diluting structural transformations. Exponential weighting preserves crisis memory (all windows still contribute, with diminishing weight) while allowing the composite to track structural evolution. The half-life parameter controls the tradeoff — short half-life emphasizes recent regime, long half-life preserves more historical memory.

**Diagnostic trigger:** OOS performance degrades for stocks that have undergone significant business model changes; latent stability metric (Spearman $\rho$ on inter-stock distances) drops below 0.80 between consecutive retrainings for a subset of stocks.

**Cost:** minimal — changes only the aggregation function. No retraining required. Can be evaluated on existing VAE outputs.

### 6.3 Iteration 2 — Σ_z estimation with regime-aware blending

**What:** Replace full-history Σ_z estimation with a blended estimator:

$$\hat{\Sigma}_z^{\text{blend}} = \omega \cdot \hat{\Sigma}_z^{\text{full}} + (1 - \omega) \cdot \hat{\Sigma}_z^{\text{recent}}$$

where $\hat{\Sigma}_z^{\text{full}}$ is the current full-history Ledoit-Wolf estimate and $\hat{\Sigma}_z^{\text{recent}}$ is a trailing-window estimate (e.g., 5 years). $\omega \in [0.3, 0.7]$ is validated in walk-forward.

**Problem addressed:** the full-history Σ_z estimation preserves crisis-era correlations permanently (anti-cyclical, by design), but may overweight correlation structures from distant regimes that are no longer representative. If factor correlations have genuinely shifted (e.g., post-GFC regulatory changes altered interbank correlation dynamics), the full-history estimate is slow to adapt.

**Diagnostic trigger:** Layer 2 validation metric — covariance forecast error vs Ledoit-Wolf baseline exceeds acceptable range; minimum variance portfolio variance ratio falls outside [0.8, 1.2].

**Cost:** low — no retraining; only changes the downstream covariance estimation step.

### 6.4 Iteration 3 — Directional mode activation (add return views)

**What:** Switch from default mode ($\mu = 0$) to directional mode ($\mu \neq 0$) by injecting external return views. Candidate signal sources: trailing 12-1 month momentum, valuation ratios (earnings yield, book-to-market), analyst consensus estimates, dividend yield.

**Problem addressed:** pure risk diversification (default mode) may underperform in equity-only contexts where factor Sharpe ratios have meaningful dispersion. Clarke, de Silva & Thorley (2013) showed that factor risk parity produces suboptimal Sharpe ratios when factor return expectations differ materially. The directional mode reuses the same SCA solver and adds one term ($w^T\mu$) to the objective — the risk infrastructure remains unchanged.

**Diagnostic trigger:** the default mode matches or exceeds benchmarks on diversification metrics (entropy, max drawdown) but lags on risk-adjusted return (Sharpe, Calmar) relative to minimum variance or classical risk parity.

**Cost:** moderate — requires sourcing and validating return signals, but the optimization pipeline is already designed for this mode (Section 4.7).

### 6.5 Iteration 4 — Additional input features (F > 2)

**What:** Extend the input tensor beyond $F = 2$ (return + realized volatility) to include additional features per time step: trading volume (z-scored), intraday range, sector-relative return deviation, or realized skewness. The encoder's Inception head and decoder output layer adjust automatically (Section 4.2); the rest of the pipeline is unchanged.

**Problem addressed:** the VAE may fail to capture risk dimensions that are invisible in return and volatility time series alone. Volume dynamics carry information about liquidity risk and informed trading; intraday range captures microstructure stress; sector deviation isolates idiosyncratic vs systematic behavior. If the co-movement scaffolding (Section 4.4) shows low Spearman correlation between latent distances and observed fundamental similarities, additional features may provide the encoder with richer signal.

**Diagnostic trigger:** Layer 1 validation — factor explanatory power EP stagnates near $\text{EP}_{\text{PCA}} + 0.10$ (VAE barely exceeds linear baseline); AU stabilizes at a low level relative to AU_max_stat, suggesting the encoder is not finding much structure beyond what returns alone provide.

**Cost:** moderate — requires extending data preprocessing and retraining the VAE. Walk-forward cost scales linearly with the number of F configurations tested.

### 6.6 Iteration 5 — Contrastive loss (InfoNCE) replacing co-movement scaffolding

**What:** Replace the Spearman-based co-movement loss (Section 4.4) with InfoNCE contrastive loss. Positive pairs: windows from different stocks in the same temporal block whose trailing empirical correlation exceeds a threshold (e.g., $\rho > 0.5$). Negative pairs: randomly sampled from different temporal blocks or uncorrelated stocks. The contrastive objective directly pushes correlated stocks' latent representations closer and uncorrelated stocks' representations apart, without assuming a specific functional form for the dependency.

**Problem addressed:** the Spearman co-movement loss imposes a linear correlation structure as scaffolding. While the curriculum schedule (scaffolding → annealing → free) is designed to release this constraint, the scaffolding phase may bias the latent space geometry in ways that the free phase cannot fully undo. InfoNCE is metric-agnostic — it optimizes for mutual information between positive pairs without prescribing the nature of the dependency. This could better capture tail co-movements, conditional correlations, and non-linear dependencies that Spearman misses.

**Diagnostic trigger:** latent stability metric (Spearman $\rho$ on inter-stock distances) is high, but OOS covariance forecast quality is poor — suggesting the latent space is geometrically stable but not aligned with actual risk dependencies. Alternatively, the free refinement phase (Phase 3) shows significant latent space drift from the scaffolding-phase structure, indicating the Spearman scaffolding was a poor initial geometry.

**Cost:** moderate to high — requires implementing InfoNCE loss, tuning temperature parameter, and defining positive/negative pair sampling strategy. Full retraining required.

### 6.7 Iteration 6 — PCA factor risk parity (VAE replacement)

**What:** Replace the entire VAE pipeline with PCA on rolling return windows. Extract $k$ factors (Bai & Ng IC₂, typically 5–15), compute factor risk contributions, and apply the same entropy-based optimization (Section 4.7) on PCA factors. The optimization infrastructure (SCA solver, constraints, cardinality enforcement) is reused entirely.

**Problem addressed:** if the VAE consistently fails to outperform the PCA baseline from Phase 0 (Section 6.1) on OOS metrics, the added complexity of the neural network is not justified. The VAE's theoretical advantage — capturing non-linearities, tail dependencies, and conditional co-movements — may not materialize on financial data with low signal-to-noise ratio (typical cross-sectional $R^2$ of 1–3%). PCA is interpretable, computationally cheap, and estimation-error-robust.

**Diagnostic trigger:** after Iterations 1–5, the VAE strategy still does not statistically outperform PCA factor risk parity across walk-forward folds (paired test on $\bar{H}_{\text{OOS}}$ scores, $p < 0.05$).

**Cost:** low — PCA is already implemented as a sanity check (Section 4.4). The main effort is connecting PCA factor outputs to the existing optimization pipeline.

### 6.8 Iteration 7 — Learned aggregation (attention-weighted composites)

**What:** Replace the fixed aggregation function (mean or exponential) with a learned attention mechanism over the trajectory of local latent vectors. A lightweight attention module (single-head, 1–2 layers) takes the sequence $\{\mu_i^{(t_1)}, ..., \mu_i^{(t_M)}\}$ as input and produces the composite $\bar{\mu}_i$ as a weighted combination where the weights are learned end-to-end. The attention weights are interpretable (they reveal which historical windows the model considers most relevant).

**Problem addressed:** both simple mean and exponential weighting impose a fixed weighting scheme that cannot adapt to the content of the latent vectors. A learned aggregation could identify which historical windows carry the most discriminative risk information — potentially assigning high weight to crisis windows while downweighting routine periods, without a fixed decay structure. This addresses both the stationarity problem and the dormant factor memory problem by letting the model learn the optimal memory structure.

**Diagnostic trigger:** Iteration 2 (exponential weighting) improves performance for structurally-transformed stocks but degrades crisis-period performance (the exponential decay forgets crisis-era vectors too aggressively). The optimal half-life varies significantly across stocks.

**Cost:** high — requires architectural modification, joint training with the VAE, and adds hyperparameters (attention dimension, number of heads). Risk of overfitting to historical crisis timing. Should only be attempted if simpler aggregation alternatives have been exhausted.

### Summary of iteration priority

| Priority | Alternative | Problem targeted | Cost | Prerequisite |
|----------|-------------|-----------------|------|--------------|
| 0 | Benchmark baselines (1/N, min-var, risk parity, PCA) | No performance reference | Low | None |
| 1 | Exponentially-weighted aggregation | Non-stationarity of factor structure | Minimal | Phase 0 |
| 2 | Regime-aware Σ_z blending | Stale factor correlations | Low | Phase 0 |
| 3 | Directional mode (return views) | Suboptimal risk-adjusted return in equity-only | Moderate | Phase 0 |
| 4 | Additional input features (F > 2) | Insufficient signal in returns + vol alone | Moderate | Iterations 1–2 |
| 5 | Contrastive InfoNCE loss | Linear scaffolding bias; non-linear dependencies missed | Moderate–High | Iterations 1–3 |
| 6 | PCA factor risk parity (VAE replacement) | VAE complexity not justified by OOS performance | Low | Phase 0 + Iterations 1–5 |
| 7 | Learned aggregation (attention) | Fixed weighting suboptimal; heterogeneous stock histories | High | Iterations 1–2 |

**Guiding principle:** at each iteration, the simplest alternative that addresses the diagnosed failure mode should be tried first. The walk-forward protocol (Section 4.8) provides the evaluation framework for every iteration. If the VAE system does not outperform the PCA baseline after Iterations 1–5, Iteration 6 (abandoning the VAE) should be adopted — the intellectual elegance of the architecture does not justify operational complexity without measurable OOS benefit.

---

*Working document — Version 4.1*

---

## Appendix A — `build_vae.py`

This appendix contains the complete Python function that constructs the VAE from the 5 variable parameters $(n, T, T_{\text{année}}, F, K)$ plus the optional training stride $s_{\text{train}}$. All sizing rules from Section 4.3 are implemented here; the function first verifies the capacity-data constraint $r = P_{\text{total}} / N \leq r_{\max}$ (always at $s = 1$) before instantiating the model, and reports the actual training dataset size $N_{\text{train}}$ for computational planning.

**Usage:**
```python
from build_vae import build_vae

model, info = build_vae(n=1000, T=504, T_annee=30, F=2, K=200, s_train=1)
# info = {'L': 5, 'channels': [...], 'P_total': ..., 'r': 2.31,
#         'N': 7_057_000, 'N_train': 7_057_000, 's_train': 1, ...}

model, info = build_vae(n=1000, T=504, T_annee=30, F=2, K=200, s_train=21)
# info = {'r': 2.31,                    ← unchanged (always computed at s=1)
#         'N': 7_057_000,               ← capacity reference
#         'N_train': 336_000, ...}      ← actual dataset size for planning
```

**Source code:**

```python
"""
build_vae.py — Construction du VAE pour la découverte de facteurs de risque latents.

L'architecture est entièrement déterminée par 5 paramètres variables :
    n        : nombre d'actions dans l'univers
    T        : longueur de la fenêtre d'entrée (jours de trading)
    T_annee  : profondeur historique en années (converti en T_hist = T_annee × 252)
    F        : features par pas de temps (1 = rendement seul, 5 = OHLCV)
    K        : dimension de l'espace latent (nombre de facteurs)

Tous les autres hyperparamètres sont fixés à des valeurs justifiées par la
littérature (cf. Section 4.3 du document). La fonction vérifie la contrainte
capacité-données P_total / N ≤ r_max avant d'instancier le modèle.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F_torch


# ═══════════════════════════════════════════════════════════════════════════════
# Hyperparamètres fixes
# ═══════════════════════════════════════════════════════════════════════════════

K_HEAD      = (5, 21, 63)    # Kernels Inception (≈ semaine / mois / trimestre)
C_BRANCH    = 48             # Filtres par branche (InceptionTime default)
C_HEAD      = len(K_HEAD) * C_BRANCH  # = 144
K_BODY      = 7              # Kernel corps résiduel (He et al., 2016)
STRIDE      = 2              # Sous-échantillonnage par bloc
ALPHA_PROJ  = 1.3            # Ratio de compression
C_MIN       = 384            # Largeur minimale dernière couche
DROPOUT     = 0.1            # Dropout (Srivastava et al., 2014)
BETA        = 1.0            # Poids KL (Kingma & Welling, 2014)
WEIGHT_DECAY = 1e-5          # Régularisation L2


# ═══════════════════════════════════════════════════════════════════════════════
# Règles de dimensionnement
# ═══════════════════════════════════════════════════════════════════════════════

def _round_to(x: float, base: int = 16) -> int:
    """Arrondi au multiple de base le plus proche, plancher = base."""
    return max(base, int(round(x / base)) * base)


def compute_depth(T: int) -> int:
    """Règle 1 — L(T) = max(3, ⌈log₂(T / k_max)⌉ + 2)."""
    k_max = max(K_HEAD)
    return max(3, math.ceil(math.log2(T / k_max)) + 2)


def compute_final_width(K: int) -> int:
    """Règle 2 — C_L(K) = round₁₆(max(C_min, ⌈α × 2K⌉))."""
    return _round_to(max(C_MIN, math.ceil(ALPHA_PROJ * 2 * K)))


def compute_channel_progression(L: int, C_L: int) -> list[int]:
    """Règle 3 — Interpolation géométrique C_head → C_L."""
    channels = [C_HEAD]
    for l in range(1, L + 1):
        c = C_HEAD * (C_L / C_HEAD) ** (l / L)
        channels.append(_round_to(c))
    return channels


def compute_temporal_sizes(T: int, L: int) -> list[int]:
    """Taille temporelle après chaque bloc stride-2."""
    sizes = [T]
    t = T
    for _ in range(L):
        t = (t - 1) // 2 + 1
        sizes.append(t)
    return sizes


# ═══════════════════════════════════════════════════════════════════════════════
# Comptage analytique des paramètres
# ═══════════════════════════════════════════════════════════════════════════════

def count_encoder_params(F: int, K: int, channels: list[int]) -> int:
    p = 0
    # Tête Inception
    for k in K_HEAD:
        p += F * C_BRANCH * k + C_BRANCH + 2 * C_BRANCH
    # Corps résiduel
    L = len(channels) - 1
    for l in range(1, L + 1):
        c_in, c_out = channels[l - 1], channels[l]
        p += c_in * c_out * K_BODY + c_out + 2 * c_out   # Conv1 + BN
        p += c_out * c_out * K_BODY + c_out + 2 * c_out   # Conv2 + BN
        p += c_in * c_out + c_out + 2 * c_out              # Skip 1×1 + BN
    # Projection (μ et log σ²)
    p += channels[-1] * K + K    # μ
    p += channels[-1] * K + K    # log σ²
    return p


def count_decoder_params(F: int, K: int, channels: list[int],
                         T_compressed: int) -> int:
    p = 0
    C_L = channels[-1]
    # Projection initiale
    p += K * (C_L * T_compressed) + (C_L * T_compressed)
    # Corps transposé (canaux inversés)
    L = len(channels) - 1
    for l in range(L):
        c_in = channels[L - l]
        c_out = channels[L - l - 1]
        p += c_in * c_out * K_BODY + c_out + 2 * c_out   # ConvT + BN
        p += c_out * c_out * K_BODY + c_out + 2 * c_out   # Conv + BN
        p += c_in * c_out + c_out + 2 * c_out              # Skip + BN
    # Tête de sortie
    p += C_HEAD * F + F
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# Modules PyTorch
# ═══════════════════════════════════════════════════════════════════════════════

class InceptionHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in K_HEAD:
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_channels, C_BRANCH, k, padding=k // 2),
                nn.BatchNorm1d(C_BRANCH),
                nn.GELU(),
            ))

    def forward(self, x):
        return torch.cat([b(x) for b in self.branches], dim=1)


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        pad = K_BODY // 2
        self.conv1 = nn.Sequential(
            nn.Conv1d(c_in, c_out, K_BODY, stride=STRIDE, padding=pad),
            nn.BatchNorm1d(c_out), nn.GELU())
        self.conv2 = nn.Sequential(
            nn.Conv1d(c_out, c_out, K_BODY, stride=1, padding=pad),
            nn.BatchNorm1d(c_out))
        self.skip = nn.Sequential(
            nn.Conv1d(c_in, c_out, 1, stride=STRIDE),
            nn.BatchNorm1d(c_out))
        self.act = nn.GELU()
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.dropout(self.act(self.conv2(self.conv1(x)) + self.skip(x)))


class TransposeResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        pad, out_pad = K_BODY // 2, STRIDE - 1
        self.conv1 = nn.Sequential(
            nn.ConvTranspose1d(c_in, c_out, K_BODY, stride=STRIDE,
                               padding=pad, output_padding=out_pad),
            nn.BatchNorm1d(c_out), nn.GELU())
        self.conv2 = nn.Sequential(
            nn.Conv1d(c_out, c_out, K_BODY, stride=1, padding=pad),
            nn.BatchNorm1d(c_out))
        self.skip = nn.Sequential(
            nn.ConvTranspose1d(c_in, c_out, 1, stride=STRIDE,
                               output_padding=out_pad),
            nn.BatchNorm1d(c_out))
        self.act = nn.GELU()
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.dropout(self.act(self.conv2(self.conv1(x)) + self.skip(x)))


class Encoder(nn.Module):
    def __init__(self, F: int, K: int, channels: list[int]):
        super().__init__()
        self.head = InceptionHead(F)
        self.body = nn.Sequential(*[
            ResBlock(channels[l], channels[l + 1])
            for l in range(len(channels) - 1)])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.proj_mu = nn.Linear(channels[-1], K)
        self.proj_logvar = nn.Linear(channels[-1], K)

    def forward(self, x):
        h = self.gap(self.body(self.head(x))).squeeze(-1)
        return self.proj_mu(h), self.proj_logvar(h)


class Decoder(nn.Module):
    def __init__(self, F: int, K: int, channels: list[int],
                 T_compressed: int, T_target: int):
        super().__init__()
        C_L = channels[-1]
        self.C_L, self.T_c, self.T_t = C_L, T_compressed, T_target
        self.fc = nn.Linear(K, C_L * T_compressed)
        rev = list(reversed(channels))
        self.body = nn.Sequential(*[
            TransposeResBlock(rev[l], rev[l + 1])
            for l in range(len(rev) - 1)])
        self.out_conv = nn.Conv1d(channels[0], F, kernel_size=1)

    def forward(self, z):
        h = self.fc(z).view(z.size(0), self.C_L, self.T_c)
        h = self.body(h)
        if h.size(-1) != self.T_t:
            h = h[:, :, :self.T_t] if h.size(-1) > self.T_t else \
                F_torch.pad(h, (0, self.T_t - h.size(-1)))
        return self.out_conv(h)


class LatentRiskVAE(nn.Module):
    def __init__(self, encoder, decoder, beta=BETA, learn_obs_var=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.learn_obs_var = learn_obs_var
        # Recon/KL balancing modes (mutually exclusive):
        #   Mode P (primary): learn_obs_var=True,  beta=1.0  → σ² learned
        #   Mode F (fallback): learn_obs_var=False, beta<1.0 → KL annealing, D/2 retained
        #   Mode A (advanced): learn_obs_var=True,  beta>1.0 → β-VAE + σ²
        # Init σ² = 1.0 (neutral for standardized inputs).
        self.log_obs_var = nn.Parameter(
            torch.tensor(0.0), requires_grad=learn_obs_var)

    @property
    def obs_var(self):
        """Observation noise σ², clamped to [1e-4, 10]."""
        return torch.clamp(torch.exp(self.log_obs_var), min=1e-4, max=10.0)

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def loss(self, x, x_hat, mu, logvar, gamma=3.0, crisis_frac=None):
        """
        Parameters
        ----------
        x, x_hat   : (B, F, T) — original and reconstructed windows.
        mu, logvar  : (B, K) — encoder outputs.
        gamma       : float — peak crisis overweighting factor.
        crisis_frac : (B,) float tensor, optional — fraction f_c of
            crisis days per window (values in [0, 1]).

        Balancing modes (set at __init__):
        - Mode P (learn_obs_var=True, beta=1):  D/(2σ²)·MSE + D/2·ln(σ²) + KL
        - Mode F (learn_obs_var=False, beta<1):  D/2·MSE + β_t · KL  (pass β_t as beta)
        - Mode A (learn_obs_var=True, beta>1):   D/(2σ²)·MSE + D/2·ln(σ²) + β·KL
        """
        # Reconstruction: MSE per sample, averaged over (T, F)
        recon_per_sample = F_torch.mse_loss(
            x_hat, x, reduction='none').mean(dim=(1, 2))     # (B,)
        # Continuous crisis weighting: γ_eff = 1 + f_c · (γ − 1)
        if crisis_frac is not None:
            gamma_eff = 1.0 + crisis_frac * (gamma - 1.0)    # (B,)
            recon = (recon_per_sample * gamma_eff).mean()
        else:
            recon = recon_per_sample.mean()
        # Scale reconstruction by observation noise (Modes P/A only).
        # Mode F (learn_obs_var=False): σ² frozen at 1.0, D/2 scaling
        # retained (D/(2σ²)|_{σ²=1} = D/2) — balance controlled by β_t
        # passed as self.beta. Without D/2, MSE (per-element mean ≈ 0.5)
        # is dwarfed by KL (sum over AU ≈ 80 nats) → posterior collapse.
        if self.learn_obs_var:
            D = x.size(1) * x.size(2)                        # F × T
            recon_term = D * recon / (2.0 * self.obs_var) \
                       + 0.5 * D * torch.log(self.obs_var)
        else:
            D = x.size(1) * x.size(2)                        # F × T
            recon_term = (D / 2.0) * recon
        # KL divergence: sum over K, mean over batch
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
        total = recon_term + self.beta * kl
        return total, recon, kl, self.obs_var.item()


# ═══════════════════════════════════════════════════════════════════════════════
# Fonction principale
# ═══════════════════════════════════════════════════════════════════════════════

def build_vae(n: int, T: int, T_annee: int, F: int, K: int,
              s_train: int = 1, r_max: float = 5.0,
              beta: float = BETA, learn_obs_var: bool = True) -> tuple[LatentRiskVAE, dict]:
    """
    Construit le VAE à partir des 5 paramètres variables.

    1. Conversion T_annee → T_hist = T_annee × 252
    2. Calcul N = n × (T_hist − T + 1)          (capacité, toujours à s=1)
    3. Calcul N_train = n × ⌊(T_hist − T)/s⌋ + n (taille réelle du dataset)
    4. Dimensionnement : L(T), C_L(K), canaux
    5. Comptage P_total = P_enc + P_dec
    6. Vérification P_total / N ≤ r_max
    7. Instanciation du modèle

    Modes d'équilibrage recon/KL (Section 4.4) :
    - Mode P (défaut) : learn_obs_var=True,  beta=1.0 → σ² appris
    - Mode F (fallback): learn_obs_var=False, beta<1.0 → KL annealing (β_t externe), D/2 conservé
    - Mode A (avancé) :  learn_obs_var=True,  beta>1.0 → β-VAE + σ² appris

    Le ratio r utilise toujours N(s=1) : le stride sous-échantillonne des
    fenêtres redondantes (T-1 jours partagés) sans réduire le contenu
    informationnel. La protection contre l'overfitting vient du KL, dropout,
    weight decay, early stopping et walk-forward OOS.

    Raises ValueError si la contrainte est violée.
    """
    T_hist = T_annee * 252
    N = n * (T_hist - T + 1)                          # capacity (s=1)
    N_train = n * ((T_hist - T) // s_train + 1)       # actual training set
    if N <= 0:
        raise ValueError(f"N = {N} ≤ 0 : T_hist doit être > T.")

    L = compute_depth(T)
    C_L = compute_final_width(K)
    channels = compute_channel_progression(L, C_L)
    temporal = compute_temporal_sizes(T, L)
    T_compressed = temporal[-1]

    P_enc = count_encoder_params(F, K, channels)
    P_dec = count_decoder_params(F, K, channels, T_compressed)
    P_total = P_enc + P_dec

    r = P_total / N
    if r > r_max:
        raise ValueError(
            f"Contrainte violée : r = {P_total:,}/{N:,} = {r:.2f} > {r_max}\n"
            f"  Canaux : {' → '.join(map(str, channels))}\n"
            f"  Leviers : ↑n, ↑T_annee, ↓K, ou ↑r_max.")

    encoder = Encoder(F, K, channels)
    decoder = Decoder(F, K, channels, T_compressed, T)
    model = LatentRiskVAE(encoder, decoder, beta=beta,
                          learn_obs_var=learn_obs_var)

    return model, {
        "L": L, "channels": channels, "temporal_sizes": temporal,
        "C_L": C_L, "T_compressed": T_compressed,
        "P_enc": P_enc, "P_dec": P_dec, "P_total": P_total,
        "N": N, "N_train": N_train, "s_train": s_train,
        "r": round(r, 2), "r_max": r_max, "T_hist": T_hist,
    }
```

---

## References

[1] Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022). Training Compute-Optimal Large Language Models. *NeurIPS 2022*. (Chinchilla scaling laws — LLMs)

[2] Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *The Review of Financial Studies*, 33(5), 2223–2273. (FC networks — asset pricing)

[3] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*. (ResNet — ImageNet)

[4] Nakkiran, P., Kaplun, G., Bansal, Y., et al. (2021). Deep Double Descent: Where Bigger Models and More Data Can Hurt. *J. Stat. Mech.*, 2021, 124003. (Double descent)

[5] Kingma, D. P. & Welling, M. (2014). Auto-Encoding Variational Bayes. *ICLR 2014*. (VAE)

[6] Higgins, I., Matthey, L., Pal, A., et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. *ICLR 2017*. (β-VAE)

[7] Fawaz, H. I., Lucas, B., Forestier, G., et al. (2020). InceptionTime: Finding AlexNet for Time Series Classification. *Data Mining and Knowledge Discovery*, 34, 1936–1962.

[8] Srivastava, N., Hinton, G., Krizhevsky, A., et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR*, 15, 1929–1958.

[9] Hendrycks, D. & Gimpel, K. (2016). Gaussian Error Linear Units (GELUs). *arXiv:1606.08415*.

[10] Rahaman, N., Baratin, A., Arpit, D., et al. (2019). On the Spectral Bias of Neural Networks. *ICML 2019*. (Frequency Principle)

[11] Gu, S., Kelly, B., & Xiu, D. (2021). Autoencoder Asset Pricing Models. *Journal of Econometrics*, 222(1), 429–450.

[12] Dai, B. & Wipf, D. (2019). Diagnosing and Enhancing VAE Models. *ICLR 2019*. (Two-Stage VAE — learned observation noise σ²)

[13] Lucas, J., Tucker, G., Grosse, R., & Norouzi, M. (2019). Don't Blame the ELBO! A Linear VAE Perspective on Posterior Collapse. *NeurIPS 2019*. (σ² controls posterior collapse)

[14] Asperti, A. & Trentin, M. (2020). Balancing Reconstruction Error and Kullback-Leibler Divergence in Variational Autoencoders. *IEEE Access*, 8, 199440–199448. (Dynamic reconstruction normalization)

[15] Fu, H., Li, C., Liu, X., et al. (2019). Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing. *NAACL 2019*. (Cyclical KL annealing)

[16] Bowman, S. R., Vilnis, L., Vinyals, O., et al. (2016). Generating Sentences from a Continuous Space. *CoNLL 2016*. (Monotonic KL annealing)

[17] Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019). Class-Balanced Loss Based on Effective Number of Samples. *CVPR 2019*. (Effective number of samples, √inverse-frequency weighting)

[18] Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411. (Shrinkage toward scaled identity)

[19] Almgren, R. & Chriss, N. (2001). Optimal Execution of Portfolio Transactions. *Journal of Risk*, 3(2), 5–39. (Linear + quadratic transaction cost decomposition — spread + market impact)

[20] Gârleanu, N. & Pedersen, L. H. (2013). Dynamic Trading with Predictable Returns and Transaction Costs. *The Journal of Finance*, 68(6), 2309–2340. (Closed-form optimal trading with quadratic costs — "aim portfolio" result)

[21] Olivares-Nadal, A. V. & DeMiguel, V. (2018). Technical Note — A Robust Perspective on Transaction Costs in Portfolio Optimization. *Operations Research*, 66(3), 733–739. (Quadratic transaction costs equivalent to robust optimization)

[22] Shumway, T. (1997). The Delisting Bias in CRSP Data. *Journal of Finance*, 52(1), 327–340. (Delisting return corrections: -30% NYSE/AMEX, -55% Nasdaq)

[23] Shumway, T. & Warther, V. (1999). The Delisting Bias in CRSP's Nasdaq Data and Its Implications for the Size Effect. *Journal of Finance*, 54(6), 2361–2379. (Survivorship bias impact on size premium)

[24] Boynton, W. & Oppenheimer, H. R. (2006). Anomalies in Stock Market Pricing: Problems in Return Measurements. *Journal of Business*, 79(5). (Over 40% of reported size premium attributable to delisting bias and bid-ask spread)

[25] Ince, O. S. & Porter, R. B. (2006). Individual Equity Return Data from Thomson Datastream: Handle with Care! *Journal of Financial Research*, 29(4), 463–479. (Data quality issues for international delisted stocks)

[26] Sicks, R., Korn, R. & Schwaar, S. (2023). Estimating the Value-at-Risk by Temporal VAE. *Risks*, 11(5), 79. (Auto-pruning property of VAEs on financial time series; TempVAE demonstrates that KL annealing enables proper auto-pruning, with inactive latent dimensions collapsing to the prior)

[27] Bai, J. & Ng, S. (2002). Determining the Number of Factors in Approximate Factor Models. *Econometrica*, 70(1), 191–221. (Information criteria IC1, IC2, IC3 for estimating the number of latent factors in large panels — used as PCA-based sanity check for AU)

[28] Bansal, N. & Stivers, C. (2023). Time-varying Equity Premia with a High-VIX Threshold and Sentiment. *SSRN Working Paper* 4477652. (Optimal VIX threshold for identifying market stress episodes is near the 80th to 85th percentile; equity risk premium steps up sharply above this threshold)

[29] Kingma, D. P. & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR 2015*. (Adaptive moment estimation optimizer — de facto standard for VAE training)

[30] Duan, Y., Wang, L., Zhang, Q. & Li, J. (2022). FactorVAE: A Probabilistic Dynamic Factor Model Based on Variational Autoencoder for Predicting Cross-Sectional Stock Returns. *AAAI 2022*. (VAE-based latent factor model for stock returns; Adam optimizer)

[31] Duan, Y. et al. (2024). RVRAE: A Dynamic Factor Model Based on Variational Recurrent Autoencoder for Stock Returns Prediction. *arXiv:2403.02500*. (Recurrent VAE for dynamic factor extraction in noisy markets)

[32] Koa, K. J. et al. (2023). Diffusion Variational Autoencoder for Tackling Stochasticity in Multi-Step Regression Stock Price Prediction. *CIKM 2023*. (Adam with LR $5 \times 10^{-4}$, 20 epochs per stock, best-validation checkpoint)

[33] Kritzman, T. & Cen, Y. (2025). Probabilistic Forecasting with VAR-VAE: Advancing Time Series Forecasting under Uncertainty. *Information Sciences*. (Adam with LR $10^{-3}$, 5000 epochs; minimal sensitivity to hyperparameter choice)

[34] Li, C. et al. (2024). Stockformer: A Price-Volume Factor Stock Selection Model Based on Wavelet Transform and Multi-Task Self-Attention Networks. *arXiv:2401.06139*. (Adam with LR $10^{-3}$, decay 0.1, 100 epochs)

[35] Wang, Y. et al. (2025). Agent-Based Modelling for Real-World Stock Markets under Behavioral Economic Principles. *arXiv:2307.12987*. (VAE backbone trained with Adam LR $10^{-3}$, 30 epochs)

[36] Feng, Y. & Palomar, D. P. (2015). SCRIP: Successive Convex Optimization Methods for Risk Parity Portfolio Design. *IEEE Transactions on Signal Processing*, 63(19), 5285–5300. (General nonconvex risk parity formulations solved via successive convex approximation; convergence to KKT point guaranteed)

[37] Roncalli, T. & Weisang, G. (2012). Risk Parity Portfolios with Risk Factors. *SSRN Working Paper* 2155159. (Factor risk parity via Shannon entropy, Herfindahl, and Gini indices; multiple solutions possible; two-stage decomposition factor space → asset space)

[38] Spinu, F. (2013). An Algorithm for Computing Risk Parity Weights. *SSRN Working Paper* 2297383. (Convex log-barrier formulation for vanilla ERC; Newton's method, provably convergent in <5 iterations for $n < 1000$)

[39] Meucci, A. (2009). Managing Diversification. *Risk*, 22(5), 74–79. (Effective Number of Bets — entropy of principal portfolio contributions as diversification index; mean-diversification efficient frontier)

[40] Locatello, F., Bauer, S., Lucic, M., Rätsch, G., Gelly, S., Schölkopf, B. & Bachem, O. (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations. *ICML 2019*. (Impossibility of unsupervised disentanglement without inductive bias — theoretical and empirical; motivates explicit co-movement scaffolding)

[41] Zhao, P. & Zhang, T. (2014). Stochastic Optimization with Importance Sampling for Regularized Loss Minimization. *ICML 2014*. (Stratified sampling for SGD: variance reduction 2–5× via optimal sampling from partitioned clusters)

[42] Zhang, C., Kjellström, H. & Mandt, S. (2017). Determinantal Point Processes for Mini-Batch Diversification. *UAI 2017*. (DPP-based mini-batch sampling reduces gradient variance by promoting intra-batch diversity via repulsive point processes)

[43] Burda, Y., Grosse, R. & Salakhutdinov, R. (2015). Importance Weighted Autoencoders. *ICLR 2016*. (Analysis of VAE latent space utilization; K should exceed intrinsic dimensionality by factor 1.5–2× for reliable auto-pruning)

[44] DeMiguel, V., Garlappi, L. & Uppal, R. (2009). Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy? *Review of Financial Studies*, 22(5), 1915–1953. (With typical estimation windows, 1/N outperforms most sophisticated portfolio optimization methods — complexity benefit absorbed by estimation error)

[45] Clarke, R., de Silva, H. & Thorley, S. (2013). Risk Parity, Maximum Diversification, and Minimum Variance: An Analytic Perspective. *Journal of Portfolio Management*, 39(3), 39–53. (Factor risk parity produces suboptimal Sharpe ratios when factor return expectations differ materially; equity-only risk parity less empirically supported than multi-asset)

---

## TODO

- [x] Expliquer d'abord l'architecture globale (encodeur-décodeur) avant de parler spécifiquement de l'encodeur — le lecteur doit comprendre dans quel cadre l'encodeur s'inscrit avant d'en voir les détails
- [x] Lister tous les éléments non encore définis clairement dans le document
- [x] Demander un esprit critique et objectif sur la stratégie et les affirmations faites en vérifiant leur véracité par la littérature scientifique