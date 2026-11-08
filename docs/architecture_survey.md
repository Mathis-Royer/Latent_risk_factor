# Deep latent risk-factor models for factor risk budgeting: a literature-grounded architecture study

Independent study, 2026-11-08. Written without reading the project's existing redesign or audit documents
(blindness rule). Inputs used from the repository: `docs/strategy_philosophy.md`, DVT v4.1 §1-4, and a skim of `src/vae`,
`src/inference`, `src/risk_model`. Every paper cited was found online and at least its abstract was read; the
reference list marks what was checked and how deeply. Statements that rest on my own reasoning or arithmetic are
labelled **[own analysis]**; recollections I could not confirm are labelled **[unverified]**.

---

## 0. Summary

1. **The learning target is wrong in the current design, not only the tuning.** A per-stock autoencoder that
   reconstructs z-scored windows learns a code for *the shape of one stock's path in one window*. The risk model
   needs something else: a *loading* `β_i` such that `Cov(r_i, r_j) ≈ β_iᵀ Σ_f β_j` holds **out of sample, across
   stocks**. No term of the reconstruction ELBO rewards this. The closest published deep risk models (Deep Risk Model,
   NeuralFactors, the neural covariance-cleaning network of Bongiorno et al.) all train on a
   **cross-sectional, forward-looking objective**: explained variance of next-period returns, joint likelihood of the
   next-period cross-section, or realized portfolio variance.
2. **Ranked shortlist** (details in §4):
   1. **Conditional probabilistic deep factor model (CPDFM)**: a network maps each stock's recent price/volume history
      and a market-state vector to time-varying loadings `β_{i,t}`, idiosyncratic scale and tail parameters. Factor
      returns are latent variables with a state-dependent covariance. Training maximizes the exact (Gaussian) or
      variational (Student-t) likelihood of the *next* cross-section. This is the VAE-family formulation that fits the
      goal.
   2. **Neural rotation-invariant covariance cleaner**, trained end-to-end on realized portfolio variance, then
      extended to the factor-risk-parity decision. It has the strongest published out-of-sample evidence on US equities
      of a few hundred to 1,000 names, but it is less a "factor discovery" model.
   3. **Regime-mixture latent factor model**: CPDFM plus a discrete regime latent with an explicit stress component.
      This is the most direct answer to "dormant risks that reactivate", and it can generate stress scenarios. It
      carries the highest overfitting risk because there are few crisis episodes.
   4. **Contrastive co-movement pretraining of the stock encoder**, used as initialization or regularizer for 1. It is
      cheap and safe, but on its own it is not a risk model.
3. **The VAE question.** A variational latent model is valuable when **the latent variable is the vector of factor
   returns `f_t` (or a regime `s_t`) shared by the whole cross-section on a date**, and the likelihood is that of
   observed returns. Then the KL/ELBO machinery gives a proper multivariate likelihood (a covariance loss), handles
   heavy tails, and yields a generative model for stress scenarios. When the latent variable is a per-stock window
   code, as now, the variational machinery adds nothing useful for risk. Its "automatic number of factors" (active
   units) is mostly a function of the decoder noise level, as the linear-VAE analysis of Lucas et al. (2019) shows.
4. **Capacity.** Successful deep covariance models on equities are **small**: about 40k parameters for the
   Bongiorno et al. network, a 2-layer GRU of width 32 for the Deep Risk Model, 256-wide layers with heavy dropout for
   NeuralFactors. The gains they report over strong linear estimators are real but modest: about 0.5-1.0 points of
   annualized volatility, or a few percent of likelihood. With price-only inputs, NeuralFactors' own ablation shows a
   large loss of joint likelihood relative to its full feature set. Expect small gains, and test for them with
   high-power statistics (likelihood/QLIKE, realized variance), not Sharpe ratios.
5. **Evaluation.** Use a hierarchy: (i) multivariate QLIKE / Gaussian NLL of the next cross-section; (ii) realized
   variance of minimum-variance and random portfolios, plus bias statistics; (iii) realized-versus-predicted factor
   risk contributions of the actual strategy, overall and in crisis subperiods. Use Diebold-Mariano/MCS tests on
   daily loss differentials. A 0.1 Sharpe difference between strategies is essentially undetectable in 25 years; a
   5% volatility reduction is detectable (§6.3).

---

## 1. First-principles diagnosis of the current formulation

What the code does (from the DVT and `src/`): windows of T=504 days × F=2 (returns, 21-day realized volatility),
z-scored per window, are encoded by a 1D-CNN into `μ(x) ∈ R^K`, K up to 200. A decoder reconstructs the window
(MSE, crisis-weighted). The KL term prunes dimensions, and optional co-movement / cross-sectional-R² auxiliary losses
exist. The window codes of each stock are **averaged over its history** (`inference/composite.py`, `method="mean"`)
to form a static B. Factor returns come from cross-sectional regression of returns on a volatility-rescaled B,
followed by `Σ = B Σ_z Bᵀ + D` and entropy maximization over principal factor risk contributions.

**[own analysis]** Six structural reasons why this cannot be expected to beat simple estimators, independent of any
tuning:

1. **Objective-target mismatch.** For one stock on daily data, the common component typically explains a minority
   of the variance. Industry-level R² in the 20-40% range is common, and the DRM paper reports cross-sectional R² of
   about 30% even with 10-20 factors (Lin et al., 2021). Minimizing window MSE therefore spends most of its capacity
   on idiosyncratic noise and on the temporal shape of the path. Only the systematic part matters for covariance, and
   it is never isolated by the loss. Per-window z-scoring also removes each window's volatility level, which is the
   first-order quantity of a risk model.
2. **A window code is not a loading.** In a factor model the window is `x_i ≈ β_i f_{t:t+T} + ε_i`. Its code depends
   on both `β_i` and the factor path `f_{t:t+T}` of that window. Two windows of the same stock from 2008 and 2017 have
   different codes even if `β_i` did not change, because `f` differs. Averaging codes across eras therefore mixes
   coordinates that do not mean the same thing. The averaged code is then used as a regressor to *define* factor
   returns, which assumes it is a loading. Nothing in training enforces equivariance: "same exposure, different factor
   path, same code".
3. **Static B destroys the regime information the project wants.** The "memory" principle is implemented as an
   arithmetic mean of exposures. Dormant crisis exposures are diluted by about (crisis windows)/(all windows), and the
   model cannot switch to a crisis configuration when a crisis returns. Memory belongs in the **function** that maps
   state to structure, trained on all history, not in a time average of outputs (§3.3).
4. **Active units do not count market factors.** For linear-Gaussian VAEs the optimum recovers the probabilistic-PCA
   solution, and the latent dimensions that stay active are those whose eigenvalue exceeds the observation-noise
   level (Lucas et al., NeurIPS 2019, which also ties posterior collapse to the noise variance). AU is therefore
   controlled by the assumed or learned noise variance and the β-annealing schedule. It is not a data-driven factor
   count in any identifiable sense. Locatello et al. (ICML 2019) show more generally that unsupervised latent
   coordinates are not identifiable without inductive biases.
5. **Factor risk parity amplifies the noisiest directions.** Equal risk contributions over principal factors imply
   `|β'_k| ∝ 1/√λ_k`. The smallest eigenvalues of an estimated covariance are the most biased downward and the least
   stable (the rationale of nonlinear shrinkage; Ledoit & Wolf 2017, 2022). With AU in the tens, the optimizer bets
   heavily on directions that are mostly estimation noise. Meucci's own effective-number-of-bets construction on
   principal components has been criticized because these components are statistical artefacts (the Meucci &
   Santangelo minimum-torsion line, seen only through secondary sources here). The risk model must therefore deliver
   **well-conditioned small eigenvalues**, which is exactly what reconstruction training does not target.
6. **The headline benchmark comparison conflates return and risk.** With μ=0 the strategy optimizes risk structure.
   Comparing it to 1/N by Sharpe ratio mixes risk-model quality with return premia (size, low-volatility, market
   beta) that 1/N harvests by construction. A risk model should be judged first by risk metrics (§6). That does not
   excuse underperformance, but it changes which test can falsify the model.

None of these points is fixed by more capacity, better tuning or a different encoder backbone. They are fixed by
changing **what is learned and against which loss**.

---

## 2. What a deep risk-factor model should learn, and why (Question 1)

### 2.1 The target

For portfolio risk budgeting with μ=0, the decision needs a one-step-ahead (horizon h = 1-21 days) conditional
covariance `Σ_{t+h|t}` for the current universe. It must be well conditioned in its small eigen-directions and
decomposable into factor contributions. The natural factor parameterization is

  `r_{t+1} = B_t f_{t+1} + ε_{t+1}`,  `f_{t+1} | 𝓕_t ~ D(0, Σ_f(s_t))`,  `ε_{i,t+1} ~ D(0, σ²_{i,t})`,

with `B_t = g_θ(stock features up to t, market state s_t)`. The deep part is `g_θ`, and optionally the maps
`s_t → Σ_f` and `s_t → regime`. The training objective should be **a proper scoring rule for the next cross-section**:
the joint Gaussian or Student-t log-likelihood of `r_{t+1}`. For Gaussians this is the multivariate QLIKE / Stein loss,
the loss that ranks covariance forecasts consistently under noisy proxies (Patton 2011; Laurent, Rombouts & Violante
2013). Decision-focused alternatives use realized variance of the portfolio built from `Σ`.

### 2.2 Families compared

| Family | What is learned | Objective | Fit to this goal | Evidence most relevant here |
|---|---|---|---|---|
| Reconstruction AE / VAE on per-stock windows (current) | Code of a window's shape | Window MSE + KL | **Poor**: no cross-sectional or forward term; codes are not loadings (§1) | No published evidence that it improves covariance forecasts that I could find |
| Conditional autoencoder, GKX-style | `β_{i,t} = g(z_{i,t})` from characteristics; factors = managed portfolios | Next-period return reconstruction across the cross-section | **Good form**, built for pricing (monthly, fundamentals) | Gu, Kelly & Xiu (2021): lower out-of-sample pricing errors than IPCA and others; IPCA (Kelly, Pruitt & Su 2019) is its linear parent |
| Probabilistic latent factor model with neural loadings (NeuralFactors; FactorVAE family) | `β_{i,t}, σ_{i,t}, ν_{i,t}`; latent `f_t` | Exact or IWAE likelihood of the next cross-section | **Best match** when trained for risk; FactorVAE/RVRAE/HireVAE target return prediction | NeuralFactors (Gopal 2024): best joint NLL and covariance MSE vs PPCA and a strong deep baseline on S&P 500, 2019-23 test |
| Explained-variance deep factors (Deep Risk Model) | K factor exposures from GRU + GAT | Multi-horizon cross-sectional R² + VIF penalty | **Good**; deterministic, no tail model | Lin et al. (2021): R² 31.7% vs 29.8% (Barra-style) and 30.1% (PCA); GMV vol 11.6% vs 12.6% / 18.6%, China A-shares 2017-20 |
| Neural spectral covariance cleaning, end-to-end | Lag weighting, marginal vols, eigenvalue map | Realized out-of-sample GMV variance | **Very strong evidence**; factors only implicit (eigenvectors) | Bongiorno, Manolakis & Mantegna (2025/26): 11.9% vs 12.4-12.6% vol (DCC, QIS, sample), Sharpe 1.06 vs 0.85-0.93, 1,000 US stocks, 2000-2024, with costs |
| Neural shrinkage / decision-focused covariance | Eigenvalue shrinkage or Cholesky factor | Portfolio risk | Promising; thin evidence so far | Yang et al. (2026), Kim et al. (2025): abstracts claim lower realized risk than Ledoit-Wolf / MSE-trained estimators |
| Neural GARCH / DCC hybrids | Volatilities (NN), correlations (DCC) | Likelihood | Good for the **dynamics** layer; most papers use small N | Boulet (2021) argues NNs struggle at hundreds of assets and keeps DCC for correlations; Engle, Ledoit & Wolf (2019) DCC-NLS is the linear reference |
| Low-rank copula / attention copula forecasters | Low-rank + diagonal covariance per step | Likelihood | Same structure as CPDFM, generic domain | Salinas et al. (NeurIPS 2019) low-rank Gaussian copula RNN scales to thousands of series; TACTiS / TACTiS-2 for non-Gaussian dependence |
| Graph / attention over the cross-section | Inter-stock message passing | Mostly return ranking | Useful as an encoder block; not a risk objective | MASTER (AAAI 2024) and HIST target return ranking; DRM's GAT ablation adds only ~0.2 pts of R² |
| Transformers / time-series foundation models | Generic sequence representations | Forecasting | Weak zero-shot on returns; better for volatility after fine-tuning or financial pretraining | Rahimikia, Ni & Wang (2025): off-the-shelf TSFMs weak; pretraining on financial data helps. Goel et al. (2025): fine-tuned TimesFM beats econometric RV benchmarks. Kronos (2025): 9% lower volatility MAE (authors' claim) |
| Generative (diffusion, GAN) | Joint return distribution | Score matching / adversarial | For **stress scenarios**, not the base risk model | Diffusion factor models (Chen, Xu, Xu & Zhang 2025): error rates driven by factor dimension, not N. Tail-GAN (Cont et al., Mgmt Sci): VaR/ES-consistent scenarios |
| End-to-end portfolio learning | Weights directly | Decision loss | Useful as a **final fine-tuning stage** with the risk-budgeting layer | Uysal, Li & Mulvey (2024): differentiable risk-budgeting layer, Sharpe 1.16 vs 0.79 for nominal RP on ETFs (small N) |
| Contrastive asset embeddings | Static similarity embedding | InfoNCE on co-movement pairs | Good **pretraining** for the stock encoder; static | Dolphin, Smyth & Dong (2024): embeddings beat return-correlation baselines on industry classification and portfolio tasks |

### 2.3 Recommended objective (answer to Q1)

**[own analysis, built from the evidence above]** Learn `B_t`, `D_t` and `Σ_f(s_t)` jointly by maximizing the
likelihood of next-period cross-sections over the whole training history. Use a multi-horizon variant
(h ∈ {1, 5, 21}, as in DRM's multi-task stability term), plus a light orthogonality/VIF penalty on `B_t`. Optionally
fine-tune with a decision loss: realized variance or realized risk-contribution dispersion of the portfolio the
optimizer would build. The reasons:

* It is a **proper covariance score**: the Gaussian NLL equals multivariate QLIKE up to constants, so improving it
  means improving `Σ`.
* It is **cross-sectional and forward**: it rewards exactly `Cov(r_i, r_j | 𝓕_t)`, including the small eigenvalues
  that factor risk parity depends on.
* It identifies the loadings up to a rotation, `B → BR`, `Σ_f → R⁻¹Σ_f R⁻ᵀ`. That is harmless because factor risk
  parity can be defined in the **principal basis of the systematic covariance `B_tΣ_fB_tᵀ`**, which is
  rotation-invariant, or with minimum-torsion bets.
* It makes "dormant risk" a **learnable conditional statement**: crisis-period cross-sections are in the training
  likelihood, and the network learns how loadings and factor covariance change with the stress state.

---

## 3. The VAE question (Question 2)

### 3.1 When a variational latent model adds real value here

| Potential benefit | Real here? | Condition |
|---|---|---|
| A **proper joint likelihood** for returns with latent factors (amortized factor analysis) | **Yes** | Latent = factor returns `f_t` shared across the cross-section; decoder = `B_t f_t + ε`. NeuralFactors does exactly this, and with a Gaussian prior the posterior is exact by moment matching |
| **Heavy tails / tail dependence** through Student-t factors and residuals | **Yes, moderately** | Student-t prior on `f` and t-residuals need IWAE/ELBO training. NeuralFactors uses IWAE (k=20) and its VaR calibration beats most baselines, though a univariate GARCH still wins on portfolio VaR calibration |
| **Regime posteriors** (`q(s_t | data)`) | **Potentially** | Discrete regime latent with a state-dependent `Σ_f`. Evidence in finance comes mostly from return-prediction models (HireVAE, IJCAI 2023) and nonparametric state-varying factors (Pelger & Xiong 2022: VIX-state factors explain more variation than constant ones) |
| **Generative stress scenarios** | **Yes** | Sampling `s = stress`, then `f ~ p(f | stress)` and `r = B f + ε`, gives coherent cross-sectional stress tests. Diffusion factor models and Tail-GAN are alternatives |
| **Uncertainty on exposures** (posterior variance of `β_i`) | **Small** | Can feed a robust optimizer, but no paper I found shows a portfolio gain from it |
| **Automatic factor count** (active units) | **No** | AU depends on the noise level and the KL schedule (Lucas et al. 2019). Use likelihood-based selection on validation folds instead |
| **Disentangled, interpretable factors** | **No** | Not identifiable without inductive biases (Locatello et al. 2019), and not needed for risk parity in a principal basis |
| **Smooth latent space for per-stock windows** | **No** | Smoothness of window codes has no link to covariance accuracy |

### 3.2 Best VAE-family formulation

A **conditional latent factor model whose latent variables are the date-level factor returns and (optionally) a
regime**, with amortized, neural loadings:

* Generative: `s_t ~ p(s_t | s_{t-1})` (optional), `f_{t+1} | s_t ~ t_ν(0, Σ_f(s_t))`,
  `r_{i,t+1} | f ~ t_{ν_i}(β_{i,t}ᵀ f, σ_{i,t})`, with `β_{i,t}, σ_{i,t}, ν_{i,t} = g_θ(x_{i,≤t}, m_{≤t})`.
* Inference: with a Gaussian `f` and Gaussian residuals the marginal is exact,
  `r ~ N(0, B Σ_f Bᵀ + D)`. Evaluate it in O(NK²) with Woodbury, with no encoder and no ELBO gap. With Student-t
  components, use a Gaussian moment-matched `q(f | r_{t+1}, B, σ)` as in NeuralFactors, trained by IWAE.
* This **is** a VAE in the technical sense: amortized variational inference of latents, ELBO/IWAE objective. The
  variational machinery serves the tails and the regime posterior, not the loadings.

So the owner's belief holds in a precise form: a probabilistic latent factor model is well suited **if the latent is
`f_t`, not a per-stock code**. The current architecture is a VAE in name but not in the sense that would deliver the
claimed benefits.

### 3.3 Where "memory of dormant risks" should live

**[own analysis]** Memory can be implemented in three ways, in increasing order of commitment:

1. **Long-memory estimation of `Σ_f`**: full history or a long half-life. This is cheap but blurs regimes.
2. **Regime-blended covariance**: `Σ = (1-π) Σ_calm + π Σ_stress`, with `Σ_stress` estimated on turbulent days.
   Chow, Jacquier, Kritzman & Lowry (FAJ 1999) blend outlier-sample and inlier-sample covariances so that the
   investor's regime weighting is explicit. `π` becomes a policy choice (anti-cyclical insurance) rather than a
   forecast.
3. **State-conditional structure learned on all history**: `B_t = g(x, s_t)` and `Σ_f(s_t)`. When the state returns
   to stress, the learned stress configuration reappears immediately. Pelger & Xiong (JBES 2022) is the
   nonparametric evidence that factor structure differs by state.

Combining 3 for the forecast with 2 for the policy gives a principled version of the anti-cyclical principle. The
optimizer sees the stress structure with a chosen weight `π` even in calm times.

Caveat from the literature: part of the observed "correlations rise in crises" is a volatility-conditioning artefact
(Forbes & Rigobon 2002). Downside co-movement is nevertheless genuinely asymmetric: Longin & Solnik (2001) reject
normality for the negative tail only, and Ang & Chen (2002) find downside correlations on average 11.6% above those
implied by normality. A model with state-dependent volatility but static correlations would capture only the first
effect. The dormant-risk claim must therefore be tested specifically on crisis subperiods (§6).

---

## 4. Architecture shortlist (Question 3)

Notation: N = 400-1,000 stocks per date, daily data 1995-2025 (~7,500 days), walk-forward with yearly refits.
Inputs are restricted to price and volume. Proxies replace missing data: dollar volume for size, Amihud
illiquidity, trailing beta, downside beta, volatility, momentum, reversal, maximum drawdown, skewness, and turnover
from volume.

### Candidate 1 (rank 1): Conditional probabilistic deep factor model (CPDFM)

* **Inputs per stock i at date t.** (a) The last L=252 daily returns, absolute returns and log dollar volume.
  Scale by a trailing volatility, not per-window z-scoring, and pass log-volatility as a separate input so the level
  survives. (b) About 15 hand-computable price/volume characteristics, cross-sectionally ranked (rank-normalization
  as in IPCA/GKX). **Market state `m_t`**: trailing market volatility at 5/21/63 days, cross-sectional dispersion,
  average pairwise correlation (top-eigenvalue share) over 21/63 days, market drawdown from its 1-year high, and the
  spread between short- and long-horizon realized volatility.
* **Blocks and sizes.**
  * Stock temporal encoder: 3-4 dilated causal conv blocks or a 2-layer GRU, width 32-64, giving `h_i` (dim 32).
    About 20-60k parameters, shared across stocks.
  * State encoder: 2-layer MLP (64), giving `e_t` (dim 8-16).
  * Loading head: `β_{i,t} = W_2 · φ(W_1[h_i, c_i] ⊙ FiLM(e_t))`. FiLM modulation by the state is what lets
    exposures switch by regime. K = 8-32 factors. A K=1 column fixed to the market (trailing beta) helps
    identification and stability.
  * Scale and tail heads: `σ_{i,t}` (softplus), optionally `ν_{i,t}` (softplus + 4).
  * Factor covariance: `Σ_f(e_t) = L(e_t) L(e_t)ᵀ` via a Cholesky head, **or** a DCC/EWMA recursion on the latent
    factor returns with a learned state-dependent intercept. The second option is safer.
  * Optional cross-sectional block: one attention layer over stocks on the date. DRM's ablation says the gain is
    marginal, so this is the first block to drop.
  * Total: about 50-150k parameters.
* **Output.** `Σ_{t+1|t} = B_t Σ_f(e_t) B_tᵀ + D_t` (N×N, positive definite by construction), factor-level
  decomposition, and per-stock tails.
* **Loss.** Exact Gaussian NLL of the next-day cross-section, `½ log|Σ| + ½ rᵀΣ⁻¹r` via Woodbury, averaged over dates.
  Add multi-horizon terms on 5- and 21-day cumulative returns, weight decay, dropout 0.1-0.25, and a light
  `tr((BᵀB)⁻¹)` VIF penalty (DRM uses λ=0.01). In a Student-t variant, replace the exact NLL by IWAE with k=10-20
  (NeuralFactors). Crisis weighting, if any, should apply to the likelihood of *dates*, not to reconstructions.
* **Regime change and dormant risks.** Through `e_t → (B_t, Σ_f)`, the training likelihood includes 2000-02, 2008-09,
  2011, 2020 and 2022, so the mapping "stress state → stress structure" is learned once and reused whenever the state
  recurs. Policy-level memory (§3.3) blends `Σ(e_stress)` with the current `Σ`.
* **Identification.** Only `B Σ_f Bᵀ` is identified. Normalize `Σ_f(calm reference) = I`, fix factor 1 to the
  market, apply a VIF penalty, and **define the risk-parity factors as eigenvectors of `B_tΣ_fB_tᵀ`** (or by
  minimum torsion), not as raw network outputs.
* **Data and compute.** One date's cross-section is one batch. About 6,000 training dates per late fold, ×500 stocks
  × 252 lags. One epoch is about 6,000 small forward passes, which runs in minutes on MPS. Thirty walk-forward refits
  × a few seeds fit Colab comfortably. Memory is dominated by the N×L input (500×252×3 floats, under 2 MB per
  batch).
* **Failure modes.**
  1. Low SNR overfitting of `β` to noise. Mitigate with small K, shared weights, dropout and early stopping on
     validation NLL.
  2. State-head overfitting to the few crisis episodes, since only about 5 independent stress regimes exist in 30
     years. Mitigate with low-dimensional `e_t`, monotone or smooth constraints, and leave-one-crisis-out validation.
  3. Degenerate solutions where `D` absorbs everything. Mitigate by flooring idiosyncratic variance and
     initializing `B` from PCA.
  4. Survivorship: without delisting returns, crisis tails are understated.
  5. Gains confined to the volatility (diagonal) part rather than correlations. Check the decomposition (§6).
* **Closest published evidence.** NeuralFactors (Gopal 2024): S&P 500 constituents, train 1996-2013, test 2019-2023,
  64 factors, attention/LSTM encoder. Test covariance MSE 0.246-0.282 vs 0.332 (deep BDG baseline) and 0.881 (PPCA).
  Joint NLL 0.556 vs 0.620/0.664. VaR calibration is worse than univariate GARCH on the portfolio. **Critically, its
  ablation shows returns-only inputs give validation joint NLL 0.466 vs 0.324 with all features**, and those features
  include fundamentals and options that the project cannot use. Deep Risk Model (Lin et al. 2021): explained-variance
  objective, 1.9 pts higher R² and 1 pt lower GMV volatility than a Barra-style model, on China A-shares with style
  factors as inputs. Gu, Kelly & Xiu (2021) and IPCA (Kelly, Pruitt & Su 2019) establish the conditional-loading
  form. Filipovic & Schneider (2024, arXiv) report that most of their out-of-sample gain over IPCA comes from **the
  covariance**, not the mean.
* **Matched baseline it must beat.** **The same model with linear loadings**: `B_t = C_t Γ` (IPCA on the same
  price/volume characteristics), the same state-dependent `Σ_f` and the same likelihood. Also beat static PCA-K, QIS,
  DCC-NLS and the AFM-DCC factor-plus-GARCH estimator of De Nard, Ledoit & Wolf (2021). Only the gap to the linear
  twin measures the value of the neural part.

### Candidate 2 (rank 2): End-to-end neural covariance cleaner, extended to factor risk budgeting

* **Inputs.** The N×L matrix of past returns (L≈1,200 days in the paper) and per-stock trailing volatilities. Market
  state can be appended for the extension.
* **Blocks** (Bongiorno, Manolakis & Mantegna; sizes from the paper):
  1. A per-lag transform `r̃ = (α_ℓ/β_ℓ)·tanh(252·β_ℓ·r)`, giving learned time weighting and winsorization
     (2,400 parameters).
  2. Eigendecomposition of the transformed correlation, with a **bidirectional LSTM over sorted eigenvalues**
     (width 64, about 34k parameters) that outputs cleaned inverse eigenvalues.
  3. A shared MLP (64-32-16) for marginal volatilities (about 2.8k parameters).

  The network is rotation-invariant and independent of N: train on random universes of 50-350 stocks, apply to
  1,000.
* **Output.** A cleaned `Σ` (or `Σ⁻¹`). Its eigenvectors are the implicit latent factors, and the cleaned
  eigenvalues are their variances, which is exactly what principal-factor risk parity consumes.
* **Loss.** Realized out-of-sample variance of the GMV portfolio over the next 5 days. **Extension [own analysis]**:
  a second head or fine-tune whose loss is the realized variance of the *factor risk parity* portfolio, plus a penalty
  on the realized dispersion of factor risk contributions. Use a differentiable risk-budgeting layer as in Uysal, Li
  & Mulvey (2024), or the closed-form GMV gradient route of Kim, Tae & Lee (2025).
* **Regime and dormant risk.** Implicit at best: the learned lag weights determine how much the past is remembered.
  Add the state input to the eigenvalue LSTM for conditioning. Dormant risks survive only if the lag window spans
  them, so this is weaker than Candidate 1 on the memory requirement.
* **Identification.** No identification problem: eigenvectors come from the data, and the network only cleans
  spectra and volatilities.
* **Data and compute.** The paper trains on CPU at about 200 s per epoch, 100 epochs per yearly model (several hours
  per model on a laptop, faster on GPU). 240 yearly models were trained in the paper. This fits the 16 GB constraint.
* **Failure modes.**
  1. GMV-specific training may not transfer to risk parity, which loads on the small eigenvalues differently.
  2. Seed sensitivity: the paper averages 10 runs per year, and sensitivity to initialization is also reported for
     end-to-end risk budgeting by Parra-Diaz & Castro-Iragorri (2025).
  3. Higher turnover: 57% vs 23% for QIS in the paper.
  4. Little interpretability.
* **Closest published evidence.** Long-only, 1,000 US stocks, 2000-2024, weekly rebalancing, realistic costs:
  annualized volatility 11.9% vs 12.4% (DCC), 12.5% (sample, analytical-optimal), 12.6% (QIS). Sharpe 1.06 vs
  0.85-0.93. In the frictionless random-universe test (n=300) the top rank is significant by bootstrap (p<10⁻⁶).
  Neural nonlinear shrinkage (Yang, Zhao & Chai 2026) and DFL covariance (Kim et al. 2025) point in the same
  direction, but I read only their abstracts.
* **Matched baseline.** QIS (Ledoit & Wolf, Bernoulli 2022) and DCC-NLS on the same lag window, plus the same
  network with identity eigenvalue map and equal lag weights. This is also the **strongest deep competitor** that
  Candidate 1 must beat.

### Candidate 3 (rank 3): Regime-mixture latent factor VAE (stress-aware, generative)

* **Structure.** Candidate 1 plus a discrete latent regime `s_t ∈ {calm, stress, (recovery)}` with a learned
  transition matrix (HMM prior). `Σ_f` and the FiLM modulation of `B` depend on `s_t`. A filtering posterior
  `q(s_t | m_{≤t})` comes from a small GRU over the market state. The ELBO covers the joint likelihood of the
  cross-sections and the KL of the regime posterior against the HMM prior. Student-t factors are optional.
* **Size.** About 60-200k parameters. The extra pieces are tiny (transition matrix, per-regime `Σ_f`, regime-specific
  FiLM parameters).
* **Output.** A regime probability `π_t`, the per-regime covariances `Σ^{(s)}`, and a mixture
  `Σ_t = Σ_s π_t^{(s)} Σ^{(s)} + between-regime term`. A sampler gives cross-sectional stress scenarios.
* **Dormant risk.** This is the most explicit treatment available. The stress regime's `B` and `Σ_f` are fit only on
  stress dates and retained forever. The portfolio layer can apply a floor `π ≥ π_min` (policy memory, §3.3).
* **Identification.** Label switching between regimes is resolved by ordering on market volatility. Within a regime,
  as Candidate 1.
* **Failure modes.**
  1. Few stress episodes (about 5-6 since 1995), so the stress-regime parameters are estimated from a few hundred
     dates. Keep the stress head low-rank (e.g. 2-4 extra factors) and regularize it toward the calm parameters.
  2. Regime collapse, where the posterior always picks one regime.
  3. Lag: filtering detects stress after it starts, which is why the policy floor is needed.
* **Evidence.** Indirect. HireVAE (IJCAI 2023) is a hierarchical regime-switching VAE factor model, evaluated on
  returns only. Pelger & Xiong (2022) find VIX-state-varying factors explain significantly more variation in
  individual stocks. Chow et al. (1999) support turbulent-sample covariance blending. Asymmetric downside
  correlation (Ang & Chen 2002; Longin & Solnik 2001) motivates a separate stress structure. I found **no**
  published out-of-sample covariance or portfolio test of a regime-VAE risk model on equities. Treat it as a research
  bet.
* **Matched baseline.** Candidate 1 without the regime latent, and the linear two-regime blend: PCA/QIS covariance
  estimated separately on turbulent vs calm days, blended with the same `π_t` rule.

### Candidate 4 (rank 4): Contrastive co-movement pretraining of the stock encoder

* **Idea.** Pretrain the temporal encoder of Candidate 1 with an InfoNCE objective. Two stocks are positives when
  their returns co-move significantly over sub-windows (the sampling scheme of Dolphin, Smyth & Dong 2024). Stress
  sub-windows are oversampled so that crisis co-movement is in the embedding. Then fine-tune within Candidate 1.
* **Size and compute.** Same encoder, about 30-60k parameters; pretraining runs in minutes to an hour.
* **Value.** It helps the encoder separate industries without sector labels, and it is an inductive bias toward
  co-movement, which is what reconstruction lacks. On its own it does not produce `Σ`.
* **Evidence.** Dolphin et al. (2024) report better industry classification and portfolio results than
  return-correlation baselines. No covariance-loss evaluation was found.
* **Matched baseline.** Candidate 1 with random initialization, and static correlation-clustering embeddings.

### Not shortlisted, with reasons

* **Per-stock window reconstruction (current) and its variants** (larger CNN, transformer encoder, more K): the
  objective mismatch of §1 persists whatever the backbone.
* **Return-prediction deep factor models** (FactorVAE, RVRAE, MASTER, AIPM transformer SDF): they optimize expected
  returns or SDF Sharpe. Their uncertainty outputs were not validated as covariance forecasts. Kelly et al.'s AIPM
  uses rich characteristics unavailable here.
* **Zero-shot time-series foundation models**: weak on daily financial returns zero-shot (Rahimikia, Ni & Wang
  2025). They are univariate in design and say nothing about cross-sectional covariance. A fine-tuned TSFM or Kronos
  could be tested later as a per-stock *volatility* head inside Candidate 1, against HAR-type and GARCH heads.
* **GAN / diffusion as the base risk model**: valuable for scenario generation (Tail-GAN; diffusion factor models),
  but harder to train and evaluate. They can be added as a stress-testing layer on top of Candidates 1 or 3.

---

## 5. Honesty on complexity (Question 4)

**Evidence that more capacity helps:**

* **Return prediction and SDF estimation.** Kelly, Malamud & Zhou (JF 2024) and Didisheim, Ke, Kelly & Malamud
  argue that out-of-sample performance improves with parameterization under ridge-type regularization. Kelly et al.
  (Large and Deep Factor Models) report gains with depth when data are long enough, and shallow models in short
  windows. **Contested**: Nagel (NBER 2025) shows the RFF "complexity" forecast degenerates into a volatility-timed,
  recency-weighted momentum rule that would fail on data with reversals, so the gain was mechanical. This literature
  is about first moments with rich characteristics and transfers weakly to covariance from price data.
* **Volatility forecasting.** Christensen, Siggaard & Veliyev (JFEc 2023) find ML (including NNs) beats HAR even with
  only HAR lags, more at longer horizons, on 29 Dow stocks. Goel et al. (2025) find fine-tuned TimesFM beats
  econometric RV models. The diagonal part of `Σ` is where non-linear models have the most consistent wins.
* **Joint distribution.** NeuralFactors beats PPCA by a wide margin and a strong deep baseline by about 10% in joint
  NLL. It also beats univariate GARCH on individual-stock NLL in some configurations, with a model trained for about
  24 GPU-hours on hundreds of stocks.

**Evidence that complexity is not the binding constraint, or overfits:**

* **The best covariance results are tiny networks with strong structure**: Bongiorno et al.'s 40k parameters built
  around the analytical GMV solution, and DRM's 32-unit GRU. Their gains over QIS/DCC are **0.5-1.0 points of
  volatility**, not multiples.
* **Inputs matter more than depth.** NeuralFactors' returns-only ablation loses a large part of the joint-NLL gain
  (0.466 vs 0.324). Its attention variant with *fewer* features had a better test covariance MSE (0.246) than with
  more features (0.282), which suggests variance across configurations comparable to the gains themselves.
* **Disentanglement and AU-based capacity selection are unreliable** (Locatello et al.; Lucas et al.).
* **Neural multivariate GARCH at hundreds of assets is hard**: Boulet (2021) keeps correlations in DCC for that
  reason.
* **Linear shrinkage-plus-dynamics baselines are strong and hard to beat**: QIS, DCC-NLS, and AFM-DCC, the last
  tested up to 1,000 stocks by De Nard et al. (2021).

**[own analysis] Rough information budget.** About 7,500 dates × 500 stocks gives about 3.75M return observations.
The covariance information is closer to *dates × effective factors*: one cross-section per day reveals about K
factor realizations. For crisis structure the budget is about 5-6 episodes and a few hundred dates. That supports
tens of thousands of shared parameters for the loading map, a handful of parameters for state effects, and very few
crisis-specific parameters. The current design, K up to 200 with a 504-day window CNN and decoder, spends capacity
where the target has no information. **Recommendation:** start at about 50k parameters and K≈8-16. Grow width or depth
only while validation NLL on held-out *years* improves, and report the capacity curve (NLL vs parameters) as part of
the evidence.

---

## 6. Evaluation that can falsify a deep risk model (Question 5)

### 6.1 Metrics, from most to least statistically powerful

1. **Multivariate likelihood / QLIKE** on the next cross-section (daily, plus 5- and 21-day aggregation), computed on
   the out-of-sample fold for every model in the same universe. QLIKE and MSE are the robust losses under a noisy
   covariance proxy (Patton 2011; Laurent, Rombouts & Violante 2013). Decompose the result into a **volatility part**
   (diagonal) and a **correlation part** (likelihood of returns standardized by forecast volatilities), to see where
   any gain comes from.
2. **Realized variance of covariance-dependent portfolios**: unconstrained and long-only GMV, plus 1,000 random
   long-only portfolios and the eigenportfolios. Report the bias statistic: the standard deviation of
   `r_p / σ̂_p` should be about 1. This follows the economic loss approach of Engle & Colacito (2006).
3. **The strategy's own claim**: realized vs predicted factor risk contributions. Track the entropy / effective
   number of bets realized ex post (Meucci 2009), the realized volatility of the risk-parity portfolio, and its
   drawdown.
4. **Crisis and dormant-risk tests**: the same metrics restricted to 2000-02, 2008-09, 2011, 2020 and 2022, *and*
   on the first 21 days after a regime switch (the "already diversified before the shock" claim). Use
   leave-one-crisis-out training to test whether memory generalizes to an unseen crisis.
5. **Sharpe, return and turnover**: reported, but not the deciding statistic for a μ=0 risk model.

### 6.2 Tests

* Diebold-Mariano on daily loss differentials with HAC errors, model by model against the matched linear twin.
* Model Confidence Set (Hansen, Lunde & Nason 2011) over all candidates and baselines.
* Ledoit & Wolf (2008) studentized time-series bootstrap for Sharpe-ratio differences, and its variance analogue
  for realized-variance differences.
* Seeds: at least 5 per configuration, reporting the spread. Bongiorno et al. average 10 runs.

### 6.3 Expected statistical power **[own analysis, approximate]**

* **Variance comparison.** For two daily return series with correlation ρ, `Var(log σ̂²_A/σ̂²_B) ≈ 2(κ−1)(1−ρ²)/T_eff`.
  Here κ is a kurtosis-like factor and T_eff the effective sample after volatility clustering. Take T=6,300 days
  (25 years), ρ=0.95, κ≈6 and T_eff≈T/3. The standard error is about 3.8% of variance, about 1.9% of volatility. A
  5% reduction in volatility (12.6% → 12.0%, the size of the Bongiorno gain) is about 10% of variance, a t-stat of
  about 2.5-3. **Detectable.**
* **QLIKE differences** use thousands of daily cross-sections and are typically even more powerful.
* **Sharpe comparison.** The standard error of a Sharpe difference is roughly `√(2(1−ρ)/years)`, annualized. With
  ρ=0.9 and 25 years it is about 0.09, so the gap of 0.11 between the last run (0.38) and 1/N (0.49) is not even
  significant. With ρ=0.8 it is about 0.13. **Undetectable** for differences of 0.1-0.2 unless the strategies are
  very highly correlated.
* **Crisis-only tests** have little power: five episodes. Report them as effect sizes with bootstrap intervals, not
  as pass/fail.

### 6.4 Protocol hygiene for these data

* Point-in-time universe with a dollar-volume rank (no market caps). Flag survivorship: without delisting returns,
  crisis losses and tails are understated for all models alike, so relative comparisons remain meaningful but
  absolute tail metrics do not.
* All features computed from data up to t, a 21-day embargo between training and test, and hyperparameters selected
  on a validation fold preceding the test fold. The final holdout stays untouched.
* Every deep model is compared with its **matched linear twin** (same inputs, same loss, same `Σ_f` dynamics), so
  the network's contribution is isolated from the contribution of the new objective. **[own analysis]** I expect much
  of any first improvement to come from the objective change alone, which is why the twin is mandatory.

---

## 7. Suggested order of experiments **[own analysis]**

1. Build the evaluation harness of §6: QLIKE/NLL decomposition, GMV and random-portfolio realized variance, bias
   statistic, DM/MCS tests. Run it on static PCA-K, QIS, DCC-NLS and the IPCA twin.
2. Implement Candidate 1 with a Gaussian likelihood and no state input. Compare it to its linear twin. This single
   experiment decides whether non-linear, price-derived loadings carry covariance information beyond linear
   characteristics.
3. Add the state input (FiLM) and the state-dependent `Σ_f`, then run leave-one-crisis-out tests.
4. Run Candidate 2 as the strong deep competitor. If it dominates Candidate 1 on QLIKE and GMV variance, use it for
   the covariance and keep Candidate 1 for factor decomposition, or hybridize: CPDFM loadings with a cleaned residual
   and factor spectrum.
5. Only if 3 shows a state effect, try the Student-t / IWAE version, then Candidate 3 with stress scenarios.
6. Last, fine-tune with the factor-risk-parity decision loss, and check that the portfolio layer's small-eigenvalue
   bets are stable (§1, point 5).

Kill criteria: if the deep model does not beat its linear twin on out-of-sample NLL/QLIKE and GMV realized variance
(DM p<0.05 across folds), the non-linearity is not earning its place *for this data*. That says nothing against the
probabilistic latent factor formulation itself, which also works with linear loadings.

---

## 8. References

Status legend: **V-full** = full text or detailed HTML read; **V-abs** = abstract and bibliographic record read;
**V-sec** = known only through secondary summaries found in search; **U** = not verified.

| # | Title | Authors | Year | Venue | Link | Status |
|---|---|---|---|---|---|---|
| 1 | Deep Risk Model: A Deep Learning Solution for Mining Latent Risk Factors to Improve Covariance Matrix Estimation | H. Lin, D. Zhou, W. Liu, J. Bian | 2021 | ICAIF '21 | https://arxiv.org/abs/2107.05201 | V-full (HTML: architecture, loss, Table 1) |
| 2 | NeuralFactors: A Novel Factor Learning Approach to Generative Modeling of Equities | A. Gopal | 2024 | arXiv 2408.01499 | https://arxiv.org/abs/2408.01499 | V-full (HTML: architecture, ablations, results) |
| 3 | End-to-End Large Portfolio Optimization for Variance Minimization with Neural Networks through Covariance Cleaning | C. Bongiorno, E. Manolakis, R. N. Mantegna | 2025/2026 | J. Finance and Data Science 12, 100179 (per arXiv) | https://arxiv.org/abs/2507.01918 | V-full (HTML: modules, sizes, Tables 1-3) |
| 4 | FactorVAE: A Probabilistic Dynamic Factor Model Based on Variational Autoencoder for Predicting Cross-Sectional Stock Returns | Y. Duan, L. Wang, Q. Zhang, J. Li | 2022 | AAAI 36(4) | https://ojs.aaai.org/index.php/AAAI/article/view/20369 | V-abs |
| 5 | Autoencoder Asset Pricing Models | S. Gu, B. Kelly, D. Xiu | 2021 | J. Econometrics 222(1) | https://papers.ssrn.com/abstract=3335536 | V-abs |
| 6 | Characteristics Are Covariances: A Unified Model of Risk and Return (IPCA) | B. Kelly, S. Pruitt, Y. Su | 2019 | JFE 134(3) | https://www.nber.org/papers/w24540 | V-abs |
| 7 | Deep Learning in Asset Pricing | L. Chen, M. Pelger, J. Zhu | 2024 | Management Science 70(2) | https://arxiv.org/abs/1904.00745 | V-abs |
| 8 | Empirical Asset Pricing via Machine Learning | S. Gu, B. Kelly, D. Xiu | 2020 | RFS 33(5) | https://www.nber.org/papers/w25398 | V-abs (depth comparison U) |
| 9 | The Virtue of Complexity in Return Prediction | B. Kelly, S. Malamud, K. Zhou | 2024 | J. Finance 79(1) | https://econpapers.repec.org/RePEc:bla:jfinan:v:79:y:2024:i:1:p:459-503 | V-abs |
| 10 | Seemingly Virtuous Complexity in Return Prediction | S. Nagel | 2025 | NBER WP 34104 | https://www.nber.org/papers/w34104 | V-sec (abstract-level summaries) |
| 11 | Large (and Deep) Factor Models | B. Kelly, B. Kuznetsov, S. Malamud, T. A. Xu (+Y. Zhang in 2026 rev.) | 2024/2026 | arXiv 2402.06635; SFI WP | https://arxiv.org/abs/2402.06635 | V-abs |
| 12 | Complexity in Factor Pricing Models (later "APT or AIPT?") | A. Didisheim, S. Ke, B. Kelly, S. Malamud | 2023-25 | NBER WP 31689 | https://www.nber.org/papers/w31689 | V-abs |
| 13 | Artificial Intelligence Asset Pricing Models | B. Kelly, B. Kuznetsov, S. Malamud, T. A. Xu | 2025 | NBER WP 33351 | https://www.nber.org/papers/w33351 | V-abs |
| 14 | Estimating Latent Asset-Pricing Factors (RP-PCA) | M. Lettau, M. Pelger | 2020 | J. Econometrics 218(1) | https://www.nber.org/papers/w24618 | V-abs |
| 15 | State-Varying Factor Models of Large Dimensions | M. Pelger, R. Xiong | 2022 | JBES 40(3) | https://arxiv.org/abs/1807.02248 | V-abs |
| 16 | Large Dynamic Covariance Matrices | R. Engle, O. Ledoit, M. Wolf | 2019 | JBES 37(2) | https://econpapers.repec.org/RePEc:taf:jnlbes:v:37:y:2019:i:2:p:363-375 | V-abs |
| 17 | Factor Models for Portfolio Selection in Large Dimensions: The Good, the Better and the Ugly | G. De Nard, O. Ledoit, M. Wolf | 2021 | J. Fin. Econometrics 19(2) | https://econpapers.repec.org/RePEc:zur:econwp:290 | V-abs |
| 18 | Nonlinear Shrinkage of the Covariance Matrix for Portfolio Selection: Markowitz Meets Goldilocks | O. Ledoit, M. Wolf | 2017 | RFS 30(12) | https://papers.ssrn.com/abstract=2383361 | V-abs |
| 19 | Quadratic Shrinkage for Large Covariance Matrices (QIS) | O. Ledoit, M. Wolf | 2022 | Bernoulli 28(3) | https://projecteuclid.org/journals/bernoulli/volume-28/issue-3/Quadratic-shrinkage-for-large-covariance-matrices/10.3150/20-BEJ1315.full | V-abs |
| 20 | The Power of (Non-)Linear Shrinking: A Review and Guide to Covariance Matrix Estimation | O. Ledoit, M. Wolf | 2022 | J. Fin. Econometrics 20(1) | https://econpapers.repec.org/RePEc:oup:jfinec:v:20:y:2022:i:1:p:187-218. | V-abs |
| 21 | Neural Nonlinear Shrinkage of Covariance Matrices for Minimum Variance Portfolio Optimization | L. Yang, S. Zhao, S. Chai | 2026 | arXiv 2601.15597 | https://arxiv.org/abs/2601.15597 | V-abs |
| 22 | Estimating Covariance for Global Minimum Variance Portfolio: A Decision-Focused Learning Approach | J. Kim, I. Tae, Y. Lee | 2025 | arXiv 2508.10776 | https://arxiv.org/abs/2508.10776 | V-abs |
| 23 | End-to-End Risk Budgeting Portfolio Optimization with Neural Networks | A. S. Uysal, X. Li, J. M. Mulvey | 2024 | Annals of OR 339 | https://arxiv.org/abs/2107.04636 | V-abs |
| 24 | Deep Declarative Risk Budgeting Portfolios | Parra-Diaz, Castro-Iragorri | 2025 | arXiv 2504.19980 | https://arxiv.org/abs/2504.19980 | V-sec |
| 25 | Conditional Moment Kernels: Joint Modeling of Expected Returns and Covariances in Unbalanced Panels | D. Filipovic, P. Schneider | 2024-25 | arXiv 2410.21858 | https://arxiv.org/abs/2410.21858 | V-abs |
| 26 | RVRAE: A Dynamic Factor Model Based on Variational Recurrent Autoencoder for Stock Returns Prediction | Y. Wang, S. Guo | 2024 | arXiv 2403.02500 | https://arxiv.org/abs/2403.02500 | V-abs |
| 27 | HireVAE: An Online and Adaptive Factor Model Based on Hierarchical and Regime-Switch VAE | Z. Wei, A. Rao, B. Dai, D. Lin | 2023 | IJCAI 2023 | https://arxiv.org/abs/2306.02848 | V-abs |
| 28 | Diffusion Factor Models: Generating High-Dimensional Returns with Factor Structure | M. Chen, R. Xu, Y. Xu, R. Zhang | 2025 | arXiv 2504.06566 | https://arxiv.org/abs/2504.06566 | V-abs |
| 29 | Tail-GAN: Learning to Simulate Tail Risk Scenarios | R. Cont, M. Cucuringu, R. Xu, C. Zhang | 2022/2026 | Management Science | https://arxiv.org/abs/2203.01664 | V-abs |
| 30 | High-Dimensional Multivariate Forecasting with Low-Rank Gaussian Copula Processes | D. Salinas et al. | 2019 | NeurIPS | https://neurips.cc/virtual/2019/poster/13787 | V-abs |
| 31 | TACTiS: Transformer-Attentional Copulas for Time Series; TACTiS-2 | A. Drouin et al.; A. Ashok et al. | 2022; 2024 | ICML; ICLR | https://arxiv.org/abs/2202.03528 | V-abs |
| 32 | Hybrid GARCH-LSTM for large portfolios (DCC with neural volatilities) | Boulet | 2021 | arXiv 2109.01044 | https://arxiv.org/abs/2109.01044 | V-sec |
| 33 | Deep Learning Enhanced Multivariate GARCH | Wang et al. | 2025 | arXiv 2506.02796 | https://arxiv.org/abs/2506.02796 | V-sec |
| 34 | A Machine Learning Approach to Volatility Forecasting | K. Christensen, M. Siggaard, B. Veliyev | 2023 | J. Fin. Econometrics 21(5) | https://econpapers.repec.org/RePEc:aah:create:2021-03 | V-abs |
| 35 | Re(Visiting) Time Series Foundation Models in Finance | E. Rahimikia, H. Ni, W. Wang | 2025 | arXiv 2511.18578 | https://arxiv.org/abs/2511.18578 | V-abs |
| 36 | Foundation Time-Series AI Model for Realized Volatility Forecasting | Goel et al. | 2025 | arXiv 2505.11163 | https://arxiv.org/abs/2505.11163 | V-sec |
| 37 | Kronos: A Foundation Model for the Language of Financial Markets | Shi et al. (Tsinghua) | 2025 | arXiv 2508.02739; AAAI | https://arxiv.org/abs/2508.02739 | V-abs |
| 38 | MASTER: Market-Guided Stock Transformer for Stock Price Forecasting | T. Li et al. | 2024 | AAAI-24 | https://ojs.aaai.org/index.php/AAAI/article/view/27767 | V-abs |
| 39 | Contrastive Learning of Asset Embeddings from Financial Time Series | R. Dolphin, B. Smyth, R. Dong | 2024 | arXiv 2407.18645 (ICAIF header) | https://arxiv.org/abs/2407.18645 | V-abs |
| 40 | Don't Blame the ELBO! A Linear VAE Perspective on Posterior Collapse | J. Lucas, G. Tucker, R. Grosse, M. Norouzi | 2019 | NeurIPS | https://arxiv.org/abs/1911.02469 | V-abs |
| 41 | Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations | F. Locatello et al. | 2019 | ICML | https://arxiv.org/abs/1811.12359 | V-abs |
| 42 | Importance Weighted Autoencoders | Y. Burda, R. Grosse, R. Salakhutdinov | 2016 | ICLR | https://arxiv.org/abs/1509.00519 | V-abs |
| 43 | Risk Parity Portfolios with Risk Factors | T. Roncalli, G. Weisang | 2016 | Quantitative Finance 16(3) | https://mpra.ub.uni-muenchen.de/44017/ | V-abs |
| 44 | Managing Diversification | A. Meucci | 2009 | Risk | https://risk.net/node/1500233 | V-sec |
| 45 | Minimum-torsion bets (Meucci, Santangelo, Deguest) | A. Meucci et al. | 2015 | SSRN/Risk | (seen only via citing abstracts) | U |
| 46 | Optimal Portfolios in Good Times and Bad | G. Chow, E. Jacquier, M. Kritzman, K. Lowry | 1999 | FAJ 55(3) | https://econpapers.repec.org/RePEc:taf:ufajxx:v:55:y:1999:i:3:p:65-73 | V-abs |
| 47 | Extreme Correlation of International Equity Markets | F. Longin, B. Solnik | 2001 | J. Finance 56(2) | https://econpapers.repec.org/RePEc:bla:jfinan:v:56:y:2001:i:2:p:649-676 | V-abs |
| 48 | Asymmetric Correlations of Equity Portfolios | A. Ang, J. Chen | 2002 | JFE 63 | https://www.nber.org/papers/w11824 (related WP); record via SUFE newsletter | V-abs |
| 49 | No Contagion, Only Interdependence: Measuring Stock Market Comovements | K. Forbes, R. Rigobon | 2002 | J. Finance 57(5) | https://www.nber.org/papers/w7267 | V-abs |
| 50 | Realized Semibetas: Disentangling "Good" and "Bad" Downside Risks | T. Bollerslev, A. Patton, R. Quaedvlieg | 2022 | JFE 144(1) | https://econpapers.repec.org/RePEc:eee:jfinec:v:144:y:2022:i:1:p:227-246 | V-abs |
| 51 | Dynamic Conditional Beta | R. Engle | 2016 | J. Fin. Econometrics 14(4) | https://econpapers.repec.org/RePEc:oup:jfinec:v:14:y:2016:i:4:p:643-667. | V-abs |
| 52 | Volatility Forecast Comparison Using Imperfect Volatility Proxies | A. Patton | 2011 | J. Econometrics 160(1) | https://econpapers.repec.org/RePEc:eee:econom:v:160:y:2011:i:1:p:246-256 | V-abs |
| 53 | On Loss Functions and Ranking Forecasting Performances of Multivariate Volatility Models | S. Laurent, J. Rombouts, F. Violante | 2013 | J. Econometrics 173 | https://econpapers.repec.org/RePEc:eee:econom:v:173:y:2013:i:1:p:1-10 | V-abs |
| 54 | Testing and Valuing Dynamic Correlations for Asset Allocation | R. Engle, R. Colacito | 2006 | JBES 24(2) | DOI 10.1198/073500106000000017 | V-sec |
| 55 | The Model Confidence Set | P. Hansen, A. Lunde, J. Nason | 2011 | Econometrica 79(2) | https://www.econometricsociety.org/publications/econometrica/2011/03/01/model-confidence-set | V-abs |
| 56 | Robust Performance Hypothesis Testing with the Sharpe Ratio | O. Ledoit, M. Wolf | 2008 | J. Empirical Finance 15(5) | https://econpapers.repec.org/RePEc:eee:empfin:v:15:y:2008:i:5:p:850-859 | V-abs |
| 57 | Eigenvalue Ratio Test for the Number of Factors | S. Ahn, A. Horenstein | 2013 | Econometrica | https://www.econometricsociety.org/publications/econometrica/2013/05/01/eigenvalue-ratio-test-number-factors | V-abs |
| 58 | Robust performance hypothesis testing with the variance (variance analogue of #56) | O. Ledoit, M. Wolf | 2011 | Wilmott | — | U |

### Not verified or only partly verified

* The depth result in Gu, Kelly & Xiu (2020), that around 3 hidden layers do best: from memory, not confirmed.
* The Gu, Kelly & Xiu (2021) comparison of autoencoder depths (CA0-CA3): not confirmed.
* FactorVAE architecture and numbers: abstract only.
* Engle & Colacito (2006) content: from citing theses.
* Meucci (2009) and the minimum-torsion critique: from secondary sources.
* The Ledoit-Wolf variance test: not searched.
* Numerical results of Yang et al., Kim et al. and Filipovic & Schneider: abstracts give no figures.
* The ICAIF venue of Dolphin et al.: header only.
* The Barra USE4 stability figure quoted in the DVT: not checked.
* All power calculations in §6.3 and the information budget in §5 are my own approximations, to be replaced by a
  bootstrap on the project's actual out-of-sample series.
