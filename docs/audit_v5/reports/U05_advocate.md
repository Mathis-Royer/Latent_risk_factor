# U05 — advocate

## Scope read
- `docs/DVT_strategie_facteurs_risque_latents_v4_1.md` — §4.1 (lines 148–200), §4.3 (293–436), §4.4 co-movement/curriculum (494–543), §4.6 inference and B (642–676), Appendix A (1200–1344), references (1576–1664).
- `docs/strategy_philosophy.md` — §6 (73–82).
- `docs/ISD_vae_latent_risk_factors.md` — MOD-002 (910–1118), MOD-006 AU sub-task (1632–1677).
- `src/vae/build_vae.py`, `src/vae/encoder.py`, `src/vae/decoder.py`, `src/vae/model.py`, `src/vae/loss.py` — full.
- `src/inference/active_units.py`, `src/inference/composite.py` — AU measurement/truncation, inference aggregation.
- `src/integration/pipeline.py` — `_adapt_vae_params` (283–375), AU derivation/truncation (1995–2074).
- `src/config.py` — VAE config (166–214), loss defaults (242), EWMA and Bai–Ng defaults (458–459).
- `src/risk_model/factor_regression.py` (1–110), `conditioning.py` (1–70), `rescaling.py` (1–95), `covariance.py` (header + assembly references).
- `src/walk_forward/metrics.py` — explanatory-power definitions (184–263, 491–550).
- `tests/unit/test_vae_architecture.py` — sizing/capacity/param-count tests (1–140, 411–442).
- `results/diagnostic/config.json`, `diagnostic_report.md`, `diagnostic_data.json`, `health_checks.csv`, checkpoint filenames.
- `notebooks/dashboard.ipynb` — VAE/loss/training configuration cells.

## Five questions

**1. Purpose and original intent of this unit (cite the spec).**
The unit specifies the latent factor model and its estimator. The data-generating assumption is `r_{i,t} = f(z_{1,t},...,z_{K,t}, β_{i,1},...,β_{i,K}) + ε_{i,t}`, with `f` allowed to be nonlinear and the VAE used to estimate the latent factors, the exposures and the dependence structure simultaneously (`DVT:154-158`). The VAE is chosen because it maps each window to a distribution rather than a point, and the encoder is explicitly the only component retained at inference: "the decoder reconstructs the original series... It is discarded after training — only the encoder is used" (`DVT:166-174`). The architecture is fully determined by five variables `(n, T, T_année, F, K)` with all other hyperparameters fixed (`DVT:321`, `ISD:915-917`). `K` is defined as a capacity ceiling, not a factor count; dimensions with marginal KL > 0.01 nats are "active units", and the effective factor count is read as AU after training (`DVT:402-420`, `ISD:1632-1657`). Philosophy §6 states the intent unambiguously: "The number of risk factors is not chosen in advance... the model discovers the answer from the data" (`philosophy:73-82`). The implementation contract is MOD-002's six sub-tasks, invariants and tests (`ISD:919-1118`).

**2. What a state-of-the-art version of this unit would look like for this objective (cite literature if you can).**
For a *risk* factor model, the SOTA object is a conditional panel factor model estimated from the cross-section, not a per-stock window autoencoder: `r_t = B(c_t) f_t + ε_t`, with loadings instrumented by characteristics (IPCA, Kelly–Pruitt–Su 2019; conditional autoencoder asset pricing, Gu–Kelly–Xiu 2021, cited in this repo as `[11]`, `DVT:1598`) or a VAE factor model trained on the cross-section (FactorVAE, Duan et al. 2022 `[30]`; RVRAE 2024 `[31]`; TempVAE, Sicks et al. 2023 `[26]`). Rank would be selected by an information criterion or by out-of-sample covariance fit (Bai–Ng `[27]`, `DVT:1630`; Onatski 2009, already referenced in `active_units.py:189`), not by a KL threshold followed by a truncation. The factor covariance would be a shrunk, EWMA/dynamic estimate (Ledoit–Wolf `[18]`, `DVT:1612`; the code already uses a DGJ spiked shrinker in `covariance.py`), and the unit would be evaluated on covariance-forecast loss (Frobenius/QLIKE, variance ratio, EP) in addition to portfolio Sharpe. A defensible SOTA stack for this objective is therefore: (i) a stock-agnostic nonlinear feature map trained with a cross-sectional likelihood or a permanent relational term; (ii) an explicit linear factor model with cross-sectional regression; (iii) a shrunk dynamic `Σ_z`; (iv) rank selection at the risk-model/validation layer. The repo's own roadmap already names the key alternative — replacing the VAE with PCA factor risk parity (`DVT:1159-1167`) and replacing co-movement scaffolding with InfoNCE (`DVT:1149-1157`) — which confirms that the authors know the current VAE is not the only candidate.

**3. What prevents the current unit from reaching it (evidenced, not assumed) — which obstacles are only apparent.**
Evidenced blockers:
- **No cross-sectional pressure at convergence.** The only term that ties latent geometry to cross-sectional co-movement is the curriculum scaffold, and it is set to zero for the last ~40% of training by design (`DVT:508-512`; `loss.py:611-642`). The stronger cross-sectional R² loss exists (`loss.py:201-278`) but is disabled: `lambda_cs: float = 0.0` (`config.py:242`), and the notebook says "DISABLED — confirmed harmful in V2" (`notebooks/dashboard.ipynb:251`). The final encoder is therefore trained on reconstruction + KL only.
- **The factor count is externally truncated, not discovered.** In the real run, 47 of 52 marginal KLs exceed 0.01 nats (`diagnostic_data.json:38-90`), but AU is reported as 26 (`diagnostic_data.json:8`) because the pipeline truncates to `AU_max_stat` (`pipeline.py:2017-2029`, `active_units.py:140-163`). `AU_max_stat = floor(sqrt(2 N_eff / r_min))` with `N_eff` from the EWMA half-life (`active_units.py:92-137`); with half-life 252 and `r_min=2`, `N_eff≈727`, so `AU_max≈26`, and `K = min(75, 2×26) = 52` (`pipeline.py:307-314`). The KL spectrum has no elbow at 26 (26th = 2.057, 27th = 1.969); the only sharp break is at 41 (0.592 → 0.128) (`diagnostic_data.json:38-90`).
- **The capacity-data constraint is not the guard it appears to be.** The denominator counts overlapping windows, `N = n(T_hist − T + 1)` (`DVT:368`, `build_vae.py:314`), while the same document states the number of independent windows is `N_eff ≈ T_hist/T ≈ 15` per stock (`DVT:651`). With `n=788` that is ~12k independent windows, not 5.7M; `P_total` is O(10^7) (`DVT:378` gives 10.4M for K=100), so the meaningful ratio is O(10^2–10^3), not the O(1) the constraint implies.
- **The only real run's factor span is weak.** `EP_oos = 0.3563` (`health_checks.csv:5`, `diagnostic_report.md:59`), below the spec's 0.50 elimination threshold and 0.60 target (`DVT:414,418`); variance ratio 2.74 is outside [0.5, 2.0] (`health_checks.csv:3`).

Obstacles that are only apparent:
- The decoder being discarded is not a defect; it is a legitimate representation-learning design (`DVT:170`). The defect is its *size*, not its fate.
- Rotation non-identifiability of the VAE is benign for the risk model: `Σ = BΣ_zB′ + D` is invariant to `B→BG`, `Σ_z→G⁻¹Σ_zG⁻ᵀ`, and the pipeline rotates to the principal-factor basis anyway (`DVT:191-197`). The SVD-based regression guard (`conditioning.py:23-59`) handles rank deficiency.
- The 1D-CNN vs transformer choice is not the binding constraint: the constraints are cross-sectional signal and effective sample size, not sequence-model topology.
- "K must be large for auto-pruning" is not needed if rank is selected downstream; the code already caps `K` at `2×AU_max_stat`, which is numerically equivalent to the doc's 1.5–2× heuristic (`DVT:420`, `pipeline.py:314`).
- The capacity constraint (`r_max=5`) is non-binding in the actual run, so it is not what is protecting the model.

**4. Which constraints are real (data, math, compute, market) and which are self-imposed design choices?**
Real:
- Data: 788 stocks, 1995–2026 daily, 35.3% missing values and 410 stocks with >20% missing (`health_checks.csv:10`, `diagnostic_report.md:30-31`); at `T=504` there are only ~15 non-overlapping windows per stock (`DVT:651`). The effective sample is small.
- Mathematical/identification: AU must be `< n_stocks` for the cross-sectional OLS (`pipeline.py:2022-2028`); `Σ` must be PSD; `Σ_z` has `AU(AU+1)/2` free parameters, which is why a statistical guard is needed (`DVT:422-426`).
- Compute: 16 GB; stride-1 inference produces ~5.5M windows and the code explicitly frees them to avoid a 32 GB+ blow-up (`pipeline.py:1998-2004`).
- Market: non-stationarity; a 504-day window spans regimes; factor loadings drift.

Self-imposed:
- One global `K` for all folds and regimes.
- Reconstructing the full `T×F` window with per-element MSE, instead of a sufficient statistic or a covariance/pricing objective.
- A 5.6M-parameter transposed residual decoder that is thrown away (`decoder.py:93-170`), which roughly doubles `P_total` and is the main driver of the capacity ratio.
- Defining AU by `KL > 0.01` and ranking factors by KL (`active_units.py:31,77-87`) rather than by contribution to return covariance.
- Removing the co-movement term in Phase 3 (`DVT:508-512`) and disabling the cross-sectional R² term (`config.py:242`, notebook line 251).
- Capping `K` at `2×AU_max_stat` and truncating AU to `AU_max_stat` (`pipeline.py:2017-2029`) — internally coherent, but it makes the factor count a function of `sigma_z_ewma_half_life` and `r_min`, not of the data.
- Fixing all architecture hyperparameters from `(n,T,T_annee,F,K)` — good for reproducibility, but it forbids per-regime capacity or a learned decoder width.

**5. What should be removed, merged or moved (not added)?**
- **Remove the transposed residual decoder body** (or replace it with a linear/MLP decoder). The decoder is discarded at inference (`DVT:170`), yet it is roughly half the parameters (`build_vae.py:177-228`) and the dominant term in the capacity ratio. A much smaller decoder preserves the encoder's multi-scale feature map and the Gaussian likelihood, while making the capacity constraint meaningful and halving training cost.
- **Remove the capacity-data constraint as currently defined**, or replace its denominator with an effective-sample-size budget (`n × T_hist / T`). It is non-binding in the run and its current denominator contradicts the document's own `N_eff≈15` statement (`DVT:368` vs `DVT:651`). Removing it loses nothing because the real rank control is `AU_max_stat`/`Σ_z` shrinkage.
- **Remove the dead AU post-filter and the KL-ranking truncation.** `post_filter_au_bai_ng` is off by default (`config.py:459`), and the KL ranking is not a factor-importance measure. Rank should be selected once, downstream, by OOS explanatory power or covariance loss.
- **Merge AU measurement, Bai–Ng/Onatski, and `Σ_z` shrinkage into one rank-selection module** in the risk model (`U09`), instead of having the VAE choose a rank and the risk model truncate it again.
- **Move the cross-sectional pressure out of the VAE curriculum**: either keep a small permanent relational term, or accept the "feature map" interpretation and stop calling the latent dimensions risk factors. The current half-measure — a strong scaffold that is switched off before the final representation is formed — is the worst of both.
- **Do not remove** the Inception/residual encoder, the stock-agnostic window encoder, the deterministic `encode()`/mean aggregation, the per-window z-scoring plus dual rescaling (shape/scale separation), or the exact analytical sizing/parameter accounting. These are the unit's genuine assets.

## Load-bearing assumptions

| ID | Assumption | Where stated | Evidenced? | Comment |
|----|------------|--------------|------------|---------|
| A1 | Risk-relevant structure is a function of a single stock's window shape, learnable by a shared encoder. | `DVT:154-174`, `DVT:307` | partially | The model runs; `EP_oos=0.356` (`health_checks.csv:5`) is weak but not zero. If false, the VAE is an expensive feature map. |
| A2 | Marginal KL ranks factor importance for return covariance. | `DVT:404-406`; `active_units.py:77-87` | no | KL measures divergence from the prior, not variance contribution. No test in the repo correlates KL rank with covariance rank. |
| A3 | The number of observations is `N=n(T_hist−T+1)`. | `DVT:368`; `build_vae.py:314` | no | Contradicted by the document's own `N_eff≈15` per stock (`DVT:651`). |
| A4 | The VAE latent space is continuous, gap-free, and distances are meaningful. | `DVT:164` | no | Not guaranteed by the ELBO; the cited Dai & Wipf (2019) is about prior/aggregate-posterior mismatch (`DVT:1600`). Distances are not used downstream anyway. |
| A5 | Removing the co-movement term in Phase 3 preserves correlation structure. | `DVT:508-512` | no | The document itself says per-sample MSE does not guarantee correlation preservation (`DVT:496`). Testable. |
| A6 | `K` can be a generous ceiling and AU read post hoc. | `DVT:404-420`; `philosophy:77-82` | no | The code caps `K` at `2×AU_max_stat` and truncates AU (`pipeline.py:2017-2029`), so the count is imposed, not discovered. |
| A7 | Per-window z-scoring makes shape the right invariant; scale is restored by `σ_i/σ̄`. | `DVT:672`; `rescaling.py:1-13` | yes | Implemented and coherent; it is the cleanest part of the design. |
| A8 | `B_A` is full rank and `AU<n` so cross-sectional OLS is well-posed. | `pipeline.py:2022-2028`; `conditioning.py:23-59` | partially | SVD guard exists; run condition number 2.4e4 (`diagnostic_report.md:70`). |
| A9 | The decoder only needs to be a training signal, so its capacity is not a modelling decision. | `DVT:170` | partially | True conceptually, but the 5.6M-parameter decoder dominates `P_total` and the capacity constraint. |
| A10 | `Σ_z` is reliably estimable after truncation to `AU_max_stat`. | `DVT:422-436`; `covariance.py` | partially | Variance ratio 2.74 outside [0.5,2] (`health_checks.csv:3`) suggests not yet. |

## Findings

| ID | Type | Severity | Evidence register | Claim (quoted, path:line) | Assessment | Evidence (path:line) | Cross-unit |
|----|------|----------|-------------------|---------------------------|------------|----------------------|------------|
| U05-ADV-01 | modeling | high | doc+code | `r_{i,t} = f(z_{1,t},...,z_{K,t}, β_{i,1},...,β_{i,K}) + ε_{i,t}` (`DVT:156`) | solid as a formulation; conditional downstream because the risk model is linear in B | `DVT:154-158`, `DVT:162-174`; `model.py:113-150` | U01,U08,U09 |
| U05-ADV-02 | literature | high | doc+code | "1D CNN with multi-scale kernels and stride (retained)... translation invariance... multi-scale extraction" (`DVT:309-312`) | solid; the three justifications are real properties of the chosen encoder, and the rejected alternatives are argued | `DVT:309-317`; `encoder.py:22-65` | U04 |
| U05-ADV-03 | math | high | code | `L(T) = max(3, ceil(log2(T / k_max)) + 2)`, `k_max = K_HEAD[-1] = 63` (`build_vae.py:44-56`) | solid; verified guarantee: `RF(L)=63+18(2^L−1)`, and with `L=ceil(log2(T/63))+2`, `RF≥45+8T/7 > T` (T=504 → RF=621) | `build_vae.py:44-56`; `encoder.py:68-133`; `DVT:340-344` | U04 |
| U05-ADV-04 | implementation | high | code | "P_enc = P_head + Σ P_block + P_proj" (`DVT:360`); `count_encoder_params(...)` (`build_vae.py:121`) | solid; the analytical counts mirror the PyTorch modules exactly (conv+BN+skip, transposed body, projection/output heads), and a regression test exists | `build_vae.py:121-228`; `tests/unit/test_vae_architecture.py:411-442` | U15 |
| U05-ADV-05 | implementation | high | code | "Each transposed block doubles the temporal dimension exactly" (`DVT:394`) | solid; `ConvTranspose1d(k=7,s=2,p=3,op=1)` gives `2L`, and the final trim 512→504 is harmless without skip connections | `decoder.py:23-90,128,142,159-169`; `DVT:390-398` | U04 |
| U05-ADV-06 | implementation | high | code | "`encode()` returns mu directly (deterministic). `forward()` uses reparameterization" (`ISD:1035`) | solid; inference uses `model.encode()` and mean aggregation, so B is reproducible | `model.py:152-165`; `composite.py:49,88`; `ISD:1030-1035` | U08 |
| U05-ADV-07 | math | high | doc+code | `D/(2σ²)·L_recon + (D/2)·ln(σ²) + L_KL` (Mode P, `loss.py:398`) | solid; this is the correct Gaussian ELBO with learned scalar `σ²`; the optimum is `σ²=MSE`; KL weight 1.0 is the standard ELBO | `loss.py:371-401`; `model.py:74-88`; `DVT:336-337` | U06 |
| U05-ADV-08 | honesty | high | doc+code | "The decoder... is discarded after training — only the encoder is used" (`DVT:170`) | solid honesty; the document also states B is "shape-only" (`DVT:672`) and that per-sample MSE does not guarantee correlation preservation (`DVT:496`). The remaining "discovers factors" language overreaches. | `DVT:170,496,672`; `composite.py:49,88` | U01,U08,U17 |
| U05-ADV-09 | interpretation | critical | doc+code+results | "The model discovers the answer from the data" (`philosophy:82`); "K is a capacity ceiling... AU is the effective factor count" (`DVT:404-406`) | fragile; contradicted by the run. 47 of 52 marginal KLs exceed 0.01 (`diagnostic_data.json:38-90`), but AU=26 because the pipeline truncates to `AU_max_stat` (`pipeline.py:2017-2029`). The spectrum has no elbow at 26; the sharp break is at 41. | `philosophy:73-82`; `DVT:404-406,422-436`; `active_units.py:92-117,140-163`; `pipeline.py:2017-2029`; `diagnostic_data.json:8,38-95` | U09,U15 |
| U05-ADV-10 | consistency | high | doc+code | "AU_max_stat ≈ 85" for 30 years daily (`DVT:426`) | fragile; the code uses the EWMA effective sample size, not raw `N_obs`: half-life 252 gives `N_eff≈727` and `AU_max=floor(sqrt(2·727/2))=26`, hence `K=52` (`pipeline.py:314`). The doc's 85 vs the run's 26 is a 3.3× capacity discrepancy. The current default is `sigma_z_ewma_half_life=0` (`config.py:458`), while the run recorded 252 (`config.json:68`), so the run is not reproducible from current defaults. | `DVT:424-436`; `active_units.py:100-137`; `pipeline.py:307-314`; `config.json:14,68`; `config.py:458`; `diagnostic_report.md:39` | U09,U15,U17 |
| U05-ADV-11 | statistics | high | doc+code | "`N = n × (T_hist − T + 1)` is the maximum number of windows (stride s=1)" (`DVT:368`) | fragile; overlapping windows are not independent. The same document says `N_eff≈T_hist/T≈15` per stock (`DVT:651`), giving ~12k independent windows for n=788, versus `N≈5.7M`. `P_total` O(10^7) (`DVT:378`) makes the real ratio O(10^2–10^3), so `r_max=5` is non-binding and does not provide the claimed overfitting protection. Missingness (35.3%) makes it worse. | `DVT:368,651`; `build_vae.py:314,330-356`; `health_checks.csv:10` | U07,U12,U15 |
| U05-ADV-12 | statistics | medium | code+results | "Active unit k ⟺ KL_k > au_threshold" (`active_units.py:31`) | conditional; KL is a valid posterior-collapse diagnostic but not a factor-importance measure for return covariance. The run ranks 47 active dims by KL and keeps the top 26 without checking that KL rank agrees with covariance contribution. Testable: correlate per-dim KL with per-dim marginal contribution to `BΣ_zB′`/cross-sectional R². | `active_units.py:77-87,140-163`; `pipeline.py:2029`; `diagnostic_data.json:38-95`; `DVT:404-406` | U09,U10 |
| U05-ADV-13 | modeling | critical | doc+code | "Phase 3 — Free refinement: λ_co = 0. The encoder operates on reconstruction + KL alone" (`DVT:512`) | fragile; the only cross-sectional term is removed for the last ~40% of training by design, and the stronger cross-sectional R² loss is disabled (`lambda_cs=0.0`). The final representation can drift away from the correlation structure the scaffold built. Testable: measure co-movement alignment / EP at the end of Phase 2 vs Phase 3. | `DVT:496,508-512`; `loss.py:611-642`; `config.py:242`; `trainer.py:358-359`; `notebooks/dashboard.ipynb:251` | U06,U07,U08 |
| U05-ADV-14 | math | medium | doc+code | `Σ = B̃′_A·Λ·B̃′_A^T + D_ε` (`DVT:196`) | solid; rotation non-identifiability is benign because the covariance is invariant to `B→BG`, `Σ_z→G⁻¹Σ_zG⁻ᵀ`; the pipeline rotates to the principal-factor basis, and the SVD regression guard handles rank deficiency | `DVT:191-197`; `factor_regression.py:97-110`; `conditioning.py:23-59`; `covariance.py:6` | U09,U10 |
| U05-ADV-15 | justification | medium | doc | "nearby points correspond to similar return dynamics, with no empty gaps. Distances between stocks are therefore meaningful" (`DVT:164`) | fragile; the ELBO does not guarantee a gap-free aggregate posterior, and the cited Dai & Wipf (2019) is about prior/aggregate-posterior mismatch (`DVT:1600`). Distances are not used downstream: B enters a linear regression, so only its column space matters. | `DVT:164,1600,191-197` | U01,U08 |
| U05-ADV-16 | implementation | medium | code | "The architecture is fully determined by 5 variable parameters" (`DVT:321`) | solid engineering; `build_vae` returns `P_enc/P_dec/P_total/r`, raises an actionable error with remediation, and keeps `N_train` separate from `N` for planning | `build_vae.py:272-390`; `ISD:1037-1065` | U15,U17 |
| U05-ADV-17 | results | medium | results | "AU = 26 / K = 52 (50.0% utilization)" (`health_checks.csv:2`); "EP_oos = 0.3563" (`health_checks.csv:5`) | fragile; the architecture runs (B=639×52, no NaNs) but OOS explanatory power is below the spec's 0.50 elimination threshold and 0.60 target; the variance ratio 2.74 is outside [0.5,2]. Single pretrained model, no training history, so this is suggestive, not causal. | `health_checks.csv:2-5`; `diagnostic_report.md:39-70`; `metrics.py:184-211,491-506`; `DVT:414,418` | U09,U12,U14 |
| U05-ADV-18 | implementation | low | code | `log_var = log_var.clamp(-6.0, 6.0)` (`model.py:105`) | fragile/minor; the sampling distribution is clamped but `compute_kl_loss` uses the unclamped `log_var`, so the KL and the sampled posterior differ at the boundary; the post-clamp assertion is tautological | `model.py:105-107`; `loss.py:126-130` | U06 |
| U05-ADV-19 | consistency | medium | doc+code+results | "Dropout | 0.1" (`DVT:335`) vs run `dropout=0.3` (`config.json:21`) vs notebook `0.2` (`dashboard.ipynb:262`) | conditional; the document is a design description, not a run record. Same for `λ_co` (0.5 doc/config vs 0.1 notebook) and `K` (200 doc vs 75 config/notebook vs 52 run). Any audit must cite which layer it relies on. | `DVT:335,514`; `config.json:14,21,26`; `config.py:179`; `notebooks/dashboard.ipynb:247,250,262`; `diagnostic_report.md:39` | U15,U17 |
| U05-ADV-20 | implementation | medium | code | "feature_weights: [2.0, 0.5]... emphasizes returns (f=0) over volatility (f=1)" (`loss.py:158-159`) | conditional strength; the mechanism exists and the diagnostics are per-feature, but the run config does not record feature weights, so the actual weighting in the one real run is unverified | `loss.py:79-100,137-194`; `notebooks/dashboard.ipynb:253`; `config.json:24-30` | U04,U06 |

## Core strengths to preserve

1. **The encoder topology.** Inception head `(5,21,63)` + residual body with stride-2 blocks + global average pooling + linear `μ/logσ²` heads is appropriate for the task: translation invariance means a volatility cluster is recognized wherever it occurs, multi-scale kernels capture week/month/quarter structure, and the encoder has no stock-ID input, so it generalizes to new stocks without retraining (`DVT:307-313`; `encoder.py:22-65,135-194`). Any redesign should keep a stock-agnostic multi-scale temporal encoder.
2. **The exact sizing rules and parameter accounting.** `L(T)` has a verified receptive-field guarantee (`RF=63+18(2^L−1)≥T`), `C_L` enforces compression (`C_L≥1.3×2K`), and the channel progression is geometric and monotonic (`build_vae.py:44-100`). The analytical parameter count matches PyTorch and is tested (`tests:411-442`). This is the reproducibility backbone of the unit and should survive any redesign.
3. **The shape/scale separation.** Per-window z-scoring makes B shape-only, and dual rescaling by `σ_{i,t}/σ̄_t` restores cross-sectional scale for estimation and portfolio construction (`DVT:672`; `rescaling.py:1-70`). This is a genuinely good idea: it prevents the VAE from spending capacity on volatility levels and puts the level where it belongs, in a transparent univariate estimator. Keep it.
4. **Deterministic inference.** `encode()` returns `μ` only; `forward()` is used for training; inference uses `model.eval()`/`no_grad()` and mean aggregation (`model.py:152-165`; `composite.py:49,88`). This makes B reproducible and decouples training stochasticity from portfolio construction.
5. **The Gaussian likelihood with learned scalar `σ²`.** Mode P is the textbook ELBO, with `σ²` initialised at 1 and clamped to `[1e-4,10]`; the decoder output head is linear, which is correct for a Gaussian mean (`loss.py:371-401`; `model.py:74-88`). The math here is sound and should not be replaced by an ad hoc MSE.
6. **The statistical guard as a concept.** Capping the downstream factor count so that `Σ_z` does not have more free parameters than the data can support is the right instinct (`DVT:422-436`; `active_units.py:92-117`). The current implementation uses the wrong `N` and an externally imposed rank, but the *idea* of a statistical cap belongs in the redesign.
7. **Honest documentation of the VAE's limits.** The document explicitly says the decoder is discarded, B is shape-only, and per-sample MSE does not guarantee correlation preservation, citing Locatello et al. (`DVT:170,496,672`). This honesty is an asset: it makes the redesign argument possible without rewriting the intent.

## Verdict

U05 is the best-engineered component of the strategy and the least-justified one. The encoder, sizing rules, parameter accounting, decoder arithmetic, deterministic inference and Gaussian ELBO are correct, testable and internally consistent; the shape/scale separation is a genuinely good architectural idea; and the documentation is candid about the decoder being discarded and about per-sample MSE not preserving correlation. But the unit's headline claim — that the VAE discovers risk factors and decides how many exist — is not what the code does. The encoder is a per-stock window autoencoder, the only cross-sectional term is switched off for the last 40% of training, the cross-sectional R² term is disabled, and the effective factor count is `floor(sqrt(2N_eff/r_min))`, a function of the EWMA half-life and `r_min`, not of the data's factor structure. The one real run shows this precisely: 47 of 52 dimensions pass the KL threshold, yet AU=26 and K=52, and OOS explanatory power is 0.356, below the spec's own 0.50 elimination threshold. The architecture should be kept as a feature extractor, the discarded decoder should be shrunk, and rank selection plus cross-sectional pressure should move into the risk model and walk-forward validation. It is sound as an autoencoder and conditional as a factor model.

Score: 7/10

## Machine companion

```json
[
  {
    "id": "U05-ADV-01",
    "unit": "U05",
    "role": "advocate",
    "type": "modeling",
    "severity": "high",
    "evidence_register": "doc+code",
    "claim": "r_{i,t} = f(z_{1,t},...,z_{K,t}, beta_{i,1},...,beta_{i,K}) + eps_{i,t} — a latent factor model with a possibly nonlinear f.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:156",
    "assessment": "solid as a formulation; conditional downstream because the risk model is linear in B, so only the encoder's feature map is nonlinear.",
    "evidence": ["docs/DVT_strategie_facteurs_risque_latents_v4_1.md:154-158", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:162-174", "src/vae/model.py:113-150"],
    "cross_units": ["U01", "U08", "U09"],
    "root_cause": null,
    "confidence": "high"
  },
  {
    "id": "U05-ADV-02",
    "unit": "U05",
    "role": "advocate",
    "type": "literature",
    "severity": "high",
    "evidence_register": "doc+code",
    "claim": "1D CNN with multi-scale kernels and stride (retained): translation invariance, multi-scale extraction, parallelization.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:309",
    "assessment": "solid; the three justifications are real properties of the chosen encoder, and the rejected alternatives (WaveNet, LSTM/GRU, Transformer) are argued rather than asserted.",
    "evidence": ["docs/DVT_strategie_facteurs_risque_latents_v4_1.md:309-317", "src/vae/encoder.py:22-65"],
    "cross_units": ["U04"],
    "root_cause": null,
    "confidence": "high"
  },
  {
    "id": "U05-ADV-03",
    "unit": "U05",
    "role": "advocate",
    "type": "math",
    "severity": "high",
    "evidence_register": "code",
    "claim": "L(T) = max(3, ceil(log2(T / k_max)) + 2), k_max = 63.",
    "claim_location": "src/vae/build_vae.py:44",
    "assessment": "solid; verified receptive-field guarantee: RF(L)=63+18(2^L-1), and with L=ceil(log2(T/63))+2, RF >= 45+8T/7 > T (T=504 gives RF=621).",
    "evidence": ["src/vae/build_vae.py:44-56", "src/vae/encoder.py:68-133", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:340-344"],
    "cross_units": ["U04"],
    "root_cause": null,
    "confidence": "high"
  },
  {
    "id": "U05-ADV-04",
    "unit": "U05",
    "role": "advocate",
    "type": "implementation",
    "severity": "high",
    "evidence_register": "code",
    "claim": "P_enc = P_head + sum P_block + P_proj (analytical parameter count).",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:360",
    "assessment": "solid; the analytical counts mirror the PyTorch modules exactly (conv+BN+skip, transposed body, projection/output heads), and a regression test compares them with sum(p.numel()).",
    "evidence": ["src/vae/build_vae.py:121-228", "tests/unit/test_vae_architecture.py:411-442"],
    "cross_units": ["U15"],
    "root_cause": null,
    "confidence": "high"
  },
  {
    "id": "U05-ADV-05",
    "unit": "U05",
    "role": "advocate",
    "type": "implementation",
    "severity": "high",
    "evidence_register": "code",
    "claim": "Each transposed block doubles the temporal dimension exactly; final trim/pad to T.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:394",
    "assessment": "solid; ConvTranspose1d(k=7,s=2,p=3,op=1) gives 2L, and the 512->504 trim is harmless because there are no U-Net skip connections.",
    "evidence": ["src/vae/decoder.py:23-90", "src/vae/decoder.py:128", "src/vae/decoder.py:142", "src/vae/decoder.py:159-169", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:390-398"],
    "cross_units": ["U04"],
    "root_cause": null,
    "confidence": "high"
  },
  {
    "id": "U05-ADV-06",
    "unit": "U05",
    "role": "advocate",
    "type": "implementation",
    "severity": "high",
    "evidence_register": "code",
    "claim": "encode() returns mu directly (deterministic); forward() uses reparameterization.",
    "claim_location": "docs/ISD_vae_latent_risk_factors.md:1035",
    "assessment": "solid; inference uses model.encode() and mean aggregation, so B is reproducible and training stochasticity is decoupled from portfolio construction.",
    "evidence": ["src/vae/model.py:152-165", "src/inference/composite.py:49", "src/inference/composite.py:88", "docs/ISD_vae_latent_risk_factors.md:1030-1035"],
    "cross_units": ["U08"],
    "root_cause": null,
    "confidence": "high"
  },
  {
    "id": "U05-ADV-07",
    "unit": "U05",
    "role": "advocate",
    "type": "math",
    "severity": "high",
    "evidence_register": "doc+code",
    "claim": "Mode P: D/(2 sigma^2) * L_recon + (D/2) * ln(sigma^2) + L_KL.",
    "claim_location": "src/vae/loss.py:398",
    "assessment": "solid; this is the correct Gaussian ELBO with a learned scalar observation variance; the optimum is sigma^2 = reconstruction MSE, and the KL weight is 1.0.",
    "evidence": ["src/vae/loss.py:371-401", "src/vae/model.py:74-88", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:336-337"],
    "cross_units": ["U06"],
    "root_cause": null,
    "confidence": "high"
  },
  {
    "id": "U05-ADV-08",
    "unit": "U05",
    "role": "advocate",
    "type": "honesty",
    "severity": "high",
    "evidence_register": "doc+code",
    "claim": "The decoder reconstructs the original series from a sampled latent vector, serving as a training signal. It is discarded after training — only the encoder is used.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:170",
    "assessment": "solid honesty; the document also states B is shape-only and that per-sample MSE does not guarantee correlation preservation. The remaining 'discovers factors' language overreaches, but the disclosure itself is a strength.",
    "evidence": ["docs/DVT_strategie_facteurs_risque_latents_v4_1.md:170", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:496", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:672", "src/inference/composite.py:49"],
    "cross_units": ["U01", "U08", "U17"],
    "root_cause": null,
    "confidence": "high"
  },
  {
    "id": "U05-ADV-09",
    "unit": "U05",
    "role": "advocate",
    "type": "interpretation",
    "severity": "critical",
    "evidence_register": "doc+code+results",
    "claim": "The model discovers the answer from the data, constrained only by a statistical guard (AU_max_stat).",
    "claim_location": "docs/strategy_philosophy.md:82",
    "assessment": "fragile; contradicted by the run. 47 of 52 marginal KLs exceed 0.01, but AU=26 because the pipeline truncates to AU_max_stat. The KL spectrum has no elbow at 26; the sharp break is at 41. The effective factor count is a function of N_eff and r_min, not of the data's factor structure.",
    "evidence": ["docs/strategy_philosophy.md:73-82", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:404-406", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:422-436", "src/inference/active_units.py:92-117", "src/inference/active_units.py:140-163", "src/integration/pipeline.py:2017-2029", "results/diagnostic/diagnostic_data.json:8", "results/diagnostic/diagnostic_data.json:38-95"],
    "cross_units": ["U09", "U15"],
    "root_cause": "The VAE is used as a per-stock window autoencoder but interpreted as a cross-sectional factor model; the rank is imposed by the risk-model statistical guard rather than learned.",
    "confidence": "high"
  },
  {
    "id": "U05-ADV-10",
    "unit": "U05",
    "role": "advocate",
    "type": "consistency",
    "severity": "high",
    "evidence_register": "doc+code",
    "claim": "AU_max_stat ≈ 85 for 30 years of daily data.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:426",
    "assessment": "fragile; the code uses the EWMA effective sample size, not raw N_obs: half-life 252 gives N_eff≈727 and AU_max=26, hence K=52. The doc's 85 vs the run's 26 is a 3.3x capacity discrepancy. The run recorded half-life 252, while the current default is 0, so the run is not reproducible from current defaults.",
    "evidence": ["docs/DVT_strategie_facteurs_risque_latents_v4_1.md:424-436", "src/inference/active_units.py:100-137", "src/integration/pipeline.py:307-314", "results/diagnostic/config.json:68", "src/config.py:458", "results/diagnostic/diagnostic_report.md:39"],
    "cross_units": ["U09", "U15", "U17"],
    "root_cause": "The doc's statistical guard uses raw counts; the code applies it to EWMA effective sample size. The two layers were never reconciled.",
    "confidence": "high"
  },
  {
    "id": "U05-ADV-11",
    "unit": "U05",
    "role": "advocate",
    "type": "statistics",
    "severity": "high",
    "evidence_register": "doc+code",
    "claim": "N = n x (T_hist - T + 1) is the maximum number of windows (stride s=1).",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:368",
    "assessment": "fragile; overlapping windows are not independent. The same document says N_eff≈T_hist/T≈15 per stock, giving ~12k independent windows for n=788 versus N≈5.7M. With P_total O(10^7), the meaningful ratio is O(10^2-10^3), so r_max=5 is non-binding and does not provide the claimed overfitting protection. Missingness (35.3%) makes it worse.",
    "evidence": ["docs/DVT_strategie_facteurs_risque_latents_v4_1.md:368", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:651", "src/vae/build_vae.py:314", "src/vae/build_vae.py:330-356", "results/diagnostic/health_checks.csv:10"],
    "cross_units": ["U07", "U12", "U15"],
    "root_cause": "The capacity constraint uses an inflated denominator (overlapping, complete-history windows), so it is structurally non-binding for realistic universes.",
    "confidence": "high"
  },
  {
    "id": "U05-ADV-12",
    "unit": "U05",
    "role": "advocate",
    "type": "statistics",
    "severity": "medium",
    "evidence_register": "code+results",
    "claim": "Active unit k iff KL_k > au_threshold (0.01 nats).",
    "claim_location": "src/inference/active_units.py:31",
    "assessment": "conditional; KL is a valid posterior-collapse diagnostic but not a factor-importance measure for return covariance. The run ranks 47 active dims by KL and keeps the top 26 without checking that KL rank agrees with covariance contribution. Testable: correlate per-dim KL with marginal contribution to B*Sigma_z*B' or cross-sectional R^2.",
    "evidence": ["src/inference/active_units.py:77-87", "src/inference/active_units.py:140-163", "src/integration/pipeline.py:2029", "results/diagnostic/diagnostic_data.json:38-95", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:404-406"],
    "cross_units": ["U09", "U10"],
    "root_cause": "KL measures divergence from the prior, not factor importance; the same statistic is used for both jobs.",
    "confidence": "medium"
  },
  {
    "id": "U05-ADV-13",
    "unit": "U05",
    "role": "advocate",
    "type": "modeling",
    "severity": "critical",
    "evidence_register": "doc+code",
    "claim": "Phase 3 — Free refinement: lambda_co = 0. The encoder operates on reconstruction + KL alone.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:512",
    "assessment": "fragile; the only cross-sectional term is removed for the last ~40% of training by design, and the stronger cross-sectional R^2 loss is disabled (lambda_cs=0.0). The final representation can drift away from the correlation structure the scaffold built. Testable: measure co-movement alignment and EP at the end of Phase 2 vs Phase 3.",
    "evidence": ["docs/DVT_strategie_facteurs_risque_latents_v4_1.md:496", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:508-512", "src/vae/loss.py:611-642", "src/config.py:242", "src/training/trainer.py:358-359", "notebooks/dashboard.ipynb:251"],
    "cross_units": ["U06", "U07", "U08"],
    "root_cause": "The architecture relies on a temporary scaffold to create cross-sectional structure, then removes the scaffold before the final representation is fixed.",
    "confidence": "high"
  },
  {
    "id": "U05-ADV-14",
    "unit": "U05",
    "role": "advocate",
    "type": "math",
    "severity": "medium",
    "evidence_register": "doc+code",
    "claim": "Sigma = B'_A * Lambda * B'_A^T + D_eps.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:196",
    "assessment": "solid; rotation non-identifiability is benign because the covariance is invariant to B->BG, Sigma_z->G^-1 Sigma_z G^-T; the pipeline rotates to the principal-factor basis, and the SVD regression guard handles rank deficiency.",
    "evidence": ["docs/DVT_strategie_facteurs_risque_latents_v4_1.md:191-197", "src/risk_model/factor_regression.py:97-110", "src/risk_model/conditioning.py:23-59", "src/risk_model/covariance.py:6"],
    "cross_units": ["U09", "U10"],
    "root_cause": null,
    "confidence": "high"
  },
  {
    "id": "U05-ADV-15",
    "unit": "U05",
    "role": "advocate",
    "type": "justification",
    "severity": "medium",
    "evidence_register": "doc",
    "claim": "nearby points correspond to similar return dynamics, with no empty gaps. Distances between stocks are therefore meaningful and comparable.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:164",
    "assessment": "fragile; the ELBO does not guarantee a gap-free aggregate posterior, and the cited Dai & Wipf (2019) is about prior/aggregate-posterior mismatch. Distances are not used downstream: B enters a linear regression, so only its column space matters.",
    "evidence": ["docs/DVT_strategie_facteurs_risque_latents_v4_1.md:164", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:1600", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:191-197"],
    "cross_units": ["U01", "U08"],
    "root_cause": "A generic VAE intuition was carried into the justification even though the downstream model does not use latent distances.",
    "confidence": "medium"
  },
  {
    "id": "U05-ADV-16",
    "unit": "U05",
    "role": "advocate",
    "type": "implementation",
    "severity": "medium",
    "evidence_register": "code",
    "claim": "The architecture is fully determined by 5 variable parameters (n, T, T_annee, F, K); all other hyperparameters are fixed.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:321",
    "assessment": "solid engineering; build_vae returns P_enc/P_dec/P_total/r, raises an actionable error with remediation, and keeps N_train separate from N for planning.",
    "evidence": ["src/vae/build_vae.py:272-390", "docs/ISD_vae_latent_risk_factors.md:1037-1065"],
    "cross_units": ["U15", "U17"],
    "root_cause": null,
    "confidence": "high"
  },
  {
    "id": "U05-ADV-17",
    "unit": "U05",
    "role": "advocate",
    "type": "results",
    "severity": "medium",
    "evidence_register": "results",
    "claim": "AU = 26 / K = 52 (50.0% utilization); EP_oos = 0.3563, EP_is = 0.2190.",
    "claim_location": "results/diagnostic/health_checks.csv:2",
    "assessment": "fragile; the architecture runs (B=639x52, no NaNs) but OOS explanatory power is below the spec's 0.50 elimination threshold and 0.60 target, and the variance ratio 2.74 is outside [0.5,2]. Single pretrained model with no training history, so this is suggestive rather than causal.",
    "evidence": ["results/diagnostic/health_checks.csv:2-5", "results/diagnostic/diagnostic_report.md:39-70", "src/walk_forward/metrics.py:184-211", "src/walk_forward/metrics.py:491-506", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:414", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:418"],
    "cross_units": ["U09", "U12", "U14"],
    "root_cause": "The latent feature map does not span enough of the OOS return covariance; this may be the VAE, the rescaling/regression, or both.",
    "confidence": "medium"
  },
  {
    "id": "U05-ADV-18",
    "unit": "U05",
    "role": "advocate",
    "type": "implementation",
    "severity": "low",
    "evidence_register": "code",
    "claim": "log_var = log_var.clamp(-6.0, 6.0).",
    "claim_location": "src/vae/model.py:105",
    "assessment": "fragile/minor; the sampling distribution is clamped but compute_kl_loss uses the unclamped log_var, so the KL and the sampled posterior differ at the boundary; the post-clamp assertion is tautological.",
    "evidence": ["src/vae/model.py:105-107", "src/vae/loss.py:126-130"],
    "cross_units": ["U06"],
    "root_cause": "A numerical safeguard was added in the sampling path without mirroring it in the KL path.",
    "confidence": "high"
  },
  {
    "id": "U05-ADV-19",
    "unit": "U05",
    "role": "advocate",
    "type": "consistency",
    "severity": "medium",
    "evidence_register": "doc+code+results",
    "claim": "Dropout | 0.1 (DVT fixed hyperparameters).",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:335",
    "assessment": "conditional; the document is a design description, not a run record. Dropout is 0.3 in the run and 0.2 in the notebook; lambda_co is 0.5 in the doc/config and 0.1 in the notebook; K is 200 in the doc, 75 in config/notebook and 52 in the run. Any audit must state which layer it relies on.",
    "evidence": ["docs/DVT_strategie_facteurs_risque_latents_v4_1.md:335", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:514", "results/diagnostic/config.json:14", "results/diagnostic/config.json:21", "results/diagnostic/config.json:26", "src/config.py:179", "notebooks/dashboard.ipynb:247", "notebooks/dashboard.ipynb:250", "notebooks/dashboard.ipynb:262", "results/diagnostic/diagnostic_report.md:39"],
    "cross_units": ["U15", "U17"],
    "root_cause": "The document, config defaults and notebook drifted apart; the run artifact is the only reliable record of the run.",
    "confidence": "high"
  },
  {
    "id": "U05-ADV-20",
    "unit": "U05",
    "role": "advocate",
    "type": "implementation",
    "severity": "medium",
    "evidence_register": "code",
    "claim": "feature_weights: [2.0, 0.5]... emphasizes returns (f=0) over volatility (f=1).",
    "claim_location": "src/vae/loss.py:158",
    "assessment": "conditional strength; the mechanism exists and per-feature MSE diagnostics are computed, but the run config does not record feature weights, so the actual weighting in the one real run is unverified.",
    "evidence": ["src/vae/loss.py:79-100", "src/vae/loss.py:137-194", "notebooks/dashboard.ipynb:253", "results/diagnostic/config.json:24-30"],
    "cross_units": ["U04", "U06"],
    "root_cause": null,
    "confidence": "medium"
  }
]
```

END-OF-REPORT
