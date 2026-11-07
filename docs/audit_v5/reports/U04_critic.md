# U04 — critic

## Scope read

- `src/data_pipeline/windowing.py` — entire file (307 lines): two-pass window construction, NaN/zero filters, ffill/bfill, log-vol, per-window z-scoring.
- `src/data_pipeline/features.py` — entire file (87 lines): 252-day trailing vol and 21-day rolling realized vol.
- `src/data_pipeline/returns.py` — lines 1–299: log-returns, gap filling, delisting lookup/imputation, winsorization.
- `src/data_pipeline/universe.py` — lines 15–129: listing-history filter, cap/ADV filters, point-in-time loop.
- `src/inference/composite.py` — entire file (172 lines): stride-1 encoding, mean/EWM aggregation.
- `src/integration/pipeline.py` — lines 560–640 (vol computation before folds), 1240–1314 (Phase A windows + train/val split), 1755–1824 (Phase B training windows), 1940–2199 (inference windows stride=1, AU, B normalisation/clip/shrink, aggregation, rescaling, OOS MSE).
- `src/vae/loss.py` — lines 137–211: per-feature weighted reconstruction loss.
- `src/config.py` — lines 115–224 (`DataPipelineConfig`, `VAEArchitectureConfig`), 343–377 (`InferenceConfig`).
- `docs/DVT_strategie_facteurs_risque_latents_v4_1.md` §4.2 preprocessing and universe (lines 202–303), §4.3 window length and encoder rationale (295–350), §4.6 inference/aggregation/B (642–676), §4.7 rescaling (724–738), parameter table (998–1002).
- `docs/ISD_vae_latent_risk_factors.md` MOD-001 spec: conventions (46–53), parameters (218–223), sub-tasks 4–6 and invariants/pitfalls/tests (791–900), anti-cyclical rule (1800).
- `notebooks/dashboard.ipynb` — parameter cells (245–268), config construction (549–556, 647–652, 700, 749).
- `tests/unit/test_data_pipeline.py` — windowing tests (142–315, 747–780, 880–1020).
- `scripts/download_tiingo.py` — lines 848, 908, 911 (exchange/delisting columns).
- `.claude/rules/project/changelog.md` — lines 1–45 (feature/vol/log-transform history).
- `results/diagnostic/config.json` (all 137 lines), `results/diagnostic/diagnostic_report.md` (1–120).

Parameter source used throughout: the notebook is treated as the production parameter source (`T=504`, `N_FEATURES=2`, `TRAINING_STRIDE=21`, `AGGREGATION_HALF_LIFE=60`, `FEATURE_WEIGHTS=[2.0,0.5]`); `src/config.py` is cited for code defaults; where the two disagree I say so explicitly.

---

## Five questions

**1. Purpose and original intent of this unit (cite the spec).**

The unit exists to convert each stock's raw return history into a *scale-free, cross-stock-comparable "risk shape"* that a shared encoder can process, and to attach a second channel that supposedly restores the volatility information destroyed by that normalisation. The intended chain is spelled out at `docs/DVT_strategie_facteurs_risque_latents_v4_1.md:182` ("daily log returns + 21-day realized volatility (2 features per time step) → sliding windows of T days (tensor T × 2, z-scored per feature per window)") and formalised at `docs/DVT_strategie_facteurs_risque_latents_v4_1.md:226-254`. The ISD restates it as an implementation contract: "Z-score returns PER WINDOW … Z-score volatility PER WINDOW … Stack → tensor (T, F=2)" (`docs/ISD_vae_latent_risk_factors.md:804-809`).

Three sub-intents are load-bearing:
- **Cross-stock generalisation without retraining**: the encoder "operates solely on return dynamics" so "a new stock can be processed without retraining" (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:307`).
- **T as purely architectural**: "T is the length of the fixed-length sequence the encoder processes — an **architectural constraint**, not a choice about how much history to consider" (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:297`); history is supposed to live in the aggregation of many windows (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:644-656`).
- **F=2 as genuinely complementary information**, not decoration: "The z-scored return preserves *temporal shape* … but erases the absolute volatility scale. The 21-day rolling volatility captures *volatility dynamics* — clustering, cross-stock co-movements in variance, and local regime transitions. These are complementary" (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:250`).

**2. What a state-of-the-art version of this unit would look like for this objective (cite literature if you can).**

For the stated objective — *discover common latent risk factors from return panels and feed a factor risk model* — the SOTA shape is not a per-stock univariate 1-D CNN on z-scored windows. It is:

- **Cross-sectional encoding.** Autoencoder asset pricing models (Gu, Kelly & Xiu, *Journal of Econometrics*, 2021) and IPCA (Kelly, Pruitt & Su, *JFE*, 2019) encode the *cross-section at each date*: the input is a stock-date characteristics matrix, and the latent factors are extracted from the panel, not from each stock's own time series. That is the structure that makes "common" factors identifiable at the representation level; a per-stock encoder can only produce loadings whose commonality is discovered afterwards by regression.
- **Explicit shape/scale decomposition with a heteroskedastic likelihood, not double z-scoring.** Standard practice is to standardise returns by a conditional volatility estimate (GARCH/EWMA/DCC; Bollerslev 1986; Engle 2002) and keep the standardised residual as the "shape" channel while carrying the conditional scale as an explicit state variable. The current design standardises *twice* and then restores scale with one ad-hoc scalar (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:724-730`).
- **Multi-horizon realized volatility.** The HAR model (Corsi, *Journal of Financial Econometrics*, 2009) exists precisely because realized volatility has separate daily/weekly/monthly components; a single 21-day horizon discards that structure. If a vol channel is kept, it should be level-referenced (log-RV, which is approximately Gaussian — Andersen, Bollerslev, Diebold & Labys 2003) and *not* re-standardised by its own window dispersion.
- **Date-indexed exposures** (dynamic factor model / state-space loadings) rather than one frozen per-stock vector; Barra-style models (Menchero, Orr & Wang 2011, USE4) at least use point-in-time exposure estimates, not an end-of-sample average.
- **Mask-aware handling of missing data** (GRU-D, Che et al., NeurIPS 2018) or an explicit missingness channel, instead of NaN→0 plus a 5 % tolerance.
- **Validation geometry that respects the window length**: purged/embargoed splits (López de Prado, *Advances in Financial Machine Learning*, 2018) so that the validation block is disjoint from training by at least T plus an embargo.

**3. What prevents the current unit from reaching it (evidenced, not assumed).**

- The second channel adds no information: it is a deterministic transform of the first (`src/data_pipeline/features.py:76-78` computes rolling std of the *same* returns that are z-scored), so the effective input dimension is ≈524, not 1008 (finding U04-CRI-01).
- The second z-scoring re-erases exactly the level the channel was introduced to restore (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:248`; `src/data_pipeline/windowing.py:257-260`), so the doc's own 10 %→40 % example is unrepresentable (U04-CRI-02).
- The 7 000-point latent trajectory is collapsed to one vector per stock (`src/inference/composite.py:155-165`), so the "local / regime" semantics never reach the risk model (U04-CRI-03).
- The recency horizon is mis-scaled by 21×: a half-life expressed in windows is applied to stride-1 windows (`src/config.py:355-359` says "60 (≈ 1260 trading days at stride=21)"; `src/integration/pipeline.py:1976-1996` applies it to stride-1 trajectories; `results/diagnostic/config.json:57-58` records 60) (U04-CRI-04).
- The exposure vector is estimated from the *end* of the training period and then applied to the whole history; σ rescaling fixes scale but not shape (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:728-730`; `src/integration/pipeline.py:2187-2194`) (U04-CRI-05).
- The train/validation split is not independent: T=504 equals the 2-year validation span, so validation windows overlap training windows by up to 503/504 days, and the doc's own ESS formula gives ~1 independent validation window per stock (U04-CRI-06).
- The missing-data policy rests on an assumption contradicted by the run: 35.31 % missing values, 410/788 stocks above 20 % missing (`results/diagnostic/diagnostic_report.md:30-31`), while the guard that would filter this (`min_valid_fraction`) is never consumed anywhere in `src` (`src/config.py:137`) (U04-CRI-07).
- The stated consecutive-gap rule is not implemented; only a NaN fraction is checked, and NaN→0 creates fake zero returns that the zero-return filter cannot see (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:258` vs `src/data_pipeline/windowing.py:213-233`) (U04-CRI-08).
- Shumway delisting imputation is never triggered on the actual data source (`scripts/download_tiingo.py:911` sets `delisting_return` to NaN for every row; `src/data_pipeline/returns.py:187-195` only records stocks with at least one non-NaN value) (U04-CRI-09).
- T=504 is asserted three different ways, never tested (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:297` vs `:273` vs `:301`), and there is only one configuration in `results/` (U04-CRI-10).
- The log-vol transform is undocumented in both specs and was introduced as a post-mortem patch (`.claude/rules/project/changelog.md:16`, kept at `:13`) (U04-CRI-11).
- The 80/20 channel weighting is a loss-side patch for a feature-side problem, with contradictory rationales (`src/vae/loss.py:148-151` vs `.claude/rules/project/changelog.md:16`) (U04-CRI-12).
- Training stride 21 vs inference stride 1, while the doc's own table says the default training stride is 1 (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:1001`), and the doc argues the mean is invariant to stride ≤21 (`:651`) — so the 21× extra inference memory buys nothing (U04-CRI-13).
- The encoder never sees a cross-section, so "common factor discovery" is delegated to the downstream cross-sectional regression; the observed parity with a 2-feature PCA (`results/diagnostic/diagnostic_report.md:115`) is what this design predicts (U04-CRI-14, -15).

**4. Which constraints are real (data, math, compute, market) and which are self-imposed design choices?**

Real:
- Daily OHLCV + adjusted price is all that exists; no fundamentals, no intraday data (`docs/ISD_vae_latent_risk_factors.md:622-646` lists optional columns that are not actually in the Tiingo build).
- Delisted-stock data quality from Tiingo is genuinely poor (no delisting returns; 35 % missing overall; `results/diagnostic/diagnostic_report.md:30-31`).
- Look-ahead is a real constraint; per-window statistics are legitimately causal.
- 16 GB RAM is a real deployment constraint (it is why stride 1 at inference had to be memory-managed, `src/integration/pipeline.py:1998-2001`).
- Covariance estimation needs independent observations; with T=504 and 30 years, ESS ≈ 15 per stock is a real statistical limit (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:651`).

Self-imposed:
- **Per-window z-scoring of the vol channel** — mathematically optional; the level could be carried as a scalar or as a level-referenced channel.
- **F=2 with a deterministic second channel** — optional; no information is added.
- **T=504 as a single number doing four jobs** (encoder input, listing requirement, regime horizon, validation span) — a choice, not a constraint.
- **Static B between retrainings** — a choice; date-indexed exposures are affordable at this n.
- **Stride-1 inference** — the doc itself concedes the mean is invariant for stride ≤21, so this is a self-imposed 21× memory cost kept "for optionality" (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:650-652`).
- **No cross-sectional input** (market return, VIX, sector) — self-imposed; VIX and the S&P 500 index file are already in the repo and used elsewhere (`src/data_pipeline/crisis.py`).
- **The 5 % NaN and 20 % zero thresholds** — arbitrary, uncited, untested.
- **The log transform and the [2.0, 0.5] feature weights** — patches, not constraints.

**5. What should be removed, merged or moved (not added)?**

- **Remove one of the two channels.** Either drop the vol channel (the encoder's 21-day Inception branch can represent a local std in one layer) or drop the per-window z-scoring of the return channel and keep a *level-referenced* vol channel. Keeping both z-scored is strictly dominated (U04-CRI-01/-02).
- **Remove stride-1 inference** unless the aggregation becomes date-specific. At stride 21 the profile is, by the doc's own ESS argument, identical; the current code spends its largest memory allocation to compute a quantity it then averages away.
- **Remove the mean/EWM aggregation** and move the time dimension into the risk model: use date-indexed `B_t` directly in the cross-sectional regression. If aggregation is kept, delete the half-life parameter as currently specified (its units are wrong) and re-specify it in *trading days*, then re-derive the intended horizon (U04-CRI-04).
- **Remove `min_valid_fraction` from the config or wire it in** — today it is a dead field that advertises a guard the pipeline does not have (U04-CRI-07).
- **Remove the log transform and the feature weights as separate patches**, and **merge** them into the model's existing per-channel observation variance: the VAE already learns a scalar σ² (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:337`); making it per-channel is a merge, not an addition, and it subsumes both hacks.
- **Merge the two volatility definitions** (21-day input vol, 252-day rescaling vol) into one documented hierarchy, because the current pair is used inconsistently: exposures are normalised by a 504-day within-window σ and then rescaled by a 252-day trailing σ (`src/data_pipeline/windowing.py:252-260` vs `src/data_pipeline/features.py:39-40`; `src/integration/pipeline.py:2187-2194`).
- **Move the listing-history rule** out of `universe.py` (where it is `len(group)`, `src/data_pipeline/universe.py:76-79`) into the window-validity layer, or delete the claim that it guarantees 504 days of *continuous* listing (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:273`).
- **Move the validation split** from "last two years of overlapping windows" to a disjoint block of length ≥ T + embargo; with the current geometry the "validation set" is one window (U04-CRI-06).
- **Delete the "several market conditions" justification for T=504** (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:301`) and replace it with an explicit, tested horizon; two years contains at most one stress episode and usually none.

---

## Load-bearing assumptions

| ID | Assumption | Where stated | Evidenced? | Comment |
|----|-----------|--------------|------------|---------|
| A1 | Returns and 21-day realized vol carry **complementary** information | `docs/DVT_strategie_facteurs_risque_latents_v4_1.md:250` | **no** | vol is an exact deterministic function of the same returns inside the window (`src/data_pipeline/features.py:76-78`); at most 20 pre-window scalars are added. |
| A2 | Z-scoring the vol channel restores the volatility level that z-scoring returns erased | `docs/DVT_strategie_facteurs_risque_latents_v4_1.md:248-250` | **no** | the vol channel is itself z-scored within the same window (`src/data_pipeline/windowing.py:257-260`), so a multiplicative rescaling of the vol path is annihilated. |
| A3 | Missing data is "rare on the target universe" | `docs/DVT_strategie_facteurs_risque_latents_v4_1.md:258` | **no** | 35.31 % missing; 410/788 stocks > 20 % missing (`results/diagnostic/diagnostic_report.md:30-31`). |
| A4 | A 504-day window is "long enough for several market conditions (calm, crisis, recovery)" | `docs/DVT_strategie_facteurs_risque_latents_v4_1.md:301` | **no** | 2 years ≈ one regime; no T ablation exists in `results/`. |
| A5 | T is purely architectural, not a history choice | `docs/DVT_strategie_facteurs_risque_latents_v4_1.md:297` | **no** | the same T is the listing requirement (`:273`) and a regime-coherence horizon (`:301`). |
| A6 | Training stride 21 and inference stride 1 are interchangeable for the chosen mean aggregation | `docs/DVT_strategie_facteurs_risque_latents_v4_1.md:651-654` | **partially** | plausible from the ESS argument, but then stride 1 is unnecessary; no test in the repo compares profiles at the two strides. |
| A7 | `aggregation_half_life=60` means ≈ 1 260 trading days (≈5 years) | `src/config.py:355-359`; `src/inference/composite.py:135-138` | **no** | production inference is stride 1 (`src/integration/pipeline.py:1976-1980`), so 60 windows = 60 trading days. |
| A8 | A single time-invariant B plus σ_t rescaling can represent time-varying risk | `docs/DVT_strategie_facteurs_risque_latents_v4_1.md:672-676, 724-730` | **partially** | σ_t restores scale only; shape changes are unrepresentable, and σ_252 ≠ the σ_504 used in the z-score. |
| A9 | A mean over all windows "preserv[es] memory of all historical regimes" | `docs/DVT_strategie_facteurs_risque_latents_v4_1.md:656` | **no** | a mean over window-end dates is a mixture, not memory; the production half-life further truncates it to recent months. |
| A10 | The validation split is a meaningful out-of-sample signal | `docs/DVT_strategie_facteurs_risque_latents_v4_1.md:620`; `src/integration/pipeline.py:1284-1289` | **no** | T = validation span = 2 years (`src/config.py` walk-forward `val_years=2` in `results/diagnostic/config.json:121`), so ~1 independent validation window per stock. |
| A11 | The zero-return filter excludes suspended/illiquid stocks | `docs/DVT_strategie_facteurs_risque_latents_v4_1.md:242` | **partially** | NaN→0 counts against the budget (`src/data_pipeline/windowing.py:227-233`); 20 % of 504 ≈ 101 stale days are tolerated. |
| A12 | Delisting returns are Shumway-imputed before windowing | `docs/DVT_strategie_facteurs_risque_latents_v4_1.md:265`; `docs/ISD_vae_latent_risk_factors.md:891` | **no** | `scripts/download_tiingo.py:911` sets every `delisting_return` to NaN → `src/data_pipeline/returns.py:187-195` never populates `delisting_info`. |
| A13 | The 21-day vol warm-up is always based on pre-window returns | `docs/ISD_vae_latent_risk_factors.md:890` | **partially** | true at fold boundaries (global rolling vol, `src/integration/pipeline.py:610`), false for the first window of each stock (NaN tolerated at `src/data_pipeline/windowing.py:215`). |

---

## Findings

| ID | Type | Severity | Evidence register | Claim (quoted, path:line) | Assessment | Evidence (path:line) | Cross-unit |
|----|------|----------|-------------------|---------------------------|------------|----------------------|------------|
| U04-CRI-01 | math / logic | **critical** | doc+code | "The 21-day rolling volatility captures *volatility dynamics* — clustering, cross-stock co-movements in variance, and local regime transitions. These are complementary" (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:250`) | False as an information claim. `v_{i,τ}` is `std(r_{i,τ-20:τ})` of the same return series that channel 0 contains (`src/data_pipeline/features.py:76-78`), and both channels are z-scored in the same window (`src/data_pipeline/windowing.py:248-260`). Given channel 0, channel 1 is determined except for the first ~20 entries, which use pre-window returns. The nominal input is `T×F = 1008` numbers; the intrinsic dimension is ≈ 524. "Cross-stock co-movements in variance" cannot even in principle be an input feature: the encoder is univariate per stock, so it never observes another stock. Steelman: a hand-crafted 21-day realised-vol channel is an *inductive bias* that can shorten learning; the Inception head already has a `k=21` branch that can learn a local std in one layer, so the bias is largely redundant with the architecture. Rebuttal: a channel that is a deterministic function of another channel is a pure capacity tax on the decoder (see U04-CRI-12 for the 10× imbalance it produced). Testable prediction: an otherwise identical F=1 run reproduces the F=2 model's AU within ±2, its return-channel reconstruction MSE within ~5 %, and its OOS Sharpe within the fold SE. | `src/data_pipeline/features.py:76-78`; `src/integration/pipeline.py:610`; `src/data_pipeline/windowing.py:248-260`; `docs/DVT_strategie_facteurs_risque_latents_v4_1.md:250` | U05, U06, U08 |
| U04-CRI-02 | modeling / math | **major** | doc+code | "a stock with identical return shapes but transitioning from 10% to 40% annualized volatility carries fundamentally different risk, invisible to the return channel alone" (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:250`) | False in both readings. (i) If only the *level* differs (identical shapes, scaled vol path), then `log v` z-scored within the window is invariant to a constant rescaling, so the vol channel also cannot distinguish them. (ii) If the transition is *within* the window, the return channel already sees it: the z-score divides by one window-level σ and preserves relative amplitude, so a 4× rise in |r| inside the window survives. The design thus cannot distinguish the one example given to justify F=2. The doc itself concedes the mechanism: "per-window normalization erases **volatility dynamics**" (`:240`) — and then applies the same normalisation to the restoring channel (`:248`). Additional robustness flaw: a single extreme return (e.g., a −55 % delisting log-return) roughly doubles the window σ and shrinks every other standardised return by ~2×; no robust (median/MAD) scaling is used. | `docs/DVT_strategie_facteurs_risque_latents_v4_1.md:240, 246-250`; `src/data_pipeline/windowing.py:252-260`; `src/data_pipeline/returns.py:230-246` | U06 |
| U04-CRI-03 | modeling | **critical** | doc+code | "The encoder produces a local latent vector μ_i^(t) … aggregates all local vectors across its full history: μ̄_i = Agg(...)" (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:644-646`) and "Between retrainings, B is **fixed**" (`:676`) | The unit's central output is a *trajectory*, and the production path throws the trajectory away: `aggregate_profiles` reduces every stock's window stack to a single (K,) vector (`src/inference/composite.py:155-165`), and that vector is all the risk model ever sees (`src/integration/pipeline.py:1992-1996`). Consequences: (a) the "local risk profile / regime detection" language is not implemented anywhere in the risk model; (b) all time variation in the model comes from Σ_z and a scalar σ_t, i.e. this is a **static factor model**; (c) the 7 000 windows/stock machinery exists only to compute a smoother average of ~15 effective observations (the doc's own number, `docs/DVT_strategie_facteurs_risque_latents_v4_1.md:651`). Steelman: slow-moving exposures are defensible (Barra exposures are monthly/quarterly). Rebuttal: if so, stride 1, the regime language and the "local vector" framing are unnecessary; and a mean over a stock's *entire* history is not an estimate of *today's* risk. Either way, the unit's stated purpose and its implementation are not the same object. | `src/inference/composite.py:119-171`; `src/integration/pipeline.py:1976-1996`; `docs/DVT_strategie_facteurs_risque_latents_v4_1.md:644-656, 676` | U08, U09, U12 |
| U04-CRI-04 | implementation / consistency | **major** | code+results | "Typical: 60 (≈ 1260 trading days at stride=21, i.e. ~5 years)" (`src/config.py:355-359`) | The decay is indexed by *windows*, not days (`src/inference/composite.py:158-163`), and the production inference path builds windows at stride 1 (`src/integration/pipeline.py:1976-1980`, `:1992-1996`), so `half_life=60` = 60 **trading days** of window-end dates, not 1 260. Weighted span of window-end dates falls by 21×; the data span falls from ≈ 7 years (504 + 1 260 days) to ≈ 2.5–3 years (504 + ~180 days). The notebook sets 60 (`notebooks/dashboard.ipynb:268`) and the diagnostic run records 60 (`results/diagnostic/config.json:57-58`). The same parameter therefore means two different things depending on which call site constructs the windows; nothing in the code enforces a unit. This also voids the ISD's "use the FULL history (anti-cyclical principle)" claim (`docs/ISD_vae_latent_risk_factors_v4_1...` → `docs/ISD_vae_latent_risk_factors.md:1800`) for any run using half-life > 0. | `src/inference/composite.py:127-144, 155-165`; `src/config.py:355-366`; `src/integration/pipeline.py:1976-1980, 1992-1996`; `notebooks/dashboard.ipynb:268`; `results/diagnostic/config.json:57-58`; `docs/ISD_vae_latent_risk_factors.md:1800` | U08, U09, U15, U16 |
| U04-CRI-05 | modeling / leakage-adjacent | **major** | doc+code | "**For historical factor estimation** … B̃_A,i,t = (σ_i,t / σ̄_t) · μ̄_A,i … so the cross-sectional regression correctly attributes co-movement to factors" (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:728-730`) | The single profile μ̄_i is inferred from windows weighted toward `train_end` (and with half-life 60 at stride 1, almost entirely the last quarter). That vector is then applied to *all* training dates to recover ẑ_t (`src/integration/pipeline.py:2187-2194`). The σ_t factor fixes the scalar magnitude only; it cannot restore a stock's shape in 2005 if the profile was measured in 2024. So the 2005 "exposure" is a 2024 shape times a 2005 vol — a use of end-of-sample shape information at the beginning of the sample. It is not strict future-data leakage (all inputs ≤ train_end), but it is point-in-time impure in exactly the sense CONV-10 is meant to prevent, and it manufactures the "anti-cyclical" property that Σ_z is claimed to have. Second inconsistency: the z-score denominator is the **504-day within-window** σ (`src/data_pipeline/windowing.py:252-255`), while the rescaling numerator is the **252-day trailing** σ (`src/data_pipeline/features.py:39-40`); the linear map latent→raw return is only correct if the two coincide, which fails precisely in regime shifts. | `src/integration/pipeline.py:1976-1996, 2187-2194`; `src/data_pipeline/windowing.py:252-260`; `src/data_pipeline/features.py:39-40`; `docs/DVT_strategie_facteurs_risque_latents_v4_1.md:728-734` | U08, U09, U12 |
| U04-CRI-06 | statistics / validation | **major** | code+results | "the same split used for nested hyperparameter selection … the selected checkpoint is, by construction, the one with the best score on this set" (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:620`) | The split is by window **end date** only (`src/integration/pipeline.py:1284-1289`), with `val_years = 2` (`results/diagnostic/config.json:121`) and `T = 504` ≈ 2 years. Every validation window therefore overlaps the training data by up to 503/504 days; the first validation window is a near-duplicate of the last training window (only its own μ, σ differ). The doc's own ESS formula (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:651`) gives `N_eff ≈ val_span / T = 1` independent validation window per stock. Early stopping is therefore operating on ~1 effective observation, and the selected E* that Phase B then fixes (`:625`) inherits that instability. Neither the DVT's validation discussion (`:620-626`) nor the test suite mentions overlap. Testable prediction: keeping every 10th validation window changes E* by <1 epoch — i.e. the val set carries ~nothing beyond one window. | `src/integration/pipeline.py:1284-1289`; `results/diagnostic/config.json:121`; `docs/DVT_strategie_facteurs_risque_latents_v4_1.md:620-626, 651` | U07, U12 |
| U04-CRI-07 | data / justification | **major** | doc+code+results | "Missing value handling: rare on the target universe (large-cap US equities on CRSP)" (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:258`) | The actual dataset has 35.31 % missing values and 410 of 788 stocks above 20 % missing (`results/diagnostic/diagnostic_report.md:30-31`); the doc's premise is off by an order of magnitude, and the "CRSP" reference is not the data used. The config field designed for this (`min_valid_fraction: float = 0.80`, `src/config.py:137`) is never read anywhere in `src` — a dead guard that creates a false impression of quality control. Combined with the window tolerance (5 % NaN accepted, `src/data_pipeline/windowing.py:215`), a stock with 30 % missing history can still contribute a substantial number of windows, each with 5 % fake zero returns. | `results/diagnostic/diagnostic_report.md:30-31`; `src/config.py:137`; `src/data_pipeline/windowing.py:213-215`; `docs/DVT_strategie_facteurs_risque_latents_v4_1.md:258` | U03 |
| U04-CRI-08 | doc-vs-code / data | **major** | doc+code | "exclude the window if a gap exceeds 5 consecutive days (temporal continuity broken over the shortest Inception kernel scale)" (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:258`) | Not implemented. Windowing checks only a NaN **fraction** per window (`src/data_pipeline/windowing.py:213-215`), so a single 20-day contiguous gap (4 % of 504) passes and is silently converted to 20 zero returns (`:227`). The same substitution makes the zero-return filter partly a missingness filter, since NaN→0 is counted as a zero return (`:232-233`); a window may therefore combine 25 missing days with up to ~76 genuine zero days and still pass the 20 % threshold. The 20 % threshold itself is uncited and untested, and it tolerates ~101 stale days in a "risk profile". `_vectorized_ffill_bfill` then backfills the vol channel with future-within-window values (`:228-229`), producing flat 21-day-vol segments at every gap. | `src/data_pipeline/windowing.py:212-233`; `docs/DVT_strategie_facteurs_risque_latents_v4_1.md:242, 258` | U03, U05 |
| U04-CRI-09 | data / honesty | **major** | code+data | "If delisted without an available return, a corrective return is imputed (Shumway convention: -30% NYSE/AMEX, -55% Nasdaq)" (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:265`); "**DO NOT** forget the Shumway delisting return imputation" (`docs/ISD_vae_latent_risk_factors.md:891`) | On the actual source the imputation never fires. The Tiingo build writes `"delisting_return": np.nan` on every row (`scripts/download_tiingo.py:911`); `_extract_delisting_info` records a stock only if some row has a non-NaN delisting return (`src/data_pipeline/returns.py:187-195`), so `delisting_info` is empty and `_apply_delisting_returns` is a no-op. The terminal return is whatever the last price-to-price return happened to be, and post-delisting NaN days are then filled to 0 inside windows ending within 25 trading days of the delisting. Consequence: the encoder never sees the imputed −30 %/−55 % tail event that the design says it learns from (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:263`), and the stated survivorship-bias-free training property is unsupported on the shipped data. | `scripts/download_tiingo.py:911`; `src/data_pipeline/returns.py:187-195, 200-246`; `docs/DVT_strategie_facteurs_risque_latents_v4_1.md:263-265`; `docs/ISD_vae_latent_risk_factors.md:889-891` | U03, U16 |
| U04-CRI-10 | justification | **major** | doc+code+results | "T = 504 days (~2 years): long enough for several market conditions (calm, crisis, recovery), short enough for a coherent risk regime" (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:301`) | Unfalsifiable as stated and contradicted by the same document: T is called "an **architectural constraint**, not a choice about how much history to consider" (`:297`), used as the minimum listing history (`:273`), and justified by regime coherence (`:301`). Three different requirements are collapsed into one integer. No T ablation exists: `results/` contains a single configuration (`results/diagnostic/config.json:4`), and the only related machinery is a planned action (`src/integration/action_specs.py:363-372`), not a performed test. A 2-year window contains at most one stress episode; the doc's own ESS calculation (`:651`) implies ≈15 independent 21-day blocks per window, i.e. T is a *low*-information choice for a 200-dim latent. | `docs/DVT_strategie_facteurs_risque_latents_v4_1.md:273, 297, 301, 651`; `results/diagnostic/config.json:4`; `src/integration/action_specs.py:363-372` | U05, U12, U15 |
| U04-CRI-11 | implementation / doc-vs-code | **major** | code+doc | "apply log(vol + 1e-8) before z-scoring. Compresses the volatility distribution, preventing the VAE from over-allocating latent capacity to volatility patterns" (`src/data_pipeline/windowing.py:132-134`) | The log transform appears in neither specification (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:244-248` prescribes raw vol z-scoring; `docs/ISD_vae_latent_risk_factors.md:804-809` likewise); a repo-wide search finds it only in code and in the changelog, where it is introduced as one of eight V2 post-mortem fixes and explicitly **kept** through the Phase-0 revert (`.claude/rules/project/changelog.md:13, 16`). The stated benefit is a hypothesis with no measurement attached, and the change is not a no-op: z-scoring after a log re-standardises variance but leaves the channel with a different shape (skew/kurtosis) than raw-vol z-scoring. It also interacts with the channel weighting (U04-CRI-12), so two untested transforms are stacked on the same channel. Confidence medium-high: the code and changelog are unambiguous; the *effect* is untested. | `src/data_pipeline/windowing.py:110, 132-134, 244-249`; `docs/DVT_strategie_facteurs_risque_latents_v4_1.md:244-248`; `docs/ISD_vae_latent_risk_factors.md:804-809`; `.claude/rules/project/changelog.md:13, 16` | U06, U15, U17 |
| U04-CRI-12 | implementation / justification | **major** | code+doc | "This addresses reconstruction imbalance where the VAE over-fits to easy-to-reconstruct features (e.g., volatility) at the expense of harder features (e.g., returns)" (`src/vae/loss.py:148-151`) | The rationale is internally contradictory with the change's own history: the changelog says the fix was for a "10× returns/vol reconstruction imbalance" (`.claude/rules/project/changelog.md:16`), i.e. vol error was *larger*, whereas `loss.py` says vol is *easy* to reconstruct. A mean-MSE already gives less weight to an easy channel; downweighting it further is only coherent if its error is large. Production weights are `[2.0, 0.5]` → 80 %/20 % after normalisation (`notebooks/dashboard.ipynb:253, 700`; `src/vae/loss.py:177-181`), while the code default is `feature_weights: list[float] = field(default_factory=list)` = equal weights (`src/config.py:244`). So the effective objective depends on which source launches the run, by a 4× relative channel weight with no principled calibration. This is a loss-side patch for a feature-side defect: two z-scored channels of unit variance should have comparable MSE; a 10× gap is evidence that the channels are not the same kind of object (one is a smoothed deterministic function of the other), not evidence that a magic weight is needed. | `src/vae/loss.py:137-194`; `notebooks/dashboard.ipynb:253, 700`; `src/config.py:244`; `.claude/rules/project/changelog.md:16` | U06 |
| U04-CRI-13 | consistency | **major** | doc+code | "If training windows exceed capacity, **windows** are subsampled (stride 5 or 21), not stocks" (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:281`); table: `s_train` default 1, range 1–21 (`:1001`) | The code makes stride 21 the unconditional default (`src/config.py:140`) and the notebook sets it (`notebooks/dashboard.ipynb:263`), so the documented *contingency* is the permanent regime. Meanwhile inference uses stride 1 (`src/integration/pipeline.py:1976-1980`), i.e. a 21× denser sampling of a distribution the encoder never trained on; the doc argues this is harmless because "the result is quasi-identical for any s_inf ≤ 21" (`:651`) — which, if true, means the stride-1 path (the single largest memory allocation in the pipeline, `src/integration/pipeline.py:1998-2001`) computes nothing the stride-21 profile does not already contain. The unit is therefore internally inconsistent in all three directions: doc default 1, code default 21, and a claimed invariance that would make the expensive path pointless. A secondary consequence: training windows per stock scale with listing length (30 y vs 5 y ≈ 7×), and the loss averages windows uniformly (`src/vae/loss.py:189-194`), so long-history stocks dominate the learned latent space with no stated weighting. | `docs/DVT_strategie_facteurs_risque_latents_v4_1.md:281, 648-654, 1001`; `src/config.py:140`; `notebooks/dashboard.ipynb:263`; `src/integration/pipeline.py:1790, 1976-1980, 1998-2001`; `src/vae/loss.py:189-194` | U05, U07, U08 |
| U04-CRI-14 | modeling | **major** | doc+code | "Raw series let the model discover *which* statistics matter — conditional correlations, volatility synchronization, tail dependence" (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:222`) | The encoder is a per-stock univariate map `(T,2) → R^K`; at inference no other stock is in the input. It therefore cannot observe conditional correlations or synchronization; it can only produce loadings whose commonality is discovered afterwards by the cross-sectional regression on B. The only training-time cross-sectional signal is the co-movement loss on synchronous windows (`docs/ISD_vae_latent_risk_factors.md:1363-1387`), which is curriculum-scheduled to zero. The claim that the VAE "discovers latent risk factors" is thus a claim about a nonlinear univariate feature map plus a downstream regression — structurally a factor model with nonlinear PCA loadings. This is the single largest gap between the narrative and the object. | `docs/DVT_strategie_facteurs_risque_latents_v4_1.md:222, 307`; `docs/ISD_vae_latent_risk_factors.md:1363-1387`; `src/inference/composite.py:21-116` | U05, U06, U08 |
| U04-CRI-15 | results / interpretation | **major** | results | "Avg cross-sectional R² (OOS): 0.1300" (`results/diagnostic/diagnostic_report.md:61`); "Top 1 factor contribution: 81.1%" (`:92`); `pca_vol` Sharpe 0.3678 vs VAE 0.3816 (`:115`) | The downstream evidence is what the shape-only/level-free/static/univariate representation predicts: a factor model that explains 13.0 % of OOS cross-sectional variance (21.9 % IS, `:60`) and whose risk is 81.1 % concentrated in one principal factor, while a **two-component PCA on the same engineered features** reaches Sharpe 0.3678 against the VAE's 0.3816. On one fold these are not statistically distinguishable, so this is supporting evidence rather than proof; but the direction is exactly what U04-CRI-01/-02/-03/-14 predict. Caveats on the run itself: the encoder was a *pretrained* model, not trained on this data (`results/diagnostic/diagnostic_report.md:35`), and the recorded config is not the run's effective config (K=75 in `results/diagnostic/config.json:14` vs K=52 in `results/diagnostic/diagnostic_report.md:39` and B `[639, 52]` at `:48`), so these numbers describe the pipeline geometry + a given encoder, not "the strategy". | `results/diagnostic/diagnostic_report.md:35, 39, 48, 59-61, 92, 115`; `results/diagnostic/config.json:14` | U05, U09, U14, U15, U17 |
| U04-CRI-16 | implementation | **minor** | code | "**DO NOT** forget the 21-day warm-up for realized volatility — the rolling vol at the start of a window must be based on pre-window returns" (`docs/ISD_vae_latent_risk_factors.md:890`) | Not enforced. `create_windows` never checks that the 21 pre-window vol values exist; it only checks that ≤5 % of a window is NaN (`src/data_pipeline/windowing.py:215`), and 21/504 = 4.2 % is under the threshold. For every stock's first windows the entire 21-day vol warm-up is therefore NaN → bfilled to a constant (`:228-229`), producing a flat 21-day segment that the encoder can memorise as an artefact. The pipeline does compute vol globally before slicing (`src/integration/pipeline.py:610`), so fold boundaries are safe; only a stock's own listing start is affected. | `docs/ISD_vae_latent_risk_factors.md:890`; `src/data_pipeline/windowing.py:213-229`; `src/integration/pipeline.py:610` | U03 |
| U04-CRI-17 | implementation / metrics | **minor** | code | "`_oos_w, _, _ = create_windows(..., T=self.config.data.window_length, stride=T)`" (`src/integration/pipeline.py:2146-2149`) | The diagnostic's `reconstruction.train_mse` is computed on stride-21 windows and `oos_mse` on stride-T (non-overlapping) windows within the same fold, while production inference uses stride 1. Overlapping windows share 99.8 % of their data, so their per-window MSEs are strongly correlated; the two numbers are not drawn from comparable sampling grids, yet they are reported side by side as a train/OOS generalisation diagnostic (`src/integration/pipeline.py:2134-2165`). Minor because it is diagnostic-only, but it can mislead model-selection decisions if anyone thresholds on it. | `src/integration/pipeline.py:2134-2165`; `:2146-2149` | U14 |
| U04-CRI-18 | data | **minor** | code | "Winsorize extreme returns caused by Tiingo API data corruption … Clip to \|r\| <= 2.0 (≈638% move)" (`src/data_pipeline/returns.py:99-102`) | The clip is documented as a data-error control, but it is a hard distributional truncation applied *before* the 21-day vol is computed (`src/integration/pipeline.py:610` operates on the already-clipped returns), so any genuine event beyond −86 % (or a corrupted +638 %) is neutralised in both channels. At ±2.0 the practical effect on daily equities is small, and the threshold is global (no look-ahead), so this is minor; it matters only because the vol channel is then a function of the clipped series, and the clip is not mentioned in the DVT's missing-value section. | `src/data_pipeline/returns.py:99-116`; `src/integration/pipeline.py:610` | U03 |

---

## Root-cause hypotheses

**RC1 — Shape/scale separation dogma: z-score everything, restore scale afterwards with one scalar.**
The unit is built on the premise that the latent must be a pure *shape* descriptor and that all magnitude must be restored ex post by σ_t. This single decision generates: the invention of a second channel to "restore" what the first removed (U04-CRI-02), the second z-scoring that defeats it, the log-transform patch and the feature-weight patch that both try to fix the resulting channel misbehaviour (U04-CRI-11/-12), the σ_252 vs σ_504 mismatch (U04-CRI-05), and the inability to express "this stock is currently 4× riskier than usual" in the representation at all.
Structural alternatives: (i) keep a single normalised return channel and add a **scale head** whose output is the conditional σ, trained with a heteroskedastic (Gaussian/Student-t) likelihood — scale becomes explicit and no second channel exists; (ii) keep a second channel but make it **level-referenced** (log realized vol minus its own long-run EWMA, not z-scored by window dispersion), so the level survives; (iii) abandon the decomposition: encode raw returns with a learned per-stock scale and let the encoder learn the scale itself. Trade-offs: (i) needs a likelihood and two heads, but removes all ad-hoc weights; (ii) is a minimal change but re-introduces a scale parameter and must be defined point-in-time; (iii) is the most expressive but gives up cross-stock comparability of the latent and makes the risk model's rescaling unnecessary (and therefore removes a safety layer).

**RC2 — One parameter, four jobs: T = 504 is simultaneously input length, listing requirement, regime horizon and validation span.**
This generates U04-CRI-10 (untestable T), U04-CRI-06 (validation set ≈ one window), and part of U04-CRI-08 (the 5 % NaN tolerance exists because T is long). No single number can satisfy "enough samples for a 200-dim latent", "risk regime coherence", "minimum listing history" and "a validation block".
Structural alternatives: (i) decouple horizons — a short input (63–126 d) for the encoder, a long estimation horizon handled by an explicit state-space layer over the latent trajectory; (ii) keep a long T but derive the validation block from `max(T, min_val_days) + embargo` rather than a fixed 2 years; (iii) hierarchical model: short-window shape encoder + a persistence model (AR/state-space) over its outputs, so "how much history is needed" becomes an estimated parameter, not an architectural constant. Trade-offs: (i) loses the long-range context the CNN was designed for; (ii) forces the validation block to be as long as the window, shrinking the number of folds (real cost); (iii) adds a component and a new estimation problem, but makes the horizon testable.

**RC3 — Aggregate-then-freeze: build a high-resolution trajectory, collapse it, freeze it.**
This generates U04-CRI-03 (static exposures), U04-CRI-04 (half-life units), U04-CRI-05 (end-of-sample shape applied backward), and the 21× memory burden of stride 1 (U04-CRI-13). It also explains why the doc has to keep claiming "local" behaviour while the code cannot deliver it.
Structural alternatives: (i) **date-indexed exposures** B_t used directly in the cross-sectional regression (a genuinely dynamic factor model); (ii) a **Kalman/state-space smoother** over the latent trajectory producing filtered exposures with an explicit persistence parameter, so today's B is a weighted function of past B (state) rather than a mean of windows; (iii) **integrate over exposure uncertainty**: treat each window's latent as a draw, estimate the cross-sectional regression per date, and let the factor covariance absorb the exposure noise, never forming a composite B at all. Trade-offs: (i) far fewer observations per exposure snapshot and a much larger parameter count (n×K×T), requiring shrinkage; (ii) introduces state dynamics and a new tuning parameter (persistence), but is the statistically principled version of "EWM aggregation"; (iii) is the cleanest conceptually and removes the most code, but changes the risk model's interface and the AU/KL measurement, which currently relies on the aggregated latents.

**RC4 — Patch the loss, not the input.**
Every time the two channels misbehaved, the fix went into the objective or into a transform: log-vol, `feature_weights`, and (historically) a cross-sectional loss. This generates U04-CRI-11/-12 and the contradictory rationales. The VAE already has a learned observation-noise scalar (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:337`), which is the principled version of a per-channel weight.
Structural alternatives: (i) **per-channel learned σ²_f** — subsumes both patches, removes two hyperparameters, and makes the reconstruction balance a fitted quantity; (ii) one channel, one likelihood — remove the redundancy (RC1) so no balancing is needed; (iii) orthogonalise the vol channel against the return channel (or replace it with a residualised local-scale feature) so unit variance implies comparable information. Trade-offs: (i) is a small, testable change but the learned σ² is itself weakly identified with a single scalar per channel; (iii) risks throwing away the signal in the shared component.

**RC5 — Missingness treated as a nuisance, not as data.**
NaN→0, ffill/bfill, fraction thresholds, a dead `min_valid_fraction`, and no delisting imputation on the shipped source generate U04-CRI-07/-08/-09/-16 and the warm-up artefact. The unit has no representation of "I do not know", so it fabricates zeros and constants.
Structural alternatives: (i) **mask-aware encoder** — add a binary missingness channel (or GRU-D-style decay) so the model can learn to discount imputed days instead of treating them as calm; (ii) a **quality gate** — require a per-window valid fraction and a *consecutive-gap* rule, and drop rather than impute; (iii) **quality weighting downstream** — carry a per-stock coverage statistic into the risk model and shrink low-coverage rows of B, which is cheap and testable. Trade-offs: (i) changes F and therefore the architecture, and complicates the z-score (masked statistics); (ii) loses data and can bias the universe toward long-listed names; (iii) does not fix the distorted windows, only their influence.

**RC6 — Spec drift: the documents describe an aspirational system, the code implements a cheaper one, and there is no deviation register.**
This generates U04-CRI-04 (half-life), -09 (Shumway), -10 (T rationale), -11 (log transform), -13 (stride), -16 (warm-up), and the config-vs-notebook-vs-doc discrepancies. Every deviation goes in the direction of cheaper or simpler, and none is recorded in the spec. This is cross-cutting (U15, U16, U17) and is arguably the meta-root-cause of the whole audit.
Structural alternatives: (i) make the config the single source of truth and generate the spec's parameter tables from it at build time (a table that cannot drift); (ii) a **deviation register** with one entry per spec rule, an owner and a test — any "DO NOT" that is not covered by a test is a known-unenforced rule; (iii) delete the rules that the implementation will not honour instead of leaving them as aspirational prose. Trade-offs: (i) requires tooling and discipline; (ii) adds maintenance overhead but converts silent drift into visible debt; (iii) risks lowering the ambition of the spec, but the audit's purpose is architectural honesty.

**RC7 — Univariate encoder, cross-sectional ambition.**
The objective is common-factor discovery, but the representation is per-stock and univariate; commonality is imposed only downstream by the regression. This generates U04-CRI-14, contributes to U04-CRI-03 (a static B is the only thing a univariate encoder can give the risk model) and explains the PCA parity in U04-CRI-15.
Structural alternatives: (i) **cross-sectional autoencoder** (Gu–Kelly–Xiu style): input the stock-date panel of standardised return features, output factors; that is the architecture that makes "common" factors identifiable at the representation level; (ii) keep the per-stock encoder but make it **conditional on a market state** (market return, VIX, dispersion) so that the latent can express "this stock's exposure given the current common state"; (iii) **hybrid**: PCA/Bai–Ng factors as an explicit common component plus the VAE residual as an idiosyncratic/latent component, which the repo already partially has (`pca_factor_rp`, `pca_vol` benchmarks) and which would make the marginal value of the VAE measurable rather than asserted. Trade-offs: (i) removes the "generalises across stocks without retraining" selling point and requires a panel-shaped model; (ii) keeps the current architecture but adds conditioning inputs and a state estimate; (iii) is the most honest comparison, but it concedes that the VAE may be a residual model.

---

## Verdict

The code is careful and the per-window z-scoring is faithfully implemented, but the *theory* of this unit does not hold. The second feature is an exact deterministic function of the first (rolling std of the same returns), so the input's information content is ≈ 524 numbers, not the advertised 1008; the doc's "complementary information" claim is false, and its own example (a stock transitioning from 10 % to 40 % vol) is annihilated by the second z-scoring it prescribes. The high-resolution trajectory that motivates stride 1 is collapsed to a single per-stock vector, so the factor model is static and the "local risk profile" language describes machinery that never reaches the risk model. The recency weighting is silently off by 21× (a window-count half-life applied to stride-1 windows), the exposure vector is measured at the end of the training period and applied to the whole history while σ-rescaling is claimed to fix it, the validation split contains roughly one independent window, and the missing-data policy (NaN→0, 5 % tolerance, dead `min_valid_fraction`, no Shumway imputation on Tiingo) is contradicted by a dataset that is 35 % missing. None of the unit's load-bearing choices — T=504, F=2, the 20 % zero threshold, the log transform, the [2.0, 0.5] channel weights, train stride 21 — has an ablation anywhere in `results/`, and the one real-data run is consistent with the design's expected failure mode (13 % OOS cross-sectional R²; a two-component PCA matching the VAE). The unit is not salvageable by tuning; it needs the shape/scale decomposition redesigned, the trajectory either used or deleted, and the cross-sectional encoding moved to where the common factors actually are.

Score: 3/10

---

## Machine companion

```json
[
  {
    "id": "U04-CRI-01",
    "unit": "U04",
    "role": "critic",
    "type": "math",
    "severity": "critical",
    "evidence_register": "doc+code",
    "claim": "The 21-day rolling volatility captures volatility dynamics — clustering, cross-stock co-movements in variance, and local regime transitions. These are complementary.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:250",
    "assessment": "False as an information claim. The vol channel is std of the same return series that channel 0 contains, computed before windowing, and both channels are z-scored in the same window. Conditional on channel 0, channel 1 is determined except for the first ~20 pre-window entries, so the nominal 1008-dim input has intrinsic dimension ~524. Cross-stock co-movement cannot be an input at all because the encoder is univariate per stock. The Inception head already has a k=21 branch that can learn a local std.",
    "evidence": ["src/data_pipeline/features.py:76-78", "src/integration/pipeline.py:610", "src/data_pipeline/windowing.py:248-260", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:250", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:327"],
    "cross_units": ["U05", "U06", "U08"],
    "root_cause": "RC1",
    "confidence": "high"
  },
  {
    "id": "U04-CRI-02",
    "unit": "U04",
    "role": "critic",
    "type": "modeling",
    "severity": "major",
    "evidence_register": "doc+code",
    "claim": "a stock with identical return shapes but transitioning from 10% to 40% annualized volatility carries fundamentally different risk, invisible to the return channel alone.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:250",
    "assessment": "False in both readings. A pure level change is a constant scaling of the vol path, which vanishes under per-window z-scoring of the vol channel; a within-window transition is already visible in the return channel because the z-score divides by one window-level sigma and preserves relative amplitude. The document concedes that per-window normalization erases volatility dynamics and then applies the same normalization to the restoring channel. An additional robustness flaw: a single extreme return rescales the entire window.",
    "evidence": ["docs/DVT_strategie_facteurs_risque_latents_v4_1.md:240", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:246-250", "src/data_pipeline/windowing.py:252-260", "src/data_pipeline/returns.py:230-246"],
    "cross_units": ["U06"],
    "root_cause": "RC1",
    "confidence": "high"
  },
  {
    "id": "U04-CRI-03",
    "unit": "U04",
    "role": "critic",
    "type": "modeling",
    "severity": "critical",
    "evidence_register": "doc+code",
    "claim": "The encoder produces a local latent vector mu_i^(t) ... aggregates all local vectors across its full history ... Between retrainings, B is fixed.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:644-646",
    "assessment": "The unit's output is a trajectory, and production discards it: aggregate_profiles reduces every stock's stack of windows to one (K,) vector, which is all the risk model sees. The factor model is therefore static; all time variation comes from Sigma_z and a scalar sigma_t. The stride-1 machinery exists only to compute a smoother average over the ~15 effective windows the document itself counts.",
    "evidence": ["src/inference/composite.py:119-171", "src/integration/pipeline.py:1976-1996", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:644-656", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:676", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:651"],
    "cross_units": ["U08", "U09", "U12"],
    "root_cause": "RC3",
    "confidence": "high"
  },
  {
    "id": "U04-CRI-04",
    "unit": "U04",
    "role": "critic",
    "type": "consistency",
    "severity": "major",
    "evidence_register": "code+results",
    "claim": "Typical: 60 (approximately 1260 trading days at stride=21, i.e. ~5 years).",
    "claim_location": "src/config.py:355-359",
    "assessment": "Half-life is counted in windows, and production inference builds windows at stride=1, so half_life=60 means 60 trading days of window-end dates, not 1260. The weighted window-end horizon is 21x shorter than documented and the data span falls from ~7 years to ~2.5-3 years. The notebook and the diagnostic run both use 60. This voids the ISD's full-history anti-cyclical claim for any run with half-life > 0.",
    "evidence": ["src/inference/composite.py:127-144", "src/inference/composite.py:155-165", "src/config.py:355-366", "src/integration/pipeline.py:1976-1996", "notebooks/dashboard.ipynb:268", "results/diagnostic/config.json:57-58", "docs/ISD_vae_latent_risk_factors.md:1800"],
    "cross_units": ["U08", "U09", "U15", "U16"],
    "root_cause": "RC3",
    "confidence": "high"
  },
  {
    "id": "U04-CRI-05",
    "unit": "U04",
    "role": "critic",
    "type": "leakage",
    "severity": "major",
    "evidence_register": "doc+code",
    "claim": "For historical factor estimation ... B_A,i,t = (sigma_i,t / sigma_bar_t) * mu_bar_A,i ... so the cross-sectional regression correctly attributes co-movement to factors.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:728-730",
    "assessment": "The single profile is inferred from windows weighted toward train_end, then applied to all training dates. Sigma_t rescaling corrects magnitude only, not shape, so a stock's 2005 exposure is a 2024 shape times a 2005 vol. Not strict future-data leakage but point-in-time impure in the CONV-10 sense, and it manufactures the claimed anti-cyclical property. Second inconsistency: z-scoring uses the 504-day within-window sigma while rescaling uses the 252-day trailing sigma.",
    "evidence": ["src/integration/pipeline.py:1976-1996", "src/integration/pipeline.py:2187-2194", "src/data_pipeline/windowing.py:252-260", "src/data_pipeline/features.py:39-40", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:728-734"],
    "cross_units": ["U08", "U09", "U12"],
    "root_cause": "RC3",
    "confidence": "medium"
  },
  {
    "id": "U04-CRI-06",
    "unit": "U04",
    "role": "critic",
    "type": "statistics",
    "severity": "major",
    "evidence_register": "code+results",
    "claim": "the same split used for nested hyperparameter selection ... the selected checkpoint is, by construction, the one with the best score on this set.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:620",
    "assessment": "The split is by window end date only, with val_years=2 and T=504 (~2 years), so every validation window overlaps training by up to 503/504 days. The document's own ESS formula gives roughly one independent validation window per stock, so early stopping selects E* from ~1 effective observation. No overlap control is mentioned and no test exists.",
    "evidence": ["src/integration/pipeline.py:1284-1289", "results/diagnostic/config.json:121", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:620-626", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:651"],
    "cross_units": ["U07", "U12"],
    "root_cause": "RC2",
    "confidence": "high"
  },
  {
    "id": "U04-CRI-07",
    "unit": "U04",
    "role": "critic",
    "type": "data",
    "severity": "major",
    "evidence_register": "doc+code+results",
    "claim": "Missing value handling: rare on the target universe (large-cap US equities on CRSP).",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:258",
    "assessment": "The actual run has 35.31% missing values and 410 of 788 stocks above 20% missing. The config field min_valid_fraction that would guard this is never read anywhere in src, a dead parameter. With a 5% NaN tolerance per window, many windows mix two years of history with several weeks of fabricated zero returns.",
    "evidence": ["results/diagnostic/diagnostic_report.md:30-31", "src/config.py:137", "src/data_pipeline/windowing.py:213-215", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:258"],
    "cross_units": ["U03"],
    "root_cause": "RC5",
    "confidence": "high"
  },
  {
    "id": "U04-CRI-08",
    "unit": "U04",
    "role": "critic",
    "type": "doc-vs-code",
    "severity": "major",
    "evidence_register": "doc+code",
    "claim": "exclude the window if a gap exceeds 5 consecutive days (temporal continuity broken over the shortest Inception kernel scale).",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:258",
    "assessment": "Not implemented: windowing checks only a NaN fraction, so a 20-day contiguous gap passes and becomes 20 zero returns. NaN-to-zero is also counted against the zero-return budget, conflating missingness with suspension; the 20% threshold tolerates ~101 stale days in a window. The vol channel is bfilled with within-window future values, creating flat segments at every gap.",
    "evidence": ["src/data_pipeline/windowing.py:212-233", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:242", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:258"],
    "cross_units": ["U03", "U05"],
    "root_cause": "RC5",
    "confidence": "high"
  },
  {
    "id": "U04-CRI-09",
    "unit": "U04",
    "role": "critic",
    "type": "honesty",
    "severity": "major",
    "evidence_register": "code+data",
    "claim": "If delisted without an available return, a corrective return is imputed (Shumway convention: -30% NYSE/AMEX, -55% Nasdaq). DO NOT forget the Shumway delisting return imputation.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:265",
    "assessment": "On Tiingo the imputation never fires: the merge writes delisting_return as NaN on every row, and _extract_delisting_info records a stock only if some row has a non-NaN value, so delisting_info is empty. The encoder therefore never sees the imputed terminal loss the design says it learns from, and the survivorship-bias-free training claim is unsupported on the shipped data.",
    "evidence": ["scripts/download_tiingo.py:911", "src/data_pipeline/returns.py:187-195", "src/data_pipeline/returns.py:200-246", "docs/ISD_vae_latent_risk_factors.md:889-891", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:263-265"],
    "cross_units": ["U03", "U16"],
    "root_cause": "RC5",
    "confidence": "high"
  },
  {
    "id": "U04-CRI-10",
    "unit": "U04",
    "role": "critic",
    "type": "justification",
    "severity": "major",
    "evidence_register": "doc+code+results",
    "claim": "T = 504 days (~2 years): long enough for several market conditions (calm, crisis, recovery), short enough for a coherent risk regime.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:301",
    "assessment": "Three different requirements are collapsed into one integer: the document also calls T purely architectural and uses it as the minimum listing history. Two years contains at most one stress episode; the document's own ESS arithmetic implies ~15 independent 21-day blocks per window. No T ablation exists in results; only one configuration is present.",
    "evidence": ["docs/DVT_strategie_facteurs_risque_latents_v4_1.md:273", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:297", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:301", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:651", "results/diagnostic/config.json:4", "src/integration/action_specs.py:363-372"],
    "cross_units": ["U05", "U12", "U15"],
    "root_cause": "RC2",
    "confidence": "high"
  },
  {
    "id": "U04-CRI-11",
    "unit": "U04",
    "role": "critic",
    "type": "doc-vs-code",
    "severity": "major",
    "evidence_register": "code+doc",
    "claim": "apply log(vol + 1e-8) before z-scoring. Compresses the volatility distribution, preventing the VAE from over-allocating latent capacity to volatility patterns.",
    "claim_location": "src/data_pipeline/windowing.py:132-134",
    "assessment": "The log transform appears in neither specification; a repo-wide search finds it only in code and in the changelog, where it was introduced as a V2 post-mortem fix and explicitly kept after the Phase-0 revert. It is not a no-op (it changes the channel's skew/kurtosis) and its stated benefit is untested. It stacks with an untested channel-weighting patch on the same channel.",
    "evidence": ["src/data_pipeline/windowing.py:110", "src/data_pipeline/windowing.py:244-249", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:244-248", "docs/ISD_vae_latent_risk_factors.md:804-809", ".claude/rules/project/changelog.md:13", ".claude/rules/project/changelog.md:16"],
    "cross_units": ["U06", "U15", "U17"],
    "root_cause": "RC4",
    "confidence": "medium"
  },
  {
    "id": "U04-CRI-12",
    "unit": "U04",
    "role": "critic",
    "type": "justification",
    "severity": "major",
    "evidence_register": "code+doc",
    "claim": "This addresses reconstruction imbalance where the VAE over-fits to easy-to-reconstruct features (e.g., volatility) at the expense of harder features (e.g., returns).",
    "claim_location": "src/vae/loss.py:148-151",
    "assessment": "The rationale contradicts the change's own history (a 10x imbalance in which vol error was larger) and a mean-MSE already gives less weight to an easy channel. Production weights [2.0,0.5] give 80/20 while the code default is equal weights, a 4x relative channel weight that depends on the launch path. Two unit-variance z-scored channels should have comparable MSE; a 10x gap is evidence about the features, not about the weights.",
    "evidence": ["src/vae/loss.py:137-194", "notebooks/dashboard.ipynb:253", "notebooks/dashboard.ipynb:700", "src/config.py:244", ".claude/rules/project/changelog.md:16"],
    "cross_units": ["U06"],
    "root_cause": "RC4",
    "confidence": "high"
  },
  {
    "id": "U04-CRI-13",
    "unit": "U04",
    "role": "critic",
    "type": "consistency",
    "severity": "major",
    "evidence_register": "doc+code",
    "claim": "If training windows exceed capacity, windows are subsampled (stride 5 or 21), not stocks.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:281",
    "assessment": "The documented contingency is the permanent code default (21), the doc's parameter table says the default is 1, and inference runs at stride 1. If the mean profile is quasi-identical for any stride <= 21, the stride-1 inference path computes nothing new while consuming the pipeline's largest memory allocation; if it is not, the encoder is evaluated off its training distribution. Training windows per stock also scale with listing length and the loss weights them uniformly.",
    "evidence": ["docs/DVT_strategie_facteurs_risque_latents_v4_1.md:281", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:648-654", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:1001", "src/config.py:140", "notebooks/dashboard.ipynb:263", "src/integration/pipeline.py:1790", "src/integration/pipeline.py:1976-1980", "src/vae/loss.py:189-194"],
    "cross_units": ["U05", "U07", "U08"],
    "root_cause": "RC6",
    "confidence": "high"
  },
  {
    "id": "U04-CRI-14",
    "unit": "U04",
    "role": "critic",
    "type": "modeling",
    "severity": "major",
    "evidence_register": "doc+code",
    "claim": "Raw series let the model discover which statistics matter — conditional correlations, volatility synchronization, tail dependence.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:222",
    "assessment": "The encoder is a per-stock univariate map and never observes another stock, so it cannot represent conditional correlations or synchronization at the input. The only cross-sectional training signal is the co-movement loss, which is curriculum-scheduled to zero. Commonality is discovered by the downstream cross-sectional regression on B, making this a conventional factor model with nonlinear PCA loadings.",
    "evidence": ["docs/DVT_strategie_facteurs_risque_latents_v4_1.md:222", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:307", "docs/ISD_vae_latent_risk_factors.md:1363-1387", "src/inference/composite.py:21-116"],
    "cross_units": ["U05", "U06", "U08"],
    "root_cause": "RC7",
    "confidence": "high"
  },
  {
    "id": "U04-CRI-15",
    "unit": "U04",
    "role": "critic",
    "type": "interpretation",
    "severity": "major",
    "evidence_register": "results",
    "claim": "Avg cross-sectional R2 (OOS): 0.1300 ... Top 1 factor contribution: 81.1% ... pca_vol Sharpe 0.3678 vs VAE 0.3816.",
    "claim_location": "results/diagnostic/diagnostic_report.md:61",
    "assessment": "The downstream evidence matches the predicted failure mode of a shape-only, level-free, static, univariate representation: 13.0% OOS cross-sectional R2 (21.9% IS), 81.1% of risk in one principal factor, and a two-component PCA on the same features matching the VAE within one fold's noise. Supporting rather than conclusive because the encoder was pretrained and the recorded config (K=75) is not the run's effective config (K=52).",
    "evidence": ["results/diagnostic/diagnostic_report.md:35", "results/diagnostic/diagnostic_report.md:39", "results/diagnostic/diagnostic_report.md:48", "results/diagnostic/diagnostic_report.md:59-61", "results/diagnostic/diagnostic_report.md:92", "results/diagnostic/diagnostic_report.md:115", "results/diagnostic/config.json:14"],
    "cross_units": ["U05", "U09", "U14", "U15", "U17"],
    "root_cause": "RC7",
    "confidence": "medium"
  },
  {
    "id": "U04-CRI-16",
    "unit": "U04",
    "role": "critic",
    "type": "implementation",
    "severity": "minor",
    "evidence_register": "code",
    "claim": "DO NOT forget the 21-day warm-up for realized volatility — the rolling vol at the start of a window must be based on pre-window returns.",
    "claim_location": "docs/ISD_vae_latent_risk_factors.md:890",
    "assessment": "Not enforced. create_windows checks only a 5% NaN fraction; 21/504 = 4.2% passes, so the first windows of each stock have a fully NaN 21-day warm-up that is bfilled into a constant, creating a flat vol artefact the encoder can memorise. Fold boundaries are safe because vol is computed globally before slicing.",
    "evidence": ["docs/ISD_vae_latent_risk_factors.md:890", "src/data_pipeline/windowing.py:213-229", "src/integration/pipeline.py:610"],
    "cross_units": ["U03"],
    "root_cause": "RC5",
    "confidence": "high"
  },
  {
    "id": "U04-CRI-17",
    "unit": "U04",
    "role": "critic",
    "type": "implementation",
    "severity": "minor",
    "evidence_register": "code",
    "claim": "_oos_w, _, _ = create_windows(..., T=self.config.data.window_length, stride=T).",
    "claim_location": "src/integration/pipeline.py:2146-2149",
    "assessment": "The train/OOS reconstruction MSE diagnostic compares stride-21 training windows with stride-T OOS windows while production inference uses stride 1. The numbers come from different sampling grids over 99.8%-overlapping data and are not comparable; minor because it is diagnostic-only, but it can mislead model selection if thresholded.",
    "evidence": ["src/integration/pipeline.py:2134-2165", "src/integration/pipeline.py:2146-2149"],
    "cross_units": ["U14"],
    "root_cause": "RC6",
    "confidence": "medium"
  },
  {
    "id": "U04-CRI-18",
    "unit": "U04",
    "role": "critic",
    "type": "data",
    "severity": "minor",
    "evidence_register": "code",
    "claim": "Winsorize extreme returns caused by Tiingo API data corruption ... Clip to |r| <= 2.0 (approximately 638% move).",
    "claim_location": "src/data_pipeline/returns.py:99-102",
    "assessment": "A hard distributional truncation documented as a data-error control, applied before the 21-day vol is computed, so the vol channel is a function of the clipped series. At +/-2.0 the practical effect on daily equities is small and the threshold is global (no look-ahead), so this is minor; it matters only because the clipping is absent from the DVT's preprocessing specification.",
    "evidence": ["src/data_pipeline/returns.py:99-116", "src/integration/pipeline.py:610"],
    "cross_units": ["U03"],
    "root_cause": "RC5",
    "confidence": "medium"
  }
]
```

END-OF-REPORT
