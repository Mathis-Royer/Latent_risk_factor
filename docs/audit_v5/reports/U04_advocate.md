# U04 — advocate

## Scope read

- `docs/DVT_strategie_facteurs_risque_latents_v4_1.md` — §4.2 Preprocessing and frequency (210–258), §4.3 Window length T (295–303), encoder rationale (305–317), §4.6 inference/composite profiles and aggregation (642–656), §5.2 key parameter relationships (1066–1080).
- `docs/ISD_vae_latent_risk_factors.md` — MOD-001 objective/outputs (544–562), data schema (585–613), Sub-task 2 log-returns/missing values (723–742), Sub-task 4 windowing contract (791–816), Sub-task 6 volatility (844–891).
- `src/data_pipeline/windowing.py` (full file, 307 lines).
- `src/data_pipeline/features.py` (full file, 87 lines).
- `src/config.py` — `DataPipelineConfig` (123–160), `VAEArchitectureConfig` (180–200), `InferenceConfig` aggregation (355–370), `RiskModelConfig.market_intercept` (464), `PortfolioConfig.refresh_risk_model_oos` (647–650).
- `src/integration/pipeline.py` — Phase A windowing/validation split (1252–1296), fold training windows (1745–1819), inference stride 1 (1960–2014), OOS reconstruction windows (2128–2172).
- `src/inference/composite.py` (100–171).
- `src/risk_model/rescaling.py` (1–10, 22–49).
- `src/vae/loss.py` (1–13, 199–240).
- `src/validation.py` (1061–1091).
- `tests/integration/test_pipeline_stages.py` (95–164); `tests/unit/test_data_quality.py` (455–560, 1218–1262); `tests/unit/test_interface_contracts.py` (via grep, 145–175); `tests/unit/test_divergences.py` (via grep, 475–530).
- `notebooks/dashboard.ipynb` — runtime cell (220–263), config assembly (543–555, 646–652), loss cell (693–698).
- `results/diagnostic/config.json`, `results/diagnostic/diagnostic_data.json`, `results/diagnostic/vae_fold_metrics.csv`, `results/config.json`; `README.md`.
- `src/data_pipeline/returns.py` — inspected by targeted grep only (27–94), not read in full.

## Five questions

**1. Purpose and original intent.**
MOD-001's stated job is to produce the only tensor the encoder ever sees — `(N, T, F)` z-scored windows, `T=504`, `F=2` (ISD:557), plus raw returns for the co-movement loss (windowing.py:278, ISD:560). The preprocessing spec has three intents: (i) force the encoder to learn *structural* patterns rather than absolute levels — "Two windows with identical shapes but different raw volatilities will produce identical latent vectors, which is desirable: they represent the same risk *structure*" (DVT:240); (ii) reject look-ahead normalization — per-window vs expanding vs full-history is compared explicitly and full-history is marked "Look-ahead? **Yes**" (DVT:234–238); (iii) filter unusable windows — near-zero σ clamp at 1e-8 and exclusion above 20% zero-return days (DVT:242), with gaps >5 consecutive days expected to invalidate a window (DVT:258). `T` is framed as an architectural constraint, not a history choice (DVT:297), and `s_train` is explicitly decoupled from `s_inf` (DVT:654).

**2. What a state-of-the-art version would look like.**
The current design is, in modern terms, *instance normalization without the inverse*: per-window standardization is the time-series analogue of instance normalization (Ulyanov et al., 2016), and the correct modern treatment is reversible instance normalization — normalize per instance for the model and re-inject the instance statistics outside it (RevIN; Kim et al., 2021). This is exactly what the pipeline does globally: shape into the VAE, scale restored by `B̃ = (σ_i,t/σ̄_t)·μ̄` (rescaling.py:4–5). A SOTA version would make that contract explicit and *per-channel* rather than implicit. On the volatility channel, the literature-standard supervised transformations are log realized variance (Andersen, Bollerslev, Diebold & Labys, 2003) and multi-horizon HAR components 1/5/22 days (Corsi, 2009) or realized semivariance (Barndorff-Nielsen, Kinnebrock & Shephard, 2010); a single 21-day std is the crudest member of that family. On exposures, SOTA risk models condition exposures on characteristics with a cross-sectional objective (Gu, Kelly & Xiu, 2021, IPCA; Chen, Pelger & Zhu, 2024, deep factor models), i.e. the training signal itself is cross-sectional. On `T`, SOTA chooses the window by validation against the downstream objective, not by assertion. None of this is exotic; all of it is compatible with the existing pipeline.

**3. What prevents the current unit from reaching it — and which obstacles are only apparent.**
The binding obstacle is not feature engineering. The representation is per-stock and scale-free (windowing.py:189–232), so it carries zero cross-sectional information, and the training objective contains essentially no cross-sectional coupling: the co-movement loss is capped at 0.1 (notebook:250) and the cross-sectional R² loss is disabled ("confirmed harmful in V2", notebook:251; `lambda_cs=0.0`, config.py:264). Factor alignment is therefore emergent from shared weights, not trained. Second real obstacle: ~15 independent windows per stock over 30 years, ~5 over a 10-year training fold (DVT:651, DVT:1076) — the doc states this correctly and it caps how much a 504-day window can contribute. Only-apparent obstacles: (a) "add features F>2" — increases input dimension without increasing independent observations, so it cannot fix the objective/ESS constraint (DVT:1074 already frames T as the information lever, not F); (b) "T is too short/long" — because the default aggregation is a uniform mean over all history (DVT:656, config.py:366), regime coverage is re-blended downstream regardless of T; (c) "per-window z-scoring loses volatility level" — true but deliberately restored by rescaling, so not a defect while rescaling exists.

**4. Real vs self-imposed constraints.**
Real: no look-ahead (structural in the code, windowing.py:209–215); fixed `T` for a CNN; memory (stride-1 inference is ~32 GB per code comment, pipeline.py:1998–2001, on a 16 GB machine); ESS ≈ `T_hist/T` (DVT:1076); unequal listing histories and suspensions; non-stationarity. Self-imposed: inference stride 1 (DVT:651 says the result is quasi-identical for any `s_inf ≤ 21`); the 5% *aggregate* NaN tolerance; the 20% zero-return tolerance; the log-vol default (not in any spec); uniform-mean aggregation with `half_life=0`; `F=2`; `T=504` sitting exactly at the document's own upper bound (DVT:1074).

**5. What to remove, merge or move.**
- **Remove inference stride 1 as the default.** With mean aggregation (composite.py:155–165) the doc itself says `s_inf ≤ 21` is quasi-identical (DVT:651); stride 1 costs 21× compute and multi-GB allocations (pipeline.py:1976–1980, 1998–2001). Keep it as an opt-in diagnostic trajectory only. Nothing is lost.
- **Remove the 5% aggregate NaN tolerance.** The spec is a *consecutive-gap* rule (DVT:258, ISD:734), and the universe already requires 504 continuous listing days (DVT:273); the aggregate tolerance can be dropped at zero cost and eliminates fabricated zero returns.
- **Remove the log-vol default, or make it an explicit, tested option with a zero-vol guard.** As written it is undocumented, untested, and creates a real outlier mechanism (finding 07).
- **Merge/consider collapsing F=2 to F=1 + a 21-day pre-window pad.** Under per-window standardization the RV channel is a smoothed function of `|r̃|`; its unique content is the 20 pre-window days (~4% of the window) plus a smoothing prior. If the redesign wants fewer moving parts, this is the safe merge.
- **Move the shape/scale contract out of prose** (DVT:240 + rescaling.py:4) into an explicit interface between U04/U08/U09, and treat `market_intercept` (config.py:464) as the deliberate Barra-style market factor it is, not as a patch.
- **Remove the duplicate count/fill logic** (`_count_valid_windows_for_stock` duplicates the filters; windowing.py:60) — one source of truth, chunked output if memory requires it.

## Load-bearing assumptions

| ID | Assumption | Where stated | Evidenced? | Comment |
|----|------------|--------------|------------|---------|
| A1 | `T=504` is long enough to contain several regimes and short enough to be coherent | DVT:301 | no | Asserted; no 252/504/756 ablation; 504 is the doc's own upper bound (DVT:1074) |
| A2 | Shape (not level) is the risk-relevant invariant | DVT:240 | partially | True mathematically (affine invariance); that shape suffices for risk is an assumption |
| A3 | The RV channel adds information the z-scored returns lack | DVT:250 | no | False as stated: RV is itself per-window z-scored (DVT:248); incremental info is 20 pre-window days + smoothing |
| A4 | Overlapping windows do not bias learning | DVT:1076, 651 | partially | ESS is acknowledged (~15/stock) but windows are still weighted uniformly at stride 21 |
| A5 | Absolute vol is restored downstream | rescaling.py:4–5; config.py:464 | yes | `σ_i,t/median(σ_t)`, winsorized (rescaling.py:38–47) |
| A6 | Zero returns = suspension; ≤20% tolerable | DVT:242 | partially | No empirical distribution check in `results/` |
| A7 | Missing returns may be imputed as 0 without distorting shape | windowing.py:227; DVT:258 | no | Spec only permits this for gaps ≤5 consecutive days; code permits ≤5% aggregate |
| A8 | The window's own μ, σ are the right normalizers | DVT:232–238 | yes (by construction) | No look-ahead; alternatives not tested downstream |
| A9 | A uniform mean over all windows preserves risk-relevant information | DVT:656; config.py:366 | no | `half_life` exists but default 0; no ablation |
| A10 | `log(vol)` is a better input than `vol` | windowing.py:132–134 | no | Literature supports log-RV generally; no test/ablation in repo |
| A11 | A model estimated on the fold's training windows represents OOS factor structure | pipeline.py:1782–1791 | partially | Implemented; shape of B is frozen at `train_end` (config.py:357–360 comments admit staleness) |

## Findings

| ID | Type | Severity | Evidence register | Claim (quoted, path:line) | Assessment | Evidence (path:line) | Cross-unit |
|----|------|----------|-------------------|---------------------------|------------|----------------------|------------|
| U04-ADV-01 | leakage / logic | high | doc+code | "where μ̂_i^(t) and σ̂_i^(t) are the sample mean and standard deviation of stock i's raw log-returns *within that window only*" (DVT:230) | **Solid.** Look-ahead is structurally impossible: windows are indexed by end date ≤ `n_dates−1`, and both channels are standardized with within-window statistics only. Folds build windows on `returns.loc[train_start:train_end]` and inference reuses the same slice. Full-history normalization is explicitly rejected on look-ahead grounds. | windowing.py:209–215, 252–259; pipeline.py:1784–1791, 1976–1980; DVT:234–238 | U12 |
| U04-ADV-02 | modeling | high | doc+code | "Two windows with identical shapes but different raw volatilities will produce identical latent vectors, which is desirable: they represent the same risk *structure*." (DVT:240) | **Conditional — and this is the unit's central strength.** The window map is invariant to positive affine transforms of each channel; absolute scale is deliberately re-injected downstream as `σ_i,t/σ̄_t`. Holds only if (i) rescaling is always applied (it is) and (ii) shape is stable over the aggregation horizon. It cleanly separates "what co-moves" from "how much it moves". | windowing.py:252–259; rescaling.py:4–5, 38–47; config.py:464 | U08, U09 |
| U04-ADV-03 | implementation | medium | code | "CONV-02: Z-score per-window, per-feature (mean 0, std 1)" (ISD:811) | **Solid and verified.** Per-window per-feature mean `<1e-4` and std within `0.05` are asserted in tests; NaN/Inf are excluded by construction and asserted; `σ_min=1e-8` prevents division blow-up. The implementation matches the invariant exactly, including feature order `[r̃, ṽ]`. | windowing.py:252–274, 278–280; test_pipeline_stages.py:108–118; test_data_quality.py:459–477; DVT:254 | U07 |
| U04-ADV-04 | statistics / honesty | high | doc | "The effective number of independent training samples is closer to T_hist/T than to T_hist − T + 1." (DVT:1076); "N_eff ≈ T_hist/T ≈ 15 independent windows are determined by window overlap, not stride" (DVT:651) | **Solid, and unusually honest.** The quantitative consequence is decisive: ~15 independent windows per stock over 30 years, ~5 over a 10-year fold. This is the correct statistical frame and should be inherited by any redesign rather than "fixed" by adding features. | DVT:1068, 1076, 651; config.py:128–140 | U07, U12 |
| U04-ADV-05 | modeling / implementation | medium | doc+code | "Inference stride s_inf = 1 (daily) … ~7,057 for 30 years" (DVT:648), justified by "the result is quasi-identical for any s_inf ≤ 21" (DVT:651) | **Fragile.** The code uses stride 1 (pipeline.py:1979) and then must delete ~32 GB of windows (pipeline.py:1998–2001). The statistical gain over stride 21 is zero under the default mean (composite.py:155–165); the cost is 21× compute and a memory cliff. Correct in principle, wasteful in default configuration. | pipeline.py:1976–1980, 1998–2001; DVT:648–654; composite.py:165 | U08, U15 |
| U04-ADV-06 | doc-vs-code | medium | doc+code | "the second input feature is the rolling 21-day standard deviation of raw returns, z-scored per window" (DVT:244–246) vs code default `log_transform_vol: bool = True` (windowing.py:110) | **Divergence.** Neither DVT nor ISD mentions a log transform (ISD:808 specifies raw `v`). The transform is literature-defensible (log-RV), but it is unstated, untested, and its stated rationale — "prevents the VAE from over-allocating latent capacity to volatility patterns" (windowing.py:132–134) — is a hypothesis with no supporting run. The code is arguably ahead of the spec, but the audit cannot treat an undocumented default as intended design. | windowing.py:110, 132–134, 248–249; DVT:246, 248; ISD:808 | U17 |
| U04-ADV-07 | math / implementation | medium | code | `vol_clean = np.log(np.maximum(vol_clean, 1e-8))` (windowing.py:249) applied after only a 20% zero-return filter (windowing.py:232) | **Fragile.** The 21-day rolling std is exactly 0 over 21 constant returns (features.py:76–78). `log(1e-8) = −18.4` versus typical `log(vol) ≈ −4.6`, so such entries become −10σ to −30σ after standardization. The pooled validation tolerance (`std_tol=0.1`, validation.py:1087) cannot detect a few outliers among millions of values. Testable prediction: on Tiingo windows, the |z|>10 rate in channel 2 exceeds channel 1 and concentrates in windows containing ≥21 consecutive zero returns. | windowing.py:227–232, 249, 257–259; features.py:76–78; validation.py:1085–1088; test_data_quality.py:1222–1235 | U03 |
| U04-ADV-08 | doc-vs-code / data | high | doc+code | "exclude the window if a gap exceeds 5 consecutive days" (DVT:258; ISD:734) vs `nan_ok = (ret_nan_frac <= 0.05) & (vol_nan_frac <= 0.05)` (windowing.py:215) | **Real divergence.** A 6–25 day gap is admitted and then zero-imputed (`np.nan_to_num(..., nan=0.0)`, windowing.py:227), fabricating a flat segment, biasing window μ downward and σ downward. The spec's unit is *consecutive* days; the code's unit is *aggregate fraction*. The universe's 504-day continuous-listing rule (DVT:273) makes this tolerance unnecessary. This is a correctness issue, not style. | windowing.py:215, 227; returns.py:33, 43–44; DVT:258, 273; ISD:734 | U03, U17 |
| U04-ADV-09 | justification / math | medium | doc+code | "The z-scored return preserves *temporal shape* … but erases the absolute volatility scale. The 21-day rolling volatility captures *volatility dynamics*" (DVT:250), with RV "z-scored per window" (DVT:246–248) | **Wrong argument, survivable design.** Per-window z-scoring uses window constants, so a mid-window vol doubling remains visible as an amplitude change in `r̃`; and RV is standardized with the same operation, so it cannot restore the absolute level either. The incremental content of channel 2 is: 20 pre-window days (~4%), a 21-day smoothing prior, and log compression. F=2 is still defensible — for inductive bias and warm-up information — but not for the reason stated. | DVT:240, 246, 248, 250; windowing.py:252–259; features.py:76–78 | U05 |
| U04-ADV-10 | modeling | high | doc+code | "Raw time series are used rather than pre-calculated correlations … Raw series let the model discover *which* statistics matter" (DVT:222); windows are built per stock (windowing.py:189–232) | **The unit's true bottleneck, and it is not fixable inside U04.** The representation contains no cross-sectional information. The only cross-stock terms are the co-movement loss (Spearman×cosine, loss.py:9) at ≤0.1 and a cross-sectional R² loss that is disabled (`LAMBDA_CS = 0.0`, notebook:251). Factor identity is emergent from weight sharing, never optimized. Changing T, F or the normalizer will not change this. | DVT:220, 222; windowing.py:189–232; loss.py:9, 201–232; notebook:250–251 | U05, U06, U08 |
| U04-ADV-11 | statistics | medium | code | `val_mask = metadata["end_date"] >= pd.Timestamp(val_start)` (pipeline.py:1285) | **Fragile but not leakage.** Labeling windows by end date prevents information from crossing the split, but the last ~24 training windows (504/21) share ≥95% of their days with the first validation window. Early stopping and Phase-A HP selection therefore see quasi-duplicates across the boundary. The windowing design guarantees no *informational* leakage, not *statistical* independence. | pipeline.py:1284–1289; windowing.py:209; DVT:876 | U12, U07 |
| U04-ADV-12 | data | low | code | `vol_clean = _vectorized_ffill_bfill(vol_wins)` (windowing.py:228) with a 5% NaN tolerance (windowing.py:215) | **Systematic fabrication, small scope.** Rolling vol has 20 leading NaNs (`min_periods=21`, features.py:76–78) = 3.97% < 5%, so every stock's earliest window is admitted and its first 20 vol points are back-filled with a constant. Harmless in magnitude, but it demonstrates that an aggregate fraction is the wrong instrument; a max-consecutive-gap rule removes it at no cost. | windowing.py:215, 228–229; features.py:76–78; DVT:252 | U03 |
| U04-ADV-13 | justification | medium | doc | "T = 504 days (~2 years): long enough for several market conditions (calm, crisis, recovery), short enough for a coherent risk regime." (DVT:301), while "Long T (> 504 days) increases the probability that a window spans a structural break" (DVT:1074) | **Conditional.** Both criteria are asserted, not measured, and 504 sits exactly at the document's own upper bound — zero headroom. There is no ablation of 252/504/756 on explanatory power or reconstruction. The value is a reasonable prior, not a derived optimum; a redesign should select T by validation against the downstream objective. | DVT:295–303, 1074 | U15 |
| U04-ADV-14 | implementation | low | code | "This duplicates the filtering logic from create_windows but only counts" (windowing.py:60) | **Fragile maintenance hazard.** Pass 1 counts (windowing.py:76–99); pass 2 fills `np.empty` arrays (windowing.py:178–185) and slices to `current_idx` (windowing.py:290). If the two filter implementations ever diverge, over-count leaves uninitialized rows in the returned tensor; `assert_no_nan_tensor` (windowing.py:302) catches NaN but not finite garbage. Currently identical, so not a bug. | windowing.py:60, 76–99, 178–185, 290, 302 | U07 |
| U04-ADV-15 | modeling / consistency | medium | doc+code | "T is … short enough for a coherent risk regime" (DVT:301) vs "Default aggregation: simple mean (all windows contribute equally, preserving memory of all historical regimes)" (DVT:656); `aggregation_half_life: int = 0` (config.py:366) | **Conditional.** The window is regime-coherent by construction, but the profile is a 10–30-year uniform average; regime specificity is discarded at aggregation, not at windowing. This is deliberate ("preserving memory") and `half_life` exists, but with the default at 0 the T=504 coherence rationale is largely undone downstream. | DVT:656, 301; config.py:366; composite.py:130–165 | U08 |

## Core strengths to preserve

1. **Structural no-look-ahead.** End-date-indexed windows plus within-window standardization make future information physically unable to enter the representation (windowing.py:209–215, 252–259). Any redesign must keep windows labeled by their end date and normalized on their own contents; this is the cheapest and strongest guarantee in the unit.
2. **Shape/scale separation.** Standardizing both channels per window creates an affine-invariant encoder; all absolute scale is re-injected by `σ_i,t/σ̄_t` (rescaling.py:4–5). This division of labour is the correct architecture for a factor *risk* model and should be an explicit interface contract, not an accident.
3. **Honest ESS accounting.** DVT:1076 and DVT:651 state that overlap, not stride, determines effective sample size (~15 independent windows per stock over 30 years). Preserve this framing: it prevents the redesign from "buying" information it does not have by changing stride or F.
4. **Deterministic, tested normalization contract.** CONV-02 is implemented exactly and asserted per-window in tests; NaN/Inf are excluded by construction; the `σ_min` clamp is present (windowing.py:252–274; test_pipeline_stages.py:108–118). This is a verified guarantee the rest of the pipeline can rely on.
5. **Separate, decoupled strides.** Training stride and inference stride are independent parameters with documented semantics (config.py:128–140; DVT:654); the training stride is a pure compute knob, and the docs say so. Keep the decoupling, drop the wasteful default.
6. **Raw series rather than pre-computed correlations.** Preserving the raw path lets the encoder see conditional co-movement, tails and clustering rather than an average pairwise statistic (DVT:222). This is the right input primitive for a factor model even though the current objective underuses it.

## Verdict

The unit is technically sound where it matters most: no look-ahead, exact per-window per-feature standardization, correct feature ordering and shapes, and an unusually honest treatment of effective sample size. Its deepest design choice — per-window standardization that makes the encoder see only *shape*, with scale restored downstream — is correct for a factor risk model, even though the document defends it with a wrong argument about the realized-volatility channel. The concrete weaknesses are bounded and diagnosable: an aggregate 5% NaN tolerance that contradicts the spec's consecutive-gap rule, a log-vol path that can manufacture extreme outliers from zero-vol stretches, a 21×-redundant inference stride, and duplicate filtering logic with a silent-garbage failure mode. None of these is the real ceiling. The real ceiling is that the window carries no cross-sectional information and the training objective contains almost none, so the latent axes can only become factors by accident of weight sharing; that cannot be repaired by adding features, lengthening T or changing the normalizer, and any redesign that treats windowing as the bottleneck will spend effort in the wrong place. As a data-representation layer, this is a competent, defensible implementation with a clear contract; as the foundation for the stated objective, it is necessary but structurally insufficient.

Score: 7/10

## Machine companion

```json
[
  {
    "id": "U04-ADV-01",
    "unit": "U04",
    "role": "advocate",
    "type": "leakage",
    "severity": "high",
    "evidence_register": "doc+code",
    "claim": "Per-window normalization uses only within-window statistics and windows are indexed by end date; no future information can enter the representation.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:230",
    "assessment": "Solid. Window end indices are bounded by n_dates-1, both channels are standardized with within-window mean/std, and folds construct windows only on [train_start, train_end]. Full-history normalization is explicitly rejected for look-ahead in the spec comparison table.",
    "evidence": ["src/data_pipeline/windowing.py:209-215", "src/data_pipeline/windowing.py:252-259", "src/integration/pipeline.py:1784-1791", "src/integration/pipeline.py:1976-1980", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:234-238"],
    "cross_units": ["U12"],
    "root_cause": null,
    "confidence": "high"
  },
  {
    "id": "U04-ADV-02",
    "unit": "U04",
    "role": "advocate",
    "type": "modeling",
    "severity": "high",
    "evidence_register": "doc+code",
    "claim": "Two windows with identical shapes but different raw volatilities produce identical latent vectors; this is desirable because they represent the same risk structure.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:240",
    "assessment": "Conditional and central: the window map is affine-invariant, and absolute scale is deliberately re-injected downstream as sigma_i,t / median(sigma_t), winsorized. Holds only if rescaling is always applied and if shape is the stable, risk-relevant invariant over the aggregation horizon.",
    "evidence": ["src/data_pipeline/windowing.py:252-259", "src/risk_model/rescaling.py:4-5", "src/risk_model/rescaling.py:38-47", "src/config.py:464"],
    "cross_units": ["U08", "U09"],
    "root_cause": null,
    "confidence": "high"
  },
  {
    "id": "U04-ADV-03",
    "unit": "U04",
    "role": "advocate",
    "type": "implementation",
    "severity": "medium",
    "evidence_register": "code",
    "claim": "CONV-02 is implemented exactly: per-window, per-feature z-scoring with a sigma_min clamp, feature order [returns, vol], and no NaN/Inf in the output.",
    "claim_location": "docs/ISD_vae_latent_risk_factors.md:811",
    "assessment": "Solid and verified by tests: per-window means below 1e-4, per-window std within 0.05, no NaN/Inf, metadata aligned with windows, raw returns returned separately for the co-movement loss.",
    "evidence": ["src/data_pipeline/windowing.py:252-274", "src/data_pipeline/windowing.py:278-280", "tests/integration/test_pipeline_stages.py:108-118", "tests/unit/test_data_quality.py:459-477"],
    "cross_units": ["U07"],
    "root_cause": null,
    "confidence": "high"
  },
  {
    "id": "U04-ADV-04",
    "unit": "U04",
    "role": "advocate",
    "type": "statistics",
    "severity": "high",
    "evidence_register": "doc",
    "claim": "The effective number of independent training samples is closer to T_hist/T than to T_hist - T + 1; N_eff is about 15 independent windows per stock over 30 years and is determined by overlap, not stride.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:1076",
    "assessment": "Solid and unusually honest. This is the binding statistical constraint on what the unit can deliver (~5 independent windows over a 10-year training fold) and should be preserved as a design fact in any redesign.",
    "evidence": ["docs/DVT_strategie_facteurs_risque_latents_v4_1.md:1068", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:651", "src/config.py:128-140"],
    "cross_units": ["U07", "U12"],
    "root_cause": null,
    "confidence": "high"
  },
  {
    "id": "U04-ADV-05",
    "unit": "U04",
    "role": "advocate",
    "type": "implementation",
    "severity": "medium",
    "evidence_register": "doc+code",
    "claim": "Inference stride is fixed at 1 (about 7057 windows per stock over 30 years), although the document states the result is quasi-identical for any stride <= 21 under mean aggregation.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:648",
    "assessment": "Fragile by configuration, not by design: stride 1 costs 21x compute and roughly 32 GB of windows that the code must immediately delete, for no statistical gain under the default uniform mean. Keep stride 1 only as an opt-in diagnostic trajectory.",
    "evidence": ["src/integration/pipeline.py:1976-1980", "src/integration/pipeline.py:1998-2001", "src/inference/composite.py:155-165", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:651"],
    "cross_units": ["U08", "U15"],
    "root_cause": "Stride-1 inference is retained for optional dense-trajectory diagnostics but is the unconditional default even when aggregation is a plain mean.",
    "confidence": "high"
  },
  {
    "id": "U04-ADV-06",
    "unit": "U04",
    "role": "advocate",
    "type": "doc-vs-code",
    "severity": "medium",
    "evidence_register": "doc+code",
    "claim": "The spec defines the second feature as the raw rolling 21-day standard deviation of returns, z-scored per window, while the code applies a natural log before z-scoring by default.",
    "claim_location": "src/data_pipeline/windowing.py:110",
    "assessment": "Divergence: neither DVT nor ISD mentions the log transform. The transform is literature-defensible (log realized variance), but it is untested in the repo and its stated rationale (freeing latent capacity) is an unsupported hypothesis, so it cannot be counted as intended design.",
    "evidence": ["src/data_pipeline/windowing.py:110", "src/data_pipeline/windowing.py:132-134", "src/data_pipeline/windowing.py:248-249", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:246-248", "docs/ISD_vae_latent_risk_factors.md:808"],
    "cross_units": ["U17"],
    "root_cause": "An implementation-time transformation was added without propagating it to the specification or the test suite.",
    "confidence": "high"
  },
  {
    "id": "U04-ADV-07",
    "unit": "U04",
    "role": "advocate",
    "type": "math",
    "severity": "medium",
    "evidence_register": "code",
    "claim": "Volatility is floored at 1e-8 before the log transform, so windows containing 21 or more constant returns map to log(1e-8) = -18.4 and become extreme standardized outliers.",
    "claim_location": "src/data_pipeline/windowing.py:249",
    "assessment": "Fragile. A 20% zero-return tolerance admits long constant stretches, the rolling std is exactly zero over 21 constant returns, and the pooled z-score validation tolerance of 0.1 cannot detect a few extreme values among millions. Testable prediction: on Tiingo windows, the |z|>10 rate in the vol channel exceeds the return channel and concentrates in windows with long zero-return runs.",
    "evidence": ["src/data_pipeline/windowing.py:227-232", "src/data_pipeline/windowing.py:249", "src/data_pipeline/windowing.py:257-259", "src/data_pipeline/features.py:76-78", "src/validation.py:1085-1088", "tests/unit/test_data_quality.py:1222-1235"],
    "cross_units": ["U03"],
    "root_cause": "No zero-vol filter or per-window outlier check exists on the volatility channel before the log transform.",
    "confidence": "high"
  },
  {
    "id": "U04-ADV-08",
    "unit": "U04",
    "role": "advocate",
    "type": "doc-vs-code",
    "severity": "high",
    "evidence_register": "doc+code",
    "claim": "The spec excludes a window when a price gap exceeds 5 consecutive days, but the code excludes only when more than 5% of window days are NaN, then zero-imputes the missing returns.",
    "claim_location": "src/data_pipeline/windowing.py:215",
    "assessment": "Real divergence: a 6-25 day gap passes the aggregate filter (5% of 504 is 25 days), is converted to zero returns, and fabricates a flat segment that biases window mean and std downward. The consecutive-gap rule can be enforced at zero cost because the universe already requires 504 continuous listing days.",
    "evidence": ["src/data_pipeline/windowing.py:215", "src/data_pipeline/windowing.py:227", "src/data_pipeline/returns.py:33", "src/data_pipeline/returns.py:43-44", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:258", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:273", "docs/ISD_vae_latent_risk_factors.md:734"],
    "cross_units": ["U03", "U17"],
    "root_cause": "A fraction-based missing-data filter replaced the spec's run-length rule, and nan_to_num silently converts the tolerated gaps into zero returns.",
    "confidence": "high"
  },
  {
    "id": "U04-ADV-09",
    "unit": "U04",
    "role": "advocate",
    "type": "justification",
    "severity": "medium",
    "evidence_register": "doc+code",
    "claim": "The document argues that per-window z-scoring erases volatility dynamics and that the 21-day realized-vol channel restores them, although that channel is itself z-scored per window.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:250",
    "assessment": "Wrong argument, survivable design: with window-constant mu and sigma, a mid-window volatility doubling stays visible as an amplitude change in the z-scored returns, and the vol channel cannot restore the absolute level because it is standardized identically. Incremental content of channel 2 is 20 pre-window days (~4%), a 21-day smoothing prior, and log compression.",
    "evidence": ["docs/DVT_strategie_facteurs_risque_latents_v4_1.md:240", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:246-248", "src/data_pipeline/windowing.py:252-259", "src/data_pipeline/features.py:76-78"],
    "cross_units": ["U05"],
    "root_cause": "The complementarity argument conflates cross-window level with within-window dynamics and ignores that the auxiliary channel receives the same normalization.",
    "confidence": "high"
  },
  {
    "id": "U04-ADV-10",
    "unit": "U04",
    "role": "advocate",
    "type": "modeling",
    "severity": "high",
    "evidence_register": "doc+code",
    "claim": "Raw per-stock series are used so the model can discover which statistics matter, but windows are built one stock at a time and the training objective contains almost no cross-sectional coupling.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:222",
    "assessment": "The true bottleneck: the representation carries no cross-sectional information. The only cross-stock terms are the co-movement loss capped at 0.1 and a cross-sectional R-squared loss that is disabled (lambda_cs=0.0). Factor identity is emergent from weight sharing, never optimized; changing T, F or the normalizer cannot fix this.",
    "evidence": ["docs/DVT_strategie_facteurs_risque_latents_v4_1.md:220", "src/data_pipeline/windowing.py:189-232", "src/vae/loss.py:9", "src/vae/loss.py:201-232", "notebooks/dashboard.ipynb:250-251"],
    "cross_units": ["U05", "U06", "U08"],
    "root_cause": "The VAE is trained as a per-stock reconstruction model; no term in the loss rewards latent dimensions for explaining cross-sectional return variation.",
    "confidence": "high"
  },
  {
    "id": "U04-ADV-11",
    "unit": "U04",
    "role": "advocate",
    "type": "statistics",
    "severity": "medium",
    "evidence_register": "code",
    "claim": "Validation windows are selected by end_date >= val_start, which prevents informational leakage but not statistical dependence across the boundary.",
    "claim_location": "src/integration/pipeline.py:1285",
    "assessment": "Fragile: the last ~24 training windows (504/21) share at least 95% of their days with the first validation window, so early stopping and Phase-A hyperparameter selection see quasi-duplicates across the split. The windowing design guarantees no informational leakage, not independent validation.",
    "evidence": ["src/integration/pipeline.py:1284-1289", "src/data_pipeline/windowing.py:209", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:876"],
    "cross_units": ["U12", "U07"],
    "root_cause": "Date-based splits treat overlapping windows as independent samples.",
    "confidence": "high"
  },
  {
    "id": "U04-ADV-12",
    "unit": "U04",
    "role": "advocate",
    "type": "data",
    "severity": "low",
    "evidence_register": "code",
    "claim": "Each stock's earliest window contains about 20 leading NaN volatility values (3.97% of the window), passes the 5% NaN filter, and has its volatility back-filled with a constant.",
    "claim_location": "src/data_pipeline/windowing.py:228",
    "assessment": "Systematic but small fabrication: exactly one window per stock is affected. It demonstrates that an aggregate NaN fraction is the wrong instrument; a max-consecutive-gap or first-valid rule removes it at no cost.",
    "evidence": ["src/data_pipeline/windowing.py:215", "src/data_pipeline/windowing.py:228-229", "src/data_pipeline/features.py:76-78", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:252"],
    "cross_units": ["U03"],
    "root_cause": "Warm-up NaNs are treated as ordinary missing data by the aggregate fraction filter.",
    "confidence": "high"
  },
  {
    "id": "U04-ADV-13",
    "unit": "U04",
    "role": "advocate",
    "type": "justification",
    "severity": "medium",
    "evidence_register": "doc",
    "claim": "T = 504 days is justified as long enough for several market conditions and short enough for a coherent risk regime, while the same document states that T > 504 increases the probability of spanning a structural break.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:301",
    "assessment": "Conditional: both criteria are asserted rather than measured, and 504 sits exactly at the document's own upper bound with zero headroom. No ablation of 252/504/756 against explanatory power exists; the value is a reasonable prior, not a derived optimum.",
    "evidence": ["docs/DVT_strategie_facteurs_risque_latents_v4_1.md:295-303", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:1074"],
    "cross_units": ["U15"],
    "root_cause": null,
    "confidence": "medium"
  },
  {
    "id": "U04-ADV-14",
    "unit": "U04",
    "role": "advocate",
    "type": "implementation",
    "severity": "low",
    "evidence_register": "code",
    "claim": "The window-counting pass duplicates the filtering logic of the filling pass, and the filling pass writes into pre-allocated np.empty arrays.",
    "claim_location": "src/data_pipeline/windowing.py:60",
    "assessment": "Fragile maintenance hazard: if the two filter implementations diverge, an over-count leaves uninitialized rows in the returned tensor. The only guard is assert_no_nan_tensor, which catches NaN but not finite garbage. The two implementations are currently identical, so this is not an active bug.",
    "evidence": ["src/data_pipeline/windowing.py:76-99", "src/data_pipeline/windowing.py:178-185", "src/data_pipeline/windowing.py:290", "src/data_pipeline/windowing.py:302"],
    "cross_units": ["U07"],
    "root_cause": "A 16 GB memory optimization split one logical filter into two code paths.",
    "confidence": "high"
  },
  {
    "id": "U04-ADV-15",
    "unit": "U04",
    "role": "advocate",
    "type": "modeling",
    "severity": "medium",
    "evidence_register": "doc+code",
    "claim": "T = 504 is justified as short enough for a coherent risk regime, but the default composite profile is a simple mean over all windows with half-life 0, preserving all historical regimes.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:656",
    "assessment": "Conditional: the window is regime-coherent by construction, but the profile is a 10-30 year uniform average, so regime specificity is discarded at aggregation rather than at windowing. The trade-off is deliberate and the half-life option exists, but the default undermines the T choice rationale.",
    "evidence": ["docs/DVT_strategie_facteurs_risque_latents_v4_1.md:656", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:301", "src/config.py:366", "src/inference/composite.py:130-165"],
    "cross_units": ["U08"],
    "root_cause": "Window-level regime control and profile-level regime averaging were decided independently.",
    "confidence": "medium"
  }
]
```

END-OF-REPORT
