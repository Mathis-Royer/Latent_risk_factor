# U05 — critic

## Scope read
- `src/vae/build_vae.py` (full, 1–381): sizing rules, capacity constraint, parameter counts, factory.
- `src/vae/encoder.py` (full): Inception head, residual body, global average pooling, projection.
- `src/vae/decoder.py` (full): transposed residual body, trim/pad, output head.
- `src/vae/model.py` (full): reparameterization, `forward`, `encode`, scalar σ².
- `src/vae/loss.py`: 201–278 (cross-sectional loss), 370–394 (loss assembly, co-movement scaling), 620–645 (λ_co curriculum).
- `src/inference/active_units.py` (full): AU measurement, `compute_au_max_stat`, truncation, Bai-Ng post-filter.
- `src/inference/composite.py`: 119–172 (aggregation).
- `src/integration/pipeline.py`: 230–349 (auto-adaptation of K and r), 1900–2120 (inference windows, AU, truncation, z-scoring).
- `src/integration/diagnostics.py`: 99–136 (thresholds), 1496–1507 (AU health check), 1578–1591 (EP health check), 690–800 (EP computation).
- `src/walk_forward/metrics.py`: 491–546 (OOS explanatory power).
- `src/data_pipeline/windowing.py`: 1–18, 252–273 (per-window z-scoring of both features).
- `src/training/batching.py` (full): synchronous/stratified vs random phase-3 batching; `compute_strata`.
- `src/config.py`: 160–204 (`VAEArchitectureConfig`), 242 (`lambda_cs`), 349–375 (inference config).
- `tests/unit/test_vae_architecture.py`: 50–95, 142–178, 411–460, 754–900 (what is and is not tested).
- `docs/DVT_strategie_facteurs_risque_latents_v4_1.md`: 148–200 (4.1), 202–292 (4.2), 293–437 (4.3), 438–498 (4.4 baseline and co-movement), 644–656 (aggregation), 712–730 (rescaling), 998–1002, 1200–1240 (Appendix A).
- `docs/strategy_philosophy.md`: 44–56 (§4), 60–83 (§5–§6).
- `docs/ISD_vae_latent_risk_factors.md`: 910–1080 (MOD-002).
- `results/diagnostic/`: `diagnostic_report.md`, `diagnostic_data.json`, `health_checks.csv`, `config.json`, checkpoint filenames.
- `notebooks/dashboard.ipynb`: K and diagnostic overrides (grep); `src/integration/notebook_helpers.py`: 157–158.

---

## Five questions

**1. Purpose and original intent of this unit (cite the spec).**
The unit is meant to learn, without prior factor specification, a set of latent risk factors from 504-day windows of per-stock returns and realized volatility, and to output each stock's exposure vector β_i; the decoder is a training signal and is discarded, only the encoder is used at inference (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:160-174`). K is explicitly a *capacity ceiling*, not a factor count; the KL is supposed to deactivate unused dimensions, and "dimensions with marginal KL > 0.01 nats are active units (AU)" (`DVT:402-406`). The intent is stated even more strongly in the philosophy: "the model decides how many risk factors exist" and "the strategy does not require a human decision about how many risk factors exist in the market" (`docs/strategy_philosophy.md:73-82`). MOD-002 fixes the implementation contract: InceptionHead + residual body, transposed residual decoder, `encode()` returns μ deterministically (`docs/ISD_vae_latent_risk_factors.md:910-917, 1035`).

**2. What a state-of-the-art version of this unit would look like for this objective (cite literature if you can).**
For *this* objective — producing a factor risk model that spans the cross-section of returns — the state of the art is not a sequence autoencoder. It is a factor model whose decoder is literally the factor model, estimated on the cross-section:
- **Structured autoencoder / conditional factor model:** Gu, Kelly & Xiu (2021, *J. Econometrics*, "Autoencoder Asset Pricing Models") and Lettau & Pelger (2020, *RFS*, IPCA) train networks whose output is the loading vector and whose reconstruction is r_{i,t} = β_i(x_{i,t})' f_t + ε. The reconstruction target is the return cross-section, not the stock's own time series.
- **Genuine rank selection:** Bai & Ng (2002) information criteria, Onatski (2009) eigenvalue tests, or sparse Bayesian factor analysis with automatic relevance determination (Bishop 1999; Knowles & Ghahramani 2011) / the Indian buffet process (Griffiths & Ghahramani 2011). These infer the number of factors from the data; a fixed N(0,I) KL prior does not.
- **Time-varying loadings:** dynamic factor models / Kalman smoothing (e.g. Doz, Giannone & Reichlin 2011) or conditional loadings; a static B per six-month fold is a deliberate simplification, not a state-of-the-art exposure model.
- **Validation against the task:** covariance-forecast losses (QLIKE, variance ratio, cross-sectional R² out of sample), not per-window MSE. Reconstruction MSE is at best a proxy and is never the quantity the risk model is judged on.
- If a *sequence* encoder is retained, TempVAE (Sicks et al. 2023) is a reasonable reference, but it must still be trained against a cross-sectional/covariance objective and ablated against linear PCA.

**3. What prevents the current unit from reaching it (evidenced, not assumed).**
- **No cross-sectional term in the generative model.** The model reconstructs each (stock, window) independently; z is indexed by (i, window), not by t. The claimed generative equation has a shared factor path z_t (`DVT:154-158`), but the code has none (`src/vae/model.py:113-150`). The only cross-sectional coupling is a curriculum auxiliary that is switched off for the final 40% of epochs (`src/training/batching.py:4-6`; λ_co = 0 in phase 3, `src/vae/loss.py:620-645`), and the purpose-built cross-sectional R² loss is disabled (`src/config.py:242`, `lambda_cs = 0.0`).
- **A univariate encoder cannot identify covariance.** B_i = f(x_i) where f is stock-agnostic and x_i is stock i's own window. Covariance is a property of the joint law; two stocks with identical marginal window distributions but different dependence structures receive identical B. The doc's own citation of Locatello et al. concedes that the implicit mechanism is not guaranteed (`DVT:496`).
- **The pruning mechanism is inert.** With D = T·F = 1008 and the ELBO D/(2σ²)·MSE + KL, at the learned σ² = MSE a dimension survives unless it explains less than ~2/D ≈ 0.2% of residual variance. The measured spectrum confirms it: 47 of 52 dimensions are above 0.01 nats (`results/diagnostic/diagnostic_data.json:38-90`), and the inactive five carry 0.023 nats in total.
- **The capacity guard is measured against the wrong denominator.** N = n·(T_hist − T + 1) counts windows that overlap by 96%; the same document states N_eff ≈ T_hist/T ≈ 15 independent windows per stock (`DVT:651`). With 739 stocks the effective sample is ~11k, giving r ≈ 10³, not 2.7 (`src/vae/build_vae.py:312-347`).
- **The only direct test fails.** EP_OOS = 0.356, EP_IS = 0.219 against the spec target > 0.60 and the monitoring floor max(0.50, EP_PCA + 0.10) (`DVT:414,475`; `results/diagnostic/diagnostic_data.json:155-156`).
- **The factor count is not discovered.** K = 52 = 2×AU_max_stat and AU is truncated to AU_max_stat = 26 (`src/integration/pipeline.py:307-314, 2017-2029`), so 50% utilization is an identity, not a measurement.

*Steelman then rebuttal.* The strongest case for the design: a VAE gives a smooth, continuous latent with a prior; it is unsupervised and generalizes across stocks; the KL is a principled regularizer; the architecture is a standard, competently implemented InceptionTime-style encoder; the decoder is only a training aid, so its cost is acceptable; the downstream statistical caps are prudent safeguards; and the diagnostic honestly flags the variance ratio as critical. Rebuttal: the smoothness and the posterior variance are never used downstream (`model.py:152-167`); the pruning is mathematically near-inert (above); the nonlinearity is never ablated against a linear VAE or PCA (the doc proposes exactly that reference at `DVT:472`, and the diagnostic says "Training diagnostics not available: pretrained model (no training)"); "many windows" are 96% redundant; the decoder dominates the parameter budget; the statistical caps make the AU metric circular and unfalsifiable; and the one metric that tests the unit (EP) is re-thresholded by a factor of 12 so that a failure is printed as "[OK]" (`src/integration/diagnostics.py:108,1578-1591`).

**4. Which constraints are real (data, math, compute, market) and which are self-imposed design choices?**
- *Real:* daily prices only (no fundamentals in use); ~800 liquid US stocks; ~15 independent 504-day windows per stock; AU must be well below n for the cross-sectional regression; covariance estimation is noisy and non-stationary; a 16 GB machine.
- *Self-imposed:* the VAE formalism itself; KL to a fixed N(0,I) prior; MSE reconstruction of standardized windows; per-window z-scoring of *both* channels; Inception head with kernels (5,21,63); five residual blocks; the C_min = 384 floor; K = 200 in the docs; the 0.01-nat AU threshold; `AU_max_stat` with r_min = 2 and an EWMA half-life of 252; the double truncation (statistical cap then Bai-Ng); the static B per fold; mean/EWMA aggregation; the decision to disable the cross-sectional loss; the decision to zero the co-movement term in the final 40% of training.
- *Mixed:* the 504-day window is a real modelling choice with a real trade-off, but its stated benefit ("several market conditions", `DVT:301`) is contradicted by the runtime aggregation, which reduces the profile to a ~60-day snapshot (finding U05-MAJ-12).

**5. What should be removed, merged or moved (not added)?**
- **Remove** the variational machinery (posterior variance, reparameterization) from the *product*: it is never used at inference (`model.py:152-167`). Retain the VAE only if an ablation shows it beats a deterministic autoencoder on OOS EP; otherwise delete it and the associated loss modes.
- **Remove** the AU threshold (0.01 nats) and the double truncation (AU_max_stat then Bai-Ng). **Move** the factor-count decision to the risk model as one explicit rank-selection step with an OOS stability criterion. A factor count should be a property of Σ_z, not of an unidentified encoder basis.
- **Remove** the deep transposed decoder body. **Merge** the decoder with the factor model: decode z as B z + ε. Then the trained object is the factor model and the capacity constraint counts something meaningful.
- **Remove** the capacity-ratio guard or redefine it on effective independent windows and on P_enc only. At present it is a hard constraint with remediation (`build_vae.py:329-347`) computed against an inflated denominator.
- **Remove** the per-window z-scoring of the realized-volatility channel, or delete the claim that the second feature restores absolute scale (`DVT:240-250` vs `DVT:724`). The encoder is scale-free by construction.
- **Merge** training and inference objectives: enable and fix the cross-sectional loss (same-date cross-sections, applied to the *aggregated* profile, not to raw batch μ), and keep the co-movement term through the final phase or remove it as a band-aid.
- **Move** the "the model decides how many factors exist" claim out of the philosophy document; it is false as implemented and contradicts the code's own caps.
- **Remove** K = 200 from the docs and tests, or make it the runtime value; reconcile config (50), notebook (75), docs (200) and runtime (52).

---

## Load-bearing assumptions

| ID | Assumption | Where stated | Evidenced? | Comment |
|----|------------|--------------|------------|---------|
| A1 | The per-window latent z plays the role of the shared factor path z_t in r_{i,t} = f(z_t, β_i) + ε | `DVT:154-158,174` | **no** | Implemented model is p(x_{i,w}) per window; no z_t, no cross-stock term (`model.py:113-150`). |
| A2 | KL auto-pruning makes AU the effective number of factors | `DVT:402-406`; `philosophy:78-80` | **no** | 47/52 dims active; 5 dead dims carry 0.03% of KL; no elbow (`diagnostic_data.json:38-90`). |
| A3 | Reconstructing standardized windows is a sufficient proxy for spanning the return cross-section | `DVT:158,450` | **no** | EP_OOS = 0.356, EP_IS = 0.219 (`diagnostic_data.json:155-156`). |
| A4 | N = n·(T_hist − T + 1) is the correct capacity denominator | `DVT:364-368`; `build_vae.py:312-347` | **no** | Overlap 96%; doc itself says N_eff ≈ 15 per stock (`DVT:651`); effective r ~10³. |
| A5 | Latent dimensions are identifiable and per-dimension KL selection is meaningful | `DVT:404,436`; `pipeline.py:2029` | **no** | Marginal KL is basis-dependent; latent basis is unidentified; downstream rotates B (`DVT:195,712-718`). |
| A6 | The realized-vol feature restores absolute volatility scale | `DVT:240-250` | **no** | Both features z-scored per window (`windowing.py:252-273`); `DVT:724` admits "pure shape". |
| A7 | T = 504 carries several regimes and the encoder learns regime structure | `DVT:301` | **partially** | The window is long enough, but global pooling is regime-blind and the profile is a ~60-day EWMA (finding MAJ-12). |
| A8 | The nonlinear CNN encoder adds value over a linear factor model | `DVT:162-168,307-317` | **no** | No linear VAE/PCA ablation was run; the proposed linear reference (`DVT:472`) has no result. |
| A9 | K is a generous ceiling and capacity is not binding | `DVT:406,420` | **no** | K is capped at 52 = 2·AU_max_stat; doc says 200 (`pipeline.py:314`; `config.py:179`; notebook:247). |
| A10 | Exposures can be held static across a six-month OOS fold | `DVT:896` | **partially** | Code recomputes B once per fold (`pipeline.py:1992-1996,2074`); this is a choice, not a result. |
| A11 | Latent distances between stocks are meaningful and required by portfolio construction | `DVT:164` | **no** | Portfolio construction never uses latent distances, only B and Σ (`pipeline.py:1992-2096`). |
| A12 | Learned σ² yields a data-dependent factor count | `DVT:458-464` | **no** | The D-scaling makes the marginal KL cost negligible (finding MAJ-07). |

---

## Findings

| ID | Type | Severity | Evidence register | Claim (quoted, path:line) | Assessment | Evidence (path:line) | Cross-unit |
|----|------|----------|-------------------|---------------------------|------------|----------------------|------------|
| U05-CRI-01 | modeling | critical | doc+code | "The model simultaneously estimates the K latent factors, each stock's exposures, and the dependence structure induced by these common exposures" (`DVT:158`); equation r_{i,t} = f(z_{1,t},…,z_{K,t},β_{i,1},…) + ε (`DVT:156`) | The implemented generative model is not a factor model. z is indexed by (stock, window); the decoder reconstructs each stock's own window from its own z; no shared factor path and no cross-stock term exist. Exposures are read off post hoc as encoder means. The doc's equation is not the code's model. | `src/vae/model.py:113-150`; `src/vae/encoder.py:174-195`; `src/vae/decoder.py:144-170`; `src/integration/pipeline.py:1992-1996` | U06, U08, U09 |
| U05-CRI-02 | modeling | critical | doc+code | "Once trained, the encoder maps return windows to latent vectors — these are the estimated exposures β_i" (`DVT:174`) | A stock-agnostic function of a stock's own window cannot identify covariance with other stocks: B_i = f(x_i) is a function of the marginal law of x_i, while covariance is a property of the joint law. The only cross-sectional term (co-movement) is switched off for the final 40% of epochs and the cross-sectional R² loss is disabled by default. The architecture therefore cannot, even in principle, learn the object the risk model needs. | `src/training/batching.py:4-6,106-111`; `src/vae/loss.py:384-394,620-645`; `src/config.py:242`; `src/vae/encoder.py:167`; `src/data_pipeline/windowing.py:252-273` | U06, U04 |
| U05-CRI-03 | statistics | critical | code+results | "A VAE with K = 200 on data supporting 80 factors converges to AU ≈ 80 with ~120 inactive dimensions. Unlike PCA … superfluous VAE dimensions deactivate themselves" (`DVT:406`) | No auto-pruning occurs. Of 52 dimensions, 47 exceed the 0.01-nat threshold; the five inactive dimensions carry 0.023 nats (0.03% of the 81.78 total). The sorted spectrum decays smoothly (2.669, 2.631, 2.553, … 2.057, 1.969, … 0.002) with no elbow; the 26/27 boundary (2.057 vs 1.969) is a 4% step. The report's "effective latent dims 40.1" also exceeds the reported AU of 26. The criterion does not select a factor count; a constant does. | `results/diagnostic/diagnostic_data.json:38-90,92-95`; `results/diagnostic/diagnostic_report.md:39-44` | U14, U09 |
| U05-CRI-04 | logic | critical | code+results | "AU = 26 / K = 52 (50.0% utilization)" (`results/diagnostic/diagnostic_report.md:13`) | Circular by construction. K = min(K_config, 2·AU_max_stat) = 52 and AU is truncated to AU_max_stat = 26, so AU/K ≤ 0.5 always in the configured regime; the doc's capacity-exhaustion rule "AU > 0.85·K → increase K" (`DVT:474`) is unreachable. Worse, with EWMA half-life 252, `compute_au_max_stat` returns 26 even for 30 years of history, not the documented 85, so the documented "typical AU 60-120" range cannot occur. The 50% is a parameter identity, not a measurement of the data. | `src/integration/pipeline.py:307-314,2017-2029`; `src/inference/active_units.py:92-117`; `results/diagnostic/checkpoints/checkpoint_fold00_train_N739_Y24_T504_K52_F2.pt`; `DVT:426,474` | U14, U15 |
| U05-MAJ-05 | math | major | doc+code | "Dimensions with marginal KL > 0.01 nats are active units (AU)" (`DVT:404`); "truncate to the AU_max_stat most active dimensions (ranked by descending marginal KL)" (`DVT:436`) | Marginal KL per dimension is not rotation-invariant, and the VAE latent basis is unidentified (KL to a factorized prior plus decoder symmetry). Selecting dimensions by KL therefore selects an arbitrary basis; the pipeline then diagonalizes Σ_z and rotates B anyway (`DVT:195,712-718`). A factor *count* must be rotation-invariant (the rank of Σ_z); this one is not. | `src/inference/active_units.py:70-89`; `src/integration/pipeline.py:2029`; `DVT:712-718` | U09 |
| U05-MAJ-06 | statistics | major | doc+code | "N = n × (T_hist − T + 1) is the maximum number of windows (stride s = 1) … stride subsamples redundant windows … without reducing the dataset's informational content" (`DVT:368`) | The capacity constraint's denominator counts windows that overlap by 96% (stride 21: 483 of 504 days shared). The same document states N_eff ≈ T_hist/T ≈ 15 independent windows per stock (`DVT:651`). With 739 stocks, N_eff ≈ 11k and r ≈ 11M/11k ≈ 10³, not the reported 2.7. The "hard constraint" with remediation is therefore off by two to three orders of magnitude, and `DVT:368` itself concedes that overfitting protection does not come from window count. | `src/vae/build_vae.py:312-347`; `DVT:651`; `DVT:368` | U07, U15 |
| U05-MAJ-07 | math | major | doc+code | "σ² can be learned as a scalar parameter … A latent dimension remains active only if the reconstruction gain … exceeds its KL cost — which, for D = 504 and σ² ≈ 0.5, requires explaining at least ~0.1% of signal variance per dimension" (`DVT:464`) | The mechanism is near-inert by scaling. With D = T·F (defined at `DVT:448`) = 1008, and at the learned optimum σ² = MSE, the marginal threshold is ΔMSE/MSE > 2/D ≈ 0.2%: a dimension is dropped only if it explains almost nothing. The note also uses D = 504, inconsistent with F = 2 (D = 1008); AU consequently depends on F, not only on the data. This is a description of a threshold, not a discovery mechanism. | `DVT:448,460-466`; `src/vae/model.py:72-88`; `src/vae/loss.py:370-394` | U06 |
| U05-MAJ-08 | justification | major | doc+code | "This forces the latent space to be continuous and smooth … Distances between stocks are therefore meaningful and comparable, which is exactly what portfolio construction requires" (`DVT:164`) | Portfolio construction uses neither latent distances nor posterior variance. `encode()` returns μ only; the variance is discarded (`DVT:170`); B enters an OLS and a quadratic form. The VAE's defining output is unused, so the stated justification does not describe the product. The doc also cites Locatello et al. (`DVT:496`) that unsupervised disentanglement is impossible without inductive bias, which contradicts the "discovered factors" interpretation. | `src/vae/model.py:152-167`; `src/integration/pipeline.py:1992-2096`; `DVT:496` | U08, U10 |
| U05-MAJ-09 | interpretation | major | doc+code+results | "Factor explanatory power … Target > 0.60" (`DVT:414`); "should exceed max(0.50, EP_PCA + 0.10)" (`DVT:475`) | The measured EP_OOS = 0.356 and EP_IS = 0.219 are below both the target and the floor; EP_PCA is never computed. The diagnostic prints "[OK] Explanatory power" because the warning threshold is 0.05, twelve times below the spec floor. This is goal-post moving on the one metric that directly tests whether the VAE's factors span the cross-section. | `results/diagnostic/diagnostic_data.json:155-156`; `src/integration/diagnostics.py:105-110,1578-1591`; `results/diagnostic/diagnostic_report.md:16` | U14, U09 |
| U05-MAJ-10 | modeling | major | results | "AU = 26" and "n_signal = 12" (`results/diagnostic/diagnostic_data.json:8,116`) | The 26-dimension factor model is effectively one-dimensional: eigenvalue ratio #1/#2 = 10.08, top-3 variance = 72.9%, top-1 factor risk contribution = 81.1%, effective eigenvalue count = 5.78. Entropy parity across 26 dimensions is therefore mostly parity across noise, which mechanically links the architecture to the optimizer's failure to diversify risk. | `results/diagnostic/diagnostic_data.json:116,150,295`; `results/diagnostic/diagnostic_report.md:74-78,92-95` | U09, U10 |
| U05-MAJ-11 | consistency | major | doc+code+results | "Baseline: K = 200" (`DVT:420`) | K is 50 in `src/config.py`, 75 in the notebook, 200 in the docs, and 52 at runtime. The feasibility table (`DVT:376-382`), the sizing example (`DVT:356`) and the unit tests exercise a model that never runs. The documented AU range 60-120 is incompatible with the actual K = 52. The architecture is not the documented architecture. | `src/config.py:179`; `notebooks/dashboard.ipynb:247`; `DVT:376-382,420`; `results/diagnostic/checkpoints/checkpoint_fold00_train_N739_Y24_T504_K52_F2.pt` | U15, U17 |
| U05-MAJ-12 | consistency | major | doc+code+results | "Default aggregation: simple mean (all windows contribute equally, preserving memory of all historical regimes)" (`DVT:656`); "the composite profile is a mean over all windows" (`docs/strategy_philosophy.md:56`) | Runtime uses an EWMA with half-life 60 *windows* at inference stride 1, i.e. ~60 trading days, not the "~5 years" claimed in the code comment (`composite.py:135-138`). B is a ~three-month snapshot of overlapping two-year windows: the VAE's T = 504 multi-regime design is averaged into essentially one window, and the "memory of dormant risks" (`philosophy:54`) is absent. | `src/integration/pipeline.py:1976-1996`; `src/integration/notebook_helpers.py:157-158`; `src/inference/composite.py:127-165` | U08, U01, U09 |
| U05-MAJ-13 | data | major | doc+code | "a stock with identical return shapes but transitioning from 10% to 40% annualized volatility carries fundamentally different risk, invisible to the return channel alone" (`DVT:250`) | Both features are z-scored per window (`windowing.py:252-273`), so the absolute volatility level is erased; only the within-window shape of the vol series survives. `DVT:724` itself admits "pure shape, no absolute scale". The encoder is a scale-free shape descriptor; scale is injected only downstream by σ_{i,t}/σ̄_t (`DVT:730`), so the "latent risk factors" are shape clusters, not risk magnitudes. This also explains the dominant single scale factor. | `src/data_pipeline/windowing.py:252-273`; `DVT:240-250,724,730` | U04, U09 |
| U05-MAJ-14 | literature | major | doc | "Dimensions with marginal KL > 0.01 nats are active units (AU). This is confirmed by TempVAE on financial time series (Sicks et al., 2023)" (`DVT:404`); "the VAE needs K ≥ 1.5–2× the intrinsic dimensionality to auto-prune reliably (Burda et al. 2015, Sicks et al. 2023)" (`DVT:420`) | AU as implemented is the *mean* marginal KL, whereas Burda et al.'s active-unit criterion is a covariance between input and KL; the two are not equivalent, and the 0.01 constant is borrowed without calibration. No derivation is given for the 1.5-2× sizing rule, and one paper on a different architecture cannot "confirm" this mechanism. Literal appeals to literature are doing the work of an ablation. | `src/inference/active_units.py:29-31,77-80`; `DVT:404,420` | U06, U17 |
| U05-MIN-15 | implementation | minor | code | "C_L(K) = round_16(max(C_min, ceil(α_proj × 2K)))" (`DVT:348`); "Dominant scaling: O(K² × L)" (`DVT:362`) | C_min = 384 binds for all K ≤ 147, so at the operating K = 52 the final width, the channel progression and most of P_enc are independent of K; the claimed quadratic capacity scaling does not hold in the regime actually used. Meanwhile the decoder's fc term (K·C_L·T_comp = 325,632 at K = 52) is the largest single tensor and is discarded at inference. | `src/vae/build_vae.py:58-70,199-227`; `src/vae/decoder.py:128`; `DVT:348-362` | U15 |
| U05-MIN-16 | math | minor | doc+code | "AU_max_stat = floor((-1 + sqrt(1 + 4·N_obs/r_min))/2)" (`DVT:424`) | The printed formula gives 59 for N_obs = 7309, r_min = 2, not the tabulated 85; the correct expression is (-1 + sqrt(1 + 8·N/r_min))/2. The code uses floor(sqrt(2·N_eff/r_min)) = 85, matching the table but not the spec. The "binding constraint" is therefore asserted from a misstated formula, and the table cannot be reproduced from it. | `DVT:424-436`; `src/inference/active_units.py:100-117` | U15 |
| U05-MIN-17 | implementation | minor | code | `measure_active_units` computes AU from mean KL; the pipeline "Derive AU from pre-computed KL" inline (`pipeline.py:2010`) | The tested function is not the production path: the pipeline reimplements the threshold, sorting and truncation inline and measures KL in-sample on 96%-overlapping inference windows. The unit test therefore does not exercise the AU that is actually used downstream. | `src/inference/active_units.py:19-89`; `src/integration/pipeline.py:2010-2029` | U14 |
| U05-MIN-18 | logic | minor | doc+code | "Translation invariance: the same kernel recognizes a volatility cluster whether it occurs at day 10 or day 200 — the risk profile should reflect what patterns occur, not when" (`DVT:311`) | For a covariance model, *when* is the information: a volatility cluster ending today and one ending two years ago imply different current beta. Global average pooling (`encoder.py:167`) enforces exactly the invariance the document praises, and two nested averages (pooling, then profile aggregation) discard the recency the risk model needs. | `src/vae/encoder.py:167`; `src/vae/decoder.py:144-170`; `DVT:311` | U04 |

---

## Root-cause hypotheses

**RC1 — The unit solves the wrong generative problem: a per-window univariate autoencoder where a cross-sectional factor model is required.** This single cause explains CRI-01, CRI-02, MAJ-08 and MAJ-10: there is no shared z_t; the encoder cannot represent covariance; the posterior variance and latent geometry are unused; and the resulting B is a shape descriptor whose column space captures only ~36% of OOS return variance, collapsing to one dominant direction. Symptoms across units: the co-movement scaffolding (U06), the "factors are uninterpretable" problem (U09), and the optimizer's one-factor risk concentration (U10).
*Structural alternatives:* (a) a structured autoencoder whose decoder is B z + ε, trained on the return cross-section (Gu-Kelly-Xiu / IPCA style) — directly optimizes the right object, but needs careful conditioning and loses the simple sequence-decoder training signal; (b) a variational *factor* model (r_t ~ N(B z_t, D), estimated by EM or amortized VI) — statistically principled and gives a likelihood for rank selection, but assumes linearity or a specified nonlinear loading map; (c) a two-stage design: unsupervised representation, then B fitted by constrained cross-sectional regression on returns — cheap and interpretable, but the representation is no longer end-to-end optimized for risk.

**RC2 — The factor count is produced by a basis-dependent heuristic and then overridden by downstream caps, so "the model decides" is not true.** Explains CRI-03, CRI-04, MAJ-05, MAJ-14 and MIN-16: the KL spectrum has no elbow; the 0.01 threshold is arbitrary; AU is truncated to a cap that is itself the reason K = 52; the cap is fed by an EWMA effective sample size that makes it ~26 regardless of history; and the printed formula for the cap is wrong. The whole chain creates the illusion of a discovered dimensionality.
*Structural alternatives:* (a) estimate rank on Σ_z or the residual covariance with Bai-Ng/Onatski plus a bootstrap stability criterion — rotation-invariant and testable, but requires a separate estimation step and can be unstable in short samples; (b) sparse Bayesian factor analysis with ARD or an IBP prior — genuinely infers rank, but adds inference complexity and may be slow on 800×7000 panels; (c) cross-validated rank by OOS covariance forecast — directly tied to the decision, but expensive and exposes the rank instability rather than hiding it.

**RC3 — The objective and capacity accounting are misaligned with the product.** Explains MAJ-06, MAJ-07, MIN-15 and MAJ-13: MSE reconstruction of standardized windows, scaled by D/(2σ²), makes KL nearly free and makes AU a function of F; the capacity guard counts 96%-overlapping windows and a decoder that is thrown away; the scale of the input is removed and re-injected downstream. Symptoms: weak pruning, over-parameterization relative to effective data, and a factor model driven by a scalar vol rescaling.
*Structural alternatives:* (a) train on a covariance/cross-sectional loss (fix and enable the existing L_cs on same-date cross-sections and on the aggregated B) — aligns the objective with the task, but is harder to optimize and needs a well-conditioned differentiable OLS; (b) fit B directly by cross-sectional regression after representation learning — simple and aligned, but loses the end-to-end nonlinearity; (c) use the factor-model likelihood itself as the objective — statistically clean, but requires specifying D and the factor dynamics.

**RC4 — Exposures are treated as static, path-averaged descriptors rather than time-varying loadings.** Explains MAJ-12, MIN-18 and the low IS explanatory power: B is computed once per fold, aggregated with a ~60-day EWMA, and held fixed across the OOS block; the architecture's global pooling is designed to ignore *when* patterns occur. The doc's anti-cyclical "memory" claim and the code's short memory are opposite.
*Structural alternatives:* (a) a state-space/dynamic factor model with slowly varying B_t — captures regime change, but increases estimation variance and compute; (b) conditional loadings B_{i,t} = g(characteristics_{i,t}) (IPCA style) — parsimonious and time-varying, but requires characteristics that the project explicitly does not use; (c) rolling re-estimation of B with a half-life selected by OOS covariance fit — simple, but adds a hyperparameter and a look-ahead-sensitive tuning surface.

**RC5 — The evidence regime is unfalsifiable: acceptance thresholds are set far below the spec, and the documented architecture is not the one that runs.** Explains MAJ-09, MAJ-11 and MIN-17: EP target 0.60 becomes a warning at 0.05; "capacity exhaustion" is unreachable by construction; K is 200 in docs and 52 at runtime; AU is measured by a function the pipeline does not call. This is a governance cause, not a modelling one, and it propagates to every unit that relies on `results/diagnostic`.
*Structural alternatives:* (a) pre-register hard gates tied to the spec (EP > 0.60 and > EP_PCA + 0.10) and make the run fail, not warn; (b) replace single-number health checks with spectra and rank curves so that the absence of an elbow is visible; (c) run the ablations the spec already proposes (linear VAE vs CNN VAE vs PCA; F = 1 vs 2; K sweep) and report them alongside the headline.

---

## Verdict

The unit is well implemented and internally documented, but its theory does not survive contact with its own code and results. The VAE is not the latent factor model of section 4.1: there is no shared factor path, no cross-sectional term in the generative model, and the only mechanism that could inject dependence is a curriculum auxiliary that is switched off for the last 40% of training while the purpose-built cross-sectional loss sits disabled at `lambda_cs = 0.0`. The "model decides how many factors exist" claim is empirically false — 47 of 52 dimensions survive the KL test, the KL spectrum has no elbow, and the reported AU of 26 is exactly the statistical cap that also sets K = 52, making the 50% utilization an identity and the capacity-exhaustion check unreachable. The one metric that directly tests whether the factors span returns, explanatory power, is 0.356 against a spec target of 0.60 and a floor of 0.50, yet the report prints "[OK]" because the warning threshold was set at 0.05. The resulting 26-factor model is effectively one-dimensional (top-1 risk contribution 81%), which is the correct diagnosis of a scale-free univariate shape encoder followed by a scalar volatility rescaling. There is no look-ahead in the VAE path — inference windows are restricted to `train_returns` (`pipeline.py:1976-1980`) and B is fixed at `train_end` — and the code quality is good; the problem is architectural, not implementational. This unit needs replacement, not repair: a structured factor-model objective, an explicit rotation-invariant rank decision, and a capacity accounting based on effective independent samples and the encoder alone.

Score: 2/10

---

## Machine companion

```json
[
  {
    "id": "U05-CRI-01",
    "unit": "U05",
    "role": "critic",
    "type": "modeling",
    "severity": "critical",
    "evidence_register": "doc+code",
    "claim": "The model simultaneously estimates the K latent factors, each stock's exposures, and the dependence structure induced by these common exposures.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:158",
    "assessment": "The implemented model has no shared factor path z_t: z is indexed by (stock, window) and the decoder reconstructs each stock's own window from its own z. Exposures are read off post hoc as encoder means. The documented generative equation is not the code's model.",
    "evidence": ["src/vae/model.py:113-150", "src/vae/encoder.py:174-195", "src/vae/decoder.py:144-170", "src/integration/pipeline.py:1992-1996"],
    "cross_units": ["U06", "U08", "U09"],
    "root_cause": "RC1",
    "confidence": "high"
  },
  {
    "id": "U05-CRI-02",
    "unit": "U05",
    "role": "critic",
    "type": "modeling",
    "severity": "critical",
    "evidence_register": "doc+code",
    "claim": "Once trained, the encoder maps return windows to latent vectors — these are the estimated exposures beta_i.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:174",
    "assessment": "A stock-agnostic function of a stock's own window cannot identify covariance with other stocks: B_i = f(x_i) depends on the marginal law of x_i, while covariance is a property of the joint law. The only cross-sectional term is a curriculum auxiliary switched off in phase 3, and the cross-sectional R2 loss is disabled by default.",
    "evidence": ["src/training/batching.py:4-6", "src/training/batching.py:106-111", "src/vae/loss.py:384-394", "src/config.py:242", "src/vae/encoder.py:167"],
    "cross_units": ["U06", "U04"],
    "root_cause": "RC1",
    "confidence": "high"
  },
  {
    "id": "U05-CRI-03",
    "unit": "U05",
    "role": "critic",
    "type": "statistics",
    "severity": "critical",
    "evidence_register": "code+results",
    "claim": "A VAE with K = 200 on data supporting 80 factors converges to AU ~ 80 with ~120 inactive dimensions. Unlike PCA, superfluous VAE dimensions deactivate themselves.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:406",
    "assessment": "No auto-pruning occurs: 47 of 52 dimensions exceed 0.01 nats; the five inactive dimensions carry 0.023 nats (0.03% of total KL); the sorted spectrum decays smoothly with no elbow, and the 26/27 boundary is a 4% step. The reported effective latent dims (40.1) even exceeds the reported AU (26).",
    "evidence": ["results/diagnostic/diagnostic_data.json:38-90", "results/diagnostic/diagnostic_data.json:92-95", "results/diagnostic/diagnostic_report.md:39-44"],
    "cross_units": ["U14", "U09"],
    "root_cause": "RC2",
    "confidence": "high"
  },
  {
    "id": "U05-CRI-04",
    "unit": "U05",
    "role": "critic",
    "type": "logic",
    "severity": "critical",
    "evidence_register": "code+results",
    "claim": "AU = 26 / K = 52 (50.0% utilization).",
    "claim_location": "results/diagnostic/diagnostic_report.md:13",
    "assessment": "Circular by construction: K = min(K_config, 2*AU_max_stat) = 52 and AU is truncated to AU_max_stat = 26, so AU/K <= 0.5 always in the configured regime. The doc's AU > 0.85K capacity-exhaustion rule is unreachable. With EWMA half-life 252, compute_au_max_stat returns 26 even for 30 years, not the documented 85.",
    "evidence": ["src/integration/pipeline.py:307-314", "src/integration/pipeline.py:2017-2029", "src/inference/active_units.py:92-117", "results/diagnostic/checkpoints/checkpoint_fold00_train_N739_Y24_T504_K52_F2.pt"],
    "cross_units": ["U14", "U15"],
    "root_cause": "RC2",
    "confidence": "high"
  },
  {
    "id": "U05-MAJ-05",
    "unit": "U05",
    "role": "critic",
    "type": "math",
    "severity": "major",
    "evidence_register": "doc+code",
    "claim": "Dimensions with marginal KL > 0.01 nats are active units (AU); truncate to the AU_max_stat most active dimensions ranked by descending marginal KL.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:404",
    "assessment": "Marginal KL per dimension is not rotation-invariant, and the VAE latent basis is unidentified. Selecting dimensions by KL selects an arbitrary basis; downstream the pipeline diagonalizes Sigma_z and rotates B anyway, so a rotation-invariant rank (not a KL threshold) is the correct notion of factor count.",
    "evidence": ["src/inference/active_units.py:70-89", "src/integration/pipeline.py:2029", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:712-718"],
    "cross_units": ["U09"],
    "root_cause": "RC2",
    "confidence": "high"
  },
  {
    "id": "U05-MAJ-06",
    "unit": "U05",
    "role": "critic",
    "type": "statistics",
    "severity": "major",
    "evidence_register": "doc+code",
    "claim": "N = n x (T_hist - T + 1) is the maximum number of windows (stride s = 1) ... stride subsamples redundant windows without reducing the dataset's informational content.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:368",
    "assessment": "The denominator counts windows overlapping by 96%; the same document states N_eff ~ T_hist/T ~ 15 independent windows per stock. With 739 stocks the effective sample is ~11k, giving r ~ 10^3 rather than the reported 2.7. The capacity guard is off by two to three orders of magnitude and is not a real constraint.",
    "evidence": ["src/vae/build_vae.py:312-347", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:651", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:368"],
    "cross_units": ["U07", "U15"],
    "root_cause": "RC3",
    "confidence": "high"
  },
  {
    "id": "U05-MAJ-07",
    "unit": "U05",
    "role": "critic",
    "type": "math",
    "severity": "major",
    "evidence_register": "doc+code",
    "claim": "A latent dimension remains active only if the reconstruction gain it provides exceeds its KL cost — which, for D = 504 and sigma^2 ~ 0.5, requires explaining at least ~0.1% of signal variance per dimension.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:464",
    "assessment": "With D = T*F = 1008 and the ELBO D/(2 sigma^2)*MSE + KL, at the learned optimum sigma^2 = MSE the marginal threshold is dMSE/MSE > 2/D ~ 0.2%. A dimension survives unless it explains almost nothing. The text uses D = 504 despite F = 2, so AU also depends on F. The mechanism is a threshold, not a discovery.",
    "evidence": ["docs/DVT_strategie_facteurs_risque_latents_v4_1.md:448", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:460-466", "src/vae/model.py:72-88", "src/vae/loss.py:370-394"],
    "cross_units": ["U06"],
    "root_cause": "RC3",
    "confidence": "high"
  },
  {
    "id": "U05-MAJ-08",
    "unit": "U05",
    "role": "critic",
    "type": "justification",
    "severity": "major",
    "evidence_register": "doc+code",
    "claim": "This forces the latent space to be continuous and smooth ... Distances between stocks are therefore meaningful and comparable, which is exactly what portfolio construction requires.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:164",
    "assessment": "Portfolio construction uses neither latent distances nor posterior variance: encode() returns mu only, the variance is discarded, and B enters an OLS and a quadratic form. The VAE's defining output is unused, so the stated justification does not describe the product. The doc also cites Locatello et al. that unsupervised disentanglement is impossible without inductive bias.",
    "evidence": ["src/vae/model.py:152-167", "src/integration/pipeline.py:1992-2096", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:496"],
    "cross_units": ["U08", "U10"],
    "root_cause": "RC1",
    "confidence": "high"
  },
  {
    "id": "U05-MAJ-09",
    "unit": "U05",
    "role": "critic",
    "type": "interpretation",
    "severity": "major",
    "evidence_register": "doc+code+results",
    "claim": "Factor explanatory power ... Target > 0.60; should exceed max(0.50, EP_PCA + 0.10).",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:414",
    "assessment": "Measured EP_OOS = 0.356 and EP_IS = 0.219, below both the target and the floor; EP_PCA is never computed. The health check prints OK because the warning threshold is 0.05, twelve times below the spec floor. This is goal-post moving on the one metric that tests the unit.",
    "evidence": ["results/diagnostic/diagnostic_data.json:155-156", "src/integration/diagnostics.py:105-110", "src/integration/diagnostics.py:1578-1591", "results/diagnostic/diagnostic_report.md:16"],
    "cross_units": ["U14", "U09"],
    "root_cause": "RC5",
    "confidence": "high"
  },
  {
    "id": "U05-MAJ-10",
    "unit": "U05",
    "role": "critic",
    "type": "modeling",
    "severity": "major",
    "evidence_register": "results",
    "claim": "AU = 26 and n_signal = 12.",
    "claim_location": "results/diagnostic/diagnostic_data.json:8",
    "assessment": "The 26-dimension factor model is effectively one-dimensional: eigenvalue ratio #1/#2 = 10.08, top-3 variance = 72.9%, top-1 factor risk contribution = 81.1%, effective eigenvalue count = 5.78. Entropy parity across 26 dimensions is mostly parity across noise, which mechanically links the architecture to the optimizer's failure.",
    "evidence": ["results/diagnostic/diagnostic_data.json:116", "results/diagnostic/diagnostic_data.json:150", "results/diagnostic/diagnostic_data.json:295", "results/diagnostic/diagnostic_report.md:74-78", "results/diagnostic/diagnostic_report.md:92-95"],
    "cross_units": ["U09", "U10"],
    "root_cause": "RC2",
    "confidence": "high"
  },
  {
    "id": "U05-MAJ-11",
    "unit": "U05",
    "role": "critic",
    "type": "consistency",
    "severity": "major",
    "evidence_register": "doc+code+results",
    "claim": "Baseline: K = 200.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:420",
    "assessment": "K is 50 in config, 75 in the notebook, 200 in the docs and 52 at runtime. The feasibility table, sizing example and unit tests exercise a model that never runs; the documented AU range 60-120 is incompatible with the actual K = 52.",
    "evidence": ["src/config.py:179", "notebooks/dashboard.ipynb:247", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:376-382", "results/diagnostic/checkpoints/checkpoint_fold00_train_N739_Y24_T504_K52_F2.pt"],
    "cross_units": ["U15", "U17"],
    "root_cause": "RC5",
    "confidence": "high"
  },
  {
    "id": "U05-MAJ-12",
    "unit": "U05",
    "role": "critic",
    "type": "consistency",
    "severity": "major",
    "evidence_register": "doc+code+results",
    "claim": "Default aggregation: simple mean (all windows contribute equally, preserving memory of all historical regimes).",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:656",
    "assessment": "Runtime uses an EWMA with half-life 60 windows at inference stride 1, i.e. ~60 trading days, not the ~5 years claimed in the code comment. B is a ~three-month snapshot of overlapping two-year windows; the VAE's multi-regime window is averaged into essentially one window, and the claimed memory of dormant risks is absent.",
    "evidence": ["src/integration/pipeline.py:1976-1996", "src/integration/notebook_helpers.py:157-158", "src/inference/composite.py:127-165"],
    "cross_units": ["U08", "U01", "U09"],
    "root_cause": "RC4",
    "confidence": "high"
  },
  {
    "id": "U05-MAJ-13",
    "unit": "U05",
    "role": "critic",
    "type": "data",
    "severity": "major",
    "evidence_register": "doc+code",
    "claim": "A stock with identical return shapes but transitioning from 10% to 40% annualized volatility carries fundamentally different risk, invisible to the return channel alone.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:250",
    "assessment": "Both features are z-scored per window, so the absolute volatility level is erased; only the within-window shape of the vol series survives. DVT:724 itself admits pure shape with no absolute scale. The encoder is a scale-free shape descriptor; scale is injected only downstream by the volatility rescaling.",
    "evidence": ["src/data_pipeline/windowing.py:252-273", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:240-250", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:724", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:730"],
    "cross_units": ["U04", "U09"],
    "root_cause": "RC4",
    "confidence": "high"
  },
  {
    "id": "U05-MAJ-14",
    "unit": "U05",
    "role": "critic",
    "type": "literature",
    "severity": "major",
    "evidence_register": "doc",
    "claim": "Dimensions with marginal KL > 0.01 nats are active units (AU). This is confirmed by TempVAE on financial time series (Sicks et al., 2023); the VAE needs K >= 1.5-2x the intrinsic dimensionality to auto-prune reliably.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:404",
    "assessment": "AU as implemented is the mean marginal KL, whereas Burda et al.'s active-unit criterion is a covariance between input and KL; the two are not equivalent and the 0.01 constant is uncalibrated. No derivation is given for the 1.5-2x sizing rule, and one paper on a different architecture cannot confirm this mechanism.",
    "evidence": ["src/inference/active_units.py:29-31", "src/inference/active_units.py:77-80", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:420"],
    "cross_units": ["U06", "U17"],
    "root_cause": "RC2",
    "confidence": "medium"
  },
  {
    "id": "U05-MIN-15",
    "unit": "U05",
    "role": "critic",
    "type": "implementation",
    "severity": "minor",
    "evidence_register": "code",
    "claim": "C_L(K) = round_16(max(C_min, ceil(alpha_proj x 2K))); dominant scaling O(K^2 x L).",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:348",
    "assessment": "C_min = 384 binds for all K <= 147, so at the operating K = 52 the final width and most of P_enc are independent of K; the claimed quadratic scaling does not hold in the regime used. The decoder fc term (325,632 parameters at K = 52) is the largest single tensor and is discarded at inference.",
    "evidence": ["src/vae/build_vae.py:58-70", "src/vae/build_vae.py:199-227", "src/vae/decoder.py:128", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:362"],
    "cross_units": ["U15"],
    "root_cause": "RC3",
    "confidence": "high"
  },
  {
    "id": "U05-MIN-16",
    "unit": "U05",
    "role": "critic",
    "type": "math",
    "severity": "minor",
    "evidence_register": "doc+code",
    "claim": "AU_max_stat = floor((-1 + sqrt(1 + 4 N_obs / r_min)) / 2).",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:424",
    "assessment": "The printed formula gives 59 for N_obs = 7309 and r_min = 2, not the tabulated 85; the correct expression is (-1 + sqrt(1 + 8 N / r_min)) / 2. The code uses floor(sqrt(2 N_eff / r_min)) = 85, matching the table but not the spec, so the binding constraint cannot be reproduced from the stated formula.",
    "evidence": ["docs/DVT_strategie_facteurs_risque_latents_v4_1.md:424-436", "src/inference/active_units.py:100-117"],
    "cross_units": ["U15"],
    "root_cause": "RC2",
    "confidence": "high"
  },
  {
    "id": "U05-MIN-17",
    "unit": "U05",
    "role": "critic",
    "type": "implementation",
    "severity": "minor",
    "evidence_register": "code",
    "claim": "measure_active_units computes marginal KL and identifies AU; the pipeline derives AU from pre-computed KL.",
    "claim_location": "src/integration/pipeline.py:2010",
    "assessment": "The tested function is not the production path: the pipeline reimplements the threshold, sorting and truncation inline and measures KL in-sample on 96%-overlapping inference windows. The unit test therefore does not exercise the AU used downstream.",
    "evidence": ["src/inference/active_units.py:19-89", "src/integration/pipeline.py:2010-2029"],
    "cross_units": ["U14"],
    "root_cause": "RC5",
    "confidence": "high"
  },
  {
    "id": "U05-MIN-18",
    "unit": "U05",
    "role": "critic",
    "type": "logic",
    "severity": "minor",
    "evidence_register": "doc+code",
    "claim": "Translation invariance: the same kernel recognizes a volatility cluster whether it occurs at day 10 or day 200 — the risk profile should reflect what patterns occur, not when.",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:311",
    "assessment": "For a covariance model, when is the information: a volatility cluster ending today and one ending two years ago imply different current beta. Global average pooling enforces exactly this invariance, and two nested averages (pooling then profile aggregation) discard the recency the risk model needs.",
    "evidence": ["src/vae/encoder.py:167", "src/vae/decoder.py:144-170", "src/data_pipeline/windowing.py:252-273"],
    "cross_units": ["U04"],
    "root_cause": "RC4",
    "confidence": "medium"
  }
]
```

END-OF-REPORT
