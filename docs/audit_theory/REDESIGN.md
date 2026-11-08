> Source document. The redesign of record is [docs/REDESIGN.md](../REDESIGN.md).

# Latent Risk Factor Discovery — Systemic Redesign (theory audit, 2026-11-07)

**Scope.** Theory only. The audit covers the nine corpus documents (DVT v4.1, philosophy, ISD, decision log, two
literature reviews, Σ_z note, strategy comparison, benchmark procedure). It uses no code, no results and no earlier
audit. Numerical outcomes stated in the corpus are not used as evidence.

**User decision (not challenged).** The strategy stays a deep-learning risk model. The architecture, the loss, the
learned object and the coupling to the risk model and the allocation may all change. PCA, Ledoit-Wolf, DCC, IPCA, 1/N,
ERC and inverse-volatility appear only as baselines, protocols and stopping criteria. The PCA-replacement branch
(DVT §6.7, Benchmark scenario C) is therefore not an admissible outcome. Where a defect suggests a linear model would
suffice, this document names the deep formulation that could add value, says why, and gives the experiment that
decides it.

**Evidence base.**
- 19 units, each audited by one blind pair of agents (advocate and devil's advocate), 38 reports.
- 59 critical findings, each checked by the orchestrator: the cited line, the derivation (recomputed) and the
  literature (web).
- Outcome: 43 confirmed, 14 downgraded to major, 2 rejected. See `INDEX.md` and `findings.jsonl`.
- Finding IDs below (`Uxx-CRI-nn`, `Uxx-ADV-nn`) refer to the raw reports in `reports/`.

---

## 1. Executive summary

The corpus describes a coherent pipeline: a variational autoencoder learns per-stock latent vectors from z-scored
windows; they are averaged into an exposure matrix B and rescaled by volatility; cross-sectional regressions give
factor returns and a full-history Σ_z; and an entropy-of-principal-factor-contributions reward is traded against
variance.

The audit finds that the pipeline's weaknesses are not local. Seven root causes generate almost every confirmed
critical finding. One sentence summarises the diagnosis: **the network is trained to compress single-stock window
shapes, while the strategy needs a conditional cross-sectional covariance; every layer downstream compensates for
that mismatch with heuristics whose outputs then serve as evidence for the design.**

1. **The learned object is not the risk object (RC-1).** No term of the final objective ties the latent vector to a
   cross-sectional loading. The window encoder is scale-blind by construction, so scale is re-imported by a
   heuristic. The only cross-sectional term, the co-movement scaffold, is negligible in magnitude, pulls toward an
   isotropic metric the risk model does not use, and is designed to be removed.
2. **A static, unconditional object is asked to answer a conditional, episodic question (RC-2).** Full-history mean
   aggregation, a full-history Σ_z and a constant D_ε dilute or cancel episodic exposures. Equal-contribution
   allocation then puts the largest exposure on the lowest-variance (dormant) factor, which is amplified when it
   reactivates. A covariance-only Gaussian object cannot express the tail dependence the narrative invokes.
3. **Unidentified coordinates feed the objective and the factor count (RC-3).** H(w) depends on the arbitrary latent
   scale and on the eigenbasis choice under near-degeneracy. AU depends on the latent basis, on D = T·F and on a
   guard whose printed formula is wrong. In practice the guard, not the data, sets the count.
4. **Sample accounting treats overlapping and dependent observations as independent (RC-4).** This affects the
   capacity ratio, the Ledoit-Wolf intensity and the paired Wilcoxon test.
5. **Evaluation happens inside the model's own coordinates and on the selection sample (RC-5).**
   - The primary metric Ĥ = H/ln AU is computed on the frozen model and cannot be compared across arms.
   - The configuration is selected on the same OOS period that is reported.
   - Absence of significance at n = 34 dependent folds is read as equivalence.
   - The validated checkpoint and the deployed model are trained under different objectives.
6. **The default objective has negative in-model value (RC-6).** With μ = 0, any α > 0 is weakly dominated in model
   variance. The entropy reward can only pay through model error, which is never formalised or tested.
7. **The data layer is plumbing, not a measurement contract (RC-7).** Market capitalisation is built from
   dividend-adjusted prices. Delisting returns are imputed without knowing the cause, which matters because the
   user has no CRSP.

**Redesign.** Keep the two-stage contract the advocates converge on: a nonlinear, amortised loading map, then an
explicit, exactly solved low-rank-plus-diagonal covariance. Keep the shape/scale separation, the return-free default,
point-in-time hygiene and the benchmark-parity harness. Change four things:

- **The learned object.** Make the network output a *conditional* loading matrix B_t (and a conditional factor
  covariance), trained end-to-end on the **out-of-window cross-sectional predictive likelihood** of returns, not on
  window reconstruction.
- **Every downstream quantity becomes invariant.** Use asset-space quantities only: the diversification measure,
  the factor count and the stability diagnostics no longer depend on latent coordinates.
- **Diversification changes status.** It becomes a stated robustness hypothesis, tested as such, not an additive
  reward whose value is assumed.
- **Evaluation is redone.** Use proper scoring rules on the covariance (where the data has statistical power),
  nested selection, a family-wise test across all variants, and the holdout as the confirmatory sample.

The hypothesis that decides the scope of the redesign is **H1a (conditionality)**: do conditional loadings beat the
static full-history object out of sample on a covariance loss? Its minimal experiment uses daily data, needs no large
network, and is described in §8.

---

## 2. Method in one paragraph

**Units and agents.**
- 19 units cover vision, nature of risk, data, features, architecture, factor count, core loss, auxiliary losses,
  training, inference, risk model, diversification objective, optimisation, validation, benchmarks, metrics and
  tests, conclusions and roadmap, parameters, and cross-document consistency.
- Each unit got one advocate (what is correct and under which conditions) and one critic (everything except the DL
  decision).
- The agents were DeepSeek v4.1 Flash at maximum effort. They worked blind and read-only on the closed corpus, with
  evidence restricted to `doc`, `math`, `literature` (flagged `[unverified]` when from memory) and `hypothesis`
  (with a testable prediction).

**Validation and verification.**
- Every report passed the structural validator with ≥ 80% of citations resolved. Two critic reports were rerun
  fresh because they abbreviated file names.
- The orchestrator re-derived every critical claim and checked every cited reference on the web.
- Downgrades and rejections, with reasons, are in `INDEX.md` §3. Two examples:
  - U12-CRI-02 is rejected because it treats sample eigenvalues as true ones.
  - U18-CRI-04 is rejected because under the violation INV-001 describes, σ² = MSE/D, which is consistent with the
    detector.

---

## 3. Root-cause map

**Defect types:**
- **[I]** internal inconsistency (within or between documents, or formula vs prose);
- **[L]** disagreement with the literature;
- **[U]** unjustified load-bearing choice.

"Conf." lists the confirmed critical findings; "Supp." lists downgraded or major findings that support the root cause.

### RC-1 — The learned object is not the risk object

The network is optimised to reconstruct single-stock, per-window z-scored shapes. The risk model needs loadings on
shared factors and their covariance.

**Mechanism.**
- The final loss (Phase 3) is a sum of per-window ELBOs. It is permutation-invariant across stocks and contains no
  shared z_t (DVT L446-496, L512).
- The corpus itself concedes that the shared encoder "is not guaranteed to preserve empirical correlation structure"
  (L496). Yet it names μ "the estimated exposures β_i" (L174).
- Both input channels are z-scored per window, so the input is exactly invariant to r → c·r. The realised-volatility
  channel cannot restore scale (U04-CRI-01), and scale must be re-imported by σ_{i,t}/σ̄_t (L728).
- The co-movement term is too small to matter:
  - it is ~10⁻³ of the reconstruction term at the baseline λ_co (U07-CRI-01);
  - its fixed point cos(z_i, z_j) = ρ_ij matches the model's implied correlation only if Σ_z ∝ I with no residual
    (U08-CRI-01, counterexample: target 0.707 vs implied 0.894);
  - it is removed in Phase 3.
- In-sample diagnostics cannot tell a factor basis from a noise basis of the same dimension:
  - E[R²] = K/(n−1) ≈ 8% for a pure-noise basis at AU = 80, n = 1000 (U10-CRI-01);
  - at equal dimension, EP_VAE ≤ EP_PCA by the Ky Fan principle (U05-CRI-04).

| Type | Findings |
|---|---|
| [I] | L174 vs L496; L240 vs L250 (U04-CRI-01); the scaffold "temporary" (L508) yet permanent if E* < 60% of the budget (U09-CRI-02) |
| [L] | Locatello et al. (cited by the corpus at L496) and Lucas et al. 2019 (verified): the ELBO does not identify factors; posterior collapse is governed by the noise variance |
| [U] | per-window reconstruction chosen as the training signal for a covariance model |

Conf.: U05-CRI-01 (narrowed), U07-CRI-01, U04-CRI-01, U08-CRI-01, U10-CRI-01, U05-CRI-04.
Supp.: U02-CRI-01, U05-CRI-02, U08-CRI-02, U10-CRI-02, U07-CRI-02.

### RC-2 — A static, unconditional risk object answers a conditional, episodic question

**Mechanism.**
- **Aggregation dilutes episodic loadings.** Mean aggregation over a lifetime (L656) gives an episodic loading b a
  weight q equal to its window fraction, so its risk contribution scales as q² (≈ 1% for a 6-month episode in
  30 years) (U01-CRI-03).
- **Sign changes cancel.** Mean aggregation charges loading variation to D_ε:
  D_ε,i = Var(e_i) + E[(β_{i,t} − β̄_i)²]·Var(z) (U10-CRI-03).
- **The episodic structure is frozen.** Σ_z and D_ε are full-history constants. Correlation changes come only from
  the scalar volatility rescaling, a functional-form artefact: the model cannot fail to show "contagion"
  (U02-CRI-04).
- **The allocation amplifies the dormant factor.** It equalises *average* contributions, so |β′_k| ∝ λ_k^{-1/2}
  (philosophy L113) maximises exposure to the lowest-variance factor. At reactivation that factor's share is
  1/(AU·p + 1 − p) ≈ 20% (U02-CRI-02) or 38.8% vs a 1.25% target (U12-CRI-01). "Anti-cyclical by construction"
  (L17) is inverted.
- **Rescaling is reactive.** Current-volatility rescaling (L742) de-weights stocks after their volatility rises,
  with the 252-day lag the DVT concedes (L734) (U17-CRI-03).
- **Gaussian objects have no tail dependence.** A covariance-only object has zero tail dependence for ρ < 1
  (Embrechts, McNeil & Straumann 2002, verified). The "tail co-movement" the narrative relies on (L11, L511) cannot
  be represented (U02-CRI-03). Longin & Solnik (2001, verified) document that stress correlation is a lower-tail
  phenomenon, not a volatility one.

| Type | Findings |
|---|---|
| [I] | L724's argument against a single volatility snapshot applies verbatim to the fixed shape vector (U10-CRI-03); "protected before" (L78) vs reactive rescaling (L734/L742) |
| [L] | Embrechts et al. 2002; Longin & Solnik 2001 |
| [U] | the full-history mean as estimand |

Conf.: U01-CRI-03, U02-CRI-02, U02-CRI-03, U02-CRI-04, U10-CRI-03, U12-CRI-01, U17-CRI-03.
Supp.: U03-CRI-01.

### RC-3 — Unidentified coordinates feed the objective and the factor count

**Mechanism.**
- **Latent scale.** B → BS, Σ_z → S⁻¹Σ_zS⁻ᵀ leaves Σ_assets unchanged but changes the eigenvectors of Σ_z, hence
  the principal factors and H. Recomputed: H = 0.207 vs 0.298 for the same portfolio and asset covariance
  (U01-CRI-02).
- **Eigenbasis.** "The principal factors of Σ_z are unique (up to sign)" (L692) is false under eigenvalue
  degeneracy (H = 0 or ln 2 for the same portfolio) and ill-conditioned near it. Eigenvector error grows as
  1/gap, and the low-variance tail is exactly where exposures are largest.
  - Correction to three critics: identity-target shrinkage does not change eigenvectors, so it does not create the
    problem; the sample spectrum does.
  - Findings: U12-CRI-03, U13-CRI-03, U17-CRI-02.
- **Factor count.**
  - AU depends on the latent basis (U06-CRI-01) and on D through the activation threshold 2σ²ΔKL/D: the same data
    gives more "factors" with a longer window (U05-CRI-03).
  - The printed guard formula gives 59, not 85: coefficient 4 instead of 8; the table uses √(2N/r) from ISD L222
    (U11-CRI-01, U18-CRI-03, U19-CRI-01).
  - The guard is declared binding (L436), so the count is a function of sample length (U06-CRI-02).
  - The literature behind it is misattributed: Burda et al. define activity as Cov_x(E_q[u]) > 10⁻², not KL > 0.01
    nats, and state no "K ≥ 1.5-2× intrinsic dimension" rule (verified; U19-CRI-02). In the linear limit the
    retained count is a pPCA elbow λ_k > σ² (Tipping & Bishop 1999, verified), not a population factor count.
- **Shrinkage as a hidden tuning knob.** The shrinkage intensity δ compresses the eigenvalue spread by exactly
  (1 − δ), so it moves H and the α elbow, while the PCA arm is unshrunk (U11-CRI-03).

| Type | Findings |
|---|---|
| [I] | printed AU_max formula (DVT L424) vs its table and the ISD form (ISD L222, L1646), and a third parameter count in R_Σ (DVT L935); L692 vs L908 ("AU may differ between retrainings") |
| [L] | Burda et al. misattribution; Meucci's ENB (cited [39]) presupposes factors with a meaningful scale; with an unnormalised latent basis its value is not identified |
| [U] | KL > 0.01 nats as a factor count |

Conf.: U01-CRI-02, U12-CRI-03, U13-CRI-03, U17-CRI-02, U06-CRI-01, U06-CRI-02, U05-CRI-03, U11-CRI-01, U18-CRI-03,
U19-CRI-01, U19-CRI-02, U11-CRI-03.

### RC-4 — Sample accounting treats dependent observations as independent

**Mechanism.**
- The capacity ratio uses N = n·(T_hist − T + 1) = 7.06M (L368), while the same document states the effective count
  is ≈ T_hist/T per stock (L651, L1076). The ratio is then r_eff ≈ 693-1,087 instead of 1.5-2.3, and the risk-model
  parameters are absent from any capacity check (U04-CRI-02, U18-CRI-01).
- The Ledoit-Wolf intensity assumes i.i.d. returns (Ledoit & Wolf 2003, Assumption 1, verified) and is applied to
  volatility-deflated, clustered factor returns (U11-CRI-02, downgraded: its N_eff ≈ 30 derivation is wrong, the
  assumption violation is real).
- The signed-rank test is applied to adjacent, nested-window folds. Under AR(1) with ρ = 0.4 the actual size of a
  nominal 5% test is ≈ 0.20 (U16-CRI-03).

| Type | Findings |
|---|---|
| [I] | L368 vs L651/L1076 |
| [L] | Diebold & Mariano 1995 and Politis & Romano 1994 (both verified) give the dependence-robust alternatives |

Conf.: U04-CRI-02, U18-CRI-01, U16-CRI-03. Supp.: U11-CRI-02, U09-CRI-01.

### RC-5 — Evaluation inside the model's own coordinates and on the selection sample

**Mechanism.**
- **Ĥ is not an out-of-sample quantity.**
  - It is the optimiser's own objective, evaluated on the frozen B and Σ_z, so it measures solver success.
  - It is homogeneous of degree 0, so it cannot distinguish a 5% from a 50% volatility portfolio (U12-CRI-04).
  - Its denominator ln AU vs ln k is model-chosen, which makes it non-comparable across arms and across AU
    (U14-CRI-04, U15-CRI-02, U16-CRI-01).
  - It is undefined for 1/N and inverse-volatility, so the composite-score test cannot be run against those two
    arms. Scenario A stays decidable through the per-metric tests.
- **Selection on the reported sample.** Phase A picks the best of up to 32 configurations on the fold's OOS
  (L894). Phase B reports the same OOS (L898), and the benchmarks are not selected. Expected inflation ≈ 2.07σ for
  m = 32 (recomputed; Cawley & Talbot 2010, verified) (U14-CRI-01, U15-CRI-01, U16-CRI-02, U18-CRI-02, U06-CRI-03).
- **Decision logic.**
  - p ≥ 0.05 at n = 34 is read as "the VAE adds nothing". Power for d = 0.3 is ≈ 0.40 (recomputed) (U15-CRI-03).
  - Five iterations are triggered and retained on the same folds before the terminal test: family-wise error
    ≈ 0.26 (U17-CRI-05; White 2000 and Hansen 2005, verified).
  - The clean 3-year holdout has no decision role.
- **The validated model is not the deployed model.** The curriculum clock is a fraction of total_epochs (ISD
  L1400), while Phase A stops at E* and Phase B trains for E* epochs (U09-CRI-02).

| Type | Findings |
|---|---|
| [I] | DVT L874 ("not used for HP comparison") vs L894; DVT L894 vs ISD L2434 ("nested validation OOS"); median Ĥ vs per-fold score (U14-CRI-03) |
| [L] | Cawley & Talbot 2010; White 2000; Hansen 2005; Hansen, Lunde & Nason 2011 |
| [U] | normalising by the model's own capacity |

Conf.: U12-CRI-04, U14-CRI-01, U14-CRI-04, U15-CRI-01, U15-CRI-02, U15-CRI-03, U16-CRI-01, U16-CRI-02, U17-CRI-05,
U18-CRI-02, U06-CRI-03, U09-CRI-02. Supp.: U14-CRI-03, U11-CRI-05.

### RC-6 — The default objective has negative in-model value

**Mechanism.**
- With μ = 0 the model assigns the same expected return to every portfolio. Any α > 0 solution therefore has weakly
  higher model variance than minimum variance (U13-CRI-02).
- The entropy reward can pay only through (i) robustness to Σ misspecification, (ii) an implicit return bet, which
  contradicts L776, or (iii) a non-variance risk measure. The corpus asserts (i) and (iii) in prose and formalises
  neither.
- The scalarisation adds variance and nats with uncalibrated exchange rates. α is described as a "price … in return
  units" (L1052), and λ is calibrated from an equity risk premium inside a forecast-free mode (U13-CRI-01,
  downgraded).

| Type | Findings |
|---|---|
| [I] | "neutral" (L776) vs ERP-calibrated λ (L786) |
| [U] | additive entropy reward without a stated loss it minimises |

Conf.: U13-CRI-02. Supp.: U13-CRI-01.

### RC-7 — The data layer is plumbing, not a measurement contract

**Mechanism.**
- **Market capitalisation.** It is built as Adjusted_close × shares (ISD L649) from a split- and dividend-adjusted
  price (ISD L596), so historical capitalisation is scaled by the cumulative dividend factor (≈ 0.54 at 2.5% yield
  over 25 years). Universe membership becomes a function of dividend yield (U03-CRI-02).
- **Delisting returns.** The Shumway figures (−30% NYSE/AMEX, −55% Nasdaq) are corrections for *performance-related*
  delistings (Shumway & Warther 1999, verified; the −30% figure confirmed indirectly). Without CRSP the delisting
  cause is unknown, so mergers get the penalty. Per-window z-scoring then compresses the other 503 observations by
  ≈ 0.78 (U03-CRI-03).
- **The corpus assumes CRSP.** The DVT names CRSP as the data source (L736); the user has none.

| Type | Findings |
|---|---|
| [I] | float-adjusted definition (ISD L600) vs construction (ISD L649) |
| [L] | Shumway 1997; Shumway & Warther 1999 |
| [U] | unconditional imputation |

Conf.: U03-CRI-02, U03-CRI-03.

---

## 4. What to preserve

Each item below is the consensus of several advocates and survives the verified critic findings, possibly with a
stated condition.

| # | Preserved element | Advocates | Condition imposed by verified critiques |
|---|---|---|---|
| P1 | **Two-stage contract**: a learned (nonlinear) loading map feeding an explicit, exactly solved low-rank + diagonal covariance BΣBᵀ + D. Guarantees PSD, auditability and a linear counterpart | U05-ADV, U01-ADV, U07-ADV, U11-ADV | The map must be trained on a cross-sectional objective (RC-1). The diagonal residual becomes a tested hypothesis, not an axiom (U05-CRI-02, U11-CRI-04) |
| P2 | **Decision invariance to latent coordinates**: only span(B) and Σ_assets may feed decisions | U01-ADV-04, U07-ADV, U11-ADV-05 | Currently violated by H and AU (RC-3); the redesign enforces it everywhere |
| P3 | **Shape/scale separation**: a scale-free learned representation plus a measured scale path | U04, U05, U07, U10, U11, U17, U18 advocates | Keep as an architectural contract; the realised-volatility channel is no longer justified as "restoring scale" (U04-CRI-01) |
| P4 | **Return-free default with a removable return layer** | U01, U12, U13, U19 advocates | Drop "neutral" and "risk-minimising" for α > 0 (U13-CRI-02) |
| P5 | **Non-negative contributions from an orthogonal decomposition** (no entropy on signed contributions) | U12, U13, U17, U19 advocates | Move the decomposition to an identification-invariant asset-space basis (§5, L3) |
| P6 | **Point-in-time causal features and a delisted-inclusive universe** | U03, U04, U14, U15 advocates | Repair the market-capitalisation and delisting constructions (RC-7) |
| P7 | **Benchmark parity**: same universe, folds, constraints, solver, α procedure and penalties for every arm | U13, U14, U15, U16, U19 advocates | Extend parity to the selection budget (RC-5) |
| P8 | **One untouched holdout**, "the only uncontaminated evaluation" (L880) | U14, U16, U18 advocates | Promote it from structural check to confirmatory inference |
| P9 | **A single observability budget**: free covariance parameters against observations | U03, U06, U11, U19 advocates | Correct formula, effective (not counted) observations, one formula across documents |
| P10 | **Gauge-invariant stability diagnostic** (rank correlation of pairwise inter-stock distances) | U05, U06, U10, U17 advocates | Unchanged |
| P11 | **Regime-conditioned evaluation and a pre-registered outcome → action map** | U02, U15, U16, U17 advocates | Re-point the actions: no PCA adoption |
| P12 | **Likelihood discipline in training**: learned scalar σ², D-scaling, β = 1, mutually exclusive loss modes, auxiliaries excluded from the selection signal | U07, U09, U18, U19 advocates | Keep if a reconstruction component survives as an auxiliary or pretraining loss; stop reading AU as a factor count |

**Advocate claims not preserved, because the verified critiques defeat them.**
- **Normalised entropy Ĥ = H/ln AU as a fairness device** (U06-ADV-07, U18-ADV, U19-ADV). It is non-comparable
  across arms and AU and is not an out-of-sample quantity (U15-CRI-02, U14-CRI-04, U12-CRI-04).
- **"The principal factors of Σ_z resolve rotational indeterminacy"** (U12-ADV, U19-ADV-02). They do not resolve
  scale indeterminacy (U01-CRI-02), and they are ill-posed under degeneracy (U12-CRI-03).
- **EP ≥ EP_PCA + 0.10 as the falsifier of nonlinearity** (U06-ADV-09). It is unattainable at equal dimension
  (Ky Fan) and vacuous at unequal dimension (U05-CRI-04, U11-CRI-05).
- **AU as a model-derived effective dimension** (U06-ADV-01, U05-ADV-04). It is basis-dependent, D-dependent and
  guard-bound (RC-3). The *principle* of a K ceiling separate from a data-derived effective dimension survives,
  with a different estimator (§5, L1).

---

## 5. Target architecture

One system, five layers. Each layer has an explicit input/output contract. The deep-learning content sits in L1
(conditional loading map) and L2 (conditional factor covariance and, optionally, a tail head). The linear models are
the counterparts that every deep layer must beat.

```
L0 Data contract ─► L1 Conditional loading map g_θ ─► L2 Conditional covariance Σ_t (+ tail head) ─► L3 Allocation ─► L4 Evaluation
   (point-in-time)    (deep, cross-sectionally          (low-rank+diag, PSD;               (invariant         (nested, proper
                       trained)                          deep state dynamics)                diversification)   scoring, holdout)
```

### L0 — Data contract (non-CRSP)

**Output.** A point-in-time panel {r_{i,t}, raw close, shares outstanding as known at t, listing status, delisting
date, delisting type when known}.

**Contract.**
- (a) Capitalisation is built from **unadjusted** close × point-in-time shares, never from adjusted prices.
- (b) Delisting returns are imputed only for delistings flagged performance-related. When the cause is unknown,
  every downstream result is computed under two bounds (0 and the Shumway value) and reported as an interval.
- (c) Every feature statistic is computed from data dated ≤ t. Preserved from P6.

This layer has no deep-learning content. It is the measurement contract the other layers rely on.

### L1 — The learned object: a conditional, cross-sectionally trained loading map

**What the network computes.**
- Per stock and date, B_{i,t} = s_{i,t} · g_θ(x_{i,t}).
  - x_{i,t} is a price-derived state of stock i at t: its recent z-scored return and volatility path (the current
    window encoder can serve here as a feature extractor), plus price-derived characteristics (size, liquidity,
    trailing beta, momentum, volatility).
  - s_{i,t} is the measured scale: the shape/scale contract P3 is kept, and scale is supplied, not learned.
- g_θ is shared across stocks (P2/U05-ADV-05) and outputs a K-vector.

**Training signal.**
- The cross-sectional Gaussian negative log-likelihood of **next-period returns**:
  −log N(r_{t+1}; 0, B_t Σ_{f,t} B_tᵀ + D_t), with D_t diagonal.
- Equivalent alternatives: a factor-model regression loss with factors as cross-sectional projections, in the
  conditional-autoencoder architecture of **Gu, Kelly & Xiu (2021, J. Econometrics 222:429-450, verified)**, or the
  low-rank + diagonal likelihood head of **Salinas et al. (2019, NeurIPS, verified)**.
- The loss is the risk object, evaluated out of window. This dissolves RC-1 (μ ≡ β is now what is optimised) and
  the co-movement scaffold (removed).

**Linear counterpart.** IPCA (Kelly, Pruitt & Su 2019, JFE, verified): the same instruments, with a linear map.
"Nonlinearity earns its keep" becomes precisely the comparison g_θ vs IPCA on identical inputs, loss and Σ_f
estimator.

**What deep learning adds that linear cannot.**
- Interactions between state variables in the loading, for example exposure to a funding factor that switches on
  only when leverage proxies and volatility are jointly high.
- Path features (shapes of the drawdown and volatility trajectory) that no linear instrument encodes.
- These are exactly the "episodic" exposures the theory wants. Being conditional, they are represented at the date
  they matter instead of being averaged (dissolves RC-2 at the source).

**Identification.**
- Normalise Σ_{f,t} = I, or impose BᵀB diagonal with ordered columns, and compute every downstream object from
  B_tΣ_{f,t}B_tᵀ.
- The latent scale and basis then cannot reach the objective (dissolves RC-3).

**Effective dimension.**
- K is a ceiling (P9 principle).
- The effective dimension k* is chosen by the inner-loop predictive likelihood on held-out dates, subject to the
  observability budget with *effective* observations.
- AU and the KL threshold are dropped as a factor count (they may remain diagnostics of a pretraining encoder).

**Relation to the current VAE.**
- The VAE may survive as a pretraining stage that produces x_{i,t} features, so it is not discarded.
- What changes is that its latent vector is no longer *declared* the loading.
- FactorVAE (Duan et al. 2022, AAAI, verified) is the published template for a VAE whose latent factors are trained
  against future cross-sectional returns rather than window reconstruction.

### L2 — Conditional covariance and optional tail head

**Output.** Σ_t = B_t Σ_{f,t} B_tᵀ + D_t: PSD, low-rank + diagonal, with:
- Σ_{f,t} from a small recurrent or attention module over market-state variables (index volatility, dispersion,
  credit and term proxies available without CRSP), conditioned on state instead of averaged over history;
- D_t from a per-stock volatility model.

**Episodic memory, made explicit.**
- Replace "average everything" with **state-conditional parameters**: a mixture-of-experts over latent regimes, with
  experts that persist across retrainings, so a stress regime's Σ_f is recalled when the state resembles it.
- This is the deep formulation of the corpus's dormant-factor intuition. It is testable by whether stress-day
  covariance forecasts improve (H2c).

**Optional tail head, only if H2 passes.**
- A Student-t or factor-copula output for joint exceedances. The deep version conditions copula parameters on the
  same state.
- Linear/parametric counterpart: Oh & Patton (2017, JBES, verified) factor copulas, which document stronger
  dependence in crashes.

**Baselines (same universe, same refit dates).**
- Ledoit-Wolf, the current choice and the corpus's own baseline;
- DCC with nonlinear shrinkage (Engle, Ledoit & Wolf 2019, JBES, verified);
- static PCA, rolling PCA, IPCA;
- the current design's full-history mean-B pipeline, kept as an arm so the redesign is measured against what it
  replaces.

### L3 — Allocation: invariant diversification as a tested robustness device

**Input.** Σ_t only (and the tail object, if adopted). Nothing in latent coordinates.

**Diversification measure.**
- Use the effective number of **minimum-torsion bets** (Meucci, Santangelo & Deguest 2015, Risk, verified), or
  entropy over the **principal portfolios of the asset-level systematic covariance** B_tΣ_{f,t}B_tᵀ (Meucci 2009;
  Lohre, Opfer & Ország 2014, J. Risk, verified).
- Both are invariant to latent reparametrisation (RC-3 dissolved).
- Minimum-torsion bets are also continuous through eigenvalue degeneracy, unlike principal factors (removes
  U12-CRI-03 and U17-CRI-02 as turnover drivers).

**Objective status.**
- Minimum variance under the constraint ENB(w) ≥ e.
- e is pre-registered from the inner loop, replacing the additive α·H with its uncalibrated exchange rate
  (U13-CRI-01).
- The *reason* for the constraint is stated as hypothesis H3: diversification protects against model error. It is
  tested against α = 0 (U13-CRI-02).

**Preserved.** The SCA/Armijo solver class, soft penalties plus hard caps, the return-free default and the removable
directional layer (P4, U13-ADV).

### L4 — Evaluation

**Primary metrics are representation-free.**
- **Layer-2 accuracy** on daily OOS returns:
  - multivariate Gaussian log-score;
  - QLIKE (Patton 2011, verified) on portfolio variances;
  - realised variance of the global-minimum-variance portfolio (Engle & Colacito 2006, verified);
  - calibration ratio var(r_p)/ŵᵀΣ̂ŵ overall and on stress days.
- **Layer-3 outcomes** on a common footing for every arm: realised volatility, maximum drawdown, stress-episode
  loss, turnover, and ENB of minimum-torsion bets *computed from a common reference covariance* (the OOS realised
  covariance), never from each model's own.

**Selection and inference.**
- **Nested walk-forward.** Hyperparameters and k* are selected in an inner split. Outer OOS folds are scored once.
  Every arm, including benchmarks, receives the same tuning budget (P7 extended).
- **Inference.**
  - Diebold-Mariano with HAC variance on daily loss differentials, or a stationary bootstrap (Politis & Romano
    1994, verified).
  - White's Reality Check or Hansen's SPA across *all* deep variants and benchmarks, so the iteration path is in
    the null (U17-CRI-05).
  - Hansen-Lunde-Nason Model Confidence Set (2011, verified) to report which arms are statistically
    indistinguishable.
- **Equivalence is a pre-registered margin** (two one-sided tests), never non-significance (U15-CRI-03).
- **The holdout is the confirmatory sample** for one pre-registered configuration (P8).
- **One clock.** Any schedule horizon (curriculum, LR) is fixed independently of the stopping epoch, so the
  validated and deployed objectives coincide (U09-CRI-02).

---

## 6. What to remove, merge or move

| Action | Element (location) | Reason |
|---|---|---|
| Remove | Reading AU (KL > 0.01 nats) as the factor count, and the "converges to AU ≈ 80" claim (DVT L404-406, philosophy L79) | RC-3; misattributed literature |
| Remove | The printed AU_max^stat formula, its ISD variant and the R_Σ count (DVT L424, ISD L222/L1646, DVT L935) | wrong algebra; replaced by one effective-observation budget (merge) |
| Remove | The co-movement scaffold and its three-phase curriculum (DVT L494-540) | superseded by the cross-sectional predictive likelihood (L1); fixed point inconsistent with the model (U08-CRI-01) |
| Remove | Crisis weighting γ in the reconstruction term (DVT L544-588) | it cannot couple stocks (U08-CRI-02); replaced by state conditioning (L2) and stress-stratified evaluation (L4) |
| Remove | Ĥ = H/ln AU as the primary and decision metric (DVT L933-951; Benchmark L172, L213) | RC-5 |
| Remove | Phase A selection on the fold's OOS (DVT L894) | RC-5; replaced by nested selection |
| Remove | Scenario C "adopt PCA" and DVT §6.7 as an action (Benchmark L299-301; DVT L1159-1168) | inadmissible under the user decision; re-pointed to "next DL formulation" (§7) |
| Remove | The EP ≥ EP_PCA + 0.10 gate (DVT L475, L910) | unattainable or vacuous (U05-CRI-04) |
| Remove | The capacity ratio r computed on overlapping windows (DVT L364-382; Appendix A) | RC-4; replaced by the effective-sample budget |
| Remove | Burda et al. as support for K ≥ 1.5-2× (DVT L420, L1662) | not in the paper (verified) |
| Remove | λ calibrated from an equity risk premium in the μ = 0 mode (DVT L786) | inconsistent with the return-free mode |
| Move | "Anti-cyclical by construction", "low turnover", "no a priori bias" (DVT L17) → hypotheses H2c, H3b, H1b | RC-2, RC-6: claims become tests |
| Move | The VAE window encoder → optional feature extractor or pretraining for g_θ (L1) | its latent vector is no longer declared the loading |
| Move | The full-history mean-B pipeline → baseline arm | it is what the redesign must beat |
| Move | The holdout → confirmatory inference | P8 |
| Move | Layer-2 covariance metrics → primary; portfolio Ĥ → diagnostic, computed in a common basis | RC-5 |
| Merge | Validation ELBO, early stopping and HP selection → one inner loop with the deployment objective | RC-5 (U09-CRI-02) |
| Merge | The three observability rules (truncation, R_Σ penalty, shrinkage regime) → one budget on effective observations | RC-3, RC-4 (U11-ADV-03 consensus) |
| Merge | The benchmark set: add DCC-NL, rolling PCA and IPCA; keep 1/N, LW min-variance, ERC, inverse-volatility and static PCA-FRP | RC-5; strongest linear and conditional counterparts present |

---

## 7. Falsifiable hypotheses and stopping criteria, per layer

**Realistic data assumptions (no CRSP).**
- A survivorship-free vendor panel of US equities with delisted names, daily, about 25-30 years. The 3-year holdout
  is excluded from everything except the confirmatory test.
- Independent stress episodes in that span: 1998, 2000-02, 2008-09, 2010-11, 2015-16, 2018 Q4, 2020, 2022. That is
  **about 6-8 episodes**; the holdout may contain none.
- Power arithmetic, for a two-sided α = 0.05 and 80% power, the minimum detectable standardised effect is
  d_MDE ≈ (1.96 + 0.84)/√N_eff:
  - **Layer-2, daily loss differentials.** Roughly 4,000-5,000 OOS days. With loss-differential autocorrelation
    decaying over 20-50 days, N_eff ≈ 100-250, so d_MDE ≈ 0.18-0.28. Usable.
  - **Layer-3, 34 overlapping semester folds.** N_eff ≈ 15-20, so d_MDE ≈ 0.6-0.7. Only large effects are
    detectable; claims are framed as non-inferiority.
  - **Episode-level claims, one-sided sign test.** With n = 7, p < 0.05 requires 7/7 favourable episodes
    (p = 0.0078; 6/7 gives p = 0.0625). With n = 8, 7/8 is required (p = 0.035).
- The power therefore lives in Layer 2. Portfolio and crisis claims are corroborative, not decisive.

| Layer | Hypothesis (refutable) | Prediction | Refuted if | Stopping criterion | Next DL formulation if it fails |
|---|---|---|---|---|---|
| L0 | **H0**: the non-CRSP panel supports the conclusions | Layer-2 rankings of arms are identical under the two delisting bounds and under adjusted vs unadjusted capitalisation | Any pairwise ranking flips between bounds | Stop all claims that flip; report them as data-undetermined | — (measurement layer) |
| L1 | **H1a (conditionality)**: time-varying loadings beat the static full-history object | OOS daily log-score: conditional arms (IPCA, rolling PCA, g_θ) beat the static mean-B arm and static PCA | DM-HAC p ≥ 0.05 *and* the 90% CI of the improvement lies within a pre-registered equivalence margin (TOST) for every conditional arm | If refuted, the episodic/conditional narrative is dropped; the redesign scope shrinks to L2-L4 (static loadings, deep Σ_f dynamics only) | A deep model of the **factor covariance dynamics** on static loadings: Σ_{f,t} from a sequence model (Salinas et al. style), where nonlinearity sits in time, not in the cross-section |
| L1 | **H1b (nonlinearity)**: g_θ beats IPCA on identical instruments, loss and Σ_f estimator | Daily log-score and GMV realised variance improve | DM-HAC p ≥ 0.05 with equivalence (TOST) within the margin, or SPA p ≥ 0.05 across all deep variants | Stop architecture search on L1; deploy IPCA loadings as the L1 baseline inside the DL system and move the deep effort to L2 | A **path-based encoder** (the current CNN/Inception window encoder as x-extractor, trained on the L1 likelihood), then an attention model over the cross-section at t (relational loadings); each tested against IPCA under the same rule |
| L2 | **H2 (risk-model accuracy)**: the deep conditional Σ_t beats DCC-NL and LW | Daily log-score, QLIKE on portfolio variances, GMV realised variance | MCS at 90% contains DCC-NL or LW and excludes nothing deep with a better mean | Stop claiming risk-model superiority; Layer-3 claims must then rest on the allocation rule alone (H3) | A **hybrid**: DCC-NL correlation target plus a deep correction of the dynamics (state-conditioned shrinkage intensity and eigenvalue path) |
| L2 | **H2c (episodic memory)**: state-conditional Σ_f improves stress-day forecasts | Calibration ratio on stress days within [0.8, 1.2] and better than static; stress-day log-score improves | Stress-day log-score not better (DM-HAC on stress days only) *and* the sign test across episodes fails (fewer than 7/7) | Drop the "protected before reactivation" claim (L78) | A regime **mixture-of-experts** with persistent experts across retrainings; then a tail head (H2t) |
| L2 | **H2t (tail)**: a tail-dependent head beats the Gaussian head on joint exceedances | Out-of-sample joint lower-tail exceedance log-score and CVaR coverage improve | No improvement on the tail log-score at the 5% level (HAC) | Keep the Gaussian Σ_t; drop tail-dependence claims (L11, L511) | A deep factor-copula head conditioned on state (Oh & Patton structure, deep parameter map) |
| L3 | **H3a (diversification as robustness)**: min-variance with ENB(w) ≥ e (minimum-torsion) is non-inferior on realised variance and superior on stress-episode loss vs α = 0, *with the same Σ_t* | Realised volatility within a +5% non-inferiority margin; lower drawdown and stress loss | Non-inferiority fails, or stress-loss superiority fails (sign test) | Drop the entropy constraint; deploy GMV on the deep Σ_t | **Decision-focused learning**: train L1/L2 end-to-end through a differentiable allocation layer (OptNet, Amos & Kolter 2017, verified) on realised portfolio variance, so model error and allocation are learned jointly |
| L3 | **H3b (stability)**: minimum-torsion ENB is more stable across retrainings than principal-factor entropy | Lower turnover at equal realised variance | Turnover not lower (paired, HAC) | Keep the measure only if H3a holds | — |
| L4 | **H4 (end-to-end)**: the full system beats every benchmark on the outer folds and in the holdout | SPA rejects "no arm beats the best benchmark"; the holdout confirms the sign with the pre-registered configuration | SPA p ≥ 0.05, or the holdout sign is opposite | Report the MCS and stop deployment claims | The highest-ranked failing layer's "next formulation" above, re-entered through the same nested protocol |

---

## 8. Falsification order and the minimal deciding experiment

**Order.** Each step decides the scope of the next.

1. **H0** (data). If rankings flip between delisting or capitalisation bounds, nothing downstream is interpretable.
2. **H1a (conditionality): scope-deciding.**
   - If static loadings are not beaten by conditional ones on Layer-2 loss, the corpus's central temporal and
     episodic premise has no out-of-sample content. The redesign reduces to deep *covariance dynamics* on static
     loadings, plus the L3/L4 corrections.
   - If conditionality wins, the full L1 redesign is warranted, and H1b decides whether the *deep* map earns its
     place.
3. **H1b** (nonlinearity of the loading map).
4. **H2, then H2c** (risk-model accuracy, then stress-day memory).
5. **H3a** (diversification earns its place).
6. **H2t** (tail head) and **H4** (end-to-end).

**Minimal experiment for H1a (not run).**
- **Data.** Universe: the top ~500 names by unadjusted capitalisation, point-in-time, from the non-CRSP panel.
  Quarterly refits on an expanding window. Daily OOS evaluation over all post-burn-in dates, excluding the holdout.
- **Arms**, all with the same factor-covariance estimator (sample Σ_f with LW) and the same D_t (trailing 252-day
  residual variance):
  - (i) the current design's static object: full-history mean loadings, volatility-rescaled;
  - (ii) static PCA;
  - (iii) rolling-window PCA (504 days);
  - (iv) IPCA with price-derived instruments;
  - (v) optionally, a small g_θ with the same instruments.
- **Score.** Daily multivariate Gaussian log-score of r_{t+1} and GMV realised variance. Arms (i)/(ii) are compared
  with (iii)/(iv)/(v) by Diebold-Mariano with a Newey-West bandwidth chosen from the loss-differential
  autocorrelation, plus TOST with a pre-registered margin (for example, 1% of the static arm's average log-score).
- **Why it is minimal.** It needs no large network, uses only daily data available without CRSP, has the most
  statistical power available in the problem, and its outcome changes the redesign's scope rather than a parameter.

---

## 9. Appendix A — critical finding → redesign element

| Finding | Status | Dissolved by |
|---|---|---|
| U01-CRI-02 | confirmed | L1 identification normalisation; L3 asset-space invariant ENB |
| U01-CRI-03, U10-CRI-03 | confirmed | L1 conditional loadings (no lifetime mean); mean-B kept as a baseline arm (H1a) |
| U02-CRI-02, U12-CRI-01 | confirmed | L3 ENB constraint instead of equalised average contributions; H2c stress calibration; H3a stress-loss test |
| U02-CRI-03 | confirmed | L2 optional tail head, gated by H2t; tail claims removed until then |
| U02-CRI-04 | confirmed | L2 state-conditional Σ_{f,t} and D_t |
| U03-CRI-02, U03-CRI-03 | confirmed | L0 contract (a), (b); H0 bounds |
| U04-CRI-01 | confirmed | P3 scale supplied by contract; the realised-volatility channel kept only if L1 ablation shows value |
| U04-CRI-02, U18-CRI-01 | confirmed | merged effective-observation budget; capacity ratio removed |
| U05-CRI-01, U07-CRI-01, U10-CRI-01 | confirmed | L1 cross-sectional predictive likelihood |
| U05-CRI-03, U06-CRI-01, U06-CRI-02, U19-CRI-02 | confirmed | AU dropped as factor count; k* by inner-loop likelihood under the budget |
| U05-CRI-04 | confirmed | EP gate removed; Layer-2 proper scoring rules |
| U06-CRI-03 | confirmed | nested selection; representation-free primary metrics |
| U08-CRI-01 | confirmed | co-movement scaffold removed |
| U09-CRI-02 | confirmed | one clock: schedule horizon independent of stopping |
| U11-CRI-01, U18-CRI-03, U19-CRI-01 | confirmed | single corrected budget formula, one source of truth |
| U11-CRI-03 | confirmed (narrowed) | the same Σ estimator family and regularisation budget for every arm (P7); invariant ENB in L3 |
| U12-CRI-03, U13-CRI-03, U17-CRI-02 | confirmed (narrowed) | minimum-torsion bets (continuous through degeneracy) |
| U12-CRI-04, U14-CRI-04, U15-CRI-02, U16-CRI-01 | confirmed | Ĥ removed as primary; common-basis portfolio metrics |
| U13-CRI-02 | confirmed | H3a: the entropy constraint must beat α = 0 on stress loss, or it is dropped |
| U14-CRI-01, U15-CRI-01, U16-CRI-02, U18-CRI-02 | confirmed | nested walk-forward; equal tuning budget for every arm |
| U15-CRI-03 | confirmed | TOST equivalence margins; power statement in §7; PCA adoption removed |
| U16-CRI-03 | confirmed | DM-HAC and stationary bootstrap |
| U17-CRI-03 | confirmed | "anti-cyclical" moved to hypothesis H2c/H3a; volatility rescaling inside L2 dynamics, tested |
| U17-CRI-05 | confirmed | SPA over all variants; holdout confirmatory |
| U02-CRI-01, U05-CRI-02, U07-CRI-02, U08-CRI-02, U08-CRI-03, U09-CRI-01, U10-CRI-02, U11-CRI-02, U11-CRI-04, U11-CRI-05, U13-CRI-01, U14-CRI-03, U01-CRI-01, U03-CRI-01 | downgraded | addressed by the same elements (respectively L1; P1 residual test; L1; L2/L4; L1; L4 inner loop; L1; RC-4 budget and HAC; P1 residual test with leverage correction; L4; L3 constraint form; L4 one budget; framing; L0/L2) |
| U12-CRI-02, U18-CRI-04 | rejected | — (see `INDEX.md` §3) |

## 10. Comparison with the v5 audit (`docs/audit_v5/REDESIGN.md`)

This section was written after the redesign above was complete, on first reading of the v5 document.

**How the two audits differ in method.**
- **v5**:
  - read the theory together with the code and the single surviving real-data run;
  - verified 15 of its 78 critical findings against code;
  - had no deep-learning mandate.
- **This audit**:
  - read the theory only, with code and results evidence forbidden;
  - verified all 59 critical findings by re-derivation and checked every reference on the web (14 downgraded,
    2 rejected);
  - works under the user decision to stay deep learning.

### 10.1 Root causes found by both audits

| This audit | v5 | Agreement | Remaining difference |
|---|---|---|---|
| RC-1 learned object ≠ risk object | RC-A estimand mismatch | Same diagnosis: a per-stock window model asked to do a cross-sectional covariance job. Same remedy family: a cross-sectional conditional autoencoder (Gu-Kelly-Xiu) matched against IPCA | v5 makes the deep model a *candidate* that is dropped if it loses ("adopt the linear model"). Here a loss triggers the next deep formulation (§7), as the user decision requires |
| RC-2 static object for an episodic question | RC-C time mismatch (+ the tail bullet of RC-D) | Static averaging cannot implement "memory of dormant risks"; memory must become an explicit, testable mechanism (v5 H4 ≈ our H2c); covariance cannot express tail contagion | v5 shows the code's half-life unit error (60 windows = 60 days). We show the theory-level dilution (q²), sign cancellation and the dormant-factor amplification at reactivation (20-39% of risk vs a 1.25% target) |
| RC-3, factor-count part (AU ≠ factor count, the guard sets the count) | RC-B identification mismatch | Same conclusion: AU and its caps produce the count, and the count must be chosen out of sample | We add the printed-formula error (59 vs 85), the dependence of AU on D = T·F and on the latent basis, and the Burda misattribution |
| RC-5 evaluation in own coordinates and on the selection sample | RC-F evidence mismatch | Selection on the reported OOS; the single path treated as many samples; DM-HAC and stationary bootstrap; an "inconclusive" state; the holdout opened once; one external yardstick for risk concentration | v5 adds code facts (Scenario C emitted from zero comparisons, ERC falling back to 1/N, separate benchmark harness). We add the theory-level counterparts: power 0.40 at n = 34, the validated ≠ deployed curriculum, and Ĥ as the optimiser's own objective |
| RC-4 sample accounting | P6 (preserved) + RC-F | Both treat ≈ T_hist/T effective windows per stock as the honest bound | v5 preserves the overlap-aware ESS guard as a strength (found in code). We show the DVT's own capacity ratio contradicts it (N = 7.06M), i.e. the theory never applied its own bound |
| RC-7 data contract | RC-E data-contract mismatch | Unadjusted prices, a point-in-time universe, delisting sensitivity bounds instead of CRSP assumptions | v5 finds the implemented failures (NaN delisting returns, full-sample universe, `vix_data=None`). We find the specification-level ones (capitalisation from dividend-adjusted prices; cause-blind Shumway imputation) |

### 10.2 Found only by v5, with the likely explanation

- **RC-G governance (four parameter sources, lost artifacts, no run manifest).**
  - Explanation: dependence on code and results. The defect lives in `config.py`, notebooks and run artefacts, which
    this audit was forbidden to read.
  - Its theory-level shadow did appear here as cross-document inconsistencies (U19: three AU-guard forms,
    F = 1 / F = 2 arithmetic, a stale changelog value).
- **Implementation-specific failures**: the 2.74× variance under-forecast, PC1 excluded from H while carrying 81% of
  factor risk, the α_eff rescaling, the fast sub-solver without a turnover constraint, ERC falling back to 1/N.
  - Explanation: code and results.
  - PC1 exclusion and α_eff do not appear in the corpus as specified; they are implementation drifts. The
    under-forecast is a result, inadmissible here by rule.
- **RMT / Donoho-Gavish-Johnstone signal-noise separation as a strength to preserve (v5 P5).**
  - Explanation: it exists in the code's PCA benchmark, not in the corpus's normative design.
  - Its absence from this redesign is a **gap in this audit's baseline list**. A spiked-covariance shrinker (Donoho, Gavish & Johnstone 2018, as cited by v5; not re-verified here)
    or analytical nonlinear shrinkage (Ledoit-Wolf 2020) should join DCC-NL among the L2 baselines.
- **A fundamentals-based control (v5 H5).**
  - Explanation: a genuine gap in this audit. The U03 advocate noted that characteristics may condition loadings, but
    the synthesis kept only price-derived instruments.
  - Adding sector/style exposures as a *benchmark arm* (not an input) is compatible with the price-only discovery
    principle and should be added to L2's baseline set.

### 10.3 Found only by this audit, with the likely explanation

- **The latent-scale and eigenbasis indeterminacy of H** (U01-CRI-02, U12-CRI-03, U13-CRI-03, U17-CRI-02).
  - v5 *preserves* "principal-factor-basis risk contributions … rotation-resolving" (v5 P4) and puts its budgets "in
    the principal basis of Σ_f".
  - Our verified counterexamples show this basis depends on the arbitrary column scale of B, and is ill-posed under
    eigenvalue near-degeneracy.
  - Likely explanation: a gap in v5. Its verification covered code-anchored findings only, and its critics' math
    findings were "not individually re-verified".
  - Consequence for v5's design: with orthonormal PCA loadings the issue disappears, but with v5's own nonlinear
    candidate (conditional AE) it returns unless the factors are normalised. Our L1 normalisation and L3
    minimum-torsion bets close it.
- **The in-model dominance of α = 0 under μ = 0** (U13-CRI-02): entropy can only pay through model error.
  - v5 sees the unit incoherence (α_eff) but not the dominance argument.
  - Likely explanation: a derivation-level finding that a code-anchored verification does not reach.
  - v5's H2 (factor budgeting vs ERC/1/N on stress outcomes) tests a neighbouring question; our H3a (vs α = 0 with
    the same Σ) isolates the entropy term itself.
- **The validated model is not the deployed model** (curriculum clock vs E*, U09-CRI-02).
  - Explanation: probably masked in v5 because the surviving run used a pretrained checkpoint without walk-forward,
    so the Phase A/B pair was never exercised.
- **The dormant-factor amplification at reactivation** (U02-CRI-02, U12-CRI-01). v5 states the general noise version
  ("largest exposures to the noisiest directions").
  - Explanation: the reactivation version is a pure theory consequence of λ^{-1/2} exposures, visible from the
    philosophy text alone.
- **Literature misattribution** (Burda et al. for KL-based active units and the K ≥ 1.5-2× rule).
  - Explanation: only this audit checked references on the web.
- **The algebraic error in the AU_max formula** (59 vs 85).
  - Explanation: v5 read the guard as implemented (with N_eff and a different value, 26), not the printed equation.

### 10.4 Net assessment

The two audits converge on the core:
- the estimand mismatch;
- static memory;
- AU is not a factor count;
- the evaluation cannot fail the theory;
- the data contract.

They also converge on the same remedy family:
- a cross-sectional conditional model scored on an out-of-sample covariance loss;
- a matched linear control;
- a single-path inference;
- the holdout opened once.

This convergence was reached from disjoint evidence: code and results for v5, documents, mathematics and literature
here. It is the strongest support either audit has.

**Where they differ.**
- **(i) Deep-learning mandate.** v5's kill criterion ends the deep line; under the user decision, this redesign
  replaces that branch with the next deep formulation per layer.
- **(ii) Identification.** v5 preserves the Σ_f principal-factor basis that this audit shows to be non-identified
  for a nonlinear factor model.
- **(iii) Governance and implementation.** v5's governance and implementation findings have no counterpart here by
  construction and should be imported unchanged.
- **(iv) Baselines.** This audit should import v5's RMT-shrinkage baseline and fundamentals control.

