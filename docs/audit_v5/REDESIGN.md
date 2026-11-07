# Strategy re-audit v5 — Systemic redesign proposal

> **Date:** 2026-11-07 · **Inputs:** 34 blind reports (17 units × advocate/critic, DeepSeek v4.1 Flash, max effort,
> via Command Code), 728 findings of which 78 graded critical, cross-checked by the orchestrator (Claude) against code
> and results. Reports: [reports/](reports/) · Index and scores: [INDEX.md](INDEX.md).
> **Scope:** the theory of the strategy (DVT v4.1, ISD, Benchmark spec, philosophy, post-mortems) confronted with the
> code and the only surviving real-data run (`results/diagnostic/`, Tiingo, 788 stocks, 1995–2026, one OOS block).
> **Status:** proposal. Nothing in this document has been implemented.

---

## 1. Executive summary

The strategy does not fail because of a bad hyperparameter, a bug or an unlucky run. It fails because **no layer of
the pipeline is defined by the object the next layer needs**, and the gaps were bridged with knobs. Seven mismatches
recur across all 17 units; they share one meta-cause:

> **The design is doctrine-first, not estimand-first.** Principles ("no prior", "the model decides how many factors
> exist", "memory of every risk ever observed", "equalize factor risk") were turned into mechanisms without a declared
> target quantity, a scoring rule for that quantity, or a pre-registered way to be proven wrong. Each layer therefore
> optimized a proxy (reconstruction, active units, entropy of a normalized sub-block, Sharpe), and the evaluation layer
> was built so that it cannot return "the theory failed".

What is worth keeping is substantial and precise (§4): the μ=0 separation of risk and return views, price-only
discovery with shape/scale separation and dual rescaling, the principal-factor-basis risk contributions with exact
gradients, random-matrix signal/noise separation, the overlap-aware effective-sample-size guard, the walk-forward
skeleton, the benchmark ladder with a PCA null and a kill criterion, and the invariant/contract machinery.

The redesign (§5) rebuilds the strategy as **five layers with one shared estimand — the out-of-sample covariance of
returns at the rebalancing horizon — and one shared scoring rule**. The VAE is not deleted by decree: it is demoted to
one *candidate* estimator in the factor layer, reformulated as a cross-sectional model, and kept only if it beats the
linear baseline on the shared score (Principle 10, finally made operational). The migration (§7) starts with the
foundations (data contract, single harness, evaluation endpoint) before any modelling, because today no result can be
attributed to the model.

**What the evidence says today** (only real run, one OOS block of 6.25 years, pretrained checkpoint):

| Strategy | Sharpe | Vol | Max DD | Eff. positions |
|---|---|---|---|---|
| VAE strategy | 0.38 | 16.4% | 35.3% | 50 |
| 1/N (and ERC, byte-identical: ERC fell back to 1/N) | 0.49 | 19.7% | 37.8% | 477 |
| Inverse vol | 0.50 | 19.1% | 37.5% | 420 |
| Min variance (LW) | 0.20 | 14.2% | 26.3% | 42 |
| PCA factor RP | 0.30 | 15.1% | 27.3% | 41 |
| S&P 500 | 0.66 | 20.6% | 33.9% | 500 |

The risk model under-forecast realized variance by **2.74×** (health check CRITICAL), PC1 carries **81%** of factor
risk while being excluded from the objective, and the pipeline's decision engine nevertheless emitted "Scenario C —
adopt PCA" from **zero** valid comparisons. None of these numbers can confirm or refute the theory; they show that the
current system cannot test it.

---

## 2. Method

1. 17 units covering every principle, hypothesis, model, data choice, pipeline stage, evaluation and conclusion.
2. Per unit, two blind agents with the same briefing: an **advocate** (what is correct, graded solid / conditional /
   fragile) and a **devil's advocate** (errors, inconsistencies, unjustified choices, dishonest framing; severity
   critical / major / minor; root causes with structural alternatives). Five mandatory questions per report, including
   "what should be removed". Evidence register per finding (doc / code / results / hypothesis).
3. Deterministic validation (sections, score, citation resolution). After expanding the agents' abbreviations
   (`DVT:`, `ISD:`, `philosophy.md:`), every report resolves ≥ 80% of its `path:line` citations to an existing file and
   line range, except the U12 advocate (71%), whose misses are elided file names (`DVT_...v4_1.md`), not invented
   lines. Resolution proves the cited line exists, not that it supports the claim — hence step 4.
4. Orchestrator verification of the load-bearing critical findings at the cited lines (§9.1): 15 checked, 13 confirmed,
   1 confirmed with nuance, 1 downgraded.
5. Synthesis by **root cause across units**, not unit by unit.

Limits: one model family wrote all 34 reports (role diversity, not error independence); critics' scores (2–3/10) and
advocates' (4–7/10) are opinions, not measurements; the only real run is a single path.

---

## 3. Root-cause map

Each root cause is supported by several units independently. Severity counts refer to critic findings.

### RC-A — Estimand mismatch: a per-stock shape model asked to do a cross-sectional covariance job
*Units: U02, U04, U05, U06, U08 (+U09).*
The portfolio needs Σ — a property of the **joint** law of returns. The VAE encodes one stock's own z-scored window and
reconstructs that same window (`src/vae/model.py` forward: `encoder(x) → z → decoder(z)`); there is no shared factor
path f_t and no cross-stock term. Exposures B are a function of the **marginal** law of each stock, then regressed on
returns post hoc. Every loss add-on (co-movement Spearman target, cross-sectional R² loss, crisis γ, feature weights)
exists to inject the missing cross-sectional structure into a model class that cannot represent it.
**Consequences:** low and unstable explanatory power (EP_is 0.22), factor quality incidental to training, loss knobs
that accumulate and cannot be retired.

### RC-B — Identification mismatch: "active units" are not "number of risk factors"
*Units: U05, U08, U09, U15.*
AU measures how many latent coordinates the encoder uses — a generative-modelling diagnostic. In the real run 47 of 52
dimensions exceed the 0.01-nat threshold with a smooth KL spectrum (no elbow), K is set to `2·AU_max_stat`, and AU is
then truncated to `AU_max_stat = 26` (`src/inference/active_units.py:92-117`, `src/integration/pipeline.py:307-314`).
The factor count is produced by caps, not by the model; coordinate-wise selection is applied to a rotationally
unidentified representation. **"The model decides how many factors exist" is false as implemented.**

### RC-C — Time mismatch: "memory of dormant risks" implemented as static averaging in inconsistent units
*Units: U01, U04, U05, U08, U09, U11, U15.*
The doctrine promises full-history memory. The code collapses each stock's trajectory into one static vector with an
exponential half-life of 60 **windows** — documented as "1260 days at stride 21" — applied to **stride-1** inference
windows, i.e. 60 trading days (`src/inference/composite.py` docstring vs `src/integration/pipeline.py:1975-1980`).
Σ_z uses half-life 0 in config, 252 in the real run, while a dedicated document argues it must be 0. B and α are frozen
per fold while Σ is refreshed at every rebalance: state and policy are decoupled. The memory claim is neither
implemented nor testable.

### RC-D — Risk-functional mismatch: a normalized, second-moment objective that cannot express the stated goal
*Units: U01, U09, U10, U11, U15.*
- The entropy is computed on **normalized sub-blocks** (signal factors excluding PC1; a separate idiosyncratic block
  blended with weight 0.05/0.20/0.0 depending on the source), so it is invariant to how much of total risk each block
  carries. PC1 (81% of factor risk) is removed and no budget replaces it.
- The objective `−λ w'Σw + α H(w)` mixes return² with nats; the code repairs units with a data-dependent
  `α_eff` (`sca_solver.py:578-633`) and the cardinality stage then optimizes with nominal α — a different objective.
- Uniform budgets over weakly identified eigen-directions give the largest exposures to the noisiest directions.
- Covariance cannot represent the tail/contagion risk that motivates the strategy (DVT §2.3), and per-window
  z-scoring removes the scale information that tails live in.
- The risk model under-forecasts variance 2.74× in the only real run.

### RC-E — Data-contract mismatch: a CRSP-shaped specification fed with a vendor snapshot
*Units: U02, U03, U06, U13.*
The spec assumes CRSP (float-adjusted cap, share codes, delisting returns, point-in-time universe, VIX). Reality:
Tiingo adjusted OHLCV; `delisting_return` is NaN for every row (`scripts/download_tiingo.py:911`) and the imputation
only fires on non-NaN values (`src/data_pipeline/returns.py:186-198`); "market cap" is a turnover proxy; the
point-in-time universe module has **no production caller**; the production universe is the top-n by **full-sample
median** proxy cap (`src/data_pipeline/data_loader.py:344-348`), a look-ahead/survivorship filter; back-adjusted prices
make historical inclusion depend on future splits; every entry point passes `vix_data=None`, so crisis weighting is a
no-op. The tail-risk claims of the theory have no data to stand on.

### RC-F — Evidence mismatch: an evaluation layer that cannot fail the theory
*Units: U12, U13, U14, U16 (+U01, U09).*
- The decision rule is Sharpe while the theory is about risk diversification.
- One backtest path is treated as 34 samples; the paired test cannot reject at all with 9 pairs and 18 Holm-corrected
  tests (minimum Wilcoxon p 0.0039 > 0.05/18).
- Phase A selects configurations on the same OOS block that is then reported; its score terms are placeholders.
- Benchmarks run in a different harness (constant mix, free daily rebalancing, no delistings, no costs) from the VAE
  (drift, costs, quarterly rebalancing); ERC silently falls back to 1/N and is byte-identical to it in the real run.
- The decision engine has no "inconclusive" state: with zero valid comparisons it emits "Scenario C — adopt PCA"
  (`src/walk_forward/selection.py:112-136`, verified).
- Health checks pass self-referential metrics (H_norm 0.399 "OK", EP 0.36 "OK" against a 0.60 spec target).

### RC-G — Governance mismatch: four parameter sources, prose specification, unattributable changes
*Units: U07, U15, U16, U17 (+U03, U04).*
Parameters live in DVT, ISD, `config.py`, the notebook and the run config, and disagree (K 200/75/50/52; λ_co, LR,
patience, shrinkage method). Divergences are legitimized by being listed. The V2 post-mortem changed eight things at
once, the V1/V2 artifacts are lost, the surviving run uses a pretrained checkpoint of unknown provenance. No number can
be traced to a (data, code, config, checkpoint) tuple.

### How the causes chain
```
RC-G governance ──► no provenance, no single spec ──► every other cause can persist unnoticed
RC-E data contract ─► no PIT universe / delistings / VIX ─► RC-D tail claims unsupported, RC-F biased comparison
RC-A estimand ──────► B incidental to training ──► RC-B factor count from caps ──► RC-D objective on noisy directions
RC-C time ──────────► static B / frozen α / refreshed Σ ──► RC-D under-forecast risk (var ratio 2.74)
RC-F evidence ──────► cannot reject ──► patches accepted or reverted on one aggregate number (V1→V2→Phase 0)
```

---

## 4. What must be preserved (advocate consensus, cross-checked)

| # | Strength | Why it survives the redesign | Evidence |
|---|---|---|---|
| P1 | **μ = 0 separation**: return views enter as one additive term, risk model independent | Makes the risk model independently testable; prevents return signals from contaminating factors | U01, U11, U15 advocates |
| P2 | **Price-only discovery, no fundamental input** | A structural, verifiable "no imposed taxonomy" guarantee; fundamentals only for a posteriori interpretation | U02 |
| P3 | **Shape/scale separation with dual rescaling** (σ_i,t/σ̄_t at estimation dates vs current) | Correct handling of heteroskedasticity across time; keep as a feature of any exposure model | U02, U04, U05, U09 |
| P4 | **Principal-factor-basis risk contributions** c'_k = λ_k β'_k² with exact analytic gradient | Exact decomposition, rotation-resolving, tested against finite differences | U10, U11 |
| P5 | **Random-matrix signal/noise separation and standalone-shrinker discipline** (DGJ/LW, no chaining) | Prevents equalizing risk across noise; keeps Σ well-conditioned | U09, U15 |
| P6 | **Overlap-aware effective sample size** (≈15 independent windows per stock over 30 years) | The honest capacity bound; the cap belongs to the data, not to the encoder | U04, U08, U15 |
| P7 | **Walk-forward skeleton**: expanding window, contiguous OOS blocks, 21-day embargo, untouched holdout | Literature-standard; the problem is how it was used, not the skeleton | U01, U12 |
| P8 | **Benchmark ladder with a PCA null and the kill criterion** ("complexity must earn its place") | The right governing principle; must become executable | U13, U15, U16 |
| P9 | **Invariant/contract system** (INV triples, CONV conventions, executable interface tests) | Turns theory into checks; must be re-pointed at the new estimand | U17 |
| P10 | **Leak-free mechanics already built**: causal features, within-window standardization, PIT crisis labelling, deterministic inference, profiled-σ² ELBO, Holm step-down decision logic | Correct, tested; reuse as components | U03, U04, U06, U07, U08, U14 |

---

## 5. Target architecture

One estimand, one score, five layers with explicit contracts. Every layer is replaceable; none may change the
estimand or the score.

```
L0 Data contract ──► L1 Risk estimand & score ◄── judges ── L2 Factor estimators (PCA baseline | candidates)
                              │                                      │
                              ▼                                      ▼
                      L4 Evaluation & decision ◄── portfolios ── L3 Risk-budgeting policy
                              ▲
                      L5 Governance (provenance, single config, executable spec, change protocol)
```

### L0 — Data contract (replaces the CRSP-shaped assumptions)
- **Output object:** a versioned panel artifact with a content hash: daily returns from *unadjusted* prices with
  point-in-time corporate-action factors; a universe **time series** U_t (eligibility evaluated with information ≤ t:
  listing age, trailing ADV, price, liveness); explicit delisting events; a stress/regime indicator series.
- **Rule:** if a field the theory relies on is unavailable (delisting returns with Tiingo), the contract states the
  bias and every result carries a **sensitivity bound** (e.g. −30%/−55% Shumway imputation on all disappearances vs
  none). No silent NaN→0, no full-sample selection.
- **Reuse:** `src/data_pipeline/universe.py` (correct PIT design, currently orphaned), `returns.py`, `crisis.py`.

### L1 — Risk estimand and scoring rule (new; the keystone)
- **Estimand:** Σ_{t→t+h}, the covariance of returns over the next rebalancing horizon h (21 or 63 trading days),
  for the universe U_t.
- **Score:** a proper, robust loss on portfolio variance forecasts — QLIKE (Patton 2011) and the bias ratio
  realized/predicted — evaluated on (i) random long-only portfolios, (ii) the optimized portfolios of L3, (iii) stress
  sub-periods separately. Optional tail extension: realized vs forecast CVaR on stress windows.
- **Contract:** every factor estimator, every hyperparameter, every "memory" choice is judged by this score in an inner
  validation that never touches the reported OOS. Acceptance band for deployment: bias ratio in [0.8, 1.2] across
  folds (the repo's own target, currently violated at 2.74).

### L2 — Factor estimators as plug-ins: `(data up to t) → (B_t, Σ_f,t, D_t)`
- **Baseline (must exist first):** statistical PCA on the PIT panel with RMT signal count and standalone shrinkage
  (already built for the PCA benchmark), plus dual rescaling (P3).
- **Time handled explicitly, not by averaging:** exposures conditional on information at t; factor covariance as a
  **declared blend** of a recent-regime estimate and a stressed-history estimate, with the blend weight chosen by the
  L1 score in stress and calm sub-periods. "Memory of dormant risks" becomes a falsifiable hypothesis (H4 below).
- **Nonlinear candidate (replaces the current VAE):** a *cross-sectional* conditional autoencoder in the spirit of
  Gu, Kelly & Xiu (2021): a beta network maps each stock's price-derived characteristics at t (including the current
  shape encoder as a feature extractor — P2/P3 preserved) to β_{i,t}; a factor network maps the cross-section of
  returns at t to f_t; the decoder reconstructs **the cross-section of returns** r_t ≈ B_t f_t. The training target is
  then the object the risk model needs. IPCA (Kelly, Pruitt & Su 2019) is the linear counterpart and the natural
  matched control. RP-PCA (Lettau & Pelger 2020) is an optional second linear control.
- **Factor count:** chosen by the L1 score out of sample (with the overlap-aware capacity bound P6 as a hard ceiling),
  never by active units. Interpretation and budgets live in the principal basis of Σ_f (P4) — rotation is resolved
  there, not by coordinate-wise KL.

### L3 — Risk-budgeting policy (replaces the normalized-entropy objective)
- **One objective in consistent units:** total portfolio risk decomposed into explicit components — market factor,
  each signal factor, the noise bulk as one block, idiosyncratic risk — each with an **explicit budget** b
  (Roncalli 2013 risk budgeting; Meucci 2009 effective number of bets as a *reported* diagnostic, not the target).
  The market factor gets a declared budget instead of silent exclusion.
- Default budgets: equal across signal factors (the original intent), with the noise bulk and idiosyncratic blocks at
  declared fractions; alternatives (eigenvalue-proportional, downside-weighted) are separate, named policies compared
  in L4 — never a mode switch inside one policy.
- No α_eff reparameterization, no entropy-vs-variance frontier calibrated on the object being judged: the trade-off,
  if any, is a single declared target volatility or tracking constraint.
- **Constraints in one place:** long-only, caps, turnover and cardinality as hard constraints of the solver used in
  every rebalance (the default fast path currently has no turnover constraint and cardinality is applied once per
  fold). State and policy are refreshed together at each rebalance.
- **Reuse:** P4 entropy/contribution maths and gradients, SCA + Armijo + multi-start, projection operators.

### L4 — Evaluation and decision (rebuilt around pre-registration)
- **One harness for every strategy:** the same simulator (drift, transaction costs, delistings, rebalancing calendar)
  for the candidate and all benchmarks. A failed solve is recorded as a failure, never replaced by 1/N.
- **Primary endpoints aligned with the theory, measured with a common external yardstick:** (1) L1 risk-forecast score;
  (2) realized diversification: realized volatility, CVaR and drawdown in stress periods, and risk concentration
  measured with **one common reference risk model** for all strategies (not each model's own Σ). Sharpe, turnover and
  costs are secondary.
- **Statistics for a single path:** loss-differential tests on daily/weekly series (Diebold–Mariano with HAC errors)
  and stationary block bootstrap, instead of Wilcoxon on 9–34 pseudo-replicate folds; combinatorial purged CV
  (López de Prado 2018) for model selection; deflated Sharpe (Bailey & López de Prado 2014) when Sharpe is reported.
  A power calculation precedes every comparison.
- **Decision states:** adopt / reject / **inconclusive** / invalid run. Inconclusive is the default when evidence is
  insufficient. The final holdout is opened once, after the decision rule is frozen.
- **Matched-control ablations instead of a narrative ladder:** factorial design model × input features × policy, so
  every Δ is attributable (PCA vs IPCA vs conditional AE at fixed features and policy; policies compared at fixed risk
  model).

### L5 — Governance
- **Single configuration source** (frozen dataclasses) with a content hash; the notebook only selects named configs.
- **Run manifest** for every result: data hash, code commit, config hash, seed, checkpoint hash. A number without a
  manifest is not evidence.
- **Executable specification:** each spec statement has an owner test (INV/CONV machinery P9 re-pointed at L0–L4);
  a divergence requires an ADR with a decision, not a list entry.
- **Change protocol:** one change per experiment, with its pre-registered hypothesis and kill criterion; bundles of
  changes are forbidden as evidence.
- DVT v4.1 and ISD become **history**; a short v5 specification (this document's §5–§8) becomes the normative source.

---

## 6. What is removed, merged or moved

| Removed / merged | Why | Replaced by |
|---|---|---|
| AU as factor count; AU caps cascade (K = 2·AU_max_stat, truncation) | Capacity ≠ factors; circular by construction | L2 factor count by L1 score, P6 ceiling |
| Aggregation half-life over window stacks | 21× unit error; destroys the trajectory it claims to remember | Conditional exposures B_t; declared memory blend in Σ_f |
| Co-movement scaffolding, cross-sectional R² loss, crisis γ as loss patches | Patches for RC-A; γ is a no-op (no VIX) | Cross-sectional objective in the L2 candidate; stress handled in L1 scoring |
| Normalized two-layer entropy, idio blend weight, PC1 exclusion, uniform/proportional mode switch | Unit-incoherent, invariant to block shares, ignores 81% of risk | L3 explicit budgets over total risk |
| α_eff rescaling and α frontier calibration | Patches dimensional incoherence; nominal/effective α mismatch | Single declared risk target |
| Phase A placeholder scoring, composite diagnostic scores without performance terms, scenario engine with default branch | Self-referential, cannot fail | L4 pre-registered decision states |
| ERC/solver fallbacks to 1/N, NaN→0 metrics, skipped tests that still emit a verdict | Silent degradation | Recorded failures, "invalid run" state |
| Separate benchmark harness | Asymmetric comparison | One L4 harness |
| Production universe = full-sample top-n | Look-ahead and survivorship | L0 U_t time series |
| Multiple entropy normalizations (ln AU, ln K, ln n_signal), several EP thresholds, two Sharpe formulas | Every construct had several definitions | One definition per metric, in L4 |
| Parameters duplicated in DVT/ISD/config/notebook | No provenance | L5 single config + manifest |

---

## 7. Migration path (each phase has a gate; failing a gate stops the next phase)

| Phase | Content | Gate (pre-registered) |
|---|---|---|
| **0 — Foundations** | L0 data contract (PIT universe wired, delisting sensitivity, unadjusted prices + factors, regime series); L4 single harness with costs/delistings; L5 config + manifest; decision states | The 6 benchmarks + S&P run end-to-end in the single harness with manifests; ERC solves (no 1/N fallback) on every fold |
| **1 — Risk baseline** | L1 score implemented; PCA/RMT baseline + dual rescaling + declared memory blend | Bias ratio in [0.8, 1.2] and QLIKE better than sample covariance and LW on random portfolios in ≥ 75% of folds. Failing → the problem is risk estimation itself; no portfolio work |
| **2 — Policy** | L3 risk budgeting on the Phase 1 risk model | Beats ERC and 1/N on stress CVaR / drawdown at matched volatility (DM test, block bootstrap), turnover within budget. Failing → the diversification policy is the research question, not the factor model |
| **3 — Nonlinear candidate** | Cross-sectional conditional AE vs IPCA vs PCA at fixed features and policy | Kill criterion (Principle 10): adopt only if the L1 score improves significantly OOS **and** the Phase 2 endpoint does not degrade. Otherwise stop the VAE line and keep the linear model |
| **4 — Extensions** | Tail-dependence module, downside budgets, return views through the μ term | Each as a single pre-registered change |
| **Holdout** | Open the 3-year holdout once with the frozen pipeline | Report whatever it says |

Phase 0 changes no model and is the precondition for any conclusion. The current VAE code is not deleted during
Phases 0–2; it is simply not on the critical path.

---

## 8. Falsifiable hypotheses and kill criteria

| ID | Hypothesis | Test | Kill / consequence |
|---|---|---|---|
| H0 | Conclusions are robust to the missing delisting data | Results under Shumway imputation vs none | If the ranking flips, no conclusion is reported without CRSP-grade data |
| H1 | A price-only factor model forecasts portfolio risk without bias | L1 bias ratio, QLIKE vs sample/LW covariance | Outside [0.8, 1.2]: fix L1/L2 before anything downstream |
| H2 | Factor risk budgeting improves stress outcomes vs naive diversification | Stress CVaR/drawdown vs ERC, 1/N at matched vol; DM + block bootstrap | Not better: drop the factor-parity claim, keep the risk model |
| H3 | A nonlinear cross-sectional model beats its linear counterpart | L1 score vs IPCA/PCA at fixed features and policy | Not better: Principle 10, adopt the linear model |
| H4 | Stressed-history memory improves risk forecasts in stress without degrading calm periods | L1 by regime for blend weight > 0 vs = 0 | Not better: drop the "dormant risk" doctrine |
| H5 | Price-only discovery suffices | Add a fundamentals-based control (sector/style exposures) as a benchmark, not as an input | If it dominates, revisit P2 explicitly |

---

## 9. Appendix

### 9.1 Orchestrator verification of load-bearing critical findings

| Finding | Claim (short) | Verified at | Verdict |
|---|---|---|---|
| U05-CRI-01 | VAE reconstructs each stock's own window; no shared factor | `src/vae/model.py` forward | Confirmed |
| U03-CRI-01 | PIT universe module has no production caller | grep over `src/`, `scripts/`, notebooks: 0 callers | Confirmed |
| U03-CRI-02 | Production universe = top-n by full-sample median proxy cap | `src/data_pipeline/data_loader.py:344-348` | Confirmed |
| U03-CRI-04 | Delisting returns NaN everywhere; imputation never fires | `scripts/download_tiingo.py:911`, `returns.py:186-198` | Confirmed |
| U06-CRI-01 / U16-CRI-04 | Crisis weighting is a no-op (`vix_data=None` in every entry point) | `run_walk_forward.py:203`, `run_diagnostic.py:444`, notebook ×2, `pipeline.py:1449-1455` | Confirmed |
| U01-CRI-03 / U08-CRI-04 / U15-CRI-04 | Aggregation half-life 60 windows applied at stride 1 = 60 days, documented as 1260 | `composite.py` docstring, `pipeline.py:1975-1995`, run config | Confirmed |
| U13-CRI-01 | ERC identical to 1/N in the real run | `cmp` of both CSVs: identical; `erc.py:296-305` fallback | Confirmed |
| U14-CRI-02 / U16-CRI-01 / U12-CRI-04 | Scenario C emitted from zero valid comparisons | `selection.py:112-136` (`n<5 → continue`, `wins<=1 → C`) | Confirmed |
| U14-CRI-04 | Paired test cannot reject (min Wilcoxon p with 9 pairs 0.0039 > 0.05/18) | Arithmetic | Confirmed |
| U14-CRI-05 | Holm correction not applied | `statistical_tests.py`: step-down `reject` logic is correct; only the reported p is raw | **Downgraded to major** (reporting defect) |
| U09-CRI-01 | Realized/predicted variance 2.74 (CRITICAL) | `results/diagnostic/health_checks.csv` | Confirmed |
| U09-CRI-04 | Off-by-one in the signal slice when PC1 is skipped | `pipeline.py:2296-2303` | Confirmed if `n_signal` counts PC1 (DGJ spike count does) |
| U11-CRI-05 | Default fast sub-solver has no turnover constraint | `tau_max` only in the CVXPY builder; `use_fast_subproblem=True` | Confirmed |
| U12-CRI-03 | The real-data diagnostic bypasses walk-forward (run_direct, pretrained, one block) | `run_diagnostic.py:436-455` | Confirmed |
| U10-CRI-01 | PC1 excluded from H while carrying 81% of factor risk | `pipeline.py:2290-2303`, diagnostic data | Confirmed |

The remaining 63 critical findings were not individually re-verified; they are consistent with the verified ones and
with each other across units, and each is listed with its citations in the unit reports.

### 9.2 Root cause → units → critical findings

| Root cause | Units | Representative critical findings |
|---|---|---|
| RC-A estimand | U02 U04 U05 U06 U08 | U05-CRI-01/02, U04-CRI-01/03, U06-CRI-01, U08-CRI-05 |
| RC-B identification | U05 U08 U09 U15 | U05-CRI-03/04, U08-CRI-01/02/03, U09-CRI-03, U17-CRI-03 |
| RC-C time | U01 U04 U08 U09 U11 U15 | U01-CRI-03/04, U08-CRI-04, U15-CRI-04, U04-CRI-03 |
| RC-D risk functional | U01 U09 U10 U11 U15 | U01-CRI-02/06, U09-CRI-01/04, U10-CRI-01..07, U11-CRI-01..07, U15-CRI-02/03/06, U17-CRI-02 |
| RC-E data contract | U02 U03 U06 U13 | U03-CRI-01..05, U06-CRI-01, U16-CRI-04 |
| RC-F evidence | U12 U13 U14 U16 | U12-CRI-01..04, U13-CRI-01..05, U14-CRI-01..09, U16-CRI-01/02/05 |
| RC-G governance | U07 U15 U16 U17 | U07-CRI-01..03, U15-CRI-01/05, U16-CRI-03, U17-CRI-01/04 |

Every critical finding maps to at least one root cause above; none requires a fix outside the redesign.

### 9.3 References
- Gu, S., Kelly, B., Xiu, D. (2021). Autoencoder asset pricing models. *Journal of Econometrics*.
- Kelly, B., Pruitt, S., Su, Y. (2019). Characteristics are covariances. *Journal of Financial Economics*.
- Lettau, M., Pelger, M. (2020). Factors that fit the time series and cross-section of stock returns. *Review of Financial Studies*.
- Ledoit, O., Wolf, M. (2020). Analytical nonlinear shrinkage of large-dimensional covariance matrices. *Annals of Statistics*.
- Donoho, D., Gavish, M., Johnstone, I. (2018). Optimal shrinkage of eigenvalues in the spiked covariance model. *Annals of Statistics*.
- Patton, A. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*.
- Diebold, F., Mariano, R. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*.
- Meucci, A. (2009). Managing diversification. *Risk*.
- Roncalli, T. (2013). *Introduction to Risk Parity and Budgeting*. CRC Press.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Bailey, D., López de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management*.
- Shumway, T. (1997). The delisting bias in CRSP data. *Journal of Finance*.
