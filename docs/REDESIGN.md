# Latent Risk Factor Discovery — Redesign of record

> **Status:** proposal, nothing implemented. **Date:** 2026-11-08 (revised the same day, see below).
> **Supersedes as normative proposal:** [audit_v5/REDESIGN.md](audit_v5/REDESIGN.md) (code + results audit) and
> [audit_theory/REDESIGN.md](audit_theory/REDESIGN.md) (theory-only audit). Both remain the evidence base, together
> with [architecture_survey.md](architecture_survey.md) (literature survey, written blind to this document) and
> [redesign_review.md](redesign_review.md) (independent review of the first version of this document).
> **Method:** independent arbitration. Every claim kept as load-bearing was re-checked at the cited spec or code line, or
> re-derived; what could not be checked is marked. Literature cited from memory is marked [unverified].
> **Binding user decision (not discussed here):** the strategy stays a deep-learning (DL) risk model. Linear and
> shrinkage models (PCA, IPCA, Ledoit-Wolf, RMT, DCC, 1/N, ERC, inverse-vol) are baselines, controls and stopping
> criteria, never the destination. The owner's further decisions of 2026-11-08 are in §0.2.

---

## 0. Revision and owner decisions

### 0.1 Revision 2026-11-08

The first version of this document (same date) was reviewed independently ([redesign_review.md](redesign_review.md)),
using the literature survey ([architecture_survey.md](architecture_survey.md)) as evidence and checking it in turn.
The owner then decided the open points (§0.2). What changed (details: review §2-§3):

| Area | Change | Why |
|---|---|---|
| L1 | MLP on 13 ranked characteristics replaced by a three-block conditional probabilistic factor model (market, structural, conditional block from characteristics + path encoder + FiLM state; ≈ 27k parameters) with a matched linear twin | Ranked characteristics cannot carry industry-like structure or regimes; GKX nonlinear loadings do not beat IPCA on total R² |
| L2 | Cholesky head → state-modulated factor DCC; deep volatility head and neural spectral cleaner added; tail head merged into the E7 VAE | Lower variance; strongest price-only deep evidence |
| VAE | Latents = date-level factor returns; exact Gaussian case E2, variational case E7 | OD3 |
| E1 | QIS, DCC-NL, AFM-DCC, IPCA-LS vs IPCA-NLL, IPCA + residual PCA move into E1 | All bars fixed before any DL test |
| E2, D8 | Factorial {twin, deep} × {no state, state} instead of × {LS, NLL}; adoption = beats twin *and* in the MCS vs the E1 bar | The open question is state and nonlinearity, not the loss |
| §6 | H1b split (H1b/c/d); new H1ℓ, H2v, H2n, H2r, H3m; H0 restricted; TOST margin in nats per stock-day; 2% and +5% margins replaced; Sharpe SE 0.18 / 0.105 | Non-identifiable or underpowered gates |
| D1, L3 | Amplification note; ENB floor relative to the feasible maximum; P6 | Review; OD4 |
| L4 | Log-score decomposition, seed ensembles, first-occurrence crisis tests, Sharpe bootstrap | Review |
| D10 | SIC codes as baseline/diagnostic | OD5 |
| Facts | `results/` is tracked (was reported deleted); C16 line updated; references re-graded | `git ls-files results`; review |

Unchanged: the diagnosis (R1-R8), the arithmetic (C1-C22, D7, power and sign tests), the asset-space measure (D1),
the data contract (L0, D9), the evaluation core (L4) and governance (L5).

### 0.2 Owner decisions (2026-11-08)

| ID | Question | Decision | Status |
|---|---|---|---|
| OD1 | What L1 is if every deep L1 rung (F1a, F1b, F2) fails | Left open on purpose, decided on the results (below) | **Open** |
| OD2 | Fixed statistical block inside the deep model | Both acceptable, as two named arms: fixed residual-PCA block (E2), learned cross-stock block (E3, tested as a replacement; not a fallback) | Decided |
| OD3 | Scope of the VAE | Probabilistic latent factor model, latents = date-level factor returns (FactorVAE / NeuralFactors form): exact Gaussian (E2), variational with regimes and Student-t tails (E7) | Decided |
| OD4 | Policy-level memory | P6 (GMV on a declared calm/stress blend, π fixed ex ante) is a named arm | Decided |
| OD5 | Industry information | Point-in-time SEC SIC codes as baseline and diagnostic now; as an input in one named arm only if E2 and E3 fail | Decided |
| OD6 | Compute | Five-seed ensembles × annual refits run on Colab; each formulation capped at 8 inner configurations | Decided |
| OD7 | Pre-registered margins | δ = 0.002 nats per stock-day (H1a), the rule that sets the H3a margin from the SE measured in E1, the P6 π, the H3m calm-cost budget and the ENB fractions e_rel are recorded in `docs/implementation_decisions.md` before E1 runs | Decided (to be recorded) |

**OD1, the three options and the evidence each needs.** None is pre-selected. In every option linear models stay
baselines; none makes a linear model the strategy.
- **(a) Linear L1 in operation, DL in L2, research changes the information set.** Needs: the deep L1 rungs *failed*
  their gates with adequate power (measured MDE, not "inconclusive"); at least one deep L2 arm (H2, H2v, H2n) passed
  and is in the MCS against the E1 bar; the full system passes H4. The research line then continues on L1 with a new
  information set (SIC codes per OD5, intraday data).
- **(b) No deployment; research only, with a new information set.** Needs: every deep rung in L1 *and* L2 failed or
  was inconclusive, or H4 failed. The next information set and its pre-registered ladder are defined before any
  further run.
- **(c) Keep a deep L1 on non-inferiority grounds.** Needs: a TOST non-inferiority margin for deep vs twin and vs the
  E1 bar pre-registered before E2, MCS membership, and a benefit fixed in advance that the twin lacks (stress-day
  calibration, the first-occurrence crisis test). Without that pre-registration, (c) would contradict "evidence
  before narrative".

---

## 1. Executive summary

**The shared diagnosis.** Both audits, from disjoint evidence, and the survey independently, reach the same
conclusion: *the network is trained to compress each stock's own z-scored return window, while the portfolio needs a
forecast of the joint, time-varying covariance of returns.* Every later stage (lifetime averaging into a static B,
active units as factor count, entropy in latent coordinates, patched units, a decision engine that cannot say
"inconclusive") is a heuristic bridging that gap, and the evaluation cannot show the theory is wrong.

**What the evidence actually shows today.** One real run (Tiingo, 788 stocks, one 1,575-day OOS block, pretrained
checkpoint): VAE Sharpe 0.38 vs 1/N 0.49; realized variance 2.74× the model forecast; one principal factor carries 81%
of systematic risk and is excluded from the objective; the engine printed "Scenario C — Adopt PCA" from zero valid
comparisons (all verified in `results/diagnostic/`, tracked in git, see §2). Under a stated correlation assumption the
0.11 Sharpe gap is about 0.6 standard errors (§6.1), so "the VAE loses" is not statistically established either. The
system cannot currently test its own theory in either direction.

**What changes.**
1. **The learned object** becomes a *conditional probabilistic factor model*: B_t = [market | structural |
   conditional] loadings, the conditional block produced by a small shared network from point-in-time
   characteristics, a causal path encoder and an unranked market-state vector, trained on the **out-of-sample
   cross-sectional predictive likelihood** of next-day returns. Its latents are date-level factor returns: this is
   where the VAE idea lives (§5, "Where the VAE lives"). The second DL layer is the factor covariance and
   idiosyncratic volatility (L2).
2. **Every downstream quantity is computed in asset space** (Σ_t = B_tΣ_f,tB_tᵀ + D_t). No latent coordinate, active
   unit or latent eigenbasis reaches a decision.
3. **Diversification becomes a tested hypothesis** (a constraint on an invariant measure inside minimum variance),
   not an additive reward assumed to pay.
4. **Evaluation is rebuilt** on proper scoring rules of daily covariance forecasts, with nested selection, one
   harness, a matched linear twin for every deep arm, a family-wise test over every variant ever tried, an
   "inconclusive" state, and the holdout opened once.

**The decisive next experiment (E2, §7).** After the foundations (E0) and the full linear bar (E1, including
DCC-NL, QIS and AFM-DCC), train a small conditional probabilistic factor model (path encoder + characteristics,
with and without FiLM state conditioning, ≈ 27k parameters) and its matched linear twin (IPCA with state-interacted
instruments, same NLL) on the same inputs, structural block, Σ_f and D_t construction and nested protocol, and
compare daily OOS Gaussian log-scores. **Gate:** deep beats its twin (one-sided DM-HAC, SPA over the cell's variants)
with calibration in [0.8, 1.2] and no significant degradation of realized GMV variance; adoption also requires
membership of the Model Confidence Set against the E1 bar. **If it fails:** the learned structural block (E3) and the
L2 ladder (E4, including the neural spectral cleaner) are tested under the same rule; the plan never adopts a linear
model as the strategy, and what L1 becomes if every deep L1 rung fails is owner decision OD1 (open).

---

## 2. What was checked, and what it showed

The table lists the load-bearing claims re-checked for the first version; the review re-opened C3, C6, C7, C8, C16,
C20 and C22 and confirmed them. **`results/` (including the only real-run artifacts and both checkpoints) is tracked in
git and present in the working tree as of 2026-11-08; L5 archives it with a manifest.**

| # | Claim | Source audit | Checked at | Verdict |
|---|---|---|---|---|
| C1 | Real-run numbers (Sharpe 0.38 / 0.49 / 0.50 / 0.20 / 0.30 / 0.66; ERC row byte-identical to 1/N) | v5 | `results/diagnostic/strategy_comparison.csv` | Confirmed |
| C2 | var ratio 2.742, AU = 26 / K = 52, EP_is 0.219, EP_oos 0.356, n_signal 12, top-1 risk share 0.811, OOS 1,575 dates | v5 | `results/diagnostic/health_checks.csv`, `diagnostic_data.json` | Confirmed |
| C3 | "Scenario C — Adopt PCA" emitted with 0 comparisons | v5 | `results/diagnostic/report.txt`; `src/walk_forward/selection.py:110-136` (`n<5 → continue`, default B, `wins<=1 → C`) | Confirmed (twice) |
| C4 | VAE reconstructs each stock's own window; no shared factor path | both | `src/vae/model.py:113-138` | Confirmed |
| C5 | Aggregation half-life 60 counted in windows, applied to stride-1 inference windows (≈ 60 days, documented as ≈ 1,260) | v5 | `src/integration/pipeline.py:1976-1995`, `src/inference/composite.py:122-165`, `src/config.py:355-357`, run config `aggregation_half_life: 60` | Confirmed |
| C6 | Principal factors taken from eig(Σ_z) in latent space, B′ = B V | both | `src/risk_model/covariance.py:605-622` | Confirmed (twice) |
| C7 | H over principal factors of Σ_z changes under a latent column rescaling that leaves Σ_assets unchanged | theory (U01-CRI-02) | Numerically (n = 30, k = 3, S = diag(1, 3, 0.2)): Σ_assets identical; H_latent 0.421 → 0.259; asset-space H 0.977 → 0.977 (review, non-orthogonal map: same result) | Confirmed; asset-space version invariant |
| C8 | PC1 excluded from the entropy, off-by-one in the signal slice | v5 | `src/integration/pipeline.py:2286-2303` (`signal_end = n_signal + 1` when PC1 skipped) | Confirmed (twice) |
| C9 | `vix_data=None` in every entry point | both (v5 code, theory spec) | `scripts/run_diagnostic.py:444`, `scripts/run_walk_forward.py:203`, `notebooks/dashboard.ipynb:1151,1217` | Confirmed |
| C10 | Production universe = top-n by full-sample median "market cap" | v5 | `src/data_pipeline/data_loader.py:344-348` | Confirmed (look-ahead) |
| C11 | "market_cap" = close × 63-day trailing mean volume × 20; delisting_return NaN everywhere | v5 | `scripts/download_tiingo.py:893-911` | Confirmed. *Raw* close and a *trailing* window: a causal dollar-volume proxy, not a dividend-adjusted cap (D9) |
| C12 | PIT universe module has no production caller | v5 | grep `construct_universe\|build_training_universe` in `src`, `scripts`, `notebooks`: only `src/data_pipeline/universe.py` | Confirmed |
| C13 | ERC falls back silently to 1/N | v5 | `src/benchmarks/erc.py:299-304` | Confirmed |
| C14 | Validated ≠ deployed objective: curriculum and β warm-up are fractions of `total_epochs`; Phase A runs with `max_epochs`, Phase B with `max_epochs=e_star` | both | `src/vae/loss.py:611,708`, `src/integration/pipeline.py:1457-1459, 1910-1912`, ISD L1395-1402 | Confirmed in spec *and* code |
| C15 | Phase B "trains on all data" but splits windows 90/10 positionally (stock-major ⇒ last stocks held out) | v5 (U07-CRI-03) | `src/integration/pipeline.py:1887-1891` | Confirmed (stock-major order taken from v5) |
| C16 | Capacity ratio uses N = n·(T_hist − T + 1), overlapping windows counted as independent | both | DVT L368-380; `src/integration/pipeline.py:318-319` | Confirmed in spec and code |
| C17 | `analytical_nonlinear` (config default) silently falls back to spiked shrinkage | v5 + project instructions | `src/risk_model/covariance.py:446-477`; `covShrinkage` absent from `pyproject.toml` | Confirmed |
| C18 | α_eff rescaling; default fast sub-solver has no turnover constraint | v5 | `src/portfolio/sca_solver.py:578-643, 796, 877` (`tau_max` used only in the CVXPY builder, L510) | Confirmed |
| C19 | AU_max_stat values 59 / 83.5 / 84 / 84.99 / 85 | theory | DVT L424-434, ISD L222, L1646, `src/inference/active_units.py:92-117` | Resolved in D7: **84** |
| C20 | Real-run AU_max_stat = 26 | v5 | Kish N_eff = (1+λ)/(1−λ), λ = 0.5^{1/252}, over 7,309 days = 727.1; ⌊√727.1⌋ = 26 | Confirmed; explains 26 and K = 52 |
| C21 | DVT §6.7 and Benchmark scenario C prescribe replacing the VAE by PCA | both | DVT L1159-1168, Benchmark L299-301 | Confirmed; inadmissible under the user decision |
| C22 | Dormant-factor amplification: share at reactivation = 1/(AU·p + 1 − p) | both (theory U02-CRI-02; v5 U01-CRI-01) | Equal ex-ante shares 1/AU, factor variance ×(1/p) ⇒ share (1/p)/(AU − 1 + 1/p); AU = 80, p = 0.05 ⇒ 20.2% | Confirmed |

**Literature.** Four sources checked for the first version, 30 opened by the review (three in full); see Appendix B.

---

## 3. Root causes, consolidated

Eight root causes. "Convergent" means both audits support it from their own evidence; this is the strongest support
available, because one audit never read code or results and the other did not check math or literature.

| ID | Root cause | v5 | Theory audit | Support |
|---|---|---|---|---|
| R1 | Learned object ≠ risk object | RC-A | RC-1 | **Convergent** + code (C4) |
| R2 | Static, unconditional object for a conditional, episodic question | RC-C, RC-D (tail) | RC-2 | **Convergent** + code (C5) |
| R3 | Unidentified latent coordinates feed the factor count and the objective | RC-B, U08-CRI-01 | RC-3 | **Convergent** on the count; scale/eigenbasis part theory-only, now code- and math-verified (C6, C7) |
| R4 | The risk functional and policy cannot express the stated goal | RC-D | RC-6, RC-2 (amplification) | **Convergent** (C8, C22) |
| R5 | Dependent observations counted as independent | U15-CRI-05, U14-CRI-04 (findings, not in v5's root-cause list) | RC-4 | **Convergent** (C16) |
| R6 | Evaluation cannot fail the theory; selection on the reported sample; validated ≠ deployed | RC-F, U07-CRI-01/02 | RC-5 | **Convergent** (C3, C14) |
| R7 | Data layer is plumbing, not a measurement contract | RC-E | RC-7 | **Convergent** on the principle, different specifics (C10-C12, D9) |
| R8 | Governance: several parameter sources, no provenance, unattributable changes | RC-G | (shadow only: U19 cross-document inconsistencies) | **v5 only**, by construction; partially verified (C17, C19) |

### R1 — The learned object is not the risk object
- **Mechanism.** Per-window ELBO, permutation-invariant across stocks, no shared z_t (DVT L446-512; `model.py:113-138`).
  Both input channels are z-scored per window, so the input is invariant to r → c·r; scale is re-imported by
  σ_i,t/σ̄_t. The only cross-sectional term (co-movement Spearman) is ~10⁻³ of reconstruction, has a fixed point
  inconsistent with the model's implied correlation (0.707 vs 0.894), and is scheduled to vanish.
- **In-sample diagnostics cannot separate a factor basis from noise:** E[R²] = K/(n−1) under the null; at equal
  dimension EP_VAE ≤ EP_PCA in sample (Ky Fan), so the DVT's "EP ≥ EP_PCA + 0.10" gate is unattainable.
- **What DL can add here (user decision).** The fix is not "use PCA". It is a *cross-sectionally trained* deep
  conditional factor model (templates: Gu-Kelly-Xiu conditional autoencoder; FactorVAE and NeuralFactors for the
  probabilistic form) whose nonlinearity can encode interactions between state variables that a linear instrument
  map (IPCA) cannot. GKX's own nonlinear loadings do not beat IPCA on total R² (12.6% vs 13.3%), so the
  nonlinearity must earn its place on covariance scores, and price-only data needs a structural block (L1).
- **Findings.** v5: U05-CRI-01, U05-CRI-02, U08-CRI-05, U04-CRI-01, U02-CRI-02, U01-CRI-02. Theory: U05-CRI-01
  (narrowed), U07-CRI-01, U04-CRI-01, U08-CRI-01, U10-CRI-01, U05-CRI-04; supporting U02-CRI-01, U05-CRI-02, U08-CRI-02.

### R2 — A static object answers a conditional, episodic question
- **Mechanism (theory).** Lifetime mean aggregation gives an episode of window fraction q a risk weight ∝ q²; sign
  changes cancel into D_ε; Σ_z and D_ε are full-history constants, so "contagion" can only appear through the scalar
  volatility rescaling, which is reactive (lags by construction).
- **Mechanism (code).** The run did the opposite of the doctrine: half-life 60 windows at stride 1 ≈ 60 days (C5), Σ_z
  EWMA half-life 252 against a repository note arguing it must be 0. The "memory" is neither the documented one nor
  tested.
- **Tails.** A covariance-only Gaussian object has zero tail dependence for ρ < 1 (Embrechts, McNeil & Straumann 2002,
  verified by the theory audit); the motivating risk (DVT §2.3) is a lower-tail phenomenon.
- **Findings.** v5: U01-CRI-03, U01-CRI-04, U01-CRI-06, U04-CRI-03, U08-CRI-04, U15-CRI-04. Theory: U01-CRI-03,
  U02-CRI-03, U02-CRI-04, U10-CRI-03, U17-CRI-03; supporting U03-CRI-01.

### R3 — Unidentified coordinates feed decisions
- **Factor count.** AU (KL > 0.01 nats) depends on the latent basis (per-coordinate KL is not rotation-invariant),
  on D = T·F through the threshold, and in practice on the guard: in the real run K = 2·AU_max_stat = 52 and AU is
  truncated to 26 (C20). 47 of 52 dimensions passed the threshold with a smooth spectrum (v5, not re-verified). The
  count is produced by caps.
- **Objective.** H is computed over eig(Σ_z) in latent space (C6). It changes under a latent rescaling that leaves
  Σ_assets unchanged (C7), and it is ill-posed under eigenvalue near-degeneracy (eigenvector error ∝ 1/gap,
  Davis-Kahan [unverified exact constant]). Shrinkage applied to Σ_z in latent coordinates is equivariant only under
  orthogonal maps, so with a non-orthogonal latent reparametrization even Σ_assets changes after shrinkage (derivation).
- **Literature.** Burda et al. (2015) define activity as Cov_x(E_q[u]) > 10⁻² and state no "K ≥ 1.5-2×" rule
  (theory audit, verified on the paper); the DVT attribution (L420, L1662) is wrong.
- **Findings.** v5: U05-CRI-03, U05-CRI-04, U08-CRI-01, U08-CRI-02, U08-CRI-03, U09-CRI-03, U17-CRI-03. Theory:
  U01-CRI-02, U06-CRI-01, U06-CRI-02, U05-CRI-03, U11-CRI-01, U18-CRI-03, U19-CRI-01, U19-CRI-02, U12-CRI-03,
  U13-CRI-03, U17-CRI-02, U11-CRI-03.

### R4 — The risk functional and policy cannot express the goal
- Entropy over **normalized sub-blocks** (signal factors without PC1; a separately normalized idiosyncratic block
  blended with a weight that was 0.20 / 0.05 / 0.0 depending on the source): invariant to how much risk each block
  carries; the excluded PC1 carried 81% of systematic risk (C2, C8).
- Units: variance and nats added with an uncalibrated exchange rate, patched by a data-dependent α_eff; the
  cardinality stage then optimizes with nominal α (C18).
- **Equal contributions ⇒ |β′_k| ∝ λ_k^{-1/2}**: the largest exposure goes to the lowest-variance (dormant or noisiest)
  direction; at reactivation that factor dominates (C22). The objective is even in β′, so bet signs are set by solver
  initialization (v5 U01-CRI-08, major). This amplification is a property of equal contributions over *any* principal
  basis, including the asset-space projectors of D1, and GMV loads even harder on small eigen-directions (exposure
  ∝ λ⁻¹). The protection must come from the covariance estimator's small eigenvalues (L2), not from the measure
  [review's own analysis, Ledoit-Wolf rationale].
- With μ = 0, any α > 0 has weakly higher *model* variance than minimum variance (theory U13-CRI-02). Correct, but see
  D1: it is true of every non-GMV portfolio, including every benchmark except min-variance.
- **Findings.** v5: U01-CRI-01, U02-CRI-03, U09-CRI-01, U09-CRI-04, U10-CRI-01..07 (05 major), U11-CRI-01..07,
  U15-CRI-02, U15-CRI-03, U15-CRI-06, U17-CRI-02. Theory: U02-CRI-02, U12-CRI-01, U13-CRI-02; supporting U13-CRI-01.

### R5 — Dependent observations counted as independent
- DVT L368 uses N = 7.06M overlapping windows while L651/L1076 state ≈ T_hist/T independent windows per stock; the
  code reproduces L368 (`pipeline.py:318-319`) and relaxes the guard to r·1.1 when it binds (`:338-353`).
- Ledoit-Wolf intensity assumes i.i.d. returns (LW 2003, Assumption 1, verified by the theory audit).
- Wilcoxon on adjacent, nested-window folds: under AR(1) ρ = 0.4 the actual size of a nominal 5% test ≈ 0.20
  (theory, recomputed there); with 9 pairs and Holm over 18 tests it cannot reject at all (min p 2/2⁹ = 0.0039 >
  0.05/18 = 0.0028, arithmetic re-checked).
- **Findings.** v5: U15-CRI-05, U14-CRI-04. Theory: U04-CRI-02, U18-CRI-01, U16-CRI-03; supporting U11-CRI-02.

### R6 — Evaluation cannot fail the theory
- Selection on the reported OOS (DVT L894/L898 vs ISD L2434; code: v5 U12-CRI-01, U14-CRI-08). Expected optimism of
  the best of 32 configs ≈ 2.07σ (re-derived).
- Ĥ = H/ln AU is the optimizer's own objective on frozen B and Σ_z, homogeneous of degree 0, non-comparable across AU,
  undefined for 1/N and inverse-vol.
- Decision engine without an "inconclusive" state (C3); benchmarks in a different harness (constant mix, free daily
  rebalancing) from the VAE (drift, costs); ERC = 1/N (C13); power 0.40 at n = 34, d = 0.3, yet p ≥ 0.05 is read as
  equivalence; five iterations on the same folds give family-wise error ≈ 0.23-0.26.
- Validated ≠ deployed objective (C14), and the "no validation set" deployment run silently holds out stocks (C15).
- **Findings.** v5: U12-CRI-01..04, U13-CRI-01..05, U14-CRI-01..09, U16-CRI-01, U16-CRI-02, U16-CRI-05, U07-CRI-01..03.
  Theory: U12-CRI-04, U14-CRI-01, U14-CRI-04, U15-CRI-01..03, U16-CRI-01, U16-CRI-02, U17-CRI-05, U18-CRI-02,
  U06-CRI-03, U09-CRI-02.

### R7 — Data contract
- **Implemented (v5, verified):** full-sample universe (C10), turnover proxy for cap (C11), no delisting returns and an
  imputation that never fires (C11), orphaned PIT universe (C12), $1 filter on back-adjusted prices (v5 U03-CRI-05, not
  re-verified), VIX never wired (C9). Tiingo often has no prices for dead tickers, so survivorship is not controlled
  (`.agents/project/reference/data.md`).
- **Specified (theory):** cap from dividend-adjusted price × shares (ISD L596, L649); Shumway −30%/−55% applied without
  knowing the delisting cause.
- **Findings.** v5: U03-CRI-01..05, U06-CRI-01, U16-CRI-04. Theory: U03-CRI-02, U03-CRI-03.

### R8 — Governance (v5 only)
- Parameters in DVT, ISD, `config.py`, notebook and run config disagree (K 200 / 75 / 50 / 52; shrinkage method;
  λ_co; entropy blend weight). Divergences are "legitimate by being listed". V2 changed eight things at once. The
  real run used a checkpoint of unknown provenance; V1/V2 artifacts were lost. The surviving artifacts (`results/`)
  are tracked but carry no manifest.
- Open defects from the project instructions: `analytical_nonlinear` silent fallback (C17), VIX not wired (C9), no
  market caps or delisting returns (C11), lost V1/V2 runs.
- **Findings.** v5: U07-CRI-01..03, U15-CRI-01, U15-CRI-05, U16-CRI-03, U17-CRI-01, U17-CRI-04.

**Meta-cause (both audits, in their own words).** The design is doctrine-first: principles were turned into mechanisms
without a declared target quantity, a scoring rule for it, or a pre-registered way to be wrong.

---

## 4. Divergences, resolved

### D1 — Diversification measure
- **v5:** principal-factor-basis contributions c′_k = λ_k β′_k² ("rotation-resolving") with explicit budgets.
  **Theory:** H depends on latent scale and eigenbasis; with μ = 0 any α > 0 is weakly dominated; use an asset-space
  measure as a constraint inside minimum variance.
- **Check.** The scale dependence is in the code (C6, C7); "rotation-resolving" holds for orthogonal maps only. The
  dominance argument holds for every non-GMV portfolio, so it only shows that diversification must pay through model
  error or non-variance risk, to be tested out of sample. Minimum-torsion bets are invariant only on asset-space or
  identified factors.
- **Decision.**
  1. The measure is computed from **Σ_sys,t = B_tΣ_f,tB_tᵀ (n×n, rank k) only**. Contributions are taken over the
     **spectral projectors** of Σ_sys,t, with eigenvalues whose relative gap is below a pre-registered threshold
     merged into one block: c_G(w) = wᵀP_GΣ_sysP_Gw. Σ_sys is invariant to B → BA, Σ_f → A⁻¹Σ_fA⁻ᵀ, hence so are
     its projectors (verified twice, C7), and the measure is continuous through degeneracy inside a block. The exact
     gradient machinery of `src/portfolio/entropy.py` is reused, re-pointed to asset space. MT-bets ENB is a second,
     named measure, reported, not optimized.
  2. **Total risk is decomposed explicitly:** market block (top projector), signal blocks, noise bulk, idiosyncratic
     D_t. Each block's share is reported and constrained where stated; nothing is excluded silently (fixes C8).
  3. **Status:** the policy is minimum variance subject to ENB(w) ≥ e (and, as a separate named arm, explicit
     budgets b over the blocks). The hypothesis "diversification pays through model error" (H3a) is tested against
     α = 0 (GMV on the same Σ_t) and ERC. Equal-contribution budgets over signal factors are *not* the default (R4
     amplification); they are one arm.
  4. **Amplification is not cured by the measure.** Equal contributions over any principal basis, including the
     asset-space projectors, imply |v_kᵀw| ∝ λ_k^{-1/2}; GMV loads even harder on small eigen-directions. Protection
     against noise-driven small eigenvalues must come from L2 (shrinkage, cleaning) and is measured by realized
     variance, not by the ENB.
  5. **Feasibility.** For long-only portfolios the market block dominates Σ_sys. The ENB floor is e = e_rel ·
     ENB_max,t, the per-date maximum feasible ENB under the constraint set C, with e_rel chosen in the inner loop from
     {0.5, 0.7, 0.9} (fixed before E1, OD7); the problem is nonconvex, so multi-start dispersion is reported.

### D2 — Stopping criteria
- **v5:** Phase 3 / H3 "not better → stop the VAE line and keep the linear model". **DVT §6.7 / Benchmark scenario C:**
  replace the VAE by PCA.
- **Decision.** Removed. Every gate on a DL layer has three outcomes: **pass** (adopt, go to the next layer),
  **fail** (this DL formulation is rejected; the next formulation of the ordered ladder in §6 is tested under the same
  protocol and the same family-wise budget), **inconclusive** (power insufficient; report, and do not iterate on the
  same folds). The ladder per layer is finite and pre-registered. When a ladder is exhausted, the research result is
  "no DL formulation tested beat its matched linear twin on this data", reported with the Model Confidence Set. What
  L1 then becomes is owner decision OD1 (open, §0.2); no option makes a linear model the strategy.
- **Corollary.** Scenario C, the `pca_tie` branch of `selection.py` and DVT §6.7 are deleted as actions. PCA and IPCA
  remain as arms and as the bar.

### D3 — Status of the VAE
- **v5:** demoted to a *candidate* reformulated as a cross-sectional model. **Theory:** the current window encoder may
  survive as a feature extractor; the latent vector stops being *declared* the loading.
- **Decision (OD3).** The per-window ELBO is not the loading objective. The VAE's place is the date-level latent
  (f_t, s_t): the exact Gaussian case is the E2 model, the variational regime/Student-t case is E7 (§5, "Where the VAE
  lives"). The current CNN encoder survives as the **temporal path encoder inside L1** (both state arms of E2), never
  as the loading itself, and only with the scale contract: inputs scaled by the trailing volatility of date t (no
  per-window z-score), window L ≤ 252.

### D4 — Factor count
- Both: not AU. **v5:** by the L1 score, with the overlap-aware capacity bound as ceiling. **Theory:** k* by inner-loop
  predictive likelihood under an effective-observation budget.
- **Decision.** Same thing; k = 1 + k_s + k_c is selected in the inner loop on the daily log-score; the ceiling is one
  budget on *effective* observations expressed in dates × factors, not windows (one formula, one place: L5 config).
  AU, KL thresholds, AU_max_stat and the K = 2·AU_max cascade are deleted.

### D5 — Memory of dormant risks
- **v5:** a declared blend of a recent-regime Σ_f and a stressed-history Σ_f, weight chosen by the score per regime.
  **Theory:** state-conditional Σ_f,t from a small recurrent/attention model; mixture of experts with experts
  persistent across retrainings.
- **Decision.** Memory lives in two places, both tested. **Forecast memory:** v5's declared blend is a linear bar in
  E1; the deep state-modulated Σ_f,t (L2 F1) and the regime-mixture VAE (L2 F3) must beat it on stress-day log-score
  without degrading calm days (H2c, H2r). **Policy memory (OD4):** P6, GMV on (1 − π)Σ_t + πΣ_stress with π fixed ex
  ante (Chow et al. 1999). π is a preference, not a forecast; it is the only form of the anti-cyclical principle that
  does not depend on forecasting crises from about 7 episodes (H3m).

### D6 — Primary score
- **v5:** QLIKE and bias ratio of portfolio variance at the rebalancing horizon h (21 or 63 days). **Theory:** daily
  multivariate Gaussian log-score, QLIKE on portfolio variances, GMV realized variance, calibration ratio.
- **Decision.** Two roles, both pre-registered. **Selection and inference:** daily 1-step Gaussian log-score (and
  QLIKE on fixed random long-only portfolios), because only daily loss differentials have power (§6.1); every
  log-score difference is split into a volatility part and a correlation part (L4). **Deployment calibration gate:**
  bias ratio realized/predicted portfolio variance at horizon h on the policy's own portfolios, in [0.8, 1.2] overall
  and reported on stress days. A model that wins daily but fails at h is not adopted.

### D7 — The AU_max_stat number
- Circulating: 59 (printed DVT formula), 83 / 83.5 (N = 7,057 windows per stock, the wrong N), 84 / 84.99, 85
  (ISD, code, √(2N/r)), 26 in the real run (C20).
- **Recomputation.** Free parameters of a symmetric AU×AU covariance: AU(AU+1)/2 ≤ N_obs/r_min ⇔ AU ≤
  (−1 + √(1 + 8·N_obs/r_min))/2. With N_obs = 30·252 − 252 + 1 = 7,309 and r_min = 2: √29,237 = 170.99, so AU ≤ 84.99
  and **AU_max_stat = 84** (check: 84·85/2 = 3,570 ≤ 3,654.5 < 85·86/2 = 3,655). The DVT's printed coefficient 4 gives
  59 (wrong algebra); the ISD/code √(2N/r) drops the linear term and gives 85.
- **Decision.** State 84 once, as a historical correction. The quantity itself is retired with AU (D4).

### D8 — The decisive experiment
- **Theory H1a** compares conditional vs static objects; its arms are static mean-B, static PCA, rolling PCA, IPCA, and
  an *optional* small g_θ. A pass would show that conditionality matters, which a linear model can deliver alone. It
  does not test deep learning.
- **Decision.** H1a stays in E1 (it fixes the bar and decides whether the conditional/episodic premise has OOS
  content), together with the loss effect (H1ℓ: IPCA-NLL vs IPCA-LS, a linear arm isolates it alone). The DL test is
  **E2**, mandatory: a **2 × 2 factorial** {matched linear twin, deep} × {no state, FiLM state}, all arms with the same
  market + structural block, inputs, NLL, Σ_f and D_t construction, and nested protocol. The DL effect is the
  deep-minus-twin contrast within each state arm (H1b); the state effect is tested in both rows (H1c). If H1a shows no
  conditional arm beats static ones, E2 still runs (a deep map might find conditionality a linear one misses) with a
  lower prior.

### D9 — Data specifics
- The theory audit's U03-CRI-02 (cap from dividend-adjusted price × shares) is a **spec** defect; the code does not do
  that. It uses raw close × trailing 63-day volume × 20 (C11), a causal dollar-volume proxy, because Tiingo has no
  shares outstanding. The theory L0 contract "unadjusted close × PIT shares" is **infeasible** with the current data.
- **Decision.** The universe is declared a **liquidity universe**: top-N by trailing 63-day dollar volume computed on
  raw close, re-evaluated at each refit date with information ≤ t, plus listing age ≥ 504 days and liveness. Level
  filters ($1) use raw close. Returns come from adjusted close. "Size" instruments are named "dollar-volume rank",
  never "market cap". Delisting: since the cause is unknown, every portfolio result is reported under three
  terminal-return bounds for tickers that stop trading {0, −30%, −100%}; portfolio rankings must be stable across
  bounds (H0). Daily risk scores are computed on names alive at t+1 and do not depend on the bounds.

### D10 — Gaps each audit had that the other covers
- **Imported from v5:** RMT baselines (DGJ; LW 2020 once `covShrinkage` is installed or fails loudly), governance,
  and the implementation defects (ERC fallback, solver turnover, cardinality, harness, validation split, VIX).
- **Fundamentals and industry control (v5 H5; OD5).** Tiingo has no sector or fundamental fields (v5 U02-CRI-01;
  `data.md`). A **price-only style model** (market beta, dollar-volume rank, momentum, volatility, reversal,
  liquidity; a Barra-like regression model) is a baseline arm in E1. **Point-in-time SEC SIC codes** are added now as a
  baseline (industry factors in the style model) and a diagnostic (does the learned structural block recover what SIC
  codes give?) [PIT quality of EDGAR SIC histories unverified; checked in E0]. They become an *input* in one named arm
  only if E2 and E3 fail. NeuralFactors and the Deep Risk Model draw much of their gain from industry inputs (review §2.8).
- **Imported from the theory audit:** latent-scale indeterminacy (D1), validated ≠ deployed (C14), AU_max algebra
  (D7), Burda misattribution (R3), dominance of α = 0 (D1), effective-sample accounting (R5).

### D11 — Claims in each audit's self-assessment that do not hold
- **v5 P6** ("overlap-aware effective sample size guard … already built") is false: the code counts overlapping
  windows (C16) and v5's own U15-CRI-05 says so. The overlap-aware bound exists only in DVT prose.
- **v5 P4** ("rotation-resolving") holds for orthogonal maps only (C7).
- **v5 Phase 1 gate** ("QLIKE better … in ≥ 75% of folds") counts dependent folds; replaced by DM-HAC on daily series.
- **Theory §10.3** calls amplification and validated ≠ deployed theory-only; both are in v5 (U01-CRI-01, U07-CRI-01/02).
  **Theory §10.1** says v5's overlap-aware guard is in code; it is not.
- **Survey (CPDFM)** "a per-stock encoder suffices; the cross-sectional block is the first to drop": rejected by the
  review. The Deep Risk Model ablation it relies on had 29 industry factors present; with price-only data a per-stock
  encoder cannot form low-rank industry loadings (L1).

---

## 5. Target architecture

One system, six layers (L0-L5), one estimand, one score. The DL content sits in L1 (conditional loading map: path
encoder, state conditioning, optionally a learned structural block) and L2 (state-modulated factor dynamics,
idiosyncratic volatility head, spectral cleaner, regime-mixture VAE); L3 may become DL in its last formulation
(decision-focused learning). Every deep arm has a matched linear twin, and the twins and E1 arms are the bar.

```
L0 Data contract ─► L1 Loading map B_t ─► L2 Covariance Σ_t ─► L3 Allocation w_t ─► L4 Evaluation & decision
   (PIT, liquidity    (market + structural   (low-rank + diag, PSD;   (asset space only;      (one harness, proper scores,
    universe, bounds,  + conditional blocks;  state-modulated Σ_f,t,    invariant diversification twins + MCS, nested
    state vector m_t)  probabilistic, NLL)    deep D_t, cleaner, VAE)   as a constraint; P6)    selection, SPA, holdout once)
                                          L5 Governance: one config, run manifest, executable spec, change protocol
```

**Estimand.** Σ_{t→t+h}, the covariance of returns over the next holding period h, for the universe U_t. Trained and
scored at h = 1 day; calibration checked at h = 21 and 63 days.

### L0 — Data contract
- **Output:** a versioned panel with content hash: adjusted-close returns r_i,t; raw close; dollar volume; U_t mask
  (D9); listing and last-trade dates; a PIT stress indicator; the market-state vector m_t; a manifest.
- **Stress indicator.** Pre-registered, in-data and PIT: trailing 21-day realized volatility of the equal-weight index
  above its expanding 80th percentile. VIX (when wired) is a robustness check, not a dependency.
- **Market-state vector m_t (8, PIT):** log realized volatility of the equal-weight index at 5/21/63 d; return
  dispersion (21 d); top-eigenvalue share of the 63-day correlation; index drawdown from its 252-day high;
  log(vol21/vol252); the stress indicator. Expanding-window standardized, **not ranked**; feeds L1 and L2.
- **Delisting bounds** {0, −30%, −100%} (D9). No NaN → 0, no full-sample selection, no back-adjusted level filter.
- **SIC codes (OD5):** point-in-time SEC EDGAR industry codes, for the baseline and the diagnostic only.
- **Reuse:** `src/data_pipeline/universe.py` (re-pointed to dollar volume), `returns.py`, `features.py`.

### L1 — Conditional loading map (DL layer 1)

**Contract.** For each date t, B_t = [b_mkt,t | B^S_t | B^C_t] (n_t × k, k = 1 + k_s + k_c ≤ 25), computed only from
information ≤ t. Nothing downstream reads latent coordinates; L2 exposes Σ_t and Σ_sys,t only. The shape/scale
separation (both audits' P3) is kept: every learned column is multiplied by the measured scale s_i,t (trailing 63-day
volatility, ratio winsorized per INV-008). The model is a conditional probabilistic deep factor model (CPDFM, the
survey's term): its latents are the date-level factor returns (§5, "Where the VAE lives").

**Why three blocks.** The doctrine asks for two kinds of co-movement that a per-stock map of scalar characteristics
cannot represent [review's own analysis]:
1. *Structural* factors (industry, supply chain): membership is not a function of beta, volatility, momentum or
   liquidity ranks, and a per-stock path encoder cannot form low-rank industry loadings either, because the factor
   paths of a window change at every date and the encoder never sees the other stocks. A cross-sectional operation
   (PCA of the trailing correlation, or attention over stocks) is required. This is the block rolling PCA provides
   and IPCA on scalar instruments does not.
2. *Regime-dependent* exposures: per-date ranks remove market-wide levels, so a regime can only enter through the
   unranked state vector m_t.

The conditional block B^C is where the DL claim is tested first. The structural block has two accepted forms (OD2):
the **fixed statistical block** (residual PCA, identical in both arms of E2) and the **learned cross-stock block**
(attention over a correlation graph, E3). Both are named arms; E3 tests whether the learned block can replace the
fixed one. Neither is excluded, and the learned block is not a fallback.

**Inputs (all point in time).**
- c_i,t (13): market beta, downside beta, volatility 21/63/252 d, momentum 12-1, 1-month reversal, dollar-volume
  rank, Amihud illiquidity, max daily return (21 d), skewness (252 d), distance to 252-day high, correlation to the
  equal-weight index; rank-normalized to [−0.5, 0.5] per date.
- P_i,t: last L = 252 days (variant L = 126) of five channels: r_i,τ / s_i,t; |r_i,τ| / s_i,t; log dollar volume
  minus its 252-day mean; equal-weight index return / its trailing volatility; its absolute value. Scaling by the
  trailing volatility of date t (not per-window z-scoring) keeps the level. Missing days are masked. Windows are
  built per date on the fly (500 × 252 × 5 floats ≈ 2.5 MB per date at n = 500).
- m_t (8, from L0), unranked.

**Blocks (deep arm).**

| Block | Definition | Size |
|---|---|---|
| Market column b_mkt | Trailing 252-day beta to the equal-weight index × s_i,t (fixed, all arms) | 0 |
| Structural block B^S, fixed form (E2) | Loadings on the top k_s principal components of the trailing 252-day correlation matrix of market residuals, × s_i,t; k_s ∈ {0, 5, 10, 20} chosen in the inner loop | 0 trainable |
| Path encoder E_ψ | 4 residual causal 1D-conv blocks, kernel 3, dilations 1/2/4/8, width 32, GELU + LayerNorm, attention pooling over time → h_i,t ∈ R³² | ≈ 15k |
| State encoder | MLP 8 → 32 → 16, GELU → e_t | ≈ 0.8k |
| Loading head g_θ | u = GELU(W₁[c_i,t, h_i,t] + b₁) (45 → 64); FiLM u ← γ(e_t) ⊙ u + β(e_t) (γ, β: linear 16 → 64, γ initialized at 1); dropout 0.1-0.2; GELU(64 → 64); linear 64 → k_c; B^C_i,t = s_i,t · output; k_c ∈ {2, 4, 8} | ≈ 11k |
| Total trainable | | ≈ 25-30k (≈ 27k without FiLM) |

Combinations with k > 25 (k_s = 20 with k_c = 8) are excluded, and each cell draws at most 8 inner configurations from
the grid, chosen before E2 (OD6).

**Σ_f,t and D_t inside training.** Factor returns are the GLS cross-sectional estimates
f̂_s = (B_sᵀD_s⁻¹B_s)⁻¹B_sᵀD_s⁻¹r_{s+1}; Σ_f,t is their EWMA covariance (half-life chosen in E1, linear shrinkage
toward its diagonal); D_t is the EWMA of squared residuals, floored at 5% of each stock's total variance. Both are
computed on a rolling buffer of past dates with gradients truncated beyond the current chunk of 64 consecutive
training dates (engineering choice, to be documented in `implementation_decisions.md`). With GLS factor returns,
B_tΣ_f,tB_tᵀ is invariant to B → BA, so no identification constraint is needed for the score.

**Loss.** Per-date, per-stock Gaussian NLL of the next-day cross-section,
ℓ_t(θ) = (1/2n_t)[log det Σ_t + r_{t+1}ᵀΣ_t⁻¹r_{t+1}], Σ_t = B_tΣ_f,tB_tᵀ + D_t, computed by Woodbury and the
matrix-determinant lemma in O(n_tk²) (no n×n matrix stored); averaged over training dates (one date = one batch).
Regularization: AdamW (lr 3·10⁻⁴, weight decay 10⁻⁴), VIF penalty 0.01·tr((B^CᵀB^C/n)⁻¹) (Deep Risk Model default),
early stopping on the inner validation NLL (last two years before the 21-day embargo), schedules independent of the
stopping epoch ("one clock"). Inner-loop variant: add NLL terms on 5- and 21-day cumulative returns (weight 0.5 each),
the multi-horizon stability device of the Deep Risk Model. Five seeds per configuration; the model is the average of
the five Σ_t (PSD by convexity), never the best seed.

**Matched linear twin (mandatory, same everything else).** B^C_i,t = s_i,t · (z_i,tΓ), z_i,t = [c_i,t, p_i,t,
c_i,t ⊗ m_t], where p_i,t are 12 fixed linear path summaries (EWMA means of the five channels at half-lives 21/63, and
trailing betas to the index at 21/63/252 d); Γ trained under the same NLL, the same Σ_f/D_t construction, the same
structural block, the same inner loop and budget. Without the state arm the c ⊗ m_t columns are dropped. This is
IPCA with state-interacted instruments estimated by likelihood instead of least squares (Kelly, Pruitt & Su 2019 is
the parent). Attribution diagnostic (not a gate): IPCA on the deep arm's frozen h_i,t, which separates "deep
features" from "deep map".

**Formulation ladder (pre-registered, each rung against its own twin).**
- **F1a (E2):** deep B^C without state input vs twin without state.
- **F1b (E2):** deep B^C with FiLM state conditioning vs twin with c ⊗ m_t.
- **F2 (E3):** the learned cross-stock structural block, as a named arm against the fixed one (OD2). Two attention
  layers over stocks on date t: h_i ← h_i + Σ_{j∈N_i} α_ij W_v h_j, α_ij = softmax_j(q_iᵀk_j/√32 + φ(ρ²⁵²_ij, ρ⁶³_ij)),
  N_i = the 32 stocks with the largest trailing |ρ|, φ an MLP 2 → 8 → 1; head 32 → k_s, × s_i,t. ≈ 20k extra
  parameters; permutation-equivariant and rotation-free. Twin = the F1 winner with the fixed PCA block. One inner
  configuration initializes E_ψ by contrastive co-movement pretraining (InfoNCE, positives = stock pairs in the top
  decile of sub-window correlation, stress sub-windows oversampled ×3; Dolphin, Smyth & Dong 2024).
- **Exhausted** (F1a, F1b and F2 all fail): the deep L1 claim is rejected for this information set; what L1 then is
  is owner decision OD1 (open, §0.2).

**Evidence and priors.** GKX: nonlinear loadings beat IPCA on predictive but not total R² (12.6% vs 13.3%, three
factors): low prior for F1a. Pelger & Xiong (2022): state-varying loadings explain more variation, supporting F1b.
NeuralFactors (Gopal 2024) beats PPCA on joint NLL, but returns-only inputs lose most of the gain (validation NLL 0.466
vs 0.324, an ablation that also removes industry); the Deep Risk Model's gains sit on top of 29 industry factors. F2
rests on design reasoning, not results. Expect small effects; report the capacity curve.

### L2 — Conditional covariance (DL layer 2)

**Contract.** Σ_t = B_tΣ_f,tB_tᵀ + D_t (+ an optional residual correlation term, F2), PSD by construction. API:
log-score of a return vector, variance of a portfolio, Σ_sys,t, stress scenarios (F3 only). Never exposes latent
coordinates to L3. Residual diagonality is a tested assumption (residual cross-correlation reported per refit).

**Linear bars (scored in E1).** Shrunk EWMA Σ_f (as in L1); DGJ and LW-2020 shrinkage; the declared calm/stressed
blend (D5); factor DCC on the L1 factor returns; asset-level DCC-NL (Engle, Ledoit & Wolf 2019); AFM-DCC (De Nard,
Ledoit & Wolf 2021, tested up to 1,000 stocks); QIS (Ledoit & Wolf 2022) on the same lookback. D_t controls: EWMA,
GARCH(1,1).

**DL formulations.**
- **F1 — state-modulated factor dynamics (E4).** Σ_f,t = V_t^{1/2} R_t V_t^{1/2}. Factor variances
  log V_k,t = log EWMA_k,t + a_k(e_t), a = MLP 16 → 16 → k (≈ 0.5k parameters). Correlations follow a DCC recursion
  whose target is R̄_t = (1 − π_t)R̄_calm + π_t R̄_stress, π_t = sigmoid(MLP 16 → 8 → 1 (e_t)); R̄_calm and R̄_stress are
  sample correlations of the L1 factor returns on calm and stress dates (L0 indicator), shrunk. B_t frozen from L1;
  optional joint fine-tune. Twin: a_k ≡ 0 and π_t = sigmoid(wᵀm_t) (logistic, linear in the state). A free Cholesky
  head is not used: it puts k(k+1)/2 free outputs on a handful of crisis episodes.
- **F1-vol — deep idiosyncratic volatility head (E4).** log σ²_i,t+1 = MLP([log RV_i at 1/5/22/63 d, |r_i,t|, log
  dollar volume, m_t]) (≈ 3k parameters, shared), trained on the diagonal QLIKE; twin: HAR-log regression on the
  same inputs; controls: EWMA, GARCH(1,1). Nonlinear volatility forecasts from price data have the most consistent
  published gains (Christensen, Siggaard & Veliyev 2023).
- **F2 — neural spectral cleaner (E4).** The rotation-invariant network of Bongiorno, Manolakis & Mantegna (BiLSTM
  over sorted eigenvalues, ≈ 40k parameters), trained on realized 5-day GMV variance on random sub-universes of
  50-350 stocks; (a) standalone Σ_t arm, the strongest published deep result on price-only US equities (1,000 stocks:
  volatility 11.9% vs 12.4% DCC, 12.6% QIS; turnover 57% vs 23% for QIS); (b) hybrid cleaning the L1 residual
  correlation, Σ_t = BΣ_fBᵀ + D^{1/2}C̃_res D^{1/2}. Twins: QIS; the same network with the identity eigenvalue map.
- **F3 — regime-mixture latent factor model, the variational VAE rung (E7).** Latents per date: s_t ∈ {calm, stress}
  with an HMM prior (learned 2 × 2 transition) and f_{t+1} | s ~ Student-t_ν(0, Σ_f^{(s)}); residuals Student-t. The
  stress regime adds 2-4 low-rank loading columns through regime-specific FiLM parameters, regularized toward the calm
  parameters. Inference: q(s_t | m_≤t) by a GRU(16) filter over m_t (point in time), q(f_{t+1} | r_{t+1}, B, D)
  Gaussian moment-matched, IWAE with 10-20 samples (NeuralFactors). Output: π_t, Σ^{(s)}, the mixture Σ_t, and a
  scenario sampler. ≈ 60-100k parameters. Twins: the Gaussian CPDFM without regime, and the linear two-regime blend
  at the same π_t rule. Label switching resolved by ordering on factor volatility. Runs only if H1c or H2c shows state
  value. It absorbs the former tail head; the parametric tail control is Oh & Patton (2017) factor copulas.

### Where the VAE lives (OD3)

**What the old VAE got wrong.** Its latent was a code for *one stock's z-scored path in one window*: each window was
encoded and reconstructed on its own, with no variable shared by the stocks on a date (C4). Codes were then averaged
over a stock's lifetime into a static loading, active units were read as a factor count, and entropy was computed in
the coordinates of that code (R1-R3). It was a VAE in name, but its latent was not a risk object, so nothing in its
likelihood rewarded a correct covariance.

**What is kept.** The VAE idea is a generative model with latent variables, an amortized (learned) mapping from data
to the model's parameters, and training by likelihood. The redesign keeps all three and puts the latent where risk
lives: the **date-level factor returns f_t** shared by every stock, and, at E7, a **regime s_t**. The decoder is the
factor model r_{t+1} = B_t f_{t+1} + ε_{t+1}, with B_t and D_t produced by the network (L1). This is the FactorVAE /
NeuralFactors form.

**Why E2 needs no ELBO.** With Gaussian factors and residuals the marginal likelihood is exact
(r ~ N(0, BΣ_fBᵀ + D)), so the posterior over f_t is exact and the ELBO adds nothing: the Gaussian CPDFM is the
exact-inference case of a VAE whose latent is f_t. Variational machinery (Student-t components, a discrete regime,
IWAE) becomes necessary, and useful, only for tails and regimes (F3, E7), and only if H1c or H2c shows state value.
What the old design took from VAEs and the redesign does not keep: active units as a factor count (Lucas et al. 2019
tie them to the noise level), disentangled factors, a smooth latent space for windows.

### L3 — Allocation
- **Contract:** input Σ_t only (and the scenario sampler if F3 is adopted); output w_t; constraints C (long-only,
  caps, turnover, cardinality) enforced **as hard constraints in the solver used at every rebalance** (fixes v5
  U11-CRI-05, U11-CRI-06). State and policy refreshed together.
- **Policies (named arms, never mode switches):** P0 GMV (α = 0); P1 GMV s.t. ENB_proj(w) ≥ e (D1, with the feasibility
  rule of D1.5); P2 explicit risk budgets over market / signal / noise / idio blocks (v5 L3); P3 ERC on assets; P4 1/N;
  P5 inverse-vol; **P6** GMV on Σ_policy = (1 − π)Σ_t + πΣ_stress, π fixed ex ante (policy-level memory, Chow et al.
  1999; OD4). Return views stay an additive μ-term layer, off by default.
- **Amplification.** Equal-contribution arms load on small eigen-directions (D1.4); their protection is the L2
  estimator, judged by realized variance.
- **DL formulation (last):** decision-focused learning — train L1/L2 end-to-end through a differentiable GMV or
  constrained layer on realized portfolio variance (Amos & Kolter 2017, OptNet; for GMV, Kim, Tae & Lee 2025;
  risk-budgeting layers, Uysal, Li & Mulvey 2024, small N).
- **Reuse:** SCA + Armijo + multi-start, projections, exact gradients (re-pointed to asset-space projectors).

### L4 — Evaluation and decision
- **One harness** for every arm: same universe, calendar, drift, costs, delisting bounds; a failed solve is a recorded
  failure, never 1/N.
- **Scores:** daily Gaussian log-score; QLIKE on 1,000 fixed random long-only portfolios and on GMV; GMV realized
  variance (Engle & Colacito 2006); calibration ratio overall / stress / calm. Portfolio endpoints: realized
  volatility, stress-episode loss, max drawdown, turnover, and ENB computed with **one common reference covariance**
  for every arm (the OOS realized covariance of the block). Sharpe is secondary.
- **Decomposition.** Every log-score difference is split into a volatility part (univariate log-scores of the
  diagonal) and a correlation part (log-score of returns standardized by forecast volatilities).
- **Selection:** nested walk-forward (inner split for hyperparameters and k; outer folds scored once); equal tuning
  budget for every arm, deep and twin.
- **Seeds.** Five seeds per configuration, averaged in Σ space; seed spread reported.
- **Inference:** DM with Newey-West bandwidth set from the measured loss-differential autocorrelation; stationary block
  bootstrap; Hansen SPA over **all variants ever scored** (a registry in L5); Model Confidence Set to report ties;
  TOST for equivalence claims.
- **Adoption needs two results.** (i) the deep arm beats its matched twin (one-sided DM-HAC, SPA over the arm's
  variants); (ii) the deep arm belongs to the 90% Model Confidence Set formed with every E1 arm. (i) alone shows that
  the nonlinearity has value; only (i) + (ii) makes the arm a candidate for the system.
- **Crisis tests.** First-occurrence test: walk-forward forecasts of each crisis from models trained before it (2008
  forecast with 1995-2007 only); plus leave-one-crisis-out for any state head (L1 F1b, L2 F1, F3).
- **Sharpe inference.** Ledoit-Wolf (2008) studentized block bootstrap; deflated Sharpe ratio (Bailey & López de
  Prado 2014) for anything selected.
- **Decision states:** adopt / reject / inconclusive / invalid-run. Default is inconclusive. The holdout is opened once,
  for one frozen configuration.

### L5 — Governance
- **Single configuration source** (frozen dataclasses in `src/config.py`) with a content hash; notebooks select named
  configs only; the spec documents reference config names, not numbers.
- **Run manifest** with every result: data hash, code commit, config hash, seeds, checkpoint hash, device. No manifest,
  no evidence.
- **Variant registry** feeding SPA (R6 forking paths).
- **Fail loudly:** missing optional dependencies (`covShrinkage`), missing stress data, solver fallbacks.
- **Artifacts:** results needed as evidence are committed or archived with their manifest; the tracked `results/`
  tree gets a retroactive manifest (what is known of its provenance, and what is not).
- **Pre-registration:** the OD7 margins are recorded in `docs/implementation_decisions.md` before E1.
- **Spec change protocol:** adopting this redesign retires or rewrites ISD invariants (below); each change needs an
  entry in `docs/implementation_decisions.md`. One change per experiment.

| ISD invariant | Fate |
|---|---|
| INV-001, INV-002, INV-006 (likelihood discipline of the window ELBO) | Retired with the per-window ELBO; the E7 variational model gets its own likelihood invariants, written with E7 |
| INV-003 (AU ≤ AU_max_stat) | Retired (D4, D7) |
| INV-004 (date-specific vs current rescaling) | Kept as the L1 scale contract |
| INV-005 (no look-ahead) | Kept and extended to universe, level filters, m_t and stress labels |
| INV-007 (principal factor basis of Σ_z, c′ ≥ 0) | Replaced: contributions over asset-space spectral projectors |
| INV-008 (ratio winsorization) | Kept for the scale s_i,t |
| INV-009 (exact gradient) | Kept, re-pointed |
| INV-010, INV-011 (curriculum, validation ELBO) | Retired with the co-movement curriculum; replaced by "one clock" (schedules independent of the stopping epoch) |
| INV-012 (identical constraints) | Kept and extended to harness, costs, tuning budget, twins |

### What is kept, removed, merged

| Element | Fate | Reason |
|---|---|---|
| μ = 0 default, removable return layer | Keep | Makes the risk model separately testable (both audits) |
| Price-only inputs; shape/scale separation, dual rescaling | Keep; SIC codes as baseline/diagnostic, input in one arm only if E2 and E3 fail | Both audits' P2/P3; OD5 |
| CNN window encoder | Move → temporal path encoder inside L1 (E2, both state arms); trailing-volatility scaling, L ≤ 252 | D3 |
| Per-window ELBO as the loading objective | Remove | R1 |
| VAE | Move to the date-level latent (f_t, s_t): exact Gaussian case E2, variational case E7 | OD3 |
| Co-movement scaffold and its curriculum; crisis γ in reconstruction | Remove | R1; γ is a no-op without VIX and cannot couple stocks |
| Lifetime/EWMA aggregation of latent trajectories into a static B | Remove; static mean-B kept as a baseline arm | R2 |
| AU, KL threshold, AU_max_stat, K = 2·AU_max_stat | Remove | R3, D4, D7 |
| Entropy over eig(Σ_z); PC1 exclusion; idio blend weight; uniform/proportional mode switch; α_eff; α frontier | Remove | R3, R4, D1 |
| Exact contribution gradients, SCA solver, projections | Keep, re-point to asset space | Correct machinery |
| DGJ spiked shrinkage, LW, PCA-RMT benchmark | Keep as baselines; add QIS, DCC-NL, AFM-DCC in E1 | v5 P5 |
| Fixed statistical (residual-PCA) structural block | Add, in every E2 arm | OD2 |
| Learned cross-stock structural block (correlation-graph attention) | Add as a second named arm (E3), tested as a replacement for the fixed block; not a fallback | OD2 |
| Neural spectral cleaner | Add as an L2 deep arm (E4) | Strongest price-only deep evidence |
| Walk-forward skeleton, embargo, holdout | Keep; add nesting; holdout confirmatory | Both audits |
| Phase A selection on OOS; Ĥ as decision metric; scenario engine with default branch; Wilcoxon on folds | Remove | R6 |
| Separate benchmark harness; ERC → 1/N fallback; NaN → 0 metrics | Remove | R6 |
| Full-sample universe; back-adjusted $1 filter | Remove | R7 |
| Capacity ratio on overlapping windows; three observability rules | Merge into one effective-observation budget (dates × factors) | R5 |
| Parameters duplicated across DVT/ISD/config/notebook | Merge into one config + manifest | R8 |
| DVT §6.7, Benchmark scenario C | Remove as actions | User decision, D2 |

---

## 6. Hypotheses, stopping criteria and statistical power

### 6.1 Power, stated honestly
Data: Tiingo, 1995-01-03 to 2026-02-20, no CRSP. Burn-in 10 years ⇒ OOS from 2005; holdout = last 3 years
(≈ 2023-02 to 2026-02). Non-holdout OOS ≈ 18 years ≈ 4,500 trading days.

| Evidence type | Sample | Effective size | Minimum detectable effect (α = 0.05 two-sided, 80% power) | Role |
|---|---|---|---|---|
| Daily loss differentials (log-score, QLIKE) | ≈ 4,500 days | N_eff ≈ 100-250 if the differential's integrated autocorrelation time is 18-45 days (assumption; **must be measured in E1**) | d ≈ 2.8/√N_eff ≈ 0.18-0.28 SD of the differential | **Decisive** |
| Holdout, daily | ≈ 750 days | N_eff ≈ 15-40 (same assumption) | d ≈ 0.45-0.7 | Confirms the sign only |
| Stress episodes in OOS (2008-09, 2010, 2011, 2015-16, 2018 Q4, 2020, 2022) | 6-7 | 6-7 | One-sided sign test: 7/7 needed at n = 7 (p = 0.0078); 6/7 gives 0.0625; at n = 6, 6/6 gives 0.016 | Corroborative |
| Sharpe difference, one 6.25-year block | — | — | SE ≈ √(2(1−ρ)/years) ≈ 0.18 for ρ = 0.9 (Jobson-Korkie/Memmel with the SR² term in per-period units, negligible; Monte Carlo 0.177) | The current 0.11 gap ≈ 0.6 SE: not evidence either way |
| Sharpe difference, 18 years | — | — | SE ≈ 0.105 for ρ = 0.9 ⇒ MDE ≈ 0.29 | Secondary only |
| Realized-variance ratio of two portfolios (H3a) | ≈ 4,500 days | — | SE of the log ratio ≈ 1.5-4.5% depending on the portfolios' correlation [review's estimate]; **measured in E1** | Sets the H3a margin |

Consequences: model choices are decided on daily covariance scores; portfolio and crisis claims are corroborative and
framed as non-inferiority or sign consistency. Missing delisting returns bias portfolio outcomes upward, more for
strategies holding small or distressed names; the daily risk score is less affected (it scores stocks alive on t+1),
but the liquidity universe still under-represents dead firms, so all claims are "within the Tiingo-survivable
liquidity universe".

### 6.2 Per-layer hypotheses

Notation: "DM" = one-sided Diebold-Mariano with HAC variance on daily differentials; SPA over the variant registry.
Every deep arm is judged against its matched twin and, for adoption, by MCS membership against the E1 bar (L4).

| Layer | Hypothesis | Refuted if | If refuted: stopping rule | Next DL formulation |
|---|---|---|---|---|
| L0 | **H0** Rankings are robust to data limits | L2 rankings (log-score) flip between liquidity universes of N = 300 and 500 or between the two halves of the OOS period; L3/L4 portfolio rankings flip across the three delisting bounds | Claims that flip are reported as data-undetermined; no adoption based on them | — (measurement) |
| L1 | **H1a** A conditional linear object beats the static object (E1) | DM p ≥ 0.05 for every conditional linear arm (rolling PCA, IPCA, IPCA + residual PCA) vs static PCA and static mean-B, *and* TOST equivalence within δ = 0.002 nats per stock-day (OD7) | The episodic premise has no linear OOS content; E2 still runs with a lower prior | F1a/F1b (E2) |
| L1 | **H1ℓ** The NLL objective beats least squares for the same linear map (E1) | DM p ≥ 0.05, IPCA-NLL vs IPCA-LS | Keep NLL anyway (it is the scoring rule); report | — |
| L1 | **H1b** Deep conditional loadings beat the matched twin, without state (E2, F1a) | DM p ≥ 0.05 after SPA over F1a variants, or calibration outside [0.8, 1.2], or GMV realized variance significantly worse (one-sided 5%) | F1a rejected | F1b |
| L1 | **H1c** State conditioning adds value (E2, F1b) | Deep-with-state does not beat deep-without-state *and* twin-with-state does not beat twin-without-state (DM, SPA), on all days and on stress days | Drop state conditioning from L1; regime claims move to L2 (H2c) | F2 (learned structural block) |
| L1 | **H1d** The learned cross-stock block beats the fixed PCA block (E3, F2) | DM p ≥ 0.05 vs the F1 winner with the fixed block | The fixed block stays the structural component; the learned block stays a reported named arm (OD2) | If F1a and F1b also failed: L1 ladder exhausted, OD1 (open) |
| L2 | **H2** State-modulated factor dynamics beat the linear bars (E4, F1) | 90% MCS contains a linear arm and no deep arm has a better mean, or DM p ≥ 0.05 vs the best of DCC-NL, AFM-DCC, QIS, declared blend | F1 rejected | F2 hybrid |
| L2 | **H2v** The deep volatility head beats HAR/GARCH on the diagonal (E4) | Diagonal QLIKE DM p ≥ 0.05 vs the HAR twin | Keep EWMA/GARCH D_t | — |
| L2 | **H2n** The neural spectral cleaner, alone or as a residual hybrid, beats QIS and the CPDFM (E4, F2) | DM p ≥ 0.05 on log-score *and* realized GMV variance not lower | Cleaner rejected | F3 (E7) |
| L2 | **H2c** State conditioning improves stress-day forecasts | Stress-day DM p ≥ 0.05, the first-occurrence crisis test fails, and the episode sign test fails (< 7/7 at n = 7, < 6/6 at n = 6) | Drop the "protected before reactivation" claim (DVT L78) | F3 only if H1c or H2c passed on at least one criterion |
| L2 | **H2r** The regime-mixture Student-t VAE beats the Gaussian CPDFM (E7, F3) | Joint log-score DM p ≥ 0.05 overall *and* on stress days, *and* lower-tail exceedance calibration not better | Keep the Gaussian model; delete tail-dependence claims (DVT L11, L511) | — |
| L3 | **H3a** The diversification constraint pays through model error | P1 vs P0 on the same Σ_t: realized-variance non-inferiority fails at the margin set from the SE measured in E1 (rule fixed before E1, OD7), or the stress-loss sign test fails | Drop the constraint; P0 | E6 decision-focused training |
| L3 | **H3b** The projector measure is more stable than the latent-eigenbasis entropy | Turnover at equal realized variance not lower (paired, HAC) | Keep the measure only if H3a holds | — |
| L3 | **H3m** Policy-level memory (P6) reduces stress losses at bounded calm-period cost | Stress-loss sign test fails or calm-period variance cost above the budget fixed before E1 | Drop P6 | — |
| L4 | **H4** The full system beats every benchmark | SPA p ≥ 0.05 on outer folds, or the holdout sign is opposite | Report the MCS; no deployment claim | Highest-ranked failing layer's next formulation, same protocol and registry |

**Budget against forking paths.** Each formulation gets at most 8 inner-loop configurations (OD6), counted separately
for each factorial cell; only the inner winner reaches the outer folds; a five-seed ensemble counts as one variant;
every outer-scored variant enters the SPA registry for the rest of the project. Iterating on the same outer folds
after a failure is not allowed without adding the variant to the registry.

---

## 7. Experiment plan

Ordered from cheapest and most decisive to most expensive. Each step is one heavy job at a time, wrapped in
`heavy_lock.py`. "Local" means the 16 GB laptop with `--device mps` or `auto`; never the 7,716-stock universe. Seed
ensembles and refit sweeps run on Colab (OD6). The OD7 margins are recorded before E1.

| Step | Content | Gate (pre-registered) | Where | Cost |
|---|---|---|---|---|
| **E0 — Foundations** | L0 (universe, filters, delisting bounds, stress indicator, m_t, PIT SIC codes). L4 (harness, decision states, DM/SPA/MCS, failure recording, log-score decomposition). L5 (config, manifests incl. `results/`, registry, loud failures). Fix ERC fallback, solver turnover, cardinality at every rebalance. No model change. | All benchmarks + S&P + incumbent (static mean-B) run end-to-end with manifests (50 stocks/8 years and 300 stocks); ERC solves on every fold; two runs identical; a synthetic panel with a planted *regime-dependent* factor structure ranks the true model first. | Local (minimal config ≈ 13 min, 4.7 GB peak; 300 stocks fits) | Engineering |
| **E1 — Linear bars, H1a, H1ℓ** | 500 stocks, 1995-2023, nested walk-forward: sample + LW; static PCA (DGJ); rolling PCA (504 d); IPCA-LS and IPCA-NLL (13 characteristics, + state interactions); IPCA + residual PCA; price-style model, with and without SIC industry factors; static mean-B; QIS; DCC-NL; AFM-DCC; declared memory blend; EWMA/GARCH D_t. Measure loss-differential autocorrelation, N_eff, MDE, and the SE of realized-variance ratios. | Bias ratio of the best arm in [0.8, 1.2] (else fix L2/D_t before any DL work); H1a, H1ℓ decided; output: the bar (MCS of linear arms), measured power, the H3a margin. | Local (DCC-NL/AFM-DCC on Colab if slow; their cost at this size is not verified) | Low |
| **E2 — Decisive DL test for L1 (H1b, H1c)** | 2 × 2 {linear twin, deep} × {no state, FiLM state}, all with the fixed market + residual-PCA structural block; same Σ_f/D_t construction and NLL; k_c ∈ {2, 4, 8}, k_s ∈ {0, 5, 10, 20}, k ≤ 25, at most 8 configurations per cell; 5 seeds averaged; annual refits, 21-day embargo. | Pass = deep beats twin within a state arm (DM-HAC one-sided, SPA over that cell's variants), calibration in [0.8, 1.2], GMV variance not significantly worse; *adoption* also requires MCS membership vs the E1 bar. Pass ⇒ E3 on the winner and E4. Fail on both cells ⇒ E3 still runs (structural block), L1 prior lowered. Inconclusive ⇒ report; no re-runs on the same folds. | Local for one seed (≈ 27k parameters, one date per batch); Colab for 5 seeds × 18 refits | Medium |
| **E3 — Learned structural block (H1d)** | Correlation-graph attention block as a named arm against the fixed PCA block, on the E2 winner (or on the deep no-state arm if E2 failed); one inner configuration with contrastive pretraining of the path encoder. Twin = same model with the fixed block. | Same gate as E2. Fail ⇒ the fixed block stays the structural component, the learned block stays a reported arm (OD2); if E2 also failed, the L1 ladder is exhausted and OD1 is decided on the results. | Local smoke on 50 stocks; Colab for sweeps (windows built on the fly, no materialized stride-1 windows) | Medium |
| **E4 — L2 (H2, H2v, H2n, H2c)** | On the best L1 map: state-modulated factor DCC vs its twin; deep volatility head vs HAR; neural spectral cleaner standalone and as residual hybrid vs QIS; all against the E1 bars. First-occurrence and leave-one-crisis-out tests. | §6.2 rows. Fail ⇒ next rung of the L2 ladder. | Colab for the cleaner (hours per yearly model on CPU in the paper); local for the rest | Medium-high |
| **E5 — Allocation (H3a, H3b, H3m)** | P0-P6 on the adopted Σ_t, one harness, costs, three delisting bounds. | §6.2 rows; H0 for portfolio rankings. | Local | Low |
| **E6 — Decision-focused (L3 DL)** | Only if H3a fails or after the E4 ladder fails: end-to-end fine-tuning of L1/L2 through a differentiable GMV or constrained layer on realized variance. | Beats the two-stage system on realized GMV variance (DM) without calibration loss. | Colab | High |
| **E7 — Regime-mixture VAE (H2r)** | Only if H1c or H2c passed on at least one criterion. | §6.2 H2r. | Colab | High |
| **E8 — End-to-end and holdout (H4)** | Frozen pipeline on outer folds; SPA vs every benchmark; then the holdout once. | SPA p < 0.05 and holdout sign consistent; otherwise report the MCS. | Colab for the full retrain sweep; local for scoring | Medium |

**Why E2 is decisive:** it first answers the user's question (does a *deep* conditional model beat its matched
linear twin?) with inputs that carry state and path information, on the most powerful evidence, and fits the laptop
for one seed.

---

## 8. Appendix A — Traceability of confirmed critical findings

Element codes refer to this document: R = root cause (§3), D = divergence decision (§4), L0-L5 = layers (§5),
H = hypotheses (§6), E = experiments (§7), OD = owner decisions (§0.2). Verification column: **T** = confirmed by the
theory audit's orchestrator; **V** = verified by v5's orchestrator; **Here** = re-checked for this document (first
version or review); **—** = not re-verified (accepted on consistency with verified findings, at the stated risk).

### A.1 Theory audit — 43 confirmed critical findings

| Finding | Short claim | Addressed by | Verif. |
|---|---|---|---|
| U01-CRI-02 | H depends on latent scale | D1 (asset-space projectors), L1 identification | T, Here (C7) |
| U01-CRI-03 | Lifetime mean dilutes episodic loadings (q²) | R2; L1 conditional B_t; static mean-B as baseline (E1) | T |
| U02-CRI-02 | Dormant factor ≈ 20% at reactivation | D1.3-D1.4 (equal contributions not default; protection from L2); H2c; H3a | T, Here (C22) |
| U02-CRI-03 | Gaussian object has no tail dependence | L2 F3 regime-mixture Student-t VAE gated by H2r; claims removed until then | T |
| U02-CRI-04 | Constant Σ_z, D_ε: contagion is a functional-form artefact | L2 state-modulated Σ_f,t, deep D_t (F1, F1-vol) | T |
| U03-CRI-02 | Cap from dividend-adjusted price | D9 (spec defect; code differs; liquidity universe) | T, Here (C11) — corrected |
| U03-CRI-03 | Shumway imputation applied without cause | D9 delisting bounds; H0 (portfolio rankings) | T |
| U04-CRI-01 | Input invariant to c·r; RV cannot restore scale | L1 scale contract s_i,t; path channels scaled by trailing volatility | T |
| U04-CRI-02 | N = 7.06M vs N_eff ≈ T_hist/T | R5; D4 single effective-observation budget | T, Here (C16) |
| U05-CRI-01 | No term ties μ to a cross-sectional loading | R1; L1 predictive NLL | T, Here (C4) |
| U05-CRI-03 | AU depends on D = T·F | D4 (AU removed) | T |
| U05-CRI-04 | EP_VAE ≤ EP_PCA at equal dimension (Ky Fan) | EP gate removed; L4 proper scores | T |
| U06-CRI-01 | AU depends on the latent basis | D4 | T |
| U06-CRI-02 | Guard binds; count = f(sample length) | D4, D7 | T, Here (C20) |
| U06-CRI-03 | Selection on reported OOS (component e) | L4 nested selection | T |
| U07-CRI-01 | Co-movement term ~10⁻³ of reconstruction | Scaffold removed; structural block in L1 | T |
| U08-CRI-01 | Co-movement fixed point ≠ model correlation | Scaffold removed | T |
| U09-CRI-02 | Validated ≠ deployed (curriculum clock) | L1 "one clock"; INV-010/011 retired | T, Here (C14) |
| U10-CRI-01 | E[R²] = K/(n−1) under the null | L4 OOS proper scores replace in-sample R² | T |
| U10-CRI-03 | Mean aggregation charges loading variation to D_ε | L1 conditional B_t | T |
| U11-CRI-01 | Printed AU_max formula wrong | D7 (84; retired) | T, Here |
| U11-CRI-03 | Shrinkage δ moves H; PCA arm unshrunk | D1 (asset-space measure); L4 equal tuning budget | T (narrowed) |
| U12-CRI-01 | 38.8% vs 1.25% target at reactivation | D1; H2c; H3a | T |
| U12-CRI-03 | H ill-posed under degeneracy | D1 block projectors | T (narrowed) |
| U12-CRI-04 | Ĥ_OOS on frozen model, degree-0 homogeneous | L4: Ĥ removed as metric | T |
| U13-CRI-02 | With μ = 0, α > 0 weakly dominated | D1 status change; H3a vs α = 0 | T, Here (argument re-derived; scope corrected) |
| U14-CRI-01 | Best-of-32 optimism ≈ 2.07σ | L4 nested; SPA registry | T, Here |
| U14-CRI-04 | Ĥ non-comparable across AU | L4 | T (narrowed) |
| U15-CRI-01 | Selection on reported OOS | L4 nested | T |
| U15-CRI-02 | Ĥ undefined for 1/N, inverse-vol | L4 common reference covariance | T (narrowed) |
| U15-CRI-03 | p ≥ 0.05 at n = 34 read as equivalence; power 0.40 | L4 TOST; §6.1 power; D2 | T, Here |
| U16-CRI-03 | Wilcoxon on dependent folds: size ≈ 0.20 | L4 DM-HAC, block bootstrap | T |
| U17-CRI-03 | Current-vol rescaling is reactive | L2 F1 dynamics; "anti-cyclical" becomes H2c and P6 (H3m) | T |
| U17-CRI-05 | Five iterations on same folds: FWER ≈ 0.23-0.26 | L5 registry; SPA | T, Here |
| U18-CRI-01 | = U04-CRI-02 + risk-model parameters unchecked | D4 budget covers L1 and L2 parameters | T |
| U18-CRI-03 | AU table uses √(2N/r) | D7 | T (narrowed), Here |
| U19-CRI-01 | = U11-CRI-01 (83 with N = 7,057) | D7 (N = 7,057 is the wrong N) | T, Here |
| U19-CRI-02 | Burda misattribution | R3; DVT L420/L1662 citation removed with AU | T |

**Duplicates** (same element as the row named): U13-CRI-03 and U17-CRI-02 = U12-CRI-03; U16-CRI-01 = U15-CRI-02;
U16-CRI-02 and U18-CRI-02 = U14-CRI-01.
**Downgraded by the theory audit (14):** U01-CRI-01 (L1 estimand), U02-CRI-01 (L1), U03-CRI-01 (stated limit, H2c),
U05-CRI-02 (L2 residual check), U07-CRI-02 (L1), U08-CRI-02 and U08-CRI-03 (γ and scaffold removed), U09-CRI-01 (L4),
U10-CRI-02 (D4), U11-CRI-02 (R5), U11-CRI-04 (L2 leverage-corrected residuals), U11-CRI-05 (EP gate removed),
U13-CRI-01 (D1), U14-CRI-03 (L5).
**Rejected by the theory audit (2):** U12-CRI-02, U18-CRI-04 — rejection reasons checked and accepted (sample vs true
eigenvalues; σ² = MSE/D under the INV-001 violation).

### A.2 v5 audit — 78 critical findings

| Findings | Short claim | Addressed by | Verif. |
|---|---|---|---|
| U01-CRI-01 | \|β′\| ∝ λ^{-1/2}: short-volatility exposure to dormant factors | D1.3-D1.4; H2c; H3a; P6 | Here (C22) |
| U01-CRI-02, U04-CRI-01, U02-CRI-02 | Per-window z-scoring removes level; RV channel redundant | L1 scale contract; path channels scaled by trailing volatility, \|r\| channel kept as one of five | — (consistent with T U04-CRI-01) |
| U01-CRI-03, U08-CRI-04, U15-CRI-04 | Half-life 60 windows = 60 days | R2; aggregation removed | V, Here (C5) |
| U01-CRI-04 | Σ_z half-life 252 vs doctrine 0 | L1/L2: EWMA half-life chosen in E1 by the score; memory tested (D5) | — |
| U01-CRI-06 | Covariance cannot represent tail risk | L2 F3; H2r | — (= T U02-CRI-03) |
| U02-CRI-01 | No characteristic/sector data; interpretation unit void | D10 price-style control; SIC codes baseline/diagnostic (OD5) | — |
| U02-CRI-03 | Never-co-moving factor invisible | Stated limit of price-only data; H2c scope | — |
| U03-CRI-01 | PIT universe orphaned | L0 | V, Here (C12) |
| U03-CRI-02 | Full-sample universe | L0, D9 | V, Here (C10) |
| U03-CRI-03 | Cap is a turnover proxy | D9 liquidity universe | Here (C11) |
| U03-CRI-04 | Delisting NaN; imputation never fires | D9 bounds; H0 (portfolio rankings) | V, Here (C11) |
| U03-CRI-05 | $1 filter on back-adjusted prices | L0 raw-close level filters | — |
| U04-CRI-03 | Trajectory discarded by aggregation | L1 conditional B_t, path encoder | — |
| U05-CRI-01, U05-CRI-02, U08-CRI-05 | Not a factor model; B from marginal law | R1; L1; "Where the VAE lives" | V, Here (C4) |
| U05-CRI-03 | No auto-pruning (47/52 above threshold) | D4 | — |
| U05-CRI-04 | AU/K ≤ 0.5 by construction | D4, D7 | Here (C20) |
| U06-CRI-01, U16-CRI-04 | γ no-op (vix_data=None) | γ removed; L0 stress indicator and m_t | V, Here (C9) |
| U07-CRI-01 | Phase 3 never ran (curriculum vs patience) | Curriculum removed; one clock | — |
| U07-CRI-02 | Phase B executes a different curriculum | L1 one clock | Here (C14) |
| U07-CRI-03 | Positional 90/10 split holds out stocks | L1/L4 time-based inner split only | Here (C15) |
| U08-CRI-01, U08-CRI-02, U08-CRI-03, U17-CRI-03 | AU selection not rotation-invariant; cut unresolvable | D4 | — (convergent with T U06-CRI-01) |
| U09-CRI-01 | var ratio 2.74 | L4 calibration gate; E1 harness gate | V, Here (C2) |
| U09-CRI-02 | Spec LW, run spiked, default NL falls back | L5 fail loudly; D10 baselines | Here (C17) |
| U09-CRI-03 | n_signal from raw vs effective γ | DGJ kept as baseline only; fix in E0 | — |
| U09-CRI-04 | Off-by-one in signal slice | Slice removed with PC1 exclusion (D1) | V, Here (C8) |
| U10-CRI-01 | PC1 (81%) excluded | D1 explicit market block | V, Here (C2, C8) |
| U10-CRI-02, U10-CRI-03, U10-CRI-04 | Block-normalized entropy; idio weight in nats; undefined mode | D1; L3 named arms | — |
| U10-CRI-06, U11-CRI-02, U11-CRI-07, U15-CRI-06 | α calibration inoperative; flat frontier | α removed; L3 constraint form, D1.5 feasibility | — |
| U10-CRI-07 | Reported H_norm mismatched numerator/denominator | L4 one definition per metric | — |
| U11-CRI-01 | Idio entropy blends toward inverse-vol | D1 (no idio entropy; idio share reported) | — |
| U11-CRI-03, U11-CRI-04, U15-CRI-02 | α_eff patch; nominal α in cardinality stage | α removed | Here (C18) |
| U11-CRI-05 | Fast solver has no turnover constraint | L3 hard constraints; E0 | V, Here (C18) |
| U11-CRI-06 | Cardinality once per fold | L3; E0 | — |
| U12-CRI-01, U14-CRI-08 | Selection on reported OOS | L4 nested | — (convergent with T) |
| U12-CRI-02, U14-CRI-07 | Phase A score terms identically zero | Phase A scoring removed; L4 | — |
| U12-CRI-03 | Real-data diagnostic bypasses walk-forward | E1-E8 run the nested protocol | V |
| U12-CRI-04, U13-CRI-02, U14-CRI-02, U16-CRI-01 | Scenario C from 0 comparisons / 0 wins | L4 decision states; D2 | V, Here (C3) |
| U13-CRI-01 | ERC = 1/N | L4 failure recording; E0 gate | V, Here (C1, C13) |
| U13-CRI-03 | Benchmarks in a different harness | L4 one harness | — |
| U13-CRI-04 | PCA arm run with a different tuning budget | L4 equal budget | — |
| U13-CRI-05 | VAE vs PCA contrast unidentifiable | E2 factorial; matched twins; L4 adoption rule | — |
| U14-CRI-01 | Verdict never uses a p-value | L4 | — |
| U14-CRI-03 | Strategy's own objective never compared | L4 common reference covariance | — |
| U14-CRI-04 | Test cannot reject (0.0039 > 0.0028) | R5; L4 DM-HAC | V, Here (arithmetic) |
| U14-CRI-05 | Holm p not adjusted (v5: downgraded to major) | L4 reporting | V |
| U14-CRI-06 | Positional pairing after independent dropna | L4 calendar-aligned pairing | — |
| U14-CRI-09 | Health checks and decision engine disconnected | L4 calibration gate inside the decision | — |
| U15-CRI-01 | No instance of the documented parameters exists | L5 single config | — |
| U15-CRI-03 | "Recommended" proportional budget inverts the formula | L3 named arms | — |
| U15-CRI-05 | Capacity guard off by ~470× and auto-relaxed | R5; D4 | Here (C16) |
| U16-CRI-02 | Post-mortem's own matrix selects Path A, not C | D2 (scenario logic removed) | — |
| U16-CRI-03 | V2 changed eight things at once | L5 one change per experiment | — |
| U16-CRI-05 | Decision rules pre-authorize the conclusion | L4 pre-registration; OD7 | — |
| U17-CRI-01 | Conformance = whatever the code does | L5 executable spec + ADRs | — |
| U17-CRI-02 | Deployed objective (two-layer, idio weight) not in spec | D1; L5 | — |
| U17-CRI-04 | One-fold run, checkpoint of unknown provenance | L5 manifest; retroactive manifest for the tracked `results/` | V |

No confirmed critical finding of either audit is rejected outright. Four are **corrected in scope**: theory U03-CRI-02
(spec, not code), theory U13-CRI-02 (true of every non-GMV portfolio), v5 P4 and P6 (see D11).

---

## 9. Appendix B — References

Statuses: **web** = verified on the web for the first version; **review** = opened for the review (abstract, or full
text where marked); **survey** = abstract read by the survey, not re-opened; **theory audit** = verified there (its
INDEX §3), not re-checked. Full records and links: [redesign_review.md](redesign_review.md) §5 and
[architecture_survey.md](architecture_survey.md) §8.

- **Verified on the web (first version):** Gu, Kelly & Xiu (2021, J. Econometrics 222(1):429-450; total-R² figures
  12.6/13.3% secondary source only); Engle, Ledoit & Wolf (2019, JBES 37(2):363-375, DCC-NL); Meucci, Santangelo &
  Deguest (2015, *Risk*, minimum-torsion bets); Duan et al. (2022, AAAI 36(4):4468-4476, FactorVAE; body not read).
- **Opened for the review, full text:** Gopal 2024 (NeuralFactors, arXiv 2408.01499); Bongiorno-Manolakis-Mantegna
  2025 (arXiv 2507.01918; venue not confirmed); Lin-Zhou-Liu-Bian 2021 (Deep Risk Model, arXiv 2107.05201).
- **Opened for the review, abstract:** Kelly-Pruitt-Su 2019 (IPCA); Pelger-Xiong 2022; Lucas et al. 2019;
  Fan-Liao-Wang 2016; De Nard-Ledoit-Wolf 2021 (AFM-DCC label not in the abstract); Ledoit-Wolf 2020 (analytical
  nonlinear shrinkage); Donoho-Gavish-Johnstone 2018; Lohre-Opfer-Ország 2014; Hansen 2005; Engle-Colacito 2006;
  Ledoit-Wolf 2008; Bailey-López de Prado 2014; Kim-Tae-Lee 2025; Yang-Zhao-Chai 2026; Uysal-Li-Mulvey 2024;
  Dolphin-Smyth-Dong 2024; Wei et al. 2023 (HireVAE, return prediction only).
- **Survey only (abstract), not re-opened:** Ledoit-Wolf 2022 (QIS); Chow et al. 1999; Hansen-Lunde-Nason 2011
  (MCS); Christensen-Siggaard-Veliyev 2023; Burda-Grosse-Salakhutdinov 2016 (IWAE); Boulet 2021 (secondary);
  Korangi-Mues-Bravo 2024 (snippet only; thin evidence for F2).
- **Verified by the theory audit (not re-checked):** Embrechts-McNeil-Straumann 2002; Shumway 1997; Shumway-Warther
  1999; Ledoit-Wolf 2003/2004; Diebold-Mariano 1995; White 2000; Politis-Romano 1994; Patton 2011; Tipping-Bishop 1999;
  Burda et al. 2016 (activity definition); Salinas et al. 2019; Oh-Patton 2017; Amos-Kolter 2017 (OptNet);
  Cawley-Talbot 2010; Meucci 2009.
- **Secondary or [unverified]:** Memmel (2003) / Jobson-Korkie variance (formula checked by Monte Carlo in the review,
  not on the text); Perez et al. (2018, FiLM); Lettau-Pelger 2020; Roncalli 2013; López de Prado 2018; Davis-Kahan
  constants.
- **Not found by the review or the survey:** any published out-of-sample covariance test of a regime-mixture VAE on
  equities; any covariance-loss evaluation of a correlation-graph attention block on price-only data.
