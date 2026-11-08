# Latent Risk Factor Discovery — Redesign of record

> **Status:** proposal, nothing implemented. **Date:** 2026-11-08.
> **Supersedes as normative proposal:** [audit_v5/REDESIGN.md](audit_v5/REDESIGN.md) (code + results audit) and
> [audit_theory/REDESIGN.md](audit_theory/REDESIGN.md) (theory-only audit). Both remain the evidence base.
> **Method:** independent arbitration. Every claim kept as load-bearing was re-checked at the cited spec or code line, or
> re-derived; what could not be checked is marked. Literature cited from memory is marked [unverified].
> **Binding user decision (not discussed here):** the strategy stays a deep-learning (DL) risk model. Linear and
> shrinkage models (PCA, IPCA, Ledoit-Wolf, RMT, DCC, 1/N, ERC, inverse-vol) are baselines, controls and stopping
> criteria, never the destination.

---

## 1. Executive summary

**The shared diagnosis.** Both audits, from disjoint evidence (v5: code and the one surviving real run; theory audit:
documents, math and literature), reach the same conclusion: *the network is trained to compress each stock's own
z-scored return window, while the portfolio needs a forecast of the joint, time-varying covariance of returns.* Nothing
in the training loss ties the latent vector to a cross-sectional loading. Every later stage (lifetime averaging into a
static B, active units as factor count, entropy computed in latent coordinates, a normalized objective with patched
units, a decision engine that cannot say "inconclusive") is a heuristic bridging that gap, and the evaluation was built
so that it cannot show the theory is wrong.

**What the evidence actually shows today.** One real run (Tiingo, 788 stocks, one 1,575-day OOS block, pretrained
checkpoint): VAE Sharpe 0.38 vs 1/N 0.49; realized variance 2.74× the model forecast; one principal factor carries 81%
of systematic risk and is excluded from the objective; the engine printed "Scenario C — Adopt PCA" from zero valid
comparisons (all verified in `git show HEAD:results/diagnostic/…`, see §2). Under a stated correlation assumption the
0.11 Sharpe gap is about 0.6 standard errors (§6.1), so "the VAE loses" is not statistically established either. The
system cannot currently test its own theory in either direction.

**What changes.** The DL content moves to where it can be judged against the right object:
1. **The learned object** becomes a *conditional* loading map B_t = s_t ⊙ g_θ(x_t) (shared across stocks, from
   point-in-time price-derived state), trained on the **out-of-sample cross-sectional predictive likelihood** of
   next-day returns, with a deep conditional factor covariance Σ_f,t as the second DL layer.
2. **Every downstream quantity is computed in asset space** (Σ_t = B_tΣ_f,tB_tᵀ + D_t). No latent coordinate, active
   unit or latent eigenbasis reaches a decision.
3. **Diversification becomes a tested hypothesis** (a constraint on an invariant measure inside minimum variance),
   not an additive reward assumed to pay.
4. **Evaluation is rebuilt** on proper scoring rules of the covariance forecast, where the data has statistical
   power (daily loss differentials), with nested selection, one harness for every arm, a family-wise test over every
   variant ever tried, an "inconclusive" state, and the holdout opened once.

**The decisive next experiment (E2, §7).** After the foundations (E0) and the linear ladder (E1), train a small
conditional deep loading map g_θ and its exact linear counterpart IPCA on **the same instruments, the same loss, the
same Σ_f and D_t estimators, and the same nested walk-forward protocol**, and compare their daily OOS Gaussian
log-scores. **Gate:** g_θ beats IPCA (one-sided Diebold-Mariano with HAC variance, p < 0.05 after an SPA correction
over all g_θ variants tried) *and* its portfolio-variance calibration ratio lies in [0.8, 1.2] *and* realized GMV
variance does not degrade by more than 2%. **If it fails:** the loading-map nonlinearity on scalar instruments is
rejected and the next DL formulation (path encoder over return windows, with a matched linear control on the same
learned features) is tested under the same rule; the plan never falls back to "adopt the linear model".

---

## 2. What was checked, and what it showed

The table lists the load-bearing claims I re-checked myself. "HEAD" means the file exists in the last commit:
**the whole `results/` tree, including the only real-run artifacts and both checkpoints, is currently deleted from the
working tree (unstaged deletion, not made by this work).** If that deletion is committed, the only surviving real-run
evidence disappears from the branch tip. This needs a decision by the owner before anything else (L5).

| # | Claim | Source audit | Checked at | Verdict |
|---|---|---|---|---|
| C1 | Real-run numbers (Sharpe 0.38 / 0.49 / 0.50 / 0.20 / 0.30 / 0.66; ERC row byte-identical to 1/N) | v5 | `HEAD:results/diagnostic/strategy_comparison.csv` | Confirmed |
| C2 | var ratio 2.742, AU = 26 / K = 52, EP_is 0.219, EP_oos 0.356, n_signal 12, top-1 risk share 0.811, OOS 1,575 dates | v5 | `HEAD:results/diagnostic/health_checks.csv`, `diagnostic_data.json` | Confirmed |
| C3 | "Scenario C — Adopt PCA" emitted with 0 comparisons | v5 | `HEAD:results/diagnostic/report.txt`; `src/walk_forward/selection.py:110-136` (`n<5 → continue`, default B, `wins<=1 → C`) | Confirmed |
| C4 | VAE reconstructs each stock's own window; no shared factor path | both | `src/vae/model.py:113-138` | Confirmed |
| C5 | Aggregation half-life 60 counted in windows, applied to stride-1 inference windows (≈ 60 days, documented as ≈ 1,260) | v5 | `src/integration/pipeline.py:1976-1995`, `src/inference/composite.py:122-165`, `src/config.py:355-357`, run config `aggregation_half_life: 60` | Confirmed |
| C6 | Principal factors taken from eig(Σ_z) in latent space, B′ = B V | both | `src/risk_model/covariance.py:607-622` | Confirmed |
| C7 | H over principal factors of Σ_z changes under a latent column rescaling that leaves Σ_assets unchanged | theory (U01-CRI-02) | Re-derived numerically (n = 30, k = 3, S = diag(1, 3, 0.2)): Σ_assets identical; H_latent 0.421 → 0.259; H over asset-space principal portfolios 0.977 → 0.977 | Confirmed; asset-space version invariant |
| C8 | PC1 excluded from the entropy, off-by-one in the signal slice | v5 | `src/integration/pipeline.py:2286-2303` (`signal_end = n_signal + 1` when PC1 skipped) | Confirmed |
| C9 | `vix_data=None` in every entry point | both (v5 code, theory spec) | `scripts/run_diagnostic.py:444`, `scripts/run_walk_forward.py:203`, `notebooks/dashboard.ipynb:1151,1217` | Confirmed |
| C10 | Production universe = top-n by full-sample median "market cap" | v5 | `src/data_pipeline/data_loader.py:344-348` | Confirmed (look-ahead) |
| C11 | "market_cap" = close × 63-day trailing mean volume × 20; delisting_return NaN everywhere | v5 | `scripts/download_tiingo.py:893-911` | Confirmed. Note: *raw* close and a *trailing* window, so it is a causal dollar-volume proxy, not a dividend-adjusted cap (see D9) |
| C12 | PIT universe module has no production caller | v5 | grep `construct_universe\|build_training_universe` in `src`, `scripts`, `notebooks`: only `src/data_pipeline/universe.py` | Confirmed |
| C13 | ERC falls back silently to 1/N | v5 | `src/benchmarks/erc.py:299-304` | Confirmed |
| C14 | Validated ≠ deployed objective: curriculum and β warm-up are fractions of `total_epochs`; Phase A runs with `max_epochs`, Phase B with `max_epochs=e_star` | both (theory: spec; v5: code) | `src/vae/loss.py:611,708`, `src/integration/pipeline.py:1457-1459, 1910-1912`, ISD L1395-1402 | Confirmed in spec *and* code |
| C15 | Phase B "trains on all data" but splits windows 90/10 positionally (stock-major ⇒ last stocks held out) | v5 (U07-CRI-03) | `src/integration/pipeline.py:1887-1891` | Confirmed (positional split; stock-major order taken from v5) |
| C16 | Capacity ratio uses N = n·(T_hist − T + 1), overlapping windows counted as independent | both | DVT L368-380; `src/integration/pipeline.py:316-317` | Confirmed in spec and code |
| C17 | `analytical_nonlinear` (config default) silently falls back to spiked shrinkage | v5 + project instructions | `src/risk_model/covariance.py:446-477`; `covShrinkage` absent from `pyproject.toml` | Confirmed |
| C18 | α_eff rescaling; default fast sub-solver has no turnover constraint | v5 | `src/portfolio/sca_solver.py:578-643, 796, 877` (`tau_max` used only in the CVXPY builder, L510) | Confirmed |
| C19 | AU_max_stat values 59 / 83.5 / 84 / 84.99 / 85 | theory | DVT L424-434, ISD L222, L1646, `src/inference/active_units.py:92-117` | Resolved in D7: **84** |
| C20 | Real-run AU_max_stat = 26 | v5 | Kish N_eff for EWMA half-life 252 over 7,309 days = 727.1 (computed); ⌊√727.1⌋ = 26 | Confirmed; explains 26 and K = 52 |
| C21 | DVT §6.7 and Benchmark scenario C prescribe replacing the VAE by PCA | both | DVT L1159-1168, Benchmark L299-301 | Confirmed; inadmissible under the user decision |
| C22 | Dormant-factor amplification: share at reactivation = 1/(AU·p + 1 − p) | both (theory U02-CRI-02; v5 U01-CRI-01) | Re-derived: equal ex-ante shares 1/AU, factor variance ×(1/p) ⇒ share (1/p)/(AU − 1 + 1/p) = 1/(AU·p + 1 − p); AU = 80, p = 0.05 ⇒ 20.2% | Confirmed |

**Literature checked on the web for this document:** Gu, Kelly & Xiu (2021, J. Econometrics 222(1):429-450,
conditional autoencoder built on IPCA); Engle, Ledoit & Wolf (2019, JBES 37(2):363-375, DCC with nonlinear
shrinkage of the correlation target); Meucci, Santangelo & Deguest (2015, *Risk*, minimum-torsion bets); Duan et al.
(2022, AAAI 36(4):4468-4476, FactorVAE trained against future cross-sectional returns). Other references are either
verified by the theory audit (its INDEX §3, not re-checked here) or marked [unverified].

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
| R8 | Governance: several parameter sources, no provenance, unattributable changes | RC-G | (shadow only: U19 cross-document inconsistencies) | **v5 only**, by construction; partially verified (C17, C19, results deletion) |

### R1 — The learned object is not the risk object
- **Mechanism.** Per-window ELBO, permutation-invariant across stocks, no shared z_t (DVT L446-512; `model.py:113-138`).
  Both input channels are z-scored per window, so the input is invariant to r → c·r; scale is re-imported by
  σ_i,t/σ̄_t. The only cross-sectional term (co-movement Spearman) is ~10⁻³ of reconstruction, has a fixed point
  inconsistent with the model's implied correlation (0.707 vs 0.894), and is scheduled to vanish.
- **In-sample diagnostics cannot separate a factor basis from noise:** E[R²] = K/(n−1) under the null; at equal
  dimension EP_VAE ≤ EP_PCA in sample (Ky Fan), so the DVT's "EP ≥ EP_PCA + 0.10" gate is unattainable.
- **Findings.** v5: U05-CRI-01, U05-CRI-02, U08-CRI-05, U04-CRI-01, U02-CRI-02, U01-CRI-02. Theory: U05-CRI-01
  (narrowed), U07-CRI-01, U04-CRI-01, U08-CRI-01, U10-CRI-01, U05-CRI-04; supporting U02-CRI-01, U05-CRI-02, U08-CRI-02.
- **What DL can add here (user decision).** The fix is not "use PCA". It is a *cross-sectionally trained* deep loading
  map (Gu-Kelly-Xiu conditional autoencoder; FactorVAE as the VAE-family template), whose nonlinearity can encode
  interactions between state variables that a linear instrument map (IPCA) cannot.

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
  initialization (v5 U01-CRI-08, major).
- With μ = 0, any α > 0 has weakly higher *model* variance than minimum variance (theory U13-CRI-02). Correct, but see
  D1: it is true of every non-GMV portfolio, including every benchmark except min-variance.
- **Findings.** v5: U01-CRI-01, U02-CRI-03, U09-CRI-01, U09-CRI-04, U10-CRI-01..07 (05 major), U11-CRI-01..07,
  U15-CRI-02, U15-CRI-03, U15-CRI-06, U17-CRI-02. Theory: U02-CRI-02, U12-CRI-01, U13-CRI-02; supporting U13-CRI-01.

### R5 — Dependent observations counted as independent
- DVT L368 uses N = 7.06M overlapping windows while L651/L1076 state ≈ T_hist/T independent windows per stock; the
  code reproduces L368 (`pipeline.py:316-317`) and relaxes the guard to r·1.1 when it binds (`:338-353`).
- Ledoit-Wolf intensity assumes i.i.d. returns (LW 2003, Assumption 1, verified by the theory audit).
- Wilcoxon on adjacent, nested-window folds: under AR(1) ρ = 0.4 the actual size of a nominal 5% test ≈ 0.20
  (theory, recomputed there); with 9 pairs and Holm over 18 tests it cannot reject at all (min p 2/2⁹ = 0.0039 >
  0.05/18 = 0.0028, arithmetic re-checked).
- **Findings.** v5: U15-CRI-05, U14-CRI-04. Theory: U04-CRI-02, U18-CRI-01, U16-CRI-03; supporting U11-CRI-02.

### R6 — Evaluation cannot fail the theory
- Selection on the reported OOS (DVT L894/L898 vs ISD L2434; code: v5 U12-CRI-01, U14-CRI-08). Expected optimism of
  the best of 32 configs ≈ 2.07σ (theory, recomputed there).
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
  real run used a checkpoint of unknown provenance; V1/V2 artifacts were lost; the surviving artifacts are now deleted
  in the working tree.
- Open defects from the project instructions: `analytical_nonlinear` silent fallback (C17), VIX not wired (C9), no
  market caps or delisting returns (C11), lost V1/V2 runs.
- **Findings.** v5: U07-CRI-01..03, U15-CRI-01, U15-CRI-05, U16-CRI-03, U17-CRI-01, U17-CRI-04.

**Meta-cause (both audits, in their own words).** The design is doctrine-first: principles were turned into mechanisms
without a declared target quantity, a scoring rule for it, or a pre-registered way to be wrong.

---

## 4. Divergences, resolved

### D1 — Diversification measure
- **v5:** keeps the principal-factor-basis contributions c′_k = λ_k β′_k² as a strength (P4, "rotation-resolving"),
  replaces the normalized entropy by explicit risk budgets over total risk.
- **Theory:** H depends on the latent scale and on the eigenbasis under near-degeneracy; with μ = 0 any α > 0 is
  weakly dominated in model variance; use minimum-torsion ENB or entropy over principal portfolios of the asset-level
  systematic covariance, as a constraint inside minimum variance.
- **Check.** The scale dependence is real and present in the code (C6, C7). v5's "rotation-resolving" is true only
  for orthogonal latent maps, which is not the model class. The dominance argument is correct but nearly tautological:
  every portfolio other than GMV is weakly dominated in model variance, so it does not single out entropy; what it
  shows is that the value of diversification can only come from model error or non-variance risk, which must be
  tested out of sample. The theory audit's claim that minimum-torsion bets are invariant to latent
  reparametrization holds only if they are computed on asset-space or already-identified factors; with raw latent
  factors MT depends on the chosen factor set by construction (Meucci et al. choose the uncorrelated set closest to the
  manager's factors).
- **Decision.**
  1. The measure is computed from **Σ_sys,t = B_tΣ_f,tB_tᵀ (n×n, rank k) only**. Contributions are taken over the
     **spectral projectors** of Σ_sys,t, with eigenvalues whose relative gap is below a pre-registered threshold
     merged into one block: c_G(w) = wᵀP_GΣ_sysP_Gw. The projector of a block is unique even when its eigenvectors
     are not, so the measure is invariant to latent reparametrization (verified, C7) and continuous through
     degeneracy inside a block. The exact gradient machinery of `src/portfolio/entropy.py` is reused, re-pointed to
     asset space. MT-bets ENB is a second, named measure, reported, not optimized.
  2. **Total risk is decomposed explicitly:** market block (top projector), signal blocks, noise bulk, idiosyncratic
     D_t. Each block's share is reported and constrained where stated; nothing is excluded silently (fixes C8).
  3. **Status:** the policy is minimum variance subject to ENB(w) ≥ e (and, as a separate named arm, explicit
     budgets b over the blocks). e is chosen in the inner loop. The hypothesis "diversification pays through model
     error" (H3a) is tested against α = 0 (GMV on the same Σ_t) and ERC. Equal-contribution budgets over signal
     factors are *not* the default (R4 amplification); they are one arm.

### D2 — Stopping criteria
- **v5:** Phase 3 / H3 "not better → stop the VAE line and keep the linear model". **DVT §6.7 / Benchmark scenario C:**
  replace the VAE by PCA.
- **Decision.** Removed. Every gate on a DL layer has three outcomes: **pass** (adopt, go to the next layer),
  **fail** (this DL formulation is rejected; the next formulation of the ordered ladder in §6 is tested under the same
  protocol and the same family-wise budget), **inconclusive** (power insufficient; report, and do not iterate on the
  same folds). The ladder per layer is finite and pre-registered. When a ladder is exhausted, the research result is
  "no DL formulation tested beat the best linear control on this data", reported with the Model Confidence Set; the
  next step is to change the information set (intraday realized covariances, PIT fundamentals as instruments), not to
  promote the linear control to the strategy. Deploying anything is a separate business decision outside this plan.
- **Corollary.** Scenario C, the `pca_tie` branch of `selection.py` and DVT §6.7 are deleted as actions. PCA and IPCA
  remain as arms and as the bar.

### D3 — Status of the VAE
- **v5:** demoted to a *candidate* reformulated as a cross-sectional model. **Theory:** the current window encoder may
  survive as a feature extractor; the latent vector stops being *declared* the loading.
- **Decision.** Theory position, with v5's discipline. The current CNN encoder is kept as a candidate **path-feature
  extractor** for the second DL formulation (E3), never as the loading itself. Its reconstruction ELBO survives only
  as optional pretraining; INV-001/002/006 (likelihood discipline) apply only if that pretraining is used.

### D4 — Factor count
- Both: not AU. **v5:** by the L1 score, with the overlap-aware capacity bound as ceiling. **Theory:** k* by inner-loop
  predictive likelihood under an effective-observation budget.
- **Decision.** Same thing; k* is selected in the inner loop on the daily log-score; the ceiling is one budget on
  *effective* observations (one formula, one place: L5 config). AU, KL thresholds, AU_max_stat and the K = 2·AU_max
  cascade are deleted.

### D5 — Memory of dormant risks
- **v5:** a declared blend of a recent-regime Σ_f and a stressed-history Σ_f, weight chosen by the score per regime.
  **Theory:** state-conditional Σ_f,t from a small recurrent/attention model; mixture of experts with experts
  persistent across retrainings.
- **Decision.** v5's declared blend is the **linear control** for memory (it is cheap and interpretable); the deep
  state-conditional Σ_f,t must beat it on stress-day log-score without degrading calm days (H2c). Memory is therefore a
  test, not a doctrine, in both forms.

### D6 — Primary score
- **v5:** QLIKE and bias ratio of portfolio variance at the rebalancing horizon h (21 or 63 days). **Theory:** daily
  multivariate Gaussian log-score, QLIKE on portfolio variances, GMV realized variance, calibration ratio.
- **Decision.** Two roles, both pre-registered. **Selection and inference:** daily 1-step Gaussian log-score (and
  QLIKE on fixed random long-only portfolios), because only daily loss differentials have power (§6.1). **Deployment
  calibration gate:** bias ratio realized/predicted portfolio variance at horizon h on the policy's own portfolios, in
  [0.8, 1.2] overall and reported on stress days. A model that wins daily but fails at h is not adopted.

### D7 — The AU_max_stat number
- The quantities circulating: 59 (printed DVT formula), 83 / 83.5 (exact formula with N = 7,057, which is the
  windows-per-stock count, not the number of dates), 84 / 84.99 (exact formula with N_obs = 7,309), 85 (ISD, code and
  DVT table, √(2N/r)), and 26 in the real run (Kish N_eff = 727 for EWMA half-life 252, C20).
- **Recomputation.** Free parameters of a symmetric AU×AU covariance: AU(AU+1)/2 ≤ N_obs/r_min ⇔ AU ≤
  (−1 + √(1 + 8·N_obs/r_min))/2. With the corpus's N_obs = 30·252 − 252 + 1 = 7,309 and r_min = 2: √29,237 = 170.99,
  so AU ≤ 84.99 and **AU_max_stat = 84** (check: 84·85/2 = 3,570 ≤ 3,654.5 < 85·86/2 = 3,655). The DVT's printed
  coefficient 4 gives 59 (wrong algebra); the ISD/code √(2N/r) drops the linear term and gives 85.
- **Decision.** State 84 once, as a historical correction. The quantity itself is retired with AU (D4); it counted
  dates, not effective observations, and only Σ_z's parameters.

### D8 — The decisive experiment
- **Theory H1a** compares conditional vs static objects; its arms are static mean-B, static PCA, rolling PCA, IPCA, and
  an *optional* small g_θ. A pass would show that conditionality matters, which a linear model (IPCA, rolling PCA) can
  deliver alone. It does not test deep learning.
- **Decision.** H1a stays as E1 (it fixes the bar and decides whether the conditional/episodic premise has OOS
  content). The DL test is added as **E2 / H1b**, mandatory: g_θ vs IPCA in a **2 × 2 factorial** {linear, deep} ×
  {reconstruction loss (GKX/IPCA style), predictive Gaussian NLL}, same instruments, Σ_f and D_t estimators, nested
  protocol. The DL effect is the deep-minus-linear contrast within each loss. Gate, stopping rule and next formulation
  in §6-§7. If H1a shows *no* conditional arm beats static ones, E2 still runs (a deep map might find conditionality a
  linear one misses), but its prior is low and the ladder moves early to deep Σ_f dynamics on static loadings (H2).

### D9 — Data specifics
- The theory audit's U03-CRI-02 (cap from dividend-adjusted price × shares) is a **spec** defect; the code does not do
  that. It uses raw close × trailing 63-day volume × 20 (C11), a causal dollar-volume proxy, because Tiingo has no
  shares outstanding. The theory L0 contract "unadjusted close × PIT shares" is **infeasible** with the current data.
- **Decision.** The universe is declared a **liquidity universe**: top-N by trailing 63-day dollar volume computed on
  raw close, re-evaluated at each refit date with information ≤ t, plus listing age ≥ 504 days and liveness. Level
  filters ($1) use raw close. Returns come from adjusted close (correct for returns). "Size" instruments are named
  "dollar-volume rank", never "market cap". Delisting: since the cause is unknown, every portfolio result is reported
  under three terminal-return bounds for tickers that stop trading {0, −30%, −100%}; the risk-score rankings must be
  stable across bounds (H0).

### D10 — Gaps each audit had that the other covers
- **Imported from v5 into the theory design:** RMT baselines (DGJ spiked shrinkage, already in code; LW 2020 analytical
  nonlinear shrinkage, after `covShrinkage` is installed or the fallback made to fail loudly); all governance (single
  config, manifest, change protocol); implementation defects (ERC fallback, fast solver without turnover constraint,
  cardinality once per fold, separate benchmark harness, positional validation split, `vix_data=None`).
- **Fundamentals control (v5 H5).** Tiingo has no sector or fundamental fields (v5 U02-CRI-01; `data.md`). Decision:
  a **price-only style model** (market beta, dollar-volume rank, momentum, volatility, reversal, liquidity; a
  Barra-like regression model) is a baseline arm now; an industry control is deferred until a point-in-time industry
  source exists (SEC EDGAR SIC codes are a candidate [unverified PIT quality]). This is a baseline, not an input,
  so the price-only discovery principle is kept.
- **Imported from the theory audit into v5's design:** latent-scale/eigenbasis indeterminacy (D1), validated ≠ deployed
  (C14; note v5's findings U07-CRI-01/02 *had* it in code, its REDESIGN did not carry it), AU_max algebra (D7), Burda
  misattribution (R3), in-model dominance of α = 0 (D1), effective-sample accounting for LW and the tests (R5).

### D11 — Claims in each audit's self-assessment that do not hold
- **v5 P6** ("overlap-aware effective sample size guard … already built") is false: the code counts overlapping
  windows (C16) and v5's own U15-CRI-05 says so. The overlap-aware bound exists only in DVT prose.
- **v5 P4** ("rotation-resolving") holds for orthogonal maps only (C7).
- **v5 Phase 1 gate** ("QLIKE better … in ≥ 75% of folds") counts dependent folds; replaced by DM-HAC on daily series.
- **Theory §10.3** lists the dormant-factor amplification and validated ≠ deployed as "found only by this audit". Both
  are in v5's findings (U01-CRI-01; U07-CRI-01/02). They are convergent, not theory-only.
- **Theory §10.1** says v5 "preserves the overlap-aware ESS guard as a strength (found in code)"; it was not in code.

---

## 5. Target architecture

One system, six layers (L0-L5), one estimand, one score. The DL content sits in L1 (conditional loading map) and L2
(conditional factor covariance, optional tail head); L3 may become DL in its last formulation (decision-focused
learning). Linear counterparts exist for each DL layer and are the bar.

```
L0 Data contract ─► L1 Loading map B_t ─► L2 Covariance Σ_t (+ tail head) ─► L3 Allocation w_t ─► L4 Evaluation & decision
   (PIT, liquidity    (deep g_θ, shared,     (low-rank + diag, PSD;            (asset space only;      (one harness, proper scores,
    universe, bounds)  cross-sectionally      deep Σ_f,t dynamics, D_t)         invariant diversification nested selection, SPA/MCS,
                       trained)                                                  as a constraint)        holdout once)
                                          L5 Governance: one config, run manifest, executable spec, change protocol
```

**Estimand.** Σ_{t→t+h}, the covariance of returns over the next holding period h, for the universe U_t. Trained and
scored at h = 1 day; calibration checked at h = 21 and 63 days.

### L0 — Data contract
- **Output:** a versioned panel with content hash: adjusted-close returns r_i,t; raw close; dollar volume; U_t mask
  (D9); listing and last-trade dates; a PIT stress indicator s_t; a manifest.
- **Stress indicator.** Pre-registered, in-data and PIT: trailing 21-day realized volatility of the equal-weight index
  above its expanding 80th percentile. VIX (when wired) is a robustness check, not a dependency.
- **Delisting bounds** {0, −30%, −100%} (D9). No NaN → 0, no full-sample selection, no back-adjusted level filter.
- **Reuse:** `src/data_pipeline/universe.py` (re-pointed to dollar volume), `returns.py`, `features.py`.

### L1 — Conditional loading map (DL layer 1)
- **Contract:** for each date t, B_t (n_t × k) computed only from information ≤ t; B_i,t = s_i,t · g_θ(x_i,t), g_θ
  shared across stocks. s_i,t is the measured scale (trailing volatility) — the shape/scale separation (both audits'
  P3) is kept as a contract.
- **Instruments x_i,t (price-only, rank-normalized per date):** market beta, downside beta, volatility 21/63/252 d,
  momentum 12-1, 1-month reversal, dollar-volume rank, Amihud illiquidity, max daily return, skewness, distance to
  252-day high, correlation to the index. Formulation 2 adds learned path features (D3).
- **Training loss (primary):** −Σ_t log N(r_t+1; 0, B_tΣ_f,tB_tᵀ + D_t), Σ_f,t and D_t from trailing estimates on
  data ≤ t (differentiable through Woodbury and the matrix-determinant lemma: O(nk²) per date, no n×n matrix stored).
  **Alternative loss:** GKX reconstruction ‖r_t+1 − B_t f_t+1‖² with f the cross-sectional projection. Both appear in
  the E2 factorial.
- **Identification:** nothing downstream reads latent coordinates; L2 exposes only Σ_t and Σ_sys,t. For reporting,
  factors are the principal portfolios of Σ_sys,t (asset space).
- **Linear counterpart:** IPCA (Kelly, Pruitt & Su 2019 [verified by the theory audit]) on the same x. **Templates:**
  Gu, Kelly & Xiu (2021, verified); FactorVAE (Duan et al. 2022, verified).
- **Why DL can add value:** loadings that switch on under joint conditions (high volatility *and* illiquidity *and*
  deep drawdown) and path shapes no scalar instrument summarizes: the episodic exposures the doctrine wants, at the
  date they matter.

### L2 — Conditional covariance (DL layer 2)
- **Contract:** Σ_t = B_tΣ_f,tB_tᵀ + D_t, PSD by construction; API: log-score of a return vector, variance of a
  portfolio, Σ_sys,t; never exposes latent coordinates to L3.
- **Σ_f,t:** baseline = shrunk trailing sample (DGJ / LW-2020) with EWMA weights chosen in the inner loop; memory
  control = v5's declared recent/stressed blend (D5); **DL formulation** = small GRU or attention module over
  market-state variables computed from the panel (index volatility, dispersion, average pairwise correlation,
  drawdown), outputting a Cholesky factor of Σ_f,t.
- **D_t:** per-stock volatility model (EWMA or GARCH(1,1)); residual diagonality is a tested assumption (residual
  cross-correlation reported per refit).
- **Optional tail head (only if H2 passes):** Student-t or factor-copula output with state-conditioned parameters;
  parametric control: Oh & Patton (2017) factor copulas [verified by the theory audit].

### L3 — Allocation
- **Contract:** input Σ_t only (and the tail object if adopted); output w_t; constraints C (long-only, caps,
  turnover, cardinality) enforced **as hard constraints in the solver used at every rebalance** (fixes v5 U11-CRI-05,
  U11-CRI-06). State and policy refreshed together.
- **Policies (named arms, never mode switches):** P0 GMV (α = 0); P1 GMV s.t. ENB_proj(w) ≥ e (D1); P2 explicit risk
  budgets over market / signal / noise / idio blocks (v5 L3); P3 ERC on assets; P4 1/N; P5 inverse-vol. Return views
  stay an additive μ-term layer, off by default.
- **DL formulation (last):** decision-focused learning — train L1/L2 end-to-end through a differentiable GMV or
  constrained layer on realized portfolio variance (Amos & Kolter 2017, OptNet [verified by the theory audit]).
- **Reuse:** SCA + Armijo + multi-start, projections, exact gradients (re-pointed to asset-space projectors).

### L4 — Evaluation and decision
- **One harness** for every arm: same universe, calendar, drift, costs, delisting bounds; a failed solve is a recorded
  failure, never 1/N.
- **Scores:** daily Gaussian log-score; QLIKE on 1,000 fixed random long-only portfolios and on GMV; GMV realized
  variance (Engle & Colacito 2006 [verified by the theory audit]); calibration ratio overall / stress / calm. Portfolio
  endpoints: realized volatility, stress-episode loss, max drawdown, turnover, and ENB computed with **one common
  reference covariance** for every arm (the OOS realized covariance of the block). Sharpe is secondary, reported with
  the deflated Sharpe ratio.
- **Selection:** nested walk-forward (inner split for hyperparameters and k*; outer folds scored once); equal tuning
  budget for every arm.
- **Inference:** DM with Newey-West bandwidth set from the measured loss-differential autocorrelation; stationary block
  bootstrap; Hansen SPA over **all variants ever scored** (a registry in L5); Model Confidence Set to report ties;
  TOST for equivalence claims.
- **Decision states:** adopt / reject / inconclusive / invalid-run. Default is inconclusive. The holdout is opened once,
  for one frozen configuration.

### L5 — Governance
- **Single configuration source** (frozen dataclasses in `src/config.py`) with a content hash; notebooks select named
  configs only; the spec documents reference config names, not numbers.
- **Run manifest** with every result: data hash, code commit, config hash, seeds, checkpoint hash, device. No manifest,
  no evidence.
- **Variant registry** feeding SPA (R6 forking paths).
- **Fail loudly:** missing optional dependencies (`covShrinkage`), missing stress data, solver fallbacks.
- **Artifacts:** results needed as evidence are committed or archived with their manifest; resolve the current
  deletion of `results/` first.
- **Spec change protocol:** adopting this redesign retires or rewrites ISD invariants (below); each change needs an
  entry in `docs/implementation_decisions.md`. One change per experiment.

| ISD invariant | Fate |
|---|---|
| INV-001, INV-002, INV-006 (likelihood discipline) | Kept only for the optional encoder pretraining |
| INV-003 (AU ≤ AU_max_stat) | Retired (D4, D7) |
| INV-004 (date-specific vs current rescaling) | Kept as the L1 scale contract |
| INV-005 (no look-ahead) | Kept and extended to universe, level filters and stress labels |
| INV-007 (principal factor basis of Σ_z, c′ ≥ 0) | Replaced: contributions over asset-space spectral projectors |
| INV-008 (ratio winsorization) | Kept if s_i,t uses the ratio |
| INV-009 (exact gradient) | Kept, re-pointed |
| INV-010, INV-011 (curriculum, validation ELBO) | Retired with the co-movement curriculum; replaced by "one clock" (schedules independent of the stopping epoch) |
| INV-012 (identical constraints) | Kept and extended to harness, costs, tuning budget |

### What is kept, removed, merged

| Element | Fate | Reason |
|---|---|---|
| μ = 0 default, removable return layer | Keep | Makes the risk model separately testable (both audits) |
| Price-only inputs; shape/scale separation, dual rescaling | Keep | Both audits' P2/P3 |
| CNN window encoder | Move → path-feature extractor (E3) | D3 |
| Per-window ELBO as the loading objective | Remove | R1 |
| Co-movement scaffold and its curriculum; crisis γ in reconstruction | Remove | R1; γ is a no-op without VIX and cannot couple stocks |
| Lifetime/EWMA aggregation of latent trajectories into a static B | Remove; static mean-B kept as a baseline arm | R2 |
| AU, KL threshold, AU_max_stat, K = 2·AU_max_stat | Remove | R3, D4, D7 |
| Entropy over eig(Σ_z); PC1 exclusion; idio blend weight; uniform/proportional mode switch; α_eff; α frontier | Remove | R3, R4, D1 |
| Exact contribution gradients, SCA solver, projections | Keep, re-point to asset space | Correct machinery |
| DGJ spiked shrinkage, LW, PCA-RMT benchmark | Keep as baselines | v5 P5 |
| Walk-forward skeleton, embargo, holdout | Keep; add nesting; holdout confirmatory | Both audits |
| Phase A selection on OOS; Ĥ as decision metric; scenario engine with default branch; Wilcoxon on folds | Remove | R6 |
| Separate benchmark harness; ERC → 1/N fallback; NaN → 0 metrics | Remove | R6 |
| Full-sample universe; back-adjusted $1 filter | Remove | R7 |
| Capacity ratio on overlapping windows; three observability rules | Merge into one effective-observation budget | R5 |
| Parameters duplicated across DVT/ISD/config/notebook | Merge into one config + manifest | R8 |
| DVT §6.7, Benchmark scenario C | Remove as actions | User decision, D2 |

---

## 6. Hypotheses, stopping criteria and statistical power

### 6.1 Power, stated honestly
Data: Tiingo, 1995-01-03 to 2026-02-20, no CRSP. Burn-in 10 years ⇒ OOS from 2005; holdout = last 3 years
(≈ 2023-02 to 2026-02). Non-holdout OOS ≈ 18 years ≈ 4,500 trading days.

| Evidence type | Sample | Effective size | Minimum detectable effect (α = 0.05 two-sided, 80% power) | Role |
|---|---|---|---|---|
| Daily loss differentials (log-score, QLIKE) | ≈ 4,500 days | N_eff ≈ 100-250 if the differential's autocorrelation decays over 20-50 days (theory audit's assumption; **must be measured in E1**) | d ≈ 2.8/√N_eff ≈ 0.18-0.28 SD of the differential | **Decisive** |
| Holdout, daily | ≈ 750 days | N_eff ≈ 15-40 (same assumption) | d ≈ 0.45-0.7 | Confirms the sign only |
| Stress episodes in OOS (2008-09, 2010, 2011, 2015-16, 2018 Q4, 2020, 2022) | 6-7 | 6-7 | One-sided sign test: 7/7 needed at n = 7 (p = 0.0078); 6/7 gives 0.0625; at n = 6, 6/6 gives 0.016 | Corroborative |
| Sharpe difference, one 6.25-year block | — | — | SE of SR_a − SR_b ≈ √[(2(1−ρ) + ½(SR_a² + SR_b² − 2ρ²SR_aSR_b))/T] ≈ 0.20 for ρ = 0.9 [Jobson-Korkie/Memmel form, unverified; ρ assumed] | The current 0.11 gap ≈ 0.6 SE: not evidence either way |
| Sharpe difference, 18 years | — | — | SE ≈ 0.12 for ρ = 0.9 ⇒ gaps below ≈ 0.3 are not reliably detectable | Secondary only |

Consequences: model choices are decided on daily covariance scores; portfolio and crisis claims are corroborative and
framed as non-inferiority or sign consistency. Missing delisting returns bias portfolio outcomes upward, more for
strategies holding small or distressed names; the daily risk score is less affected (it scores stocks alive on t+1),
but the liquidity universe still under-represents dead firms, so all claims are "within the Tiingo-survivable
liquidity universe".

### 6.2 Per-layer hypotheses

Notation: "DM" = one-sided Diebold-Mariano with HAC variance on daily differentials; SPA over the variant registry.

| Layer | Hypothesis | Refuted if | If refuted: stopping rule | Next DL formulation |
|---|---|---|---|---|
| L0 | **H0** Rankings are robust to data limits | Any pairwise L2 ranking flips across the three delisting bounds or between liquidity universes of N = 300 and 500 | Claims that flip are reported as data-undetermined; no adoption based on them | — (measurement) |
| L1 | **H1a** Conditional loadings beat the static object (E1) | DM p ≥ 0.05 for every conditional arm (rolling PCA, IPCA) vs static PCA and static mean-B, *and* TOST within a pre-registered margin (1% of the static arm's mean log-score) | The episodic premise has no linear OOS content; E2 still runs, but the ladder moves early to H2 | g_θ (E2), then path encoder (E3) |
| L1 | **H1b** Deep loading map beats IPCA (E2), within each loss of the 2×2 | DM p ≥ 0.05 after SPA over g_θ variants, or calibration outside [0.8, 1.2], or GMV realized variance worse by > 2% | Formulation 1 (scalar-instrument MLP) rejected | **F2:** path encoder (CNN over return windows) feeding g_θ, with matched control = IPCA on the *same frozen learned features* (isolates the deep map from the deep features). **F3:** cross-sectional attention over stocks at t (relational loadings), same control. Ladder exhausted ⇒ L1 deep claim rejected; IPCA loadings serve as the L1 component and the DL claim moves to L2 |
| L2 | **H2** Deep Σ_f,t beats DCC-NL, LW-2020/DGJ and the declared memory blend | The 90% MCS contains a linear arm and no deep arm has a better mean, or DM p ≥ 0.05 vs the best linear arm | Formulation rejected | **F2:** hybrid — DCC-NL target with a deep state-conditioned shrinkage intensity and eigenvalue path. **F3:** regime mixture-of-experts with experts persistent across refits |
| L2 | **H2c** State conditioning improves stress-day forecasts | Stress-day DM p ≥ 0.05 *and* the episode sign test fails (fewer than 7/7 favourable at n = 7, 6/6 at n = 6) | Drop the "protected before reactivation" claim (DVT L78) | Mixture-of-experts (as above); then H2t |
| L2 | **H2t** A tail head beats the Gaussian head on joint lower-tail exceedances | Tail log-score DM p ≥ 0.05 | Keep Gaussian Σ_t; delete tail-dependence claims (DVT L11, L511) | Deep factor-copula head conditioned on state |
| L3 | **H3a** Diversification constraint pays through model error | P1 vs P0 on the same Σ_t: realized variance non-inferiority (+5% margin) fails, or stress-loss sign test fails | Drop the constraint; P0 GMV on the deep Σ_t | Decision-focused end-to-end training of L1/L2 on realized GMV variance (E6) |
| L3 | **H3b** The projector-based measure is more stable than the latent-eigenbasis entropy | Turnover at equal realized variance not lower (paired, HAC) | Keep the measure only if H3a holds | — |
| L4 | **H4** The full system beats every benchmark | SPA p ≥ 0.05 on outer folds, or the holdout sign is opposite | Report the MCS; no deployment claim | Highest-ranked failing layer's next formulation, re-entered through the same nested protocol and the same registry |

**Budget against forking paths.** Each formulation gets at most 8 inner-loop configurations; only the inner winner
reaches the outer folds; every outer-scored variant enters the SPA registry for the rest of the project. Iterating on
the same outer folds after a failure is not allowed without adding the variant to the registry.

---

## 7. Experiment plan

Ordered from cheapest and most decisive to most expensive. Each step is one heavy job at a time, wrapped in
`heavy_lock.py`. "Local" means the 16 GB laptop with `--device mps` or `auto`; never the 7,716-stock universe.

| Step | Content | Gate (pre-registered) | Where | Cost |
|---|---|---|---|---|
| **E0 — Foundations** | Resolve the `results/` deletion. L0: PIT liquidity universe, raw-close level filters, delisting bounds, stress indicator. L4: one harness for all arms, decision states, DM/SPA/MCS, failure recording. L5: single config, manifest, variant registry, loud failures (`covShrinkage`, solver fallbacks). Fix ERC fallback, fast-solver turnover, cardinality at every rebalance. No model change. | All six benchmarks + S&P + incumbent (static mean-B) run end-to-end in the harness with manifests on the minimal real config (50 stocks, 8 years) and on 300 stocks; ERC solves on every fold; two identical runs give identical outputs; synthetic panel with a planted factor structure is scored with the true model ranked first. | Local (minimal config ≈ 13 min, 4.7 GB peak; 300 stocks fits) | Engineering only |
| **E1 — Linear ladder and H1a** | Score, under L4, on 500 stocks 1995-2023: sample + LW; DGJ-PCA static; rolling PCA (504 d); IPCA (same x as E2); price-style model; incumbent static mean-B from the existing pipeline. Same Σ_f estimator (shrunk trailing) and D_t (EWMA). Measure the loss-differential autocorrelation, N_eff and the minimum detectable effect. | Harness gate: bias ratio of the best arm in [0.8, 1.2] (else fix L2/D_t before any DL work). H1a decided per §6.2. Output: the bar (best linear arm) and the measured power. | Local (no training; O(nk²) per date; 500 stocks is not the full universe) | Low |
| **E2 — Decisive DL test (H1b)** | 2 × 2: {IPCA, g_θ MLP (2-3 layers, ≤ 64 units, dropout)} × {reconstruction loss, predictive NLL}; same x, Σ_f, D_t; nested walk-forward, annual refits, 21-day embargo; k* ∈ {3, 5, 8, 12, 20} in the inner loop; 3 seeds. | g_θ beats IPCA within at least the predictive-NLL loss: DM p < 0.05 after SPA; calibration in [0.8, 1.2]; GMV realized variance not worse by > 2%. **Pass ⇒ E4 on g_θ loadings. Fail ⇒ E3.** Inconclusive (power below the measured MDE) ⇒ report, go to E3 without re-running E2 variants. | Local for one seed (small MLP, ≈ 500 × 4,500 × 13 inputs ≈ 120 MB); Colab for the seed sweep | Medium |
| **E3 — Path encoder (H1b, F2)** | CNN window encoder (current architecture, shorter windows allowed) trained under the predictive NLL as a feature extractor for g_θ; control = IPCA on the same frozen features. | Same gate as E2, deep vs linear on the same features. Fail ⇒ F3 (cross-sectional attention); ladder exhausted ⇒ IPCA loadings become the L1 component and the deep claim moves to L2. | Colab (stride-1 windows for 500 stocks ≈ 12 GB in float32); local smoke on 50 stocks with stride 5 | High |
| **E4 — Deep Σ_f dynamics (H2, H2c)** | On the best loading map from E2/E3 (or IPCA): GRU/attention Σ_f,t vs shrunk trailing, declared memory blend, DCC-NL (Engle-Ledoit-Wolf 2019). | §6.2 H2 and H2c. Fail ⇒ hybrid DCC-NL + deep correction, then mixture-of-experts. | Local single seed; Colab for sweeps and DCC-NL on 500 stocks if slow | Medium |
| **E5 — Allocation (H3a, H3b)** | Policies P0-P5 on the adopted Σ_t, one harness, costs, three delisting bounds. | §6.2 H3a/H3b; H0 stability across bounds. | Local | Low |
| **E6 — Decision-focused (L3 F2)** | Only if H3a fails or as the next formulation after E4 fails: end-to-end training through a differentiable GMV layer on realized variance. | Beats the two-stage system on realized GMV variance (DM) without calibration loss. | Colab | High |
| **E7 — Tail head (H2t)** | Only if H2 passes. | §6.2 H2t. | Colab | High |
| **E8 — End-to-end and holdout (H4)** | Frozen pipeline on outer folds; SPA vs every benchmark; then the holdout once. | SPA p < 0.05 and holdout sign consistent; otherwise report the MCS. | Colab for the full retrain sweep; local for scoring | Medium |

**Why E2 is decisive:** it is the first step that answers the question the user fixed (does a *deep* conditional
model beat its matched linear control?), on the evidence with the most power, and it fits the laptop for one seed.

---

## 8. Appendix A — Traceability of confirmed critical findings

Element codes refer to this document: R = root cause (§3), D = divergence decision (§4), L0-L5 = layers (§5),
H = hypotheses (§6), E = experiments (§7). Verification column: **T** = confirmed by the theory audit's orchestrator;
**V** = verified by v5's orchestrator; **Here** = re-checked for this document; **—** = not re-verified (accepted on
consistency with verified findings, at the stated risk).

### A.1 Theory audit — 43 confirmed critical findings

| Finding | Short claim | Addressed by | Verif. |
|---|---|---|---|
| U01-CRI-02 | H depends on latent scale | D1 (asset-space projectors), L1 identification | T, Here (C7) |
| U01-CRI-03 | Lifetime mean dilutes episodic loadings (q²) | R2; L1 conditional B_t; static mean-B as baseline (E1) | T |
| U02-CRI-02 | Dormant factor ≈ 20% at reactivation | D1 (equal contributions not default); H2c; H3a | T, Here (C22) |
| U02-CRI-03 | Gaussian object has no tail dependence | L2 tail head gated by H2t; claims removed until then | T |
| U02-CRI-04 | Constant Σ_z, D_ε: contagion is a functional-form artefact | L2 state-conditional Σ_f,t, D_t | T |
| U03-CRI-02 | Cap from dividend-adjusted price | D9 (spec defect; code differs; liquidity universe) | T, Here (C11) — corrected |
| U03-CRI-03 | Shumway imputation applied without cause | D9 delisting bounds; H0 | T |
| U04-CRI-01 | Input invariant to c·r; RV cannot restore scale | L1 scale contract s_i,t | T |
| U04-CRI-02 | N = 7.06M vs N_eff ≈ T_hist/T | R5; D4 single effective-observation budget | T, Here (C16) |
| U05-CRI-01 | No term ties μ to a cross-sectional loading | R1; L1 predictive NLL | T, Here (C4) |
| U05-CRI-03 | AU depends on D = T·F | D4 (AU removed) | T |
| U05-CRI-04 | EP_VAE ≤ EP_PCA at equal dimension (Ky Fan) | EP gate removed; L4 proper scores | T |
| U06-CRI-01 | AU depends on the latent basis | D4 | T |
| U06-CRI-02 | Guard binds; count = f(sample length) | D4, D7 | T, Here (C20) |
| U06-CRI-03 | Selection on reported OOS (component e) | L4 nested selection | T |
| U07-CRI-01 | Co-movement term ~10⁻³ of reconstruction | Scaffold removed (L1) | T |
| U08-CRI-01 | Co-movement fixed point ≠ model correlation | Scaffold removed | T |
| U09-CRI-02 | Validated ≠ deployed (curriculum clock) | L4 "one clock"; INV-010/011 retired | T, Here (C14) |
| U10-CRI-01 | E[R²] = K/(n−1) under the null | L4 OOS proper scores replace in-sample R² | T |
| U10-CRI-03 | Mean aggregation charges loading variation to D_ε | L1 conditional B_t | T |
| U11-CRI-01 | Printed AU_max formula wrong | D7 (84; retired) | T, Here |
| U11-CRI-03 | Shrinkage δ moves H; PCA arm unshrunk | D1 (asset-space measure); L4 equal regularization budget | T (narrowed) |
| U12-CRI-01 | 38.8% vs 1.25% target at reactivation | D1; H2c; H3a | T |
| U12-CRI-03 | H ill-posed under degeneracy | D1 block projectors | T (narrowed) |
| U12-CRI-04 | Ĥ_OOS on frozen model, degree-0 homogeneous | L4: Ĥ removed as metric | T |
| U13-CRI-02 | With μ = 0, α > 0 weakly dominated | D1 status change; H3a vs α = 0 | T, Here (argument re-derived; scope corrected) |
| U13-CRI-03 | = U12-CRI-03 | D1 | T (narrowed) |
| U14-CRI-01 | Best-of-32 optimism ≈ 2.07σ | L4 nested; SPA registry | T |
| U14-CRI-04 | Ĥ non-comparable across AU | L4 | T (narrowed) |
| U15-CRI-01 | Selection on reported OOS | L4 nested | T |
| U15-CRI-02 | Ĥ undefined for 1/N, inverse-vol | L4 common reference covariance | T (narrowed) |
| U15-CRI-03 | p ≥ 0.05 at n = 34 read as equivalence; power 0.40 | L4 TOST; §6.1 power; D2 | T |
| U16-CRI-01 | = U15-CRI-02 | L4 | T (narrowed) |
| U16-CRI-02 | = U14-CRI-01 | L4 | T |
| U16-CRI-03 | Wilcoxon on dependent folds: size ≈ 0.20 | L4 DM-HAC, block bootstrap | T |
| U17-CRI-02 | = U12-CRI-03 | D1 | T (narrowed) |
| U17-CRI-03 | Current-vol rescaling is reactive | L2 dynamics; "anti-cyclical" becomes H2c | T |
| U17-CRI-05 | Five iterations on same folds: FWER ≈ 0.23-0.26 | L5 registry; SPA | T |
| U18-CRI-01 | = U04-CRI-02 + risk-model parameters unchecked | D4 budget covers L1 and L2 parameters | T |
| U18-CRI-02 | = U14-CRI-01 | L4 | T |
| U18-CRI-03 | AU table uses √(2N/r) | D7 | T (narrowed), Here |
| U19-CRI-01 | = U11-CRI-01 (83 with N = 7,057) | D7 (N = 7,057 is the wrong N) | T, Here |
| U19-CRI-02 | Burda misattribution | R3; DVT L420/L1662 citation removed with AU | T |

**Downgraded by the theory audit (14):** U01-CRI-01 (L1 estimand), U02-CRI-01 (L1), U03-CRI-01 (stated limit, H2c),
U05-CRI-02 (L2 residual check), U07-CRI-02 (L1), U08-CRI-02 and U08-CRI-03 (γ and scaffold removed), U09-CRI-01 (L4),
U10-CRI-02 (D4), U11-CRI-02 (R5), U11-CRI-04 (L2 leverage-corrected residuals), U11-CRI-05 (EP gate removed),
U13-CRI-01 (D1), U14-CRI-03 (L5).
**Rejected by the theory audit (2):** U12-CRI-02, U18-CRI-04 — rejection reasons checked and accepted (sample vs true
eigenvalues; σ² = MSE/D under the INV-001 violation).

### A.2 v5 audit — 78 critical findings

| Findings | Short claim | Addressed by | Verif. |
|---|---|---|---|
| U01-CRI-01 | |β′| ∝ λ^{-1/2}: short-volatility exposure to dormant factors | D1; H2c; H3a | Here (C22) |
| U01-CRI-02, U04-CRI-01, U02-CRI-02 | Per-window z-scoring removes level; RV channel redundant | L1 scale contract; RV channel kept only if ablation in E3 shows value | — (consistent with T U04-CRI-01) |
| U01-CRI-03, U08-CRI-04, U15-CRI-04 | Half-life 60 windows = 60 days | R2; aggregation removed | V, Here (C5) |
| U01-CRI-04 | Σ_z half-life 252 vs doctrine 0 | L2: memory chosen by the score (D5) | — |
| U01-CRI-06 | Covariance cannot represent tail risk | H2t | — (= T U02-CRI-03) |
| U02-CRI-01 | No characteristic/sector data; interpretation unit void | D10 price-style control; industry deferred | — |
| U02-CRI-03 | Never-co-moving factor invisible | Stated limit of price-only data; H2c scope | — |
| U03-CRI-01 | PIT universe orphaned | L0 | V, Here (C12) |
| U03-CRI-02 | Full-sample universe | L0, D9 | V, Here (C10) |
| U03-CRI-03 | Cap is a turnover proxy | D9 liquidity universe | Here (C11) |
| U03-CRI-04 | Delisting NaN; imputation never fires | D9 bounds; H0 | V, Here (C11) |
| U03-CRI-05 | $1 filter on back-adjusted prices | L0 raw-close level filters | — |
| U04-CRI-03 | Trajectory discarded by aggregation | L1 conditional B_t | — |
| U05-CRI-01, U05-CRI-02, U08-CRI-05 | Not a factor model; B from marginal law | R1; L1 | V, Here (C4) |
| U05-CRI-03 | No auto-pruning (47/52 above threshold) | D4 | — |
| U05-CRI-04 | AU/K ≤ 0.5 by construction | D4, D7 | Here (C20) |
| U06-CRI-01, U16-CRI-04 | γ no-op (vix_data=None) | γ removed; L0 stress indicator | V, Here (C9) |
| U07-CRI-01 | Phase 3 never ran (curriculum vs patience) | Curriculum removed; one clock | — |
| U07-CRI-02 | Phase B executes a different curriculum | L4 one clock | Here (C14) |
| U07-CRI-03 | Positional 90/10 split holds out stocks | L4 time-based inner split only | Here (C15) |
| U08-CRI-01, U08-CRI-02, U08-CRI-03, U17-CRI-03 | AU selection not rotation-invariant; cut unresolvable | D4 | — (convergent with T U06-CRI-01) |
| U09-CRI-01 | var ratio 2.74 | L4 calibration gate; E1 harness gate | V, Here (C2) |
| U09-CRI-02 | Spec LW, run spiked, default NL falls back | L5 fail loudly; D10 baselines | Here (C17) |
| U09-CRI-03 | n_signal from raw vs effective γ | DGJ kept as baseline only; fix in E0 | — |
| U09-CRI-04 | Off-by-one in signal slice | Slice removed with PC1 exclusion (D1) | V, Here (C8) |
| U10-CRI-01 | PC1 (81%) excluded | D1 explicit market block | V, Here (C2, C8) |
| U10-CRI-02, U10-CRI-03, U10-CRI-04 | Block-normalized entropy; idio weight in nats; undefined mode | D1; L3 named arms | — |
| U10-CRI-06, U11-CRI-02, U11-CRI-07, U15-CRI-06 | α calibration inoperative; flat frontier | α removed; L3 constraint form | — |
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
| U13-CRI-05 | VAE vs PCA contrast unidentifiable | E2 factorial; matched controls | — |
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
| U16-CRI-05 | Decision rules pre-authorize the conclusion | L4 pre-registration | — |
| U17-CRI-01 | Conformance = whatever the code does | L5 executable spec + ADRs | — |
| U17-CRI-02 | Deployed objective (two-layer, idio weight) not in spec | D1; L5 | — |
| U17-CRI-04 | One-fold run, checkpoint of unknown provenance | L5 manifest; results deletion flagged | V |

No confirmed critical finding of either audit is rejected outright. Four are **corrected in scope**: theory U03-CRI-02
(spec, not code), theory U13-CRI-02 (true of every non-GMV portfolio), v5 P4 and P6 (see D11).

---

## 9. Appendix B — References

- **Verified for this document (web, 2026-11-08):** Gu, Kelly & Xiu (2021, J. Econometrics 222(1):429-450);
  Engle, Ledoit & Wolf (2019, JBES 37(2):363-375); Meucci, Santangelo & Deguest (2015, *Risk*); Duan et al. (2022,
  AAAI 36(4):4468-4476, FactorVAE).
- **Verified by the theory audit (its INDEX §3; not re-checked):** Kelly-Pruitt-Su 2019; Embrechts-McNeil-Straumann
  2002; Shumway 1997; Shumway-Warther 1999; Ledoit-Wolf 2003/2004; Diebold-Mariano 1995; White 2000; Hansen 2005;
  Hansen-Lunde-Nason 2011; Politis-Romano 1994; Patton 2011; Engle-Colacito 2006; Lucas et al. 2019; Tipping-Bishop
  1999; Burda et al. 2016; Salinas et al. 2019; Oh-Patton 2017; Amos-Kolter 2017; Cawley-Talbot 2010; Meucci 2009;
  Lohre-Opfer-Ország 2014.
- **[unverified]:** Donoho-Gavish-Johnstone 2018; Ledoit-Wolf 2020; Lettau-Pelger 2020; Roncalli 2013; López de Prado
  2018; Bailey-López de Prado 2014; Davis-Kahan constants; Jobson-Korkie/Memmel variance; De Nard et al. 2022.
