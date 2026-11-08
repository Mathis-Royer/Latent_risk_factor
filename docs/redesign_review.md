# Independent review of the redesign of record

> **Document under review:** [REDESIGN.md](REDESIGN.md) (2026-11-08, "proposal, nothing implemented").
> **Second input, used as evidence and checked in turn:** [architecture_survey.md](architecture_survey.md) (written
> without reading the redesign).
> **Reviewer stance:** independent; neither the owner's intuition nor the redesign is taken as the default. A
> claim is kept only if it survives a re-derivation, a code check, or a source read for this review.
> **Date:** 2026-11-08. **Scope:** review only. The redesign is not edited; section 3 gives replacement text for the
> writer who applies the review.

Conventions. Line references `R:n` point to `docs/REDESIGN.md`, `S:n` to `docs/architecture_survey.md`. Classes:
**M** = mathematical or internal fact (re-derived or checked at the cited line), **L** = literature-dependent claim,
**C** = design choice. Verdicts: keep / correct / replace / reject / owner decision. Literature status: **V-abs**
(abstract and record read for this review), **V-full** (body or tables read for this review), **V-sec** (secondary
source only), **[unverified]** (from memory or not found). "[own analysis]" marks reasoning of this review that no
source states.

What was actually checked for this review:
- Code lines: `src/walk_forward/selection.py:110-136` (C3), `src/integration/pipeline.py:310-320` (C16),
  `src/risk_model/covariance.py:605-622` (C6), `src/integration/pipeline.py:2284-2304` (C8),
  `docs/strategy_philosophy.md` §2-§4, DVT L9-12 and L76-79, and `git status` / `git ls-files results`.
- Numerical checks (numpy, scratchpad, no training): the C7 invariance claim on a random 30×3 example with a
  non-orthogonal latent map (asset-space entropy 0.6614 → 0.6614; latent-eigenbasis entropy 0.680 → 0.040;
  Σ_assets identical to 4e-15); a Monte Carlo of the Sharpe-difference standard error (4,000 replications).
- Arithmetic re-derived: C20 (Kish N_eff = (1+λ)/(1−λ) = 727.1 for half-life 252), C22, D7, the §6.1 sign tests, the
  MDE formula, the best-of-32 optimism (≈ 2.07σ), the FWER of five looks (1 − 0.95⁵ = 0.226), the Wilcoxon minimum
  p (2/2⁹ = 0.0039 > 0.05/18 = 0.0028), the power of 0.40-0.42 at n = 34, d = 0.3.
- Literature: 30 sources opened (section 5).

---

## 1. Verdict in 10 lines

1. **Stands:** the diagnosis (R1-R8), the arithmetic (C1-C22, D7, power and sign-test numbers), the asset-space,
   reparametrization-invariant risk measure (D1, verified numerically), the proper-scoring evaluation with DM/SPA/MCS,
   nested selection and an "inconclusive" state (L4, D6), the data contract (L0, D9) and governance (L5).
2. **Must change, L1:** the first deep formulation (an MLP on 13 rank-normalized scalar characteristics, E2) is
   structurally unable to represent the two things the doctrine asks for: industry-like *structural* co-movement
   (no scalar characteristic encodes it) and *regime* switching (per-date ranks erase market-wide levels; no state
   input). Its linear parent has a weak prior too: in Gu-Kelly-Xiu the nonlinear beta network does not beat IPCA on
   total R² (12.6% vs 13.3%, 3 factors). Replace it with a conditional probabilistic factor model (path encoder +
   characteristics + FiLM state conditioning, ~30k parameters) that keeps a structural block, with a matched linear twin.
3. **Must change, L2:** add the strongest published deep covariance result on price-only US equities (Bongiorno et
   al.: 11.9% vs 12.4-12.6% volatility on 1,000 stocks) as a deep arm; replace the free Cholesky head by a
   state-modulated factor DCC; add a deep per-stock volatility head, where price-only nonlinear gains are best documented.
4. **The VAE has a precise place:** latent = date-level factor returns f_t and a regime s_t (E7), not per-stock
   windows. The Gaussian conditional factor model is already its exact-posterior special case.
5. **Must change, evaluation:** require "deep beats its linear twin" *and* "deep is in the MCS against the best E1
   arm"; move DCC-NL, QIS and AFM-DCC into E1; fix three non-identifiable or underpowered gates (H0 on delisting
   bounds, the TOST margin, the 2% GMV-variance margin); minor fix to the Sharpe SE (0.18, not 0.20).
6. **Stale fact:** `results/` is no longer deleted in the working tree (§2, L5, E0 must be updated).
7. **Price/volume-only inputs:** they support a deep *covariance* claim (Bongiorno; volatility forecasting) better
   than a deep *loading* claim; NeuralFactors' returns-only ablation loses most of its gain, but that ablation also
   removes industry and volume. The deep loading claim is plausible only with a cross-sectional structural block.
8. **Order:** E0 → E1 (all linear bars) → E2 (2×2 {linear, deep} × {no state, state}) → E3 (learned structural block)
   → E4 (L2 dynamics, volatility head, spectral cleaner) → E5 → E6/E7 → E8.
9. **Owner decisions (section 4):** the fallback when an L1 ladder is exhausted; whether a fixed statistical block
   inside the deep model is acceptable; the VAE scope; a policy-level memory arm; industry codes as inputs.
10. **Complexity:** the evidence favors small, structured networks (30-60k parameters, seed ensembles). It does not
    support larger or deeper models on this information set; the owner's "too basic" intuition is right about the
    information content of the inputs, not about depth.

---

## 2. Element-by-element table

### 2.1 Front matter, §1-§2

| ID | Class | Verdict | Reason | Evidence |
|---|---|---|---|---|
| Header, binding decision | C | keep | Consistent with the owner's decisions | R:8-10 |
| §1 shared diagnosis | M | keep | Re-checked through C4 and the code of the encoder; matches the survey's independent §1 | R:16-22; S:63-101 |
| §1 evidence paragraph | M | correct (minor) | Numbers confirmed; "≈ 0.6 standard errors" holds, but the SE is 0.18 not 0.20 (see §6.1 row) | R:24-29; Monte Carlo in this review |
| §1 "What changes" item 1 | C | replace | The learned object must include a structural block and state conditioning (section 3.1) | R:32-34 |
| §1 items 2-4 | C | keep | Asset-space quantities, diversification as hypothesis, proper scores | R:35-41 |
| §1 decisive experiment E2 | C | replace | See E2 row and section 3.6 | R:43-50 |
| §2 note on `results/` deletion | M | correct | `git status` shows only `.claude/settings.json` and `docs/architecture_survey.md`; `results/` and checkpoints are tracked and present | R:56-59; `git ls-files results` |
| C1, C2, C4, C5, C9-C15, C17-C19 | M | keep | Accepted on the redesign's citations; not all re-opened here | R:63-81 |
| C3 | M | keep | Verified: `n < 5 → continue`, default `"B"`, `pca_tie = wins <= 1 → "C"` | `src/walk_forward/selection.py:110-136` |
| C6 | M | keep | Verified: `eigh(Sigma_z)`, `B_prime_port = B_A_port @ V` | `src/risk_model/covariance.py:605-622` |
| C7 | M | keep | Reproduced independently with a non-orthogonal latent map: asset-space entropy invariant, latent version not | numpy check (section 0) |
| C8 | M | keep | Verified: `signal_start = 1`, `signal_end = n_signal + 1` when the market intercept is on | `src/integration/pipeline.py:2296-2299` |
| C16 | M | keep | Verified: `N_capacity = n_stocks * (T_hist - T + 1)` | `src/integration/pipeline.py:318-319` |
| C20 | M | keep | (1+λ)/(1−λ) with λ = 0.5^{1/252} = 727.1; ⌊√727.1⌋ = 26 | re-derived |
| C21 | M | keep | Accepted | R:83 |
| C22 | M | keep | (1/p)/(AU − 1 + 1/p) = 1/(AU·p + 1 − p); 0.202 at AU = 80, p = 0.05 | re-derived |
| "Literature checked" paragraph | L | keep, extend | GKX and FactorVAE confirmed here (V-abs/V-sec); add the sources of section 5 | R:86-90 |

### 2.2 Root causes (§3)

| ID | Class | Verdict | Reason | Evidence |
|---|---|---|---|---|
| R1 mechanism | M | keep | No shared z_t in the per-window ELBO (C4); E[R²] = K/(n−1) under the null and the Ky Fan bound are standard | R:111-116 |
| R1 "What DL can add" | L | correct | GKX is the right template, but its own evidence is against nonlinear loadings for *total* R² (the covariance-relevant metric): 12.6% (autoencoder) vs 13.3% (IPCA), 3 factors; its gain is in predictive R² (means). State this, and name the structural-block gap | R:119-121; GKX 2021 (V-sec for the figures) |
| R2 | M + L | keep | Mechanism and the code fact (C5) hold; zero tail dependence of the Gaussian for ρ < 1 is standard | R:123-133 |
| R3 | M | keep | Basis dependence of per-coordinate KL is correct; C20 confirms the count is produced by caps | R:135-148 |
| R4 | M | correct (add) | Correct as stated. Add: the λ^{-1/2} amplification is a property of equal contributions over *any* principal basis, including the redesign's asset-space projectors, and GMV loads even harder on small eigen-directions (w ∝ Σ⁻¹1, exposure ∝ λ⁻¹). The protection must come from the covariance estimator's small eigenvalues (L2), not from the measure [own analysis; Ledoit-Wolf 2017/2020 rationale] | R:150-162; S:89-94 |
| R5 | M | keep | Arithmetic re-checked | R:164-171 |
| R6 | M | keep | Best-of-32 ≈ 2.07σ, FWER ≈ 0.23 re-derived | R:173-184 |
| R7 | M | keep | Accepted | R:186-193 |
| R8 | M | correct (minor) | The "surviving artifacts are now deleted" sentence is stale | R:198-199 |
| Meta-cause | C | keep | — | R:204-205 |

### 2.3 Divergence decisions (§4)

| ID | Class | Verdict | Reason | Evidence |
|---|---|---|---|---|
| D1.1 projector measure | M | keep | Σ_sys = BΣ_fBᵀ is invariant to B → BA, Σ_f → A⁻¹Σ_fA⁻ᵀ, hence so are its spectral projectors; verified numerically | R:226-231 |
| D1.2 total-risk decomposition | C | keep | Fixes C8 | R:232-233 |
| D1.3 GMV s.t. ENB ≥ e | C | correct | Three additions: (a) for long-only portfolios the market block takes most of Σ_sys, so e must be set relative to the per-date maximum feasible ENB, otherwise the constraint is infeasible or always slack; (b) the constraint is nonconvex, SCA gives a local solution, report multi-start dispersion; (c) the R4 amplification note above | R:234-237 [own analysis] |
| D1 MT-bets as reported measure | L | keep | Meucci-Santangelo-Deguest 2015 confirmed (V-abs: "uncorrelated factors optimized to track the factors used to allocate") | Risk 2015 |
| D2 three outcomes, ladder | C | keep, owner decision on exhaustion | The logic is sound. "IPCA loadings serve as the L1 component" (R:499, R:523) is a linear component at layer level; whether that is compatible with "linear never the fallback" is the owner's call (section 4, OD1) | R:241-250 |
| D3 VAE status | C | correct | Agree the per-window ELBO is not the loading objective. Add where the VAE does belong: latent = (f_t, s_t), E7. The CNN encoder survives only with the scale contract (no per-window z-score) and L ≤ 252 | R:252-257; S:159-189 |
| D4 factor count | C | keep | Inner-loop log-score selection is the literature's recommendation (S:170); express the budget in dates × factors, not windows | R:259-264 |
| D5 memory | C | keep, extend | Add a *policy-level* memory arm (Chow et al. 1999 blend at fixed π), owner decision OD4 | R:266-272; S:191-211 |
| D6 primary score | C | keep, extend | Add the volatility/correlation decomposition of the log-score (S:426-428) | R:274-280 |
| D7 AU_max_stat = 84 | M | keep | √29,237 = 170.99 → 84.99 → 84; 84·85/2 = 3,570 ≤ 3,654.5 < 3,655 | R:282-291 |
| D8 decisive experiment | C | replace | The {linear, deep} × {LS, NLL} factorial isolates the loss, which a linear arm can isolate alone in E1. The open question for the doctrine is {linear, deep} × {no state, state}, with a structural block (section 3.6) | R:293-302 |
| D9 liquidity universe, delisting bounds | M + C | keep | C11 checked by the redesign; correct reading of the data | R:304-313 |
| D10 imports, price-only style model | C | keep | — | R:315-327 |
| D11 self-assessment corrections | M | keep | Consistent with C7, C16 | R:329-336 |

### 2.4 Target architecture (§5)

| ID | Class | Verdict | Reason | Evidence |
|---|---|---|---|---|
| §5 diagram, one estimand | C | keep | — | R:342-355 |
| L0 | C | keep, extend | Add the market-state vector m_t (8 PIT variables) to the panel output; it feeds L1 and L2 | R:357-363 |
| L1 contract B_t = s_t ⊙ g_θ(x_t) | C | replace | Lacks a structural block and any market-state input (section 3.1) | R:366-368 |
| L1 instruments (13, ranked per date) | C | keep as one input group | Ranking is right for characteristics (IPCA/GKX), but ranks erase market-wide levels, so a separate unranked state vector is needed | R:369-371 |
| L1 loss: Gaussian NLL via Woodbury | M | keep | O(nk²) per date by Woodbury and the determinant lemma is correct. Specify how Σ_f,t and D_t are computed inside training (they depend on θ) | R:372-374 |
| L1 alternative loss (GKX LS) | C | move to E1 | Loss effect testable on the linear arm alone | R:374-375 |
| L1 identification | M | keep | Only Σ_t is used; GLS factor returns make BΣ_fBᵀ invariant to B → BA | R:376-377 |
| L1 linear counterpart IPCA | L + C | correct | IPCA confirmed (V-abs). The twin must be IPCA *with the same inputs* (characteristics, linear path summaries, characteristic × state interactions) under the same NLL and the same structural block | R:378-379; KPS 2019 |
| L1 "Why DL can add value" | L | correct | Argument requires state input and cross-sectional context that F1 lacks | R:380-382 |
| L2 contract | C | keep | — | R:385-386 |
| L2 Σ_f,t ladder (GRU → Cholesky) | C | replace | A free Cholesky head over k(k+1)/2 entries driven by a few crisis episodes is the highest-variance option; the survey and Boulet's evidence favor a DCC/EWMA recursion with state modulation | R:387-390; S:238-239, S:404-405 |
| L2 D_t (EWMA/GARCH) | C | extend | Add a deep per-stock volatility head: the best-documented nonlinear gain from price-only inputs (Christensen-Siggaard-Veliyev 2023, V-abs via survey) | R:391-392 |
| L2 tail head | C | merge into E7 | Student-t factors and residuals with IWAE are the VAE rung (NeuralFactors) | R:393-394 |
| L2 missing arm | L | add | Neural rotation-invariant covariance cleaner (Bongiorno et al.), price-only, 1,000 US stocks, 2000-2024: volatility 11.9% vs 12.4% (DCC), 12.6% (QIS); turnover 57% vs 23% (QIS) | arXiv 2507.01918 (V-full, Table 2) |
| L3 contract, hard constraints | C | keep | Fixes C18 | R:397-399 |
| L3 policies P0-P5 | C | keep, add P6 | P6 = GMV on a declared calm/stress blend (owner decision OD4) | R:400-402 |
| L3 decision-focused formulation | L + C | keep | OptNet (theory audit); DFL for GMV now has direct support (Kim-Tae-Lee 2025, V-abs); risk-budgeting layers (Uysal-Li-Mulvey, V-abs, small N) | R:403-404 |
| L4 one harness, scores, selection, inference, states | C + L | keep, extend | Add: log-score decomposition; 5-seed ensembles in Σ space; Ledoit-Wolf (2008) studentized bootstrap for Sharpe; adoption = beats twin *and* in the MCS vs the best E1 arm; first-occurrence crisis test | R:408-421 |
| L5 | C | keep, correct | Remove "resolve the current deletion of `results/` first" (resolved) | R:424-433 |
| INV table | M | keep | Consistent with the layers | R:435-445 |

### 2.5 Kept / removed / merged table (§5)

| Row | Class | Verdict | Reason |
|---|---|---|---|
| μ = 0 default | C | keep | — |
| Price-only inputs; shape/scale separation | C | keep, owner decision OD5 | Evidence that industry information carries much of the deep gain (NeuralFactors, DRM) |
| CNN window encoder → path-feature extractor (E3) | C | correct | Becomes the temporal encoder inside E2 (both state arms), L ≤ 252, trailing-volatility scaling instead of per-window z-score; not a separate later rung |
| Per-window ELBO as loading objective | C | keep (remove) | R1 |
| Co-movement scaffold; crisis γ | C | keep (remove) | R1 |
| Lifetime aggregation into static B | C | keep (remove; baseline) | R2 |
| AU, KL threshold, AU_max_stat, K cascade | C | keep (remove) | R3, D4, D7 |
| Entropy over eig(Σ_z), PC1 exclusion, α_eff, frontier | C | keep (remove) | R3, R4, D1 |
| Exact gradients, SCA, projections | C | keep | Verified module exists (`src/portfolio/entropy.py`, `sca_solver.py`) |
| DGJ, LW, PCA-RMT baselines | L | keep, extend | Add QIS, DCC-NL, AFM-DCC, IPCA+PCA to E1 |
| Walk-forward, embargo, holdout | C | keep | — |
| Phase A on OOS, Ĥ, scenario engine, Wilcoxon | C | keep (remove) | R6 |
| Separate harness, ERC → 1/N, NaN → 0 | C | keep (remove) | R6, C13 |
| Full-sample universe, back-adjusted $1 | C | keep (remove) | R7 |
| Capacity ratio on windows → one budget | C | keep (merge) | R5 |
| Duplicated parameters → one config | C | keep (merge) | R8 |
| DVT §6.7, scenario C | C | keep (remove) | Owner decision |
| *Missing row* | C | add | "Fixed statistical (residual-PCA) block inside every L1 arm — add; replaced by a learned block only if E3 passes" |
| *Missing row* | C | add | "Neural spectral cleaner (Bongiorno et al.) — add as L2 deep arm" |
| *Missing row* | C | add | "VAE — moved to the date-level latent (f_t, s_t), E7" |

### 2.6 Hypotheses, power, stopping (§6)

| ID | Class | Verdict | Reason | Evidence |
|---|---|---|---|---|
| §6.1 daily differentials row | M | keep | MDE = (1.96 + 0.84)/√N_eff = 2.8/√N_eff; N_eff 100-250 from 4,500 days if integrated autocorrelation time is 18-45 days; "must be measured" is right | R:479 |
| §6.1 holdout row | M | keep | Same arithmetic | R:480 |
| §6.1 stress-episode sign test | M | keep | 1/128 = 0.0078; 8/128 = 0.0625; 1/64 = 0.016 | R:481 |
| §6.1 Sharpe rows | M | correct | The SR² term of the Jobson-Korkie/Memmel variance is in per-period units; with annual Sharpe ratios it must be divided by 252, so it is negligible. SE ≈ √(2(1−ρ)/years) = 0.18 (6.25 y) and 0.105 (18 y). Monte Carlo (iid normal, ρ = 0.9): 0.177 and 0.103. The redesign's 0.20 / 0.12 overstate by ~10%; conclusions unchanged (0.11 ≈ 0.6 SE; MDE ≈ 0.29 over 18 y) | R:482-483; scratchpad Monte Carlo; formula as cited by Ledoit-Wolf 2008 (formula itself V-sec) |
| §6.1 survivorship paragraph | C | keep | — | R:485-489 |
| H0 | M | correct | Delisting bounds change only portfolio returns of names that stop trading; the daily log-score at t+1 is computed on names alive at t+1, so L2 rankings cannot flip across bounds by construction. Apply the bounds to L3/L4 portfolio rankings; test L2 rankings across universe sizes and sub-periods | R:497 |
| H1a | C | correct | Keep; the TOST margin "1% of the static arm's mean log-score" scales with n and units and is not interpretable; use nats per stock-day | R:498 |
| H1b | C | replace | Split into H1b (deep vs twin, no state), H1c (state conditioning), H1d (learned structural block); add the MCS adoption condition | R:499 |
| H2 | C | correct | Keep the test; add AFM-DCC, QIS as bars and the neural cleaner as a deep arm (H2n) | R:500 |
| H2c | C | keep, extend | Add the first-occurrence test (walk-forward predicts 2008 with only 1995-2007 in training) | R:501 |
| H2t | C | merge | Into H2v (regime/Student-t VAE) | R:502 |
| H3a | C | correct | The +5% non-inferiority margin on realized variance is close to the detection limit (SE of the log variance ratio ≈ 1.5-4.5% depending on the correlation of the two portfolios [own analysis, S:450-454]); fix the margin after E1 measures that SE | R:503 |
| H3b | C | keep | — | R:504 |
| H4 | C | keep | — | R:505 |
| Forking-path budget (8 configs) | C | keep | Count the factorial cells separately; every seed ensemble counts as one variant | R:507-509 |

### 2.7 Experiments (§7)

| ID | Class | Verdict | Reason | Evidence |
|---|---|---|---|---|
| E0 | C | correct | Remove "resolve the `results/` deletion"; add the market-state vector m_t and a planted *regime-dependent* factor structure to the synthetic test | R:520 |
| E1 | C | correct | Add QIS, DCC-NL, AFM-DCC (linear L2 bars belong at the start, not in E4), IPCA-LS vs IPCA-NLL (loss effect), IPCA + residual PCA (structural block) | R:521 |
| E2 | C | replace | See section 3.6 | R:522 |
| E3 | C | replace | Learned structural block; the "≈ 12 GB" cost assumes materialized stride-1 windows: building windows per date on the fly needs 500 × 252 × 5 floats ≈ 2.5 MB per batch, so local runs are feasible [own analysis] | R:523 |
| E4 | C | correct | Add the deep volatility head and the neural cleaner; DCC-NL moves to E1 | R:524 |
| E5 | C | keep, add P6 | — | R:525 |
| E6 | C | keep | — | R:526 |
| E7 | C | replace | Regime-mixture Student-t VAE (absorbs the tail head) | R:527 |
| E8 | C | keep | — | R:528 |
| "Why E2 is decisive" | C | correct | E2 remains decisive for L1 under the new definition | R:530-531 |
| Appendix A | M | keep | Traceability only; not re-audited line by line | R:535-659 |
| Appendix B | L | correct | Move DGJ 2018 and LW 2020 to verified (this review); add section 5 sources | R:663-674 |

### 2.8 Claims of the survey checked

| Survey claim | Verdict | Evidence |
|---|---|---|
| NeuralFactors returns-only validation joint NLL 0.466 vs 0.324 | Confirmed (Table 3). Caveats: the returns-only arm also drops industry, GICS indices and volume; the paper's prose gives 0.3618 for a returns-only figure, inconsistent with its table; the returns-only model is never compared with PPCA on the same split | arXiv 2408.01499 (V-full) |
| NeuralFactors uses fundamentals and options | Confirmed: financial ratios, industry, GICS indices, VIX, put/call open interest | same |
| Bongiorno et al. figures | Confirmed (11.9% / 12.4% / 12.6% / 12.5%; Sharpe 1.058 vs 0.85-0.94; turnover 57% vs 23%). Venue "J. Finance and Data Science" not visible on arXiv v3 | arXiv 2507.01918 (V-full) |
| DRM figures and GAT ablation | Confirmed. Omitted by the survey: DRM inputs are 10 Barra CNE5 style factors and the covariance step adds country + 29 industry factors, so DRM is not price-only and its small GAT gain is measured *with* industry already in the model | arXiv 2107.05201v2 (V-full) |
| Pelger-Xiong VIX-state factors explain more variation | Confirmed | arXiv 1807.02248 (V-abs) |
| Lucas et al.: AU tied to noise level | Partly confirmed (abstract: linear-VAE link to pPCA, collapse explained by the linear analysis) | arXiv 1911.02469 (V-abs) |
| Sharpe SE √(2(1−ρ)/years) | Confirmed by Monte Carlo | this review |
| CPDFM per-stock encoder suffices; cross-sectional block "first to drop" | **Rejected as stated**: the DRM ablation it relies on had industry factors present; with price-only data a per-stock encoder cannot form low-rank industry loadings (section 3.1) | [own analysis] |

---

## 3. Replacement text, ready to paste

### 3.1 New §5 L1 (replaces R:365-382)

```markdown
### L1 — Conditional loading map (DL layer 1)

**Contract.** For each date t, B_t = [b_mkt,t | B^S_t | B^C_t] (n_t × k, k = 1 + k_s + k_c ≤ 25), computed only from
information ≤ t. Nothing downstream reads latent coordinates; L2 exposes Σ_t and Σ_sys,t only. The shape/scale
separation is kept: every learned column is multiplied by the measured scale s_i,t (trailing 63-day volatility,
ratio winsorized per INV-008).

**Why three blocks [own analysis].** The doctrine asks for two kinds of co-movement that a per-stock map of scalar
characteristics cannot represent:
1. *Structural* factors (industry, supply chain): membership is not a function of beta, volatility, momentum or
   liquidity ranks, and a per-stock path encoder cannot form low-rank industry loadings either, because the factor
   paths of a window change at every date and the encoder never sees the other stocks. A cross-sectional operation
   (PCA of the trailing correlation, or attention over stocks) is required. This is the block rolling PCA provides
   and IPCA on scalar instruments does not.
2. *Regime-dependent* exposures: per-date ranks remove market-wide levels, so a regime can only enter through an
   unranked state vector.
The conditional block B^C is where the DL claim is tested first; the structural block is fixed in E2 (identical in
both arms) and learned in E3.

**Inputs (all point in time).**
- c_i,t (13): market beta, downside beta, volatility 21/63/252 d, momentum 12-1, 1-month reversal, dollar-volume
  rank, Amihud illiquidity, max daily return (21 d), skewness (252 d), distance to 252-day high, correlation to the
  equal-weight index; rank-normalized to [−0.5, 0.5] per date.
- P_i,t: last L = 252 days (variant L = 126) of five channels: r_i,τ / s_i,t; |r_i,τ| / s_i,t; log dollar volume
  minus its 252-day mean; equal-weight index return / its trailing volatility; its absolute value. Scaling by the
  trailing volatility of date t (not per-window z-scoring) keeps the level. Missing days are masked. Windows are
  built per date on the fly (≈ 2.5 MB per date at n = 500).
- m_t (8, from L0): log realized volatility of the equal-weight index at 5/21/63 d; cross-sectional return
  dispersion (21 d); top-eigenvalue share of the 63-day correlation matrix; index drawdown from its 252-day high;
  log(vol21/vol252); the L0 stress indicator. Standardized with expanding-window moments, **not ranked**.

**Blocks (deep arm).**
| Block | Definition | Size |
|---|---|---|
| Market column b_mkt | Trailing 252-day beta to the equal-weight index × s_i,t (fixed, all arms) | 0 |
| Structural block B^S (E2: fixed) | Loadings on the top k_s principal components of the trailing 252-day correlation matrix of market residuals, × s_i,t; k_s ∈ {0, 5, 10, 20} chosen in the inner loop | 0 trainable |
| Path encoder E_ψ | 4 residual causal 1D-conv blocks, kernel 3, dilations 1/2/4/8, width 32, GELU + LayerNorm, attention pooling over time → h_i,t ∈ R³² | ≈ 15k |
| State encoder | MLP 8 → 32 → 16, GELU → e_t | ≈ 0.8k |
| Loading head g_θ | u = GELU(W₁[c_i,t, h_i,t] + b₁) (45 → 64); FiLM u ← γ(e_t) ⊙ u + β(e_t) (γ, β: linear 16 → 64, γ initialized at 1); dropout 0.1-0.2; GELU(64 → 64); linear 64 → k_c; B^C_i,t = s_i,t · output; k_c ∈ {2, 4, 8} | ≈ 11k |
| Total trainable | | ≈ 25-30k (≈ 27k without FiLM) |

**Σ_f,t and D_t inside training.** Factor returns are the GLS cross-sectional estimates
f̂_s = (B_sᵀD_s⁻¹B_s)⁻¹B_sᵀD_s⁻¹r_{s+1}; Σ_f,t is their EWMA covariance (half-life chosen in E1, linear shrinkage
toward its diagonal); D_t is the EWMA of squared residuals, floored at 5% of each stock's total variance. Both are
computed on a rolling buffer of past dates with gradients truncated beyond the current chunk of 64 consecutive
training dates (engineering choice, to be documented in `implementation_decisions.md`). With GLS factor returns,
B_tΣ_f,tB_tᵀ is invariant to B → BA, so no identification constraint is needed for the score.

**Loss.** Per-date, per-stock Gaussian NLL of the next-day cross-section,
ℓ_t(θ) = (1/2n_t)[log det Σ_t + r_{t+1}ᵀΣ_t⁻¹r_{t+1}], Σ_t = B_tΣ_f,tB_tᵀ + D_t, computed by Woodbury and the
matrix-determinant lemma in O(n_tk²); averaged over training dates (one date = one batch). Regularization:
AdamW (lr 3·10⁻⁴, weight decay 10⁻⁴), VIF penalty 0.01·tr((B^CᵀB^C/n)⁻¹) (Deep Risk Model default), early stopping
on the inner validation NLL (last two years before the 21-day embargo). Inner-loop variant: add NLL terms on 5- and
21-day cumulative returns (weight 0.5 each), the multi-horizon stability device of the Deep Risk Model. Five seeds
per configuration; the model is the average of the five Σ_t (PSD by convexity), never the best seed.

**Matched linear twin (mandatory, same everything else).** B^C_i,t = s_i,t · (z_i,tΓ), z_i,t = [c_i,t,
p_i,t, c_i,t ⊗ m_t], where p_i,t are 12 fixed linear path summaries (EWMA means of the five channels at
half-lives 21/63, and trailing betas to the index at 21/63/252 d); Γ trained under the same NLL, the same
Σ_f/D_t construction, the same structural block, the same inner loop and budget. Without the state arm the
c ⊗ m_t columns are dropped. This is IPCA with state-interacted instruments estimated by likelihood instead of
least squares (Kelly, Pruitt & Su 2019 is the parent). Attribution diagnostic (not a gate): IPCA on the deep arm's
frozen h_i,t, which separates "deep features" from "deep map".

**Formulation ladder (pre-registered, each rung against its own twin).**
- **F1a (E2):** deep B^C without state input vs twin without state.
- **F1b (E2):** deep B^C with FiLM state conditioning vs twin with c ⊗ m_t.
- **F2 (E3):** learned structural block replaces the fixed PCA block. Two attention layers over stocks on date t:
  h_i ← h_i + Σ_{j∈N_i} α_ij W_v h_j, α_ij = softmax_j(q_iᵀk_j/√32 + φ(ρ²⁵²_ij, ρ⁶³_ij)), N_i = 32 stocks with the
  largest trailing |ρ|, φ an MLP 2 → 8 → 1; head 32 → k_s, × s_i,t. ≈ 20k extra parameters; permutation-equivariant
  and rotation-free. Twin = the F1 winner with the fixed PCA block. One of the inner configurations initializes E_ψ
  by contrastive co-movement pretraining (InfoNCE, positives = stock pairs in the top decile of sub-window
  correlation, stress sub-windows oversampled ×3; Dolphin, Smyth & Dong 2024).
- **Exhausted:** the deep L1 claim is rejected for this information set (owner decision OD1 on what L1 then is).

**Evidence and priors.** Gu, Kelly & Xiu (2021): nonlinear loadings beat IPCA on predictive R² but not on total R²
(12.6% vs 13.3%, three factors), so the prior for F1a on scalar characteristics alone is low. Pelger & Xiong (2022):
state-varying loadings explain more variation than constant ones, which supports F1b. NeuralFactors (Gopal 2024): a
neural conditional factor model beats PPCA and a deep baseline on joint NLL, but returns-only inputs lose most of the
gain (validation NLL 0.466 vs 0.324) in an ablation that also removes industry. The Deep Risk Model's gains sit on top
of 29 industry factors. Expect small effects; size networks at 25-60k parameters; report the capacity curve.
```

### 3.2 New §5 L2 (replaces R:384-394)

```markdown
### L2 — Conditional covariance (DL layer 2)

**Contract.** Σ_t = B_tΣ_f,tB_tᵀ + D_t (+ an optional residual correlation term, F2), PSD by construction. API:
log-score of a return vector, variance of a portfolio, Σ_sys,t, stress scenarios (F3 only). Never exposes latent
coordinates to L3.

**Linear bars (scored in E1, not E4).** Shrunk EWMA Σ_f (as in L1); the declared calm/stressed blend (D5); factor
DCC on the L1 factor returns; asset-level DCC-NL (Engle, Ledoit & Wolf 2019); AFM-DCC (De Nard, Ledoit & Wolf 2021,
tested up to 1,000 stocks); QIS (Ledoit & Wolf 2022) on the same lookback.

**DL formulations.**
- **F1 — state-modulated factor dynamics (E4).** Σ_f,t = V_t^{1/2} R_t V_t^{1/2}. Factor variances
  log V_k,t = log EWMA_k,t + a_k(e_t), a = MLP 16 → 16 → k (≈ 0.5k parameters). Correlations follow a DCC recursion
  whose target is R̄_t = (1 − π_t)R̄_calm + π_t R̄_stress, π_t = sigmoid(MLP 16 → 8 → 1 (e_t)); R̄_calm and R̄_stress are
  sample correlations of the L1 factor returns on calm and stress dates (L0 indicator), shrunk. B_t frozen from L1;
  optional joint fine-tune. Twin: a_k ≡ 0 and π_t = sigmoid(wᵀm_t) (logistic, linear in the state). A free Cholesky
  head is not used: it puts k(k+1)/2 free outputs on a handful of crisis episodes.
- **F1-vol — deep idiosyncratic volatility head (E4).** log σ²_i,t+1 = MLP([log RV_i at 1/5/22/63 d, |r_i,t|, log
  dollar volume, m_t]) (≈ 3k parameters, shared), trained on the diagonal QLIKE; twin: HAR-log regression on the
  same inputs; control: EWMA, GARCH(1,1). Nonlinear volatility forecasts from price data have the most consistent
  published gains (Christensen, Siggaard & Veliyev 2023).
- **F2 — neural spectral cleaner (E4).** The rotation-invariant network of Bongiorno, Manolakis & Mantegna
  (per-lag soft clipping, BiLSTM over sorted eigenvalues, volatility MLP; ≈ 40k parameters), trained on realized
  5-day GMV variance on random sub-universes of 50-350 stocks, applied (a) as a standalone Σ_t arm, the strongest
  published deep competitor on price-only US equities, and (b) as a hybrid that cleans the residual correlation of
  the L1 model: Σ_t = BΣ_fBᵀ + D^{1/2}C̃_res D^{1/2}. Twins: QIS on the same lookback; the same network with the
  identity eigenvalue map.
- **F3 — regime-mixture latent factor model, the VAE rung (E7).** Latents per date: s_t ∈ {calm, stress} with an
  HMM prior (learned 2 × 2 transition) and f_{t+1} | s ~ Student-t_ν(0, Σ_f^{(s)}); residuals Student-t. The stress
  regime adds 2-4 low-rank loading columns through regime-specific FiLM parameters, regularized toward the calm
  parameters. Inference: q(s_t | m_≤t) by a GRU(16) filter over m_t (point in time), q(f_{t+1} | r_{t+1}, B, D)
  Gaussian moment-matched, IWAE with 10-20 samples (NeuralFactors). Output: π_t, Σ^{(s)}, the mixture Σ_t, and a
  scenario sampler. ≈ 60-100k parameters. Twins: the Gaussian F1 model without regime, and the linear two-regime
  blend at the same π_t rule. Label switching resolved by ordering on factor volatility. Runs only if H1c or H2c
  shows state value.

**Note on the VAE.** With Gaussian factors and residuals the marginal likelihood of the L1/L2 model is exact
(r ~ N(0, BΣ_fBᵀ + D)), so the posterior over f_t is exact and no ELBO is needed: the Gaussian conditional factor
model is the exact-inference case of a VAE whose latent is f_t. The variational machinery becomes necessary, and
useful, only with Student-t components or a discrete regime (F3).
```

### 3.3 Additions to D1 and L3 (append to R:237 and R:402)

```markdown
4. **Amplification is not cured by the measure.** Equal contributions over any principal basis, including the
   asset-space projectors, imply |v_kᵀw| ∝ λ_k^{-1/2}; GMV loads even harder on small eigen-directions. Protection
   against noise-driven small eigenvalues must come from L2 (shrinkage, cleaning) and is measured by realized
   variance, not by the ENB.
5. **Feasibility.** For long-only portfolios the market block dominates Σ_sys. The ENB floor is set as a fraction
   e_rel ∈ {0.5, 0.7, 0.9} of the per-date maximum feasible ENB under the constraint set C; the problem is
   nonconvex, so multi-start dispersion is reported.
```

L3 policies, add after P5: `P6 GMV on Σ_policy = (1 − π)Σ_t + πΣ_stress, π fixed ex ante (policy-level memory,
Chow et al. 1999); a named arm, owner decision OD4.`

### 3.4 Additions to L4 (append to R:421)

```markdown
- **Decomposition.** Every log-score difference is split into a volatility part (univariate log-scores of the
  diagonal) and a correlation part (log-score of returns standardized by forecast volatilities).
- **Adoption needs two results.** (i) the deep arm beats its matched twin (one-sided DM-HAC, SPA over the arm's
  variants); (ii) the deep arm belongs to the 90% Model Confidence Set formed with every E1 arm. (i) alone shows
  that the nonlinearity has value; only (i) + (ii) makes the arm a candidate for the system.
- **Seeds.** Five seeds per configuration, averaged in Σ space; seed spread reported.
- **Crisis tests.** First-occurrence test: walk-forward forecasts of each crisis from models trained before it;
  plus leave-one-crisis-out for any state head (F1b, L2 F1, F3).
- **Sharpe inference.** Ledoit-Wolf (2008) studentized block bootstrap, deflated Sharpe ratio (Bailey & López de
  Prado 2014) for anything selected.
```

### 3.5 Revised §6.1 rows and §6.2 table

§6.1, replace the two Sharpe rows (R:482-483):

```markdown
| Sharpe difference, one 6.25-year block | — | — | SE ≈ √(2(1−ρ)/years) ≈ 0.18 for ρ = 0.9 (Jobson-Korkie/Memmel with the SR² term in per-period units, negligible; Monte Carlo 0.177) | The current 0.11 gap ≈ 0.6 SE: not evidence either way |
| Sharpe difference, 18 years | — | — | SE ≈ 0.105 for ρ = 0.9 ⇒ MDE ≈ 0.29 | Secondary only |
```

§6.2, replacement table:

```markdown
| Layer | Hypothesis | Refuted if | If refuted: stopping rule | Next DL formulation |
|---|---|---|---|---|
| L0 | **H0** Rankings are robust to data limits | L2 rankings (log-score) flip between liquidity universes of N = 300 and 500 or between the two halves of the OOS period; L3/L4 portfolio rankings flip across the three delisting bounds | Claims that flip are reported as data-undetermined | — (measurement) |
| L1 | **H1a** A conditional linear object beats the static object (E1) | DM p ≥ 0.05 for every conditional linear arm (rolling PCA, IPCA, IPCA + residual PCA) vs static PCA and static mean-B, *and* TOST equivalence within δ = 0.002 nats per stock-day (margin fixed before E1) | The episodic premise has no linear OOS content; E2 still runs with a lower prior | F1a/F1b (E2) |
| L1 | **H1ℓ** The NLL objective beats least squares for the same linear map (E1) | DM p ≥ 0.05, IPCA-NLL vs IPCA-LS | Keep NLL anyway (it is the scoring rule); report | — |
| L1 | **H1b** Deep conditional loadings beat the matched twin, without state (E2, F1a) | DM p ≥ 0.05 after SPA over F1a variants, or calibration outside [0.8, 1.2], or GMV realized variance significantly worse (one-sided 5%) | F1a rejected | F1b |
| L1 | **H1c** State conditioning adds value (E2, F1b) | Deep-with-state does not beat deep-without-state *and* twin-with-state does not beat twin-without-state (DM, SPA), on all days and on stress days | Drop state conditioning from L1; regime claims move to L2 (H2c) | F2 (learned structural block) |
| L1 | **H1d** A learned cross-sectional structural block beats the fixed PCA block (E3, F2) | DM p ≥ 0.05 vs the F1 winner with the fixed block | Keep the fixed block (owner decision OD2) | Ladder exhausted: L1 deep claim rejected for this information set (OD1) |
| L2 | **H2** State-modulated factor dynamics beat the linear bars (E4, F1) | 90% MCS contains a linear arm and no deep arm has a better mean, or DM p ≥ 0.05 vs the best of DCC-NL, AFM-DCC, QIS, declared blend | F1 rejected | F2 hybrid |
| L2 | **H2v** The deep volatility head beats HAR/GARCH on the diagonal (E4) | Diagonal QLIKE DM p ≥ 0.05 vs the HAR twin | Keep EWMA/GARCH D_t | — |
| L2 | **H2n** The neural spectral cleaner, alone or as a residual hybrid, beats QIS and the CPDFM (E4, F2) | DM p ≥ 0.05 on log-score *and* realized GMV variance not lower | Cleaner rejected | F3 (E7) |
| L2 | **H2c** State conditioning improves stress-day forecasts | Stress-day DM p ≥ 0.05, the first-occurrence crisis test fails, and the episode sign test fails (< 7/7 at n = 7, < 6/6 at n = 6) | Drop the "protected before reactivation" claim (DVT L78) | F3 only if H1c or H2c passed on at least one criterion |
| L2 | **H2r** The regime-mixture Student-t VAE beats the Gaussian CPDFM (E7, F3) | Joint log-score DM p ≥ 0.05 overall *and* on stress days, *and* lower-tail exceedance calibration not better | Keep the Gaussian model; delete tail-dependence claims (DVT L11, L511) | — |
| L3 | **H3a** The diversification constraint pays through model error | P1 vs P0 on the same Σ_t: realized-variance non-inferiority fails at a margin fixed after E1 measures its SE, or the stress-loss sign test fails | Drop the constraint; P0 | E6 decision-focused training |
| L3 | **H3b** The projector measure is more stable than the latent-eigenbasis entropy | Turnover at equal realized variance not lower (paired, HAC) | Keep the measure only if H3a holds | — |
| L3 | **H3m** Policy-level memory (P6) reduces stress losses at bounded calm-period cost | Stress-loss sign test fails or calm-period variance cost above the pre-set budget | Drop P6 | — |
| L4 | **H4** The full system beats every benchmark | SPA p ≥ 0.05 on outer folds, or the holdout sign is opposite | Report the MCS; no deployment claim | Highest-ranked failing layer's next formulation, same protocol and registry |
```

### 3.6 Revised §7 rows and executive-summary paragraph

```markdown
| **E0 — Foundations** | L0: PIT liquidity universe, raw-close level filters, delisting bounds, stress indicator, market-state vector m_t. L4: one harness, decision states, DM/SPA/MCS, failure recording, log-score decomposition. L5: single config, manifest, variant registry, loud failures. Fix ERC fallback, fast-solver turnover, cardinality at every rebalance. No model change. | All benchmarks + incumbent run end-to-end with manifests (50 stocks/8 years and 300 stocks); ERC solves on every fold; two runs identical; a synthetic panel with a planted *regime-dependent* factor structure ranks the true model first. | Local | Engineering |
| **E1 — Linear bars, H1a, H1ℓ** | 500 stocks, 1995-2023, nested walk-forward: sample + LW; static PCA (DGJ); rolling PCA; IPCA-LS and IPCA-NLL (13 characteristics, + state interactions); IPCA + residual PCA; price-style model; static mean-B; QIS; DCC-NL; AFM-DCC; declared memory blend. Measure loss-differential autocorrelation, N_eff, MDE, and the SE of realized-variance ratios. | Bias ratio of the best arm in [0.8, 1.2]; H1a, H1ℓ decided; output: the bar (MCS of linear arms), measured power, margins for H3a. | Local (DCC-NL/AFM-DCC on Colab if slow) | Low |
| **E2 — Decisive DL test for L1 (H1b, H1c)** | 2 × 2 {linear twin, deep} × {no state, FiLM state}, all with the fixed market + residual-PCA structural block; same Σ_f/D_t construction and NLL; k_c ∈ {2, 4, 8}, k_s ∈ {0, 5, 10, 20} in the inner loop; 5 seeds averaged; annual refits, 21-day embargo. | Pass = deep beats twin within a state arm (DM-HAC one-sided, SPA over that cell's variants), calibration in [0.8, 1.2], GMV variance not significantly worse; *adoption* also requires MCS membership vs the E1 bar. Pass ⇒ E3 on the winner and E4. Fail on both cells ⇒ E3 still runs (structural block), L1 prior lowered. Inconclusive ⇒ report; no re-runs on the same folds. | Local for one seed (≈ 27k parameters, one date per batch); Colab for 5 seeds × 18 refits | Medium |
| **E3 — Learned structural block (H1d)** | Correlation-graph attention block replaces the fixed PCA block on the E2 winner (or on the deep no-state arm if E2 failed); one inner configuration with contrastive pretraining of the path encoder. Twin = same model with the fixed block. | Same gate as E2. Fail ⇒ fixed block kept (OD2); if E2 also failed, the L1 deep claim is rejected for this information set (OD1). | Local smoke on 50 stocks; Colab for sweeps (windows built on the fly, no 12 GB materialization) | Medium |
| **E4 — L2 (H2, H2v, H2n, H2c)** | On the best L1 map: state-modulated factor DCC vs its twin; deep volatility head vs HAR; neural spectral cleaner standalone and as residual hybrid vs QIS; all against the E1 bars. First-occurrence and leave-one-crisis-out tests. | §6.2 rows. | Colab for the cleaner (≈ hours per yearly model on CPU, 10 runs in the paper); local for the rest | Medium-high |
| **E5 — Allocation (H3a, H3b, H3m)** | P0-P6 on the adopted Σ_t, one harness, costs, three delisting bounds. | §6.2 rows; H0 for portfolio rankings. | Local | Low |
| **E6 — Decision-focused (L3 F2)** | End-to-end fine-tuning of L1/L2 through a differentiable GMV or constrained layer on realized variance. | Beats the two-stage system on realized GMV variance (DM) without calibration loss. | Colab | High |
| **E7 — Regime-mixture VAE (H2r)** | Only if H1c or H2c passed on at least one criterion. | §6.2 H2r. | Colab | High |
| **E8 — End-to-end and holdout (H4)** | Frozen pipeline on outer folds; SPA vs every benchmark; holdout once. | SPA p < 0.05 and holdout sign consistent; otherwise report the MCS. | Colab/local | Medium |
```

Executive summary, replace R:43-50:

```markdown
**The decisive next experiment (E2, §7).** After the foundations (E0) and the full linear bar (E1, including
DCC-NL, QIS and AFM-DCC), train a small conditional probabilistic factor model (path encoder + characteristics,
with and without FiLM state conditioning, ≈ 27k parameters) and its matched linear twin (IPCA with state-interacted
instruments, same NLL) on the same inputs, structural block, Σ_f and D_t construction and nested protocol, and
compare daily OOS Gaussian log-scores. **Gate:** deep beats its twin (one-sided DM-HAC, SPA over the cell's variants)
with calibration in [0.8, 1.2] and no significant degradation of realized GMV variance; adoption also requires
membership of the Model Confidence Set against the E1 bar. **If it fails:** the learned structural block (E3) and the
L2 ladder (E4, including the neural spectral cleaner) are tested under the same rule; the plan never adopts a linear
model as the strategy.
```

### 3.7 Small corrections

- R:56-59 and R:431: replace the deletion warning by "`results/` (including the only real-run artifacts and both
  checkpoints) is tracked in git and present in the working tree as of 2026-11-08; L5 archives it with a manifest."
- R:119-121 (R1, "What DL can add"): append "Gu, Kelly & Xiu's own results show nonlinear loadings on
  characteristics improve predictive R² but not total R² over IPCA (12.6% vs 13.3%, three factors), so the
  nonlinearity must earn its place on covariance scores; and on price-only data a cross-sectional structural block is
  needed (L1)."
- R:150-162 (R4): append the amplification note of section 3.3, item 4.
- R:252-257 (D3): append "The VAE's place is the date-level latent (f_t, s_t): exact Gaussian case in E2, variational
  regime/Student-t case in E7."
- R:663-674 (Appendix B): move DGJ 2018 and LW 2020 to verified; add section 5 of this review.

---

## 4. Decisions that belong to the owner

**OD1 — What L1 is if every deep L1 rung fails.** The redesign makes IPCA loadings the L1 component (R:499, R:523).
- (a) Accept a linear L1 component while DL lives in L2 (state dynamics, volatility head, cleaner) or L3.
- (b) Declare the deep L1 claim rejected and change the information set (industry codes, intraday) before
  any further L1 work.
- (c) Keep a deep L1 anyway on non-inferiority grounds.
- **Recommendation: (a) for operation, (b) for research.** (c) contradicts "evidence before narrative".

**OD2 — A fixed statistical block inside the deep model.** E2 includes a rolling residual-PCA block in both arms.
- (a) Accept it as a component. It handles structural co-movement, and the DL claim is tested on the conditional
  part.
- (b) Require a learned block from the start (E3 first).
- **Recommendation: (a).** Without the block, both arms of E2 lack the dominant non-market structure, and the
  comparison measures little. E3 then tests whether DL can replace the block.

**OD3 — Scope of the VAE.**
- (a) The VAE is the probabilistic latent factor model. Its exact Gaussian case comes first (E2), and its
  variational regime and Student-t case comes at E7.
- (b) Insist on variational training from E2.
- **Recommendation: (a).** The ELBO adds nothing when the posterior is exact. Variational inference earns its
  cost only for tails and regimes, and only if H1c or H2c show state value.

**OD4 — Policy-level memory (anti-cyclical insurance).**
- (a) Add P6, a declared calm/stress blend with π fixed ex ante, as a named arm. π is a preference, not a forecast.
- (b) Leave memory to forecasting only.
- **Recommendation: (a).** It is the only form of the doctrine's anti-cyclical principle that does not depend on
  forecasting crises from about 7 episodes. Fix π before E5.

**OD5 — Industry information.** NeuralFactors and the Deep Risk Model draw much of their gain from industry or
fundamental inputs.
- (a) Strictly price-only, as now.
- (b) PIT SEC SIC codes as a baseline and diagnostic only (redesign D10).
- (c) Industry codes as an input in one named arm after E3.
- **Recommendation: (b) now, (c) only if E2 and E3 fail.** This keeps the discovery principle testable: does the
  learned structural block recover what SIC codes give?

**OD6 — Compute budget.** Five-seed ensembles × 18 annual refits × (2 × 2 + E3) cells require Colab.
- **Recommendation:** accept, and cap each formulation at 8 inner configurations (as in the redesign).

**OD7 — Pre-registered margins.**
- Fix δ = 0.002 nats per stock-day (H1a), the H3a variance margin (set after E1 measures its SE), the P6 π, and
  the ENB fractions e_rel **before E1 runs**.
- **Recommendation:** sign them in `implementation_decisions.md`.

---

## 5. References

| # | Title | Authors | Year | Venue | Link | Status |
|---|---|---|---|---|---|---|
| 1 | NeuralFactors: A Novel Factor Learning Approach to Generative Modeling of Equities | A. Gopal | 2024 | arXiv 2408.01499 | https://arxiv.org/abs/2408.01499 | V-full (features, Table 3 ablation, sizes, test results) |
| 2 | End-to-End Large Portfolio Optimization for Variance Minimization with Neural Networks through Covariance Cleaning | C. Bongiorno, E. Manolakis, R. N. Mantegna | 2025 (v3 2026) | arXiv 2507.01918 | https://arxiv.org/abs/2507.01918 | V-full (modules, sizes, Table 2); journal venue not confirmed |
| 3 | Deep Risk Model: A Deep Learning Solution for Mining Latent Risk Factors to Improve Covariance Matrix Estimation | H. Lin, D. Zhou, W. Liu, J. Bian | 2021 | ICAIF '21; arXiv 2107.05201 | https://arxiv.org/abs/2107.05201 | V-full (inputs, loss, Table 1, GAT ablation) |
| 4 | Autoencoder Asset Pricing Models | S. Gu, B. Kelly, D. Xiu | 2021 | J. Econometrics 222(1):429-450 | https://econpapers.repec.org/RePEc:eee:econom:v:222:y:2021:i:1:p:429-450 | V-abs; total-R² figures (12.6/13.3/3.4%) V-sec (paper text quoted in a search summary, partly confirmed on a Scribd copy); CA0-CA3 sizes V-sec (https://heewonh.github.io/autoencoder) |
| 5 | Characteristics Are Covariances: A Unified Model of Risk and Return | B. Kelly, S. Pruitt, Y. Su | 2019 | JFE 134(3) | https://www.nber.org/papers/w24540 | V-abs |
| 6 | FactorVAE: A Probabilistic Dynamic Factor Model Based on Variational Autoencoder for Predicting Cross-Sectional Stock Returns | Y. Duan, L. Wang, Q. Zhang, J. Li | 2022 | AAAI 36(4) | https://ojs.aaai.org/index.php/AAAI/article/view/20369 | V-sec (abstract via search; page blocked) |
| 7 | State-Varying Factor Models of Large Dimensions | M. Pelger, R. Xiong | 2022 | JBES 40(3) | https://arxiv.org/abs/1807.02248 | V-abs |
| 8 | Don't Blame the ELBO! A Linear VAE Perspective on Posterior Collapse | J. Lucas, G. Tucker, R. Grosse, M. Norouzi | 2019 | NeurIPS | https://arxiv.org/abs/1911.02469 | V-abs |
| 9 | Projected Principal Component Analysis in Factor Models | J. Fan, Y. Liao, W. Wang | 2016 | Annals of Statistics 44(1):219-254 | https://arxiv.org/abs/1406.3836 | V-abs (loadings = covariate part + orthogonal residual part; supports the structural + conditional split) |
| 10 | Factor Models for Portfolio Selection in Large Dimensions: The Good, the Better and the Ugly | G. De Nard, O. Ledoit, M. Wolf | 2021 | J. Fin. Econometrics 19(2):236-257 | https://econpapers.repec.org/RePEc:oup:jfinec:v:19:y:2021:i:2:p:236-257 | V-abs (the "AFM-DCC" label not confirmed in the abstract) |
| 11 | Large Dynamic Covariance Matrices | R. Engle, O. Ledoit, M. Wolf | 2019 | JBES 37(2):363-375 | https://econpapers.repec.org/RePEc:taf:jnlbes:v:37:y:2019:i:2:p:363-375 | verified by the redesign; not re-opened |
| 12 | Analytical Nonlinear Shrinkage of Large-Dimensional Covariance Matrices | O. Ledoit, M. Wolf | 2020 | Annals of Statistics 48(5):3043-3065 | https://projecteuclid.org/journals/annals-of-statistics/volume-48/issue-5/Analytical-nonlinear-shrinkage-of-large-dimensional-covariance-matrices/10.1214/19-AOS1921.full | V-abs |
| 13 | Optimal Shrinkage of Eigenvalues in the Spiked Covariance Model | D. Donoho, M. Gavish, I. Johnstone | 2018 | Annals of Statistics 46(4):1742-1778 | https://arxiv.org/abs/1311.0851 | V-abs |
| 14 | Quadratic Shrinkage for Large Covariance Matrices | O. Ledoit, M. Wolf | 2022 | Bernoulli 28(3) | https://projecteuclid.org/journals/bernoulli/volume-28/issue-3/Quadratic-shrinkage-for-large-covariance-matrices/10.3150/20-BEJ1315.full | V-abs by the survey; not re-opened |
| 15 | Risk Budgeting and Diversification Based on Optimised Uncorrelated Factors | A. Meucci, A. Santangelo, R. Deguest | 2015 | Risk 11:70-75; SSRN 2276632 | https://www.risk.net/risk-magazine/technical-paper/2433224/risk-budgeting-and-diversification-based-on-optimised-uncorrelated-factors | V-abs |
| 16 | Diversifying Risk Parity | H. Lohre, H. Opfer, G. Ország | 2014 | J. Risk 16(5):53-79 | https://ideas.repec.org/a/rsk/journ4/2347710.html | V-abs |
| 17 | Optimal Portfolios in Good Times and Bad | G. Chow, E. Jacquier, M. Kritzman, K. Lowry | 1999 | FAJ 55(3) | https://econpapers.repec.org/RePEc:taf:ufajxx:v:55:y:1999:i:3:p:65-73 | V-abs by the survey; not re-opened |
| 18 | A Test for Superior Predictive Ability | P. R. Hansen | 2005 | JBES 23(4):365-380 | https://doi.org/10.1198/073500105000000063 | V-abs |
| 19 | The Model Confidence Set | P. Hansen, A. Lunde, J. Nason | 2011 | Econometrica 79(2) | https://www.econometricsociety.org/publications/econometrica/2011/03/01/model-confidence-set | V-abs by the survey; not re-opened |
| 20 | Testing and Valuing Dynamic Correlations for Asset Allocation | R. Engle, R. Colacito | 2006 | JBES 24(2):238-253 | https://doi.org/10.1198/073500106000000017 | V-abs (partial abstract) |
| 21 | Robust Performance Hypothesis Testing with the Sharpe Ratio | O. Ledoit, M. Wolf | 2008 | J. Empirical Finance 15(5) | https://econpapers.repec.org/RePEc:eee:empfin:v:15:y:2008:i:5:p:850-859 | V-abs (via record); cites the Memmel (2003) correction |
| 22 | Performance Hypothesis Testing with the Sharpe Ratio | C. Memmel | 2003 | Finance Letters 1:21-23 | — | V-sec (formula checked by Monte Carlo here, not on the text) |
| 23 | The Deflated Sharpe Ratio | D. Bailey, M. López de Prado | 2014 | J. Portfolio Management 40(5):94-107 | https://papers.ssrn.com/abstract=2460551 | V-abs |
| 24 | Estimating Covariance for Global Minimum Variance Portfolio: A Decision-Focused Learning Approach | J. Kim, I. Tae, Y. Lee | 2025 | arXiv 2508.10776 | https://arxiv.org/abs/2508.10776 | V-abs (no figures in abstract) |
| 25 | Neural Nonlinear Shrinkage of Covariance Matrices for Minimum Variance Portfolio Optimization | L. Yang, S. Zhao, S. Chai | 2026 | arXiv 2601.15597 | https://arxiv.org/abs/2601.15597 | V-abs (S&P 500 constituents; no figures in abstract) |
| 26 | End-to-End Risk Budgeting Portfolio Optimization with Neural Networks | A. S. Uysal, X. Li, J. M. Mulvey | 2024 | Annals of OR 339 | https://arxiv.org/abs/2107.04636 | V-abs |
| 27 | Contrastive Learning of Asset Embeddings from Financial Time Series | R. Dolphin, B. Smyth, R. Dong | 2024 | arXiv 2407.18645 | https://arxiv.org/abs/2407.18645 | V-abs |
| 28 | HireVAE: An Online and Adaptive Factor Model Based on Hierarchical and Regime-Switch VAE | Z. Wei, A. Rao, B. Dai, D. Lin | 2023 | IJCAI | https://arxiv.org/abs/2306.02848 | V-abs (return prediction only) |
| 29 | A Machine Learning Approach to Volatility Forecasting | K. Christensen, M. Siggaard, B. Veliyev | 2023 | J. Fin. Econometrics 21(5) | https://econpapers.repec.org/RePEc:aah:create:2021-03 | V-abs by the survey; not re-opened |
| 30 | Hybrid GARCH-LSTM for large portfolios | Boulet | 2021 | arXiv 2109.01044 | https://arxiv.org/abs/2109.01044 | V-sec (survey); not re-opened |
| 31 | Graph attention networks for portfolio weights (distance-correlation network) | Korangi, Mues, Bravo | 2024 | arXiv 2407.15532 | https://arxiv.org/abs/2407.15532 | V-sec (search snippet); weights, not covariance: thin evidence for the F2 graph block |
| 32 | OptNet: Differentiable Optimization as a Layer in Neural Networks | B. Amos, J. Z. Kolter | 2017 | ICML | https://arxiv.org/abs/1703.00443 | verified by the theory audit; not re-opened |
| 33 | FiLM: Visual Reasoning with a General Conditioning Layer | E. Perez et al. | 2018 | AAAI | https://arxiv.org/abs/1709.07871 | [unverified] (cited for the conditioning block only) |
| 34 | Importance Weighted Autoencoders | Y. Burda, R. Grosse, R. Salakhutdinov | 2016 | ICLR | https://arxiv.org/abs/1509.00519 | V-abs by the survey; not re-opened |

**Not verified in this review.**
- The full Gu-Kelly-Xiu tables.
- The FactorVAE body.
- The DCC-NL and AFM-DCC computational cost at 500 stocks over 18 years on a laptop.
- The exact Memmel formula text.
- The journal venue of Bongiorno et al.
- Any published out-of-sample covariance test of a regime-mixture VAE on equities. None was found, as the survey
  also reports.
- Any covariance-loss evaluation of a correlation-graph attention block on price-only data. The evidence for
  F2/E3 is design reasoning, not results.
