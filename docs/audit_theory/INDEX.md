# Theory audit 2026-11-07 — Index

Blind paired audit (advocate / devil's advocate) of the Latent Risk Factor Discovery **theory**. The audit read a
closed corpus of 9 documents only: no code, no results, no earlier audit.

**Deliverables**
- [REDESIGN.md](REDESIGN.md): systemic redesign, including §10, the comparison with `docs/audit_v5/REDESIGN.md`.
- [findings.jsonl](findings.jsonl): 429 findings with the reduced schema, plus `unit`, `report`, `prompt_version`,
  `prompt_sha`, `verification`, `final_severity` and `verification_reason`.
- [reports/](reports/): the 38 raw reports.

**Agents.**
- DeepSeek v4.1 Flash at effort `max`, run with CommandCode 1.88.5 in headless mode, permission mode `dont-ask`
  (read, grep and glob only), working directory = corpus, 4 agents in parallel.
- The kit launcher `run_agent.sh` was used unchanged: timeout per agent, then session resume when END-OF-REPORT is
  missing, then a fresh run.
- Prompts come from adapted copies of the kit template and generator, in `campaign/`.

## 1. Units and scores

Columns:
- **Scores** are each agent's own `Score: X/10`, read from its role: an opinion, not a measurement.
- **Findings** gives the number of findings per role (advocate / critic).
- **Critical → c / d / r** gives the critic's critical findings, then how verification classified them: confirmed
  (including narrowed) / downgraded / rejected.
- **Prompt version**:
  - `v1-uncapped` is the first template;
  - `v2-capped` adds the ~25 KB cap, at most 12 findings, full derivations only for critical findings, and the
    reduced JSON companion.
  - The step shown is the one that produced the report: `f1` first run; `.rN` session resume; `f2` fresh run.

| Unit | Topic | Adv score | Crit score | Findings adv / crit | Critical → c / d / r | Advocate prompt (step, size) | Critic prompt (step, size) | Reports |
|---|---|---|---|---|---|---|---|---|
| U01 | Vision and premises | 7 | 4 | 11 / 10 | 3 → 2 / 1 / 0 | v2-capped (f1, 23 KB) | v2-capped (f1, 25 KB) | [adv](reports/U01_advocate.md) · [crit](reports/U01_critic.md) |
| U02 | Nature of risk | 6 | 3 | 11 / 16 | 4 → 3 / 1 / 0 | v2-capped (f1, 25 KB) | v1-uncapped (f1, 58 KB) | [adv](reports/U02_advocate.md) · [crit](reports/U02_critic.md) |
| U03 | Data choice and data requirements | 7 | 3 | 10 / 10 | 3 → 2 / 1 / 0 | v2-capped (f1, 28 KB) | v2-capped (f1, 25 KB) | [adv](reports/U03_advocate.md) · [crit](reports/U03_critic.md) |
| U04 | Feature engineering and cleaning | 7 | 3 | 12 / 12 | 2 → 2 / 0 / 0 | v2-capped (f1, 24 KB) | v2-capped (f1, 31 KB) | [adv](reports/U04_advocate.md) · [crit](reports/U04_critic.md) |
| U05 | Model choice and architecture | 7 | 4 | 10 / 10 | 4 → 3 / 1 / 0 | v2-capped (f1, 26 KB) | v2-capped (f1.r1, 28 KB) | [adv](reports/U05_advocate.md) · [crit](reports/U05_critic.md) |
| U06 | Latent dimension and number of factors | 6 | 3 | 11 / 12 | 3 → 3 / 0 / 0 | v2-capped (f1, 28 KB) | v2-capped (f1, 32 KB) | [adv](reports/U06_advocate.md) · [crit](reports/U06_critic.md) |
| U07 | Core loss function | 7 | 3 | 12 / 11 | 2 → 1 / 1 / 0 | v2-capped (f1, 27 KB) | v2-capped (f1, 30 KB) | [adv](reports/U07_advocate.md) · [crit](reports/U07_critic.md) |
| U08 | Auxiliary loss terms | 6 | 3 | 12 / 11 | 3 → 1 / 2 / 0 | v2-capped (f1, 26 KB) | v2-capped (f2.r1, 27 KB) | [adv](reports/U08_advocate.md) · [crit](reports/U08_critic.md) |
| U09 | Training | 6.5 | 3 | 11 / 12 | 2 → 1 / 1 / 0 | v2-capped (f1, 32 KB) | v2-capped (f1, 32 KB) | [adv](reports/U09_advocate.md) · [crit](reports/U09_critic.md) |
| U10 | Inference and composite risk profiles | 6.5 | 3 | 12 / 10 | 3 → 2 / 1 / 0 | v2-capped (f1, 31 KB) | v2-capped (f1, 27 KB) | [adv](reports/U10_advocate.md) · [crit](reports/U10_critic.md) |
| U11 | Risk model | 7 | 3 | 10 / 11 | 5 → 2 / 3 / 0 | v2-capped (f1, 28 KB) | v2-capped (f1.r1, 36 KB) | [adv](reports/U11_advocate.md) · [crit](reports/U11_critic.md) |
| U12 | Diversification objective | 8 | 3 | 11 / 12 | 4 → 3 / 0 / 1 | v2-capped (f1, 26 KB) | v2-capped (f1, 32 KB) | [adv](reports/U12_advocate.md) · [crit](reports/U12_critic.md) |
| U13 | Optimization and constraints | 6.5 | 3 | 12 / 11 | 3 → 2 / 1 / 0 | v2-capped (f1, 32 KB) | v2-capped (f1, 29 KB) | [adv](reports/U13_advocate.md) · [crit](reports/U13_critic.md) |
| U14 | Validation protocol | 6 | 3 | 11 / 12 | 3 → 2 / 1 / 0 | v2-capped (f1, 31 KB) | v2-capped (f1, 30 KB) | [adv](reports/U14_advocate.md) · [crit](reports/U14_critic.md) |
| U15 | Benchmarks and fairness of comparison | 7.5 | 3 | 10 / 12 | 3 → 3 / 0 / 0 | v2-capped (f1, 27 KB) | v2-capped (f1, 29 KB) | [adv](reports/U15_advocate.md) · [crit](reports/U15_critic.md) |
| U16 | Metrics, statistical tests and decision criteria | 6 | 3 | 11 / 11 | 3 → 3 / 0 / 0 | v2-capped (f1, 26 KB) | v2-capped (f1, 32 KB) | [adv](reports/U16_advocate.md) · [crit](reports/U16_critic.md) |
| U17 | Conclusions, framing and contingency roadmap | 7 | 3 | 12 / 11 | 3 → 3 / 0 / 0 | v2-capped (f1, 32 KB) | v2-capped (f1, 30 KB) | [adv](reports/U17_advocate.md) · [crit](reports/U17_critic.md) |
| U18 | Parameters and their dependencies | 7 | 3 | 12 / 10 | 4 → 3 / 0 / 1 | v2-capped (f1, 34 KB) | v2-capped (f1, 30 KB) | [adv](reports/U18_advocate.md) · [crit](reports/U18_critic.md) |
| U19 | Cross-document consistency | 6 | 3 | 12 / 12 | 2 → 2 / 0 / 0 | v2-capped (f1, 25 KB) | v2-capped (f1.r1, 34 KB) | [adv](reports/U19_advocate.md) · [crit](reports/U19_critic.md) |

**Totals.**
- 429 findings: 213 advocate, 216 critic.
- 59 critic critical findings: 43 confirmed (34 as stated, 9 narrowed), 14 downgraded, 2 rejected.
- One advocate finding (U10-ADV-01) was labelled with the critic scale ("critical"). It is kept as reported and marked
  `not-applicable`.

## 2. Validation and accepted deviations

- **Validator.** `validate_report.py --repo corpus` checks sections, sentinel, score, JSON schema and citations. Every
  report resolves at least 80% of its `file:line` citations and has its score and sentinel. The only flags left are
  the accepted deviations below.
- **Reads outside the corpus.** None. Every `tool_queued` input of every transcript was scanned for paths outside the
  corpus.
- **Corpus.** Unchanged: the shasum fingerprint was checked before and after. The harness side folder
  `corpus/.commandcode/` was deleted at the end.
- **Accepted deviations** (user decisions):
  - `U02_critic` was produced by the **v1-uncapped** prompt (58 KB, 16 findings, old JSON with full text). Its JSON
    was normalised to the reduced keys in `findings.jsonl`.
  - `U04_critic` is 31 KB with the v2 prompt.
- **Size cap.** 16 further reports exceed the 30 KB tolerance of the 25 KB cap (30-36 KB). They were accepted on the
  same basis as U04_critic: the content is complete and the cap is advisory for the model.
- **`U10_critic`.** The `## Machine companion` heading is missing; the fenced JSON block, score and sentinel are
  present. Accepted.
- **Reruns.** The first `U03_critic` and `U05_critic` reports resolved only 22% and 39% of their citations, because
  they abbreviated file names (`DVT_...md`). Both were rerun as fresh runs (`runs_rerun/`); the reruns resolve
  ≥ 80%. The rejected originals are kept in `campaign/runs_failed/citations_lt80/`.

## 3. Verification of critical findings (step 3)


Checks for each finding: (1) the quoted claim is at the cited line (all 59: yes, see `claim_lines.txt`);
(2) the derivation or counterexample, recomputed by the orchestrator; (3) every literature reference checked on the web.

Verdicts: **confirmed** (kept critical), **confirmed-narrowed** (kept critical, part of the argument removed),
**downgraded** (valid residue kept at major or minor, with the reason), **rejected** (the argument does not hold).

### Literature checks (web)

| Reference (as cited by critics) | Exists? | Says what is claimed? |
|---|---|---|
| Michaud 1989, FAJ 45(1):31-42, "The Markowitz Optimization Enigma" | yes | yes: optimizers as estimation-error maximizers |
| Embrechts, McNeil & Straumann 2002, "Correlation and dependence in risk management: properties and pitfalls" | yes | yes: correlation is adequate only for elliptical laws; the Gaussian copula has zero tail dependence for ρ<1 |
| Shumway 1997, JF 52(1):327-340 | yes | the −30% figure is the standard correction for missing *performance-related* delisting returns (confirmed indirectly; full text not readable) |
| Shumway & Warther 1999, JF 54(6):2361-2379 | yes | yes: −55% for missing *performance-related* Nasdaq delisting returns |
| Ledoit & Wolf (identity target), JMVA 88 (2004) 365-411 | yes | identity-target estimator; the (π−ρ)/γ intensity quoted by U11-CRI-03 belongs to the 2003/2004 single-index and constant-correlation papers, not to this target → **misattributed detail** |
| Ledoit & Wolf 2003, JEF: i.i.d. returns assumption | yes | yes: Assumption 1, returns i.i.d. through time |
| White 2000, Econometrica 68(5):1097-1126 (Reality Check) | yes | yes |
| Hansen 2005, JBES 23:365-380 (SPA) | yes | yes |
| Diebold & Mariano 1995, JBES 13(3):253-263 | yes | yes |
| Lucas, Tucker, Grosse & Norouzi 2019, NeurIPS, "Don't Blame the ELBO" | yes | yes: linear-VAE / pPCA correspondence; observation-noise variance governs the stability of collapsed solutions |
| Tipping & Bishop 1999, pPCA | yes | yes: ML σ² = mean of the discarded eigenvalues |
| Burda, Grosse & Salakhutdinov 2015 (ICLR 2016), IWAE | yes | **no** for the corpus's use: the paper defines activity as Cov_x(E_q[u]) > 10⁻² (not KL > 0.01 nats) and contains no "K ≥ 1.5-2× intrinsic dimensionality" rule. The corpus misattribution (DVT L420, L1662) is confirmed |

### Verdicts

| ID | Verdict | Final severity | Reason |
|---|---|---|---|
| U01-CRI-01 | downgraded | major | "We are not forecasting" is false: the pipeline forecasts second moments. But the comparative premise (second moments are more tractable than means) is standard (Merton 1980, from memory), so "no theorem links…" overstates. |
| U01-CRI-02 | confirmed | critical | Recomputed: same BΣ_zBᵀ, H = 0.207 vs 0.298 after a column rescaling. The principal factors of Σ_z depend on the arbitrary latent scale. |
| U01-CRI-03 | confirmed | critical | The q² dilution algebra is correct. Window fraction 8.9% rather than 8.3%: immaterial. |
| U02-CRI-01 | downgraded | major | The logical argument is valid, but it has no math or literature evidence, which the evidence rule requires for critical. |
| U02-CRI-02 | confirmed | critical | Recomputed 1/(AU·p+1−p) = 20.2%. |
| U02-CRI-03 | confirmed | critical | Gaussian tail independence verified (Embrechts et al.). A covariance-only risk functional cannot express tail dependence. |
| U02-CRI-04 | confirmed | critical | Constant Σ_z and D_ε with diagonal rescaling: the correlation change is a functional-form artefact. |
| U03-CRI-01 | downgraded | major | The counterexample only covers factors never active in sample. The corpus (DVT L78) limits its claim to "all factors ever observed". |
| U03-CRI-02 | confirmed | critical | ISD L596 (adj_price is dividend-adjusted) and L649 (cap = Adjusted_close × shares). The ranking is biased by dividend yield. The current-vs-historical shares point is conditional. |
| U03-CRI-03 | confirmed | critical | Shumway figures are corrections for performance-related delistings. Without CRSP the delisting cause is unknown, so the imputation hits mergers. z-score arithmetic recomputed. |
| U04-CRI-01 | confirmed | critical | Both channels are z-scored per window (DVT L226-246), so the input is invariant to c·r. RV cannot restore scale. |
| U04-CRI-02 | confirmed | critical | DVT L368 (N = 7.06M) contradicts L651/L1076 (N_eff ≈ T_hist/T). r_eff ≈ 693-1,087. |
| U05-CRI-01 | confirmed-narrowed | critical | The counterexample in the row is self-contradictory ("sign-blind? No —"). The retained claim, also made by U07-CRI-01 and U10-CRI-01: no term of the final objective ties μ to a cross-sectional loading. |
| U05-CRI-02 | downgraded | major | span(D_tB) = span(B) is false for non-scalar D_t. Invariance under B→BA⁻¹ leaves Σ_assets unchanged, so it is not a defect. Valid residue: diagonal D_ε is never tested. |
| U05-CRI-03 | confirmed | critical | θ = 2σ²ΔKL/D recomputed (0.099%, 0.033%). AU depends on D. D = 504 vs 1008 inconsistency confirmed (L448, L464, L490). |
| U05-CRI-04 | confirmed | critical | Ky Fan bound: at equal dimension, in sample and with a static projector, EP_VAE ≤ EP_PCA. The comparison at unequal dimension is vacuous. |
| U06-CRI-01 | confirmed | critical | Recomputed (1 vs 2 active units). Caveat: the rotation counterexample needs an isotropic posterior (σ²=1). A diagonal posterior with σ²≠1 breaks the rotation symmetry, but permutation and the split-over-m-coordinates argument still hold. |
| U06-CRI-02 | confirmed | critical | Arithmetic slip: the correct formula floors at 84, not 85. The structural point holds: the guard binds (L436), so the count is a function of sample length. |
| U06-CRI-03 | confirmed | critical | (c) holds only if the guard is applied per fold, which the corpus does not specify (see U14-CRI-03). |
| U07-CRI-01 | confirmed | critical | Order-of-magnitude comparison consistent with DVT L490 and ISD. |
| U07-CRI-02 | downgraded | major | The math is correct but duplicates U05-CRI-03. The independence point is a likelihood misspecification, not an invalidation. |
| U08-CRI-01 | confirmed | critical | Recomputed: target 0.707 vs implied correlation 0.894. |
| U08-CRI-02 | downgraded | major | γ enters only the reconstruction term (L586-588), so it cannot couple stocks. But upweighting synchronous crisis windows does emphasise crisis-period patterns: the mechanism is misstated, not void. |
| U08-CRI-03 | downgraded | major | Pairwise-deletion targets need not be PSD (correct; eigenvalue −1 example verified). "Imposes an elliptical copula" is overstated, since Spearman is rank-based. Applies to Phases 1-2 only. |
| U09-CRI-01 | downgraded | major | Count correct: the validation set covers one 2-year regime. That is standard for time-series early stopping; "calibrated on noise / degenerate" overstates. |
| U09-CRI-02 | confirmed | critical | ISD L1400: the schedule is a fraction of total_epochs. Phase A stops at E* < 60 inside Phase 2. Phase B semantics are unspecified, so the validated and deployed objectives differ in at least one reading. |
| U10-CRI-01 | confirmed | critical | E[R²] = K/(n−1) under the null (standard). 8% > the corpus's own 1-3%. |
| U10-CRI-02 | downgraded | major | (ii) is wrong: collapse sends q to the prior (σ²→1), and the KL penalises σ² = 0.01 (KL = 1.81). (i) is a borderline hypothetical. Valid residue (iii): KL ranks dimensions by within-stock temporal variance, not by composite SNR. |
| U10-CRI-03 | confirmed | critical | Residual-inflation identity and sign cancellation are correct. The DVT's own L724 argument applies. |
| U11-CRI-01 | confirmed | critical | The printed formula gives 59.95, the correct one 84.99. The table matches √(2N/r) (ISD L222). |
| U11-CRI-02 | downgraded | major | The derivation is wrong: it applies the deflator autocorrelation to the whole variance and treats the observed deflator path as random, so N_eff ≈ 30 is not established. Valid residue: the LW intensity assumes i.i.d. returns (verified); there is no HAC or volatility-clustering treatment. |
| U11-CRI-03 | confirmed-narrowed | critical | Core verified: identity-target shrinkage keeps eigenvectors and compresses the spread by (1−δ), so δ moves H; the PCA arm is unshrunk; ln AU vs ln k. Removed: the (π−ρ)/γ formula (wrong target, see literature) and "δ* ≈ p/T ≈ 0.43" (unsupported). |
| U11-CRI-04 | downgraded | major | The leverage bias E[ε̂²] = σ²(1−h_ii) is standard. Magnitudes hold under the stated isotropy assumption. It is correctable and does not invalidate the unit. |
| U11-CRI-05 | downgraded | major | Mixed scalings (L753 vs L757) and the in-sample floor AU/n are verified. EP is a gate and a Layer-2 target, not the decision metric. |
| U12-CRI-01 | confirmed | critical | Recomputed: 38.8% vs a 1.25% target. |
| U12-CRI-02 | rejected | — | It treats sample eigenvalues as true ones, whereas shrinkage is used because the sample spectrum is over-dispersed. Arithmetic slip (0.109, not 0.101). The residue (1/√λ amplification of small-eigenvalue error) is absorbed by U12-CRI-01 / RC-3. |
| U12-CRI-03 | confirmed-narrowed | critical | The degenerate counterexample is correct (H: 0 vs ln 2). Correction: identity-target shrinkage does not change the eigenvectors, so it does not create the indeterminacy. Instability comes from close sample eigenvalues (eigenvector error ∝ 1/gap, Davis-Kahan, from memory). |
| U12-CRI-04 | confirmed | critical | Ĥ_OOS is evaluated on the frozen B and Σ_z (DVT L878, L898); H is homogeneous of degree 0. |
| U13-CRI-01 | downgraded | major | Comparing levels is not a marginal analysis, and the "step frontier" is unproven. Valid: the units mismatch (α "in return units", L1052) and uncalibrated ranges. |
| U13-CRI-02 | confirmed | critical | With μ = 0, any α > 0 is weakly dominated in model variance. Entropy can only add value through model error, which is never formalised or tested. |
| U13-CRI-03 | confirmed-narrowed | critical | Same as U12-CRI-03, with the same correction on shrinkage. |
| U14-CRI-01 | confirmed | critical | E[max] recomputed: 2.07σ (m=32), 1.03σ (m=4). |
| U14-CRI-03 | downgraded | major | The penalty arithmetic assumes AU = 80 at fold 1, which a per-fold guard forbids (cap 47). If the guard is per fold, the penalty is identically 0 (U06-CRI-03c); if global (85), it bites. The true defect is underspecification. The median vs per-fold inconsistency is confirmed (doc). |
| U14-CRI-04 | confirmed-narrowed | critical | Counterexample recomputed (0.722 → 0.608). It shows non-comparability across AU, not "mechanically decreasing". |
| U15-CRI-01 | confirmed | critical | DVT L894/L898 vs ISD L2434: selection on the reported OOS. |
| U15-CRI-02 | confirmed-narrowed | critical | Ĥ is non-comparable and undefined for 1/N and inverse-vol. Correction: Scenario A stays decidable through the per-metric tests (Benchmark §4.2); only the composite-score test is undefined for two arms. |
| U15-CRI-03 | confirmed | critical | Power recomputed: 0.40 (n=34, d=0.3). The PCA switch is not admissible under the user decision, but the inferential error stands. |
| U16-CRI-01 | confirmed-narrowed | critical | Merged with U15-CRI-02, same correction. |
| U16-CRI-02 | confirmed | critical | Merged with U14-CRI-01. |
| U16-CRI-03 | confirmed | critical | AR(1) size inflation recomputed (≈0.20 at ρ = 0.4). Diebold-Mariano, White and Hansen verified. |
| U17-CRI-02 | confirmed-narrowed | critical | Merged with U12-CRI-03, same correction on shrinkage. |
| U17-CRI-03 | confirmed | critical | Current-volatility rescaling (L742) is reactive de-weighting with a lag the DVT concedes (L734). "Anti-cyclical by construction" (L17) is false for that channel. |
| U17-CRI-05 | confirmed | critical | 1−0.95⁵ = 0.226 and 0.265 recomputed. White and Hansen verified. The holdout (L880) is unused by the rule. |
| U18-CRI-01 | confirmed | critical | Duplicates U04-CRI-02, plus the risk-model parameters are absent from any capacity check. |
| U18-CRI-02 | confirmed | critical | Merged with U14-CRI-01. √(2 ln 32) = 2.63 is an asymptotic approximation; the exact value is 2.07. |
| U18-CRI-03 | confirmed-narrowed | critical | Correction: the table is computed with √(2N/r) (ISD), not with coefficient 8, which gives 84/77/68/58/35. The "79" from sigma_z_ewma_justification.md L76 is a code changelog value, inadmissible as evidence. |
| U18-CRI-04 | rejected | — | Under the violation INV-001 describes (D missing from the reconstruction numerator, (D/2)·ln σ² kept), ∂L/∂σ² = 0 gives σ² = MSE/D, which heads to the lower clamp. The detector is therefore consistent. Only the F=1 vs F=2 doc point survives (minor, duplicates U05-CRI-03). |
| U19-CRI-01 | confirmed | critical | Duplicates U11-CRI-01. 83 with N = 7,057 recomputed (83.51). |
| U19-CRI-02 | confirmed | critical | Burda misattribution verified on the paper, which also uses a different activity criterion. Lucas et al. and Tipping & Bishop verified. |

### Tally

- Confirmed (including narrowed): 43. Downgraded: 14. Rejected: 2. Total: 59.
- Duplicate clusters after merging, all on the same evidence:
  - selection on OOS: U06-03e, U14-01, U15-01, U16-02, U18-02;
  - AU guard: U11-01, U18-03, U19-01;
  - degenerate-eigenbasis H: U12-03, U13-03, U17-02;
  - Ĥ comparability: U14-04, U15-02, U16-01;
  - N vs N_eff: U04-02, U18-01.

## Annex — Stream cuts and step failures (data for fixing the kit)

**What was counted.**
- A *cut* is a CommandCode `stream_restart` event: the provider stream broke and CommandCode restarted the request.
  It does this at most 3 times per request.
- "Interrupted response" gives the phase of the request when the stream broke:
  - `thinking`: a thinking block was open;
  - `visible`: the thinking block had closed, so visible text or a tool call was streaming;
  - `pre-stream`: nothing had been received yet.
- "Resolution" says how the agent eventually produced its report:
  - `internal retry`: the same request completed after the restart, in the same step;
  - `session resume (fN.rM)`: the launcher's `--resume` step produced the report;
  - `fresh run (fN…)`: a later fresh run produced it;
  - `killed (step interrupted from outside)`, or unresolved because a later step was killed: the step was stopped by
    the orchestrator, either by the former stop-on-cut rule or for a prompt change.
- Classification comes from `campaign/scripts/cut_report.py` over `runs/`, `runs_rerun/`,
  `runs_failed/U02_advocate_uncapped/` and `runs_failed/stopped_by_watcher/`.

| Agent | Step | Turn | Retry attempt | Interrupted response | discardedVisibleContent | Resolution | Run dir |
|---|---|---|---|---|---|---|---|
| U01_advocate | f1 | 10 | 1 | visible | True | internal retry | runs |
| U05_critic | f1 | 6 | 1 | visible | True | internal retry | runs |
| U07_critic | f1 | 7 | 1 | visible | True | internal retry | runs |
| U07_critic | f1 | 8 | 1 | visible | True | internal retry | runs |
| U08_critic | f1 | 2 | 1 | visible | True | internal retry | runs |
| U08_critic | f1.r1 | 1 | 1 | visible | True | fresh run (f2.r1) | runs |
| U08_critic | f1.r1 | 1 | 2 | visible | True | fresh run (f2.r1) | runs |
| U08_critic | f2 | 8 | 1 | visible | True | session resume (f2.r1) | runs |
| U11_critic | f1 | 9 | 1 | visible | True | internal retry | runs |
| U16_critic | f1 | 8 | 1 | visible | True | internal retry | runs |
| U16_critic | f1 | 8 | 2 | visible | True | internal retry | runs |
| U17_critic | f1 | 15 | 1 | visible | True | internal retry | runs |
| U05_critic | f1 | 9 | 1 | visible | True | session resume (f1.r1) | runs_rerun |
| U02_advocate | f1 | 9 | 1 | visible | True | unresolved: later step killed from outside | U02_advocate_uncapped |
| U02_advocate | f1 | 9 | 2 | visible | True | unresolved: later step killed from outside | U02_advocate_uncapped |
| U02_advocate | f1 | 9 | 3 | visible | True | unresolved: later step killed from outside | U02_advocate_uncapped |
| U02_advocate | f1.r1 | 1 | 1 | pre-stream | True | unresolved: later step killed from outside | U02_advocate_uncapped |
| U02_advocate | f1.r1 | 1 | 2 | visible | True | unresolved: later step killed from outside | U02_advocate_uncapped |
| U02_advocate | f1.r1 | 1 | 3 | pre-stream | True | unresolved: later step killed from outside | U02_advocate_uncapped |
| U02_advocate | f1.r2 | 1 | 1 | pre-stream | True | unresolved: later step killed from outside | U02_advocate_uncapped |
| U02_advocate | f1.r2 | 1 | 2 | pre-stream | True | unresolved: later step killed from outside | U02_advocate_uncapped |
| U02_advocate | f1.r2 | 1 | 3 | pre-stream | True | unresolved: later step killed from outside | U02_advocate_uncapped |
| U02_advocate | f2 | 14 | 1 | visible | True | internal retry | U02_advocate_uncapped |
| U02_advocate | f2 | 15 | 1 | visible | True | killed (step interrupted from outside) | U02_advocate_uncapped |
| U02_advocate | f2 | 15 | 2 | visible | True | killed (step interrupted from outside) | U02_advocate_uncapped |
| U01_advocate | f1 | 10 | 1 | visible | True | killed (step interrupted from outside) | stopped_by_watcher |

Total cuts: 26

**Summary.** 26 cuts in total.

| Group | Cuts | Detail |
|---|---|---|
| Final campaigns (`runs/`, `runs_rerun/`) | 13 | 9 resolved by internal retry; 3 by a later fresh run plus session resume (U08_critic); 1 by session resume (U05_critic rerun) |
| `U02_advocate` v1-uncapped (60 KB-class report) | 12 | 9 on f1/f1.r1/f1.r2, never resolved; 1 internal retry on f2; 2 on f2 when the step was stopped for the prompt change |
| Former stop-on-cut rule | 1 | `U01_advocate`, stopped by the watcher; the same agent later passed its turn-10 cut by internal retry |

**Observations.**
- **Phase.** Every cut except five pre-stream restarts of the uncapped U02 advocate's resumes happened in phase
  `visible`, after the thinking block had closed. None happened while thinking was streaming.
  `discardedVisibleContent` was true in every case.
- **Correction of an earlier statement.** The U01_advocate turn-10 cut was first reported as happening "before the
  report started". The transcript shows the thinking block had closed and visible output was streaming.
- **Size cap.** With the v2 prompt (size cap), no cut ended in failure.
- **Timeouts.** Step failures other than cuts were timeouts (exit 124, 40 min fresh run / 20 min resume):

| Agent | Step | Exit | Seconds | Run dir |
|---|---|---|---|---|
| U08_critic | f1 | 124 (timeout) | 2400 | runs |
| U08_critic | f1.r1 | 124 (timeout) | 1200 | runs |
| U11_critic | f1 | 124 (timeout) | 2400 | runs |
| U08_critic | f2 | 124 (timeout) | 2400 | runs |
| U19_critic | f1 | 124 (timeout) | 2400 | runs |
| U05_critic | f1 | 124 (timeout) | 2400 | runs_rerun |

- **U08_critic sequence.** f1 timeout → f1.r1 cut twice then timeout → f1.r2 exited 0 **without** END-OF-REPORT →
  fresh run f2 timeout → f2.r1 complete (5 steps, 7,136 s in total).
- **Lesson for the kit.** An exit code of 0 is not success. The launcher correctly keys on the sentinel.
