# U12 — advocate

## Scope read

- `docs/DVT_strategie_facteurs_risque_latents_v4_1.md` §4.8 (lines 855–984), plus 1058–1064, 1192
- `docs/ISD_vae_latent_risk_factors.md` MOD-009 (2202–2311), module table 289, 329
- `docs/Benchmark_latent_risk_factor.md` §1.4 (50–61), §1.5 (63–65), §4 (203–228), §5 (232–242)
- `src/walk_forward/folds.py` (all 168 lines), `phase_a.py` (all), `phase_b.py` (all), `selection.py` (all)
- `src/walk_forward/metrics.py` 37–130, 601–710; `src/walk_forward/oos_rebalancing.py` 129–172, 640–822, 260–261
- `src/integration/pipeline.py` 284–320, 431–470, 542–803, 855–974, 1236–1497, 1722–2111, 2360–2399, 2700–2770, 3022–3071
- `src/integration/reporting.py` 42–121, 117–176, 160–219; `src/integration/composite.py` 21–116
- `src/validation.py` 632–713; `src/training/trainer.py` 812–856
- `src/data_pipeline/windowing.py` 102–192; `src/config.py` 746–790
- `notebooks/dashboard.ipynb` lines 221, 243–244, 859–862, 1144–1151, 1206–1222
- Tests: `tests/integration/test_walk_forward.py` 1–60; `tests/unit/test_invariants.py` 391–418, 450–470
- Results: `results/diagnostic/{config.json, report.json, diagnostic_report.md, vae_fold_metrics.csv}`; `results/{report.json, vae_fold_metrics.csv}`

## Five questions

**1. Purpose and original intent.** The unit exists to make every other number in the project admissible: "The validation protocol answers whether the strategy *actually* works without hindsight. Without it, all metrics are suspect" (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:862`). The design intent is explicitly distributional, not point-estimate: walk-forward "produces a **distribution** of OOS results across ~34 folds covering crisis and calm periods" (`…v4_1.md:866`), with benchmarks "to determine whether its complexity is justified" (`…v4_1.md:862`), and the architectural exit clause that if the VAE does not beat PCA, "Iteration 6 (abandoning the VAE) should be adopted" (`…v4_1.md:1192`). This is an unusually disciplined intent: a pre-registered decision rule with an abandon branch.

**2. What a state-of-the-art version would look like.** Nested walk-forward in which *every* choice that touches the test window is made strictly before that window opens: HP selection on an inner temporal split inside `[start, t_k−2yr]` (or a HP set frozen once, pre-registered, and never revisited), one model per fold, one fixed rule applied to each disjoint OOS block. The literature is unambiguous on this: White's Reality Check (2000) and Hansen's SPA (2005)/Romano–Wolf (2005) exist precisely because backtest statistics are maxima over trials; Bailey & López de Prado's Deflated Sharpe (2014) and PBO/CSCV were invented for the case where the *selection* used the reported sample; Arnott–Harvey–Markowitz (2019) codify pre-registration and "hold out the holdout"; López de Prado (2018) adds purging/embargo (already present); Gu–Kelly–Xiu (2020) demonstrate the expanding-window + genuine holdout discipline for ML asset pricing. A SOTA U12 would also publish the number of trials consumed (18 configs × 34 folds), because that number determines the haircut.

**3. What prevents the current unit from reaching it — and which obstacles are only apparent.**
The blocking obstacles, evidenced:
- Phase A never computes the quantities it claims to select on (`pipeline.py:1487-1497`), so the "18-config nested search" does not exist in the code.
- The only runnable entry point sets `skip_phase_a = True` (`notebooks/dashboard.ipynb:1144`), so even the stub is bypassed.
- Phase B is not the "no validation set" deployment run it claims (`pipeline.py:1887-1889`, `trainer.py:839-841`).
- The holdout is pooled into the fold table and the paired tests (`pipeline.py:764`, `782`).
- The schedule is anchored to a declared calendar horizon, not to the data (`folds.py:46-47`).

Obstacles that are **only apparent**:
- *"34 folds give a distribution."* Only if the folds are not reused for selection. As specified (`…v4_1.md:894`), the config is scored on the fold's own OOS, so the aggregate is a distribution of per-fold maxima, not of a strategy.
- *"Phase B retrains on all data, so the backtest is deployment-faithful."* The code restores the best `val_elbo` checkpoint, on a stock-based split, so the deployed weights are still validation-selected — just by a different split than the spec names.
- *"Fixed E\* removes a degree of freedom."* It does not: `restore_best` reintroduces epoch selection through `val_elbo`.
- *"The holdout is the uncontaminated verdict."* It is uncontaminated by fitting, but not by reporting or by the development loop that `run_direct` supports.
- *"Elimination uses four hard constraints."* Three of them are constants; only AU binds.

**4. Real vs self-imposed constraints.**
Real: the number of non-overlapping 6-month windows on 31 years of data (~34 at best, and only if the schedule is data-anchored); serial dependence of factor structure; 788-stock universe with 35.31% missing values (`results/diagnostic/diagnostic_report.md`); Tiingo's absent delisting returns; the compute cost of retraining a VAE per fold. Self-imposed: the *two-phase* per-fold design (2× training to buy a selection that cannot be measured on the same window); the 18-config grid evaluated per fold; OOS-based scoring (H and MDD terms); the 90/10 monitor inside Phase B; the 3-year holdout being simultaneously a reporting artifact and, via `run_direct`, a development test set; anchoring the horizon on `total_years` rather than `returns.index[-1]`.

**5. What should be removed, merged or moved.** (No patches; structural only.)
- **Remove** the OOS terms from the selection score. Selection on the reporting window is the one flaw that no implementation fix repairs — it must leave the architecture. H and MDD stay as *reported* metrics.
- **Remove** the Phase A / Phase B split. It costs 2× training per fold and buys nothing once selection is not OOS-based; one training per fold on `[start, t_k]`, with HP frozen by pre-registration, is strictly more honest and cheaper.
- **Remove** the `R_Σ` estimation penalty. Statistical identification is already enforced upstream by `compute_au_max_stat(..., r_min, …)` (`pipeline.py:2017-2021`), which caps AU so that N_obs/(AU(AU+1)/2) ≥ r_min; scoring it again double-counts a constraint already satisfied.
- **Remove** `run_direct` from the evidence path (keep only as a smoke test): a single split with `oos_start` = the day after `train_end` (`pipeline.py:905-926`) has no embargo, no folds, no Phase B, and cannot support any protocol claim.
- **Remove** zero-filled failure records (`_empty_metrics`, `pipeline.py:3047-3056`); a failed fold must be NaN + an explicit count, otherwise zeros enter medians and Wilcoxon pairs.
- **Move** the holdout out of `vae_results` into a single-shot, separately reported evaluation with its own pre-registered metric set.
- **Merge** `check_deployment_criteria` and `compile_pairwise_tests` into one decision rule; the current deployment block cannot express the doc's abandon branch (see U12-ADV-17).

## Load-bearing assumptions

| # | Assumption | Where stated | Evidenced? | Comment |
|---|---|---|---|---|
| A1 | OOS windows are disjoint and tile the evaluation period | `docs/DVT_...v4_1.md:882-890` | yes | `folds.py:69-70,98` — step = OOS length, so they tile |
| A2 | Per-fold OOS selection yields an unbiased estimate of the deployed rule | `…v4_1.md:894,929` | **no** | Circular: the score *is* the reported quantity |
| A3 | The holdout is untouched by selection and reporting | `…v4_1.md:880` | partially | Not used for fitting; pooled into summaries (`pipeline.py:782`) |
| A4 | E\* transfers from the Phase A subset to the Phase B full window | `…v4_1.md:900` | partially | Untested; contradicts `trainer.py:839-841` |
| A5 | 21 trading days suffice to break residual dependence | `…v4_1.md:876` | partially | Reasoned correctly, never measured empirically |
| A6 | A 6-month block is a usable evaluation unit | `…v4_1.md:1059` | yes | ~126 days; protocol compensates by reporting a distribution |
| A7 | Fold-level paired tests need no dependence correction beyond Holm | `docs/Benchmark_...md:207-224` | partially | Disjoint OOS returns help; shared expanding histories do not |
| A8 | Phase A AU (val windows) is comparable to Phase B AU (full training) | `pipeline.py:1473-1482` vs `1986-2016` | no | Different estimation samples; the selected config is not the deployed one |
| A9 | The data ends where the schedule says it ends | implicit in `folds.py:46` | **no** | Violated at `total_years=35/40` |
| A10 | A missing return is a zero return | `metrics.py:629` | **no** | Contradicted by `oos_rebalancing.py:663-674`, which charges −30% for the same NaN |

## Findings

| ID | Type | Severity | Evidence register | Claim (quoted, path:line) | Assessment | Evidence (path:line) | Cross-unit |
|----|------|----------|-------------------|---------------------------|------------|----------------------|------------|
| U12-ADV-01 | justification | solid | doc | "Standard random train/test splits leak future information for financial time series… A single split yields one data point; walk-forward produces a **distribution**" | Correct and standard; the rejection of random splits is the right first move | `docs/DVT_strategie_facteurs_risque_latents_v4_1.md:866` | U01 |
| U12-ADV-02 | modeling | solid | doc+code | "Expanding (not rolling) to preserve all past crisis data — consistent with the anti-cyclical principle" | Correct choice; faithfully implemented (`train_start = data_start`) | `…v4_1.md:872`; `folds.py:53-58` | U01 |
| U12-ADV-03 | leakage | solid | doc+code+test | Embargo "addresses residual temporal dependence (volatility clustering, momentum), not data overlap — zero shared daily returns exist even without embargo" | The distinction is precise and the strongest justification in the document; constructed as **business** days and tested as such | `…v4_1.md:876`; `folds.py:66`; `tests/unit/test_invariants.py:391-418` | U04 |
| U12-ADV-04 | statistics | solid | code | Wilcoxon signed-rank per (benchmark, metric), 10,000-rep bootstrap CI, Holm over the 6×4 family | Exactly what `Benchmark §4` prescribes; correct test family and correct correction | `src/integration/reporting.py:55-108`; `docs/Benchmark_latent_risk_factor.md:203-224` | U13, U14 |
| U12-ADV-05 | math | solid | doc+code | "$\hat H = H(w)/\ln(AU)$ … $\in[0,1]$"; worked example 3.80/ln80 = 0.867 vs 4.20/ln120 = 0.877 | Normalization rationale (AU varies by config) is sound and the arithmetic checks out; the unit-bridge that makes λ_pen interpretable is a genuine contribution | `…v4_1.md:935-953`; `src/walk_forward/phase_a.py:44-58` | U10, U15 |
| U12-ADV-06 | implementation | solid | code | "Compute daily portfolio return (before any delisting handling)" → drift → rebalance | Causal timing is correct: day-t return uses pre-t weights; rebalance at close of t applies from t+1 | `src/walk_forward/oos_rebalancing.py:642-706,743-822` | U11 |
| U12-ADV-07 | implementation | solid | code | "Factor regression on full expanding window [train_start, current_date]" | In-OOS risk-model refresh uses only past data; no future slice | `src/walk_forward/oos_rebalancing.py:260-261` | U09 |
| U12-ADV-08 | leakage | solid | code+test | Crisis threshold invariance: adding future data must not change the threshold | Point-in-time discipline is *proved* by test, not asserted; rare in this codebase | `tests/unit/test_invariants.py:450-470` | U14 |
| U12-ADV-09 | implementation | fragile | code | `assert_no_lookahead(str(fold["train_end"]), str(fold["oos_start"]), …)` — `min_embargo_days` defaults to 0 | The embargo is never asserted at runtime; config even permits `embargo_days=0`. One deleted test removes the guarantee | `folds.py:91-93`; `pipeline.py:450-453`; `src/config.py:776` | U17 |
| U12-ADV-10 | implementation | critical | code | `"H_oos": 0.0,  # Simplified` / `"mdd_oos": 0.0` / `"explanatory_power": 0.5,  # Placeholder` | Phase A computes **no** OOS quantity. All survivors score 0 → the first grid entry (`mode=P, lr=5e-4, alpha=0.5`) always wins. The 18-config search is a no-op | `src/integration/pipeline.py:1487-1497`; `phase_a.py:139-160`; `pipeline.py:3028-3037` | U15 |
| U12-ADV-11 | math | critical | code | `"n_obs": train_windows.shape[0]` fed to `R_sigma = n_obs / max(AU*(AU+1)/2, 1)` | Wrong unit and wrong magnitude: window count (~10⁵) instead of covariance observations (~126). Penalty is identically zero — a dead constraint | `pipeline.py:1493`; `phase_a.py:49`; intent `…v4_1.md:935` | U09 |
| U12-ADV-12 | protocol | critical | doc | "Evaluate on OOS → fold score. Select best config by OOS score" | Even fully implemented, this selects on the window it reports. Ĥ and MDD become selected order statistics (MDD biased *down* by construction); the aggregate describes no deployable rule | `…v4_1.md:894,929,933-937` | U10, U15 |
| U12-ADV-13 | leakage | critical | code+notebook | `skip_phase_a = True` (hardcoded) then `skip_phase_a=(… or skip_phase_a)` | The designated entry point disables the selection protocol; the deployed config is a literal default dict. No run in this repo ever executed Phase A | `notebooks/dashboard.ipynb:1144`; `pipeline.py:576-577,637-639,654` | U15, U16 |
| U12-ADV-14 | implementation | high | code | "Use 90% for train, 10% for validation monitor (but don't stop early)"; `restore_best(self.model)` | Phase B does keep a validation set and restores its best-ELBO weights. Because `create_windows` is stock-major, the 10% is the *last stocks*, not the last 2 years — a cross-sectional monitor, contrary to `…v4_1.md:874,896` | `pipeline.py:1887-1892`; `src/data_pipeline/windowing.py:189`; `trainer.py:839-841` | U05, U07 |
| U12-ADV-15 | statistics | high | code+results | `self.record_vae_result(holdout_id, …)` then `vae_df = aggregate_fold_metrics(self.vae_results)` | The holdout is pooled with the folds: it inflates `n_folds`, enters medians/means and the Wilcoxon pairs, though it is a different model and a different horizon (232 vs ~126 OOS days) | `pipeline.py:764,782`; `results/vae_fold_metrics.csv:1-10`; `…v4_1.md:880` | U14 |
| U12-ADV-16 | robustness | high | code+results | `fold_crisis_fractions=None,  # Crisis fractions per fold not tracked yet` → `{"available": False, "reason": …}` | The mandated regime decomposition is inert in every run; the "crisis" layer of Layer 3 is computed from a VIX series that the notebook passes as `None` | `pipeline.py:790`; `reporting.py:133-134`; `notebooks/dashboard.ipynb:1151`; `…v4_1.md:961` | U13, U14 |
| U12-ADV-17 | consistency | medium | code | `scenario = "B"` … `elif pca_tie: scenario = "C"` | Scenario D ("Use 1/N") and E are unreachable; `"significant": False` is hardcoded. The doc's abandon clause cannot be produced by the code | `selection.py:132-147,243`; `…v4_1.md:1192` | U14, U16 |
| U12-ADV-18 | data/schedule | high | code+notebook+results | `data_end = data_start + pd.DateOffset(years=total_years)` | The schedule is anchored on a declared horizon, never on `returns.index[-1]`. With `N_YEARS=40` (notebook) or `total_years=35` (diagnostic) against data ending 2026-02-13, the holdout and trailing folds have empty OOS; with 30, the last 1.1 years are silently outside the holdout | `folds.py:46-47,55`; `notebooks/dashboard.ipynb:243`; `results/diagnostic/config.json`; `results/diagnostic/diagnostic_report.md` | U03, U15 |
| U12-ADV-19 | results | critical | results+code | "Training diagnostics not available: pretrained model (no training)"; `n_days_oos = 1575.0`; `holdout_fraction` split with `oos_start = test_dates[0]` | The only real-data artifact is **not** a walk-forward: one synthetic fold, zero embargo, no Phase A/B. No U12 guarantee has ever been exercised on real data. The unit's guarantees are therefore untested, not merely unproven | `results/diagnostic/diagnostic_report.md`; `results/diagnostic/vae_fold_metrics.csv:2`; `pipeline.py:897-945` | U16, U02 |
| U12-ADV-20 | interpretation | medium | code+results | `R_oos = np.nan_to_num(np.asarray(returns_oos[available].values), nan=0.0)` vs NaN→delisting at −30% | The same missing observation is a 0 return in metrics and a Shumway loss in the simulator; with 35.31% missing data the OOS P&L is not a well-defined object | `metrics.py:629`; `oos_rebalancing.py:663-674,129-172`; `results/diagnostic/diagnostic_report.md` | U03, U14 |
| U12-ADV-21 | statistics | fragile | code | `vae_vals = …dropna().values; bench_vals = …dropna().values; n = min(len(…))` | Independent `dropna` + positional truncation can misalign pairs; failed folds are recorded as `0.0` with `"error": 1.0` and are not filtered, so a NaN in one series silently shifts the pairing | `reporting.py:62-74`; `pipeline.py:3047-3056` | U14 |
| U12-ADV-22 | modeling | conditional | doc+code | "more training data implies convergence at least as fast, so $E^*$ is a conservative upper bound" | An epoch is a pass over a *larger* set, i.e. more optimizer steps; E\* can overtrain. The 20%-loss-gap check is the right response — it converts the assumption into a falsifiable diagnostic — but it only `flag`s | `…v4_1.md:900`; `phase_b.py:51-76` | U07 |

## Core strengths to preserve

1. **The fold skeleton.** Expanding window, contiguous non-overlapping 6-month OOS blocks, 21 *trading*-day embargo, 10-year minimum, 3-year holdout. This is literature-standard and correctly constructed; any redesign should keep the geometry and only re-anchor `data_end` to the data.
2. **The embargo rationale.** Stating that the embargo addresses dependence, not overlap, is the sort of precision that prevents the usual cargo-cult purge. Keep the sentence; add the empirical check it implies (window-expiry information decay).
3. **The three-layer metric taxonomy and the per-fold distribution.** Reporting every fold, not a mean, is the correct primary artifact; fold sensitivity is thereby visible by construction (`…v4_1.md:959`).
4. **The paired statistical layer** (Wilcoxon + bootstrap CI + Holm over a declared 24-test family). This is the part of U12 that is fully built and defensible; it should become the *only* decision mechanism.
5. **The two-phase separation as an architectural idea** — select somewhere, then retrain on everything and deploy deterministically. The idea is right even though the current implementation violates it; the redesign should keep "deployment model = refit on all available data" and move the *choice* out of the OOS window.
6. **The E\* transfer sanity check.** A pre-declared 20% loss-gap flag is a genuine falsification test of an assumption the protocol otherwise takes on faith. Keep it and make it blocking, not advisory.
7. **The abandon clause.** A written commitment to drop the VAE if it loses to PCA is the most valuable sentence in the spec; the redesign must keep it and make it reachable in code.

## Verdict

The specified protocol is better than most published equity backtests: expanding walk-forward, genuine trading-day embargo, a pre-registered composite score with an explicit normalization argument, a separate holdout, six benchmarks, and a paired-test family with FWER control. But it contains one design flaw no implementation can repair — per-fold selection on the very OOS window that is reported — and the implementation does not even reach that flawed design: Phase A computes no OOS score at all (`H_oos=0`, `mdd_oos=0`, `EP=0.5`), its estimation penalty is identically zero because it counts windows instead of dates, the notebook sets `skip_phase_a=True`, Phase B silently restores a validation-selected checkpoint on a stock-based split, the holdout is pooled into the fold table and the p-values, the regime decomposition is unwired, and the only real-data artifact in the repository is a single pretrained-split evaluation with no embargo. The skeleton (fold geometry, embargo, statistical layer, decision architecture) is sound and worth preserving; the selection protocol is the part that must be deleted rather than patched, because its cost (2× training per fold, 34×18 fits) buys a statistic that cannot be interpreted. As an advocate, I can defend the design of the unit and the correctness of its statistical machinery, but not its fidelity, and not its central claim that the reported aggregate is an out-of-sample estimate.

Score: 4/10

END-OF-REPORT
