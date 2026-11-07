# U01 — advocate

## Scope read

- `docs/strategy_philosophy.md` §1–§4 (lines 7–58), plus §5–§10 (60–183) for the objective, modularity and abandonment clauses.
- `docs/DVT_strategie_facteurs_risque_latents_v4_1.md` Executive Summary (7–17), §1 (21–39), §2 (43–89), §3.2/3.4 (113–143), §4.1 (150–198), §6.2–6.4 (1101–1137).
- `docs/sigma_z_ewma_justification.md` (1–75).
- `src/config.py`: data (117–158), VAE (165–214), loss (221–260), inference (346–376), risk model (383–505), portfolio (512–738), walk-forward (745–789).
- `src/inference/composite.py` (1–172); `src/inference/active_units.py` (1–218).
- `src/integration/pipeline.py` (1960–2029, 2486–2515, 2557–2565, 2776–2782); `src/integration/notebook_helpers.py` (155–196).
- `src/walk_forward/folds.py` (15–121); `src/walk_forward/oos_rebalancing.py` (1–18, 505–530).
- `src/vae/loss.py` (611–640); `src/vae/build_vae.py` (228–347).
- `src/portfolio/entropy.py` (1–52); `src/portfolio/sca_solver.py` (605–644, 700–725); `src/portfolio/constraints.py` (111–112).
- `src/risk_model/rescaling.py` (52–111); `src/risk_model/factor_regression.py` (1–70).
- `notebooks/dashboard.ipynb` production config cells (lines 242–282, 330–369, 744–772).
- `results/diagnostic/config.json` (9, 33, 58, 68, 78–79, 84, 129–131); `diagnostic_data.json:1407`; `health_checks.csv` (1–11); `vae_fold_metrics.csv` (1–2); `strategy_comparison.csv` (1–3).

Not read: benchmark implementations, tests, training loop internals. Claims about benchmarks are limited to the design assertion at `src/config.py:517`.

## Five questions

1. **Purpose and original intent.** The unit is a *requirements specification*, not a model: "Before designing the model, we need to clarify what it must capture" (`docs/DVT_strategie_facteurs_risque_latents_v4_1.md:45`). It fixes five contracts downstream units must honour: (i) the objective is risk, not expected return, and `mu = 0` by default (`strategy_philosophy.md:9-13`); (ii) exposures are continuous and multi-membership, never categorical (`DVT:62-64`, `strategy_philosophy.md:29-40`); (iii) risk has a temporal dimension including dormant episodes (`DVT:66-78`, `strategy_philosophy.md:44-56`); (iv) risk is defined by co-movement, not by downside per se (`DVT:84-89`); (v) the risk model is modular and falsifiable, with an explicit abandon-VAE rule (`strategy_philosophy.md:154-168`, `strategy_philosophy.md:183`). The intent is to constrain the architecture before it is built.

2. **State of the art for this objective.** A modern version would be a *conditional* factor risk model: time-varying exposures `B_t` and covariance `Σ_t`, estimated with GARCH/DCC (Engle 2002), regime switching (Pelletier 2006), realized covariance (Barndorff-Nielsen & Shephard 2004), shrinkage (Ledoit & Wolf 2004/2020), and factor-count selection (Bai & Ng 2002), in the Barra/Axioma production tradition (Menchero, Orr & Wang 2011). Diversification would be implemented with a recognised risk-budgeting objective (Maillard, Roncalli & Teïletche 2010; Roncalli 2013; Meucci 2009 ENB; Choueifaty & Coignard 2008), and validated on covariance accuracy (QLIKE, Frobenius, Mincer–Zarnowitz) before any Sharpe comparison. If a graph representation is genuinely wanted, the SOTA object is a sparse inverse covariance (graphical lasso, Friedman, Hastie & Tibshirani 2008) or network connectedness (Diebold & Yılmaz 2014), not a dense bilinear factor model. Deep nonlinear exposure extraction exists mainly for *return* prediction (Gu, Kelly & Xiu 2021); deep risk models remain unproven, which sets the evidential bar higher, not lower.

3. **What prevents it, and what is only apparent.** Real blockers: covariance non-stationarity and estimation error; prices identify realized co-movement, not *ex ante* vulnerability (the document says so itself, `DVT:115-117`); long-only plus a fully invested budget forces a market exposure (`constraints.py:111-112`, `config.py:464`). Only apparent: "not enough data" — 7,874 days and 788 stocks are ample for a 26-factor model; "risk evolves slowly" is not needed for a conditional model to work, and is contradicted by the document's own episodic-factor argument (`DVT:74`); "the graph is needed" is rhetoric, since a low-rank covariance graph is already a graph. The evidenced blockers are implementation-level: memory is recency-weighted (`dashboard.ipynb:268`, `pipeline.py:1979`), the latent trajectory is collapsed to one vector (`rescaling.py:62`), the entropy target is inverted (`pipeline.py:2559-2562`), and the "prior-free" phase of training never runs at the selected checkpoint (`vae_fold_metrics.csv:2`).

4. **Real versus self-imposed constraints.** Real: PSD covariance; long-only and budget; daily price data only; estimation error (DeMiguel et al. 2009); crisis-time correlation spikes; 16 GB compute. Self-imposed: the 60-window half-life, the 252-day Σ_z EWMA, the 0.01-nat AU threshold, `w_max = 3%`, `w_min = 0.1%`, quarterly rebalancing, per-window z-scoring, `F = 2` with feature weights `[2.0, 0.5]`, the proportional entropy budget, gradient normalisation, and `K = 75`. Market-imposed but unstated: a long-only risk-only portfolio is a low-beta/low-volatility bet, so "no return view" is not "no directional exposure".

5. **What to remove, merge or move.** Remove: the half-life knob and its 21×-wrong documentation; the momentum path (disabled, mis-calibrated, contradicts the vision); the idiosyncratic entropy layer (it approximates 1/N and dilutes the factor signal, `entropy.py:12-16`); the "graph" and "K = 200" rhetoric; the claim of equal risk contributions if the runtime keeps proportional budgets. Merge: the three factor-count gates (KL threshold, statistical cap, optional Bai–Ng) into one documented rule. Move: the anti-cyclical requirement from exposure averaging to Σ_z conditioning, where it can be measured. Removable without losing a strength: the co-movement scaffold — dropping it makes the "no a priori bias" claim true, and the cost (training guidance) is a testable question for U06/U07.

## Load-bearing assumptions

| ID | Assumption | Where stated | Evidenced? | Comment |
|----|------------|--------------|------------|---------|
| A1 | Risk structure evolves slowly enough to be measured between retrainings | `docs/strategy_philosophy.md:11` | no | Contradicted by the same document's episodic claim (`DVT:74`) and by the runtime's own turn to recency weighting (`dashboard.ipynb:268`). This is the linchpin. |
| A2 | Averaging latent vectors preserves dormant exposures | `docs/strategy_philosophy.md:54`; `DVT:78` | no | A mean is a first moment: a brief strong exposure and a long weak one are indistinguishable. |
| A3 | One static `B` is a valid risk model over a 6-month OOS window | implied by `DVT:13`, `DVT:15` | partially | Σ_z is refreshed, `B` is not (`oos_rebalancing.py:528-530`); exposure drift is never tested. |
| A4 | Equal factor risk contributions is the right diversification target | `docs/strategy_philosophy.md:88-124` | partially | A literature heuristic (ERC), never shown to dominate min-variance or 1/N; the runtime uses proportional budgets instead (`dashboard.ipynb:361`). |
| A5 | Prices suffice for latent factor discovery | `DVT:134-143` | partially | They reveal realized co-movement, not pre-crisis vulnerabilities. |
| A6 | "No a priori bias" is achievable and achieved | `DVT:17`, `DVT:37` | no | λ_co scaffold still active at E* (see U01-ADV-03). |
| A7 | `abs(beta) ∝ 1/sqrt(lambda)` is a property of the portfolio | `docs/strategy_philosophy.md:113-116` | no | Holds only for the unconstrained Lagrangian, not under long-only + simplex + 3% cap. |
| A8 | AU is discovered by KL, not chosen | `docs/strategy_philosophy.md:75-82` | no | AU = 26 equals the statistical cap under the recorded EWMA (see U01-ADV-05). |
| A9 | The factor covariance is "far more stable" than asset-space estimation | `DVT:13` | no | No covariance-accuracy benchmark; the only real run fails its own variance-ratio check (`health_checks.csv:3`). |
| A10 | The portfolio is stable between retrainings | `DVT:15` | no | `refresh_risk_model_oos=True` re-estimates Σ_z and D_eps every 63 days (`config.py:650`, `pipeline.py:2776`). |

## Findings

| ID | Type | Severity | Evidence register | Claim (quoted, path:line) | Assessment | Evidence (path:line) | Cross-unit |
|----|------|----------|-------------------|---------------------------|------------|----------------------|------------|
| U01-ADV-01 | doc-vs-code, implementation | critical | doc+code | "the composite profile is a mean over all windows ... Crisis correlations persist permanently" (`docs/strategy_philosophy.md:56`) | The runtime sets `AGGREGATION_HALF_LIFE = 60` (`dashboard.ipynb:268`), and inference runs at stride 1 (`pipeline.py:1979`), so 60 windows = 60 trading days, not the documented ~5 years at stride 21 (`composite.py:136`, `dashboard.ipynb:269`). Weight left after 2 years is `0.5^(504/60) = 0.3%` (computed). The flagship anti-cyclical premise is inverted by a 21× unit error present in three places. | `dashboard.ipynb:268-270`; `pipeline.py:1976-1980`; `composite.py:132-165`; `dashboard.ipynb:749` | U04, U08, U15 |
| U01-ADV-02 | modeling, implementation | critical | code | "Temporality of risk factors ... dormant for extended periods, then suddenly dominant" (`DVT:68-74`) | Historical exposures are `B̃_{A,i,t} = R_{i,t} · μ̄_{A,i}`: one time-invariant profile times a scalar volatility ratio (`rescaling.py:62`). The VAE's per-window trajectory is averaged away (`composite.py:155-165`). Within a fold the model has no state; it cannot represent a factor switching from dormant to active, so "protected before the shock" is not representable. | `src/risk_model/rescaling.py:60-63`; `src/inference/composite.py:119-171`; `src/risk_model/factor_regression.py:10` | U04, U08, U09 |
| U01-ADV-03 | doc-vs-code, statistics | critical | doc+code+results | "no a priori bias in factor discovery" (`DVT:17`; `docs/strategy_philosophy.md:17`) | The co-movement loss supervises latent distances with Spearman correlation until 60% of epochs (`loss.py:621-640`). The selected checkpoint is E* = 164 of 500 (`vae_fold_metrics.csv:2`; `config.json:33`), so phase 3 (epoch ≥ 300) never ran; λ_co at E* is ≈0.45 with the recorded λ_max = 0.5, or ≈0.09 with the notebook's 0.1. The *trained model* is correlation-scaffolded even if the architecture is not. | `src/vae/loss.py:621-640`; `notebooks/dashboard.ipynb:249-250, 255`; `results/diagnostic/config.json:33`; `results/diagnostic/vae_fold_metrics.csv:1-2` | U04, U06, U07 |
| U01-ADV-04 | math, doc-vs-code | major | code | "abs(beta'_k) proportional to 1 / sqrt(lambda_k)" (`docs/strategy_philosophy.md:113-116`) | The current notebook sets `ENTROPY_BUDGET_MODE = "proportional"` (`dashboard.ipynb:361`), and the pipeline builds `budget = eigenvalues / sum(eigenvalues)` (`pipeline.py:2557-2562`). Tilted entropy pushes `c_k → b_k`, so `beta_k^2 * lambda_k ∝ lambda_k`, i.e. **equal betas**, the exact opposite of the inverse-root rule. Gradient normalisation (`sca_solver.py:633-634`, `dashboard.ipynb:355`) further rescales α each iteration, so the optimised objective is not the documented one. | `src/integration/pipeline.py:2557-2562`; `src/portfolio/entropy.py:45-52`; `src/portfolio/sca_solver.py:633-634`; `notebooks/dashboard.ipynb:355, 361` | U10, U11 |
| U01-ADV-05 | statistics, interpretation | major | code+results | "The number of risk factors is not chosen in advance ... effective factor count discovered from data" (`docs/strategy_philosophy.md:75-82`) | With the recorded `sigma_z_ewma_half_life = 252` and 7,874 days, the code's Kish formula gives N_eff = 727.1 and `AU_max = floor(sqrt(2·727.1/2)) = 26` (`active_units.py:100,114-137`). The reported AU is exactly 26 (`health_checks.csv:2`). The factor count is indistinguishable from a cap induced by a human-chosen half-life. Testable: rerun with EWMA = 0; the cap becomes 88. If AU stays 26 it was discovered; if it rises, the cap was binding. | `src/inference/active_units.py:92-137`; `results/diagnostic/config.json:68,131`; `results/diagnostic/health_checks.csv:2` | U08, U09, U15 |
| U01-ADV-06 | honesty, reproducibility | major | results | `"data_source": "synthetic"` (`results/diagnostic/config.json:9`) | The same file says `"data_source": "tiingo"` (line 129), and several values disagree with the current runtime: `sigma_z_ewma_half_life = 252` (line 68) vs the code default 0 (`config.py:458`); `w_bar = 0.03`, `phi = 0.0` (lines 78-79) vs the notebook's 0.015 and 5.0 (`dashboard.ipynb:334,336`). The run's parameter state cannot be reconstructed from the current code, so all causal readings of that run are conditional. | `results/diagnostic/config.json:9,68,78-79,129`; `notebooks/dashboard.ipynb:334,336`; `src/integration/notebook_helpers.py:161-164` | U15, U16, U17 |
| U01-ADV-07 | honesty, validation | major | doc+results | "if ... the VAE strategy does not statistically outperform the PCA factor risk parity baseline ... the VAE is abandoned" (`docs/strategy_philosophy.md:183`; `DVT:1192`) | The principle is stated but not implemented anywhere in `src` (no match for `abandon`), and the only real-data diagnostic is a single split: one fold, 1,575 OOS days (`vae_fold_metrics.csv:1-2`). The terminal principle cannot be triggered by the evidence that exists. | `docs/strategy_philosophy.md:183`; `results/diagnostic/vae_fold_metrics.csv:1-2` | U12, U14, U16 |
| U01-ADV-08 | interpretation, doc-vs-code | moderate | doc+code | "The true structure is closer to a graph" (`DVT:64`) | The implemented object is a dense bilinear factor model with continuous loadings; no adjacency, sparsity, multi-hop or nonlinear covariance exists (grep for graph/adjacency/GNN in `src` finds only the diagnostics DAG). The continuous-loading idea is implemented and sound; the graph claim adds no testable content. | `DVT:60-64`; `src/inference/composite.py:149-171`; `src/risk_model/rescaling.py:60-63` | U05, U09 |
| U01-ADV-09 | consistency, interpretation | moderate | doc+code | "what matters is the effective correlation of assets, regardless of its origin" (`DVT:89`) | The implemented risk is second-moment variance (`entropy.py:9-10`). Downside/CVaR and conditional-tail definitions are listed (`DVT:84-88`) then never used; crisis weighting (γ = 3) only reweights the training reconstruction loss (`config.py:237`, `dashboard.ipynb:249`), not the risk metric. Conditional: defensible for convex optimisation, but the document should not imply tail capture. | `src/portfolio/entropy.py:4-16`; `src/config.py:235-244`; `DVT:84-89` | U06, U09, U10 |
| U01-ADV-10 | math, interpretation | moderate | doc+code | "For a strong factor ... the portfolio has a small exposure. For a weak factor ... a large exposure" (`docs/strategy_philosophy.md:116`) | This is the unconstrained entropy-only first-order condition with total risk held fixed. Under long-only, fully invested weights (`constraints.py:111-112`), a 3% cap (`dashboard.ipynb:332`) and a unit market column (`config.py:464`), the market exposure cannot be driven to 4.5 against 31.6. The table is not a property of the shipped portfolio. | `src/portfolio/constraints.py:111-112`; `notebooks/dashboard.ipynb:332`; `src/config.py:439-464` | U10, U11 |
| U01-ADV-11 | doc-vs-code | moderate | doc+code | "the factor covariance is estimated on full history" (`docs/strategy_philosophy.md:56`); "between retrainings, the portfolio is stable by design" (`DVT:15`) | `refresh_risk_model_oos = True` (`config.py:650`) re-estimates Σ_z, D_eps and eigenvalues at every 63-day rebalance on an expanding window (`oos_rebalancing.py:11-15, 525-530`; `pipeline.py:2776`), and the recorded run used a 252-day EWMA (`config.json:68`). Neither B nor Σ_z carries permanent memory; the stated mechanism is not the shipped one. | `src/config.py:650`; `src/integration/pipeline.py:2776`; `src/walk_forward/oos_rebalancing.py:11-15, 525-530` | U09, U12 |
| U01-ADV-12 | interpretation, economics | moderate | code+results | "the strategy produces no return forecast by default (mu = 0)" (`docs/strategy_philosophy.md:13`) | True in code (`pipeline.py:2492-2501`, `sca_solver.py:704,725`), but not return-neutral in economics: the diagnostic VAE returned 6.3% with 16.4% vol versus 9.7% and 19.7% for equal weight (`strategy_comparison.csv:2-3`), the signature of a low-beta/low-vol tilt (Frazzini & Pedersen 2014). The document acknowledges the effect only later, via Clarke, de Silva & Thorley (2013) at `DVT:1133`. | `src/integration/pipeline.py:2491-2501`; `src/portfolio/sca_solver.py:704,725`; `results/diagnostic/strategy_comparison.csv:2-3` | U10, U14 |
| U01-ADV-13 | statistics, results | moderate | doc+results | "enables a factor-based covariance estimation that is far more stable than direct estimation in asset space" (`DVT:13`) | No covariance-accuracy benchmark exists in the repo, and the only real run fails the project's own check: `var_ratio = 2.742`, outside [0.5, 2.0] (`health_checks.csv:3`; `vae_fold_metrics.csv:2`). Stability is untested; calibration is demonstrably wrong by ~2.7×. | `results/diagnostic/health_checks.csv:3`; `results/diagnostic/vae_fold_metrics.csv:1-2`; `DVT:13` | U09, U14 |
| U01-ADV-14 | implementation, logic | minor | code | "Each stock's windows are assumed to be sorted chronologically" (`src/inference/composite.py:132-133`) | The grouping uses `np.argsort(stock_ids_arr)` with the default (non-stable) kind (`composite.py:102`). With `half_life > 0` (`dashboard.ipynb:268`), a permutation within a stock's block would assign recency weights to the wrong windows. Testable: compare profiles with `kind="stable"`; equality confirms no permutation for this input. | `src/inference/composite.py:102,132-162` | U04, U08 |
| U01-ADV-15 | implementation, calibration | minor | code | "With lambda_risk=252 ... momentum_weight=0.02 gives a return term ~0.02" (`src/config.py:563-565`) | The actual default is `momentum_weight = 0.30` (`config.py:611`), 15× the documented calibration, while validation suggests 0.02 (`config.py:694-695`). It is harmless only because the notebook disables it and sets 0.0 (`dashboard.ipynb:366,369`). The "removable layer" is safe by switch, not by design. | `src/config.py:563-565,608,611,694-695`; `notebooks/dashboard.ipynb:366,369` | U10, U15 |
| U01-ADV-16 | consistency | minor | doc+code+results | "K is a capacity ceiling (default: 200)" (`docs/strategy_philosophy.md:77`) | K is 50 in the code default (`config.py:179`), 75 in the notebook (`dashboard.ipynb:247`) and the run config (`config.json`), and 52 in the run's health check (`health_checks.csv:2`). The capacity ceiling is not a stable documented quantity, and the doc's 200 is never used. | `docs/strategy_philosophy.md:77`; `src/config.py:179`; `notebooks/dashboard.ipynb:247`; `results/diagnostic/health_checks.csv:2` | U05, U15 |
| U01-ADV-17 | implementation, honesty | strength | code | "the strategy produces no return forecast by default (mu = 0)" (`docs/strategy_philosophy.md:13`) | Verified guarantee: `mu` is `None` unless `momentum_enabled` (`pipeline.py:2492-2501`), and the return layer is exactly one additive term in the objective (`sca_solver.py:704,725`). Removing the layer is therefore a genuine code deletion, not a reconfiguration. This is the unit's strongest verified property. | `src/integration/pipeline.py:2491-2501`; `src/portfolio/sca_solver.py:700-725`; `src/config.py:608` | U10, U11 |
| U01-ADV-18 | modeling | strength | doc+code | "multiple simultaneous memberships with varying intensities — which is what a continuous latent space provides" (`DVT:64`) | Implemented as a continuous `B` matrix of shape (n, K), with no categorical assignment anywhere in inference (`composite.py:149-171`). This directly targets the failure mode of sector/geography diversification and is the minimal sufficient object for factor risk budgeting. | `src/inference/composite.py:119-171`; `DVT:62-64`; `docs/strategy_philosophy.md:29-40` | U05, U08 |
| U01-ADV-19 | validation protocol | strength | code | "Expanding training window ... Embargo gap (21 days) ... Final holdout" (`docs/strategy_philosophy.md:174-181`) | Implemented: `train_start` is fixed at the data start, with nested validation, a 21-business-day embargo, a separate holdout, and explicit no-lookahead assertions (`folds.py:53-70, 100-119`). Constraints are asserted identical for VAE and benchmarks (`config.py:517`). This protocol is the project's main epistemic asset. | `src/walk_forward/folds.py:15-121`; `src/config.py:512-517`; `src/validation.py` (referenced at `folds.py:12`) | U12, U13 |
| U01-ADV-20 | math | strength | code | "WARNING: Must be computed in the PRINCIPAL factor basis ... where contributions can be negative" (`src/portfolio/entropy.py:25-26`) | Correct and necessary: contributions are `c_k = beta_k^2 * lambda_k` in the diagonalised basis (`entropy.py:9-10`), guaranteeing non-negative arguments to `log` and a well-defined entropy. This is a genuine mathematical guarantee that a redesign must keep. | `src/portfolio/entropy.py:9-10,25-26,45-52` | U10 |

## Core strengths to preserve

1. **The `mu = 0` separation, implemented as exactly one additive term.** It makes the risk model independently testable and prevents return signals from contaminating factor discovery. It is verified in code, not just in prose (`pipeline.py:2491-2501`; `sca_solver.py:704,725`). Any redesign must keep the return layer removable in one line.
2. **The continuous exposure matrix with no forced membership.** `B` (n × K) is the minimal object that expresses overlapping risk memberships; it is what makes factor risk budgeting possible (`composite.py:149-171`). It must not regress to sector buckets.
3. **Time as a first-class risk dimension.** The *requirement* is right even though the *mechanism* (averaging exposures) is wrong. A redesign should move it to a conditional covariance, not delete it.
4. **The walk-forward protocol: expanding window, 21-day embargo, untouched holdout, six benchmarks, identical constraints.** This is the only reason the project can be judged honestly (`folds.py:15-121`; `config.py:517`).
5. **The mathematical hygiene of the entropy layer.** Computing contributions in the principal basis (`entropy.py:25-26`) and using non-negative contributions is a correctness guarantee; the Ledoit–Wolf/DGJ shrinkage, WLS regression and Barra-style z-scoring are production-standard remedies for estimation error that should survive any redesign.
6. **The stated abandonment clause.** Even though unimplemented, the commitment that complexity must earn its place (`strategy_philosophy.md:183`) is the correct governing principle for the synthesis.

## Verdict

The vision is epistemically sound at its core and unusually honest in its validation design, but four of its five distinctive claims are not delivered by the code. The memory/anti-cyclical premise is inverted by a 21× unit error (a 60-day, not 5-year, half-life); temporality is collapsed to one time-invariant exposure vector scaled by volatility; the "no a priori bias" claim is false for the selected checkpoint because the co-movement scaffold never annealed before E*; and the flagship factor-risk-parity is replaced at runtime by eigenvalue-proportional budgets plus gradient normalisation. The one claim that survives intact is the return/risk separation, which is genuinely and verifiably implemented. As advocate, I can defend the direction, the continuous-exposure representation, the entropy mathematics and the validation protocol; I cannot defend the mechanisms that carry the unit's distinctive identity. The correct redesign move is not to patch these mechanisms but to relocate the temporal requirement to the covariance estimator and delete the claims the code does not honour.

Score: 5/10

## Machine companion

```json
[
  {
    "id": "U01-ADV-01",
    "unit": "U01",
    "role": "advocate",
    "type": "doc-vs-code",
    "severity": "critical",
    "evidence_register": "doc+code",
    "claim": "the composite profile is a mean over all windows; crisis correlations persist permanently",
    "claim_location": "docs/strategy_philosophy.md:56",
    "assessment": "Runtime uses a 60-window half-life; inference runs at stride 1, so the half-life is 60 trading days, not the documented ~5 years at stride 21. Only 0.3% of weight survives two years. The anti-cyclical premise is inverted by a 21x unit error repeated in three places.",
    "evidence": ["notebooks/dashboard.ipynb:268-270", "notebooks/dashboard.ipynb:749", "src/integration/pipeline.py:1976-1980", "src/inference/composite.py:132-165"],
    "cross_units": ["U04", "U08", "U15"],
    "root_cause": "The half-life is expressed in windows while training and inference use different strides; no unit test ties the parameter to calendar time.",
    "confidence": "high"
  },
  {
    "id": "U01-ADV-02",
    "unit": "U01",
    "role": "advocate",
    "type": "modeling",
    "severity": "critical",
    "evidence_register": "code",
    "claim": "risk factors are dormant for extended periods then suddenly dominant",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:68-74",
    "assessment": "Historical exposures are a single time-invariant profile times a scalar volatility ratio. The per-window latent trajectory is averaged away. Within a fold there is no state variable, so dormant-to-active transitions cannot be represented and protection-before-the-shock is unrepresentable.",
    "evidence": ["src/risk_model/rescaling.py:60-63", "src/inference/composite.py:119-171", "src/risk_model/factor_regression.py:10"],
    "cross_units": ["U04", "U08", "U09"],
    "root_cause": "The aggregation step discards the trajectory; the risk model was designed for a static loading matrix.",
    "confidence": "high"
  },
  {
    "id": "U01-ADV-03",
    "unit": "U01",
    "role": "advocate",
    "type": "doc-vs-code",
    "severity": "critical",
    "evidence_register": "doc+code+results",
    "claim": "no a priori bias in factor discovery",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:17",
    "assessment": "The co-movement loss supervises latent distances with Spearman correlation until 60% of epochs. E* is 164 of 500, so phase 3 never ran; lambda_co at E* is about 0.45 with the recorded 0.5 maximum, or 0.09 with the notebook's 0.1. The trained checkpoint is correlation-scaffolded even if the architecture is not.",
    "evidence": ["src/vae/loss.py:621-640", "notebooks/dashboard.ipynb:249-250,255", "results/diagnostic/config.json:33", "results/diagnostic/vae_fold_metrics.csv:1-2"],
    "cross_units": ["U04", "U06", "U07"],
    "root_cause": "Early stopping selects a validation-ELBO optimum before the curriculum reaches its prior-free phase; no constraint ties E* to the end of the scaffold.",
    "confidence": "high"
  },
  {
    "id": "U01-ADV-04",
    "unit": "U01",
    "role": "advocate",
    "type": "math",
    "severity": "major",
    "evidence_register": "code",
    "claim": "the portfolio exposure to each factor is inversely proportional to the square root of the factor variance",
    "claim_location": "docs/strategy_philosophy.md:113-116",
    "assessment": "The runtime sets a proportional entropy budget b = eigenvalues / sum(eigenvalues). Tilted entropy pushes contributions toward b, implying equal betas across factors, the exact opposite of the inverse-root rule. Gradient normalisation additionally rescales alpha each iteration, so the optimised objective is not the documented objective.",
    "evidence": ["src/integration/pipeline.py:2557-2562", "src/portfolio/entropy.py:45-52", "src/portfolio/sca_solver.py:633-634", "notebooks/dashboard.ipynb:355,361"],
    "cross_units": ["U10", "U11"],
    "root_cause": "The objective evolved through solver-balancing fixes without reconciling the specification, so the name factor risk parity no longer describes the optimisation.",
    "confidence": "high"
  },
  {
    "id": "U01-ADV-05",
    "unit": "U01",
    "role": "advocate",
    "type": "statistics",
    "severity": "major",
    "evidence_register": "code+results",
    "claim": "the number of risk factors is not chosen in advance; the effective factor count is discovered from the data",
    "claim_location": "docs/strategy_philosophy.md:75-82",
    "assessment": "With the recorded 252-day EWMA and 7,874 days, the implemented Kish formula gives N_eff = 727.1 and AU_max = 26. The reported AU is exactly 26, so AU is indistinguishable from a cap induced by a human-chosen half-life. Testable: rerun with EWMA = 0, which raises the cap to 88.",
    "evidence": ["src/inference/active_units.py:92-137", "results/diagnostic/config.json:68,131", "results/diagnostic/health_checks.csv:2"],
    "cross_units": ["U08", "U09", "U15"],
    "root_cause": "Three independent gates (KL threshold, statistical cap, optional Bai-Ng) determine the factor count, and the effective gate is not reported.",
    "confidence": "medium"
  },
  {
    "id": "U01-ADV-06",
    "unit": "U01",
    "role": "advocate",
    "type": "honesty",
    "severity": "major",
    "evidence_register": "results",
    "claim": "the run records data_source = synthetic while the diagnostic block records tiingo",
    "claim_location": "results/diagnostic/config.json:9",
    "assessment": "The recorded parameter state is self-contradictory and also disagrees with the current runtime: sigma_z_ewma_half_life 252 versus the code default 0, and w_bar/phi 0.03/0.0 versus the notebook's 0.015/5.0. The run cannot be reconstructed from current code, so causal readings of its results are conditional.",
    "evidence": ["results/diagnostic/config.json:9,68,78-79,129", "notebooks/dashboard.ipynb:334,336", "src/integration/notebook_helpers.py:161-164", "src/config.py:458"],
    "cross_units": ["U15", "U16", "U17"],
    "root_cause": "Configuration is assembled in several code paths with divergent defaults, and exports are not validated against the executing code.",
    "confidence": "high"
  },
  {
    "id": "U01-ADV-07",
    "unit": "U01",
    "role": "advocate",
    "type": "honesty",
    "severity": "major",
    "evidence_register": "doc+results",
    "claim": "if the VAE does not statistically outperform the PCA factor risk parity baseline the VAE is abandoned",
    "claim_location": "docs/strategy_philosophy.md:183",
    "assessment": "The rule is stated but not implemented anywhere in src, and the only real-data diagnostic is a single split with one fold and 1,575 OOS days. The terminal principle cannot be triggered by the evidence that exists.",
    "evidence": ["docs/strategy_philosophy.md:183", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:1192", "results/diagnostic/vae_fold_metrics.csv:1-2"],
    "cross_units": ["U12", "U14", "U16"],
    "root_cause": "The decision principle lives in prose; the report generator has no corresponding gate.",
    "confidence": "high"
  },
  {
    "id": "U01-ADV-08",
    "unit": "U01",
    "role": "advocate",
    "type": "interpretation",
    "severity": "moderate",
    "evidence_register": "doc+code",
    "claim": "the true structure is closer to a graph",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:64",
    "assessment": "The implemented object is a dense bilinear factor model with continuous loadings; there is no adjacency, sparsity, multi-hop or nonlinear covariance. The continuous-loading representation is sound and implemented; the graph language adds no testable content.",
    "evidence": ["docs/DVT_strategie_facteurs_risque_latents_v4_1.md:60-64", "src/inference/composite.py:149-171", "src/risk_model/rescaling.py:60-63"],
    "cross_units": ["U05", "U09"],
    "root_cause": "Vision text uses a stronger structural metaphor than the model implements.",
    "confidence": "high"
  },
  {
    "id": "U01-ADV-09",
    "unit": "U01",
    "role": "advocate",
    "type": "consistency",
    "severity": "moderate",
    "evidence_register": "doc+code",
    "claim": "what matters is the effective correlation of assets, regardless of its origin",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:89",
    "assessment": "The implemented risk is second-moment variance. Downside, CVaR and conditional-tail definitions are listed then unused; crisis weighting only reweights the training reconstruction loss, not any risk metric. Defensible for convex optimisation, but the document should not imply tail capture.",
    "evidence": ["src/portfolio/entropy.py:4-16", "src/config.py:235-244", "notebooks/dashboard.ipynb:249", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:84-89"],
    "cross_units": ["U06", "U09", "U10"],
    "root_cause": "Tail risk was introduced in the training loss and never propagated to the risk measure.",
    "confidence": "high"
  },
  {
    "id": "U01-ADV-10",
    "unit": "U01",
    "role": "advocate",
    "type": "math",
    "severity": "moderate",
    "evidence_register": "doc+code",
    "claim": "for a strong factor the portfolio has a small exposure, for a weak factor a large exposure",
    "claim_location": "docs/strategy_philosophy.md:116",
    "assessment": "This is the unconstrained entropy-only first-order condition with total risk held fixed. Under long-only, fully invested weights, a 3% cap and a unit market column, the market exposure cannot be driven to the documented ratio. The table is not a property of the shipped portfolio.",
    "evidence": ["src/portfolio/constraints.py:111-112", "notebooks/dashboard.ipynb:332", "src/config.py:439-464"],
    "cross_units": ["U10", "U11"],
    "root_cause": "The specification derives an unconstrained optimum and presents it as a portfolio property.",
    "confidence": "high"
  },
  {
    "id": "U01-ADV-11",
    "unit": "U01",
    "role": "advocate",
    "type": "doc-vs-code",
    "severity": "moderate",
    "evidence_register": "doc+code",
    "claim": "the factor covariance is estimated on full history; between retrainings the portfolio is stable by design",
    "claim_location": "docs/strategy_philosophy.md:56",
    "assessment": "refresh_risk_model_oos is enabled and re-estimates Sigma_z, D_eps and eigenvalues at every 63-day rebalance on an expanding window, with a 252-day EWMA in the recorded run. Neither the exposures nor the covariance carries permanent memory; the stated mechanism is not the shipped one.",
    "evidence": ["src/config.py:650", "src/integration/pipeline.py:2776", "src/walk_forward/oos_rebalancing.py:11-15,525-530", "results/diagnostic/config.json:68"],
    "cross_units": ["U09", "U12"],
    "root_cause": "The anti-cyclical design was overridden for evaluation symmetry with the benchmarks without updating the philosophy document.",
    "confidence": "high"
  },
  {
    "id": "U01-ADV-12",
    "unit": "U01",
    "role": "advocate",
    "type": "interpretation",
    "severity": "moderate",
    "evidence_register": "code+results",
    "claim": "the strategy produces no return forecast by default (mu = 0)",
    "claim_location": "docs/strategy_philosophy.md:13",
    "assessment": "True in code but not return-neutral in economics: the VAE produced 6.3% return and 16.4% volatility versus 9.7% and 19.7% for equal weight, the signature of a low-beta low-volatility tilt. The document acknowledges the effect only in the iteration section via Clarke, de Silva and Thorley.",
    "evidence": ["src/integration/pipeline.py:2491-2501", "src/portfolio/sca_solver.py:704,725", "results/diagnostic/strategy_comparison.csv:2-3", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:1133"],
    "cross_units": ["U10", "U14"],
    "root_cause": "The vision equates no return forecast with no directional exposure, which is false under a long-only budget.",
    "confidence": "medium"
  },
  {
    "id": "U01-ADV-13",
    "unit": "U01",
    "role": "advocate",
    "type": "statistics",
    "severity": "moderate",
    "evidence_register": "doc+results",
    "claim": "the exposure matrix enables covariance estimation far more stable than direct estimation in asset space",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:13",
    "assessment": "No covariance-accuracy benchmark exists in the repository, and the only real run fails the project's own variance-ratio check at 2.742 versus a [0.5, 2.0] band. Stability is untested; calibration is wrong by roughly 2.7x.",
    "evidence": ["results/diagnostic/health_checks.csv:3", "results/diagnostic/vae_fold_metrics.csv:1-2", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:13"],
    "cross_units": ["U09", "U14"],
    "root_cause": "The project validates portfolio outcomes, not covariance forecasts, so a miscalibrated risk model can persist without blocking progress.",
    "confidence": "high"
  },
  {
    "id": "U01-ADV-14",
    "unit": "U01",
    "role": "advocate",
    "type": "implementation",
    "severity": "minor",
    "evidence_register": "code",
    "claim": "each stock windows are assumed to be sorted chronologically",
    "claim_location": "src/inference/composite.py:132-133",
    "assessment": "The per-stock grouping uses np.argsort with the default non-stable kind. With half_life greater than zero, a permutation within a stock block would assign recency weights to the wrong windows. Testable by comparing profiles against kind=stable.",
    "evidence": ["src/inference/composite.py:102,132-162", "notebooks/dashboard.ipynb:268"],
    "cross_units": ["U04", "U08"],
    "root_cause": "Chronological order is an unchecked precondition of the decay weighting.",
    "confidence": "medium"
  },
  {
    "id": "U01-ADV-15",
    "unit": "U01",
    "role": "advocate",
    "type": "calibration",
    "severity": "minor",
    "evidence_register": "code",
    "claim": "with lambda_risk 252, momentum_weight 0.02 gives a return term of about 0.02",
    "claim_location": "src/config.py:563-565",
    "assessment": "The actual default momentum_weight is 0.30, fifteen times the documented calibration, while validation suggests 0.02. It is harmless only because the notebook disables momentum and sets the weight to 0.0. The removable return layer is safe by switch, not by design.",
    "evidence": ["src/config.py:563-565,608,611,694-695", "notebooks/dashboard.ipynb:366,369"],
    "cross_units": ["U10", "U15"],
    "root_cause": "The default drifted from the documented calibration and no test binds the weight to the risk-term scale.",
    "confidence": "high"
  },
  {
    "id": "U01-ADV-16",
    "unit": "U01",
    "role": "advocate",
    "type": "consistency",
    "severity": "minor",
    "evidence_register": "doc+code+results",
    "claim": "K is a capacity ceiling (default 200)",
    "claim_location": "docs/strategy_philosophy.md:77",
    "assessment": "K is 50 in the code default, 75 in the notebook and run config, and 52 in the run health check. The capacity ceiling is not a stable documented quantity, and the documented 200 is never used.",
    "evidence": ["docs/strategy_philosophy.md:77", "src/config.py:179", "notebooks/dashboard.ipynb:247", "results/diagnostic/health_checks.csv:2"],
    "cross_units": ["U05", "U15"],
    "root_cause": "No single source of truth for the architecture parameter.",
    "confidence": "high"
  },
  {
    "id": "U01-ADV-17",
    "unit": "U01",
    "role": "advocate",
    "type": "implementation",
    "severity": "strength",
    "evidence_register": "code",
    "claim": "the strategy produces no return forecast by default (mu = 0)",
    "claim_location": "docs/strategy_philosophy.md:13",
    "assessment": "Verified guarantee: mu is None unless momentum is enabled, and the return layer is exactly one additive term in the objective. Removing the layer is a genuine deletion, not a reconfiguration. This is the unit's strongest verified property.",
    "evidence": ["src/integration/pipeline.py:2491-2501", "src/portfolio/sca_solver.py:700-725", "src/config.py:608"],
    "cross_units": ["U10", "U11"],
    "root_cause": null,
    "confidence": "high"
  },
  {
    "id": "U01-ADV-18",
    "unit": "U01",
    "role": "advocate",
    "type": "modeling",
    "severity": "strength",
    "evidence_register": "doc+code",
    "claim": "multiple simultaneous memberships with varying intensities, which is what a continuous latent space provides",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:64",
    "assessment": "Implemented as a continuous B matrix of shape n by K with no categorical assignment in inference. It directly targets the failure mode of sector and geography diversification and is the minimal sufficient object for factor risk budgeting.",
    "evidence": ["src/inference/composite.py:119-171", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:62-64", "docs/strategy_philosophy.md:29-40"],
    "cross_units": ["U05", "U08"],
    "root_cause": null,
    "confidence": "high"
  },
  {
    "id": "U01-ADV-19",
    "unit": "U01",
    "role": "advocate",
    "type": "validation",
    "severity": "strength",
    "evidence_register": "code",
    "claim": "expanding training window, embargo gap of 21 days, final holdout",
    "claim_location": "docs/strategy_philosophy.md:174-181",
    "assessment": "Implemented with a fixed data start, nested validation, a 21-business-day embargo, a separate holdout and explicit no-lookahead assertions. Constraints are asserted identical for VAE and benchmarks. This protocol is the project's main epistemic asset.",
    "evidence": ["src/walk_forward/folds.py:15-121", "src/config.py:512-517"],
    "cross_units": ["U12", "U13"],
    "root_cause": null,
    "confidence": "high"
  },
  {
    "id": "U01-ADV-20",
    "unit": "U01",
    "role": "advocate",
    "type": "math",
    "severity": "strength",
    "evidence_register": "code",
    "claim": "entropy must be computed in the principal factor basis, where contributions can be negative otherwise",
    "claim_location": "src/portfolio/entropy.py:25-26",
    "assessment": "Correct and necessary: contributions are beta squared times lambda in the diagonalised basis, guaranteeing non-negative arguments to log and a well-defined entropy. A redesign must preserve this guarantee.",
    "evidence": ["src/portfolio/entropy.py:9-10,25-26,45-52"],
    "cross_units": ["U10"],
    "root_cause": null,
    "confidence": "high"
  }
]
```

END-OF-REPORT
