# Implementation Decisions Log

> **Purpose:** Document all implementation decisions made when the ISD/DVT specifications were insufficient or ambiguous.

## Protocol

### When to Log

Log an entry when:
- A specification is **ambiguous** or **incomplete**
- Multiple valid interpretations exist and you chose one
- You had to **infer** behavior not explicitly specified
- You discovered an **edge case** not covered by the spec
- You made a **conservative assumption** to proceed

### Entry Format

Each entry MUST follow this format:

```markdown
### [MOD-XXX] Title of the decision

**Date:** YYYY-MM-DD
**Agent/Author:** (agent name or human)
**Spec Reference:** (ISD section, DVT section, or "none")

**Context:** (What were you trying to implement?)

**Gap:** (What was missing or ambiguous in the specification?)

**Decision:** (What did you decide to do?)

**Rationale:** (Why this choice over alternatives?)

**Alternatives Considered:** (Other options you rejected)

**Impact:** (Files/modules affected, potential downstream effects)

**Status:** `provisional` | `validated` | `superseded`
```

### Responsibility by Phase

| Phase | Responsibility | Protocol |
|-------|----------------|----------|
| Phase 1 (parallel) | Each teammate (data-engineer, ml-architect, test-lead) logs their own decisions with MOD-XXX prefix | Atomic entries, no coordination needed |
| Phase 2 (sequential) | Lead session or subagent logs decisions | Sequential, no conflict |
| Phase 3 (parallel) | Each benchmark teammate logs with MOD-0XX prefix | Atomic entries, no coordination needed |
| Phase 4 (sequential) | Lead session logs and **consolidates** all entries | Review `provisional` entries, validate or flag |

### Validation at Phase Transitions

Before transitioning to the next phase:
1. **Review all `provisional` entries** from the completed phase
2. **Validate** decisions that are correct (`provisional` → `validated`)
3. **Flag** decisions that need discussion with human supervisor
4. **Supersede** decisions that were revised (`provisional` → `superseded` + new entry)

---

## Decisions Log

<!-- Entries go below, newest first within each module -->

### Phase 1 — Infrastructure

#### [MOD-001] VIX data source for crisis labeling

**Date:** 2026-02-06
**Agent/Author:** human + lead session
**Spec Reference:** ISD MOD-001 Sub-task 5 (crisis.py), DVT Section 4.4

**Context:** Crisis labeling requires VIX historical data (daily close, 1990–present). The ISD specifies the crisis mechanism (VIX threshold at P80 on expanding training window) but not the data source.

**Gap:** No concrete VIX data source specified. Need a free, reliable, programmatic source with history from 1990.

**Decision:** Use **FRED (Federal Reserve Economic Data)** as primary source via the `fredapi` Python package. Series ID: `VIXCLS`. Fallback: direct CSV download from CBOE website.

- Primary: `fredapi` package → `fred.get_series('VIXCLS', start_date, end_date)`
- Fallback: CBOE CSV from https://www.cboe.com/tradable-products/vix/vix-historical-data/
- Pre-1990 proxy: VXO (1986–1990) from FRED series `VXOCLS`, or realized vol proxy (annualized 21-day rolling std of S&P 500 returns, percentile-matched to VIX over common history 1990–2003)

**Rationale:** FRED provides free API access with no rate limit concerns, has the full VIX history from 1990-01-02, is maintained by the Federal Reserve (institutional reliability), and the `fredapi` package is well-maintained and lightweight. CBOE direct download requires manual intervention or scraping. Yahoo Finance (`^VIX` via `yfinance`) is an alternative but has had reliability issues with historical data gaps.

**Alternatives Considered:**
- `yfinance` (`^VIX`): unreliable for pre-2000 data, occasional API changes
- `vix-utils` PyPI package: overkill (designed for futures term structure)
- CBOE direct CSV: no programmatic API, requires manual download

**Impact:** `src/data_pipeline/crisis.py`, `tests/fixtures/synthetic_data.py` (VIX mock), `pyproject.toml` (add `fredapi` dependency). Requires a free FRED API key (obtainable at https://fred.stlouisfed.org/docs/api/api_key.html).

**Status:** `provisional`

---

#### [MOD-001] Volume column promoted to core schema for ADV filtering

**Date:** 2026-02-06
**Agent/Author:** human + lead session
**Spec Reference:** ISD MOD-001 Sub-task 1 (data_loader.py) data schema, DVT Section 5.1 (Universe table), DVT Section 4.2

**Context:** The universe construction requires an ADV (Average Daily Dollar Volume) filter: ADV ≥ $2M over trailing 63 trading days, where ADV = mean(adj_price × volume). This was defined in the DVT Section 5.1 Universe parameter table and ISD `construct_universe` function.

**Gap:** The ISD data schema classified `volume` as an extended-only column (F > 2), available only when volume is used as a VAE input feature. However, the F=2 baseline pipeline still needs `volume` to compute ADV for the universe filter — creating an impossible dependency: the universe filter needs data that the schema doesn't provide.

**Decision:** Promote `volume` to a **core column** (always loaded), with a clear distinction:
- **F=2 baseline:** `volume` is an **infrastructure column** for the ADV liquidity filter in universe construction. It is NOT fed to the VAE encoder.
- **F > 2 extended:** `volume` is additionally z-scored and used as a VAE input feature.

Updated ISD sections: core columns table, extended columns table, EODHD column mapping, synthetic data generator spec, loader function signature, `construct_universe` docstring.

**Rationale:** The DVT Section 5.1 Universe table lists ADV as a fundamental filter parameter alongside market cap and listing history. Universe construction is upstream of feature selection — the pipeline must know which stocks are in the universe before deciding what features to feed the VAE. Separating "infrastructure columns" (needed for pipeline logic) from "feature columns" (fed to the model) resolves the dependency cleanly.

**Alternatives Considered:**
- Hardcode a fixed universe without ADV filter (rejected — violates DVT Section 4.2 liquidity requirement)
- Load volume via a separate data path outside the main schema (rejected — unnecessary complexity, duplicates I/O)
- Defer ADV filter to F > 2 only (rejected — the DVT mandates ADV filtering for all configurations)

**Impact:** `docs/ISD_vae_latent_risk_factors.md` (6 sections updated), `src/data_pipeline/data_loader.py` (must always load volume), `tests/fixtures/synthetic_data.py` (must always generate volume). EODHD provides Volume — no data gap. No changes needed to DVT.

**Status:** `validated`

---

#### [MOD-001] EODHD production data — deferred

**Date:** 2026-02-06
**Agent/Author:** human + lead session
**Spec Reference:** ISD MOD-001 Sub-task 1 (data_loader.py), DVT Section 4.2

**Context:** Phase B of the data loading strategy uses EODHD (End-Of-Day Historical Data) as the production data source for US equity prices (26,000+ stocks, 2000–present).

**Gap:** The following EODHD-specific details are NOT yet documented:
- API endpoint URLs and authentication method (API key vs bulk download)
- Rate limits and pagination strategy
- Bulk download format (CSV columns, encoding, delimiters)
- Shares outstanding source for market cap computation (fundamentals API endpoint)
- Security type metadata for filtering common equity (ETF/ADR/REIT exclusion)
- Ticker-to-integer permno mapping strategy
- Delisting detection method (last available date vs explicit delisting flag)
- Cost: "All-In-One" plan (~100 EUR/month), subscription management

**Decision:** Defer EODHD integration to after Phase 1 is complete and the full pipeline is validated on synthetic data. Phase A (synthetic CSV) is sufficient for all development and testing. EODHD-specific implementation will be documented here when addressed.

**Rationale:** The ISD explicitly defines a two-phase data strategy (synthetic first, EODHD second). All pipeline logic, invariants, and tests can be validated on synthetic data. Integrating EODHD now would block development on an external dependency.

**Alternatives Considered:** None — this is the documented strategy.

**Impact:** `src/data_pipeline/data_loader.py` will need an EODHD-specific loader added later. All other modules are unaffected (they consume the same normalized schema).

**Status:** `provisional`

---

#### [MOD-009] Hyperparameter grid for Phase A walk-forward selection

**Date:** 2026-02-06
**Agent/Author:** lead session
**Spec Reference:** ISD MOD-009 Sub-task 2 (phase_a.py), DVT Section 4.8 + Section 5.1

**Context:** Phase A of walk-forward validation requires a grid of HP configurations to evaluate per fold. The ISD specifies `hp_configs: list[dict]` but does not enumerate the grid. The DVT (Section 5.1) provides baselines and ranges for all parameters.

**Gap:** No explicit HP grid defined. The DVT distinguishes "structural HPs" (require full retraining) from "training HPs" (selected within each fold), but does not specify exact grid values.

**Decision:** Define a two-level HP grid:

**Level 1 — Structural HPs (coarse grid, full walk-forward):**

| Parameter | Values | Justification |
|-----------|--------|---------------|
| K | {100, 200} | DVT baseline 200, range 50–300. 100 tests reduced capacity. |
| Loss mode | {P, F} | Mode P is primary, Mode F is fallback. Mode A deferred (DVT: only if P unstable). |

→ 2 × 2 = **4 structural configs**

**Level 2 — Training HPs (nested within each fold, per structural config):**

| Parameter | Values | Justification |
|-----------|--------|---------------|
| η₀ (learning rate) | {5e-4, 1e-3} | DVT range 5e-5 to 1e-3, ISD default 1e-3. Log-scale. |
| γ (crisis weight) | {2.0, 3.0} | DVT baseline 3.0, range 1.5–5.0. 2.0 tests lighter weighting. |
| λ_co_max | {0.3, 0.5} | DVT baseline 0.5, range 0.1–2.0. |

→ 2 × 2 × 2 = **8 training configs per structural config**

**Total: 4 × 8 = 32 configs per fold.** At ~34 folds, this yields ~1088 training runs for Phase A + 34 runs for Phase B = ~1122 total. With early stopping (typically 30–60 epochs), this is computationally feasible.

**Elimination before scoring (per ISD):**
- AU < max(0.15K, AU_PCA)
- Explanatory power < max(0.40, EP_PCA)
- OOS/train MSE > 3.0

**Rationale:** The grid is deliberately compact (32 configs) to stay computationally tractable while covering the most impactful parameters. Parameters not in the grid are fixed at DVT baselines: T=504 (resolved), F=2 (resolved), batch_size=128 (ISD default), optimizer=Adam (resolved), patience=10 (ISD default). Mode A is excluded from the initial grid because the DVT recommends it only if Mode P is unstable (diagnosed post-Phase 1).

**Alternatives Considered:**
- Larger grid with s_train ∈ {1, 5, 21}: adds 3× configs, questionable benefit (DVT recommends s=1)
- Random search: less interpretable, harder to compare across folds
- Bayesian optimization: requires sequential evaluation, incompatible with the nested fold structure

**Impact:** `src/walk_forward/phase_a.py`, `src/config.py` (may need a HPGridConfig dataclass). Computational budget ~1100 training runs across the full walk-forward.

**Status:** `provisional`

---

#### [MOD-008] α calibration — elbow detection on variance-entropy frontier

**Date:** 2026-02-06
**Agent/Author:** lead session
**Spec Reference:** ISD MOD-008 Sub-task 5 (frontier.py), DVT Section 4.7

**Context:** The α parameter (entropy reward weight) is calibrated at the "elbow" of the variance-entropy frontier. The ISD says "α where ΔH/ΔVar < threshold" but does not specify the threshold. The DVT was checked for details.

**Gap:** The DVT (Section 4.7) provides more context but still no explicit threshold: "The elbow — where the marginal entropy gain per unit of additional variance starts to diminish — is the natural operating point. For automated selection, use the α value where ΔH/ΔVar drops below a threshold."

**Decision:** Implement a two-step elbow detection:

1. **Compute the frontier:** For each α in the grid {0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0} at fixed λ=1.0, solve the optimization and record (Var*(α), H*(α)).

2. **Detect the elbow via maximum curvature (Kneedle algorithm):**
   - Normalize both axes to [0, 1]
   - Compute the curvature κ at each point on the piecewise-linear frontier
   - Select the α with maximum curvature (= the point of sharpest bend)
   - Fallback: if the frontier is nearly linear (max κ < 0.1), select α = 0.1 (moderate diversification)

This is equivalent to the classic "knee point" detection used in scree plots and L-curve regularization, and avoids an arbitrary ΔH/ΔVar threshold.

**Rationale:** A fixed threshold for ΔH/ΔVar would require calibration itself and would be sensitive to the scale of H and Var (which vary across folds and data configurations). Maximum curvature is scale-invariant after normalization and has a clear geometric interpretation: the point where adding more entropy becomes increasingly expensive in variance. The Kneedle algorithm (Satopää et al., 2011) is the standard reference for this.

**Alternatives Considered:**
- Fixed ΔH/ΔVar threshold (e.g., 0.5): requires calibration, not robust across folds
- Visual inspection: not automatable
- Fixed α = 0.1: loses adaptivity to the risk model structure

**Impact:** `src/portfolio/frontier.py`. May add `kneed` as optional dependency, or implement the simple max-curvature logic directly (5–10 lines of code, no external dependency needed).

**Status:** `provisional`

---

#### [MOD-013] Spinu ERC solver — log-barrier Newton formulation

**Date:** 2026-02-06
**Agent/Author:** lead session
**Spec Reference:** ISD MOD-013, DVT Reference [38]

**Context:** The ERC (Equal Risk Contribution) benchmark uses the Spinu (2013) convex log-barrier formulation. The ISD gives the objective but not the Newton step details. The DVT was checked for details.

**Gap:** The DVT (Reference [38]) cites: "Spinu, F. (2013). An Algorithm for Computing Risk Parity Weights. SSRN 2297383. Convex log-barrier formulation for vanilla ERC; Newton's method, provably convergent in <5 iterations for n < 1000." But neither the ISD nor DVT spell out the exact Newton iteration.

**Decision:** Implement the Spinu (2013) formulation as follows:

**Convex reformulation:** Instead of directly solving the non-convex ERC problem (equalize risk contributions), Spinu shows it is equivalent to:

$$\min_y \quad \frac{1}{2} y^T \Sigma y - \sum_{i=1}^{n} \ln(y_i)$$

where the unconstrained variable $y > 0$ is then normalized: $w_i = y_i / \sum_j y_j$.

This is a strictly convex problem (sum of convex quadratic + convex barrier), so Newton's method converges globally.

**Newton iteration:**
- Gradient: $g = \Sigma y - 1/y$ (element-wise inverse)
- Hessian: $H = \Sigma + \text{diag}(1/y_i^2)$
- Update: $y^{(t+1)} = y^{(t)} - H^{-1} g$
- Initialization: $y^{(0)} = \mathbf{1}$ (or $1/\sigma_i$)
- Convergence: $\|g\| < 10^{-10}$, typically 3–5 iterations

**Post-hoc constraint projection:** The raw ERC solution w = y/sum(y) may violate the hard caps (w_max=5%, w_min=0.1%). Apply iterative projection: clip at w_max, zero below w_min, renormalize, repeat 2–3 passes.

**Rationale:** This is the standard formulation from the Spinu (2013) paper, widely used in production risk parity implementations. The log-barrier makes the problem strictly convex, guaranteeing a unique solution. Newton's method on this specific problem is known to converge in <5 iterations for n < 1000 (as cited in the DVT). No need for a general-purpose nonlinear solver.

**Alternatives Considered:**
- CVXPY with log constraint: works but slower (overhead of the modeling layer)
- Roncalli (2010) cyclical coordinate descent: slower convergence, less elegant
- SCA reuse from MOD-008: overkill for the convex ERC problem

**Impact:** `src/benchmarks/erc.py`. No additional dependencies (NumPy + SciPy linalg sufficient).

**Status:** `provisional`

#### [MOD-005] Stratification strategy — k-means only, no GICS sectors

**Date:** 2026-02-06
**Agent/Author:** human + lead session
**Spec Reference:** ISD MOD-005 Sub-task 1 (batching.py), DVT Section 4.4

**Context:** Curriculum batching during Phases 1-2 requires stratified sampling across stocks within synchronized temporal blocks. The DVT originally offered two options: k-means on trailing 63-day returns, or GICS sectors as a zero-cost proxy.

**Gap:** Using predefined sector categories contradicts the strategy's core principle (DVT Section 1.3): "Let the data reveal the underlying risk structures themselves, without imposing prior categorization."

**Decision:** Use **k-means on trailing 63-day returns only**. GICS sectors removed as an option. DVT and ISD updated accordingly. Fallback for stocks with < 63 days of history: assign to nearest cluster based on available data.

**Rationale:** K-means produces data-driven strata reflecting recent co-movement patterns. Computational cost is negligible (~1s for n=1000, S=15). Injecting sector labels — even for batching only — introduces an a priori bias inconsistent with the strategy's philosophy.

**Alternatives Considered:**
- GICS sectors (rejected — philosophical inconsistency)
- No stratification (rejected — excessive gradient variance in synchronous batching)

**Impact:** `src/training/batching.py`. DVT Sections 4.4 and 4.5 updated. ISD MOD-005 updated.

**Status:** `validated`

---

#### [MOD-009] Phase B E* — expanding median, point-in-time

**Date:** 2026-02-06
**Agent/Author:** lead session
**Spec Reference:** ISD MOD-009 Sub-task 3 (phase_b.py), DVT Section 4.8

**Context:** Phase B trains on all data without validation set, for a fixed E* epochs. The DVT says E* "can be taken as the median of E*_config across folds rather than the single fold's value" but does not specify whether this median is global or expanding.

**Gap:** A global median across all folds would require future fold information, violating CONV-10 (point-in-time).

**Decision:** Three cases:
1. **Per-fold default:** E* = E*_config of the selected config in that fold's Phase A.
2. **Robust alternative:** E* = median of E*_config across all previous folds (expanding).
3. **Holdout:** E* = median of E*_config across all walk-forward folds (global, since holdout runs after all folds).

**Rationale:** Only the expanding median is compatible with CONV-10. The holdout is the single exception since it executes after the entire walk-forward.

**Alternatives Considered:**
- Global median for all folds (rejected — look-ahead violation)
- Per-fold only (rejected — DVT explicitly recommends median for robustness)

**Impact:** `src/walk_forward/phase_b.py`, `src/walk_forward/folds.py`. ISD MOD-009 updated.

**Status:** `validated`

---

#### [MOD-008] μ=0 default mode — directional deferred to Iteration 3

**Date:** 2026-02-06
**Agent/Author:** lead session
**Spec Reference:** ISD MOD-008 Sub-task 2 (sca_solver.py), DVT Sections 4.7 and 6.4

**Context:** The SCA solver accepts μ (expected returns) as a parameter. The DVT defines two modes: default (μ=0) and directional (μ≠0).

**Gap:** The ISD sub-problem formulation includes w^T μ but does not specify which mode to use for the walk-forward evaluation.

**Decision:** All walk-forward and benchmark evaluations use **μ=0**. The solver accepts μ as a parameter for future extensibility, but defaults to zero. Directional mode is deferred to DVT Iteration 3 (Section 6.4), triggered only if the default mode underperforms on Sharpe/Calmar.

**Rationale:** DVT Section 4.7: "Setting μ=0 is not an omission but a design choice: the strategy produces risk structure, not return forecasts." Section 6.4 explicitly positions directional mode as a future iteration.

**Alternatives Considered:** None — this is the documented strategy.

**Impact:** `src/portfolio/sca_solver.py` (μ parameter with default=None → treated as 0). ISD MOD-008 updated.

**Status:** `validated`

---

### Phase 2 — Core Pipeline

<!-- MOD-004 to MOD-008 entries -->

### Phase 3 — Benchmarks

<!-- MOD-010 to MOD-015 entries -->

### Phase 4 — Integration

<!-- MOD-009 and MOD-016 entries -->

---

## Corrections

> Implementations missing from the code but clearly specified in ISD/DVT. Fixed and logged here.

| # | Date | Module | Description |
|---|------|--------|-------------|
| 1 | 2026-02-07 | MOD-009 | Added `crisis_period_return()` to `metrics.py` — ISD Layer 3 primary metric |
| 2 | 2026-02-07 | MOD-009 | Added `realized_vs_predicted_correlation()` to `metrics.py` — ISD Layer 2 metric |
| 3 | 2026-02-07 | MOD-016 | Added `--config PATH` arg to `run_walk_forward.py` — ISD MOD-016 Sub-task 4 |
| 4 | 2026-02-07 | MOD-016 | Added `--device` arg to `run_benchmarks.py` — ISD MOD-016 Sub-task 4 (API consistency) |

---

## Divergences

> Code diverges from clear ISD/DVT specifications. Documented here for future resolution.

*(none identified)*

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| Total decisions | 9 |
| Provisional | 5 |
| Validated | 4 |
| Superseded | 0 |
| Corrections | 4 |

*Last updated: 2026-02-07 (post-audit)*
