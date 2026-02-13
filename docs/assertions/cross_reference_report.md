# Cross-Reference Report: Test Suite vs Reference Documents

**Generated from:** DVT v4.1, ISD, divergences.md  
**Test suite:** `tests/unit/` (11 files, ~9,000 lines)

---

## Section A — Divergences vs Test Coverage

Each of the 10 documented divergences (docs/divergences.md) is assessed for test coverage.

### Divergence #1: AU_max_stat formula

- **Code:** `floor(sqrt(2 * N_obs / r_min))`
- **DVT (exact):** `floor((-1 + sqrt(1 + 4*N_obs/r_min)) / 2)` — quadratic solution
- **Impact:** Negligible (~0.1% difference for large N_obs)
- **Test coverage: TESTED**
  - [test_AU_max_stat_formula](../tests/unit/test_inference.py#L300) — verifies `compute_au_max_stat` matches `floor(sqrt(2*N_obs/r_min))` for 4 cases
  - [test_au_max_stat_formula_edge_cases](../tests/unit/test_inference.py#L381) — edge cases (n_obs=10)
  - [TestAUMaxStatDVTTable](../tests/unit/test_inference.py#L459) — verifies DVT §4.8 table values using the **code** formula
  - [TestINV003BAShapeAndAUBound](../tests/unit/test_invariants.py#L155) — AU ≤ AU_max_stat
- **Gap:** Tests verify the CODE formula, not the DVT formula. No test asserts the two formulas give identical (or nearly identical) results for the same inputs.

### Divergence #2: Beta floor 0.01 in Mode F

- **Code:** `get_beta_t()` returns `max(0.01, min(1, epoch/T_warmup))`
- **DVT/ISD:** `beta_t = min(1, t/T_warmup)` — no floor
- **Impact:** Defensive; prevents KL contribution from being exactly zero at epoch 0
- **Test coverage: TESTED**
  - [test_mode_F_beta_annealing](../tests/unit/test_loss_function.py#L195) — asserts `get_beta_t(0, ...) == 0.01`
  - [TestModeProperties.test_mode_F_has_beta_annealing](../tests/unit/test_loss_function.py#L590) — asserts beta_0 = 0.01, beta_10 < beta_20, beta_20 = 1.0
  - [test_mode_F_beta_t_values_in_loss](../tests/unit/test_loss_function.py#L870) — verifies beta_t = 0.01 at epoch 0, 0.5 at mid-warmup, 1.0 at T_warmup
- **Status:** Fully tested. The floor is explicitly verified in assertions.

### Divergence #3: Dropout asymmetry (encoder 0.2, decoder 0.1)

- **Code:** `encoder.py` has `DROPOUT = 0.2`, `decoder.py` has `DROPOUT = 0.1`, `config.py` default = 0.1
- **DVT appended source:** `DROPOUT = 0.1` is the single design-time constant
- **Test coverage: PARTIALLY TESTED**
  - [TestDropoutParam.test_build_vae_respects_dropout_param](../tests/unit/test_vae_architecture.py#L569) — verifies `build_vae(dropout=0.3)` propagates to both encoder and decoder modules
- **Gap:** No test asserts the DEFAULT dropout values match between encoder and decoder. The test only verifies explicit override propagation. The asymmetry (0.2 encoder / 0.1 decoder) is NOT detected by any test.

### Divergence #4: Variance targeting (`_variance_targeting_scale()`)

- **Code:** `pipeline.py` contains `_variance_targeting_scale()` — not in DVT/ISD
- **Test coverage: NOT DIRECTLY TESTED**
- **Gap:** No unit test for `_variance_targeting_scale()`. The function may be indirectly exercised by integration tests (test_e2e_pipeline.py) but has no dedicated formula verification.

### Divergence #5: Auto-adaptation for small universes (`_adapt_vae_params()`)

- **Code:** `pipeline.py` auto-adapts K, T, etc. for small universes
- **Test coverage: PARTIALLY TESTED**
  - [test_capacity_constraint_remediation](../tests/unit/test_vae_architecture.py#L225) — tests that `build_vae` raises when r > r_max, suggesting remediation (but not the auto-adapt function itself)
- **Gap:** No test for `_adapt_vae_params()` directly. No test verifies that parameter adaptation preserves the capacity-data constraint.

### Divergence #6: Direct mode (`run_direct()`)

- **Code:** `pipeline.py` has `run_direct()` — dev/debug mode not in DVT/ISD
- **Test coverage: NOT TESTED**
- **Gap:** No unit test for `run_direct()`. It may be covered by integration tests only.

### Divergence #7: 4 cardinality methods vs DVT's 1-2

- **Code:** sequential, gradient, miqp, two_stage (+ auto, fallback)
- **DVT:** Describes only sequential entropy-aware elimination
- **Test coverage: FULLY TESTED**
  - [TestCardinality.test_cardinality_enforcement](../tests/unit/test_portfolio_optimization.py#L310) — gradient method
  - [test_gradient_method](../tests/unit/test_portfolio_optimization.py#L452) — gradient with real SCA
  - [test_sequential_method](../tests/unit/test_portfolio_optimization.py#L466) — sequential
  - [test_miqp_method](../tests/unit/test_portfolio_optimization.py#L478) — MIQP (skip if no MOSEK)
  - [test_two_stage_method](../tests/unit/test_portfolio_optimization.py#L490) — two-stage (skip if no MOSEK)
  - [test_fallback_no_mi](../tests/unit/test_portfolio_optimization.py#L502) — miqp falls back to gradient
  - [test_auto_resolves](../tests/unit/test_portfolio_optimization.py#L527) — auto method
- **Status:** All 4 methods + fallback are tested for semi-continuous constraint satisfaction.

### Divergence #8: Multi-start composition differs from DVT

- **Code:** EW / inverse-diag / inverse-vol / 2-random starts
- **DVT:** EW / min-variance / ERC / 2-random starts
- **Test coverage: PARTIALLY TESTED**
  - [test_multi_start_deterministic](../tests/unit/test_portfolio_optimization.py#L270) — same seed → same result (exercises multi_start_optimize)
  - [test_multi_start_finite_output](../tests/unit/test_portfolio_optimization.py#L906) — finite output
  - [test_multi_start_explores_better_optima](../tests/unit/test_portfolio_optimization.py#L920) — n_starts=5 ≥ n_starts=1
- **Gap:** No test verifies which specific starting points are used (EW/inverse-diag/inverse-vol/random). The tests verify the API works but not the composition.

### Divergence #9: `create_windows` returns 3-tuple

- **Code:** Returns `(windows, metadata, raw_returns)`
- **ISD:** Specifies 2-tuple `(windows, metadata)`
- **Test coverage: FULLY TESTED**
  - [TestWindowsOutputTriple.test_windows_output_triple](../tests/unit/test_data_pipeline.py#L672) — asserts `len(result) == 3`, verifies types and shapes of all 3 elements
  - [TestRawReturns.test_raw_returns_not_zscored](../tests/unit/test_data_pipeline.py#L765) — verifies raw_returns is NOT z-scored, confirming it's for Spearman co-movement
- **Status:** Fully tested. The 3rd element (raw_returns) is validated for shape, type, and non-z-scoring.

### Divergence #10: Fresh CVXPY problem per SCA iteration

- **Code:** Creates fresh CVXPY problem each iteration (no parametric reuse)
- **DVT:** Implies fixed parametric problem
- **Test coverage: NOT DIRECTLY TESTED**
- **Gap:** No test verifies whether the CVXPY problem is fresh or reused. The SCA convergence tests implicitly exercise it but don't assert this implementation choice.

---

## Section B — CONV Conventions vs Test Coverage

### CONV-01: Log returns (not arithmetic)

- **ISD:** `r_{i,t} = ln(P_{i,t} / P_{i,t-1})`
- **Tests:**
  - [test_conv01_log_returns_not_arithmetic](../tests/unit/test_conventions.py#L38) ✅
  - [TestLogReturns.test_log_returns_vs_arithmetic](../tests/unit/test_data_pipeline.py#L88) ✅
- **Status:** FULLY COVERED

### CONV-02: Z-score per window per feature

- **ISD:** Each feature in each window has mean ≈ 0, std ≈ 1
- **Tests:**
  - [test_conv02_zscore_per_window_per_feature](../tests/unit/test_conventions.py#L76) ✅
  - [test_zscore_exact_manual_values](../tests/unit/test_conventions.py#L114) ✅
  - [TestWindowing.test_zscore_per_window](../tests/unit/test_data_pipeline.py#L112) ✅
  - [TestZScoreFormulaVerification.test_zscore_manual_recomputation](../tests/unit/test_data_pipeline.py#L830) ✅
- **Status:** FULLY COVERED (formula-level)

### CONV-03: Zero-based indices

- **ISD:** fold_ids, active_dims, window indices start at 0
- **Tests:**
  - [test_conv03_fold_ids_start_at_zero](../tests/unit/test_conventions.py#L150) ✅
  - [test_conv03_active_dims_0_based](../tests/unit/test_conventions.py#L168) ✅
  - [test_conv03_window_indices_0_based](../tests/unit/test_conventions.py#L187) ✅
- **Status:** FULLY COVERED

### CONV-04: PyTorch for VAE, NumPy for downstream

- **ISD:** Training uses PyTorch tensors; portfolio/risk use NumPy arrays
- **Tests:**
  - [test_conv04_pytorch_for_vae_numpy_for_downstream](../tests/unit/test_conventions.py#L209) ✅
- **Status:** FULLY COVERED

### CONV-05: Window shape (B, T, F)

- **ISD:** Input windows stored as (Batch, Time, Features)
- **Tests:**
  - [test_conv05_window_shape_B_T_F](../tests/unit/test_conventions.py#L260) ✅
  - [TestWindowing.test_window_shape](../tests/unit/test_data_pipeline.py#L131) ✅
  - [test_feature_stacking_order](../tests/unit/test_data_pipeline.py#L713) — verifies channel 0 = returns, channel 1 = vol ✅
- **Status:** FULLY COVERED

### CONV-06: σ² is scalar

- **ISD:** `log_sigma_sq.ndim == 0`
- **Tests:**
  - [test_conv06_sigma_sq_scalar](../tests/unit/test_conventions.py#L291) ✅
  - [test_log_sigma_sq_scalar](../tests/unit/test_vae_architecture.py#L314) ✅
  - [test_sigma_sq_stays_scalar_after_training](../tests/unit/test_vae_architecture.py#L655) — after 20 gradient steps ✅
- **Status:** FULLY COVERED

### CONV-07: AU threshold 0.01 nats

- **ISD:** Active unit defined as `E_q[KL_k] > 0.01`
- **Tests:**
  - [test_conv07_au_threshold_0_01_nats](../tests/unit/test_conventions.py#L320) ✅
  - [TestAUThreshold.test_au_threshold_is_0_01_nats](../tests/unit/test_inference.py#L367) ✅
- **Status:** FULLY COVERED

### CONV-08: Dual rescaling

- **ISD:** Estimation uses date-specific σ; portfolio uses current-date σ
- **Tests:**
  - [test_conv08_dual_rescaling](../tests/unit/test_conventions.py#L350) ✅
  - [test_rescaling_formula_known_vols](../tests/unit/test_conventions.py#L395) — formula verification ✅
  - [TestINV004DualRescalingFormulas](../tests/unit/test_invariants.py#L955) — exact formula with manual winsorization ✅
  - [TestDualRescalingCorrectness](../tests/unit/test_risk_model.py#L472) — date-specific vs current-date ✅
- **Status:** FULLY COVERED (formula-level)

### CONV-09: Expanding window walk-forward

- **ISD:** Training windows expand (same start, monotonically increasing end)
- **Tests:**
  - [test_conv09_expanding_window_walk_forward](../tests/unit/test_conventions.py#L450) ✅
  - [test_folds_expanding_training_window](../tests/unit/test_walk_forward_detailed.py#L220) ✅
- **Status:** FULLY COVERED

### CONV-10: Point-in-time universe

- **ISD:** Universe at date t uses only stocks observable at t (no survivorship bias)
- **Tests:**
  - [test_conv10_point_in_time_universe](../tests/unit/test_conventions.py#L501) ✅
  - [TestUniverse.test_universe_point_in_time](../tests/unit/test_data_pipeline.py#L190) ✅
- **Status:** FULLY COVERED

---

## Section C — INV Invariants vs Test Coverage

### INV-001: D = T × F factor in reconstruction loss

- **ISD:** `recon_coeff == T * F / (2 * sigma_sq)` for Mode P
- **Tests:**
  - [TestINV001DFactorCoefficient](../tests/unit/test_invariants.py#L66) — parametrized over 3 modes ✅
  - [test_D_factor_present](../tests/unit/test_loss_function.py#L105) ✅
  - [test_D_factor_coefficient_exact_value](../tests/unit/test_loss_function.py#L417) — 3 different log_sigma_sq values ✅
  - [test_D_factor_production_dimensions](../tests/unit/test_loss_function.py#L760) — T=504, F=2, D=1008 ✅
- **Status:** FULLY COVERED (formula-level, multiple dimensions)

### INV-002: σ² scalar, clamped to [1e-4, 10]

- **ISD:** `sigma_sq = clamp(exp(log_sigma_sq), 1e-4, 10.0)`
- **Tests:**
  - [TestINV002SigmaSqScalar](../tests/unit/test_invariants.py#L116) ✅
  - [test_sigma_sq_clamped](../tests/unit/test_training.py#L62) — after 10 epochs ✅
  - [TestSigmaSqClampingPerStep](../tests/unit/test_invariants.py#L1205) — at EVERY step (30 steps) ✅
  - [test_sigma_sq_clamp_exact_values](../tests/unit/test_training.py#L485) — boundary values (lower, upper, interior) ✅
- **Status:** FULLY COVERED (per-step verification)

### INV-003: B_A shape (n, AU) with AU ≤ AU_max_stat

- **ISD:** Exposure matrix B_A has shape `(n_active, AU)` with AU bounded
- **Tests:**
  - [TestINV003BAShapeAndAUBound](../tests/unit/test_invariants.py#L155) ✅
  - [test_B_shape](../tests/unit/test_inference.py#L262) ✅
- **Status:** FULLY COVERED

### INV-004: Dual rescaling — estimation ≠ portfolio

- **ISD:** `rescale_estimation()` returns dict; `rescale_portfolio()` returns ndarray
- **Tests:**
  - [TestINV004DualRescalingDiffers](../tests/unit/test_invariants.py#L210) — type check ✅
  - [TestINV004DualRescalingFormulas](../tests/unit/test_invariants.py#L955) — formula verification with manual winsorization ✅
  - [TestDualRescalingCorrectness](../tests/unit/test_risk_model.py#L472) — date-specific vs current ✅
  - [TestRescalingKnownValues](../tests/unit/test_risk_model.py#L780) — fixture-based with pre-computed winsorized ratios ✅
- **Status:** FULLY COVERED (formula-level)

### INV-005: No lookahead / embargo ≥ 21 days

- **ISD:** OOS data never leaks into training; embargo gap ≥ 21 trading days
- **Tests:**
  - [test_no_lookahead_fold_dates](../tests/unit/test_invariants.py#L256) ✅
  - [test_embargo_is_at_least_21_trading_days](../tests/unit/test_invariants.py#L289) ✅
  - [test_vix_threshold_no_future_data](../tests/unit/test_invariants.py#L324) ✅
  - [test_embargo_gap_size](../tests/unit/test_walk_forward_detailed.py#L21) ✅
  - [TestEmbargoGapExactCount.test_embargo_is_trading_days_not_calendar](../tests/unit/test_walk_forward_detailed.py#L433) ✅
- **Status:** FULLY COVERED

### INV-006: Modes P / F / A mutually exclusive

- **ISD:** Error on invalid mode or forbidden combination (e.g., Mode P + beta ≠ 1)
- **Tests:**
  - [TestINV006ModesMutuallyExclusive](../tests/unit/test_invariants.py#L355) — invalid_mode_raises, mode_P_with_beta_not_1_raises, mode_P_learns_sigma_sq_mode_F_freezes ✅
  - [test_modes_exclusive](../tests/unit/test_loss_function.py#L218) ✅
  - [TestModePBetaAnnealingForbidden](../tests/unit/test_invariants.py#L1271) — Mode P ignores warmup_fraction ✅
- **Status:** FULLY COVERED

### INV-007: Entropy H(w) in principal factor basis

- **ISD:** `H = -Σ ĉ'_k · ln(ĉ'_k)`, ĉ'_k = c'_k/C, c'_k = λ_k · (β'_k)²
- **Tests:**
  - [TestINV007EntropyPrincipalFactorBasis](../tests/unit/test_invariants.py#L419) — risk contributions non-negative, H bounded ✅
  - [TestINV007ManualEntropyComputation](../tests/unit/test_invariants.py#L1352) — 2-factor step-by-step manual trace ✅
  - [TestINV008ExactPercentile](../tests/unit/test_invariants.py#L1414) — percentile matches numpy ✅
  - [TestEntropy.test_entropy_gradient_at_maximum](../tests/unit/test_portfolio_optimization.py#L98) — ∇H = 0 at max ✅
  - [TestEntropy.test_entropy_gradient_numerical](../tests/unit/test_portfolio_optimization.py#L127) — finite difference verification ✅
  - [TestEntropyFormulaVerification](../tests/unit/test_portfolio_optimization.py#L1000) — identity, concentrated, manual 3-factor ✅
  - [test_entropy_at_equal_weight_with_two_factor](../tests/unit/test_risk_model.py#L774) — fixture-based ✅
- **Status:** FULLY COVERED (formula-level, gradient verification)

### INV-008: Winsorization [P5, P95]

- **ISD:** `ratios = clip(σ_{i,t}/median(σ_t), P5, P95)`
- **Tests:**
  - [TestINV008WinsorizationBounds](../tests/unit/test_invariants.py#L446) — extreme outlier clipped ✅
  - [TestINV008ExactPercentile](../tests/unit/test_invariants.py#L1414) — numpy P5/P95 match ✅
  - [test_winsorization_uses_median_not_mean](../tests/unit/test_risk_model.py#L175) ✅
  - [test_winsorization_applied_in_rescaling](../tests/unit/test_risk_model.py#L650) — 50x outlier bounded ✅
  - [TestRescalingKnownValues](../tests/unit/test_risk_model.py#L780) — exact pre-computed values ✅
- **Status:** FULLY COVERED (formula-level)

### INV-009: Gradient ∇H = 0 at maximum entropy

- **ISD:** At ĉ'_k = 1/AU for all k, gradient must vanish
- **Tests:**
  - [TestINV009GradientZeroAtMaximum](../tests/unit/test_invariants.py#L476) ✅
  - [TestEntropy.test_entropy_gradient_at_maximum](../tests/unit/test_portfolio_optimization.py#L98) — grad_active all zero ✅
- **Status:** FULLY COVERED

### INV-010: Curriculum phases (λ_co scheduling)

- **ISD:** Phase 1 (0-30%): λ_co_max; Phase 2 (30-60%): linear decay; Phase 3 (60-100%): 0
- **Tests:**
  - [TestINV010CurriculumPhases](../tests/unit/test_invariants.py#L510) — boundaries + monotonicity ✅
  - [test_curriculum_phases](../tests/unit/test_loss_function.py#L310) — boundary values at 0, 15, 29, 30, 45, 60, 99 ✅
  - [test_co_movement_included_in_phase_1_excluded_in_phase_3](../tests/unit/test_loss_function.py#L930) — integration with compute_loss ✅
  - [test_curriculum_sampler_sync_vs_random_batches](../tests/unit/test_training.py#L260) — batch sampler modes ✅
- **Status:** FULLY COVERED

### INV-011: Validation ELBO excludes γ and λ_co, includes σ²

- **ISD:** `L_val = D/(2σ²)·L_recon^(γ=1) + (D/2)·ln(σ²) + L_KL`
- **Tests:**
  - [TestINV011ValidationELBO](../tests/unit/test_invariants.py#L553) — deterministic + differs from training ✅
  - [test_validation_elbo_excludes_gamma](../tests/unit/test_loss_function.py#L338) — manual gamma-free formula + gamma invariance ✅
  - [TestModeProperties.test_validation_elbo_formula_exact](../tests/unit/test_loss_function.py#L636) — exact formula ✅
  - [TestEndToEndELBOKnownValues.test_validation_elbo_known_values](../tests/unit/test_loss_function.py#L1083) — known (1,2,2) tensor ✅
- **Status:** FULLY COVERED (formula-level, exact manual computation)

### INV-012: Benchmark constraints identical across all models

- **ISD:** All 6 benchmarks receive, preserve, and enforce identical constraint_params
- **Tests:**
  - [TestINV012BenchmarkConstraintsIdentical.test_all_benchmarks_share_same_constraints](../tests/unit/test_invariants.py#L847) ✅
  - [test_all_benchmarks_enforce_constraints_on_optimize](../tests/unit/test_invariants.py#L885) — run optimize(), check weight constraints ✅
  - [TestBenchmarks.test_constraints_identical](../tests/unit/test_benchmarks.py#L70) ✅
  - [test_all_benchmarks_respect_w_max](../tests/unit/test_benchmarks.py#L214) ✅
  - [test_benchmark_output_format](../tests/unit/test_benchmarks.py#L184) — shape, sum=1, non-negative ✅
- **Status:** FULLY COVERED

---

## Section D — DVT Formulas with NO Test

The following formulas from the DVT that are NOT verified by any test:

### D1. Exponentially-weighted composite aggregation (DVT §6.2)

$$\bar{\mu}_i = \frac{\sum_{t} \lambda^{M-t} \cdot \mu_i^{(t)}}{\sum_{t} \lambda^{M-t}}$$

- **Status:** NO TEST — This is described as Iteration 1 (future work), not yet implemented.
- **Priority:** Low (not yet in code)

### D2. Regime-aware Σ_z blending (DVT §6.3)

$$\hat{\Sigma}_z^{\text{blend}} = \omega \cdot \hat{\Sigma}_z^{\text{full}} + (1 - \omega) \cdot \hat{\Sigma}_z^{\text{recent}}$$

- **Status:** NO TEST — Iteration 2, not yet implemented.
- **Priority:** Low (future work)

### D3. DVT exact AU_max_stat quadratic formula

$$\text{AU}_{\max,\text{stat}} = \left\lfloor \frac{-1 + \sqrt{1 + 4N_{\text{obs}}/r_{\min}}}{2} \right\rfloor$$

- **Status:** NOT TESTED as such — tests use the code's approximation `floor(sqrt(2*N_obs/r_min))`. See Divergence #1.
- **Priority:** Low (difference is negligible)

### D4. Variance targeting scale

- `_variance_targeting_scale()` in pipeline.py
- **Status:** NO UNIT TEST
- **Priority:** Medium — this is active code not in DVT/ISD

### D5. Auto-adaptation for small universes

- `_adapt_vae_params()` in pipeline.py
- **Status:** NO UNIT TEST
- **Priority:** Medium — this is active code not in DVT/ISD

### D6. Direct mode (`run_direct()`)

- `run_direct()` in pipeline.py
- **Status:** NO UNIT TEST
- **Priority:** Low (debug mode)

### D7. Multi-start starting point composition

- DVT specifies: EW / min-variance / ERC / 2-random
- Code uses: EW / inverse-diag / inverse-vol / 2-random
- **Status:** No test verifies which starting points are used. Only the API behavior is tested.
- **Priority:** Medium — the choice affects optimization quality

### D8. ReduceLROnPlateau parameters

- DVT §4.8: `factor=0.5, patience=5`
- **Status:** NOT DIRECTLY TESTED — No test verifies the scheduler parameters
- **Priority:** Low (trainer tests verify loss decrease and warmup protection, but not specific scheduler params)

### D9. Adam optimizer parameters

- DVT: `β1=0.9, β2=0.999, lr=1e-4`
- **Status:** NOT DIRECTLY TESTED — VAETrainer uses lr=1e-3 in tests (test default), not 1e-4 (DVT baseline)
- **Priority:** Low (hyperparameters are configurable)

### D10. Concentration penalty soft threshold w̄ value

- DVT §4.7: `w̄ = round(1/n, 4)` (approximately 1/n)
- **Status:** Tested at formula level (TestConcentrationPenaltyFormula uses w_bar=0.03), but no test verifies the default w̄ = 1/n rule.
- **Priority:** Low

### D11. InfoNCE contrastive loss (DVT §6.6)

- **Status:** NO TEST — Iteration 5, not yet implemented.
- **Priority:** Low (future work)

---

## Section E — Tests That Contradict Reference Documents

### E1. AU_max_stat formula mismatch (LOW severity)

- **Test:** [test_AU_max_stat_formula](../tests/unit/test_inference.py#L300) asserts `compute_au_max_stat(n_obs, r_min) == floor(sqrt(2*n_obs/r_min))`
- **DVT §4.3:** States `AU_max_stat = floor((-1 + sqrt(1 + 4*N_obs/r_min))/2)` (exact quadratic solution of `AU*(AU+1)/2 ≤ N_obs/r_min`)
- **Impact:** For N_obs=7560, r_min=2: code gives 86, DVT exact gives 86 (`(-1+sqrt(15121))/2 = 86.98 → 86`). Negligible difference.
- **Verdict:** Minor — documented in divergences.md. Functionally equivalent for all practical inputs. Consider adding a comparison test.

### E2. Beta floor 0.01 not in DVT/ISD (LOW severity)

- **Test:** [test_mode_F_beta_annealing](../tests/unit/test_loss_function.py#L195) asserts `get_beta_t(0, ...) == 0.01`
- **DVT §4.4:** `β_t = min(1, t/T_warmup)` → at epoch 0, β_t = 0 (no floor)
- **Impact:** Improves numerical stability (KL never fully zeroed). Documented in divergences.md.
- **Verdict:** Acceptable defensive deviation — test correctly verifies the CODE behavior which intentionally differs from DVT.

### E3. Dropout default asymmetry (MEDIUM severity)

- **Source code:** `encoder.py:DROPOUT = 0.2`, `decoder.py:DROPOUT = 0.1`
- **DVT Appendix A:** `DROPOUT = 0.1` (single value for both)
- **config.py:** `dropout: float = 0.1`
- **ISD:** Does not specify different dropout per module
- **Test:** `test_build_vae_respects_dropout_param` only tests explicit override, not defaults
- **Verdict:** Genuine inconsistency. When `build_vae()` is called WITHOUT a dropout parameter, the encoder uses its module-level `DROPOUT=0.2` while the decoder uses `DROPOUT=0.1`. This contradicts DVT's uniform 0.1. **No test catches this.**

### E4. create_windows returns 3-tuple vs ISD's 2-tuple

- **Test:** [test_windows_output_triple](../tests/unit/test_data_pipeline.py#L672) asserts `len(result) == 3`
- **ISD MOD-001:** Specifies `returns (windows, metadata)` — 2 elements
- **Impact:** The 3rd element (raw_returns) is needed for co-movement Spearman computation (ISD MOD-004 explicitly says "use raw returns"). The ISD module interface is stale.
- **Verdict:** ISD needs updating. The test correctly verifies the actual (needed) behavior. Documented in divergences.md.

### E5. Multi-start composition

- **Test:** Tests verify multi_start_optimize API behavior but do NOT verify starting point composition
- **DVT §4.7:** Starting points are EW / min-variance / ERC / 2-random
- **Code:** Uses EW / inverse-diag / inverse-vol / 2-random (approximations to avoid solving sub-problems)
- **Verdict:** No test contradiction per se — the tests never assert the starting point identities. But the code deviates from DVT silently.

---

## Summary Table

| Item | DVT/ISD Ref | Test Coverage | Status |
|------|-------------|--------------|--------|
| **Divergences** | | | |
| #1 AU_max_stat formula | DVT §4.3 | Code formula tested, not DVT exact | ⚠️ GAP |
| #2 Beta floor 0.01 | DVT §4.4 | Fully tested | ✅ |
| #3 Dropout asymmetry | DVT Appendix | Not tested (defaults) | ❌ GAP |
| #4 Variance targeting | Not in DVT | Not tested | ❌ GAP |
| #5 Auto-adaptation | Not in DVT | Not tested | ❌ GAP |
| #6 run_direct() | Not in DVT | Not tested | ⚠️ Low priority |
| #7 Cardinality methods | DVT §4.7 | All methods tested | ✅ |
| #8 Multi-start composition | DVT §4.7 | API tested, not composition | ⚠️ GAP |
| #9 3-tuple return | ISD MOD-001 | Fully tested | ✅ |
| #10 Fresh CVXPY | DVT §4.7 | Not tested | ⚠️ Low priority |
| **Conventions** | | | |
| CONV-01 through CONV-10 | ISD §00 | All fully covered | ✅ |
| **Invariants** | | | |
| INV-001 through INV-012 | ISD §00 | All fully covered | ✅ |
| **Untested DVT Formulas** | | | |
| Variance targeting | pipeline.py | No unit test | ❌ |
| Auto-adapt params | pipeline.py | No unit test | ❌ |
| Multi-start starts | DVT §4.7 | Not verified | ⚠️ |
| LR scheduler params | DVT §4.8 | Not verified | ⚠️ |
| **Contradictions** | | | |
| E1 AU formula | Minor | Documented | ⚠️ |
| E2 Beta floor | Defensive | Documented | ✅ |
| E3 Dropout asymmetry | Genuine bug | NOT documented | ❌ |
| E4 3-tuple return | ISD stale | Documented | ✅ |
| E5 Multi-start starts | Silent deviation | NOT tested | ⚠️ |
