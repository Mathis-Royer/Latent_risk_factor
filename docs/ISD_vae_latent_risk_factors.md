# ISD — VAE Latent Risk Factor Discovery Strategy

## Implementation Specification Document — Operational Instantiation

**DVT Source:** `strategie_facteurs_risque_latents_v4_1.md` (v4.1)
**Benchmark Source:** `Latent_risk_factor_benchmark.md`
**Methodology:** ISD Methodology v1.0

---

## 00 — Global Context and Conventions

### Project Objective

Implement an end-to-end portfolio construction pipeline based on latent risk factor discovery using a Variational Autoencoder (VAE). The pipeline includes: financial data preparation, 1D-CNN VAE training, composite risk profile inference, factor risk model estimation, portfolio optimization via factor entropy, and walk-forward validation over 30 years of history. Six benchmarks serve as reference.

### Pipeline Architecture

```
[Stock Data (synthetic / EODHD)] → [data_pipeline] → windows (N×T×F) + crisis_labels
                                        ↓
                               [vae_architecture] → VAEModel
                                        ↓
                          [loss_function] → loss computation
                                        ↓
                             [training] → trained encoder
                                        ↓
                            [inference] → B (n×K), composite profiles
                                        ↓
                           [risk_model] → B_A, Σ_z, D_ε, Σ_assets
                                        ↓
                   [portfolio_optimization] → w* (optimal weights)
                                        ↓
                        [walk_forward] → OOS metrics across ~34 folds
                                        ↓
                       [benchmarks ×6] → comparative metrics
                                        ↓
                        [integration] → final report, statistical tests
```

### Critical Conventions — DO NOT VIOLATE

| ID | Convention | Affected Modules |
|----|-----------|-------------------|
| CONV-01 | **Log returns**: $r_t = \ln(P_t^{\text{adj}} / P_{t-1}^{\text{adj}})$, never arithmetic | All |
| CONV-02 | **Z-score per-window**: each window (T, F) is normalized independently (mean 0, std 1 per feature) | data_pipeline, loss_function, inference |
| CONV-03 | **0-based indices** everywhere (Python standard) | All |
| CONV-04 | **PyTorch tensors** for the VAE; **NumPy/arrays** for the downstream pipeline (risk_model, optimization) | All |
| CONV-05 | **Window shape**: (batch, T, F) — time is dimension 1, features are dimension 2 | data_pipeline, vae_architecture, loss_function, training |
| CONV-06 | **σ² is scalar** (a single value for the entire model), not per-feature or per-dimension vector | loss_function, training, vae_architecture |
| CONV-07 | **AU** = number of active dimensions = |{k : KL_k > 0.01 nats}|, determined once per retraining | inference, risk_model, portfolio_optimization |
| CONV-08 | **Dual rescaling**: date-specific for historical estimation, current-date for portfolio construction | risk_model, portfolio_optimization |
| CONV-09 | **Expanding window** for training (all history), not rolling | walk_forward, training |
| CONV-10 | **Point-in-time**: no future data in any computation; universe reconstituted at each date | data_pipeline, walk_forward |

### Critical Invariants

```yaml
invariants:
  - id: INV-001
    category: mathematical
    description: "The factor D = T × F MUST appear in the numerator of the reconstruction loss.
                  The reconstruction loss is D/(2σ²) · L_recon, NOT 1/(2σ²) · L_recon."
    modules: [loss_function, training, monitoring]
    violation: "Posterior collapse — all dimensions converge to the prior,
               AU → 0, the model is useless"
    detection: "AU < 5 after full training; σ² → lower bound (1e-4)"
    test: "assert loss_recon_coefficient == (T * F) / (2 * sigma_sq)"

  - id: INV-002
    category: mathematical
    description: "σ² is a learned scalar (not a vector). A single parameter log_sigma_sq,
                  with σ² = exp(log_sigma_sq), clamped to [1e-4, 10]."
    modules: [vae_architecture, loss_function]
    violation: "If σ² is vectorial (per-feature or per-dimension), the self-regulated
               recon/KL balancing is broken — each dimension has its own trade-off"
    test: "assert log_sigma_sq.ndim == 0 or log_sigma_sq.numel() == 1"

  - id: INV-003
    category: inter_module
    description: "The matrix B after filtering has shape (n, AU) with AU ≤ AU_max_stat.
                  AU_max_stat = floor(sqrt(2 * N_obs / r_min)) with r_min = 2, N_obs = T_hist_days."
    modules: [inference, risk_model]
    violation: "Σ_z underestimated (observations/parameters ratio < 1), unstable optimization"
    test: |
      assert B_A.shape == (n_stocks, AU)
      assert AU <= AU_max_stat

  - id: INV-004
    category: convention
    description: "Dual rescaling: B_A_estimation[i,t] = (σ_i,t / σ_bar_t) · μ_A_i;
                  B_A_portfolio[i] = (σ_i,now / σ_bar_now) · μ_A_i.
                  DO NOT use the same rescaling for both."
    modules: [risk_model, portfolio_optimization]
    violation: "Using current-date vol for historical estimation incorrectly attributes
               systematic risk to residuals, inflating D_ε"
    test: "Verify that z_hat_t uses B_A_t (date-specific) and that Σ_assets uses B_A_port (current)"

  - id: INV-005
    category: safety
    description: "No look-ahead. Test data is never seen during training.
                  The 21-day embargo separates training and OOS. The VIX threshold is computed
                  on expanding window of the training set only."
    modules: [data_pipeline, walk_forward, training]
    violation: "Invalid backtest — artificially optimistic results"
    test: |
      assert all(train_dates < embargo_start)
      assert all(test_dates > embargo_end)
      assert vix_threshold_computed_on <= training_end_date

  - id: INV-006
    category: mathematical
    description: "The three loss modes (P/F/A) are MUTUALLY EXCLUSIVE.
                  Mode P: σ² learned, β=1 fixed.
                  Mode F: σ²=1 frozen, β_t annealing, D/2 scaling retained.
                  Mode A: σ² learned, β>1 fixed.
                  DO NOT combine learned σ² with β annealing."
    modules: [loss_function, training]
    violation: "Unpredictable σ²/β interaction — the model compensates for double KL pressure"
    test: "assert not (sigma_sq_learned and beta_annealing_enabled)"

  - id: INV-007
    category: mathematical
    description: "The entropy H(w) is computed in the PRINCIPAL FACTOR basis of Σ_z
                  (after V rotation from the eigendecomposition Σ_z = VΛV^T), NOT in the
                  raw latent basis. The contributions c'_k = (β'_p,k)² · λ_k are always ≥ 0."
    modules: [portfolio_optimization, risk_model]
    violation: "Negative contributions → undefined entropy → solver diverges"
    test: |
      assert all(c_prime_k >= 0)
      assert all(eigenvalues >= 0)

  - id: INV-008
    category: inter_module
    description: "The ratio σ_i,t / σ_bar_t is WINSORIZED cross-sectionally at each date
                  at percentiles [P5, P95] BEFORE rescaling. The same winsorized ratio is used
                  for both B_A_estimation AND B_A_portfolio."
    modules: [risk_model]
    violation: "A stock with R=15 receives 225× the regression weight, idiosyncratic noise
               contaminates Σ_z"
    test: "assert all(ratio >= percentile_5) and all(ratio <= percentile_95)"

  - id: INV-009
    category: mathematical
    description: "The gradient of H with respect to w is:
                  ∇_w H = -(2/C) · B' · φ, with φ_k = λ_k · β'_k · (ln(ĉ'_k) + H)
                  where B' = B_A_port_rotated, C = Σ_k λ_k (β'_k)²"
    modules: [portfolio_optimization]
    violation: "Incorrect SCA convergence, suboptimal solution"
    test: "Verify that ∇H = 0 when ĉ'_k = 1/AU for all k (maximum entropy)"

  - id: INV-010
    category: inter_module
    description: "The co-movement curriculum has 3 phases: Phase 1 (λ_co = λ_co_max, ~30% epochs),
                  Phase 2 (linear decay → 0, ~30%), Phase 3 (λ_co = 0, ~40%).
                  Batching changes between phases: synchronous+stratified (Ph 1-2), random (Ph 3)."
    modules: [loss_function, training]
    violation: "Synchronous batching in Phase 3 = unnecessarily high gradient variance;
               random batching in Phase 1 = co-movement loss not computable"

  - id: INV-011
    category: mathematical
    description: "The validation ELBO excludes γ (crisis weighting) and λ_co (co-movement loss).
                  It includes σ². Formula: L_val = D/(2σ²)·L_recon^(γ=1) + (D/2)·ln(σ²) + L_KL"
    modules: [training]
    violation: "Selection bias toward crisis-rich folds if γ is included in validation"

  - id: INV-012
    category: convention
    description: "Portfolio constraints are identical between the VAE and all benchmarks:
                  long-only, fully invested, w_max=5%, w_min=0.10% or 0, P_conc, P_turn, τ_max=30%"
    modules: [portfolio_optimization, benchmarks]
    violation: "Invalid comparison — differences reflect constraints, not the model"
```

### Implementation Decision Logging — MANDATORY

> **Log file:** `docs/implementation_decisions.md`

During implementation, agents will encounter situations where the ISD or DVT specifications are insufficient, ambiguous, or incomplete. **All implementation decisions made to fill these gaps MUST be logged.**

#### When to Log

- Specification is **ambiguous** or has **multiple valid interpretations**
- Behavior must be **inferred** because it is not explicitly specified
- An **edge case** is discovered that is not covered
- A **conservative assumption** is made to proceed

#### Protocol by Phase

| Phase | Agents | Protocol |
|-------|--------|----------|
| **Phase 1** (parallel) | data-engineer, ml-architect, test-lead | Each logs with `[MOD-XXX]` prefix — atomic entries, no coordination |
| **Phase 2** (sequential) | Lead session / subagent | Sequential logging, no conflict |
| **Phase 3** (parallel) | bench-simple, bench-covariance, bench-factor | Each logs with `[MOD-0XX]` prefix — atomic entries |
| **Phase 4** (sequential) | Lead session | Logs + **consolidates all entries**, validates `provisional` → `validated` |

#### Orchestrator Responsibility

At each **phase transition**, the orchestrator (lead session) MUST:
1. Review all `provisional` entries from the completed phase
2. Validate correct decisions (`provisional` → `validated`)
3. Flag decisions requiring human supervisor discussion
4. Supersede revised decisions (`provisional` → `superseded` + new entry)

#### Why This Matters

- **Traceability**: Future debugging can trace unexpected behavior to documented decisions
- **Consistency**: Prevents different agents from making conflicting assumptions
- **Validation**: Human supervisor can review and correct before deployment
- **Learning**: Identifies gaps in the specification for future projects

### Symbol Glossary

| Symbol | Definition | Default Value |
|---------|-----------|-------------------|
| $n$ | Number of stocks in the universe | 1000 |
| $T$ | Window length (days) | 504 |
| $F$ | Number of features per timestep | 2 (return + realized vol) |
| $K$ | Latent capacity (ceiling) | 200 |
| $AU$ | Active dimensions (auto-pruning) | Dynamically determined |
| $AU_{\max}^{\text{stat}}$ | Statistical guard | $\lfloor\sqrt{2 \cdot N_{\text{obs}} / r_{\min}}\rfloor$ |
| $D$ | Number of elements per window | $T \times F$ |
| $\sigma^2$ | Observation noise (learned scalar) | init 1.0, clamp [1e-4, 10] |
| $\gamma$ | Crisis overweighting | 3.0 |
| $\lambda_{\text{co}}^{\max}$ | Max co-movement weight | 0.5 |
| $L$ | Encoder depth (residual blocks) | $\max(3, \lceil\log_2(T/63)\rceil + 2)$ |
| $C_L$ | Final layer width | $\max(384, \lceil 1.3 \times 2K \rceil)$ |
| $\lambda$ | Risk aversion | 1.0 |
| $\alpha$ | Entropy weight | Elbow of the variance-entropy frontier |

### Technical Dependencies

```
Python 3.11+
PyTorch >= 2.1
NumPy >= 1.24
SciPy >= 1.11
CVXPY >= 1.4 + MOSEK (or ECOS fallback)
pandas >= 2.0
scikit-learn >= 1.3 (Ledoit-Wolf)
statsmodels >= 0.14 (statistical tests)
pytest >= 7.0
```

---

## Topology and Modular Decomposition

### Functional Components and Coupling

```
Coupling matrix (degree 0-4):

                 data  vae   loss  train infer risk  optim wf    bench
data_pipeline    -     1     1     2     1     2     0     2     1
vae_architecture 1     -     3     3     2     0     0     0     0
loss_function    1     3     -     4     0     0     0     0     0
training         2     3     4     -     1     0     0     2     0
inference        1     2     0     1     -     3     0     1     0
risk_model       2     0     0     0     3     -     4     1     0
portfolio_optim  0     0     0     0     0     4     -     1     1
walk_forward     2     0     0     2     1     1     1     -     2
benchmarks       1     0     0     0     0     0     1     2     -
```

### Critical Coupling Points (degree ≥ 3)

| Pair | Degree | Invariant |
|-------|-------|-----------|
| vae_architecture ↔ loss_function | 3 (semantic) | σ² scalar (INV-002), D in the loss (INV-001) |
| loss_function ↔ training | 4 (mathematical) | P/F/A modes mutually exclusive (INV-006), validation ELBO excludes γ and λ_co (INV-011), curriculum batching linked to λ_co curriculum (INV-010) |
| vae_architecture ↔ training | 3 (semantic) | Reparameterization trick in forward(), architecture determines the training loop |
| inference ↔ risk_model | 3 (semantic) | B_A shape (INV-003), AU filtering convention (KL > 0.01 nats) |
| risk_model ↔ portfolio_optim | 4 (mathematical) | Dual rescaling (INV-004), principal factor basis rotation (INV-007), gradient H (INV-009) |

### Module Decomposition

| ID | Module | Components | Dependencies | Context Density | Mode |
|----|--------|-----------|-------------|------------------|------|
| MOD-001 | `data_pipeline` | Data loading, returns, universe, windowing, z-scoring, VIX, crisis labels | — | Medium | teammate |
| MOD-002 | `vae_architecture` | build_vae.py, encoder, decoder, sizing rules | MOD-001 (I: shapes) | High | teammate |
| MOD-003 | `test_infrastructure` | Synthetic data, assertion framework, test fixtures | — | Low | teammate |
| MOD-004 | `loss_function` | 3 modes, crisis weighting, co-movement loss, curriculum | MOD-002 (C) | Very high | lead_session |
| MOD-005 | `training` | Training loop, batching, optimizer, early stopping, LR scheduler | MOD-004 (C), MOD-001 (D) | Very high | lead_session |
| MOD-006 | `inference` | Composite profiles, aggregation, exposure matrix B | MOD-002 (C), MOD-005 (D: trained model) | Medium | subagent |
| MOD-007 | `risk_model` | AU filtering, rescaling, factor regression, Σ_z, D_ε, Σ_assets | MOD-006 (D) | High | subagent |
| MOD-008 | `portfolio_optimization` | Entropy, gradient, SCA, Armijo, cardinality, constraints | MOD-007 (D) | Very high | lead_session |
| MOD-009 | `walk_forward` | Fold scheduling, Phase A/B, HP selection, metrics, holdout | MOD-001–008 (D) | High | lead_session |
| MOD-010 | `bench_equal_weight` | 1/N benchmark | MOD-001 (D), shared infra | Low | teammate |
| MOD-011 | `bench_inverse_vol` | Inverse-volatility benchmark | MOD-001 (D) | Low | teammate |
| MOD-012 | `bench_min_variance` | Minimum-variance Ledoit-Wolf | MOD-001 (D) | Medium | teammate |
| MOD-013 | `bench_erc` | Equal Risk Contribution (Spinu) | MOD-001 (D), MOD-012 (C: LW) | Medium | teammate |
| MOD-014 | `bench_pca_factor_rp` | PCA factor risk parity (Bai-Ng IC₂ + SCA) | MOD-001 (D), MOD-008 (C: SCA solver) | High | teammate |
| MOD-015 | `bench_pca_vol` | PCA + realized vol feature | MOD-014 (C) | Low | teammate |
| MOD-016 | `integration` | E2E orchestration, reporting, statistical tests | All | High | lead_session |

### Dependency Graph (DAG)

```
Phase 1 (parallel — Agent Team "infrastructure")
  MOD-001 (data_pipeline)
  MOD-002 (vae_architecture)
  MOD-003 (test_infrastructure)
      ↓ synchronization

Phase 2 (sequential — Subagents builder-validator)
  MOD-004 (loss_function) ← MOD-002(C)
      ↓
  MOD-005 (training) ← MOD-004(C), MOD-001(D)
      ↓
  MOD-006 (inference) ← MOD-002(C), MOD-005(D)
      ↓
  MOD-007 (risk_model) ← MOD-006(D)
      ↓
  MOD-008 (portfolio_optimization) ← MOD-007(D)
      ↓ synchronization

Phase 3 (parallel — Agent Team "benchmarks")
  MOD-010 (bench_equal_weight) ← MOD-001(D)
  MOD-011 (bench_inverse_vol) ← MOD-001(D)
  MOD-012 (bench_min_variance) ← MOD-001(D)
  MOD-013 (bench_erc) ← MOD-012(C)
  MOD-014 (bench_pca_factor_rp) ← MOD-001(D), MOD-008(C: SCA)
  MOD-015 (bench_pca_vol) ← MOD-014(C)
      ↓ synchronization

Phase 4 (sequential — lead session)
  MOD-009 (walk_forward) ← MOD-001–008(D)
      ↓
  MOD-016 (integration) ← All
```

### Code Structure

```
latent_risk_factors/
├── CLAUDE.md
├── pyproject.toml
├── docs/
│   ├── isd/
│   │   ├── 00_global.md          ← this file
│   │   ├── 01_data_pipeline.md
│   │   ├── 02_vae_architecture.md
│   │   ├── ...
│   │   └── 16_integration.md
│   └── assertions/
│       └── contracts.yaml
├── src/
│   ├── __init__.py
│   ├── config.py               # Centralized configuration (dataclasses)
│   ├── data_pipeline/
│   │   ├── __init__.py
│   │   ├── data_loader.py      # Stock data loading (synthetic / EODHD)
│   │   ├── returns.py          # Log-return calculation
│   │   ├── universe.py         # Point-in-time universe construction
│   │   ├── windowing.py        # Sliding window + z-scoring
│   │   ├── crisis.py           # VIX threshold + crisis labeling
│   │   └── features.py         # Realized volatility + additional features
│   ├── vae/
│   │   ├── __init__.py
│   │   ├── build_vae.py        # Architecture construction (sizing rules)
│   │   ├── encoder.py          # InceptionHead + ResidualBlocks + projection
│   │   ├── decoder.py          # Transposed architecture
│   │   ├── model.py            # VAEModel (forward, reparameterize, loss modes)
│   │   └── loss.py             # Loss computation (3 modes, crisis weight, co-movement)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training loop
│   │   ├── batching.py         # Curriculum batching (synchronous + stratified / random)
│   │   ├── early_stopping.py   # Patience + best checkpoint
│   │   └── scheduler.py        # ReduceLROnPlateau wrapper
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── composite.py        # Sliding inference + aggregation → B
│   │   └── active_units.py     # AU measurement (KL > 0.01 nats)
│   ├── risk_model/
│   │   ├── __init__.py
│   │   ├── rescaling.py        # Dual rescaling (estimation + portfolio)
│   │   ├── factor_regression.py # Cross-sectional OLS → z_hat_t
│   │   ├── covariance.py       # Σ_z (Ledoit-Wolf), D_ε, Σ_assets assembly
│   │   └── conditioning.py     # Conditioning guard + ridge fallback
│   ├── portfolio/
│   │   ├── __init__.py
│   │   ├── entropy.py          # H(w), ∇H(w), principal factor rotation
│   │   ├── sca_solver.py       # SCA + Armijo + multi-start
│   │   ├── constraints.py      # P_conc, P_turn, hard caps
│   │   ├── cardinality.py      # Sequential entropy-aware elimination
│   │   └── frontier.py         # Variance-entropy frontier + α calibration
│   ├── walk_forward/
│   │   ├── __init__.py
│   │   ├── folds.py            # Fold scheduling
│   │   ├── phase_a.py          # HP selection
│   │   ├── phase_b.py          # Deployment run
│   │   ├── metrics.py          # All metrics (3 layers)
│   │   └── selection.py        # Scoring + selection criterion
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract benchmark class
│   │   ├── equal_weight.py
│   │   ├── inverse_vol.py
│   │   ├── min_variance.py
│   │   ├── erc.py              # Spinu solver
│   │   ├── pca_factor_rp.py    # PCA + Bai-Ng IC₂ + SCA
│   │   └── pca_vol.py          # PCA + realized vol variant
│   └── integration/
│       ├── __init__.py
│       ├── pipeline.py         # End-to-end orchestrator
│       ├── statistical_tests.py # Wilcoxon, Holm-Bonferroni
│       └── reporting.py        # Results compilation
├── tests/
│   ├── unit/
│   │   ├── test_data_pipeline.py
│   │   ├── test_vae_architecture.py
│   │   ├── test_loss_function.py
│   │   ├── ...
│   │   └── test_benchmarks.py
│   ├── integration/
│   │   ├── test_vae_training.py
│   │   ├── test_risk_pipeline.py
│   │   ├── test_portfolio_pipeline.py
│   │   └── test_walk_forward.py
│   └── fixtures/
│       ├── synthetic_data.py   # Deterministic synthetic data generator
│       └── known_solutions.py  # Known analytical solutions for verification
└── scripts/
    ├── run_walk_forward.py
    └── run_benchmarks.py
```

---

## Inter-Module Interface Assertions

### MOD-001 → MOD-004/005 (data_pipeline → loss/training)

```python
def verify_data_pipeline_output(windows, crisis_labels, returns_df):
    """Data pipeline output assertions."""
    N, T, F = windows.shape
    assert T == 504, f"Window length {T} != 504"
    assert F == 2, f"Feature count {F} != 2"
    assert windows.dtype == torch.float32

    # Z-scored per window: mean ≈ 0, std ≈ 1 per feature
    for i in range(min(100, N)):
        for f in range(F):
            feat = windows[i, :, f]
            assert abs(feat.mean()) < 1e-5, f"Window {i} feature {f} mean {feat.mean():.6f} != 0"
            assert abs(feat.std() - 1.0) < 1e-3, f"Window {i} feature {f} std {feat.std():.6f} != 1"

    # Crisis labels: fraction in [0, 1]
    assert crisis_labels.shape == (N,)
    assert (crisis_labels >= 0).all() and (crisis_labels <= 1).all()

    # Returns are log-returns
    # Verification: sum of log-returns ≈ log(P_T/P_0)
    # (checked on raw returns before windowing, in data_pipeline tests)
```

### MOD-002 → MOD-004 (vae_architecture → loss_function)

```python
def verify_vae_forward(model, sample_input):
    """VAE ↔ loss compatibility assertions."""
    x = sample_input  # (batch, T, F)
    x_hat, mu, log_var = model(x)

    assert x_hat.shape == x.shape, f"Reconstruction shape {x_hat.shape} != input {x.shape}"
    assert mu.shape[1] == model.K, f"mu dim {mu.shape[1]} != K={model.K}"
    assert log_var.shape == mu.shape

    # σ² is scalar
    assert model.log_sigma_sq.ndim == 0 or model.log_sigma_sq.numel() == 1
    sigma_sq = torch.exp(model.log_sigma_sq).item()
    assert 1e-4 <= sigma_sq <= 10, f"σ² = {sigma_sq} outside [1e-4, 10]"
```

### MOD-006 → MOD-007 (inference → risk_model)

```python
def verify_inference_output(B, AU, K, n_stocks, AU_max_stat):
    """Inference output assertions."""
    assert B.shape == (n_stocks, K), f"B shape {B.shape} != ({n_stocks}, {K})"
    assert AU == (np.abs(B[:, :K]).sum(axis=0) > 1e-6).sum()  # Approximation

    # Active dimensions count
    assert 0 < AU <= K
    assert AU <= AU_max_stat, f"AU={AU} > AU_max_stat={AU_max_stat}"

    # B_A has correct shape after filtering
    B_A = B[:, active_dims]
    assert B_A.shape == (n_stocks, AU)
```

### MOD-007 → MOD-008 (risk_model → portfolio_optimization)

```python
def verify_risk_model_output(Sigma_z, D_eps, B_A_port, eigenvalues, V, n_stocks, AU):
    """Risk model output assertions."""
    assert Sigma_z.shape == (AU, AU)
    assert D_eps.shape == (n_stocks,)
    assert B_A_port.shape == (n_stocks, AU)

    # Σ_z is symmetric positive semi-definite
    assert np.allclose(Sigma_z, Sigma_z.T, atol=1e-10)
    assert all(eigenvalues >= -1e-10), f"Negative eigenvalue: {eigenvalues.min()}"

    # D_ε has floor
    assert (D_eps >= 1e-6).all(), f"D_eps min {D_eps.min()} < 1e-6"

    # V is orthogonal
    assert np.allclose(V @ V.T, np.eye(AU), atol=1e-8)

    # Rotated B preserves asset covariance
    B_prime = B_A_port @ V
    Lambda = np.diag(eigenvalues)
    Sigma_from_rotation = B_prime @ Lambda @ B_prime.T + np.diag(D_eps)
    Sigma_from_original = B_A_port @ Sigma_z @ B_A_port.T + np.diag(D_eps)
    assert np.allclose(Sigma_from_rotation, Sigma_from_original, rtol=1e-6)
```

### MOD-008 output assertions (portfolio_optimization)

```python
def verify_portfolio_output(w, n_stocks, w_min=0.001, w_max=0.05):
    """Optimization output assertions."""
    assert w.shape == (n_stocks,)
    assert abs(w.sum() - 1.0) < 1e-8, f"Weights sum {w.sum():.8f} != 1"
    assert (w >= -1e-10).all(), "Negative weight found"

    # Semi-continuous: w_i = 0 or w_i >= w_min
    active = w > 1e-10
    assert (w[active] >= w_min - 1e-8).all(), f"Active weight below w_min"
    assert (w <= w_max + 1e-8).all(), f"Weight exceeds w_max"
```

---

## ISD Sections — Phase 1 (parallel: infrastructure)

---

### MOD-001 — data_pipeline

**Phase:** 1 | **Mode:** teammate | **Dependencies:** — | **Density:** medium
**Files:** `src/data_pipeline/*.py`, `tests/unit/test_data_pipeline.py`

#### Objective

Implement end-to-end financial data preparation: stock data loading (synthetic for development, EODHD for production) with survivorship-bias-free handling, log-return computation, point-in-time universe construction, sliding window creation with per-window z-scoring, crisis labeling via VIX threshold, and trailing realized volatility computation. This module feeds every downstream component — errors here silently corrupt the entire pipeline.

#### Outputs

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `windows` | `torch.Tensor` | `(N, T, F)` | Z-scored sliding windows (T=504, F=2) |
| `window_metadata` | `pd.DataFrame` | `(N, 3)` | stock_id, start_date, end_date per window |
| `crisis_fractions` | `torch.Tensor` | `(N,)` | $f_c^{(w)} \in [0, 1]$ per window |
| `returns_df` | `pd.DataFrame` | `(n_dates, n_stocks)` | Raw log-returns (for co-movement loss and risk model) |
| `trailing_vol` | `pd.DataFrame` | `(n_dates, n_stocks)` | 252-day trailing annualized vol |
| `universe_snapshots` | `dict` | `date → list[str]` | Point-in-time universe at each date |

#### Sub-task 1: Data loading (data_loader.py)

**Development strategy — two phases:**

1. **Phase A (development):** Use **synthetic CSV data** to validate the entire pipeline
   end-to-end without requiring any external data subscription. A helper function
   generates a CSV file with the exact same schema, allowing all downstream code
   to be tested for correctness, shapes, and invariants.
2. **Phase B (production):** Once the pipeline is bug-free on synthetic data, switch to
   **EODHD** (End-Of-Day Historical Data) as the real data source. EODHD provides
   26,000+ US equities including delisted stocks, with history from ~2000 onward.
   Subscription: "All-In-One" plan (~100 EUR/month). API or bulk CSV download.

**Backtest period adjustment:** With EODHD data starting ~2000, the backtest period
becomes **2000–2025** (25 years) instead of 1993–2023 (30 years). Impact:
- Walk-forward folds: ~21 (vs ~34), still statistically sufficient
- AU_max_stat ≈ 77 (vs 85), adequate capacity check
- All pipeline logic, fold scheduling, and metrics remain identical

---

**Data schema — Core columns (required for F=2 baseline):**

The baseline model uses F=2 features per time step: **log-returns** and **21-day rolling
realized volatility** (DVT Section 4.2). The following columns are always required,
including infrastructure columns needed for universe construction (DVT Section 5.1,
Universe parameters):

| Column | Type | Description | Used by | Example |
|--------|------|-------------|---------|---------|
| `permno` | int | Unique stock identifier | All | `10001` |
| `date` | str (YYYY-MM-DD) | Trading date | All | `2005-03-15` |
| `adj_price` | float | Split- and dividend-adjusted closing price (> 0) | Returns, vol | `42.57` |
| `volume` | int | Daily trading volume in shares | Universe (ADV filter) | `1_500_000` |
| `exchange_code` | int | 1 = NYSE, 2 = AMEX, 3 = NASDAQ | Shumway imputation | `1` |
| `share_code` | int | Share type code (10/11 = common equity) | Universe (equity filter) | `11` |
| `market_cap` | float | Float-adjusted market cap in USD | Universe (top-n ranking) | `2.5e9` |
| `delisting_return` | float or NaN | Return on delisting day (NaN if unknown) | Delisting handling | `-0.30` |

Notes:
- `exchange_code` is used by the Shumway (1997) delisting return imputation:
  -30% for NYSE/AMEX (codes 1, 2), -55% for NASDAQ (code 3). No `delisting_code`
  column is needed — the strategy never uses it.
- `adj_price` is the sole input for computing log-returns (CONV-01) and
  21-day rolling realized volatility (DVT Section 4.2).
- `market_cap` is used for universe construction (top-n ranking).
- `volume` is an **infrastructure column** for the ADV liquidity filter in universe
  construction (DVT Section 4.2: ADV ≥ $2M over trailing 3 months). It is NOT used
  as a VAE input feature in the F=2 baseline — the encoder receives only (T, 2) =
  [returns, realized_vol]. Dollar volume is computed as `adj_price × volume`.

**Data schema — Extended columns (required for F > 2 iterations):**

When extending the model beyond F=2 (DVT Section 6.5, Iteration 4), additional
features per time step are introduced. These columns become required.
Note: `volume` is already in the core schema (used for ADV filtering); for F > 2
it is additionally used as a z-scored VAE input feature.

| Column | Type | Description | Feature derived |
|--------|------|-------------|-----------------|
| `high` | float | Intraday high price | Intraday range (high - low) / close |
| `low` | float | Intraday low price | Intraday range (high - low) / close |
| `sector` | str | Sector classification (GICS or SIC) | Sector-relative return deviation |

Additional derived feature (no extra column needed):
- **Realized skewness**: computed from the existing `adj_price` column
  (rolling window of raw returns).

The encoder Inception head and decoder output layer adjust automatically for F > 2
(~0.1% parameter change); all downstream pipeline components are unchanged.

---

**EODHD column mapping (Phase B):**

| EODHD field | Internal column | Notes |
|-------------|-----------------|-------|
| `Code` / `Ticker` | `permno` | Map to integer ID via lookup table |
| `Date` | `date` | Already YYYY-MM-DD |
| `Adjusted_close` | `adj_price` | Split+dividend adjusted |
| `Volume` | `volume` | Always required (core: ADV filter; F>2: also VAE feature) |
| `High` | `high` | Extended only (F > 2) |
| `Low` | `low` | Extended only (F > 2) |
| `Exchange` | `exchange_code` | Map: NYSE→1, AMEX→2, NASDAQ→3 |
| — | `share_code` | Filter common equity via security type metadata |
| — | `market_cap` | Compute from Adjusted_close × shares outstanding (fundamentals API) |
| — | `delisting_return` | Compute from last available price vs prior close; impute Shumway if missing |

---

**Synthetic data generator** (in `data_loader.py` or called from `tests/fixtures/synthetic_data.py`):

```python
def generate_synthetic_csv(
    output_path: str,
    n_stocks: int = 200,
    start_date: str = "2000-01-03",
    end_date: str = "2025-12-31",
    n_delistings: int = 20,
    seed: int = 42,
    include_extended: bool = False,
) -> str:
    """
    Generate a synthetic stock data CSV file with realistic properties.

    Price dynamics: geometric Brownian motion per stock.
      P_{t+1} = P_t × exp(μ_i + σ_i × ε_t),  ε ~ N(0,1)
      μ_i ~ U(0.0001, 0.0005) (daily drift),
      σ_i ~ U(0.005, 0.03)    (daily vol, annualized ~8%-48%)
      P_0,i ~ U(10, 200)

    Core columns (always generated):
      permno, date, adj_price, volume, exchange_code, share_code,
      market_cap, delisting_return.

    Extended columns (if include_extended=True):
      high, low, sector.

    Market cap: P × shares_outstanding, shares ~ U(10M, 500M).
    Volume: shares_traded ~ LogNormal(μ_vol_i, 1.0), with
      μ_vol_i = log(shares_outstanding_i × 0.005) (≈0.5% daily turnover).
      Correlated with market cap (larger stocks → higher volume).
    Exchange codes: random assignment (1/2/3) with realistic proportions
      (60% NYSE, 10% AMEX, 30% NASDAQ).
    Share codes: all 10 or 11 (common equity).
    Sectors: random assignment from 11 GICS sectors.

    Delistings: n_delistings stocks are delisted at random dates
      in the second half of history. Delisting return = NaN for ~50%,
      imputed value for the rest.

    Missing data: ~2% of stock-days have NaN adj_price (random gaps).

    Returns: path to the generated CSV file.
    """
```

**Loader functions:**

```python
def load_stock_data(
    data_path: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Load daily stock data from CSV or Parquet file.
    Accepts both synthetic data (Phase A) and EODHD data (Phase B) — same schema.

    Must include delisted stocks with full pre-delisting history.

    Returns: DataFrame with core columns [permno, date, adj_price,
             volume, exchange_code, share_code, market_cap,
             delisting_return].
             Extended columns [high, low, sector] present if available.
    Sorted by (permno, date). date is pd.Timestamp.
    """
```

#### Sub-task 2: Log-return computation (returns.py)

```python
def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    r_{i,t} = ln(P^adj_{i,t} / P^adj_{i,t-1})

    CONV-01: Log returns, NEVER arithmetic.

    Missing value handling:
    - Isolated gaps ≤ 5 consecutive days: forward-fill price (= 0 return)
    - Gaps > 5 consecutive days: NaN (excluded from windowing)

    Delisting returns:
    - If no return available at delisting: impute -30% (NYSE/AMEX)
      or -55% (Nasdaq) per Shumway (1997) convention.

    Returns: DataFrame (n_dates, n_stocks) of log-returns
    """
```

#### Sub-task 3: Point-in-time universe construction (universe.py)

```python
def construct_universe(
    stock_data: pd.DataFrame,
    date: pd.Timestamp,
    n_max: int = 1000,
    cap_entry: float = 500e6,
    cap_exit: float = 400e6,
    adv_min: float = 2e6,
    min_listing_days: int = 504,
) -> list[str]:
    """
    Reconstruct the universe as it existed at each historical date.
    CONV-10: Point-in-time, no future data.

    Eligibility criteria:
    1. Float-adjusted market cap ≥ cap_entry (entry) / cap_exit (exit buffer)
    2. ADV ≥ adv_min over trailing 3 months (filter, not ranking).
       ADV = mean(adj_price × volume) over trailing 63 trading days.
       Requires `volume` column in stock_data (core schema).
    3. Continuous listing ≥ min_listing_days (= T = 504)
    4. Common equities only (exclude ETFs, ADRs, REITs, preferred, warrants, SPACs)
    5. NYSE + NASDAQ + AMEX

    If more than n_max stocks qualify: top n_max by float-adjusted market cap.

    Training universe for a fold = union of all U_t' for t' in training period
    (includes since-delisted stocks).
    """

def handle_delisting(
    universe: list[str],
    w: np.ndarray,
    returns_oos: pd.DataFrame,
    H_last: float,
    alpha_trigger: float = 0.90,
) -> tuple[np.ndarray, bool]:
    """
    Between reconstitutions:
    - Delisted positions: liquidated at last available price (or imputed return).
      Freed capital held as cash.
    - Exceptional rebalancing triggered if:
      H(w_post_delisting) < alpha_trigger × H(w_last_rebalancing)
    """
```

#### Sub-task 4: Sliding windows + z-scoring (windowing.py)

```python
def create_windows(
    returns_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    stock_ids: list[str],
    T: int = 504,
    stride: int = 1,
    sigma_min: float = 1e-8,
    max_zero_frac: float = 0.20,
) -> tuple[torch.Tensor, pd.DataFrame]:
    """
    For each stock i and window ending at date t:
    1. Extract raw returns r_{i, t-T+1:t} and vol v_{i, t-T+1:t}
    2. Exclude window if > max_zero_frac of days have zero return (suspension)
    3. Z-score returns PER WINDOW: r̃ = (r - μ_r) / max(σ_r, σ_min)
    4. Z-score volatility PER WINDOW: ṽ = (v - μ_v) / max(σ_v, σ_min)
    5. Stack → tensor (T, F=2): [r̃, ṽ]

    CONV-02: Z-score per-window, per-feature.
    CONV-05: Window shape (batch, T, F) — time is dim 1, features dim 2.

    Returns: (windows tensor (N, T, F), metadata DataFrame)
    """
```

#### Sub-task 5: Crisis labeling (crisis.py)

```python
def compute_crisis_labels(
    vix_data: pd.Series,
    window_metadata: pd.DataFrame,
    training_end_date: pd.Timestamp,
    percentile: float = 80.0,
) -> torch.Tensor:
    """
    VIX threshold computed on EXPANDING window of training period only.
    CONV-10 / INV-005: No look-ahead.

    τ_VIX = Percentile_80(VIX_{t0:t_train})  [recalculated once per fold]

    Pre-1990: use VXO (1986-1990) or realized vol proxy
    (annualized 21-day rolling std of S&P 500 returns,
    percentile-matched to VIX over common history).

    For each window w:
      f_c(w) = fraction of days where VIX > τ_VIX

    Returns: crisis_fractions tensor (N,) with values in [0, 1]
    """
```

#### Sub-task 6: Trailing realized volatility (features.py)

```python
def compute_trailing_volatility(
    returns_df: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    """
    σ_{i,t} = annualized std of r_{i, t-251:t} (trailing 252 days).

    Warm-up: the first 252 days of each stock have NaN vol.
    The 21-day rolling vol for the VAE input requires an extra
    21 days of history BEFORE the encoder window start.

    Returns: DataFrame (n_dates, n_stocks) of trailing annualized vol.
    """

def compute_rolling_realized_vol(
    returns_df: pd.DataFrame,
    rolling_window: int = 21,
) -> pd.DataFrame:
    """
    v_{i,τ} = std(r_{i, τ-20:τ}) — 21-day rolling std (not annualized).
    Used as the second input feature (F=2) for the VAE.

    Warm-up requirement: needs 21 pre-window returns.

    Returns: DataFrame (n_dates, n_stocks) of 21-day rolling vol.
    """
```

#### Applicable Invariants

- **CONV-01:** Log returns `r_t = ln(P^adj_t / P^adj_{t-1})`, never arithmetic.
- **CONV-02:** Z-score per-window, per-feature (mean 0, std 1).
- **CONV-05:** Window shape `(batch, T, F)`.
- **CONV-10:** Point-in-time universe reconstruction.
- **INV-005:** No look-ahead. VIX threshold on expanding training window only.

#### Known Pitfalls

- **DO NOT** use arithmetic returns — log returns are required for additivity and correct z-scoring behavior.
- **DO NOT** compute z-score on expanding or full-history window — per-window only (DVT Section 4.2).
- **DO NOT** use future data for VIX threshold — expanding window up to `training_end_date` only.
- **DO NOT** forget the sigma_min clamp (1e-8) — near-zero volatility windows produce NaN after division.
- **DO NOT** include delisted stocks without their pre-delisting history — this introduces survivorship bias.
- **DO NOT** forget the 21-day warm-up for realized volatility — the rolling vol at the start of a window must be based on pre-window returns.
- **DO NOT** forget the Shumway delisting return imputation (-30% NYSE/AMEX, -55% Nasdaq).

#### Required Tests

1. `test_log_returns_vs_arithmetic`: verify `ln(P_t/P_{t-1})` not `(P_t - P_{t-1})/P_{t-1}`
2. `test_zscore_per_window`: mean ≈ 0, std ≈ 1 per feature per window (tolerance 1e-5, 1e-3)
3. `test_window_shape`: output shape `(N, T, F)` with T=504, F=2
4. `test_sigma_min_clamp`: window with near-zero std does not produce NaN
5. `test_zero_return_exclusion`: window with >20% zero returns is excluded
6. `test_missing_value_handling`: forward-fill ≤5 days, NaN >5 days
7. `test_universe_point_in_time`: stock not in universe before listing, removed after delisting
8. `test_universe_cap_filter`: stocks below cap_entry excluded, above cap_exit retained
9. `test_vix_threshold_no_lookahead`: threshold uses only data up to training_end_date
10. `test_crisis_fraction_range`: all f_c values in [0, 1]
11. `test_trailing_vol_warmup`: first 252 days are NaN
12. `test_delisting_return_imputed`: delisting without data gets -30% or -55%

---

### MOD-002 — vae_architecture

**Phase:** 1 | **Mode:** teammate | **Dependencies:** MOD-001 (I: window shapes) | **Density:** high
**Files:** `src/vae/build_vae.py`, `src/vae/encoder.py`, `src/vae/decoder.py`, `src/vae/model.py`, `tests/unit/test_vae_architecture.py`

#### Objective

Implement the complete VAE architecture: InceptionHead + residual body encoder, transposed residual decoder, VAEModel wrapper (forward, encode, reparameterize), and the `build_vae()` factory function that derives the entire architecture from 5 variable parameters $(n, T, T_{\text{année}}, F, K)$ with capacity-data constraint verification.

#### Sub-task 1: Sizing rules (build_vae.py)

```python
def compute_depth(T: int) -> int:
    """Rule 1 — L(T) = max(3, ceil(log2(T / k_max)) + 2), k_max = 63."""

def compute_final_width(K: int) -> int:
    """Rule 2 — C_L(K) = round_16(max(C_min=384, ceil(1.3 × 2K)))."""

def compute_channel_progression(L: int, C_L: int) -> list[int]:
    """Rule 3 — Geometric interpolation C_head(=144) → C_L, rounded to multiples of 16."""

def compute_temporal_sizes(T: int, L: int) -> list[int]:
    """Temporal dimension after each stride-2 block: t → (t-1)//2 + 1."""
```

**Fixed hyperparameters (all from DVT Section 4.3):**

| Symbol | Value | Component |
|--------|-------|-----------|
| `K_HEAD` | (5, 21, 63) | Inception head kernels |
| `C_BRANCH` | 48 | Filters per Inception branch → C_HEAD = 144 |
| `K_BODY` | 7 | Residual body kernel |
| `STRIDE` | 2 | Per-block downsampling |
| `ALPHA_PROJ` | 1.3 | Projection ratio |
| `C_MIN` | 384 | Minimum final layer width |
| `DROPOUT` | 0.1 | Dropout rate |
| Activation | GELU | All layers |
| Normalization | BatchNorm1d | After every conv |

#### Sub-task 2: Encoder (encoder.py)

```python
class InceptionHead(nn.Module):
    """
    Three parallel Conv1d branches with kernels (5, 21, 63),
    each producing C_BRANCH=48 channels → concatenated to C_HEAD=144.
    Each branch: Conv1d + BatchNorm1d + GELU.
    Padding: k // 2 (same-length output).
    """

class ResBlock(nn.Module):
    """
    Two convolutions (k=7) with BatchNorm + GELU, skip connection (1×1 conv).
    First conv: stride=2 (downsamples temporal dimension).
    Second conv: stride=1.
    Skip: 1×1 conv with stride=2 (always active since stride changes dimensions).
    Dropout(0.1) after activation.
    """

class Encoder(nn.Module):
    """
    InceptionHead(F) → L × ResBlock(c_in, c_out) → AdaptiveAvgPool1d(1)
      → Linear(C_L, K) [mu] + Linear(C_L, K) [log_var]

    Input: (B, F, T) — channels-first for Conv1d.
    Output: mu (B, K), log_var (B, K).

    NOTE: Input arrives as (B, T, F) from data_pipeline (CONV-05).
    Transpose to (B, F, T) before Conv1d in model.forward().
    """
```

#### Sub-task 3: Decoder (decoder.py)

```python
class TransposeResBlock(nn.Module):
    """
    ConvTranspose1d (stride=2, output_padding=1) + Conv1d (stride=1).
    Skip: ConvTranspose1d (1×1, stride=2, output_padding=1).
    Each with BatchNorm + GELU + Dropout.
    Doubles temporal dimension at each block.
    """

class Decoder(nn.Module):
    """
    Linear(K, C_L × T_comp) → reshape (B, C_L, T_comp)
      → L × TransposeResBlock (reversed channel progression)
      → trim/pad to target T → Conv1d(C_HEAD, F, kernel_size=1)

    T_comp = encoder's last temporal size (e.g. 16 for T=504, L=5).

    Temporal mismatch: decoder produces 16→32→64→128→256→512
    vs encoder 504→252→126→63→32→16. Final trim/pad to exact T.

    Output: (B, F, T) — channels-first.
    """
```

#### Sub-task 4: VAEModel (model.py)

```python
class VAEModel(nn.Module):
    """
    Attributes:
      encoder: Encoder
      decoder: Decoder
      K: int — latent dimension
      log_sigma_sq: nn.Parameter — scalar (INV-002), init 0.0 (σ²=1.0)
      learn_obs_var: bool — True for Mode P/A, False for Mode F

    Key properties:
      obs_var: σ² = clamp(exp(log_sigma_sq), 1e-4, 10)

    Methods:
      forward(x) → (x_hat, mu, log_var)
        x: (B, T, F) → transpose to (B, F, T) for encoder
        z = mu + exp(0.5 * log_var) * ε, ε ~ N(0,1)  [reparameterization]
        x_hat: (B, F, T) → transpose to (B, T, F) for output

      encode(x) → mu
        Deterministic. For inference only. model.eval() + no_grad().
        Returns mu (B, K) — no sampling, no reparameterization.
    """
```

**CRITICAL:** `encode()` returns mu directly (deterministic). `forward()` uses reparameterization (stochastic, for training). MOD-006 inference MUST use `encode()`.

#### Sub-task 5: Factory function (build_vae.py)

```python
def build_vae(
    n: int, T: int, T_annee: int, F: int, K: int,
    s_train: int = 1, r_max: float = 5.0,
    beta: float = 1.0, learn_obs_var: bool = True,
) -> tuple[VAEModel, dict]:
    """
    1. T_hist = T_annee × 252
    2. N = n × (T_hist − T + 1)         [capacity, always at s=1]
    3. N_train = n × floor((T_hist − T) / s_train) + n
    4. L = compute_depth(T)
    5. C_L = compute_final_width(K)
    6. channels = compute_channel_progression(L, C_L)
    7. P_total = P_enc + P_dec (analytical count)
    8. r = P_total / N
    9. Raise ValueError if r > r_max
    10. Instantiate Encoder, Decoder, VAEModel

    Mode selection via (learn_obs_var, beta):
      Mode P: learn_obs_var=True,  beta=1.0
      Mode F: learn_obs_var=False, beta=<1.0 (external β_t)
      Mode A: learn_obs_var=True,  beta=>1.0

    Returns: (model, info_dict)
    info_dict keys: L, channels, temporal_sizes, C_L, T_compressed,
                    P_enc, P_dec, P_total, N, N_train, s_train, r, r_max, T_hist
    """
```

#### Sub-task 6: Analytical parameter counting

```python
def count_encoder_params(F: int, K: int, channels: list[int]) -> int:
    """
    Inception head: Σ_branch (F × C_BRANCH × k + C_BRANCH + 2×C_BRANCH)
    Residual body: Σ_block (c_in×c_out×K_BODY + c_out + 2×c_out) × 2 convs
                   + (c_in×c_out + c_out + 2×c_out) skip
    Projection: C_L×K + K (mu) + C_L×K + K (log_var)
    """

def count_decoder_params(F: int, K: int, channels: list[int], T_compressed: int) -> int:
    """
    Initial projection: K × (C_L × T_comp) + (C_L × T_comp)
    Transposed body: same block structure, reversed channels
    Output head: C_HEAD × F + F
    """
```

#### Applicable Invariants

- **INV-002:** `log_sigma_sq` is scalar (`ndim == 0` or `numel() == 1`). Clamped to `[1e-4, 10]`.
- **CONV-05:** Input `(B, T, F)` transposed to `(B, F, T)` for Conv1d, transposed back for output.
- **CONV-06:** σ² is a single scalar for the entire model.

#### Known Pitfalls

- **DO NOT** forget to transpose `(B, T, F) → (B, F, T)` before Conv1d and back after decoder — PyTorch Conv1d expects channels-first.
- **DO NOT** make `log_sigma_sq` a vector (per-feature or per-dimension) — it must be a scalar (INV-002).
- **DO NOT** use `forward()` for inference — use `encode()` which returns mu deterministically without reparameterization noise.
- **DO NOT** forget `output_padding=1` in TransposeResBlock — without it, stride-2 transposed convolution does not double the temporal dimension correctly.
- **DO NOT** forget the final trim/pad in the decoder to ensure output length matches T exactly.
- **DO NOT** set `r_max` lower than the computed ratio without checking the feasibility table (DVT Section 4.3).

#### Required Tests

1. `test_sizing_rule_depth`: L=5 for T=504, L=4 for T=252, L=6 for T=756
2. `test_sizing_rule_width`: C_L=384 for K≤147, C_L=round_16(1.3×2K) for K>147
3. `test_channel_progression_monotonic`: channels strictly increasing
4. `test_capacity_constraint`: ValueError raised when r > r_max
5. `test_capacity_constraint_table`: known (K, T) combinations match DVT table
6. `test_encoder_output_shape`: mu and log_var both (B, K)
7. `test_decoder_output_shape`: x_hat same shape as input (B, T, F)
8. `test_forward_roundtrip_shape`: forward(x) returns (x_hat, mu, log_var) with correct shapes
9. `test_encode_deterministic`: two encode() calls on same input produce identical mu
10. `test_forward_stochastic`: two forward() calls produce different x_hat (reparameterization)
11. `test_log_sigma_sq_scalar`: `model.log_sigma_sq.ndim == 0`
12. `test_log_sigma_sq_clamped`: obs_var in [1e-4, 10] after extreme gradient steps
13. `test_param_count_analytical_vs_pytorch`: `count_encoder_params()` matches `sum(p.numel() for p in encoder.parameters())`
14. `test_transpose_convention`: input (B, T, F) → encoder sees (B, F, T) → output (B, T, F)
15. `test_build_vae_modes`: Mode P (learn=True, β=1), Mode F (learn=False, β<1), Mode A (learn=True, β>1)

---

### MOD-003 — test_infrastructure

**Phase:** 1 | **Mode:** teammate | **Dependencies:** — | **Density:** low
**Files:** `tests/fixtures/synthetic_data.py`, `tests/fixtures/known_solutions.py`, `tests/__init__.py`

#### Objective

Provide deterministic synthetic data generators and known analytical solutions that all unit and integration tests rely on. Every test in the project should be reproducible with fixed seeds and independent of external data.

#### Sub-task 1: Synthetic data generator (synthetic_data.py)

```python
def generate_synthetic_returns(
    n_stocks: int = 50,
    n_days: int = 2520,       # 10 years
    n_factors: int = 5,
    noise_std: float = 0.02,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Generate returns from a known factor model:
      r_{i,t} = B_true[i, :] @ z_true[t, :] + ε_{i,t}

    Where:
    - B_true (n_stocks, n_factors): fixed loadings from N(0, 1)
    - z_true (n_days, n_factors): factor returns from N(0, 0.01)
    - ε (n_days, n_stocks): idiosyncratic noise ~ N(0, noise_std)

    Returns: (returns_df, B_true, z_true) — for verification of
    factor regression recovery (z_hat ≈ z_true).

    Deterministic: np.random.seed(seed) at start.
    """

def generate_synthetic_windows(
    n_windows: int = 1000,
    T: int = 504,
    F: int = 2,
    K_true: int = 10,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate pre-z-scored windows with known latent structure.
    CONV-02: each window has mean ≈ 0, std ≈ 1 per feature.

    Returns: (windows (N, T, F), true_latents (N, K_true))
    """

def generate_crisis_labels(
    n_windows: int = 1000,
    crisis_fraction: float = 0.20,
    seed: int = 42,
) -> torch.Tensor:
    """
    Generate realistic crisis fractions:
    - ~80% of windows with f_c ≈ 0 (calm)
    - ~20% of windows with f_c ∈ [0.5, 1.0] (crisis)
    - Bimodal distribution (VIX is autocorrelated)

    Returns: crisis_fractions (N,) in [0, 1]
    """

def generate_synthetic_universe(
    n_stocks: int = 50,
    n_days: int = 2520,
    n_delistings: int = 5,
    seed: int = 42,
) -> dict:
    """
    Generate point-in-time universe with delistings.

    Returns: dict with prices, market_caps, volumes, exchange_codes,
             delisting_dates, universe_snapshots
    """
```

#### Sub-task 2: Known analytical solutions (known_solutions.py)

```python
def diagonal_covariance_solution() -> dict:
    """
    Σ = diag(σ²_1, ..., σ²_n), B = I (identity).
    Known minimum-variance: w_i ∝ 1/σ²_i.
    Known ERC: w_i ∝ 1/σ_i.
    Known entropy at ERC: H = ln(n) (maximum, all contributions equal).
    """

def two_factor_solution() -> dict:
    """
    n=4 stocks, 2 factors, known B and Σ_z.
    Analytical portfolio that maximizes entropy.
    Useful for verifying the SCA solver converges to the right solution.
    """

def entropy_gradient_verification() -> dict:
    """
    At maximum entropy (ĉ'_k = 1/AU ∀k):
    - H = ln(AU)
    - ∇H = 0 (gradient vanishes)
    Provides w, B_prime, eigenvalues for this test case.
    """

def factor_regression_identity() -> dict:
    """
    B = I (identity), r = z (no noise).
    z_hat = (B^T B)^{-1} B^T r = r → z_hat ≈ z.
    """

def rescaling_verification() -> dict:
    """
    Known σ_{i,t}, σ_bar_t → expected B_A_estimation and B_A_portfolio
    with winsorization at [P5, P95].
    """
```

#### Sub-task 3: Test configuration and helpers

```python
# Shared constants for all tests
TEST_SEED = 42
TEST_N_STOCKS = 50
TEST_N_DAYS = 2520  # 10 years
TEST_T = 504
TEST_F = 2
TEST_K = 20        # small K for fast tests
TEST_BATCH_SIZE = 32

def set_deterministic(seed: int = TEST_SEED):
    """
    Set all random seeds for reproducibility:
    np.random.seed, torch.manual_seed, torch.cuda.manual_seed_all,
    torch.backends.cudnn.deterministic = True
    """
```

#### Applicable Invariants

- All synthetic data must be deterministic (fixed seeds).
- Known solutions must be analytically verified (not just numerically computed).
- Test fixtures must be independent of external data (CRSP, VIX files).

#### Known Pitfalls

- **DO NOT** use random data without a fixed seed — tests become non-reproducible.
- **DO NOT** generate synthetic windows that are NOT z-scored — downstream code expects CONV-02.
- **DO NOT** make known solutions depend on floating point precision beyond 1e-10 — use tolerance in assertions.
- **DO NOT** generate factor models with collinear factors — the regression tests become ill-conditioned.

#### Required Tests

1. `test_synthetic_returns_deterministic`: same seed → identical output
2. `test_synthetic_returns_shape`: correct shapes (n_days, n_stocks) and (n_stocks, n_factors)
3. `test_synthetic_windows_zscore`: mean ≈ 0, std ≈ 1 per feature per window
4. `test_diagonal_solution_analytical`: min-var weights match 1/σ²_i (normalized)
5. `test_entropy_at_maximum`: H = ln(AU) when all contributions equal

---

---

## ISD Sections — Phase 2 (sequential)

---

### MOD-004 — loss_function

**Phase:** 2 | **Mode:** lead_session | **Dependencies:** MOD-002 (C: VAEModel) | **Density:** very high
**Files:** `src/vae/loss.py`, `tests/unit/test_loss_function.py`

#### Objective

Implement the three VAE loss computation modes (P/F/A), the crisis weighting mechanism, the co-movement loss, and the curriculum scheduling. This module is the mathematical core of the VAE — any error here silently invalidates training.

#### Inputs

| Name | Type | Shape | Description |
|-----|------|-------|-------------|
| `x` | `torch.Tensor` | `(B, T, F)` | Z-scored input windows |
| `x_hat` | `torch.Tensor` | `(B, T, F)` | Reconstruction |
| `mu` | `torch.Tensor` | `(B, K)` | Encoder mean |
| `log_var` | `torch.Tensor` | `(B, K)` | Encoder log-variance |
| `log_sigma_sq` | `torch.Tensor` | scalar | Model log σ² |
| `crisis_fractions` | `torch.Tensor` | `(B,)` | $f_c^{(w)}$ per window |
| `epoch` | int | — | Current epoch (for curriculum) |
| `mode` | str | — | "P", "F", or "A" |

#### Outputs

| Name | Type | Description |
|-----|------|-------------|
| `total_loss` | `torch.Tensor` | scalar, total loss for backprop |
| `loss_components` | dict | recon, kl, co_mov, sigma_sq (for monitoring) |

#### Sub-task 1: Crisis-weighted reconstruction loss

```
MSE(w) = (1/(T×F)) · Σ_{t,f} (x_{w,t,f} - x̂_{w,t,f})²    [per-element mean]

γ_eff(w) = 1 + f_c(w) · (γ - 1)                             [continuous crisis weight]

L_recon_weighted = (1/|B|) · Σ_w γ_eff(w) · MSE(w)          [batch mean, weighted]
```

**WARNING:** MSE is a per-element mean (divided by T×F), not a sum. The factor D = T×F is applied separately as a multiplicative coefficient.

#### Sub-task 2: KL Loss

$$\mathcal{L}_{\text{KL}} = \frac{1}{N_{\text{batch}}} \sum_{i=1}^{N_{\text{batch}}} \frac{1}{2} \sum_{k=1}^{K} \left( \mu_{ik}^2 + \exp(\log\_var_{ik}) - \log\_var_{ik} - 1 \right)$$

The KL is averaged over the batch (dimension 0) and summed over the latent dimensions (dimension 1). The 1/2 is outside the sum over k.

#### Sub-task 3: Assembly — Mode P (primary)

$$\mathcal{L} = \frac{D}{2\sigma^2} \cdot \mathcal{L}_{\text{recon, weighted}} + \frac{D}{2}\ln\sigma^2 + \mathcal{L}_{\text{KL}} + \lambda_{\text{co}}(t) \cdot \mathcal{L}_{\text{co-mov}}$$

Where:
- $D = T \times F$ (for T=504, F=2: D=1008)
- $\sigma^2 = \text{clamp}(\exp(\log\_sigma\_sq), 10^{-4}, 10)$
- $\beta = 1$ fixed (DO NOT modify)
- $\lambda_{\text{co}}(t)$ follows the curriculum (sub-task 5)

The term $(D/2)\ln\sigma^2$ is the Gaussian log-normalization. It penalizes high $\sigma^2$ (the model cannot "cheat" by increasing σ² to reduce the reconstruction cost).

#### Sub-task 4: Assembly — Mode F (fallback)

$$\mathcal{L}_t = \frac{D}{2} \cdot \mathcal{L}_{\text{recon, weighted}} + \beta_t \cdot \mathcal{L}_{\text{KL}} + \lambda_{\text{co}}(t) \cdot \mathcal{L}_{\text{co-mov}}$$

Where:
- $\sigma^2 = 1$ FROZEN (no gradient on log_sigma_sq)
- $D/2$ is RETAINED (dimensional scaling)
- $\beta_t = \min(1, t / T_{\text{warmup}})$, linear ramp
- $T_{\text{warmup}}$: 10-30% of total epochs

**CRUCIAL:** DO NOT remove D/2. Without D/2, MSE ≈ 0.3-0.7 would be dominated by KL ≈ 60-120 nats → immediate posterior collapse.

#### Sub-task 5: Assembly — Mode A (advanced)

$$\mathcal{L} = \frac{D}{2\sigma^2} \cdot \mathcal{L}_{\text{recon, weighted}} + \frac{D}{2}\ln\sigma^2 + \beta \cdot \mathcal{L}_{\text{KL}} + \lambda_{\text{co}}(t) \cdot \mathcal{L}_{\text{co-mov}}$$

Identical to Mode P but with $\beta > 1$ (fixed, not annealed). Range: β ∈ [1.0, 4.0].

#### Sub-task 6: Co-movement loss

```python
def compute_co_movement_loss(
    mu: torch.Tensor,           # (B, K) latent means
    window_metadata: ...,       # stock_ids + dates for pairing
    returns_data: ...,          # raw returns for Spearman computation
    max_pairs: int = 2048,
    delta_sync: int = 21,       # max date gap for synchronization
) -> torch.Tensor:
    """
    For each eligible pair (i, j) in the batch:
      1. Verify: distinct stocks, |end_date_i - end_date_j| ≤ δ_sync,
         ≥ 80% valid data in the common period
      2. ρ_ij = Spearman rank correlation on RAW returns (not z-scored)
         in the common time segment
      3. d(z_i, z_j) = cosine distance = 1 - cos_sim(μ_i, μ_j)
      4. g(ρ_ij) = 1 - ρ_ij  (target distance)
      5. L_co = (1/|P|) · Σ (d(z_i, z_j) - g(ρ_ij))²

    Subsample to max_pairs if |P| > max_pairs.
    """
```

**Note:** Computing Spearman correlation is expensive in the forward pass. Pre-compute pairwise correlations for synchronous windows and store them in a lookup table, updated at each fold.

#### Sub-task 7: Curriculum scheduling

```python
def get_lambda_co(epoch: int, total_epochs: int, lambda_co_max: float = 0.5) -> float:
    """
    Phase 1 (0 → 30% epochs): λ_co = λ_co_max
    Phase 2 (30% → 60% epochs): λ_co decays linearly from λ_co_max to 0
    Phase 3 (60% → 100% epochs): λ_co = 0

    Returns: lambda_co for this epoch
    """
    phase1_end = int(0.30 * total_epochs)
    phase2_end = int(0.60 * total_epochs)

    if epoch < phase1_end:
        return lambda_co_max
    elif epoch < phase2_end:
        progress = (epoch - phase1_end) / (phase2_end - phase1_end)
        return lambda_co_max * (1.0 - progress)
    else:
        return 0.0
```

#### Sub-task 8: Validation ELBO

```python
def compute_validation_elbo(x, x_hat, mu, log_var, log_sigma_sq, T, F):
    """
    Validation ELBO — EXCLUDES γ and λ_co, INCLUDES σ².

    L_val = D/(2σ²) · L_recon^(γ=1) + (D/2)·ln(σ²) + L_KL

    Where L_recon^(γ=1) is the mean MSE NOT weighted by crisis.
    """
```

#### Applicable Invariants

- **INV-001:** D = T × F appears as coefficient of L_recon. Verify: `assert recon_coeff == T * F / (2 * sigma_sq)`.
- **INV-002:** σ² is scalar. `assert log_sigma_sq.ndim == 0`.
- **INV-006:** Modes mutually exclusive. `assert not (mode == 'P' and beta != 1.0)`.
- **INV-010:** The λ_co curriculum returns 0 in Phase 3.
- **INV-011:** Validation ELBO excludes γ and λ_co.

#### Known Pitfalls

- **DO NOT** omit the factor D = T × F. This is the most critical error in the entire pipeline. Without D, the reconstruction (~0.3-0.7 nats) is overwhelmed by the KL (~60-120 nats), and the model collapses immediately.
- **DO NOT** confuse per-element MSE (correct) with sum MSE (incorrect). D is applied as a multiplicative coefficient, not integrated into the MSE.
- **DO NOT** combine learned σ² with β annealing (Mode P + Mode F simultaneously).
- **DO NOT** include γ in the validation ELBO.
- **DO NOT** include λ_co in the validation ELBO.
- **DO NOT** compute Spearman correlation on z-scored data — use raw returns.

#### Required Tests

1. `test_D_factor_present`: loss_recon_coeff == T*F / (2*sigma_sq) for Mode P
2. `test_mode_P_gradients`: σ² receives a gradient; β is fixed at 1
3. `test_mode_F_sigma_frozen`: σ² has no gradient, D/2 is retained
4. `test_mode_F_beta_annealing`: β_t = min(1, t/T_warmup) correctly computed
5. `test_mode_A_beta_applied`: KL multiplied by β > 1
6. `test_modes_exclusive`: error if invalid mode or forbidden combination
7. `test_crisis_weight_gamma_1`: γ=1 → γ_eff=1 for all windows
8. `test_crisis_weight_gamma_3`: γ=3, f_c=1 → γ_eff=3; f_c=0 → γ_eff=1
9. `test_curriculum_phases`: λ_co correct at each phase boundary
10. `test_validation_elbo_excludes_gamma`: same L_val for γ=1 and γ=3
11. `test_co_movement_symmetric`: L_co(i,j) == L_co(j,i)
12. `test_loss_finite`: no NaN or Inf for valid inputs

---

### MOD-005 — training

**Phase:** 2 | **Mode:** lead_session | **Dependencies:** MOD-004 (C), MOD-001 (D: windows) | **Density:** very high
**Files:** `src/training/*.py`, `tests/unit/test_training.py`

#### Objective

Implement the complete training loop with curriculum batching (synchronous+stratified for co-movement phases, random for free refinement), early stopping on validation ELBO, ReduceLROnPlateau, and the checkpoint saving protocol.

#### Sub-task 1: Curriculum batching (batching.py)

```python
class CurriculumBatchSampler:
    """
    Phases 1-2 (λ_co > 0): SYNCHRONOUS + STRATIFIED batching
      - Select a random time block (δ_sync = 21 days)
      - Pre-cluster stocks into S strata (10-20 groups, k-means on
        trailing 63d returns). Recalculated per fold.
      - Sample B/S windows per stratum
      - Guarantee temporal synchronization for co-movement loss

    Phase 3 (λ_co = 0): Standard RANDOM SHUFFLING
      - Windows drawn uniformly across all stocks and periods
    """
```

**Transition:** The sampler switches strategy at the exact moment λ_co reaches 0 (start of Phase 3 of the curriculum).

#### Sub-task 2: Training loop (trainer.py)

```python
class VAETrainer:
    def __init__(self, model, optimizer, loss_mode, config):
        """
        optimizer: Adam(model.parameters(), lr=η_0, betas=(0.9, 0.999), eps=1e-8)
        weight_decay: 1e-5 (on all weights)
        """

    def train_epoch(self, train_loader, epoch, total_epochs) -> dict:
        """
        One epoch:
        1. Determine λ_co via curriculum (get_lambda_co)
        2. Determine batching type (synchronous if λ_co > 0, random otherwise)
        3. For each batch:
           a. Forward pass
           b. Compute loss (mode P/F/A per config)
           c. Backward + optimizer step
           d. Clamp log_sigma_sq to ensure σ² ∈ [1e-4, 10]:
              with torch.no_grad():
                model.log_sigma_sq.clamp_(math.log(1e-4), math.log(10))
        4. Collect metrics: loss_total, loss_recon, loss_kl,
           loss_co, sigma_sq, mse_crisis, mse_normal

        Returns: dict of epoch-level metrics
        """

    def validate(self, val_loader) -> float:
        """
        Compute validation ELBO (INV-011: excludes γ and λ_co, includes σ²).
        Returns: validation_elbo (lower is better)
        """

    def fit(self, train_dataset, val_dataset, max_epochs=100) -> dict:
        """
        Full loop:
        1. For each epoch 1..max_epochs:
           a. train_epoch()
           b. validate() → val_elbo
           c. scheduler.step(val_elbo)  [EXCEPT if Mode F and epoch < T_warmup]
           d. early_stopping.check(val_elbo) [EXCEPT if Mode F and epoch < T_warmup]
           e. If early stopping triggered → restore best weights → stop
        2. Return: best_epoch (E*), best_val_elbo, training_history

        Mode F specific:
        - Scheduler and early stopping DISABLED during T_warmup
        - Activated only once β_t = 1.0
        """
```

#### Sub-task 3: Early stopping (early_stopping.py)

```python
class EarlyStopping:
    """
    Patience: 10 epochs without validation ELBO improvement.
    Restore best weights: yes.
    Record E*: the epoch of the best checkpoint.
    """
```

#### Sub-task 4: Monitoring

Metrics to log at each epoch:
- `train_loss`, `val_elbo`
- `sigma_sq` (current value)
- `AU` (number of dimensions with KL_k > 0.01 nats, computed on the batch)
- `mse_crisis / mse_normal` (ratio, target [0.5, 2.0])
- `effective_eta` (effective crisis contribution)
- `lambda_co` (current curriculum value)
- `learning_rate` (current value)

**Overfit diagnostic:** if val_elbo_best / train_loss_best < 0.85 or > 1.5, flag the fold.

#### Applicable Invariants

- **INV-010:** Synchronous batching in Phases 1-2, random in Phase 3.
- **INV-011:** Validation ELBO excludes γ and λ_co.
- **INV-005:** Training on data ≤ training_end_date only.
- **INV-006:** Scheduler and early stopping disabled during warmup (Mode F).

#### Known Pitfalls

- **DO NOT** forget to clamp log_sigma_sq after each optimizer step — without clamping, σ² can diverge.
- **DO NOT** activate the scheduler during Mode F warmup — mechanical loss changes trigger artificial plateaus.
- **DO NOT** use random batching during co-movement phases — the co-movement loss is not computable without temporal synchronization.
- **DO NOT** forget to switch to `model.eval()` + `torch.no_grad()` for validation.

#### Required Tests

1. `test_sigma_sq_clamped`: σ² stays in [1e-4, 10] after 100 steps with random data
2. `test_curriculum_batching_transition`: batching changes at the right time
3. `test_early_stopping_patience`: stops after 10 epochs without improvement
4. `test_best_checkpoint_restored`: best epoch weights restored
5. `test_mode_F_warmup_protection`: scheduler/early_stopping inactive during warmup
6. `test_training_loss_decreases`: loss decreases over 5 epochs (simple synthetic data)

---

### MOD-006 — inference

**Phase:** 2 | **Mode:** subagent | **Dependencies:** MOD-002 (C), MOD-005 (D: trained model) | **Density:** medium
**Files:** `src/inference/*.py`, `tests/unit/test_inference.py`

#### Objective

Pass all windows for each stock through the trained encoder (forward pass without sampling), aggregate local latent vectors into composite profiles, build the exposure matrix B (n × K), and measure AU.

#### Sub-task 1: Stride-1 inference

```python
def infer_latent_trajectories(
    model: VAEModel,
    windows: torch.Tensor,         # (N_windows, T, F)
    window_metadata: pd.DataFrame,  # stock_id, start_date, end_date
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    """
    Forward pass (encode only, no sampling) for all windows.

    model.eval() + torch.no_grad() — inference only.
    Uses model.encode(x) which returns mu directly.

    Returns: dict mapping stock_id → ndarray of shape (n_windows_for_stock, K)
    """
```

#### Sub-task 2: Aggregation → composite profiles

```python
def aggregate_profiles(
    trajectories: dict[str, np.ndarray],  # stock_id → (n_windows, K)
    method: str = "mean",
) -> np.ndarray:
    """
    Aggregates local latent vectors into composite profiles.

    Default: mean (all windows contribute equally,
    preserving memory of all historical regimes).

    Returns: B of shape (n_stocks, K)
    """
```

#### Sub-task 3: AU measurement (active_units.py)

```python
def measure_active_units(
    model: VAEModel,
    windows: torch.Tensor,
    batch_size: int = 512,
) -> tuple[int, np.ndarray, list[int]]:
    """
    Computes the marginal KL per dimension:
      KL_k = (1/N) Σ_i (1/2)(μ²_ik + exp(log_var_ik) - log_var_ik - 1)

    Active unit k ⟺ KL_k > 0.01 nats.

    AU_max_stat = floor(sqrt(2 × N_obs / r_min))
    with N_obs = number of historical days, r_min = 2.

    If AU > AU_max_stat, truncate A to the AU_max_stat dimensions
    with the highest marginal KL.

    Returns:
      AU: int (number of active dimensions, possibly truncated)
      kl_per_dim: ndarray (K,) — marginal KL per dimension
      active_dims: list[int] — indices of active dimensions
    """
```

#### Applicable Invariants

- **INV-003:** B_A.shape == (n_stocks, AU), AU ≤ AU_max_stat.
- **CONV-07:** AU = |{k : KL_k > 0.01 nats}|, determined once per retraining.
- **INV-005:** Inference uses only training set windows.

#### Known Pitfalls

- **DO NOT** use `model.forward()` (which samples z) — use `model.encode()` (which returns mu directly). Inference must be deterministic.
- **DO NOT** forget `model.eval()` and `torch.no_grad()` — otherwise dropout is active and results are not reproducible.
- **DO NOT** forget statistical truncation: if AU > AU_max_stat, keep only the AU_max_stat dimensions with the highest KL_k.

#### Required Tests

1. `test_inference_deterministic`: two identical passes on the same data
2. `test_B_shape`: B.shape == (n_stocks, K)
3. `test_AU_measurement`: AU correct on synthetic data (known factors)
4. `test_AU_truncation`: if AU > AU_max_stat, correct truncation
5. `test_active_dims_ordering`: dimensions sorted by decreasing KL

---

### MOD-007 — risk_model

**Phase:** 2 | **Mode:** subagent | **Dependencies:** MOD-006 (D) | **Density:** high
**Files:** `src/risk_model/*.py`, `tests/unit/test_risk_model.py`

#### Objective

Transform the exposure matrix B_A (shape, unscaled) into a complete factor risk model: dual rescaling, factor estimation via cross-sectional OLS, Σ_z estimation (Ledoit-Wolf), D_ε estimation, Σ_assets assembly.

#### Sub-task 1: Dual rescaling (rescaling.py)

**For historical estimation** (date-specific):

$$\tilde{B}_{\mathcal{A},i,t} = \frac{\sigma_{i,t}}{\bar{\sigma}_t} \cdot \bar{\mu}_{\mathcal{A},i}$$

- $\sigma_{i,t}$: trailing 252d annualized vol of stock i at date t (from MOD-001)
- $\bar{\sigma}_t$: cross-sectional median of $\sigma_{i,t}$ over the universe $\mathcal{U}_t$
- $\bar{\mu}_{\mathcal{A},i}$: composite profile filtered to AU active dimensions

**Ratio bounding:** $R_{i,t} = \sigma_{i,t} / \bar{\sigma}_t$ is winsorized cross-sectionally at each date at percentiles $[P_5(R_{\cdot,t}), P_{95}(R_{\cdot,t})]$.

**For portfolio construction** (current-date):

$$\tilde{B}_{\mathcal{A},i}^{\text{port}} = \frac{\sigma_{i,\text{now}}}{\bar{\sigma}_{\text{now}}} \cdot \bar{\mu}_{\mathcal{A},i}$$

Same formula but with volatilities at the rebalancing date only. Same winsorization.

```python
def rescale_estimation(
    B_A: np.ndarray,           # (n, AU) shape-only exposures
    trailing_vol: pd.DataFrame, # (n_dates, n_stocks) σ_{i,t}
    universe_snapshots: dict,   # date → list of active stocks
    percentile_bounds: tuple = (5, 95),
) -> dict[str, np.ndarray]:
    """
    Returns: dict mapping date_str → B_A_t of shape (n_active_t, AU)
    Each B_A_t uses only stocks active at that date.
    """

def rescale_portfolio(
    B_A: np.ndarray,
    trailing_vol: pd.DataFrame,
    current_date: pd.Timestamp,
    universe: list[str],
    percentile_bounds: tuple = (5, 95),
) -> np.ndarray:
    """Returns: B_A_port of shape (n, AU)"""
```

#### Sub-task 2: Cross-sectional regression (factor_regression.py)

$$\hat{z}_t = (\tilde{B}_{\mathcal{A},t}^T \tilde{B}_{\mathcal{A},t})^{-1} \tilde{B}_{\mathcal{A},t}^T r_t$$

OLS at each date t, using only stocks active at that date.

```python
def estimate_factor_returns(
    B_A_by_date: dict[str, np.ndarray],  # date → (n_active, AU)
    returns: pd.DataFrame,                # (n_dates, n_stocks) log-returns
) -> np.ndarray:
    """
    Returns: z_hat of shape (n_dates, AU)
    """
```

**Conditioning guard:** if $\kappa(\tilde{B}_{\mathcal{A},t}^T \tilde{B}_{\mathcal{A},t}) > 10^6$, apply a minimal ridge: $\lambda = 10^{-6} \cdot \text{tr}(\tilde{B}_{\mathcal{A},t}^T \tilde{B}_{\mathcal{A},t}) / \text{AU}$.

#### Sub-task 3: Σ_z and D_ε estimation (covariance.py)

**Σ_z:** Empirical covariance of $\{\hat{z}_t\}$ over the FULL history + Ledoit-Wolf shrinkage:

$$\hat{\Sigma}_z = (1 - \delta^*) S_{\text{emp}} + \delta^* \frac{\text{tr}(S_{\text{emp}})}{\text{AU}} I_{\text{AU}}$$

$\delta^*$ computed analytically (use `sklearn.covariance.LedoitWolf` or direct implementation).

**D_ε:** Idiosyncratic variances from residuals:

$$\varepsilon_{i,t} = r_{i,t} - \tilde{B}_{\mathcal{A},i,t} \hat{z}_t$$

$D_{\varepsilon,i} = \text{Var}(\varepsilon_{i,\cdot})$ over all dates where stock i is active. Floor: $D_{\varepsilon,i} \geq 10^{-6}$.

**WARNING:** The residuals use the DATE-SPECIFIC rescaling $\tilde{B}_{\mathcal{A},i,t}$, not the portfolio rescaling.

#### Sub-task 4: Σ_assets assembly and eigendecomposition

$$\Sigma_{\text{assets}} = \tilde{B}_{\mathcal{A}}^{\text{port}} \cdot \Sigma_z \cdot (\tilde{B}_{\mathcal{A}}^{\text{port}})^T + D_\varepsilon$$

Eigendecomposition of Σ_z: $\Sigma_z = V \Lambda V^T$, with $\Lambda = \text{diag}(\lambda_1, ..., \lambda_{\text{AU}})$.

Rotation of B_A_port into the principal factor basis:

$$\tilde{B}'^{\text{port}}_{\mathcal{A}} = \tilde{B}_{\mathcal{A}}^{\text{port}} \cdot V$$

```python
def assemble_risk_model(
    B_A_port: np.ndarray,     # (n, AU)
    Sigma_z: np.ndarray,      # (AU, AU)
    D_eps: np.ndarray,        # (n,)
) -> dict:
    """
    Returns dict with:
      Sigma_assets: (n, n)
      eigenvalues: (AU,) — λ_k, sorted descending
      V: (AU, AU) — eigenvectors
      B_prime_port: (n, AU) — rotated exposures
    """
```

#### Applicable Invariants

- **INV-003:** B_A.shape == (n, AU), AU ≤ AU_max_stat.
- **INV-004:** Date-specific rescaling for estimation, current-date for portfolio.
- **INV-007:** Eigenvalues of Σ_z ≥ 0 (PSD). Rotation preserves Σ_assets.
- **INV-008:** Ratio σ_i,t / σ_bar_t winsorized [P5, P95].

#### Known Pitfalls

- **DO NOT** use portfolio rescaling (current-date) for historical estimation — this incorrectly attributes systematic risk to residuals.
- **DO NOT** forget the floor D_ε ≥ 1e-6 — without the floor, a stock with zero idiosyncratic variance receives infinite weight in the optimization.
- **DO NOT** compute Σ_z on a rolling window — use the FULL history (anti-cyclical principle).
- **DO NOT** forget the conditioning guard on B_A_t^T B_A_t.

#### Required Tests

1. `test_rescaling_dual`: estimation vs portfolio rescaling produce different results
2. `test_winsorization`: ratios bounded to [P5, P95]
3. `test_factor_regression_identity`: for B = I and r = z, z_hat ≈ z
4. `test_Sigma_z_psd`: all eigenvalues ≥ 0
5. `test_D_eps_floor`: min(D_eps) ≥ 1e-6
6. `test_covariance_reconstruction`: B'ΛB'^T + D_ε == BΣ_zB^T + D_ε
7. `test_conditioning_guard`: ridge applied when κ > 1e6
8. `test_rotation_preserves_covariance`: Σ_assets identical before and after rotation

---

### MOD-008 — portfolio_optimization

**Phase:** 2 | **Mode:** lead_session | **Dependencies:** MOD-007 (D) | **Density:** very high
**Files:** `src/portfolio/*.py`, `tests/unit/test_portfolio_optimization.py`

#### Objective

Implement portfolio optimization via factor entropy: computation of H(w) and ∇H(w) in the principal factor basis, SCA solver with Armijo backtracking, multi-start (5 initializations), cardinality enforcement via sequential entropy-aware elimination, and α calibration via the variance-entropy frontier.

#### Sub-task 1: Entropy and gradient (entropy.py)

**Entropy:**

$$H(w) = -\sum_{k=1}^{\text{AU}} \hat{c}'_k \ln \hat{c}'_k$$

Where:
- $B' = \tilde{B}'^{\text{port}}_{\mathcal{A}}$ (rotated, shape n × AU)
- $\beta' = B'^T w$ (portfolio exposure, shape AU)
- $c'_k = (\beta'_k)^2 \cdot \lambda_k$ (risk contribution from principal factor k, always ≥ 0)
- $C = \sum_k c'_k$ (total systematic risk)
- $\hat{c}'_k = c'_k / C$ (normalized risk contributions)

**Gradient:**

$$\nabla_w H = -\frac{2}{C}\; B'\; \phi, \qquad \phi_k = \lambda_k \, \beta'_k \,(\ln \hat{c}'_k + H)$$

**Verification:** At maximum entropy ($\hat{c}'_k = 1/\text{AU}$ ∀k), $\ln \hat{c}'_k = -\ln(\text{AU}) = -H$, so $\phi_k = 0$ and $\nabla_w H = 0$.

```python
def compute_entropy_and_gradient(
    w: np.ndarray,              # (n,)
    B_prime: np.ndarray,        # (n, AU)
    eigenvalues: np.ndarray,    # (AU,)
    eps: float = 1e-30,         # for numerical stability of ln
) -> tuple[float, np.ndarray]:
    """
    Returns: (H, grad_H) where H is scalar and grad_H is (n,)

    Numerical stability: if c'_k < eps, set ln(c'_k) = ln(eps)
    and exclude from normalization to avoid NaN.
    """
```

#### Sub-task 2: SCA solver (sca_solver.py)

**Multi-start (M=5):**
1. Equal-weight: w_i = 1/n
2. Minimum variance: w* = argmin w^T Σ w (convex QP)
3. Approximate ERC (Spinu): w ∝ solution of $w_i (Σw)_i = \text{const}$
4-5. Random: cardinality n_r ~ U(30, 300), uniform stocks, Dirichlet(1) weights, projection onto constraints (clip w_max, renormalize, zero < w_min, renormalize, iterate 2-3 passes)

**SCA iterations:**
```
For each start m = 1..M:
  w = w_init_m
  For t = 1..max_iter:
    1. Linearize H(w) around w^(t): H_lin(w) = H(w^(t)) + ∇H(w^(t))^T (w - w^(t))
    2. Solve the convex sub-problem (CVXPY):
       w* = argmax [ w^T μ - λ w^T Σ w + α · ∇H^T w - φ P_conc(w) - P_turn(w, w_old) ]
       s.t. w_i ∈ [0, w_max], 1^T w = 1, ||w - w_old||_1 / 2 ≤ τ_max
       (Note: the term α · ∇H^T w is LINEAR in w — this is what makes the sub-problem convex)
    3. Armijo backtracking to find η_step:
       η = max{ρ^j : j=0,...,j_max} such that
       f(w^(t) + η(w* - w^(t))) ≥ f(w^(t)) + c · η · Δ_surr
       with c=1e-4, ρ=0.5, j_max=20
       Δ_surr = f_surr(w*) - f_surr(w^(t)) ≥ 0
    4. Update: w^(t+1) = w^(t) + η(w* - w^(t))
    5. Convergence: |f(w^(t+1)) - f(w^(t))| < 1e-8

  Apply cardinality enforcement (sub-task 3)
  Store (w_final_m, f_final_m, H_final_m)

Select m* = argmax_m f(w_final_m)
```

**Objective f:**
- Default (μ=0): $f = -\lambda w^T \Sigma w + \alpha H(w) - \phi P_{\text{conc}}(w) - P_{\text{turn}}(w, w^{\text{old}})$
- Directional: $f = w^T \mu - \lambda w^T \Sigma w + \alpha H(w) - \phi P_{\text{conc}}(w) - P_{\text{turn}}(w, w^{\text{old}})$

**Current implementation: μ=0 only** (DVT Section 4.7: "Setting μ=0 is not an omission
but a design choice"). The solver accepts μ as a parameter for future extensibility
(DVT Iteration 3, Section 6.4), but all walk-forward and benchmark evaluations use μ=0.

#### Sub-task 3: Cardinality enforcement (cardinality.py)

```python
def enforce_cardinality(
    w: np.ndarray,
    w_min: float,
    sca_solver_fn,        # callable to re-optimize on reduced active set
    B_prime, eigenvalues,  # for entropy computation
    max_eliminations: int = 100,
) -> np.ndarray:
    """
    Repeat:
    1. S_sub = {i : 0 < w_i < w_min}. If empty → stop.
    2. For each i ∈ S_sub: ΔH_i = H(w) - H(w^(-i)) (entropy cost of elimination)
    3. Eliminate i* = argmin ΔH_i (the least costly)
    4. Re-optimize via SCA on the reduced active set
    5. Return to step 1

    Convergence guarantee: active set strictly decreases at each iteration.
    In practice: 5-15 iterations for n=1000.
    """
```

#### Sub-task 4: Constraints (constraints.py)

**Concentration penalty:**
$$P_{\text{conc}}(w) = \sum_i \max(0, w_i - \bar{w})^2, \quad \bar{w} = 3\%$$

**Turnover penalty (Almgren-Chriss):**
$$P_{\text{turn}}(w, w^{\text{old}}) = \kappa_1 \cdot \frac{1}{2}\sum_i |w_i - w_i^{\text{old}}| + \kappa_2 \cdot \sum_i \max(0, |w_i - w_i^{\text{old}}| - \bar{\delta})^2$$

First rebalancing: κ₁ = κ₂ = 0.

**Hard constraints:**
- $w_i \geq 0$ (long-only)
- $\sum_i w_i = 1$ (fully invested)
- $w_i \leq w_{\max}^{\text{hard}} = 5\%$
- $\frac{1}{2}\|w - w^{\text{old}}\|_1 \leq \tau_{\max}^{\text{hard}} = 30\%$

#### Sub-task 5: Variance-entropy frontier (frontier.py)

```python
def compute_variance_entropy_frontier(
    Sigma_assets, B_prime, eigenvalues, D_eps,
    lambda_risk: float = 1.0,
    alpha_grid: list = [0, 0.01, 0.05, 0.1, 0.5, 1, 5],
    **constraint_params,
) -> pd.DataFrame:
    """
    For each α, solve the default optimization (μ=0).
    Return DataFrame with columns: alpha, variance, entropy, n_active_positions.

    The elbow of the frontier = operating point.
    Automatic selection: α where ΔH/ΔVar < threshold.
    """
```

#### Applicable Invariants

- **INV-007:** H(w) computed in the principal factor basis. c'_k ≥ 0 ∀k.
- **INV-009:** Gradient exactly as specified. ∇H = 0 at maximum.
- **INV-012:** Constraints identical for VAE and benchmarks.

#### Known Pitfalls

- **DO NOT** compute entropy in the raw latent basis — contributions c_k can be negative if Σ_z is not diagonal. Always use the principal factor basis (after V rotation).
- **DO NOT** forget the constant term α · H(w^(t)) in the linearization — it does not affect w* but is necessary for computing Δ_surr in Armijo.
- **DO NOT** use a full step (η=1) without Armijo — in early iterations, the full step can DECREASE the actual objective.
- **DO NOT** forget the convergence condition |f(w^(t+1)) - f(w^(t))| < 1e-8, not |w^(t+1) - w^(t)|.
- **DO NOT** re-optimize constraint parameters (φ, κ₁, κ₂) per benchmark — they are set once for the VAE and applied identically to all models (INV-012).

#### Required Tests

1. `test_entropy_gradient_at_maximum`: ∇H = 0 when ĉ'_k = 1/AU ∀k
2. `test_entropy_gradient_numerical`: analytical gradient ≈ numerical gradient (finite diff)
3. `test_sca_convergence`: convergence in < 100 iterations on simple problem
4. `test_armijo_sufficient_decrease`: f(w_{t+1}) ≥ f(w_t) at each iteration
5. `test_multi_start_deterministic`: same results with same seed
6. `test_cardinality_enforcement`: no w_i in (0, w_min) after enforcement
7. `test_constraints_satisfied`: all hard constraints satisfied
8. `test_turnover_first_rebalancing`: κ₁ = κ₂ = 0 at first rebalancing
9. `test_known_solution`: for diagonal Σ and B = I, analytical solution recovered


---

## ISD Sections — Phase 3 (parallel: benchmarks)

---

### MOD-010 to MOD-015 — Benchmarks

**Phase:** 3 | **Mode:** teammates (Agent Team "benchmarks") | **Density:** low to high
**Files:** `src/benchmarks/*.py`, `tests/unit/test_benchmarks.py`

All benchmarks inherit from a common abstract class and share the infrastructure from MOD-001 (data_pipeline) and the portfolio constraints from MOD-008.

#### Common Abstract Class (base.py)

```python
from abc import ABC, abstractmethod

class BenchmarkModel(ABC):
    """
    Common interface for all benchmarks.

    Each benchmark receives exactly the same inputs (universe, returns,
    constraints) and produces exactly the same output format (weights).
    """

    def __init__(self, constraint_params: dict):
        """
        constraint_params: w_max, w_min, phi, kappa_1, kappa_2,
        delta_bar, tau_max, lambda_risk (identical to the VAE — INV-012).
        """
        self.constraint_params = constraint_params

    @abstractmethod
    def fit(self, returns: pd.DataFrame, universe: list[str], **kwargs) -> None:
        """Estimate the risk model (if applicable)."""

    @abstractmethod
    def optimize(self, w_old: np.ndarray = None, is_first: bool = False) -> np.ndarray:
        """Produce optimal weights under shared constraints."""

    def evaluate(self, w: np.ndarray, returns_oos: pd.DataFrame) -> dict:
        """Compute all OOS metrics (shared)."""
```

---

### MOD-010 — bench_equal_weight

```python
class EqualWeight(BenchmarkModel):
    def fit(self, returns, universe, **kwargs):
        self.n = len(universe)

    def optimize(self, w_old=None, is_first=False):
        w = np.ones(self.n) / self.n
        # Hard cap at 5% non-binding for n=1000
        return np.clip(w, 0, self.constraint_params['w_max'])
```

No risk model, no turnover penalty (turnover intrinsically low).

---

### MOD-011 — bench_inverse_vol

```python
class InverseVolatility(BenchmarkModel):
    def fit(self, returns, universe, trailing_vol, current_date, **kwargs):
        """
        σ_i = trailing 252d annualized vol at current_date.
        w_i ∝ 1/σ_i, then projection onto constraints.
        """
        self.sigma = trailing_vol.loc[current_date, universe].values
        assert (self.sigma > 0).all(), "Zero or negative vol detected"

    def optimize(self, w_old=None, is_first=False):
        w = (1.0 / self.sigma)
        w /= w.sum()
        return self._project_to_constraints(w, w_old, is_first)

    def _project_to_constraints(self, w, w_old, is_first):
        """Clip w_max, zero < w_min, renormalize. Iterate 2-3 passes."""
```

---

### MOD-012 — bench_min_variance

```python
class MinimumVariance(BenchmarkModel):
    def fit(self, returns, universe, **kwargs):
        """
        Estimate Σ_LW via Ledoit-Wolf (2004) shrinkage toward scaled identity
        on the full training window (expanding, anti-cyclical).

        Use sklearn.covariance.LedoitWolf or direct implementation.
        """
        R = returns[universe].dropna(how='all').values
        lw = LedoitWolf().fit(R)
        self.Sigma_LW = lw.covariance_

    def optimize(self, w_old=None, is_first=False):
        """
        min w^T Σ_LW w
        s.t. shared constraints (P_conc, P_turn, hard caps)

        Convex QP → CVXPY + MOSEK. Global solution guaranteed.
        """
```

Expose `self.Sigma_LW` for reuse by MOD-013 (ERC).

---

### MOD-013 — bench_erc

```python
class EqualRiskContribution(BenchmarkModel):
    def fit(self, returns, universe, **kwargs):
        """Reuses Σ_LW from MOD-012."""
        # Use the same Ledoit-Wolf estimator
        R = returns[universe].dropna(how='all').values
        lw = LedoitWolf().fit(R)
        self.Sigma_LW = lw.covariance_

    def optimize(self, w_old=None, is_first=False):
        """
        ERC via Spinu (2013): convex log-barrier formulation.

        min Σ_i Σ_j (w_i(Σw)_i - w_j(Σw)_j)²
        or equivalent formulation:
        min (1/2) w^T Σ w - Σ_i ln(w_i)

        Convergence: < 5 Newton iterations for n < 1000.
        Post-hoc projection onto hard caps, then iterative renormalization.
        """
```

---

### MOD-014 — bench_pca_factor_rp

**This is the most important benchmark** — it isolates the added value of the VAE vs linear PCA.

```python
class PCAFactorRiskParity(BenchmarkModel):
    def fit(self, returns, universe, **kwargs):
        """
        1. PCA on the return matrix (T_est × n).
        2. Number of factors k via Bai & Ng (2002) IC₂.
        3. B_PCA ∈ R^(n × k): PCA loadings.
        4. Σ_z_PCA = Λ_k (diagonal — orthogonal principal components).
        5. D_ε_PCA: residuals, diagonal variance, floor 1e-6.
        6. Σ_assets = B_PCA Λ_k B_PCA^T + D_ε_PCA.
        """

    def _bai_ng_ic2(self, returns_matrix, k_max=30):
        """
        Information Criterion IC₂ (Bai & Ng, 2002).
        IC₂(k) = ln(V(k)) + k · ((n+T)/(n·T)) · ln(min(n,T))
        V(k) = (1/(n·T)) · ||R - F_k Λ_k^T||²_F
        Select k* = argmin IC₂(k).
        Typically k ∈ [5, 15].
        """

    def optimize(self, w_old=None, is_first=False):
        """
        SAME objective and solver as the VAE:
        max -λ w^T Σ w + α H(w)  (μ = 0)

        H(w) computed in the principal factor basis of Σ_z_PCA
        (but since Σ_z_PCA = Λ_k is already diagonal, the rotation is trivial: V = I).

        SCA solver identical to MOD-008 (import sca_solver).
        α calibration identical (variance-entropy frontier).
        Constraints identical.
        """
```

**Dependencies:** MOD-008 (C: SCA solver import), MOD-001 (D: returns).

---

### MOD-015 — bench_pca_vol

```python
class PCAVolRiskParity(PCAFactorRiskParity):
    """
    Variant of MOD-014 with augmented matrix (T × 2n):
    - Concatenation of z-scored returns + z-scored 21d realized volatilities.
    - PCA on this augmented matrix.
    - Rest is identical (IC₂, SCA, same constraints).

    Isolation: VAE non-linearity independently of feature enrichment.
    """

    def fit(self, returns, universe, trailing_vol, **kwargs):
        # Z-score per window for the two features, then PCA on augmented matrix
```

---

### Required Tests (all benchmarks)

1. `test_constraints_identical`: same constraint parameters for all models
2. `test_equal_weight_sum_to_one`: w.sum() == 1
3. `test_min_var_beats_random`: min-var < random portfolio variance (sanity)
4. `test_erc_equal_risk_contributions`: RC_i ≈ RC_j ∀ i,j (tolerance 5%)
5. `test_pca_ic2_range`: k ∈ [3, 30] for realistic data
6. `test_pca_factor_rp_uses_sca`: the SCA solver converges
7. `test_benchmark_output_format`: all models return w of shape (n,)

---

## ISD Sections — Phase 4 (sequential)

---

### MOD-009 — walk_forward

**Phase:** 4 | **Mode:** lead_session | **Dependencies:** MOD-001–008 (D), MOD-010–015 (D) | **Density:** high
**Files:** `src/walk_forward/*.py`, `tests/integration/test_walk_forward.py`

#### Objective

Orchestrate the complete walk-forward validation: scheduling of ~34 folds, execution of Phase A (HP selection) and Phase B (deployment) per fold, metric computation across 3 layers, composite scoring, and final holdout evaluation.

#### Sub-task 1: Fold scheduling (folds.py)

```python
def generate_fold_schedule(
    total_years: int = 30,
    min_training_years: int = 10,
    oos_months: int = 6,
    embargo_days: int = 21,
    holdout_years: int = 3,
) -> list[dict]:
    """
    Generates the walk-forward fold schedule.

    Fold k:
      - Training end: Year 10 + k × 0.5
      - Embargo: 21 trading days after training end
      - OOS start: training_end + embargo + 1
      - OOS end: OOS start + 6 months
      - Validation subset (nested): [training_end - 2yr, training_end]
      - Training subset (nested): [start, training_end - 2yr]

    Holdout: last ~3 years (Year 27 to Year 30)

    Returns: list of dicts with fold_id, train_start, train_end,
             val_start, val_end, embargo_start, embargo_end,
             oos_start, oos_end, is_holdout
    """
```

**~34 folds:** from Year 10 to Year 27, 6-month steps.

#### Sub-task 2: Phase A — HP selection (phase_a.py)

```python
def run_phase_a(
    fold: dict,
    hp_configs: list[dict],  # hyperparameter grid
    data_pipeline_output: dict,
    build_vae_fn,
) -> dict:
    """
    For each HP config:
      1. Build the VAE with these HPs
      2. Train on [start, train_end - 2yr]
         with early stopping on [train_end - 2yr, train_end]
         → record E*_config
      3. Build downstream pipeline (AU, B_A, Σ_z, portfolio)
      4. Evaluate on OOS → fold score

    Composite score:
      Score = Ĥ_OOS - λ_pen · max(0, MDD_OOS - MDD_threshold) - λ_est · max(0, 1 - R_Σ)

    where Ĥ = H(w) / ln(AU)  (normalized entropy, ∈ [0, 1])
    MDD as fraction ∈ [0, 1]
    R_Σ = N_obs / (AU(AU+1)/2)

    Baselines: MDD_threshold = 0.20, λ_pen = 5, λ_est = 2.

    Elimination: configs with AU < max(0.15K, AU_PCA), EP < max(0.40, EP_PCA),
    OOS/train MSE > 3.0 are eliminated before scoring.

    Returns: best_config, E_star, fold_scores
    """
```

#### Sub-task 3: Phase B — Deployment run (phase_b.py)

```python
def run_phase_b(
    fold: dict,
    best_config: dict,
    E_star: int,           # epochs from Phase A
    data_pipeline_output: dict,
) -> dict:
    """
    Re-train the encoder on ALL data [start, train_end]
    for E* epochs (no validation set, no early stopping).

    E* determination (DVT Section 4.8):
    - Per-fold: E* = E*_config of the selected config in Phase A.
    - Robust alternative: E* = median of E*_config across all
      PREVIOUS folds (expanding, CONV-10 point-in-time).
    - Holdout: E* = median of E*_config across ALL walk-forward folds.

    Sanity check: if training loss at E* in Phase B is > 20% lower
    than Phase A, flag the fold.

    Then: complete downstream pipeline
    (AU, B_A, Σ_z, D_ε, portfolio optimization).

    Evaluate on OOS.

    Returns: weights, metrics, AU, diagnostics
    """
```

#### Sub-task 4: Metrics (metrics.py)

**Layer 1 — VAE quality:**
- OOS reconstruction error by regime (OOS/train MSE < 1.5; crisis/normal ∈ [0.5, 2.0])
- AU (AU ≥ max(0.15K, AU_PCA); AU ≤ min(0.85K, AU_max_stat))
- Latent stability: Spearman ρ > 0.85 of pairwise inter-stock distances between retrainings

**Layer 2 — Risk model quality:**
- Realized vs predicted variance: var(r_p^OOS) / (w^T Σ̂ w) ∈ [0.8, 1.2]
- Factor explanatory power: > max(0.50, EP_PCA + 0.10)
- Realized vs predicted correlation (rank)

**Layer 3 — Portfolio quality:**
- Normalized factor entropy OOS (primary)
- Annualized volatility OOS (primary)
- Maximum drawdown OOS (primary)
- Crisis-period return (primary)
- Annualized return, Sharpe, Calmar, Sortino (diagnostic)
- Turnover at rebalancing (diagnostic, target < 30%)
- Diversification ratio DR (diagnostic)
- Effective number of positions 1/Σw²_i (diagnostic)

#### Sub-task 5: Holdout

```python
def run_holdout(
    final_config: dict,
    E_star_median: int,
    data_pipeline_output: dict,
    holdout_period: tuple,    # (start_date, end_date)
    benchmark_results: dict,  # benchmark results on holdout
) -> dict:
    """
    A SINGLE EXECUTION, at the end.
    Train each model (VAE + 6 benchmarks) on all history
    up to t_holdout. Evaluate on the last ~3 years.
    Compare holdout vs walk-forward to detect structural overfitting.
    """
```

#### Applicable Invariants

- **INV-005:** No look-ahead. Training ≤ train_end, OOS > embargo_end.
- **CONV-09:** Expanding window for training.
- **CONV-10:** Point-in-time universe at each date.
- **INV-012:** Same constraints for VAE and benchmarks.

#### Required Tests

1. `test_fold_no_overlap`: no training/OOS overlap (with embargo)
2. `test_fold_dates_sequential`: folds ordered chronologically
3. `test_holdout_untouched`: holdout data never seen during walk-forward
4. `test_score_normalization`: Ĥ ∈ [0, 1]
5. `test_phase_b_no_early_stopping`: E* epochs executed without interruption

---

### MOD-016 — integration

**Phase:** 4 | **Mode:** lead_session | **Dependencies:** all | **Density:** high
**Files:** `src/integration/*.py`, `scripts/run_walk_forward.py`, `scripts/run_benchmarks.py`

#### Objective

Orchestrate the complete end-to-end execution and produce the results report with statistical comparison tests.

#### Sub-task 1: E2E Pipeline (pipeline.py)

```python
class FullPipeline:
    def run(self, config: dict) -> dict:
        """
        1. data_pipeline.prepare(config)
        2. For each fold in walk_forward.schedule:
           a. Phase A: HP selection
           b. Phase B: Deployment run (VAE)
           c. For each benchmark: fit + optimize
        3. Aggregate results
        4. Run holdout
        5. Statistical tests
        6. Generate report
        """
```

#### Sub-task 2: Statistical tests (statistical_tests.py)

```python
def wilcoxon_paired_test(vae_scores, benchmark_scores):
    """
    Wilcoxon signed-rank test (non-parametric) on per-fold differences.
    H0: median(Δ_k) = 0.
    Threshold: p < 0.05.
    """

def bootstrap_effect_size(vae_scores, benchmark_scores, n_bootstrap=10000):
    """Median of Δ + bootstrap confidence interval (percentile method)."""

def holm_bonferroni_correction(p_values: list[float], alpha=0.05):
    """Correction for 6 benchmarks × 4 metrics = 24 tests."""

def regime_decomposition(fold_metrics, vix_data):
    """
    Separate folds into "crisis" (> 20% days VIX > P80) and "calm".
    Report metrics and tests separately.
    """
```

#### Required Tests

1. `test_pipeline_e2e_synthetic`: complete pipeline on synthetic data (50 stocks, 10 years)
2. `test_statistical_tests_known`: Wilcoxon correct on known distributions
3. `test_holm_bonferroni`: correction applied correctly

---

## CLAUDE.md — Template for the Project

```markdown
# Latent Risk Factor Discovery — VAE Strategy

## Context
Portfolio construction pipeline based on latent risk factor discovery
via VAE (1D CNN encoder-decoder). Objective: maximize factor diversification
(Shannon entropy on principal factor risk contributions).

## Architecture
Stock data (synthetic / EODHD) → data_pipeline → VAE training → inference → risk_model → portfolio_optimization
                                                                         ↓
                                                                  walk_forward (34 folds)
                                                                         ↓
                                                                  benchmarks (×6)

## Critical Conventions — DO NOT VIOLATE
- LOG returns, never arithmetic
- Z-score PER-WINDOW, PER-FEATURE
- σ² is a SCALAR, not a vector (init 1.0, clamp [1e-4, 10])
- D = T × F MUST appear in the reconstruction loss (D/(2σ²) · L_recon)
- P/F/A modes are MUTUALLY EXCLUSIVE
- DUAL rescaling: date-specific for estimation, current-date for portfolio
- Entropy H(w) computed in the PRINCIPAL FACTOR basis (after V rotation of Σ_z)
- Ratio σ_i,t / σ_bar_t WINSORIZED [P5, P95] BEFORE rescaling
- Portfolio constraints IDENTICAL between VAE and all benchmarks
- NO look-ahead — strict point-in-time

## Code Structure
- `src/data_pipeline/`: loading, returns, universe, windowing, crisis
- `src/vae/`: architecture, loss, model
- `src/training/`: training loop, batching, early stopping
- `src/inference/`: composite profiles, AU
- `src/risk_model/`: rescaling, factor regression, covariance
- `src/portfolio/`: entropy, SCA, constraints, cardinality
- `src/walk_forward/`: folds, Phase A/B, metrics
- `src/benchmarks/`: 6 benchmark models
- `tests/unit/` and `tests/integration/`

## Workflow for Agents
1. Read `docs/isd/00_global.md` AND the ISD section for your module
2. Implement interface assertions FIRST
3. TDD: test before code for each sub-task
4. Commit after each sub-task
5. If ambiguous: comment `# AMBIGUITY: ...` and conservative interpretation
6. **Log all gap decisions** in `docs/implementation_decisions.md` (see Section 00 — Implementation Decision Logging)

## Dependencies
Python 3.11+, PyTorch ≥ 2.1, NumPy, SciPy, CVXPY + MOSEK, pandas, scikit-learn, pytest

## Tests
- `pytest tests/unit/` — unit tests per module
- `pytest tests/integration/` — inter-module integration tests
- Module complete ⟺ all its tests pass + interface assertions satisfied
```

---

## Operational Execution Plan

### Phase 1 — Infrastructure (parallel, Agent Team, ~2-3 days)

| Teammate | Module | Main Deliverable |
|----------|--------|-------------------|
| `data-engineer` | MOD-001 data_pipeline | Z-scored windows, universe, crisis labels, trailing vol |
| `ml-architect` | MOD-002 vae_architecture | VAEModel with build_vae, sizing rules |
| `test-lead` | MOD-003 test_infrastructure | Synthetic data, fixtures, known solutions |

**Synchronization:** interface assertions between MOD-001 and MOD-002 (window shape = (N, T, F), F=2, T=504).

### Phase 2 — Core Pipeline (sequential, Subagents builder-validator, ~5-7 days)

| Order | Module | Sequential Justification |
|-------|--------|---------------------------|
| 1 | MOD-004 loss_function | Depends on MOD-002 (VAEModel). Lead session (very high density). |
| 2 | MOD-005 training | Depends on MOD-004 (loss). Lead session (math coupling degree 4). |
| 3 | MOD-006 inference | Depends on MOD-005 (trained model). Subagent (medium density). |
| 4 | MOD-007 risk_model | Depends on MOD-006 (B matrix). Subagent (high density). |
| 5 | MOD-008 portfolio_optimization | Depends on MOD-007 (Σ_z). Lead session (very dense SCA solver). |

**Protocol:** builder-validator for MOD-006 and MOD-007. Lead session with human supervision for MOD-004, MOD-005, MOD-008.

### Phase 3 — Benchmarks (parallel, Agent Team, ~2-3 days)

| Teammate | Modules | Notes |
|----------|---------|-------|
| `bench-simple` | MOD-010 (1/N), MOD-011 (inverse-vol) | Trivial, ~1 day |
| `bench-covariance` | MOD-012 (min-var), MOD-013 (ERC) | Share Σ_LW |
| `bench-factor` | MOD-014 (PCA factor RP), MOD-015 (PCA+vol) | Reuse SCA solver from MOD-008 |

**Dependencies:** MOD-014 and MOD-015 import the SCA solver from MOD-008 — MOD-008 must be complete and stable.

### Phase 4 — Integration (sequential, lead session, ~3-5 days)

| Order | Module | Description |
|-------|--------|-------------|
| 1 | MOD-009 walk_forward | Orchestration of 34 folds, Phase A/B |
| 2 | MOD-016 integration | E2E, statistical tests, final report |

**Mandatory human validation** before Phase 3 → Phase 4 transition and before holdout execution.

---

## Post-Execution Decision Matrix

| Scenario | Condition | Action |
|----------|-----------|--------|
| A — VAE outperforms all | p < 0.05 on ≥ 2/4 primary metrics vs all benchmarks | Production |
| B — VAE > PCA but not min-var/ERC | Non-linearity useful, optimization to review | Iterations 1-3 (doc v4.1 Section 6) |
| C — PCA ≈ VAE | Non-linearity without measurable value | Adopt PCA (cost /100) |
| D — 1/N ≥ all | Estimation error absorbs all benefit | 1/N or relax constraints |
| E — Heterogeneous by regime | VAE > in crisis, < in calm | Dual-regime system |
