# Project Structure

> **Note:** This file documents the project tree and external directories. Built incrementally.

## Project Tree

```
latent_risk_factor/
├── .claude/                          # Claude Code configuration
│   ├── settings.json                 # Hooks (pyright, pylint, flake8)
│   ├── rules/                        # Auto-loaded rules
│   │   ├── 00-claude-config.md       # Claude configuration
│   │   ├── 00-role.md               # Expert persona (quant ML engineer)
│   │   ├── 01-python-guidelines.md   # Python coding rules
│   │   ├── 02-git-commit.md          # Git commit rules
│   │   ├── 03-code-quality.md        # Quality verification
│   │   ├── 04-task-completion.md     # Task completion rules
│   │   ├── 05-comments-readme.md     # Comments and README rules
│   │   ├── 06-documentation-protocol.md  # Documentation protocol
│   │   ├── 07-environment-setup.md   # Environment setup (uv)
│   │   └── project/                  # Project-specific documentation
│   │       ├── project-overview.md   # Pipeline goals and descriptions
│   │       ├── structure.md          # This file
│   │       ├── changelog.md          # Recent changes
│   │       └── datasets.md           # Dataset documentation
│   ├── commands/                     # User slash commands
│   └── agents/                       # Specialized sub-agents
│
├── docs/                             # Technical specification documents
│   ├── ISD_vae_latent_risk_factors.md    # Implementation Specification Document
│   ├── DVT_strategie_facteurs_risque_latents_v4_1.md  # Technical Vision Document v4.1
│   ├── Benchmark_latent_risk_factor.md   # Benchmark specifications
│   └── assertions/                   # Contract assertion definitions
│       └── contracts.yaml            # Inter-module interface contracts
│
├── src/                              # Main source code
│   ├── __init__.py
│   ├── config.py                     # Centralized configuration (frozen dataclasses)
│   ├── data_pipeline/                # MOD-001: Data loading and preparation
│   │   ├── __init__.py
│   │   ├── data_loader.py            # CRSP / alternative data loading
│   │   ├── returns.py                # Log-return calculation (CONV-01)
│   │   ├── universe.py               # Point-in-time universe construction (CONV-10)
│   │   ├── windowing.py              # Sliding window + z-scoring (CONV-02)
│   │   ├── crisis.py                 # VIX threshold + crisis labeling
│   │   └── features.py               # Realized volatility + additional features
│   ├── vae/                          # MOD-002: VAE architecture
│   │   ├── __init__.py
│   │   ├── build_vae.py              # Architecture construction (sizing rules)
│   │   ├── encoder.py                # InceptionHead + ResidualBlocks + projection
│   │   ├── decoder.py                # Transposed architecture
│   │   ├── model.py                  # VAEModel (forward, reparameterize, encode)
│   │   └── loss.py                   # MOD-004: Loss (3 modes, crisis, co-movement)
│   ├── training/                     # MOD-005: Training loop
│   │   ├── __init__.py
│   │   ├── trainer.py                # Training loop (fit, train_epoch, validate)
│   │   ├── batching.py               # Curriculum batching (sync+stratified / random)
│   │   ├── early_stopping.py         # Patience + best checkpoint restore
│   │   └── scheduler.py              # ReduceLROnPlateau wrapper
│   ├── inference/                    # MOD-006: Latent factor extraction
│   │   ├── __init__.py
│   │   ├── composite.py              # Sliding inference + aggregation -> B
│   │   └── active_units.py           # AU measurement (KL > 0.01 nats)
│   ├── risk_model/                   # MOD-007: Factor risk model
│   │   ├── __init__.py
│   │   ├── rescaling.py              # Dual rescaling (estimation + portfolio)
│   │   ├── factor_regression.py      # Cross-sectional OLS -> z_hat_t
│   │   ├── covariance.py             # Sigma_z (LW), D_eps, Sigma_assets assembly
│   │   └── conditioning.py           # Conditioning guard + ridge fallback
│   ├── portfolio/                    # MOD-008: Portfolio optimization
│   │   ├── __init__.py
│   │   ├── entropy.py                # H(w), grad_H(w), principal factor rotation
│   │   ├── sca_solver.py             # SCA + Armijo + multi-start
│   │   ├── constraints.py            # P_conc, P_turn, hard caps
│   │   ├── cardinality.py            # Sequential entropy-aware elimination
│   │   └── frontier.py               # Variance-entropy frontier + alpha calibration
│   ├── walk_forward/                 # MOD-009: Walk-forward validation
│   │   ├── __init__.py
│   │   ├── folds.py                  # Fold scheduling (~34 folds)
│   │   ├── phase_a.py                # HP selection (nested validation)
│   │   ├── phase_b.py                # Deployment run (E* epochs, no early stop)
│   │   ├── metrics.py                # All metrics (3 layers)
│   │   └── selection.py              # Scoring + selection criterion
│   ├── benchmarks/                   # MOD-010 to MOD-015: Benchmarks
│   │   ├── __init__.py
│   │   ├── base.py                   # Abstract BenchmarkModel class
│   │   ├── equal_weight.py           # MOD-010: 1/N
│   │   ├── inverse_vol.py            # MOD-011: Inverse volatility
│   │   ├── min_variance.py           # MOD-012: Min-var Ledoit-Wolf
│   │   ├── erc.py                    # MOD-013: Equal Risk Contribution (Spinu)
│   │   ├── pca_factor_rp.py          # MOD-014: PCA factor risk parity (Bai-Ng IC2)
│   │   └── pca_vol.py                # MOD-015: PCA + realized vol variant
│   └── integration/                  # MOD-016: E2E orchestration
│       ├── __init__.py
│       ├── pipeline.py               # FullPipeline orchestrator
│       ├── statistical_tests.py      # Wilcoxon, Holm-Bonferroni, bootstrap
│       └── reporting.py              # Results compilation
│
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── unit/                         # Unit tests per module
│   │   ├── __init__.py
│   │   ├── test_data_pipeline.py
│   │   ├── test_vae_architecture.py
│   │   ├── test_loss_function.py
│   │   ├── test_training.py
│   │   ├── test_inference.py
│   │   ├── test_risk_model.py
│   │   ├── test_portfolio_optimization.py
│   │   └── test_benchmarks.py
│   ├── integration/                  # Integration tests
│   │   ├── __init__.py
│   │   ├── test_vae_training.py
│   │   ├── test_risk_pipeline.py
│   │   ├── test_portfolio_pipeline.py
│   │   └── test_walk_forward.py
│   └── fixtures/                     # Test fixtures
│       ├── __init__.py
│       ├── synthetic_data.py         # Deterministic synthetic data generator
│       └── known_solutions.py        # Known analytical solutions for verification
│
├── scripts/                          # CLI entry points
│   ├── run_walk_forward.py           # Run full walk-forward validation
│   └── run_benchmarks.py             # Run benchmarks only
│
├── ISD_methodology.md                # ISD Methodology v1.0
├── CLAUDE.md                         # Project-level Claude instructions
├── pyproject.toml                    # Package config, dependencies, tool settings
├── .gitignore
└── README.md
```

---

## Remote Directories

### CRSP Data

- **Absolute Path:** Not yet configured — depends on deployment environment
- **Description:** CRSP daily stock data (prices, returns, volumes) for US equities
- **Structure:**
```
{TODO: to be documented when data source is configured}
```
- **Used by:** `src/data_pipeline/data_loader.py` — loads raw price data for the universe

---

## Update History

| Date | Section | Change |
|------|---------|--------|
| 2026-02-06 | Initial | Created full project tree from ISD specification |
