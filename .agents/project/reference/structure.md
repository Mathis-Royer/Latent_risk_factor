# Project structure

Generated on 2026-11-07 from `git ls-files` (tracked and untracked, not ignored); each description is the first line of
the module docstring or of the Markdown title, so it says what the file claims to do. Regenerate after adding or moving
modules rather than editing by hand. `__init__.py` files and the 34 raw audit reports are omitted.

## Top level

```
AGENTS.md, CLAUDE.md      # agent entry point (CLAUDE.md imports AGENTS.md)
.agents/                  # agent rules: common/ and stacks/ (kit-owned copies), project/ (this project), skills/
.claude/settings.json     # Claude Code adapter: permissions, plugins (hooks come from the agent-kit plugin)
pyproject.toml, uv.lock   # dependencies (uv sync --all-extras)
README.md                 # setup, MOSEK licence, data, usage
ISD_methodology.md        # agent-orchestration method used to write the ISD (process, not strategy)
data/                     # merged panel (Git LFS) + Tiingo metadata; raw per-ticker cache ignored (data.md)
results/                  # run outputs (ignored); runs/ = TensorBoard logs
```

## Code, tests, scripts, notebooks, docs

```
docs/
    Benchmark_latent_risk_factor.md                  # Procédure de Benchmarking — Latent Risk Factor Discovery Strategy
    DVT_strategie_facteurs_risque_latents_v4_1.md    # Latent Risk Factor Discovery Strategy
    ISD_vae_latent_risk_factors.md                   # ISD — VAE Latent Risk Factor Discovery Strategy
    analyse_critique_strategie.md                    # Analyse Critique de la Stratégie de Facteurs de Risque Latents
    assertions/
        cross_reference_report.md                    # Cross-Reference Report: Test Suite vs Reference Documents
    REDESIGN.md                                      # Consolidated redesign of record (from both audits)
    audit_theory/
        INDEX.md                                     # Theory-only re-audit — units, verdicts, stream-cut annex
        REDESIGN.md                                  # Theory-only re-audit — redesign proposal (source)
        findings.jsonl
        reports/                                     # 38 raw advocate/critic reports
    audit_v5/
        INDEX.md                                     # Strategy re-audit v5 — Index
        REDESIGN.md                                  # Strategy re-audit v5 — Systemic redesign proposal (source)
        findings.jsonl
    data_preprocessing_diagnostic.md                 # Data Preprocessing Pipeline Diagnostic
    diagnostic.md                                    # Guide d'Interprétation des Diagnostics
    divergences.md                                   # DIVERGENCES CODE vs DOCUMENTS DE REFERENCE (ISD / DVT)
    implementation_decisions.md                      # Implementation Decisions Log
    literature_review_idiosyncratic_risk.md          # Idiosyncratic Risk: Literature Review
    literature_review_strategy.md                    # Literature Review: Strategy Analysis
    sigma_z_ewma_justification.md                    # Justification: sigma_z_ewma_half_life = 0
    strategic_paths_post_v2.md                       # Strategic Paths After V2 Diagnostic — Decision Framework
    strategy_comparison.md                           # Strategy Comparison: Uniform Entropy vs Sortino-Budgeted Factor Risk Parity
    strategy_philosophy.md                           # Strategy Philosophy — Latent Risk Factor Discovery
notebooks/
    dashboard.ipynb
    diagnostic_experiments.ipynb
scripts/
    download_sp500.py                                # Download S&P 500 index data for the SP500TotalReturn benchmark
    download_tiingo.py                               # Download US equity daily data from Tiingo API with persistent storage
    run_benchmarks.py                                # CLI entry point for benchmarks only (no VAE training)
    run_diagnostic.py                                # CLI entry point for comprehensive pipeline diagnostic
    run_walk_forward.py                              # CLI entry point for the complete walk-forward validation
src/
    benchmarks/
        base.py                                      # Abstract base class for all benchmark models
        equal_weight.py                              # MOD-010: Equal-weight (1/N) benchmark
        erc.py                                       # MOD-013: Equal Risk Contribution (ERC) benchmark via Spinu (2013)
        inverse_vol.py                               # MOD-011: Inverse volatility benchmark
        min_variance.py                              # MOD-012: Minimum variance benchmark (Ledoit-Wolf shrinkage)
        pca_factor_rp.py                             # MOD-014: PCA factor risk parity benchmark
        pca_vol.py                                   # MOD-015: PCA + realized volatility variant
        sp500_index.py                               # S&P 500 index benchmark (price return)
    config.py                                        # Centralized configuration for the VAE Latent Risk Factor pipeline
    data_pipeline/
        crisis.py                                    # Crisis labeling via VIX threshold on expanding training window
        data_loader.py                               # Stock data loading: synthetic CSV generation (Phase A) and file loading
        features.py                                  # Trailing realized volatility computation for the data pipeline
        returns.py                                   # Log-return computation with missing value handling and delisting imputation
        tiingo_keys.py                               # Resolve Tiingo API keys without ever hardcoding them in tracked files
        universe.py                                  # Point-in-time universe construction and delisting handling
        windowing.py                                 # Sliding window creation with per-window z-score normalization
    diagnostics/
        e2e_diagnostics.py                           # End-to-end pipeline diagnostics: determinism, alignment, universe tracking
        factor_diagnostics.py                        # Factor model diagnostics: eigenvalue analysis, shrinkage quality, regression stability
        portfolio_diagnostics.py                     # Portfolio optimization diagnostics: budget conservation, gradient analysis, frontier quality
        vae_diagnostics.py                           # VAE-specific diagnostics: posterior quality, KL analysis, sampling significance
    inference/
        active_units.py                              # Active Unit (AU) measurement and statistical truncation
        composite.py                                 # Latent factor inference: stride-1 encoding and aggregation to exposure matrix B
    integration/
        action_specs.py                              # Machine-readable action specifications for diagnostic recommendations
        colab_drive.py                               # Google Drive integration for Colab persistence
        composite_scores.py                          # Composite diagnostic scores for the VAE Latent Risk Factor pipeline
        decision_rules.py                            # Decision rules engine for automated diagnostic interpretation
        diagnostic_experiments.py                    # Diagnostic experiments for systematic component substitution and parameter sweeps
        diagnostic_plots.py                          # Diagnostic plot generation for the full pipeline
        diagnostic_report.py                         # Diagnostic report generation in Markdown, JSON, and CSV formats
        diagnostic_schema.py                         # JSON Schema for diagnostic output validation
        diagnostics.py                               # Comprehensive diagnostic collection for the full pipeline
        notebook_helpers.py                          # Notebook helper functions for config assembly, JSON export, and diagnostic execution
        pipeline.py                                  # Full end-to-end pipeline orchestrator
        pipeline_state.py                            # Pipeline state management for incremental checkpointing and resume
        reporting.py                                 # Results compilation and report generation
        statistical_tests.py                         # Statistical tests for walk-forward comparison
        visualization.py                             # Visualization helpers for notebook dashboard display
    portfolio/
        cardinality.py                               # Cardinality enforcement for semi-continuous portfolio constraints
        constraints.py                               # Portfolio constraints and penalty functions
        entropy.py                                   # Two-layer Shannon entropy on factor risk contributions and its gradient
        frontier.py                                  # Variance-entropy frontier computation
        momentum.py                                  # Cross-sectional momentum signal (12-1 month)
        sca_solver.py                                # Sequential Convex Approximation (SCA) solver with Armijo backtracking
    risk_model/
        conditioning.py                              # Conditioning guard for cross-sectional regression
        covariance.py                                # Covariance estimation: Sigma_z (Ledoit-Wolf), D_eps (idiosyncratic), Sigma_assets assembly
        factor_quality.py                            # Factor quality metrics for diagnostic purposes
        factor_regression.py                         # Cross-sectional factor regression (OLS / iterative WLS)
        rescaling.py                                 # Dual rescaling of exposure matrix B_A
    training/
        batching.py                                  # Curriculum batch sampler for VAE training
        early_stopping.py                            # Early stopping with best checkpoint restoration
        scheduler.py                                 # Learning rate scheduler wrapper
        trainer.py                                   # VAE Training loop with curriculum batching, early stopping, and LR scheduling
    utils.py                                         # Hardware adaptation utilities: device detection, backend config, DataLoader kwargs, AMP config
    vae/
        build_vae.py                                 # VAE architecture construction: sizing rules and factory function
        decoder.py                                   # VAE Decoder: Linear projection + Transposed residual body + Output head
        encoder.py                                   # VAE Encoder: InceptionHead + Residual body + Projection to latent space
        loss.py                                      # VAE Loss computation: three modes (P/F/A), crisis weighting, co-movement loss
        model.py                                     # VAEModel: top-level VAE wrapper combining encoder and decoder
    validation.py                                    # Centralized validation utilities for runtime checks
    walk_forward/
        folds.py                                     # Walk-forward fold scheduling
        metrics.py                                   # Walk-forward metrics: 3 layers of evaluation
        oos_rebalancing.py                           # Out-of-sample periodic rebalancing simulation
        phase_a.py                                   # Phase A: Hyperparameter selection via nested validation
        phase_b.py                                   # Phase B: Deployment run
        selection.py                                 # Walk-forward fold scoring and final selection criterion
tests/
    fixtures/
        known_solutions.py                           # Known analytical solutions for verification of portfolio optimization,
        synthetic_data.py                            # Deterministic synthetic data generators for all unit and integration tests
    integration/
        test_e2e_pipeline.py                         # End-to-end integration test: synthetic 100-stock, 16-year pipeline
        test_integration.py                          # Integration tests for MOD-016: E2E pipeline orchestration
        test_pipeline_stages.py                      # Integration tests: verify data flows correctly between pipeline stages
        test_walk_forward.py                         # Integration tests for MOD-009: Walk-forward validation
    unit/
        test_benchmarks.py                           # Unit tests for MOD-010 to MOD-015: Benchmark models
        test_colab_drive.py                          # Unit tests for Google Drive persistence in Colab
        test_composite_scores.py                     # Unit tests for composite diagnostic scores
        test_config_validation.py                    # Tests for config dataclass __post_init__ validation
        test_conventions.py                          # Tests for the 10 project conventions (CONV-01 to CONV-10)
        test_data_pipeline.py                        # Unit tests for MOD-001: Data pipeline
        test_data_quality.py                         # Data quality and consistency tests for the Tiingo pipeline
        test_decision_rules.py                       # Unit tests for decision rules and causal graph analysis
        test_diagnostic_additions.py                 # Unit tests for the new diagnostic additions
        test_diagnostic_experiments.py               # Unit tests for diagnostic experiments module (src/integration/diagnostic_experiments.py)
        test_diagnostics.py                          # Unit tests for the diagnostics package
        test_divergences.py                          # Tests verifying alignment between the code and the reference documents
        test_edge_cases.py                           # Edge case and defensive tests for pipeline robustness
        test_factor_quality.py                       # Unit tests for factor quality diagnostics module
        test_inference.py                            # Unit tests for the inference module (src/inference/)
        test_interface_contracts.py                  # Interface contract tests verifying dtype and shape at every pipeline stage boundary
        test_invariants.py                           # P0-priority tests for the 12 critical invariants (INV-001 to INV-012)
        test_loss_function.py                        # Unit tests for the VAE loss computation module (src/vae/loss.py)
        test_oos_rebalancing.py                      # Unit tests for OOS periodic rebalancing simulation
        test_pipeline_state.py                       # Unit tests for pipeline state management
        test_portfolio_optimization.py               # Unit tests for the portfolio optimization module (MOD-008)
        test_risk_model.py                           # Unit tests for the risk model module (MOD-007)
        test_tiingo_keys.py                          # Tests for Tiingo API key resolution (env var → Colab Secrets → gitignored file)
        test_training.py                             # Unit tests for the VAE training module (src/training/)
        test_utils.py                                # Unit tests for hardware adaptation utilities (src/utils.py)
        test_vae_architecture.py                     # Unit tests for MOD-002: VAE architecture
        test_validation.py                           # Unit tests for centralized validation utilities
        test_visualization.py                        # Unit tests for visualization helper functions
        test_walk_forward_detailed.py                # Unit tests for walk-forward validation details
```

## Data and outputs outside git

`data/tiingo_us_equities.parquet` (469 MB) is versioned with Git LFS (`.gitattributes`); the rest is local only.

| Path | Content | Used by |
|---|---|---|
| `data/tiingo_raw/*.parquet` | one raw Parquet per ticker from the Tiingo API (ignored) | `scripts/download_tiingo.py --phase merge` |
| `results/diagnostic_runs/<timestamp>/` | diagnostic reports, plots, checkpoints (on Drive when on Colab) | `scripts/run_diagnostic.py`, notebooks |
| `~/mosek/mosek.lic` | MOSEK licence | CVXPY solvers |
