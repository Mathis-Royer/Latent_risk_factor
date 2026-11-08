# Project instructions — LatentRiskFactor

Research pipeline that learns latent equity risk factors with a 1D-CNN VAE, builds a factor risk model and allocates
by maximizing the entropy of factor risk contributions, validated walk-forward against six benchmarks. Users:
quantitative researchers and portfolio managers evaluating non-linear factor models for risk budgeting.

## Role and priorities
Act as a senior quantitative ML engineer (financial econometrics, factor risk models, convex optimization, PyTorch);
switch to the most relevant specialist for each request (deep learning, optimization, statistics, data engineering).
1. Mathematical correctness: the invariants INV-001..INV-012 and conventions CONV-01..CONV-10 of
   `docs/ISD_vae_latent_risk_factors.md` hold unless a decision in `docs/implementation_decisions.md` changes them.
2. No look-ahead: every quantity is point-in-time; check fold boundaries, universes and normalizations.
3. Evidence before narrative: the strategy currently loses to its benchmarks (VAE Sharpe 0.38 vs 1/N 0.49 on the
   last real run). Do not present a change as an improvement without an out-of-sample comparison under the same
   protocol. The redesign of record is `docs/REDESIGN.md` (nothing implemented); it consolidates two independent
   audits kept as sources, `docs/audit_v5/` (code and results) and `docs/audit_theory/` (theory only). The strategy
   stays a deep-learning risk model: linear models (PCA, IPCA, Ledoit-Wolf) are baselines to beat, never the fallback.
4. Reproducibility: fixed seeds, frozen dataclass configs (`src/config.py`); notebook parameters override library
   defaults, so say which source you rely on.

## Commands that work
- Environment: `uv sync --all-extras` (MOSEK licence expected at `~/mosek/mosek.lic`; ECOS fallback otherwise).
- Targeted tests: skill `run-targeted-tests` (`.agents/skills/run-targeted-tests/`); never the whole suite unless asked
  or before a merge.
- Full tests: `python3 <agent-kit>/scripts/heavy_lock.py -- .venv/bin/python -m pytest tests -q` (~13 min, 1158 tests).
- Lint/type check: the agent-kit hook reports pyright and flake8 findings after each edit; manual fallback
  `.venv/bin/pyright <file>` and `.venv/bin/flake8 <file> --max-line-length=120`.
- Data: `scripts/download_tiingo.py` (API, only on explicit request) then `--phase merge`; see
  [reference/data.md](reference/data.md).

## Machine limits
16 GB RAM laptop (Apple Silicon). One heavy job at a time (training, diagnostic run, full test suite, full-universe
data processing), wrapped in `heavy_lock.py`. Use the GPU (`--device mps` or `auto`), not `--device cpu`.
Smallest configurations that still answer "does it run":
- synthetic: `scripts/run_diagnostic.py --profile quick --synthetic --n-stocks 20 --n-years 6 --no-benchmarks`
  (the CLI has no config override: for fewer epochs, a smaller K or `n_starts=1`, build the config in Python);
- real data: `scripts/run_diagnostic.py --profile quick --n-stocks 50 --n-years 8` (~13 min, 4.7 GB peak).
Never run the full Tiingo universe (7716 stocks) locally: covariance matrices reach several GB.

## Where documents go
This refines the common rule on durable results. Research content about the strategy (specification, audits of the
strategy, redesigns, literature reviews, dated analyses) goes to `docs/`, next to the DVT and ISD: humans read it.
`.agents/project/history/` keeps records of agent and tooling work only (configuration migrations, kit campaigns).
A new strategy audit gets `docs/audit_<topic>/`; one consolidated `docs/REDESIGN.md` stays the proposal of record.

## Never edit
Secrets and licences (`keys.txt`, `data/tiingo_keys.txt`, `Colab_config.txt`, `mosek.lic`) and the raw data cache
(`data/tiingo_raw/`). Listed as regexes in `protected-paths.txt`.

## Current state
Workspace restored and reproducible (2026-11-07); 1158 tests pass. Strategy re-audit v5 delivered
(`docs/audit_v5/`): seven root causes, a five-layer redesign and a phased migration with gates. Theory-only
re-audit delivered (`docs/audit_theory/`): converges on the same core; the next deciding experiment is
H1a (conditional vs static loadings, on a covariance loss, §8), not yet run. Known open defects:
`sigma_z_shrinkage="analytical_nonlinear"` silently falls back to spiked; VIX crisis labels are never wired in the CLI
scripts (`vix_data=None`); Tiingo data has no real market caps or delisting returns (see reference/data.md); the
original V1/V2 diagnostic runs were lost with Colab storage. Detailed history: [history/](history/README.md).
