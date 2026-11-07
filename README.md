# Latent Risk Factor

Portfolio construction based on latent risk factors discovered by a 1D-CNN
Variational Autoencoder. The VAE learns non-linear factor exposures from equity
return windows; portfolios then maximise the Shannon entropy of factor risk
contributions. Walk-forward validation compares the strategy against six
benchmarks (1/N, inverse-vol, min-variance, ERC, PCA risk parity, PCA + vol)
plus the S&P 500.

Methodology: [`ISD_methodology.md`](ISD_methodology.md) and [`docs/`](docs/).

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git lfs install            # the Tiingo dataset is stored with Git LFS
git clone https://github.com/Mathis-Royer/Latent_risk_factor.git
cd Latent_risk_factor
uv sync --all-extras       # core + dev (pytest, pyright, pylint, flake8) + notebook
```

**MOSEK licence** (portfolio solver; ECOS/Clarabel are used as fallback):
place `mosek.lic` at `~/mosek/mosek.lic`. Academic licences are free at
<https://www.mosek.com/products/academic-licenses/>.

### Data

| File | Source | How to get it |
| ---- | ------ | ------------- |
| `data/tiingo_us_equities.parquet` | Tiingo, 1995 → 2026-02 | Included (Git LFS) |
| `data/tiingo_meta/` | Tiingo ticker metadata | Included |
| `data/sp500_index.parquet` | Yahoo Finance (`^GSPC`) | `uv run python scripts/download_sp500.py` (also auto-downloaded on first use) |

Optional credentials (environment variables, never commit them):

- `TIINGO_API_KEYS` — only to refresh/extend the equity dataset
  (`scripts/download_tiingo.py --phase all`). Resolved by
  `src/data_pipeline/tiingo_keys.load_tiingo_keys()`, first source found wins:
  1. environment variable (`export TIINGO_API_KEYS=key1,key2`, e.g. in `~/.zshrc`);
  2. Colab Secrets — key icon in the Colab sidebar, secret named `TIINGO_API_KEYS`,
     with notebook access enabled;
  3. `keys.txt` at the repo root, one key per line (gitignored).

  The dashboard never calls the API unless `DOWNLOAD_TIINGO = True` in its
  configuration cell, even when keys are available.
- `FRED_API_KEY` — VIX series for crisis labels (`src/data_pipeline/crisis.py`).
  The CLI scripts currently run without it (crisis weighting disabled).

## Usage

```bash
# Quick end-to-end check on synthetic data (no external data needed)
uv run python scripts/run_diagnostic.py --profile quick --synthetic

# Diagnostic on real data (full profile: 1-3 h)
uv run python scripts/run_diagnostic.py --profile full --data-dir data/

# Walk-forward validation / benchmarks only
uv run python scripts/run_walk_forward.py --data-dir data/
uv run python scripts/run_benchmarks.py --data-dir data/
```

Each script accepts `--help`. Outputs go to `results/` and TensorBoard logs to `runs/`.

Notebooks (`notebooks/dashboard.ipynb`, `notebooks/diagnostic_experiments.ipynb`)
run locally or on Google Colab (Drive persistence is enabled only on Colab).
Select the `.venv` kernel.

## Tests

```bash
uv run pytest            # ~1160 tests, ~11 min
```

## Layout

```text
src/
  data_pipeline/   loading, returns, features, crisis labels
  vae/             model and loss
  training/        training loop
  inference/       composite profiles → exposure matrix B
  risk_model/      rescaling, factor covariance, Sigma_assets
  portfolio/       entropy objective, SCA solver
  benchmarks/      reference strategies
  walk_forward/    folds, phase A/B, OOS rebalancing
  integration/     end-to-end pipeline, diagnostics, reporting
scripts/           CLI entry points
tests/             unit and integration tests
```
