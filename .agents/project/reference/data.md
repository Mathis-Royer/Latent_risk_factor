# Data

Facts checked against the code and the data on 2026-11-07. The specification (`docs/ISD_vae_latent_risk_factors.md`,
MOD-001) assumes CRSP; the project actually runs on Tiingo, which lacks several CRSP fields. Read the quirks below
before interpreting any result.

## Pipeline schema (8 core columns)

Defined in `src/data_pipeline/data_loader.py::CORE_COLUMNS`; every loader returns them sorted by `(permno, date)`.

| Column | Type | Meaning |
|---|---|---|
| `permno` | int | stock identifier (CRSP name kept; for Tiingo a synthetic id, see below) |
| `date` | datetime | trading day |
| `adj_price` | float | split- and dividend-adjusted close |
| `volume` | int | shares traded (0 when missing) |
| `exchange_code` | int | 1 NYSE, 2 AMEX, 3 NASDAQ (CRSP convention) |
| `share_code` | int | CRSP share class (10/11 = common) |
| `market_cap` | float | capitalization, used to rank the universe |
| `delisting_return` | float | return on the delisting day, NaN otherwise |

## Tiingo panel (`data/tiingo_us_equities.parquet`, Git LFS, 469 MB)

- Built by `scripts/download_tiingo.py` (phases `prices`, `merge`; API calls only on explicit request; keys from
  `src/data_pipeline/tiingo_keys.py`). Raw per-ticker files in `data/tiingo_raw/` (ignored); metadata in
  `data/tiingo_meta/` (`supported_tickers.csv`, `ticker_to_permno.json` with 21,915 tickers, `download_progress.json`).
- Content: 24.86 M rows, 7,716 stocks, 1995-01-03 to 2026-02-20. Exchanges kept: NYSE, AMEX, NASDAQ, NYSE ARCA,
  NYSE MKT, BATS.
- Loaded by `load_tiingo_data()`: rows with `adj_price < 1` dropped; stocks with fewer than 504 trading days dropped.

### Specificities (they change results)

- **`permno` is synthetic.** Assigned per ticker (`ticker_to_permno.json`, from 10001). A ticker reused by another
  company, or a company changing ticker, is not linked as in CRSP.
- **`market_cap` is a liquidity proxy, not a capitalization.** `close × 63-day mean volume × 20`
  (`download_tiingo.py`, merge phase): the free tier has no shares outstanding. Universe ranking "by cap" is a
  ranking by dollar volume.
- **No delisting returns, and no imputation.** `delisting_return` is NaN on every row. `returns.py` imputes Shumway
  returns (-30% NYSE/AMEX, -55% NASDAQ) only for stocks that have a non-NaN `delisting_return` somewhere, so on Tiingo
  a delisted stock simply stops: its terminal loss is never booked (optimistic bias for every strategy, larger for
  those holding small or distressed names).
- **`share_code` is always 10 and unknown exchanges default to 1 (NYSE).** Share-class filters are inert.
- **Coverage of dead companies depends on Tiingo**, which lists delisted tickers but often returns no prices for them
  (empty body treated as "no data"). Survivorship is therefore not controlled.
- **VIX is not wired.** Crisis labels (`src/data_pipeline/crisis.py`) need VIX data; the CLI scripts pass
  `vix_data=None`, so crisis weighting is off in every scripted run.

## Other datasets

- `data/sp500_index.parquet` (ignored, regenerable): S&P 500 price index from `scripts/download_sp500.py`, used by
  the `sp500_index` benchmark.
- Synthetic data: `generate_synthetic_csv()` in `data_loader.py` produces the same schema with a factor structure,
  20 delistings (half with an explicit delisting return) and ~2% missing prices; used by tests and `--synthetic` runs.
