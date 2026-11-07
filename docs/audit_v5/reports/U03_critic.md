# U03 — critic

## Scope read

- `docs/DVT_strategie_facteurs_risque_latents_v4_1.md` — §4.2 in full (`:202-296`), catalogue lines `:977-992`, survivorship note `:736`, references `:1622`
- `docs/ISD_vae_latent_risk_factors.md` — MOD-001 (`:544-830`), dependency table `:281-294`, interface assertions `:2658-2662`, constraints `:889`
- `docs/data_preprocessing_diagnostic.md` — data-flow diagram `:20-80`, filtering tables `:82-155`, transformations `:157-206`, split logic `:208-260`, validation gaps `:262-380`, "Research Best Practices Conformity" `:415-505`
- `docs/divergences.md` (first 80 lines, plus grep for `tiingo|universe|delisting|market_cap`), `docs/implementation_decisions.md:100-160`, `docs/Benchmark_latent_risk_factor.md:15`
- `src/data_pipeline/data_loader.py` (whole), `returns.py` (whole), `universe.py` (whole), `windowing.py` (whole), `features.py` (whole), `crisis.py` (head), `tiingo_keys.py` (skipped — contains keys)
- `scripts/download_tiingo.py` (`:80-95`, `:295-330`, `:332-470`, `:530-560`, `:800-980`), `scripts/download_sp500.py`, `scripts/run_walk_forward.py:100-220`, `scripts/run_diagnostic.py:83-135`, `:280-400`, `:420-460`, `:600-697`, `scripts/run_benchmarks.py` (grep)
- `src/integration/pipeline.py:520-640`, `:1240-1300`, `:1440-1470`, `:1965-2010`, `:2160-2200`; `src/integration/diagnostics.py:1355-1400`; `src/walk_forward/oos_rebalancing.py:100-220`, `:640-700`; `src/benchmarks/sp500_index.py:30-200`; `src/training/trainer.py:270-300`; `src/vae/loss.py:40-75`; `src/config.py:95-160`
- `tests/unit/test_conventions.py:560-600`, `tests/unit/test_data_pipeline.py` (grep), `tests/unit/test_data_quality.py:380-470`
- `notebooks/dashboard.ipynb` (grep: `N_STOCKS`, `N_YEARS`, `DATA_SOURCE`, `vix_data`)
- Data facts computed read-only with pyarrow over `data/tiingo_us_equities.parquet` (24,864,645 rows × 8 cols), `data/sp500_index.parquet`, `data/tiingo_meta/supported_tickers.csv`, `data/tiingo_meta/ticker_to_permno.json`
- `results/diagnostic/{config.json, diagnostic_data.json, diagnostic_report.md}`, `results/{config.json, report.json}`

---

## Five questions

**1. Purpose and original intent of this unit.**
To supply an *unbiased, point-in-time* representation of the tradable US large-cap opportunity set with intact tails. The DVT is explicit: daily log-returns of dividend/split-adjusted prices (`DVT:206-208`); per-window z-scoring with F=2 (`DVT:224-258`); a universe reconstructed "as it existed at each historical date" with float-adjusted cap ≥ $500M / exit $400M, ADV ≥ $2M, 504-day listing, common equity only, NYSE/AMEX/NASDAQ (`DVT:261-291`); Shumway delisting imputation (`DVT:265-267`); CRSP as the reference source (`DVT:291`). ISD MOD-001 repeats this and adds the outputs `universe_snapshots`, `returns_df`, `trailing_vol` (`ISD:576-590`) and the warning that "errors here silently corrupt the entire pipeline" (`ISD:551-553`).

**2. What a state-of-the-art version would look like.**
A versioned panel artifact built from a survivorship-bias-free source with an explicit delisting file (CRSP daily stock file + delisting events with codes and delisting returns; Shumway 1997, *The Delisting Bias in CRSP Data*, JF; Shumway & Warther 1999, already cited at `DVT:1622`; Beaver, McNichols & Price 2007 on delisting-return corrections). Membership should come from a point-in-time index (CRSP index constituents / Russell snapshots) rather than being re-derived from a market-cap proxy; returns should be constructed from unadjusted prices plus a point-in-time corporate-action table, not from retroactively back-adjusted levels; halts, stale quotes and non-trading days should be first-class states with masks, not silent zeros; the whole panel should carry a manifest (source, vintage, row counts, membership hash) and be consumed identically by training, risk model, benchmarks and validation. Gu–Kelly–Xiu (2020, *Empirical Asset Pricing via Machine Learning*) is the standard reference for such a documented panel; Frazzini–Israel–Moskowitz (2018) / Novy-Marx–Velikov (2016) for why a liquidity screen must be applied at the date, not by an ex-post cap rank.

**3. What prevents the current unit from reaching it.**
(a) The vendor does not carry the fields the schema demands, and the code fills them with constants: no shares outstanding → `market_cap = close × avg_volume × 20` (`download_tiingo.py:890-901`); no delisting metadata → `delisting_return = NaN` on every row (`download_tiingo.py:911`); `share_code = 10` universally (`:909`). (b) The one module implementing the spec is orphaned: `construct_universe`/`build_training_universe` are called only from tests (grep: `tests/unit/test_data_pipeline.py:29`, `test_data_quality.py:26`, `test_conventions.py:17`). (c) The production universe is an ex-post full-sample top-N on a liquidity proxy (`data_loader.py:344-348`). (d) The cleaning chain destroys exactly the two things the theory needs — early winner history and the terminal collapse leg (`download_tiingo.py:930-935` on back-adjusted levels). (e) Missing data is resolved by membership exclusion plus zeros, not by masking (`windowing.py:213-228`). (f) The crisis channel is never fed (`run_diagnostic.py:444`). None of these is a compute constraint: 24.8M rows × 8 columns is comfortably in memory on this machine.

**4. Which constraints are real and which are self-imposed.**
*Real*: the vendor lacks shares outstanding and delisting codes; adjusted prices are retroactively recomputed; free sources cannot supply PIT index membership; the sample cannot start before 1995 for this vendor; 16 GB RAM caps the window tensor, not the panel.
*Self-imposed*: the `n_stocks` cap itself (a modelling choice masquerading as a data rule, `data_loader.py:323-355`); the $1 filter on *adjusted* prices (`download_tiingo.py:930-935`, `data_loader.py:230-231`); the 5% per-window NaN tolerance instead of a mask (`windowing.py:213-215`); `nan_to_num(..., 0.0)` for missing returns (`windowing.py:227`); the blanket −30%/−55% applied to *all* missing return events (`oos_rebalancing.py:661-675`); the explicit decision that "No `delisting_code` column is needed — the strategy never uses it" (`ISD:604-608`), which forecloses the correct treatment; the price-only S&P benchmark (`sp500_index.py:88-95,173-179`); and the independent-GBM synthetic generator used to "validate end-to-end" (`data_loader.py:103-105`).

**5. What should be removed, merged or moved.**
*Remove*: `src/data_pipeline/universe.py` entirely (dead), `returns.winsorize_returns` (never called), `DataPipelineConfig.min_valid_fraction` (never read), `vix_lookback_percentile` and `crisis.py` unless VIX is actually ingested, the `market_cap` field (rename to `dollar_volume_proxy` or delete), the hard `|r|≤2.0` clip (cosmetic), and the `SP500TotalReturn` name/claim (it is a price index). *Merge*: membership, calendar and price cleaning into a single **panel-build** step that emits one artifact with a manifest; delisting handling into the returns layer (one place, with event types) instead of the OOS simulator; the two volatility series (21d feature, 252d risk scaling) into one documented feature store. *Move*: universe construction out of the strategy runtime into a versioned, checksummed data artifact; fold-invariant data facts (calendar, membership, missingness mask) out of per-fold code. Adding another filter or another guard is not the answer; the unit needs a smaller surface with a stronger contract.

---

## Load-bearing assumptions

| # | Assumption | Where stated | Evidenced? | Comment |
|---|------------|--------------|------------|---------|
| A1 | The universe is point-in-time and survivorship-bias-free | `ISD:551`, `DVT:265`, `data_preprocessing_diagnostic.md:432` | no | `universe.py` is never called in production; selection is full-sample (`data_loader.py:346-347`) |
| A2 | `market_cap` is float-adjusted market capitalisation | `ISD:591-599`, `DVT:269` | no | It is `close × 63d avg volume × 20` (`download_tiingo.py:890-901`) |
| A3 | Delisting returns exist or are imputed per Shumway | `returns.py:48`, `ISD:604-611`, `DVT:265` | no | 0 non-null values in 24.8M rows; the imputation branch is unreachable (`returns.py:189-195`) |
| A4 | Missing data are rare on the target universe | `DVT:258` | no | 35.31% of the panel; 410/788 stocks >20% missing (`diagnostic_report.md:29-31`) |
| A5 | The vol channel restores absolute volatility scale and cross-sectional variance co-movement | `DVT:245-252` | no | Vol is log-transformed then z-scored per window (`windowing.py:248-260`) |
| A6 | Crisis weighting γ=3 focuses training on crisis windows | `DVT` §4.4, `dashboard.ipynb:419` | no | `vix_data=None` in every entry point (`run_diagnostic.py:444`, `dashboard.ipynb:1151,1217`) |
| A7 | 504 trading days of listing ensures a complete window | `DVT:273`, `ISD:700-712` | partially | Implemented as a *row count over the whole file* (`data_loader.py:237`), not contiguity |
| A8 | A $1 floor on adjusted prices is an innocuous quality filter | `download_tiingo.py:930-935` | no | Deletes early history of big winners and the terminal collapse leg (AAPL/NVDA below) |
| A9 | Tiingo supplies the delisted names the encoder needs | implied by `DVT:261-267` | partially | 3,886 permnos end before the sample end (computed), so dead names exist as price series — but with no delisting event/return, and the tail is truncated |
| A10 | Synthetic data validates the pipeline end-to-end | `data_loader.py:9-12` | partially | Validates shapes/invariants; contains no common factors, no clustering, no fat tails |
| A11 | S&P 500 is a comparable benchmark | `sp500_index.py:173-179`, `Benchmark_latent_risk_factor.md:15` | no | Price return vs total-return stock data |
| A12 | Exchange codes are stable over history | implied by `ISD:604-608` | no | Exchange map is read from the *current* metadata snapshot (`download_tiingo.py:836-845`) |

---

## Findings

| ID | Type | Severity | Evidence register | Claim (quoted, path:line) | Assessment | Evidence (path:line) | Cross-unit |
|----|------|----------|-------------------|---------------------------|------------|----------------------|------------|
| U03-CRI-01 | doc-vs-code / leakage | critical | doc+code | "CONV-10: No future data in any computation; universe reconstituted at each date" (`src/data_pipeline/universe.py:4`) | The point-in-time universe module is **orphaned**: no production path calls `construct_universe`/`build_training_universe`; only unit tests do. Every eligibility rule in §4.2 (cap, ADV, listing, security type, exchange, liveness) is therefore inoperative. The "tradable set" is whoever has a price print that day. | `universe.py:13-124`; call sites only `tests/unit/test_data_pipeline.py:29`, `tests/unit/test_data_quality.py:26`, `tests/unit/test_conventions.py:17`; production loaders `data_loader.py:415,425`; `run_walk_forward.py:158-160`; pipeline universe construction `pipeline.py:2170-2183` | U09, U12, U13 |
| U03-CRI-02 | leakage / selection | critical | code+results | "median_mcap = df.groupby('permno')['market_cap'].median(); top_permnos = list(median_mcap.nlargest(n_stocks).index)" (`src/data_pipeline/data_loader.py:346-347`) | The production universe is selected with **full-sample information**. Replicating the rule on the parquet: 126/788 (16.0%) of selected names are dead before 2025 vs 2,859/7,716 (37%) in the full file, and the live cross-section grows 298 (1995) → 645 (2020) — so early folds trade a survivor-tilted, ~300-name set while the model is told n=788. | `data_loader.py:344-348`; computed from `data/tiingo_us_equities.parquet`; panel size `results/diagnostic/diagnostic_report.md:22` | U12, U15 |
| U03-CRI-03 | data / modeling | critical | code+data | "Since Tiingo free tier lacks shares outstanding, we use: close × average_volume × 20 as a rough market cap proxy" (`scripts/download_tiingo.py:890-893`) | `market_cap` is a turnover proxy, not a market cap. The DVT's $500M float-adjusted threshold has no referent (and is never applied anyway, CRI-01). Rank is dominated by trading activity: BRK-A (permno 12939) ranks 9th with a ≈$41B proxy against a ≈$1T actual cap, while GOOGL and GOOG both rank top-10. | `download_tiingo.py:890-901`; `DVT:269`; `ISD:591-599`; computed top-15 table from the parquet | U09 |
| U03-CRI-04 | data / honesty | critical | doc+code+data | "`delisting_return`: np.nan" (`scripts/download_tiingo.py:911`); docstring: "If NaN: impute per Shumway (1997): -30% NYSE/AMEX, -55% NASDAQ" (`returns.py:48`) | The merge writes NaN for all 24,864,645 rows (verified: 0 non-null). `_extract_delisting_info` only registers permnos that already have a non-NaN delisting return (`returns.py:189-195`), so `_apply_delisting_returns` **never imputes**. The training data contains no delisting return at all — the exact bias §4.2 says it avoids. The only remaining treatment is the OOS heuristic that fires on *any* NaN return of a held name and applies −30%/−55% regardless of delisting type (`oos_rebalancing.py:661-675`), misapplying Shumway, whose figures are for performance delistings, not mergers. | `download_tiingo.py:911`; `returns.py:169-198`, `:200-248`; `oos_rebalancing.py:661-675`; `ISD:604-611`; `data_preprocessing_diagnostic.md:183`; parquet count 0 | U12, U16 |
| U03-CRI-05 | leakage / data | critical | code+data | "Filter 2: Remove penny stock rows (adj_price < min_price)" (`scripts/download_tiingo.py:930-931`), with `adj_price` "Split- and dividend-adjusted" (`ISD:597`) | Filtering the **back-adjusted** price level at $1 makes a stock's inclusion in each year depend on its *future* split/dividend history. Hard evidence: AAPL (permno 10061) has `startDate 1980-12-12` in the vendor metadata but its first row is 2000-03-21; NVDA (23854) lists 1999-01-22 but starts 2016-05-13. The same filter deletes rows below $1 at the *end* of a collapse, amputating the bankruptcy tail the encoder is supposed to learn. Both effects correlate with returns — this is survivorship bias reintroduced through the cleaning layer. | `download_tiingo.py:890,930-935`; `data_loader.py:230-231`; `data/tiingo_meta/supported_tickers.csv` (AAPL/NVDA startDate); computed first rows from parquet; `ISD:597`; `DVT:206-208` | U05, U12 |
| U03-CRI-06 | data / implementation | major | code+results | "`ret_nan_frac <= 0.05` … `ret_clean = np.nan_to_num(ret_wins, nan=0.0)`" (`src/data_pipeline/windowing.py:215-227`) | 35.31% of the panel is missing and 410/788 stocks exceed 20% missing. The code resolves missingness in two incompatible ways: whole windows are dropped (>5% NaN) while surviving NaNs are turned into **exact zero returns**; the vol channel is backward-filled within the window. The loss receives no mask, so fabricated flats are indistinguishable from real quiet days. Separately, the exposure matrix covers only 639 of 788 names (`B_shape [639,52]`) and the report never explains the 149-name attrition. | `windowing.py:213-233`; `diagnostics.py:1361-1368`; `results/diagnostic/diagnostic_report.md:29-31`; `results/diagnostic/diagnostic_data.json` (`latent.B_shape`) | U04, U06 |
| U03-CRI-07 | math / logic | major | code+doc | "The 21-day rolling volatility captures volatility dynamics … cross-stock co-movements in variance" (`DVT:247-252`) | Self-contradictory: the feature is log-transformed and then z-scored **per window, per stock**, which removes exactly the absolute scale and the cross-sectional co-movement it is claimed to add. Two windows with the same shape but 10% vs 40% vol produce identical inputs. F=2 therefore adds only the *shape* of the log-vol path, not a volatility-level or variance-common-factor channel. | `windowing.py:248-260`; `DVT:224-252`; `DVT:303` (F=2 baseline) | U04, U05 |
| U03-CRI-08 | consistency / data | major | code | "`vix_data=None`" (`scripts/run_diagnostic.py:444`); "if `vix_data is not None` … compute_crisis_labels" (`src/integration/pipeline.py:1450-1458`) | VIX is never ingested on any path (diagnostic, walk-forward, notebook). With `crisis_fractions=None` the trainer sets `cf = 0` (`trainer.py:288-290`) and `gamma_eff = 1` (`loss.py:69`), so γ=3 is inert in every recorded run — while the notebook UI presents it as an active mechanism (`dashboard.ipynb:419`). | `run_diagnostic.py:444`; `dashboard.ipynb:1151,1217`; `pipeline.py:1450-1458`; `trainer.py:288-290`; `loss.py:69`; `DVT` §4.4 | U06, U15 |
| U03-CRI-09 | consistency | major | code | "S&P 500 price-return benchmark. Uses ^GSPC daily prices" (`src/benchmarks/sp500_index.py:173-179`) | The strategy's stock returns come from dividend-inclusive `adjClose` (`download_tiingo.py:875-880`; `DVT:206`), the market benchmark is price-only, and the class is nevertheless named `SP500TotalReturn` (`sp500_index.py:173`). The comparison is asymmetric by roughly the dividend yield, material when Sharpe differences are ~0.1-0.3 and annual returns are compared to 13.6%. | `sp500_index.py:88-95,173-183`; `download_tiingo.py:875-880`; `DVT:206` | U13 |
| U03-CRI-10 | data / interpretation | major | code+results | "eps = rng.randn(n_days, n_stocks); log_returns = mu + sigma * eps" (`src/data_pipeline/data_loader.py:103-104`) | The synthetic generator draws an independent shock per stock per day and fixes σ per stock: **no common factors, no volatility clustering, no fat tails, no correlated delistings**. It still "validates end-to-end" (`data_loader.py:9-12`), and the only walk-forward artifacts are 20 synthetic stocks with E*=1 (`results/config.json:3,9`). Claims that the pipeline "works" are therefore claims about plumbing, not about the model's purpose. | `data_loader.py:99-115`; `results/config.json:3,9`; `results/report.json` | U05, U12, U14 |
| U03-CRI-11 | consistency | major | code | "Build universe snapshots for estimation rescaling (point-in-time)" (`src/integration/pipeline.py:2170`) | The snapshot is `valid_ids = [sid for sid in inferred_stock_ids if pd.notna(row[sid])]` (`:2175-2183`) — "has a print today", not U_t. It includes sub-$500M names, ADRs and untradeable prints, excludes stocks on a missing-data day (which the simulator then treats as delistings), and falls back to **all** ids on dates absent from the index (`:2185`). The "point-in-time" label overstates what the object is. | `pipeline.py:2170-2185` | U09, U12 |
| U03-CRI-12 | implementation | minor | code | "Winsorize extreme returns caused by Tiingo API data corruption … Clip to |r| <= 2.0" (`src/data_pipeline/returns.py:99-110`) | Three incompatible stories: the live rule is a hard symmetric clip at e^2 ≈ +639% / −86.5%, which fires on 6 of 3,959,331 stock-days in the selected panel (computed) — cosmetic; the percentile winsorizer (`returns.py:251-289`) is never called; and the diagnostic doc still lists "No outlier winsorization" as a gap (`data_preprocessing_diagnostic.md:443`). | `returns.py:99-110,251-289`; `data_preprocessing_diagnostic.md:443,452`; computed zero/outlier fractions | U04 |
| U03-CRI-13 | implementation | minor | code | "df.groupby('permno').filter(lambda g: len(g) >= min_history_days)" (`src/data_pipeline/data_loader.py:237`) | "Continuous listing ≥ 504 days" (`DVT:273`) is implemented as a row **count over the whole file**; a stock with 504 scattered prints across 30 years qualifies, and the real constraint ends up being the per-window NaN filter. `min_valid_fraction` exists and is validated (`config.py:137,150-152`) but no module reads it. | `data_loader.py:236-238`; `download_tiingo.py:954-958`; `config.py:137`; grep: no other use | U15 |
| U03-CRI-14 | data | minor | code | "`exchange_map[ticker] = EXCHANGE_CODE_MAP.get(exchange, 1)`" from `supported_tickers.csv` (`scripts/download_tiingo.py:836-845`) | The exchange code is the *current* listing venue applied to all history. A stock that moved NASDAQ→NYSE is treated as NYSE for its whole past, which flips the Shumway branch (−55% vs −30%) used by the OOS simulator. | `download_tiingo.py:836-845`; `oos_rebalancing.py:153-165` | U12 |
| U03-CRI-15 | logic | minor | code | "returns_df.at[last_date, permno] = log_ret" (`src/data_pipeline/returns.py:246`) | Delisting return is **overwritten** on the last row, not added to it; the last observed daily return is lost, and a positive (merger) delisting return replaces rather than compounds. Latent today (branch unreachable, CRI-04), but it is the kind of contract error that survives until the data is upgraded. | `returns.py:200-248` | — |
| U03-CRI-16 | honesty / doc-vs-code | major | doc+code | "Point-in-time universe | CONV-10 in universe.py | No look-ahead bias" (`docs/data_preprocessing_diagnostic.md:432`); "survivorship-bias-free handling" (`ISD:551`) | The project documents these properties as **achieved**. They are not: the module is orphaned, the source is Tiingo, selection is ex post, delisting returns are absent. The Tiingo substitution is mentioned in exactly one doc — a pipeline diagram (`data_preprocessing_diagnostic.md:23-26`) — and in no decision record (`docs/implementation_decisions.md:126-150` still treats EODHD as deferred; `docs/divergences.md` lists none of this). | `data_preprocessing_diagnostic.md:23-26,183,432`; `ISD:551,604-611`; `DVT:291,736`; `implementation_decisions.md:126-150` | U16, U17 |
| U03-CRI-17 | data / modeling | major | code+data | "share_code: 10" for every mapped row (`scripts/download_tiingo.py:909`) | `share_code ∈ {10,11}` (`universe.py:63-66`) therefore never filters: no ADR, REIT, preferred or dual-class exclusion. Dual share classes are distinct permnos with near-identical returns — GOOGL (18452) and GOOG (18451) both rank in the top-10 of the ex-post universe, BRK-A/BRK-B both present — double-counting one company's risk in B, in the cross-sectional regression, and in the entropy objective. | `download_tiingo.py:909`; `universe.py:63-66`; computed top-15 by median proxy cap | U05, U09 |
| U03-CRI-18 | reproducibility | major | results | "`"n_stocks": 2000` … `"data_source": "synthetic"`" (`results/diagnostic/config.json:3,9`) | The recorded config cannot reproduce the run: the panel has 788 names (`diagnostic_report.md:22`, `_diagnostic.n_stocks_actual=788`), the config says 2000, and the data-source field says synthetic for a Tiingo run. The selection rule that produced 788 is recorded nowhere. `results/config.json` is the same pattern at 20 stocks/synthetic with E*=1. A reader cannot tell which universe any number refers to. | `results/diagnostic/config.json:3,9`; `results/diagnostic/diagnostic_report.md:22,271-272`; `results/config.json:3,9` | U12, U15, U16 |

---

## Root-cause hypotheses

**R1 — The data contract was never renegotiated when the source changed.** The pipeline was built against a CRSP-shaped schema (float-adjusted cap, share codes, delisting returns, delisting codes) and then fed a vendor snapshot that provides only adjusted OHLCV. Instead of redefining the contract, the loader filled the required fields with constants: `share_code=10`, `delisting_return=NaN`, `market_cap=close·volume·20`, `exchange_code` from a current snapshot. Every downstream module then reasons as if it had CRSP semantics — this single decision generates CRI-01/02/03/04/04/17 and the false documentation in CRI-16.
*Structural alternatives:* (a) **Minimal honest contract** — keep only fields the vendor truly supports, delete cap/ADV/share-code/delisting logic, and rename the probe field `dollar_volume_proxy`; trade-off: the strategy loses the large-cap screen and must defend trading costs differently. (b) **Vendor upgrade with a conformance gate** — CRSP/EODHD/Norgate with a field-by-field provenance manifest and a CI test that fails on constant columns; trade-off: cost and a hard dependency, but it is the only option that restores the stated theory. (c) **Split responsibilities** — prices from a cheap vendor, membership from an independent PIT index source; trade-off: reconciliation work, but removes the circularity of deriving the universe from the same prices used to trade.

**R2 — "Universe" is a CLI parameter, not a time series.** Three entry points choose it three different ways (7,716 names in `run_walk_forward`, 4,500 in the notebook, 1,000/788 in the diagnostic), the one function that would produce U_t is orphaned, and there is no U_t object anywhere in production. Symptoms: CRI-01, 02, 11, 13, 18, and the 298→645 cross-section drift that quietly changes what the entropy objective is optimising.
*Structural alternatives:* (a) **Materialise U_t once** as a versioned artifact (date → permno set) consumed read-only by training, risk model, benchmarks and validation; trade-off: a build step and storage, but the only way to make membership auditable. (b) **Membership-file-driven** (index constituents at each reconstitution); trade-off: still needs a PIT source but decouples membership from prices entirely. (c) **Universe-free** — model the full panel and let shrinkage/regularisation decide relevance; trade-off: removes selection bias but increases n and noise and makes the "n=1,000" capacity arguments vacuous.

**R3 — Convenience cleaning: level-based filters and silent imputation.** The chain ($1 adjusted-price filter → 504-row count → ffill ≤5 → NaN→0 → 5% NaN window tolerance → 20% zero-return filter → |r|≤2 clip) is ad hoc and each step trades tail information for memory or simplicity. It explains CRI-05, 06, 12, 13 and half of CRI-04. The pattern is consistent across units: when a genuinely hard data problem appears, the code deletes evidence rather than marking it.
*Structural alternatives:* (a) **Flag, don't delete** — keep rows, attach validity/provenance flags (ok / halt / stale / missing / delisting / below-threshold), and pass a mask into the loss; trade-off: every downstream consumer must handle masks. (b) **A single-purpose cleaning layer** with a price-validity state machine and an explicit PIT corporate-action table; trade-off: real engineering effort, and it makes the panel bigger. (c) **Replace the level filter with a date-level liquidity screen** (ADV at date t), which is the economically meaningful version of what the $1 filter is trying to approximate; trade-off: still a selection rule, but at least a defensible, point-in-time one.

**R4 — Tail risk is asserted in the theory and suppressed in the data.** The DVT justifies crisis weighting and the encoder's role by tail-risk learning (`DVT:263-267`), the notebook advertises γ=3, and the loss implements it — but there is no VIX, no delisting return, the collapse leg is truncated at $1, and the universe over-samples survivors. The model is trained mostly on calm, liquid, surviving windows, then asked to diversify factor risk in crises. Symptoms: CRI-04, 05, 08, and the absent crisis diagnostics.
*Structural alternatives:* (a) **Distress sample** — deliberately harvest bankrupt/delisted windows with coded returns and weight them; trade-off: requires delisting metadata, the thing the vendor lacks. (b) **Regime-balanced sampling** by realized-volatility quantile instead of a VIX label; trade-off: no external data needed, but the "crisis" definition becomes endogenous. (c) **Drop the crisis claim** and confront tail risk at the portfolio layer (expected-shortfall constraint / EVT tail model); trade-off: intellectually honest, but abandons a core selling point of the VAE.

**R5 — Retroactively adjusted prices are treated as immutable history.** Tiingo's `adjClose` is recomputed through today, so a stock's presence in 1995 depends on 2026 split factors (CRI-05). This is a *distinct* look-ahead from universe selection and is invisible in every diagnostic because it lives in the data-availability structure rather than in a formula.
*Structural alternatives:* (a) **Unadjusted prices + PIT corporate-action table**, returns rebuilt as-of each date; trade-off: more infrastructure, but the only way to remove the effect. (b) **Per-stock total-return series with a documented adjustment vintage** and no filtering on the adjusted level; trade-off: residual vintage bias, much simpler. (c) **Accept and document** the bias with a sensitivity band estimated by re-running on unadjusted data; trade-off: cheap, but leaves a known bias in every number.

**R6 — Validation was designed for artifact completeness, not data provenance.** Runs record configs that do not describe their data (`results/diagnostic/config.json:3,9`), so the same strategy name refers to four different universes. Symptoms: CRI-18 and the impossibility of attributing V1/V2/Sharpe numbers to a defined panel.
*Structural alternatives:* (a) **Data manifest with SHA + row counts + membership hash**, checked at run start and stored with results; trade-off: mild plumbing, eliminates a whole class of ambiguity. (b) **One canonical builder** with no per-entry-point filters; trade-off: less flexibility for quick experiments. (c) **Frozen snapshot per study** with an explicit "study id"; trade-off: storage, but makes cross-run comparison meaningful.

---

## Verdict

This unit is a CRSP-shaped façade over a price-only vendor snapshot, and the gap between the façade and the substrate is where every load-bearing claim of the strategy quietly fails. The spec demands a point-in-time, survivorship-bias-free universe with float-adjusted caps, ADV and listing screens, Shumway delisting returns and CRSP-grade provenance (`DVT:261-291`, `ISD:551-611`). The code contains a correct-looking implementation of exactly that — and never calls it; production instead takes a full-sample top-N by a dollar-volume proxy, on a vendor panel whose `delisting_return` is NaN everywhere and whose `share_code` is a constant, filtered at $1 on *back-adjusted* prices, which deletes the early history of Apple and Nvidia and the final leg of every collapse. Missing data is 35% of the panel and is resolved by dropping windows and zero-filling the rest, without a mask. The crisis channel that is supposed to compensate for the missing tail is never fed. The documentation nevertheless certifies the properties as achieved (`data_preprocessing_diagnostic.md:432`, `ISD:551`) and the run configs cannot even identify the universe they used. None of this is a compute constraint; it is a design choice to keep a schema the data cannot honour. The unit is therefore not "almost right with local defects": its central assurances are false, and any conclusion drawn from the real-data runs — including the sign and magnitude of the VAE's underperformance — is confounded by selection and tail truncation of unknown but clearly material size. The strongest steelman is that Tiingo does include 3,886 since-dead permnos with full pre-death price history, so the encoder is not trained on a pure survivor panel; the rebuttal is that those histories are truncated exactly at the $1 boundary, carry no delisting event, and are then re-selected ex post, so their presence does not restore the tail the theory needs. Fixing this is not a patch to `returns.py`; it is a redefinition of the data artifact, its provenance and the contract every other unit consumes.

Score: 2/10

---

## Machine companion

```json
[
  {
    "id": "U03-CRI-01",
    "unit": "U03",
    "role": "critic",
    "type": "doc-vs-code",
    "severity": "critical",
    "evidence_register": "doc+code",
    "claim": "CONV-10: No future data in any computation; universe reconstituted at each date",
    "claim_location": "src/data_pipeline/universe.py:4",
    "assessment": "The point-in-time universe module is orphaned: construct_universe/build_training_universe are called only from unit tests. No production path applies cap, ADV, listing, security-type, exchange or liveness eligibility; the tradable set is whoever has a price print that day.",
    "evidence": ["src/data_pipeline/universe.py:13-124", "tests/unit/test_data_pipeline.py:29", "tests/unit/test_data_quality.py:26", "tests/unit/test_conventions.py:17", "src/data_pipeline/data_loader.py:415", "scripts/run_walk_forward.py:158-160", "src/integration/pipeline.py:2170-2183"],
    "cross_units": ["U09", "U12", "U13"],
    "root_cause": "R1/R2",
    "confidence": "high"
  },
  {
    "id": "U03-CRI-02",
    "unit": "U03",
    "role": "critic",
    "type": "leakage",
    "severity": "critical",
    "evidence_register": "code+results",
    "claim": "median_mcap = df.groupby('permno')['market_cap'].median(); top_permnos = list(median_mcap.nlargest(n_stocks).index)",
    "claim_location": "src/data_pipeline/data_loader.py:346-347",
    "assessment": "The production universe is selected with full-sample information. Replicating the rule, 16.0% of the 788 selected names are dead before 2025 versus 37% of all 7,716 names, and the live cross-section grows 298 (1995) to 645 (2020); early folds trade a survivor-tilted subset of a panel that is already survivor-tilted.",
    "evidence": ["src/data_pipeline/data_loader.py:344-348", "data/tiingo_us_equities.parquet (computed)", "results/diagnostic/diagnostic_report.md:22"],
    "cross_units": ["U12", "U15"],
    "root_cause": "R2",
    "confidence": "high"
  },
  {
    "id": "U03-CRI-03",
    "unit": "U03",
    "role": "critic",
    "type": "data",
    "severity": "critical",
    "evidence_register": "code+data",
    "claim": "Since Tiingo free tier lacks shares outstanding, we use: close x average_volume x 20 as a rough market cap proxy",
    "claim_location": "scripts/download_tiingo.py:890-893",
    "assessment": "market_cap is a turnover proxy, not a market cap, so the float-adjusted $500M threshold has no referent. Rank follows trading activity: BRK-A ranks 9th with a ~$41B proxy against a ~$1T actual cap, and GOOGL/GOOG both rank top-10.",
    "evidence": ["scripts/download_tiingo.py:890-901", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:269", "docs/ISD_vae_latent_risk_factors.md:591-599"],
    "cross_units": ["U09"],
    "root_cause": "R1",
    "confidence": "high"
  },
  {
    "id": "U03-CRI-04",
    "unit": "U03",
    "role": "critic",
    "type": "honesty",
    "severity": "critical",
    "evidence_register": "doc+code+data",
    "claim": "delisting_return: np.nan (merge); docstring says If NaN: impute per Shumway (1997)",
    "claim_location": "scripts/download_tiingo.py:911",
    "assessment": "All 24,864,645 rows carry a NaN delisting_return (verified 0 non-null), and _extract_delisting_info only registers permnos that already have a non-NaN value, so the Shumway imputation branch is unreachable. Training data contains no delisting return. The only residual treatment is the OOS heuristic that fires on any NaN return of a held name and applies a blanket -30%/-55% regardless of delisting type.",
    "evidence": ["scripts/download_tiingo.py:911", "src/data_pipeline/returns.py:169-198", "src/data_pipeline/returns.py:200-248", "src/walk_forward/oos_rebalancing.py:661-675", "docs/ISD_vae_latent_risk_factors.md:604-611", "docs/data_preprocessing_diagnostic.md:183"],
    "cross_units": ["U12", "U16"],
    "root_cause": "R1/R4",
    "confidence": "high"
  },
  {
    "id": "U03-CRI-05",
    "unit": "U03",
    "role": "critic",
    "type": "leakage",
    "severity": "critical",
    "evidence_register": "code+data",
    "claim": "Filter 2: Remove penny stock rows (adj_price < min_price) on split- and dividend-adjusted prices",
    "claim_location": "scripts/download_tiingo.py:930-931",
    "assessment": "Filtering the back-adjusted price level at $1 makes a stock's presence in each year depend on its future split/dividend history. AAPL (permno 10061) has vendor startDate 1980-12-12 but first row 2000-03-21; NVDA (23854) lists 1999-01-22 but starts 2016-05-13. The same filter deletes the final leg of a collapse, amputating the bankruptcy tail the encoder is meant to learn.",
    "evidence": ["scripts/download_tiingo.py:890", "scripts/download_tiingo.py:930-935", "src/data_pipeline/data_loader.py:230-231", "data/tiingo_meta/supported_tickers.csv", "docs/ISD_vae_latent_risk_factors.md:597"],
    "cross_units": ["U05", "U12"],
    "root_cause": "R5/R3",
    "confidence": "high"
  },
  {
    "id": "U03-CRI-06",
    "unit": "U03",
    "role": "critic",
    "type": "data",
    "severity": "major",
    "evidence_register": "code+results",
    "claim": "ret_nan_frac <= 0.05 ... ret_clean = np.nan_to_num(ret_wins, nan=0.0)",
    "claim_location": "src/data_pipeline/windowing.py:215-227",
    "assessment": "35.31% of the panel is missing and 410/788 stocks exceed 20% missing. Windows with >5% NaN are dropped, surviving NaNs become exact zero returns, and the vol channel is backward-filled: fabricated flats enter the loss with no mask. The exposure matrix covers only 639 of 788 names and the report never explains the 149-name attrition.",
    "evidence": ["src/data_pipeline/windowing.py:213-233", "src/integration/diagnostics.py:1361-1368", "results/diagnostic/diagnostic_report.md:29-31", "results/diagnostic/diagnostic_data.json"],
    "cross_units": ["U04", "U06"],
    "root_cause": "R3",
    "confidence": "high"
  },
  {
    "id": "U03-CRI-07",
    "unit": "U03",
    "role": "critic",
    "type": "logic",
    "severity": "major",
    "evidence_register": "code+doc",
    "claim": "The 21-day rolling volatility captures volatility dynamics and cross-stock co-movements in variance",
    "claim_location": "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:247-252",
    "assessment": "The feature is log-transformed and then z-scored per window and per stock, which removes exactly the absolute scale and cross-sectional co-movement it is claimed to restore. Two windows with identical shape but 10% vs 40% vol produce identical inputs; F=2 carries only the shape of the log-vol path.",
    "evidence": ["src/data_pipeline/windowing.py:248-260", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:224-252"],
    "cross_units": ["U04", "U05"],
    "root_cause": "R4",
    "confidence": "high"
  },
  {
    "id": "U03-CRI-08",
    "unit": "U03",
    "role": "critic",
    "type": "consistency",
    "severity": "major",
    "evidence_register": "code",
    "claim": "vix_data=None on every entry point; crisis labels are computed only if vix_data is not None",
    "claim_location": "scripts/run_diagnostic.py:444",
    "assessment": "VIX is never ingested, so crisis_fractions is None, the trainer sets cf=0 and gamma_eff=1; the advertised gamma=3 crisis weighting is inert in every recorded run while the notebook presents it as an active mechanism.",
    "evidence": ["scripts/run_diagnostic.py:444", "notebooks/dashboard.ipynb:1151", "notebooks/dashboard.ipynb:1217", "src/integration/pipeline.py:1450-1458", "src/training/trainer.py:288-290", "src/vae/loss.py:69"],
    "cross_units": ["U06", "U15"],
    "root_cause": "R4",
    "confidence": "high"
  },
  {
    "id": "U03-CRI-09",
    "unit": "U03",
    "role": "critic",
    "type": "consistency",
    "severity": "major",
    "evidence_register": "code",
    "claim": "S&P 500 price-return benchmark. Uses ^GSPC daily prices from Yahoo Finance",
    "claim_location": "src/benchmarks/sp500_index.py:173-179",
    "assessment": "Stock returns come from dividend-inclusive adjClose while the market benchmark is price-only, and the class is named SP500TotalReturn. The comparison is asymmetric by roughly the dividend yield, material when Sharpe differences are small.",
    "evidence": ["src/benchmarks/sp500_index.py:88-95", "src/benchmarks/sp500_index.py:173-183", "scripts/download_tiingo.py:875-880", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:206"],
    "cross_units": ["U13"],
    "root_cause": "R1",
    "confidence": "high"
  },
  {
    "id": "U03-CRI-10",
    "unit": "U03",
    "role": "critic",
    "type": "data",
    "severity": "major",
    "evidence_register": "code+results",
    "claim": "eps = rng.randn(n_days, n_stocks); log_returns = mu + sigma * eps",
    "claim_location": "src/data_pipeline/data_loader.py:103-104",
    "assessment": "The synthetic generator uses independent shocks per stock per day with a fixed sigma: no common factors, no volatility clustering, no fat tails, no correlated delistings. It still 'validates end-to-end', and the only walk-forward artifacts are 20 synthetic stocks with E*=1, so claims about the pipeline concern plumbing, not the model's purpose.",
    "evidence": ["src/data_pipeline/data_loader.py:99-115", "results/config.json:3", "results/config.json:9", "results/report.json"],
    "cross_units": ["U05", "U12", "U14"],
    "root_cause": "R6",
    "confidence": "high"
  },
  {
    "id": "U03-CRI-11",
    "unit": "U03",
    "role": "critic",
    "type": "interpretation",
    "severity": "major",
    "evidence_register": "code",
    "claim": "Build universe snapshots for estimation rescaling (point-in-time)",
    "claim_location": "src/integration/pipeline.py:2170",
    "assessment": "The snapshot is 'stocks with a non-NaN return today', not U_t: it includes sub-$500M names, ADRs and untradeable prints, excludes stocks on a missing-data day (which the simulator then treats as delistings), and falls back to all ids on dates absent from the index.",
    "evidence": ["src/integration/pipeline.py:2170-2185"],
    "cross_units": ["U09", "U12"],
    "root_cause": "R2",
    "confidence": "high"
  },
  {
    "id": "U03-CRI-12",
    "unit": "U03",
    "role": "critic",
    "type": "implementation",
    "severity": "minor",
    "evidence_register": "code",
    "claim": "Winsorize extreme returns ... Clip to |r| <= 2.0",
    "claim_location": "src/data_pipeline/returns.py:99-110",
    "assessment": "The live rule is a hard clip at about +639%/-86.5% that fires on 6 of 3.96M stock-days in the selected panel, i.e. cosmetic; the percentile winsorizer exists and is never called; the diagnostic doc still lists 'No outlier winsorization' as an open gap.",
    "evidence": ["src/data_pipeline/returns.py:99-110", "src/data_pipeline/returns.py:251-289", "docs/data_preprocessing_diagnostic.md:443", "docs/data_preprocessing_diagnostic.md:452"],
    "cross_units": ["U04"],
    "root_cause": "R3",
    "confidence": "high"
  },
  {
    "id": "U03-CRI-13",
    "unit": "U03",
    "role": "critic",
    "type": "implementation",
    "severity": "minor",
    "evidence_register": "code",
    "claim": "df.groupby('permno').filter(lambda g: len(g) >= min_history_days)",
    "claim_location": "src/data_pipeline/data_loader.py:237",
    "assessment": "Continuous listing of 504 days is implemented as a row count over the whole file; scattered prints qualify. min_valid_fraction is validated in config but read by no module, so no stock is ever dropped for missingness.",
    "evidence": ["src/data_pipeline/data_loader.py:236-238", "scripts/download_tiingo.py:954-958", "src/config.py:137"],
    "cross_units": ["U15"],
    "root_cause": "R3",
    "confidence": "high"
  },
  {
    "id": "U03-CRI-14",
    "unit": "U03",
    "role": "critic",
    "type": "data",
    "severity": "minor",
    "evidence_register": "code",
    "claim": "exchange_map[ticker] = EXCHANGE_CODE_MAP.get(exchange, 1) built from the current supported_tickers.csv",
    "claim_location": "scripts/download_tiingo.py:836-845",
    "assessment": "The exchange code is the current listing venue applied to all history, so exchange migrations make the Shumway -55% vs -30% branch wrong for historical delistings.",
    "evidence": ["scripts/download_tiingo.py:836-845", "src/walk_forward/oos_rebalancing.py:153-165"],
    "cross_units": ["U12"],
    "root_cause": "R1",
    "confidence": "medium"
  },
  {
    "id": "U03-CRI-15",
    "unit": "U03",
    "role": "critic",
    "type": "logic",
    "severity": "minor",
    "evidence_register": "code",
    "claim": "returns_df.at[last_date, permno] = log_ret",
    "claim_location": "src/data_pipeline/returns.py:246",
    "assessment": "The delisting return overwrites the last observed daily return instead of being added to it; a positive merger delisting return replaces rather than compounds. Latent today because the branch is unreachable, but it is a contract error that would surface on any data upgrade.",
    "evidence": ["src/data_pipeline/returns.py:200-248"],
    "cross_units": [],
    "root_cause": "R1",
    "confidence": "high"
  },
  {
    "id": "U03-CRI-16",
    "unit": "U03",
    "role": "critic",
    "type": "honesty",
    "severity": "major",
    "evidence_register": "doc+code",
    "claim": "Point-in-time universe | CONV-10 in universe.py | No look-ahead bias",
    "claim_location": "docs/data_preprocessing_diagnostic.md:432",
    "assessment": "The documentation certifies point-in-time and survivorship-free handling as achieved. Neither holds, and the Tiingo substitution appears only in a pipeline diagram, not in any decision or divergence record.",
    "evidence": ["docs/data_preprocessing_diagnostic.md:23-26", "docs/data_preprocessing_diagnostic.md:183", "docs/data_preprocessing_diagnostic.md:432", "docs/ISD_vae_latent_risk_factors.md:551", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:291", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:736", "docs/implementation_decisions.md:126-150"],
    "cross_units": ["U16", "U17"],
    "root_cause": "R1",
    "confidence": "high"
  },
  {
    "id": "U03-CRI-17",
    "unit": "U03",
    "role": "critic",
    "type": "data",
    "severity": "major",
    "evidence_register": "code+data",
    "claim": "share_code: 10 for every mapped row",
    "claim_location": "scripts/download_tiingo.py:909",
    "assessment": "The common-equity filter never filters: no ADR, REIT, preferred or dual-class exclusion. Dual share classes are separate permnos with near-identical returns (GOOGL 18452 and GOOG 18451 both top-10 by the ex-post rank; BRK-A and BRK-B both present), double-counting one company's risk in B and in the cross-sectional regression.",
    "evidence": ["scripts/download_tiingo.py:909", "src/data_pipeline/universe.py:63-66", "data/tiingo_us_equities.parquet (computed top-15)"],
    "cross_units": ["U05", "U09"],
    "root_cause": "R1",
    "confidence": "high"
  },
  {
    "id": "U03-CRI-18",
    "unit": "U03",
    "role": "critic",
    "type": "consistency",
    "severity": "major",
    "evidence_register": "results",
    "claim": "n_stocks: 2000 ... data_source: synthetic, while the run's panel has 788 stocks",
    "claim_location": "results/diagnostic/config.json:3",
    "assessment": "The recorded config cannot reproduce the run: n_stocks=2000 versus a 788-name panel, and data_source=synthetic for a Tiingo run. The rule that produced 788 is recorded nowhere, so a reader cannot tell which universe any number refers to.",
    "evidence": ["results/diagnostic/config.json:3", "results/diagnostic/config.json:9", "results/diagnostic/diagnostic_report.md:22", "results/diagnostic/diagnostic_report.md:271-272", "results/config.json:3", "results/config.json:9"],
    "cross_units": ["U12", "U15", "U16"],
    "root_cause": "R6",
    "confidence": "high"
  }
]
```

END-OF-REPORT
