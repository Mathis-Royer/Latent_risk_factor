# U03 — advocate

## Scope read

- `docs/DVT_strategie_facteurs_risque_latents_v4_1.md` §4.2 in full (data type/frequency 204–222, preprocessing 224–259, investment universe 261–291), plus §4.3 capacity/statistical guard 422–436, §4.4 crisis weighting 544–582, §4.7 σ̄ survivorship note 736, §5.1 Data/Universe tables 973–992.
- `docs/ISD_vae_latent_risk_factors.md`: conventions 41–54, invariants 58–174, MOD-001 544–907 (data schema, EODHD mapping, loader/returns/universe/windowing/crisis/vol specs, pitfalls, 12 required tests), MOD-006 AU measurement 1632–1657, MOD-007 1681–1812.
- `docs/data_preprocessing_diagnostic.md` (whole file, esp. §2 filtering 80–123, §3 transformations 126–204, §7 best-practice conformity 425–491).
- `src/data_pipeline/data_loader.py`, `returns.py`, `universe.py`, `crisis.py`, `windowing.py`, `features.py` (full).
- `scripts/download_tiingo.py` (full), `scripts/run_walk_forward.py` 130–239, `scripts/run_diagnostic.py` 600–693.
- `src/integration/pipeline.py` 300–321, 1260–1280, 1449–1483, 2010–2029; `src/integration/notebook_helpers.py` 1112–1146; `src/walk_forward/oos_rebalancing.py` 103–120, 218–247; `src/inference/active_units.py` 92–146; `src/config.py` 120–160.
- `results/diagnostic/config.json`, `diagnostic_data.json` (1–150), `diagnostic_report.md` (1–120); `data/tiingo_meta/` file inventory and counts.
- `tests/unit/test_data_quality.py` (test inventory + 760–800, 950–1034).

I did not read the FEM/business docs, notebooks, or other audit outputs.

---

## Five questions

### 1. Purpose and original intent of this unit (cite the spec)

U03 exists to make the backtest *economically valid before any modelling happens*. The DVT states the intent explicitly: "The universe defines *which stocks* enter the pipeline. An incorrectly constructed universe silently invalidates the backtest through **survivorship bias**" (`docs/DVT_...v4_1.md:263`), and enforces a "**Point-in-time principle.** The universe is reconstructed as it existed at each historical date" (`:265`), with eligibility on float-adjusted cap ($500M entry / $400M exit), ADV ≥ $2M, ≥504 days listing, common equity only, NYSE/AMEX/NASDAQ, top-n by float cap (`:267-279`), a training universe that is the *union* of historical universes "including since-delisted stocks" (`:281`), and Shumway imputation (−30% NYSE/AMEX, −55% Nasdaq) for missing delisting returns (`:265`, `:991`). The ISD restates this as CONV-01/CONV-10 (`docs/ISD_...md:45,54`) and MOD-001, whose stated purpose is "survivorship-bias-free handling... This module feeds every downstream component — errors here silently corrupt the entire pipeline" (`:551`). The second intent is representational: log returns (`:206-210`), per-window z-scoring so the encoder sees *shape* not scale (`:226-240`), and a realized-vol channel to restore volatility dynamics that z-scoring erases (`:244-250`).

### 2. What a state-of-the-art version of this unit would look like

- **Source**: CRSP daily stock file with `dlret`, `dlstcd`, `shrcd`, `exchcd`, `prc` (negative `prc` = bid/ask midpoint), plus a *point-in-time* index-membership file. Delisting returns included, not imputed where known (Shumway 1997; Beaver, McNichols & Price 2007 show delisting returns are smaller than Shumway's worst-case but economically indispensable). Fama-French-style breakpoints computed on NYSE names only; Novy-Marx & Velikov (2016) for realistic microcap treatment; Ince & Porter (2006) for individual-stock data-error rates.
- **Order of operations**: returns first, screens second. Price/liquidity screens are *portfolio-formation* decisions applied at date *t*, never row-level truncations of the return series — truncating at a price threshold deletes the crash that crossed it.
- **Universe mechanics**: eligibility computed once per reconstitution on a vectorised panel, membership held with an entry/exit buffer (already in the spec, `:269`), and the *training* universe deliberately wider than the investment universe to include dead names.
- **Adjustment discipline**: total-return series for returns; *unadjusted* price for screens and for any cap/liquidity proxy, so that screens do not depend on future corporate actions.
- **Missing data**: explicit imputation policy with a documented bias direction (Little & Rubin); forward-fill only for gaps that are short relative to the estimator's memory, and never silently zero-filled before standardisation.
- **Effective-sample accounting**: report N_eff (Kish 1965) alongside nominal N whenever a shrinkage estimator (Ledoit-Wolf 2004, DGJ spiked) is fitted, because the binding data constraint is effective observations, not calendar span.
- **Audit trail**: a machine-checkable provenance record per run (vendor, snapshot date, ticker count attempted/succeeded, delisted share) — the analogue of the backfill-bias controls used in hedge-fund databases (Fung & Hsieh 2000).

Against that standard, the unit's *construction* is close; its *provenance* is far.

### 3. What prevents the current unit from reaching it — and which obstacles are only apparent

**Real (evidenced) obstacles**

1. *The vendor cannot express the contract.* Tiingo's merge hard-codes `"delisting_return": np.nan` for every row (`scripts/download_tiingo.py:911`), hard-codes `"share_code": 10` (`:909`), defaults unknown exchanges to NYSE (`:845`, map at `:87-94`), and derives "market cap" as `close × 63-day average volume × 20` (`:899-901`). Four of the eight ISD core columns (`docs/ISD_...md:592-601`) are therefore synthetic or constant. No local code change can recover delisting returns, float, or share type.
2. *Rate limits force a sample.* The downloader caps at 45 tickers per key per hour (`scripts/download_tiingo.py:96-98`); `data/tiingo_meta/supported_tickers.csv` lists 22,790 US stock rows, and `download_progress.json` records 451 ticker statuses. A 2% draw of the vendor universe is what the data actually is.
3. *Effective depth, not span.* See finding U03-ADV-12: with the 252-day EWMA half-life the Kish effective sample is ~727 trading days, and `AU_max_stat` = 26 exactly.

**Only apparent obstacles**

- *"We lack 30 years."* The panel spans 1995-01-03 → 2026-02-13 = 31.2 years / 7,874 days (`results/diagnostic/config.json:130-131`; `results/diagnostic/diagnostic_report.md:26-29`).
- *"We lack breadth."* 788 stocks yield 639 exposure rows vs AU = 26 (`diagnostic_data.json:96-103`) — identification is comfortable, not marginal.
- *"Missing data disqualifies the sample."* 35.31% panel missingness (`diagnostic_report.md:30`) is dominated by pre-listing NaN, which the window filter is explicitly designed to drop (`windowing.py:212-215`). The design already handles it.
- *"CRSP is unaffordable."* The expensive part of CRSP here is not prices, it is the delisting-return table and share-type codes — the DVT itself names EODHD/Norgate as cheaper substitutes for exactly that (`docs/DVT_...md:291`). The obstacle is sourcing one small table, not licensing CRSP.

### 4. Which constraints are real and which are self-imposed

**Real**: vendor absence of delisting returns / shares outstanding / share codes / point-in-time index membership; API rate limits; 16 GB RAM (which genuinely shapes the two-pass window builder, `windowing.py:148-178`); the arithmetic of `AU(AU+1)/2` parameters against N_eff.

**Self-imposed (design choices, not constraints)**

- Applying the $1 price screen at *dataset build* (`scripts/download_tiingo.py:930-933`) and again in the loader (`data_loader.py:230-232`), i.e. before returns are computed — a one-line reordering that costs nothing and currently truncates the left tail.
- Choosing universe members by *full-sample* median "market cap" (`data_loader.py:344-348`) instead of by the point-in-time function that already exists.
- Implementing `construct_universe`/`build_training_universe`/`handle_delisting` and never calling them (`universe.py:13,125,182`; only definition sites appear in a repo-wide grep).
- Defining the trade-date universe as "non-NaN return today" (`notebook_helpers.py:1128-1134`, `oos_rebalancing.py:233-237`) and calling it point-in-time.
- Not loading VIX: both runners pass `vix_data=None` (`run_walk_forward.py:203`, `run_diagnostic.py:444`), so γ = 3.0 is inert.
- `×20` in the cap proxy, which silently converts the $500M cap rule into a ~$25M ADV rule (12.5× the stated $2M ADV rule).
- `n_stocks = 1000` (`config.py:132`) when 788 exist; `min_valid_fraction = 0.80` (`config.py:137`) is never read outside config validation.
- 5% NaN *per window* instead of the ISD's ">5 consecutive days excluded" (`docs/ISD_...md:734` vs `windowing.py:215`).
- `training_stride = 21` (`config.py:140`) — a legitimate compute choice, not a data constraint, but it changes the effective training distribution and should be stated as such.

### 5. What should be removed, merged or moved

**Remove outright (no strength lost, because the strength lives elsewhere):**

1. `_filter_universe` (`data_loader.py:323-350`). It is a look-ahead selector by full-sample median cap, and it is redundant with any point-in-time eligibility function. Removing it loses nothing that `construct_universe` does not do correctly. In the one real run it never bound (n_stocks = 2000 > 788), so its removal is behaviour-neutral today and prevents a future silent bias.
2. `winsorize_returns` (`returns.py:251-288`). Dead code; the ±2 log-return clip at `returns.py:99-110` already bounds the influence of vendor-error observations, and cross-sectional P1/P99 winsorisation would alter the very co-movement structure the loss is trying to learn.
3. The `×20` in the cap proxy (`download_tiingo.py:901`) and, with it, the pretence that the "$2M ADV" rule is separately binding. Keep one dollar-volume threshold.
4. The dead NaN branch of `_extract_delisting_info` / `_apply_delisting_returns` (`returns.py:187-195`, `:230-236`). It is unreachable by construction and sustains a documented guarantee that does not exist.
5. `min_valid_fraction` from config (`config.py:137`) or wire it — currently a knob with no consumer.

**Merge:**

6. Delisting handling currently lives in three places: `returns.py` (imputation), `universe.py` (`handle_delisting`, unused), and `oos_rebalancing.py` (weights dropped and renormalised, `:111-112`). Pick one semantic (liquidate at last price, hold cash, or apply a terminal return) and implement it once, in the return/valuation path.
7. The penny/extreme-price screens exist in two stages (`download_tiingo.py:930-952`, `data_loader.py:230-232`). Merge into one eligibility screen applied at date *t*.

**Move:**

8. Move the price screen *after* return computation (compute on the full raw series, then screen names at formation). This is the highest-value structural change in the unit and costs no data.
9. Move `construct_universe` from library to the single production path (or delete it); do not keep two universe definitions where the correct one is unused.
10. Move `AU_max_stat` computation into the data audit, because it is a statement about data depth (N_eff), not about the estimator — it is the real binding constraint on the whole strategy and should be reported next to "universe size", not buried in inference.
11. Move VIX loading into the runner (one call, once) or delete γ from the loss and from the documents. A core, advertised mechanism must not be inert.

---

## Load-bearing assumptions

| # | Assumption | Where stated | Evidenced? | Comment |
|---|---|---|---|---|
| A1 | Tiingo `adjClose` is a correctly split/dividend-adjusted total-return price, equivalent to CRSP `prc` with adjustment factors | `docs/DVT_...md:210`; `scripts/download_tiingo.py:875-878` | partially | The code prefers `adjClose` (correct choice), but ships a `MAX_ABS_RETURN = 2.0` clip (`returns.py:102`) and a $1M extreme-price filter (`download_tiingo.py:941-952`) precisely because this assumption fails intermittently. |
| A2 | Stocks that vanish from the panel carry no material return at disappearance | implicit | **no** | `delisting_return` is NaN for every row (`download_tiingo.py:911`); nothing supplies a terminal return. |
| A3 | The 788 downloaded names are a representative draw from the specified US large-cap universe | implicit in `docs/ISD_...md:572-581` | **no** | 451 progress entries against 22,790 supported tickers, with current-S&P-500-first ordering (`download_tiingo.py:437-445`). |
| A4 | Forward-filling ≤5 consecutive missing prices as 0 returns, and 0-filling remaining NaNs before z-scoring, leaves window statistics consistent | `returns.py:43-44`, `windowing.py:227-229` | partially | Filled zeros enter `ret_mu`/`ret_sigma` (`windowing.py:252-255`) and the zero-fraction filter counts them (`:232`), which partially offsets the distortion. |
| A5 | 5% NaN per 504-day window is an adequate continuity screen | `windowing.py:215` | partially | This is ~25 days, not the ISD's "5 consecutive days" (`docs/ISD_...md:734`). No test pins the two together. |
| A6 | A stock's *total* row count ≥ 504 in the sample is a valid proxy for 504 continuous days of listing at trade date *t* | `data_loader.py:236-237` | **no** | Full-sample count; applied once, not per date. |
| A7 | Cross-sectional OLS at date *t* is identified because n_active ≫ AU | `docs/DVT_...md:750` | yes | 639 rows vs AU = 26 (`diagnostic_data.json:96-103`). |
| A8 | 31 years of daily data give enough independent observations to estimate an AU×AU factor covariance | `docs/DVT_...md:426` | partially | Nominal N = 7,874; effective N under a 252-day EWMA is ~727 (→ `AU_max_stat` = 26, matching the run). |
| A9 | VIX is available and crisis weighting is active | `docs/DVT_...md:548-558` | **no** | `vix_data=None` in both runners (`run_walk_forward.py:203`, `run_diagnostic.py:444`). |

---

## Findings

Severity encodes the strength grade required for this role: **solid** = holds generally, **conditional** = holds only under the stated condition, **fragile** = correct in principle but easily (or already) broken.

| ID | Type | Severity | Evidence register | Claim (quoted, path:line) | Assessment | Evidence (path:line) | Cross-unit |
|----|------|----------|-------------------|---------------------------|------------|----------------------|------------|
| U03-ADV-01 | math / implementation | solid | code+doc | "r_{i,t} = ln(P^adj_{i,t} / P^adj_{i,t-1})" — implemented as `np.log(price_vals[1:] / price_vals[:-1])` on a positivity-validated panel | Log returns are computed exactly as specified, first row NaN, no arithmetic fallback, guarded by `assert_log_input_positive` before the log. This is the one convention on which every downstream estimate depends and it is not violated. | `src/data_pipeline/returns.py:81-92`; `docs/DVT_...md:206-210`; `docs/ISD_...md:45` | U02, U04 |
| U03-ADV-02 | statistics / implementation | solid | code+doc | "z-score standardization... computed **per-window**"; `ret_mu = ret_clean.mean(axis=1, keepdims=True)` … `ret_zscore = (ret_clean - ret_mu) / ret_sigma` | Per-window, per-feature standardisation is implemented as designed and *asserted* at runtime (`assert_z_score_normalized`) with a guard for degenerate variance. Because the statistic is within-window, it is look-ahead-free by construction — the DVT's central argument against expanding/full-history normalisation (`:234-240`) is honoured, not just documented. | `src/data_pipeline/windowing.py:251-273`; `docs/DVT_...md:226-240`; `docs/ISD_...md:46` | U04, U06 |
| U03-ADV-03 | leakage / implementation | solid | code | "`ret_all = sliding_window_view(ret_series, T)[::stride]`" with metadata `end_indices = start_indices + T - 1` | Windows are strictly backward-looking slices ending at *t*; no statistic crosses the window boundary. The 21-day realized-vol channel is computed on the whole series (`rolling(window=21, min_periods=21)`), so the first day of each encoder window uses 20 *pre-window* returns — the warm-up requirement at `docs/DVT_...md:252` is genuinely satisfied rather than papered over. | `src/data_pipeline/windowing.py:193-203, 209-210`; `src/data_pipeline/features.py:76-78`; `docs/DVT_...md:244-252` | U04 |
| U03-ADV-04 | data / implementation | conditional | code | "`nan_ok = (ret_nan_frac <= 0.05) & (vol_nan_frac <= 0.05)`"; then `ret_clean = np.nan_to_num(ret_wins, nan=0.0)`; "`zero_ok = zero_frac <= max_zero_frac`" | The missing-data policy is explicit, bounded, and output-asserted (`assert_no_nan_tensor`). It holds conditionally: (a) the tolerance is 5% *of the window* (~25 days) rather than the ISD's 5 *consecutive* days (`docs/ISD_...md:734`), and (b) zero-imputed gaps enter the mean/std used for z-scoring, so a 5%-gappy window is slightly mis-scaled. Both are small at 5% but they are real, and the suspension filter partly compensates by counting imputed zeros. | `src/data_pipeline/windowing.py:212-233, 232, 302`; `docs/ISD_...md:733-735` | U04 |
| U03-ADV-05 | doc-vs-code | fragile | doc+code | "Delisting returns: If NaN: impute per Shumway (1997): -30% NYSE/AMEX, -55% NASDAQ" | The constants are right (`SHUMWAY_NYSE_AMEX = np.log(0.70)`, `SHUMWAY_NASDAQ = np.log(0.45)`) and the bounds assert is sane, but the imputation is unreachable: `_extract_delisting_info` only creates an entry `if pd.notna(delist_ret)`, else from rows that are themselves notna, so the `np.isnan(delist_ret)` branch can never fire. A stock with no delisting return is therefore never penalised — the exact case Shumway exists for. | `src/data_pipeline/returns.py:26-28, 169-197, 230-238`; `docs/DVT_...md:265`; `docs/ISD_...md:737-738` | U03→U12 |
| U03-ADV-06 | data | fragile | code | "`"delisting_return": np.nan,`" | The merged Tiingo dataset carries no delisting returns at all, so finding U03-ADV-05 is not a corner case: it is the whole sample. The unit is at least honest here — no fake −30% is invented — but the survivorship-bias mitigation the design claims does not exist in the executed data. | `scripts/download_tiingo.py:903-912` | U12, U16 |
| U03-ADV-07 | consistency / data | fragile | code+results | "`stock_ids = [int(c) for c in returns.columns if str(c).isdigit()]`" (comment above: "Get universe at train_end") | The effective universe is *every stock in the loaded panel*, at every date, with no cap/ADV/listing-history/security-type screen. The trade-date snapshot is only "has a non-NaN return today" (`notebook_helpers.py:1128-1134`, `oos_rebalancing.py:233-237`). This is look-ahead-free (it is listing-only), so it does not invalidate the backtest, but the specified eligibility universe (`docs/DVT_...md:267-279`) is not the universe that was traded: 788 names, no n_max, no cap floor, no 504-day rule at the trade date. | `src/integration/pipeline.py:1265-1279`; `src/integration/notebook_helpers.py:1128-1134`; `src/walk_forward/oos_rebalancing.py:233-237`; `results/diagnostic/config.json:130` | U09, U12, U14 |
| U03-ADV-08 | logic / design | fragile | code | "`data_up_to = stock_data[stock_data["date"] <= date]`" … "`cap_threshold = cap_exit if permno in previous_set else cap_entry`" … "`universe = [permno for permno, _ in eligible[:n_max]]`" | `construct_universe` is a faithful, point-in-time implementation of the spec: date-truncated panel, share/exchange filters, 504-day listing, cap hysteresis, trailing-63d ADV, liveness ≤10 days, top-n by cap, duplicate assert; `build_training_universe` implements the union-including-delisted rule (`docs/DVT_...md:281`). It is *never called* — and `handle_delisting` (`universe.py:182-240`) likewise. This is the unit's best asset and it is dead code, while a weaker ad-hoc definition runs in production. | `src/data_pipeline/universe.py:13-122, 125-179, 182-240`; `src/integration/pipeline.py:1266` | U12, U17 |
| U03-ADV-09 | leakage | fragile | code | "`median_mcap = df.groupby("permno")["market_cap"].median()` … `top_permnos = list(median_mcap.nlargest(n_stocks).index)`" | The only *selection* rule that ever runs on real data ranks names by their market cap averaged over the **entire loaded history** — a full-sample statistic, i.e. survivorship bias reintroduced by the loader. In the diagnostic it did not bind (n_stocks 2000 > 788), so it is inert today and a trap tomorrow; any run with `n_stocks < 788` becomes biased with no warning. | `src/data_pipeline/data_loader.py:323-350`; `scripts/run_diagnostic.py:650-652` | U12 |
| U03-ADV-10 | data / interpretation | fragile | code+doc | "`merged = pd.DataFrame(merged[merged["adj_price"] >= min_price])`" applied *before* `compute_log_returns` | The $1 screen runs at dataset build, so the return that crosses $1 and every subsequent sub-$1 return are deleted, not recorded as large negatives. For a strategy whose stated purpose includes capturing "tail-risk patterns" (`docs/DVT_...md:263`), this truncates exactly the observations it wants. Standard practice is returns first, price screen at formation. Additional subtlety: the screen is applied to a *retroactively restated* price series, so which rows survive depends on future dividends/splits. | `scripts/download_tiingo.py:930-933`; `src/data_pipeline/data_loader.py:230-232`; `src/data_pipeline/returns.py:89` | U12, U06 |
| U03-ADV-11 | implementation | solid | code | "`assert not df["date"].isna().any()`" / "`assert not dup_keys.any()`" / "`assert_positive_prices(df["adj_price"], "adj_price")`" | The loader declares and enforces the 8-column contract, positive prices, non-negative volume, and duplicate `(permno,date)` rejection before anything downstream consumes the panel — cheap, correct, and it fails loudly rather than silently. One gap: date monotonicity is verified on `df["permno"].unique()[:5]` only, so it is a spot check, not a guarantee. | `src/data_pipeline/data_loader.py:269-318, 288-292`; `docs/ISD_...md:592-601` | — |
| U03-ADV-12 | statistics / math | solid | code+results | "AU_max_stat = floor(sqrt(2 × N_eff / r_min))" with "N_eff = 1 / sum(w_i^2)" | The statistical guard is correct, correctly asymptotic, and correctly *binding*: with a 252-day EWMA half-life the Kish ESS of ~7,623 daily rows is ≈727, so `floor(sqrt(727)) = 26` — exactly the AU = 26 and `B_A (639, 26)` reported. The guard truncates by descending marginal KL, as specified. Two consequences an advocate must state plainly: (i) the "VAE discovers 26 factors" is partly a *data-depth* verdict, not a modelling verdict (raw AU was 52 — every one of the K=52 dimensions had KL ≫ 0.01); (ii) the DVT's printed closed form `(-1+sqrt(1+4·N/r_min))/2` at `:424` is dimensionally wrong (it gives 60, not the 85 quoted at `:426`); the ISD form at `:222` and the code are the correct ones. | `src/inference/active_units.py:92-117, 120-137`; `src/integration/pipeline.py:2017-2021`; `results/diagnostic/diagnostic_data.json:7-9, 38-103`; `results/diagnostic/config.json:66-68, 130-131`; `docs/DVT_...md:424-426`; `docs/ISD_...md:222` | U08, U09, U15, U17 |
| U03-ADV-13 | data / interpretation | conditional | code+results | date range "1995-01-03 to 2026-02-13", 7,874 days, "Years of data: 31.2" | The 30-year depth requirement is met in calendar span, and 7,874 days is ample *raw* material for a 504-day encoder window. It is not met in continuity: "Missing data: 35.31%", "Stocks > 20% missing: 410", and only 639 of 788 names produce exposure rows. The panel is a long-span, thin-coverage sample, not the dense large-cap panel assumed at `docs/DVT_...md:263`. | `results/diagnostic/config.json:130-131`; `results/diagnostic/diagnostic_report.md:26-31`; `results/diagnostic/diagnostic_data.json:96-103` | U12, U14 |
| U03-ADV-14 | leakage / implementation | solid | code | "`vix_training = vix_data.loc[vix_data.index <= training_end_date]`" … "`threshold = float(np.percentile(..., percentile))`" … "`fractions[i] = (window_vix > threshold).mean()`" | Crisis labelling is genuinely point-in-time: an expanding-window P80 truncated at the fold's training end, an explicit `history_start` option, a hard error if no VIX exists in range, and a *continuous* per-window fraction rather than a binary flag. This is exactly what `docs/DVT_...md:554-558` specifies and it is better than the usual binary-crisis hack. The condition: it is never executed — both runners pass `vix_data=None`, so the `if` at `pipeline.py:1451` is false and γ = 3.0 is inert. | `src/data_pipeline/crisis.py:133-153, 183-194, 196-203`; `src/integration/pipeline.py:1449-1455`; `scripts/run_walk_forward.py:203`; `scripts/run_diagnostic.py:444` | U06, U07 |
| U03-ADV-15 | data / justification | conditional | code+doc | "`market_cap = close_arr * np.asarray(rolling_avg_vol) * 20.0`" — "Tiingo free tier lacks shares outstanding, we use: close × average_volume × 20 as a rough market cap proxy" | As a *ranking* device the proxy is defensible: dollar volume is monotone in visible size and the comment is honest about what it is. As a *threshold* device it is not: $500M / 20 = $25M of daily dollar volume, i.e. 12.5× the separate $2M ADV rule, so the two eligibility criteria collapse into one much stricter liquidity screen and the "$500M float-adjusted cap" claim (`docs/DVT_...md:269`) is not what the code tests. The consequence — a liquid large-cap sample — is accidentally consistent with the DVT's stated intent (exclude micro-caps where "spreads 50–200 bps"), which is why this is conditional rather than fatal. | `scripts/download_tiingo.py:890-901`; `src/data_pipeline/universe.py:87-100`; `docs/DVT_...md:269-271` | U09, U15 |
| U03-ADV-16 | data / consistency | fragile | code+doc | "`"share_code": 10,`" and "`exchange_code": exchange_map.get(ticker, 1)`" | The security-type and exchange filters are vacuous by construction: every ticker is common equity code 10, and any unmapped exchange becomes NYSE. `assetType == "Stock"` (`scripts/download_tiingo.py:430`) does exclude ETFs, which is the substantive part of the filter; ADRs, REITs, preferreds and warrants are not excluded, and the Shumway NYSE/AMEX-vs-NASDAQ split would key off a vendor string rather than CRSP `exchcd`. | `scripts/download_tiingo.py:87-94, 428-431, 841-845, 908-909`; `src/data_pipeline/universe.py:66-74`; `docs/ISD_...md:598-606` | U17 |
| U03-ADV-17 | data / hypothesis | fragile | code | "SP500 priority: %d SP500 tickers first" — `df = df.sort_values("is_sp500", ascending=False, kind="mergesort")` combined with "All %d keys exhausted... Re-run to resume" | The downloader orders *current* S&P 500 constituents first and stops when the key budget is exhausted, so every partial download is a constituent-biased sample. Observed magnitudes: 22,790 supported US stock rows vs 451 recorded ticker statuses. Testable prediction: among the 788 permnos in the parquet, the share of `is_sp500 == True` should far exceed the 2.2% base rate of the supported list, and the count of tickers whose last observation is before 2025 should be small. If confirmed, the panel is close to a *current-index* survivor set and every cross-sectional result is upward-biased. | `scripts/download_tiingo.py:437-445, 671-679, 96-98`; `data/tiingo_meta/supported_tickers.csv` (22,790 rows); `data/tiingo_meta/download_progress.json` (451 statuses) | U12, U16 |
| U03-ADV-18 | implementation | solid | code | "PASS 1: Count total valid windows per stock (memory-efficient)" … "PASS 2: Pre-allocate and fill arrays" | Two-pass counting followed by single pre-allocation, float32 throughout, `sliding_window_view` (no copies), in-place gap filling, and `groupby().filter` instead of `transform` are deliberate and correct responses to a 16 GB machine. This is the kind of engineering that makes an otherwise infeasible 5.5M-cell window tensor tractable, and it does not trade correctness for memory: the filters are reapplied identically in pass 2 and the output is asserted. | `src/data_pipeline/windowing.py:148-178, 275-305`; `src/data_pipeline/data_loader.py:234-239`; `src/data_pipeline/returns.py:85-86, 137-139` | — |

---

## Core strengths to preserve

1. **Exact log-return + positivity guard.** `returns.py:81-92`. Any redesign that keeps a return-based factor model must keep `ln(P_t/P_{t-1})` on a validated strictly-positive price series, with the first observation NaN rather than zero. This is the only place where a silent sign/level error would propagate into every factor and every weight.
2. **Within-window standardisation as a look-ahead-free operator.** `windowing.py:251-260`. The DVT's argument (`:234-240`) is correct and the code implements it. A redesign that switches to expanding or cross-sectional normalisation must re-derive the causality argument from scratch; the current choice is defensible on its own terms and must not be lost by accident.
3. **Causal window construction with a real 21-day warm-up.** `windowing.py:193-203`, `features.py:76-78`. Cheap, correct, and easy to break when refactoring the feature pipeline.
4. **The point-in-time universe function as *design*.** `universe.py:13-122, 125-179`. Date truncation, entry/exit hysteresis, trailing-63d ADV, liveness, top-n ranking, duplicate assert, and the union-including-delisted training universe are all correct. Whatever the redesign does about data vendors, this logic should be preserved and *used*, not rewritten.
5. **The Kish-ESS statistical guard, applied and reported.** `active_units.py:92-137`, `pipeline.py:2017-2021`. It is the only place in the pipeline where the *data's* information content bounds the *model's* capacity, and it is correct. It should survive and should be surfaced as a data-quality headline, not hidden inside inference.
6. **Point-in-time crisis labelling.** `crisis.py:133-153, 183-194`. Expanding percentile with an explicit training-end mask and a continuous weight. If γ stays in the design, this is the implementation to keep; if γ goes, this module is the thing to delete with it (not to leave inert).
7. **Declarative contract + loud validation at the loader boundary.** `data_loader.py:269-318`, `windowing.py:141-144, 262-273, 302`. Shape, duplication, positivity, finiteness and z-score normalisation are checked rather than assumed. This is why the unit's *bugs are discoverable* even though its data are biased.
8. **Honest, quantified preprocessing diagnostic.** `docs/data_preprocessing_diagnostic.md:425-491` names the missing winsorisation, the missing price-discontinuity check, and the un-penalised Phase-A selection. The unit knows its own gaps; a redesign should preserve that audit habit.

---

## Verdict

As an advocate I can defend the *machinery* of U03 almost line by line: log returns are exact and guarded, per-window standardisation is look-ahead-free and asserted, windows are causal with a genuine 21-day warm-up, the missing-data policy is bounded and NaN-free at output, the point-in-time universe and the crisis labeller are correctly designed, and the Kish-ESS capacity guard is both correct and — at AU = 26 with a raw AU of 52 — the actual binding constraint on the whole strategy. What I cannot defend is the *provenance*. The executed data is a 788-name partial Tiingo draw with no delisting returns (`download_tiingo.py:911`), a turnover proxy masquerading as float-adjusted market cap (`:901`), hard-coded share codes (`:909`), a $1 row filter that happens at dataset build rather than at portfolio formation (`:930-933`), a full-sample median-cap selector (`data_loader.py:344-348`), and no VIX at runtime. The specified survivorship-bias correction exists in the documents and in unreachable branches of the code, not in the sample. The unit's own correct universe function is dead code. The strengths that a redesign must carry forward are the return/standardisation/causality chain, the *design* of `construct_universe`, and the effective-sample accounting — and the single structural move that buys the most is reordering screens after returns and wiring the point-in-time eligibility function into the only data path, which is free. On balance the construction is sound and the source defeats its purpose; that is a mixed verdict, not a pass.

Score: 5/10

---

## Machine companion

```json
[
  {
    "id": "U03-ADV-01",
    "unit": "U03",
    "role": "advocate",
    "type": "math",
    "severity": "solid",
    "evidence_register": "code+doc",
    "claim": "Log returns are computed exactly as specified on a positivity-validated adjusted price panel: r_t = ln(P_t/P_{t-1}), first row NaN, no arithmetic fallback.",
    "claim_location": "src/data_pipeline/returns.py:84-92",
    "assessment": "Correct implementation of CONV-01 with an explicit log-input guard. The one invariant on which every downstream estimate depends is not violated.",
    "evidence": ["src/data_pipeline/returns.py:81-92", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:206-210", "docs/ISD_vae_latent_risk_factors.md:45"],
    "cross_units": ["U02", "U04"],
    "root_cause": null,
    "confidence": "high"
  },
  {
    "id": "U03-ADV-02",
    "unit": "U03",
    "role": "advocate",
    "type": "statistics",
    "severity": "solid",
    "evidence_register": "code+doc",
    "claim": "Per-window, per-feature z-scoring is implemented as designed and runtime-asserted, making the standardisation look-ahead-free by construction.",
    "claim_location": "src/data_pipeline/windowing.py:251-273",
    "assessment": "The DVT's argument against expanding/full-history normalisation is honoured in code, not just documented. Degenerate-variance guard and normality assertion are present.",
    "evidence": ["src/data_pipeline/windowing.py:251-273", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:226-240", "docs/ISD_vae_latent_risk_factors.md:46"],
    "cross_units": ["U04", "U06"],
    "root_cause": null,
    "confidence": "high"
  },
  {
    "id": "U03-ADV-03",
    "unit": "U03",
    "role": "advocate",
    "type": "leakage",
    "severity": "solid",
    "evidence_register": "code",
    "claim": "Windows are strictly backward-looking slices ending at date t, and the 21-day realized-volatility channel is computed on the full series so its first window day uses 20 pre-window returns.",
    "claim_location": "src/data_pipeline/windowing.py:193-203",
    "assessment": "Causality is guaranteed structurally (sliding_window_view + end index t). The warm-up requirement of the DVT is genuinely satisfied, not approximated.",
    "evidence": ["src/data_pipeline/windowing.py:193-203", "src/data_pipeline/windowing.py:209-210", "src/data_pipeline/features.py:76-78", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:244-252"],
    "cross_units": ["U04"],
    "root_cause": null,
    "confidence": "high"
  },
  {
    "id": "U03-ADV-04",
    "unit": "U03",
    "role": "advocate",
    "type": "data",
    "severity": "conditional",
    "evidence_register": "code",
    "claim": "The missing-data policy is explicit and bounded (<=5% NaN per window, then returns->0 and vol->ffill/bfill->0, suspension filter at 20% zero returns), but the 5% tolerance is per-window rather than the ISD's 5-consecutive-day rule.",
    "claim_location": "src/data_pipeline/windowing.py:212-233",
    "assessment": "Holds as a bounded, asserted policy. Condition: zero-imputed gaps enter the mean/std used for z-scoring, so gappy windows are slightly mis-scaled; the zero-fraction filter partially compensates by counting imputed zeros.",
    "evidence": ["src/data_pipeline/windowing.py:212-233", "src/data_pipeline/windowing.py:232", "src/data_pipeline/windowing.py:302", "docs/ISD_vae_latent_risk_factors.md:733-735"],
    "cross_units": ["U04"],
    "root_cause": null,
    "confidence": "high"
  },
  {
    "id": "U03-ADV-05",
    "unit": "U03",
    "role": "advocate",
    "type": "doc-vs-code",
    "severity": "fragile",
    "evidence_register": "doc+code",
    "claim": "The Shumway delisting imputation is documented as active and its constants are correct, but the imputation branch is unreachable because delisting info is only registered for stocks that already have a non-NaN delisting return.",
    "claim_location": "src/data_pipeline/returns.py:169-197",
    "assessment": "Correct in principle, broken in practice: a stock with no delisting return is the only case where imputation matters, and it is exactly the case that is skipped. The documented guarantee is not delivered.",
    "evidence": ["src/data_pipeline/returns.py:26-28", "src/data_pipeline/returns.py:169-197", "src/data_pipeline/returns.py:230-238", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:265", "docs/ISD_vae_latent_risk_factors.md:737-738"],
    "cross_units": ["U12"],
    "root_cause": "Two-stage delisting logic with an unreachable NaN branch; no test exercises imputation on a stock lacking the column value.",
    "confidence": "high"
  },
  {
    "id": "U03-ADV-06",
    "unit": "U03",
    "role": "advocate",
    "type": "data",
    "severity": "fragile",
    "evidence_register": "code",
    "claim": "The merged Tiingo dataset sets delisting_return to NaN for every row, so the entire sample carries no delisting returns and no survivorship correction can be applied.",
    "claim_location": "scripts/download_tiingo.py:903-912",
    "assessment": "Honest (no fabricated penalty is invented) but the unit's stated purpose (survivorship-bias-free universe) is not achieved. This is the single largest threat to the unit.",
    "evidence": ["scripts/download_tiingo.py:903-912"],
    "cross_units": ["U12", "U16"],
    "root_cause": "Vendor lacks delisting returns; the merge writes a constant instead of sourcing a terminal-return table.",
    "confidence": "high"
  },
  {
    "id": "U03-ADV-07",
    "unit": "U03",
    "role": "advocate",
    "type": "consistency",
    "severity": "fragile",
    "evidence_register": "code+results",
    "claim": "The effective investment universe is every stock in the loaded panel at every date, with the trade-date snapshot defined only as 'has a non-NaN return today'; the specified cap/ADV/listing/security filters are never applied at a trade date.",
    "claim_location": "src/integration/pipeline.py:1266",
    "assessment": "Look-ahead-free (listing-only), so the backtest is not invalidated by this alone, but the traded universe (788 names, no n_max, no 504-day rule) is not the universe the spec describes. The inline comment 'Get universe at train_end' misdescribes the code.",
    "evidence": ["src/integration/pipeline.py:1265-1279", "src/integration/notebook_helpers.py:1128-1134", "src/walk_forward/oos_rebalancing.py:233-237", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:267-279", "results/diagnostic/config.json:130"],
    "cross_units": ["U09", "U12", "U14"],
    "root_cause": "The eligibility function exists in universe.py but is not wired into the only executable data path.",
    "confidence": "high"
  },
  {
    "id": "U03-ADV-08",
    "unit": "U03",
    "role": "advocate",
    "type": "design",
    "severity": "fragile",
    "evidence_register": "code",
    "claim": "construct_universe and build_training_universe implement the point-in-time universe faithfully (date truncation, hysteresis, ADV, liveness, top-n, union-including-delisted) and are never called; handle_delisting is likewise dead.",
    "claim_location": "src/data_pipeline/universe.py:13-122",
    "assessment": "The best asset of this unit is unused code. A weaker ad-hoc definition runs in production while the correct one sits in the library. Preserve the logic, delete the competing definition.",
    "evidence": ["src/data_pipeline/universe.py:13-122", "src/data_pipeline/universe.py:125-179", "src/data_pipeline/universe.py:182-240", "src/integration/pipeline.py:1266"],
    "cross_units": ["U12", "U17"],
    "root_cause": "Two competing universe definitions; only the cheaper one was wired.",
    "confidence": "high"
  },
  {
    "id": "U03-ADV-09",
    "unit": "U03",
    "role": "advocate",
    "type": "leakage",
    "severity": "fragile",
    "evidence_register": "code",
    "claim": "The loader's only selection rule ranks stocks by their median market cap over the entire loaded history, a full-sample statistic applied before the walk-forward split.",
    "claim_location": "src/data_pipeline/data_loader.py:344-348",
    "assessment": "Inert in the diagnostic (n_stocks 2000 > 788 available) but actively biasing for any run with a binding cap. Positional look-ahead at the universe level, not at the return level.",
    "evidence": ["src/data_pipeline/data_loader.py:323-350", "scripts/run_diagnostic.py:650-652", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:265"],
    "cross_units": ["U12"],
    "root_cause": "Convenience filter in the loader duplicating the point-in-time universe module.",
    "confidence": "high"
  },
  {
    "id": "U03-ADV-10",
    "unit": "U03",
    "role": "advocate",
    "type": "data",
    "severity": "fragile",
    "evidence_register": "code+doc",
    "claim": "The $1 price screen is applied at dataset build, before log returns are computed, so the return crossing $1 and all subsequent sub-$1 returns are deleted rather than recorded as large negative observations.",
    "claim_location": "scripts/download_tiingo.py:930-933",
    "assessment": "One-sided truncation of the left tail, in the exact region a tail-risk strategy claims to capture. The screen is also applied to a retroactively restated adjusted price, so row survival depends on future corporate actions.",
    "evidence": ["scripts/download_tiingo.py:930-933", "src/data_pipeline/data_loader.py:230-232", "src/data_pipeline/returns.py:89", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:263"],
    "cross_units": ["U12", "U06"],
    "root_cause": "Screens applied at data construction rather than at portfolio formation.",
    "confidence": "high"
  },
  {
    "id": "U03-ADV-11",
    "unit": "U03",
    "role": "advocate",
    "type": "implementation",
    "severity": "solid",
    "evidence_register": "code",
    "claim": "The loader enforces the 8-column contract, positive prices, non-negative volume and duplicate (permno,date) rejection before any downstream consumption; date monotonicity is checked on a 5-permno sample only.",
    "claim_location": "src/data_pipeline/data_loader.py:269-318",
    "assessment": "Loud failure instead of silent corruption at the boundary. The spot-check on monotonicity is the one weakness: it is a sample, not a guarantee.",
    "evidence": ["src/data_pipeline/data_loader.py:269-318", "src/data_pipeline/data_loader.py:288-292", "docs/ISD_vae_latent_risk_factors.md:592-601"],
    "cross_units": [],
    "root_cause": null,
    "confidence": "high"
  },
  {
    "id": "U03-ADV-12",
    "unit": "U03",
    "role": "advocate",
    "type": "statistics",
    "severity": "solid",
    "evidence_register": "code+results",
    "claim": "The capacity guard AU_max_stat = floor(sqrt(2*N_eff/r_min)) with Kish effective sample size is correct, correctly applied, and binds: N_eff ~ 727 under a 252-day EWMA half-life gives AU_max_stat = 26, exactly the AU reported.",
    "claim_location": "src/inference/active_units.py:92-137",
    "assessment": "Correct math and correct truncation by descending KL. Two consequences: (i) the reported 26 factors are a data-depth verdict, not a modelling verdict (raw AU was 52, all dims above the KL threshold); (ii) the DVT's printed closed form at line 424 is dimensionally wrong (gives 60, not 85); the ISD form and the code are right.",
    "evidence": ["src/inference/active_units.py:92-117", "src/inference/active_units.py:120-137", "src/integration/pipeline.py:2017-2021", "results/diagnostic/diagnostic_data.json:7-9", "results/diagnostic/diagnostic_data.json:96-103", "results/diagnostic/config.json:130-131", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:424-426", "docs/ISD_vae_latent_risk_factors.md:222"],
    "cross_units": ["U08", "U09", "U15", "U17"],
    "root_cause": null,
    "confidence": "high"
  },
  {
    "id": "U03-ADV-13",
    "unit": "U03",
    "role": "advocate",
    "type": "data",
    "severity": "conditional",
    "evidence_register": "code+results",
    "claim": "The panel spans 1995-01-03 to 2026-02-13 (31.2 years, 7,874 days), satisfying the 30-year requirement in calendar span, but with 35.31% missing values and 410 of 788 stocks above 20% missing.",
    "claim_location": "results/diagnostic/config.json:130-131",
    "assessment": "Span yes, continuity no. Only 639 stocks produce exposure rows. The sample is long-span, thin-coverage - not the dense large-cap panel the DVT assumes.",
    "evidence": ["results/diagnostic/config.json:130-131", "results/diagnostic/diagnostic_report.md:26-31", "results/diagnostic/diagnostic_data.json:96-103"],
    "cross_units": ["U12", "U14"],
    "root_cause": null,
    "confidence": "high"
  },
  {
    "id": "U03-ADV-14",
    "unit": "U03",
    "role": "advocate",
    "type": "leakage",
    "severity": "solid",
    "evidence_register": "code",
    "claim": "Crisis labelling uses an expanding-window P80 of VIX truncated at the fold's training end, with a continuous per-window crisis fraction; it is genuinely point-in-time but is never executed because both runners pass vix_data=None.",
    "claim_location": "src/data_pipeline/crisis.py:133-153",
    "assessment": "The design is correct and better than a binary flag. Condition: gamma=3.0 is inert in every observed run, so a core advertised mechanism contributes nothing to any result.",
    "evidence": ["src/data_pipeline/crisis.py:133-153", "src/data_pipeline/crisis.py:183-194", "src/integration/pipeline.py:1449-1455", "scripts/run_walk_forward.py:203", "scripts/run_diagnostic.py:444"],
    "cross_units": ["U06", "U07"],
    "root_cause": "No VIX ingestion step in either runner; the module requires a FRED API key that nothing supplies.",
    "confidence": "high"
  },
  {
    "id": "U03-ADV-15",
    "unit": "U03",
    "role": "advocate",
    "type": "justification",
    "severity": "conditional",
    "evidence_register": "code+doc",
    "claim": "Market cap is proxied as close x 63-day average volume x 20, documented in code as a rough ranking proxy; as a threshold device it converts the $500M cap rule into a ~$25M ADV rule, 12.5x the separate $2M ADV rule.",
    "claim_location": "scripts/download_tiingo.py:890-901",
    "assessment": "Defensible as a liquidity ranking; indefensible as a float-adjusted cap. The accidental consequence (a liquid large-cap sample) is consistent with the DVT's stated intent to exclude micro-caps, which is why this is conditional rather than fatal.",
    "evidence": ["scripts/download_tiingo.py:890-901", "src/data_pipeline/universe.py:87-100", "docs/DVT_strategie_facteurs_risque_latents_v4_1.md:269-271"],
    "cross_units": ["U09", "U15"],
    "root_cause": "Vendor lacks shares outstanding; the fundamentals phase named in the script docstring was never implemented.",
    "confidence": "high"
  },
  {
    "id": "U03-ADV-16",
    "unit": "U03",
    "role": "advocate",
    "type": "consistency",
    "severity": "fragile",
    "evidence_register": "code+doc",
    "claim": "share_code is hard-coded to 10 for every ticker and any unmapped exchange defaults to NYSE, so the common-equity and exchange filters pass vacuously and the Shumway exchange split would key off a vendor string.",
    "claim_location": "scripts/download_tiingo.py:908-909",
    "assessment": "assetType == 'Stock' does exclude ETFs, which is the substantive part of the security filter; ADRs, REITs, preferreds and warrants are not excluded despite the spec.",
    "evidence": ["scripts/download_tiingo.py:87-94", "scripts/download_tiingo.py:428-431", "scripts/download_tiingo.py:841-845", "src/data_pipeline/universe.py:66-74", "docs/ISD_vae_latent_risk_factors.md:598-606"],
    "cross_units": ["U17"],
    "root_cause": "Vendor schema mapped onto CRSP column names without a real mapping for share type or exchange.",
    "confidence": "high"
  },
  {
    "id": "U03-ADV-17",
    "unit": "U03",
    "role": "advocate",
    "type": "data",
    "severity": "fragile",
    "evidence_register": "code",
    "claim": "The downloader orders current S&P 500 constituents first and stops when the key budget is exhausted; 451 ticker statuses are recorded against 22,790 supported US stock rows, so any partial download is constituent-biased.",
    "claim_location": "scripts/download_tiingo.py:437-445",
    "assessment": "Hypothesis with a testable prediction: among the 788 permnos in the parquet, the share with is_sp500 == True should far exceed the 2.2% base rate of the supported list, and few tickers should have a last observation before 2025. If confirmed, the panel is close to a current-index survivor set.",
    "evidence": ["scripts/download_tiingo.py:437-445", "scripts/download_tiingo.py:671-679", "scripts/download_tiingo.py:96-98", "data/tiingo_meta/supported_tickers.csv", "data/tiingo_meta/download_progress.json"],
    "cross_units": ["U12", "U16"],
    "root_cause": "Prioritisation rule optimised for early partial results rather than for unbiased coverage.",
    "confidence": "medium"
  },
  {
    "id": "U03-ADV-18",
    "unit": "U03",
    "role": "advocate",
    "type": "implementation",
    "severity": "solid",
    "evidence_register": "code",
    "claim": "Two-pass window counting followed by single pre-allocation, float32, sliding_window_view and in-place gap filling keep a 788x7,874 panel tractable under a 16 GB budget without trading correctness for memory.",
    "claim_location": "src/data_pipeline/windowing.py:148-178",
    "assessment": "Filters are reapplied identically in pass 2 and the output is asserted, so the memory optimisation does not fork the logic. Deliberate engineering rather than incidental.",
    "evidence": ["src/data_pipeline/windowing.py:148-178", "src/data_pipeline/windowing.py:275-305", "src/data_pipeline/data_loader.py:234-239", "src/data_pipeline/returns.py:85-86"],
    "cross_units": [],
    "root_cause": null,
    "confidence": "high"
  }
]
```

END-OF-REPORT
