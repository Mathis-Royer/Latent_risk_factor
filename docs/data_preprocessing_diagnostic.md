# Data Preprocessing Pipeline Diagnostic

> **Purpose**: Comprehensive diagnostic of all filtering, cleaning, transformations, and splits in the data preprocessing pipeline. Identifies missing runtime validations, code bugs, and gaps relative to research best practices.

---

## Table of Contents

1. [Data Flow Summary](#1-data-flow-summary)
2. [Filtering & Cleaning Analysis](#2-filtering--cleaning-analysis)
3. [Transformation Details](#3-transformation-details)
4. [Train/Val/Test Split Logic](#4-trainvaltest-split-logic)
5. [Missing Runtime Validations](#5-missing-runtime-validations)
6. [Code Bugs Beyond Validation](#6-code-bugs-beyond-validation)
7. [Research Best Practices Conformity](#7-research-best-practices-conformity)
8. [Recommended Fixes](#8-recommended-fixes)

---

## 1. Data Flow Summary

```
Stock Data (CSV/Parquet/Tiingo)
    │
    ├─ data_loader.py: load_tiingo_data() / load_stock_data()
    │   ├── Read from tiingo_us_equities.parquet
    │   ├── Parse dates → pd.Timestamp
    │   ├── Validate 8 core columns present
    │   ├── Sort by (permno, date)
    │   ├── Filter: adj_price >= min_price ($1.00)
    │   └── Filter: trading days >= min_history_days (504)
    │
    ├─ returns.py: compute_log_returns()
    │   ├── Pivot to wide format (dates × permnos)
    │   ├── Forward-fill small gaps (≤5 consecutive NaN)
    │   ├── Compute r_t = ln(P_t / P_{t-1})  [CONV-01]
    │   └── Apply delisting returns (Shumway imputation)
    │
    ├─ features.py: compute_trailing_volatility() + compute_rolling_realized_vol()
    │   ├── Trailing 252d annualized vol: σ * sqrt(252)
    │   └── Rolling 21d raw std (second VAE feature, F=2)
    │
    ├─ universe.py: construct_universe() + build_training_universe()
    │   ├── Point-in-time reconstitution (CONV-10)
    │   ├── Eligibility: share_code ∈ {10, 11}, exchange ∈ {1, 2, 3}
    │   ├── Min listing: ≥504 trading days
    │   ├── Market cap: hysteresis 500M entry / 400M exit
    │   ├── ADV: ≥$2M over trailing 63 days
    │   └── Liveness: last valid price ≤10 days ago
    │
    ├─ windowing.py: create_windows()
    │   ├── Sliding windows (N, T=504, F=2)
    │   ├── Filter: ≤5% NaN in returns OR vol → skip
    │   ├── Filter: ≤20% zero-return days → skip
    │   ├── NaN handling: returns → 0, vol → ffill/bfill → 0
    │   ├── Z-score per-window, per-feature (CONV-02)
    │   └── Clamp σ_min = 1e-8 to prevent division by zero
    │
    └─ crisis.py: compute_crisis_labels()
        ├── VIX threshold = P80 on expanding training window
        ├── Per-window crisis fraction f_c ∈ [0, 1]
        └── No look-ahead: threshold uses data up to training_end_date

    → Output: (N, 504, 2) z-scored windows + metadata + raw_returns + crisis_fractions
```

### Key Module Responsibilities

| Module | File | Primary Function |
|--------|------|------------------|
| Data Loading | `data_loader.py` | Load CSV/Parquet, validate schema, filter penny stocks |
| Returns | `returns.py` | Log-return computation, gap-fill, delisting imputation |
| Features | `features.py` | Trailing and rolling volatility |
| Universe | `universe.py` | Point-in-time eligibility, hysteresis, ADV/liveness |
| Windowing | `windowing.py` | Sliding windows, z-scoring, quality filters |
| Crisis | `crisis.py` | VIX-based crisis labeling |

---

## 2. Filtering & Cleaning Analysis

### Data Loading Stage

| Stage | Filter | Threshold | Implementation | Estimated Data Loss |
|-------|--------|-----------|----------------|---------------------|
| Tiingo load | Penny stock | `adj_price >= $1.00` | `data_loader.py:212-214` | ~5-10% rows |
| Tiingo load | Min history | `≥504 trading days` | `data_loader.py:217-219` | ~20-30% stocks |
| Date filter | Start date | `date >= start_date` | `data_loader.py:252-253` | Variable |
| Date filter | End date | `date <= end_date` | `data_loader.py:254-255` | Variable |

### Returns Stage

| Stage | Filter | Threshold | Implementation | Estimated Data Loss |
|-------|--------|-----------|----------------|---------------------|
| Gap fill | Forward-fill | `≤5 consecutive NaN` | `returns.py:72-112` | Rare (fills, not drops) |
| Gap fill | Large gaps | `>5 consecutive NaN` | `returns.py:72-112` | Remain NaN |
| Delisting | Imputation | Shumway 1997 | `returns.py:146-188` | ~50% delistings imputed |

**Shumway Delisting Returns**:
- NYSE/AMEX: `ln(1 - 0.30) ≈ -0.357` (explicit or imputed)
- NASDAQ: `ln(1 - 0.55) ≈ -0.799` (explicit or imputed)

### Universe Construction Stage

| Stage | Filter | Threshold | Implementation | Estimated Data Loss |
|-------|--------|-----------|----------------|---------------------|
| Share code | Common equity | `share_code ∈ {10, 11}` | `universe.py:65-66` | ~0-2% |
| Exchange | NYSE/AMEX/NASDAQ | `exchange_code ∈ {1, 2, 3}` | `universe.py:70-71` | ~0% |
| Listing | Min trading days | `≥504 days` | `universe.py:75-77` | ~5-10% |
| Market cap | Entry threshold | `≥$500M` (new) | `universe.py:79-91` | ~5-20% churn |
| Market cap | Exit threshold | `≥$400M` (existing) | `universe.py:89` | Hysteresis band |
| ADV | Min dollar volume | `≥$2M trailing 63d` | `universe.py:93-98` | ~2-5% |
| Liveness | Recent price | `last valid ≤10 days` | `universe.py:100-103` | ~1-3% |

### Windowing Stage

| Stage | Filter | Threshold | Implementation | Estimated Data Loss |
|-------|--------|-----------|----------------|---------------------|
| NaN fraction | Returns | `≤5% NaN` | `windowing.py:104-108` | ~10-20% windows |
| NaN fraction | Volatility | `≤5% NaN` | `windowing.py:104-108` | Combined with above |
| Zero returns | Suspension proxy | `≤20% zero-return days` | `windowing.py:122-127` | ~5-10% windows |
| σ_min clamp | Prevent div/0 | `σ ≥ 1e-8` | `windowing.py:137, 142` | <0.01% windows |

---

## 3. Transformation Details

### 3.1 Log-Returns (CONV-01)

**Formula**:
```
r_{i,t} = ln(P^adj_{i,t} / P^adj_{i,t-1})
```

**Location**: `returns.py:61-62`

**Edge Cases**:
- **Zero prices**: Would produce `ln(0) = -∞`. Prevented by penny stock filter (`adj_price >= $1.00`).
- **Negative prices**: Invalid data. No explicit check — would produce `ln(negative) = NaN`.
- **Gap handling**: Forward-fill up to 5 consecutive NaN → 0 return for those days. Gaps >5 remain NaN.

**Why log-returns?**:
- Additivity: `r_cumulative = Σ r_t` (vs multiplicative for arithmetic)
- Symmetry: -10% and +10% are equally distant from 0
- Match Fama-French / standard academic practice

### 3.2 Z-Scoring (CONV-02)

**Formula**:
```
r̃_{i,window} = (r - μ_r) / max(σ_r, σ_min)
ṽ_{i,window} = (v - μ_v) / max(σ_v, σ_min)
```

**Location**: `windowing.py:134-143`

**Critical Design Decision**: Per-window, per-feature normalization (NOT cross-sectional).

**Rationale**:
- Preserves temporal patterns within each window
- Co-movement loss requires comparable windows from same time block
- Cross-sectional z-scoring would destroy stock-specific volatility information

**σ_min Clamping**:
- `σ_min = 1e-8` prevents division by zero for nearly-constant windows
- Rare case: stock suspended for most of the window → near-zero variance

### 3.3 Delisting Imputation (Shumway 1997)

**Location**: `returns.py:146-188`

**Implementation**:
```python
SHUMWAY_NYSE_AMEX = np.log(1.0 + (-0.30))  # -30% → ln(0.70) ≈ -0.357
SHUMWAY_NASDAQ = np.log(1.0 + (-0.55))      # -55% → ln(0.45) ≈ -0.799
```

**Logic**:
1. If `delisting_return` is provided in data: convert to log-return
2. If NaN: impute based on exchange code (NYSE/AMEX → -30%, NASDAQ → -55%)
3. Apply at the last valid trading date

**Survivorship Bias Mitigation**:
- Without delisting imputation, backtests ignore the catastrophic loss from delistings
- Shumway (1997) showed this bias inflates returns by ~2-3% annually

### 3.4 Volatility Features

**Trailing Volatility** (for risk model rescaling):
```
σ_{i,t}^{trail} = std(r_{i, t-251:t}) × sqrt(252)
```
- 252-day rolling window
- Annualized
- Location: `features.py:18-37`

**Rolling Realized Volatility** (second VAE feature):
```
v_{i,τ} = std(r_{i, τ-20:τ})
```
- 21-day rolling window
- NOT annualized (raw daily std)
- Location: `features.py:40-63`

---

## 4. Train/Val/Test Split Logic

### Walk-Forward Fold Structure

```
|-------- Training (8y) --------|--- Val (3mo) ---|--- OOS (6mo) ---|
|-- Warmup --|--- Available ----|                 |                 |
     504d        ~7.0y                3mo              6mo
```

**Fold Schedule** (`walk_forward/folds.py`):
- ~34 folds over 30 years (1994-2024)
- 6-month OOS periods
- 3-month validation buffer
- 504-day warmup for volatility features

### Validation Split Logic

**Location**: `pipeline.py` (train_start/train_end/val_start/val_end/oos_start/oos_end)

**Window Assignment**:
```python
# Validation: window end_date >= val_start
val_mask = window_metadata["end_date"] >= val_start

# Training: window end_date < val_start
train_mask = window_metadata["end_date"] < val_start
```

**Critical Point**: Windows are assigned based on `end_date`, not `start_date`. This ensures:
- No look-ahead bias (window doesn't span into validation period)
- Validation windows reflect the most recent training data

### Embargo

**Implementation**: Implicit in fold scheduling, NOT explicit in `pipeline.py`.

The fold scheduler ensures:
- OOS start ≥ train end + validation buffer
- No window straddles training/validation boundary

**Gap**: No explicit embargo parameter. If validation buffer is reduced, training windows could leak into validation.

### OOS Extraction

**Location**: `pipeline.py:~1050-1080`

```python
returns_oos = returns.loc[oos_start:oos_end]
```

**Risk**: `oos_start` or `oos_end` may not exist in the returns index if:
- Market closures (holidays)
- Data gaps

**Missing Check**: No validation that `oos_start <= oos_end` or that the slice is non-empty.

---

## 5. Missing Runtime Validations

### CRITICAL (NaN/Shape/Alignment Issues)

| # | Module | Location | Missing Check | Impact | Recommended Fix |
|---|--------|----------|---------------|--------|-----------------|
| 1 | `loss.py` | L306 | NaN in `raw_returns` before Spearman | NaN gradients, training collapse | `assert not torch.isnan(raw_returns).any()` |
| 2 | `windowing.py` | L43-51 | `returns_df.shape == vol_df.shape` | Silent misalignment, wrong z-scoring | `assert returns_df.shape == vol_df.shape` |
| 3 | `model.py` | L109-118 | Input shape `(B, T, F)` validation | Wrong transpose, corrupt forward pass | `assert x.shape[1:] == (self.T, self.F)` |
| 4 | `composite.py` | L94 | `"stock_id"` column exists | KeyError crash | `assert "stock_id" in window_metadata.columns` |
| 5 | `covariance.py` | L190+ | `z_hat.ndim == 2` | Shape propagation errors | `assert z_hat.ndim == 2` |
| 6 | `pipeline.py` | L1055 | `train_start`/`end` exist in returns index | Empty slice → 0-sample error | Check dates exist or use `.loc` with bounds check |
| 7 | `entropy.py` | L130 | `eigenvalues >= 0` | Negative risk contributions | `assert np.all(eigenvalues >= 0)` |
| 8 | `sca_solver.py` | L105 | Cholesky output finite | Singular matrix → NaN | `assert np.isfinite(L).all()` |
| 9 | `trainer.py` | L329 | `raw_ret.shape == (B, T)` | Batch mismatch in co-movement | Already added in recent fix |
| 10 | `data_loader.py` | L249 | No NaT after `pd.to_datetime` | Silent NaT propagation | `assert not df["date"].isna().any()` |

### MODERATE (Value Ranges/Types)

| # | Module | Location | Missing Check | Impact | Recommended Fix |
|---|--------|----------|---------------|--------|-----------------|
| 11 | `returns.py` | L184 | `delisting_return ∈ [-0.99, 10]` | Log overflow for extreme values | Clamp before `np.log(1 + delist_ret)` |
| 12 | `universe.py` | L94 | `len(recent) == 63` for ADV | Wrong ADV with short history | Use `min_periods` or validate |
| 13 | `factor_regression.py` | L82 | `valid_mask` bounds check | IndexError potential | See Code Bugs section |
| 14 | `windowing.py` | L85-86 | `permno in both returns_df and vol_df` | Skip but no warning | Log warning for missing permnos |
| 15 | `crisis.py` | L176-183 | VIX data coverage | Zero crisis fraction if no VIX | Log warning if VIX sparse |
| 16 | `covariance.py` | L229 | `n_samples > p_dims` for shrinkage | DGJ assumptions violated | Warn if gamma > 1 |
| 17 | `entropy.py` | L121-124 | `D_eps` length matches `w` | Already added assertion | — |
| 18 | `sca_solver.py` | L580-583 | `w_init` sums to ~1 | Projection may fail | `assert abs(w_init.sum() - 1) < 1e-3` |
| 19 | `cardinality.py` | L159 | `active_indices` no duplicates | Incorrect weight assignment | `assert len(set(active_indices)) == len(active_indices)` |
| 20 | `metrics.py` | L91-92 | No duplicate stock IDs in `latent_stability()` | Distance matrix mismatch | Validate uniqueness |

### LOW (Logging/Warnings)

| # | Module | Location | Missing Check | Impact | Recommended Fix |
|---|--------|----------|---------------|--------|-----------------|
| 21 | `universe.py` | L104-108 | Eligible list empty | Silent return of `[]` | Log warning |
| 22 | `features.py` | L35-37 | Window > len(returns) | All-NaN output | Log warning |
| 23 | `windowing.py` | L160-166 | Zero windows created | Silent empty tensor | Log warning |
| 24 | `crisis.py` | L136-139 | No VIX data in range | Raises ValueError | Convert to warning + default |
| 25 | `covariance.py` | L368 | Eigenvalue sum ≤ 0 | Return input unchanged | Log warning |

---

## 6. Code Bugs Beyond Validation

### CRITICAL

**Bug 1: Index Mismatch in `factor_regression.py:82-83`**

```python
# Current code
for i in range(len(avail_stocks)):
    # ...
    B_t_valid = B_t[valid_mask]
    valid_sids = [avail_stocks[j] for j in range(len(avail_stocks)) if valid_mask[j]]
```

**Problem**: `valid_mask` is created from `r_t` which has length `len(avail_stocks)`, but after filtering, `B_t_valid` and `valid_sids` may have different lengths than `B_t`.

**Impact**: `IndexError` or silent misalignment when `B_t.shape[0] != len(avail_stocks)`.

**Fix**: Ensure `B_t` is sliced to `avail_stocks` before applying `valid_mask`.

---

**Bug 2: Metadata Slicing Assumes Sorted Order in `pipeline.py:1254,1668`**

```python
# Current code assumes metadata and windows are aligned
window_metadata_train = window_metadata.iloc[:n_train]
```

**Problem**: If windows are reordered (e.g., by CurriculumBatchSampler), metadata indices no longer match.

**Impact**: Wrong stock_id → window mapping, corrupted co-movement loss.

**Fix**: Use explicit index alignment or pass indices through the batch.

---

**Bug 3: `active_indices` Not Validated in `cardinality.py:159`**

```python
active_idx = np.array(sorted(active_set), dtype=np.intp)
```

**Problem**: `active_set` may contain duplicates if pre-screening logic has bugs.

**Impact**: Duplicate binary variables in MIQP, incorrect constraint enforcement.

**Fix**: Add `assert len(active_set) == len(set(active_set))`.

---

### MODERATE

**Bug 4: Duplicate Stock IDs in `metrics.py:91-92` (`latent_stability`)**

```python
set_current = set(ids_current)
set_previous = set(ids_previous)
common_ids = sorted(set_current & set_previous)
```

**Problem**: If `ids_current` or `ids_previous` have duplicates, `set()` removes them but the row indices computed later may be wrong.

**Impact**: Wrong rows selected for distance comparison, incorrect latent stability metric.

**Fix**: Validate uniqueness or use `dict` for index lookup.

---

**Bug 5: D_eps Shape Mismatch in `oos_rebalancing.py:196`**

```python
# When stocks missing from returns_oos
D_eps_active = D_eps[:len(available)]  # May not align
```

**Problem**: `D_eps` is ordered by `stock_ids`, but `available` filters by presence in `returns_oos.columns`. Order may differ.

**Impact**: Wrong idiosyncratic variance assigned to stocks.

**Fix**: Use explicit ID-to-D_eps mapping: `D_eps_active = np.array([D_eps_dict[sid] for sid in available])`.

---

**Bug 6: `n_active` Bounds in `metrics.py:248`**

```python
n_active = min(len(active_sids), B_t.shape[0])
for i in range(n_active):
    sid = active_sids[i]
```

**Problem**: Assumes `active_sids[:n_active]` maps to `B_t[:n_active]`. If `B_t` was filtered differently, this is wrong.

**Impact**: Wrong residuals computed for explanatory power.

**Fix**: Align by stock ID, not position.

---

**Bug 7: Degenerate Eigenvalue Spectrum in `factor_quality.py:204`**

```python
# No warning when all eigenvalues equal (no informative factors)
eigenvalues = np.maximum(eigenvalues, 0.0)
```

**Problem**: If shrinkage collapses all eigenvalues to σ² (bulk), `n_signal=0` but no diagnostic warning.

**Impact**: Pipeline continues with no informative factors, poor portfolio.

**Fix**: Add `if n_signal == 0: logger.warning("No signal eigenvalues detected")`.

---

## 7. Research Best Practices Conformity

### ✅ CORRECT (Aligned with Best Practices)

| Practice | Implementation | Reference |
|----------|----------------|-----------|
| Log returns only | `CONV-01` enforced in `returns.py` | Fama-French, Campbell et al. (1997) |
| Point-in-time universe | `CONV-10` in `universe.py` | No look-ahead bias |
| Survivorship bias handling | Shumway (1997) delisting imputation | `returns.py:146-188` |
| Walk-forward validation | ~34 folds, 6-month OOS | Prevents data snooping |
| Multiple testing correction | Holm-Bonferroni on final results | Harvey & Liu (2020) |
| VIX crisis weighting | P80 threshold, expanding window | Point-in-time, no look-ahead |
| Covariance shrinkage | Ledoit-Wolf / DGJ spiked | Modern estimation theory |

### ⚠️ GAPS (Missing or Incomplete)

| Gap | Best Practice | Impact | Priority | Reference |
|-----|---------------|--------|----------|-----------|
| **No outlier winsorization** | Winsorize at ±4σ or 1st/99th percentile | Extreme returns distort VAE latents, inflate variance | MEDIUM | Harvey & Liu (2020) |
| **Phase A HP selection not corrected** | Bonferroni across HP configurations | Spurious HP selection (p-hacking) | MEDIUM | Harvey et al. (2016) |
| **No price discontinuity validation** | Flag single-day moves >15% | Data errors propagate (splits, errors) | MEDIUM | Fama-French data cleaning |
| **Cross-sectional normalization absent** | Document as intentional design choice | Co-movement loss requires raw cross-section | LOW | Documented in CONV-02 |
| **Sector concentration not monitored** | Track sector weights, warn if >30% | Sector crash risk not hedged | LOW | Diversification literature |
| **No return clipping during crisis** | Clip extreme returns during VIX >40 | Crisis outliers dominate training | LOW | Robust statistics |

### Detailed Gap Analysis

#### Gap 1: Outlier Winsorization

**Issue**: Extreme returns (e.g., ±50% single-day) pass through unfiltered.

**Evidence**: The VAE receives z-scored windows, but z-scoring a window with one extreme return pushes all other values toward 0.

**Best Practice**: Winsorize daily returns at ±4σ (cross-sectional) or 1st/99th percentile before z-scoring.

**References**:
- Harvey, C. R., & Liu, Y. (2020). "False (and missed) discoveries in financial economics."
- Jegadeesh, N., & Titman, S. (1993). Standard momentum filtering.

**Implementation Location**: Would add to `returns.py` after log-return computation.

---

#### Gap 2: Phase A HP Selection

**Issue**: Phase A tests multiple hyperparameter configurations and selects the best. No multiple testing penalty applied.

**Current**: `selection.py` scores configurations but doesn't adjust p-values.

**Best Practice**: Apply Bonferroni or BH correction to HP selection (not just final benchmark comparison).

**References**:
- Harvey, C. R., Liu, Y., & Zhu, H. (2016). "...and the Cross-Section of Expected Returns."

**Implementation Location**: `walk_forward/phase_a.py` or `selection.py`.

---

#### Gap 3: Price Discontinuity

**Issue**: Stock splits, data errors, and corporate actions can create artificial discontinuities.

**Current**: No validation. A 10x price error propagates as ln(10) ≈ 2.3 daily return.

**Best Practice**: Flag returns > ±15% single-day, cross-check against volume/market cap.

**Implementation Location**: `returns.py` after log-return computation.

---

## 8. Recommended Fixes

### Priority 1: CRITICAL (Runtime Validations)

**Fix 1.1**: Add NaN check before Spearman in `loss.py`
```python
# At loss.py:305 (before _batch_spearman call)
assert not torch.isnan(raw_returns).any(), "NaN in raw_returns before Spearman"
```

**Fix 1.2**: Add input shape assertion in `model.py`
```python
# At model.py:118 (start of forward())
assert x.shape[1:] == (self.T, self.F), f"Expected shape (B, {self.T}, {self.F}), got {x.shape}"
```

**Fix 1.3**: Add alignment check in `windowing.py`
```python
# At windowing.py:44 (before processing)
assert returns_df.shape == vol_df.shape, "returns and vol DataFrames must have same shape"
```

**Fix 1.4**: Add stock_id column check in `composite.py`
```python
# At composite.py:93 (before groupby)
assert "stock_id" in window_metadata.columns, "window_metadata must have 'stock_id' column"
```

**Fix 1.5**: Create `src/validation.py` module
```python
# Centralized validators
def assert_finite_2d(arr: np.ndarray, name: str) -> None:
    assert arr.ndim == 2, f"{name} must be 2D, got {arr.ndim}D"
    assert np.isfinite(arr).all(), f"{name} contains NaN/Inf"

def assert_positive_semidefinite(Sigma: np.ndarray, name: str) -> None:
    eigenvalues = np.linalg.eigvalsh(Sigma)
    assert np.all(eigenvalues >= -1e-10), f"{name} is not PSD (min eigenvalue: {eigenvalues.min()})"
```

---

### Priority 2: MEDIUM (Code Bugs)

**Fix 2.1**: Fix index alignment in `factor_regression.py:82`
```python
# Ensure B_t is sliced to match avail_stocks before valid_mask
B_t_sliced = B_t[:len(avail_stocks)]  # or use explicit mapping
r_t = ret_matrix[date_loc][col_indices].astype(np.float64)
valid_mask = ~np.isnan(r_t)
B_t_valid = B_t_sliced[valid_mask]
valid_sids = [avail_stocks[j] for j in range(len(avail_stocks)) if valid_mask[j]]
```

**Fix 2.2**: Fix D_eps alignment in `oos_rebalancing.py:196`
```python
# Use explicit mapping
D_eps_dict = dict(zip(stock_ids, D_eps))
D_eps_active = np.array([D_eps_dict.get(sid, d_eps_floor) for sid in available])
```

**Fix 2.3**: Add duplicate check in `metrics.py`
```python
# At latent_stability() start
if ids_current is not None:
    assert len(ids_current) == len(set(ids_current)), "Duplicate stock IDs in ids_current"
if ids_previous is not None:
    assert len(ids_previous) == len(set(ids_previous)), "Duplicate stock IDs in ids_previous"
```

---

### Priority 3: MEDIUM (Best Practices)

**Fix 3.1**: Add optional winsorization in `returns.py`
```python
def winsorize_returns(
    returns_df: pd.DataFrame,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pd.DataFrame:
    """Winsorize returns at specified percentiles (cross-sectional per date)."""
    from scipy.stats import mstats
    return returns_df.apply(
        lambda row: mstats.winsorize(row, limits=[lower_pct, 1 - upper_pct]),
        axis=1,
        result_type='broadcast',
    )
```

**Fix 3.2**: Add Bonferroni to Phase A selection
```python
# In selection.py, when comparing HP configurations
n_configs = len(hp_grid)
alpha_adjusted = 0.05 / n_configs  # Bonferroni
```

**Fix 3.3**: Add price discontinuity warning in `data_loader.py`
```python
# After loading, check for suspicious returns
returns_check = returns_df.diff()
extreme_mask = np.abs(returns_check) > np.log(1.15)  # >15% single-day
n_extreme = extreme_mask.sum().sum()
if n_extreme > 0:
    logger.warning(f"Found {n_extreme} single-day returns >15%. Consider data quality check.")
```

---

## Appendix: Test Coverage Checklist

| Module | Unit Tests | Integration Tests | Missing Coverage |
|--------|------------|-------------------|------------------|
| `data_loader.py` | ✅ | ✅ | NaT handling edge case |
| `returns.py` | ✅ | ✅ | Extreme delisting return overflow |
| `features.py` | ✅ | ✅ | Edge case: window > data length |
| `universe.py` | ✅ | ✅ | Empty eligible list warning |
| `windowing.py` | ✅ | ✅ | returns/vol shape mismatch |
| `crisis.py` | ✅ | ✅ | Sparse VIX coverage |
| `factor_regression.py` | ✅ | ✅ | B_t / avail_stocks alignment |
| `metrics.py` | ✅ | ✅ | Duplicate stock ID handling |

---

**Document Version**: 1.0
**Last Updated**: 2026-02-21
**Author**: Claude Code Diagnostic
