"""
Download US equity daily data from Tiingo API with persistent storage.

Features:
    - Multi-key rotation: provide N API keys, round-robin with auto-failover
    - Resumable: tracks progress per ticker, restarts from where it stopped
    - Incremental updates: re-run fetches only new dates for existing tickers
    - Phase-based execution: prices / fundamentals / merge / all

Usage:
    # Download prices (1 API call per ticker)
    python scripts/download_tiingo.py --phase prices --api-keys key1,key2

    # Or via env var
    TIINGO_API_KEYS=key1,key2 python scripts/download_tiingo.py --phase prices

    # Or via file (one key per line)
    python scripts/download_tiingo.py --phase prices --keys-file keys.txt

    # Download SP500 tickers first, then remaining US equities
    python scripts/download_tiingo.py --phase all --api-keys key1 --sp500-first

    # Use multiple API keys (per-key rotation: 45 tickers per key, then stop)
    python scripts/download_tiingo.py --phase all --api-keys key1,key2,key3 --sp500-first

    # Use a local SP500 ticker list (offline, no Wikipedia fetch)
    python scripts/download_tiingo.py --phase all --api-keys key1 --sp500-first --sp500-file sp500.csv

    # Unattended mode: sleep ~62 min when all keys exhausted, then continue
    python scripts/download_tiingo.py --phase all --api-keys key1 --wait-on-rate-limit

    # Merge downloaded data into pipeline-ready Parquet (no API calls)
    python scripts/download_tiingo.py --phase merge --output-dir data/

    # Run all phases
    python scripts/download_tiingo.py --phase all --api-keys key1

Output structure:
    data/
    ├── tiingo_meta/
    │   ├── supported_tickers.csv
    │   ├── ticker_to_permno.json
    │   └── download_progress.json
    ├── tiingo_raw/
    │   ├── AAPL.parquet
    │   └── ...
    └── tiingo_us_equities.parquet
"""

import argparse
import io
import json
import logging
import os
import sys
import time
import zipfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIINGO_BASE_URL = "https://api.tiingo.com"
SUPPORTED_TICKERS_URL = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"

WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

VALID_EXCHANGES = {"NYSE", "AMEX", "NASDAQ", "NYSE ARCA", "NYSE MKT", "BATS"}

EXCHANGE_CODE_MAP: dict[str, int] = {
    "NYSE": 1,
    "NYSE ARCA": 1,
    "NYSE MKT": 2,
    "AMEX": 2,
    "NASDAQ": 3,
    "BATS": 1,
}

# Rate limit: ~50 unique symbols per hour per key (conservative)
REQUESTS_PER_BATCH = 45
BATCH_SLEEP_SECONDS = 3700  # ~61.7 minutes between batches of 45

# Global retry budget: after this many errors, skip all subsequent failures
MAX_GLOBAL_RETRIES = 2


# ---------------------------------------------------------------------------
# Key Rotator
# ---------------------------------------------------------------------------

class KeyRotator:
    """
    Round-robin API key rotation with per-key rate-limit tracking.

    Keys can be in 3 states:
    - Active: can be used for requests
    - Rate-limited: batch limit reached (temporary, resets after cooldown)
    - Disabled: permanently unusable (auth failure)

    :param keys (list[str]): List of API keys
    """

    def __init__(self, keys: list[str]) -> None:
        if not keys:
            raise ValueError("At least one API key is required.")
        self.keys = keys
        self.current_index = 0
        self.disabled: set[int] = set()
        self.rate_limited: set[int] = set()
        self.call_counts: list[int] = [0] * len(keys)

    def next_key(self) -> str | None:
        """
        Get the next available API key (round-robin, skip disabled and rate-limited).

        :return key (str | None): Next valid API key, or None if all exhausted
        """
        n = len(self.keys)
        for _ in range(n):
            idx = self.current_index % n
            self.current_index += 1
            if idx not in self.disabled and idx not in self.rate_limited:
                return self.keys[idx]
        return None

    def record_use(self, key: str) -> None:
        """
        Record a successful API call. Auto-marks key as rate-limited at batch limit.

        :param key (str): The API key that was used
        """
        for i, k in enumerate(self.keys):
            if k == key:
                self.call_counts[i] += 1
                if self.call_counts[i] >= REQUESTS_PER_BATCH:
                    self.rate_limited.add(i)
                    logger.info(
                        "Key #%d reached batch limit (%d/%d calls, masked: ...%s)",
                        i + 1, self.call_counts[i], REQUESTS_PER_BATCH, key[-4:],
                    )
                break

    def mark_rate_limited(self, key: str) -> None:
        """
        Mark a key as rate-limited (temporary — 429 response).

        :param key (str): The API key to mark
        """
        for i, k in enumerate(self.keys):
            if k == key:
                self.rate_limited.add(i)
                logger.info(
                    "Key #%d rate-limited by server (masked: ...%s)", i + 1, key[-4:],
                )
                break

    def disable_key(self, key: str) -> None:
        """
        Permanently disable a key (e.g., after 401/403 auth failure).

        :param key (str): The API key to disable
        """
        for i, k in enumerate(self.keys):
            if k == key:
                self.disabled.add(i)
                logger.warning("Disabled API key #%d (masked: ...%s)", i + 1, key[-4:])
                break

    def reset_rate_limits(self) -> None:
        """
        Reset all rate-limited keys (after cooldown). Does NOT reset disabled keys.
        """
        self.rate_limited.clear()
        self.call_counts = [0] * len(self.keys)

    @property
    def active_count(self) -> int:
        """Number of active (non-disabled) keys."""
        return len(self.keys) - len(self.disabled)

    @property
    def available_count(self) -> int:
        """Number of keys that are neither disabled nor rate-limited."""
        return len(self.keys) - len(self.disabled | self.rate_limited)

    def validate_all(self) -> int:
        """
        Test each key with a lightweight API call. Disable invalid ones.

        :return n_valid (int): Number of valid keys
        """
        for i, key in enumerate(self.keys):
            try:
                resp = requests.get(
                    f"{TIINGO_BASE_URL}/api/test",
                    headers={"Authorization": f"Token {key}"},
                    timeout=10,
                )
                if resp.status_code in (401, 403):
                    self.disabled.add(i)
                    logger.warning("Key #%d invalid (HTTP %d)", i + 1, resp.status_code)
                else:
                    logger.info("Key #%d valid (masked: ...%s)", i + 1, key[-4:])
            except requests.RequestException as e:
                self.disabled.add(i)
                logger.warning("Key #%d unreachable: %s", i + 1, e)
        return self.active_count


# ---------------------------------------------------------------------------
# Progress Tracker
# ---------------------------------------------------------------------------

class ProgressTracker:
    """
    Persistent progress tracking for resumable downloads.

    :param progress_path (str): Path to the JSON progress file
    """

    def __init__(self, progress_path: str) -> None:
        self.path = progress_path
        self.data: dict[str, dict[str, str]] = {}
        if os.path.exists(progress_path):
            with open(progress_path) as f:
                self.data = json.load(f)

    def is_complete(self, ticker: str) -> bool:
        """
        Check if a ticker has been fully downloaded.

        :param ticker (str): Ticker symbol
        :return complete (bool): True if status is "complete"
        """
        return self.data.get(ticker, {}).get("status") == "complete"

    def get_last_date(self, ticker: str) -> str | None:
        """
        Get the last downloaded date for a ticker.

        :param ticker (str): Ticker symbol
        :return last_date (str | None): Last date string or None
        """
        return self.data.get(ticker, {}).get("last_date")

    def mark_complete(self, ticker: str, last_date: str) -> None:
        """
        Mark a ticker as complete with its last date.

        :param ticker (str): Ticker symbol
        :param last_date (str): Last date in the data
        """
        self.data[ticker] = {"status": "complete", "last_date": last_date}
        self._save()

    def mark_failed(self, ticker: str, reason: str) -> None:
        """
        Mark a ticker as failed.

        :param ticker (str): Ticker symbol
        :param reason (str): Failure reason
        """
        self.data[ticker] = {"status": "failed", "reason": reason}
        self._save()

    def _save(self) -> None:
        """Persist progress to disk."""
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)


# ---------------------------------------------------------------------------
# Permno Manager
# ---------------------------------------------------------------------------

class PermnoManager:
    """
    Stable ticker-to-permno mapping, persisted to JSON.

    :param mapping_path (str): Path to the JSON mapping file
    """

    def __init__(self, mapping_path: str) -> None:
        self.path = mapping_path
        self.mapping: dict[str, int] = {}
        self.next_id = 10001
        if os.path.exists(mapping_path):
            with open(mapping_path) as f:
                self.mapping = json.load(f)
            if self.mapping:
                self.next_id = max(self.mapping.values()) + 1

    def get_or_create(self, ticker: str) -> int:
        """
        Get existing permno or create a new one for a ticker.

        :param ticker (str): Ticker symbol
        :return permno (int): Stable integer ID
        """
        if ticker not in self.mapping:
            self.mapping[ticker] = self.next_id
            self.next_id += 1
        return self.mapping[ticker]

    def save(self) -> None:
        """Persist mapping to disk."""
        with open(self.path, "w") as f:
            json.dump(self.mapping, f, indent=2)


# ---------------------------------------------------------------------------
# Tiingo API Functions
# ---------------------------------------------------------------------------

def fetch_sp500_tickers(sp500_file: str | None = None) -> set[str]:
    """
    Fetch the current S&P 500 constituent tickers.

    Sources (in priority order):
    1. Local CSV file (if ``sp500_file`` is provided) — one ticker per line
       or a CSV with a "Symbol" or "ticker" column.
    2. Wikipedia "List of S&P 500 companies" page via ``pd.read_html()``.

    Tiingo uses hyphens for multi-class tickers (BRK-B), while Wikipedia
    uses dots (BRK.B). Both variants are included in the returned set.

    :param sp500_file (str | None): Optional path to a local CSV with SP500 tickers

    :return sp500 (set[str]): Set of SP500 ticker symbols (uppercased),
        including dot→hyphen variants
    """
    raw_tickers: set[str] = set()

    # Source 1: local file
    if sp500_file and os.path.exists(sp500_file):
        logger.info("Loading S&P 500 list from local file: %s", sp500_file)
        sp_df = pd.read_csv(sp500_file)
        # Accept "Symbol", "ticker", or single-column file
        if "Symbol" in sp_df.columns:
            raw_tickers = set(sp_df["Symbol"].astype(str).str.strip().str.upper())
        elif "ticker" in sp_df.columns:
            raw_tickers = set(sp_df["ticker"].astype(str).str.strip().str.upper())
        else:
            # Single-column file: use first column
            col = sp_df.columns[0]
            raw_tickers = set(sp_df[col].astype(str).str.strip().str.upper())
        logger.info("Loaded %d tickers from local file", len(raw_tickers))
    else:
        # Source 2: Wikipedia (use requests to set proper User-Agent)
        logger.info("Fetching S&P 500 constituent list from Wikipedia...")
        try:
            resp = requests.get(
                WIKIPEDIA_SP500_URL,
                headers={"User-Agent": "LatentRiskFactor/1.0 (research project)"},
                timeout=30,
            )
            resp.raise_for_status()
            tables = pd.read_html(io.StringIO(resp.text))
            sp500_df = tables[0]
            raw_tickers = set(
                sp500_df["Symbol"].astype(str).str.strip().str.upper()
            )
            logger.info("Found %d S&P 500 tickers from Wikipedia", len(raw_tickers))
        except Exception as e:
            logger.error("Failed to fetch S&P 500 list: %s", e)
            raise RuntimeError(
                "Could not fetch S&P 500 constituents from Wikipedia. "
                "Check your internet connection, or provide a local file "
                "via --sp500-file."
            ) from e

    # Normalize: include both dot and hyphen variants (BRK.B ↔ BRK-B)
    normalized: set[str] = set()
    for t in raw_tickers:
        normalized.add(t)
        normalized.add(t.replace(".", "-"))
    return normalized


def fetch_supported_tickers(
    output_dir: str,
    sp500_first: bool = False,
    sp500_file: str | None = None,
) -> pd.DataFrame:
    """
    Download and filter the supported tickers list from Tiingo.

    This endpoint is not rate-limited. Downloads a ZIP containing a CSV.

    :param output_dir (str): Base output directory (data/)
    :param sp500_first (bool): If True, reorder tickers so S&P 500 constituents
        appear first. Non-SP500 tickers are appended after. A column
        ``is_sp500`` is added to the output DataFrame.

    :return tickers_df (pd.DataFrame): Filtered DataFrame with columns:
        ticker, exchange, assetType, priceCurrency, startDate, endDate, [is_sp500]
    """
    meta_dir = os.path.join(output_dir, "tiingo_meta")
    os.makedirs(meta_dir, exist_ok=True)
    csv_path = os.path.join(meta_dir, "supported_tickers.csv")

    logger.info("Downloading supported tickers list from Tiingo...")
    resp = requests.get(SUPPORTED_TICKERS_URL, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)

    # Filter: US equities only
    df = pd.DataFrame(df[df["exchange"].isin(list(VALID_EXCHANGES))])
    df = pd.DataFrame(df[df["assetType"] == "Stock"])
    df = pd.DataFrame(df[df["priceCurrency"] == "USD"])

    # Drop rows without a ticker
    df = df.dropna(subset=["ticker"])
    df = df.reset_index(drop=True)

    # SP500 priority: tag and reorder so SP500 tickers are downloaded first
    if sp500_first:
        sp500_tickers = fetch_sp500_tickers(sp500_file=sp500_file)
        ticker_upper: pd.Series = df["ticker"].astype(str).str.upper()  # type: ignore[assignment]
        df["is_sp500"] = ticker_upper.isin(sp500_tickers)
        n_sp500 = int(pd.Series(df["is_sp500"]).sum())
        # Sort: SP500 first, then the rest (stable sort preserves original order within groups)
        df = df.sort_values("is_sp500", ascending=False, kind="mergesort")
        df = df.reset_index(drop=True)
        logger.info(
            "SP500 priority: %d SP500 tickers first, then %d others",
            n_sp500, len(df) - n_sp500,
        )

    df.to_csv(csv_path, index=False)
    n_active = int(pd.Series(df["endDate"]).isna().sum())
    n_delisted = int(pd.Series(df["endDate"]).notna().sum())
    logger.info(
        "Saved %d US equity tickers (%d active, %d delisted) to %s",
        len(df), n_active, n_delisted, csv_path,
    )
    return df


def fetch_ticker_prices(
    ticker: str,
    api_key: str,
    start_date: str = "1995-01-01",
    end_date: str | None = None,
) -> tuple[pd.DataFrame | None, str]:
    """
    Fetch daily EOD prices for a single ticker from Tiingo.

    :param ticker (str): Ticker symbol
    :param api_key (str): Tiingo API key
    :param start_date (str): Start date (YYYY-MM-DD)
    :param end_date (str | None): End date (YYYY-MM-DD), defaults to today

    :return df (pd.DataFrame | None): DataFrame with Tiingo price columns, or None on failure
    :return status (str): "ok", "not_found", "auth_failed", "rate_limited", "error"
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    url = f"{TIINGO_BASE_URL}/tiingo/daily/{ticker}/prices"
    params = {
        "startDate": start_date,
        "endDate": end_date,
        "format": "json",
    }
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)

        if resp.status_code in (401, 403):
            return None, "auth_failed"
        if resp.status_code == 404:
            logger.debug("Ticker %s not found (404)", ticker)
            return pd.DataFrame(), "not_found"
        if resp.status_code == 429:
            logger.warning("Rate limited on ticker %s (key ...%s)", ticker, api_key[-4:])
            return None, "rate_limited"

        resp.raise_for_status()

        # Empty or non-JSON body (delisted tickers) → no data, not an error
        body = resp.text.strip() if resp.text else ""
        if not body:
            logger.debug("Empty response for %s (no data on Tiingo)", ticker)
            return pd.DataFrame(), "ok"

        try:
            data = resp.json()
        except ValueError:
            logger.debug("Non-JSON response for %s — skipping (no data)", ticker)
            return pd.DataFrame(), "ok"

        if not data:
            return pd.DataFrame(), "ok"

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df, "ok"

    except requests.RequestException as e:
        logger.warning("Request failed for %s: %s", ticker, e)
        return None, "error"


# ---------------------------------------------------------------------------
# Phase Functions
# ---------------------------------------------------------------------------

def phase_discovery(
    output_dir: str,
    permno_mgr: PermnoManager,
    sp500_first: bool = False,
    sp500_file: str | None = None,
) -> pd.DataFrame:
    """
    Phase 1: Discover all US equity tickers (no rate limit).

    :param output_dir (str): Base output directory
    :param permno_mgr (PermnoManager): Permno assignment manager
    :param sp500_first (bool): If True, reorder so SP500 tickers appear first
    :param sp500_file (str | None): Optional local CSV with SP500 tickers

    :return tickers_df (pd.DataFrame): Filtered tickers
    """
    tickers_df = fetch_supported_tickers(
        output_dir, sp500_first=sp500_first, sp500_file=sp500_file,
    )

    # Assign permnos
    for ticker in tickers_df["ticker"]:
        permno_mgr.get_or_create(str(ticker))
    permno_mgr.save()

    logger.info("Assigned permnos to %d tickers", len(tickers_df))
    return tickers_df


def phase_prices(
    output_dir: str,
    tickers_df: pd.DataFrame,
    key_rotator: KeyRotator,
    progress: ProgressTracker,
    start_date: str = "1995-01-01",
    end_date: str | None = None,
    max_tickers: int | None = None,
    wait_on_rate_limit: bool = False,
) -> None:
    """
    Phase 2: Download EOD prices for all tickers (rate-limited, resumable).

    Per-key rotation: each key gets up to REQUESTS_PER_BATCH calls, then the
    next key is used. When all keys are exhausted:
    - Default: stop and save progress (re-run to resume)
    - --wait-on-rate-limit: sleep ~62 min, reset keys, and continue

    :param output_dir (str): Base output directory
    :param tickers_df (pd.DataFrame): Ticker list from Phase 1
    :param key_rotator (KeyRotator): API key rotator
    :param progress (ProgressTracker): Progress tracker
    :param start_date (str): Earliest date to fetch
    :param end_date (str | None): Latest date to fetch (YYYY-MM-DD). None = today.
    :param max_tickers (int | None): Limit downloads (for testing)
    :param wait_on_rate_limit (bool): If True, sleep when all keys are
        exhausted instead of stopping. Default: False (stop and resume).
    """
    raw_dir = os.path.join(output_dir, "tiingo_raw")
    os.makedirs(raw_dir, exist_ok=True)

    tickers = list(tickers_df["ticker"].astype(str))
    if max_tickers is not None:
        tickers = tickers[:max_tickers]

    # Split into: needs full download, needs incremental update, already complete
    to_download: list[str] = []
    to_update: list[str] = []
    already_done = 0

    cutoff_date = end_date if end_date is not None else datetime.now().strftime("%Y-%m-%d")

    for ticker in tickers:
        if progress.is_complete(ticker):
            last_date = progress.get_last_date(ticker)
            if last_date and last_date < cutoff_date:
                to_update.append(ticker)
            else:
                already_done += 1
        else:
            to_download.append(ticker)

    total_work = len(to_download) + len(to_update)
    total_tickers = len(tickers)
    # "Downloaded" = tickers with data (already_done + to_update); "to download" = truly new
    already_downloaded = already_done + len(to_update)
    logger.info(
        "Tickers: %d new to download, %d to update, %d up-to-date | %d/%d downloaded (%.1f%%)",
        len(to_download), len(to_update), already_done,
        already_downloaded, total_tickers, 100.0 * already_downloaded / total_tickers,
    )

    if total_work == 0:
        logger.info("All %d tickers up to date. Nothing to download.", total_tickers)
        return

    n_keys = key_rotator.active_count
    total_per_cycle = REQUESTS_PER_BATCH * n_keys
    logger.info(
        "Rate limit budget: %d keys × %d requests/key = %d tickers per cycle",
        n_keys, REQUESTS_PER_BATCH, total_per_cycle,
    )

    # Process: new downloads first, then updates
    all_work = [(t, "full") for t in to_download] + [(t, "update") for t in to_update]

    n_downloaded = 0  # New tickers downloaded this session
    n_updated = 0     # Existing tickers updated this session
    n_skipped = 0     # Tickers skipped (fetch_start > end_date)
    global_retries_used = 0

    print(f"[phase_prices] Starting loop: {len(all_work)} tickers, end_date={end_date}", flush=True)

    for i, (ticker, mode) in enumerate(all_work):
        if key_rotator.active_count == 0:
            logger.error("All API keys permanently disabled. Stopping.")
            break

        # Get next available key (per-key rotation)
        api_key = key_rotator.next_key()

        if api_key is None:
            # All keys exhausted (rate-limited)
            if wait_on_rate_limit:
                wait = BATCH_SLEEP_SECONDS
                total_downloaded = already_downloaded + n_downloaded
                logger.info(
                    "All %d keys exhausted. Waiting %.0f minutes "
                    "(%d/%d downloaded, %.1f%% | +%d new, +%d updated)...",
                    n_keys, wait / 60, total_downloaded, total_tickers,
                    100.0 * total_downloaded / total_tickers, n_downloaded, n_updated,
                )
                time.sleep(wait)
                key_rotator.reset_rate_limits()
                api_key = key_rotator.next_key()
                if api_key is None:
                    logger.error("No keys available after cooldown. Stopping.")
                    break
            else:
                total_downloaded = already_downloaded + n_downloaded
                logger.info(
                    "All %d keys exhausted (%d/%d downloaded, %.1f%% | +%d new, +%d updated). "
                    "Re-run to resume.",
                    n_keys, total_downloaded, total_tickers,
                    100.0 * total_downloaded / total_tickers, n_downloaded, n_updated,
                )
                break

        # Determine start date for this ticker
        if mode == "update":
            last_date = progress.get_last_date(ticker)
            if last_date:
                fetch_start = (
                    datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
                ).strftime("%Y-%m-%d")
            else:
                fetch_start = start_date
        else:
            fetch_start = start_date

        # Skip API call if fetch window is empty (fetch_start > end_date)
        if end_date is not None and fetch_start > end_date:
            n_skipped += 1
            progress.mark_complete(ticker, cutoff_date)
            if mode == "update":
                n_updated += 1
            else:
                n_downloaded += 1
            if n_skipped % 500 == 0:
                print(f"  Skipped {n_skipped} tickers (already have data past {end_date})", flush=True)
            continue

        # Fetch prices — single attempt (retry only if global budget allows)
        df, status = fetch_ticker_prices(ticker, api_key, start_date=fetch_start, end_date=end_date)

        if status == "auth_failed":
            key_rotator.disable_key(api_key)
            progress.mark_failed(ticker, "auth_failed")
            continue

        if status == "rate_limited":
            key_rotator.mark_rate_limited(api_key)
            retry_key = key_rotator.next_key()
            if retry_key is None:
                if wait_on_rate_limit:
                    wait = BATCH_SLEEP_SECONDS
                    total_downloaded = already_downloaded + n_downloaded
                    logger.info(
                        "All keys rate-limited. Waiting %.0f minutes "
                        "(%d/%d downloaded, %.1f%% | +%d new, +%d updated)...",
                        wait / 60, total_downloaded, total_tickers,
                        100.0 * total_downloaded / total_tickers, n_downloaded, n_updated,
                    )
                    time.sleep(wait)
                    key_rotator.reset_rate_limits()
                    retry_key = key_rotator.next_key()
                if retry_key is None:
                    total_downloaded = already_downloaded + n_downloaded
                    logger.info(
                        "All keys exhausted (%d/%d downloaded, %.1f%% | +%d new, +%d updated). "
                        "Re-run to resume.",
                        total_downloaded, total_tickers,
                        100.0 * total_downloaded / total_tickers, n_downloaded, n_updated,
                    )
                    break
            if retry_key is not None:
                df, status = fetch_ticker_prices(ticker, retry_key, start_date=fetch_start, end_date=end_date)
                if status == "ok":
                    key_rotator.record_use(retry_key)
                else:
                    progress.mark_failed(ticker, f"retry_failed_{status}")
                    continue

        elif status == "error":
            # Retry once if global budget allows, otherwise skip immediately
            if global_retries_used < MAX_GLOBAL_RETRIES:
                global_retries_used += 1
                df, status = fetch_ticker_prices(ticker, api_key, start_date=fetch_start, end_date=end_date)

            if status == "error":
                logger.debug("Skipping %s (error)", ticker)
                progress.mark_failed(ticker, "request_error")
                continue

        if status == "ok":
            key_rotator.record_use(api_key)

        # Process result
        if df is None or df.empty:
            progress.mark_complete(ticker, cutoff_date)
            if mode == "update":
                n_updated += 1
            else:
                n_downloaded += 1
            continue

        # Save/append to per-ticker Parquet
        parquet_path = os.path.join(raw_dir, f"{ticker}.parquet")

        if mode == "update" and os.path.exists(parquet_path):
            existing_df = pd.read_parquet(parquet_path)
            df = pd.concat([existing_df, df], ignore_index=True)
            df = df.drop_duplicates(subset=["date"], keep="last")
            df = df.sort_values("date").reset_index(drop=True)

        df.to_parquet(parquet_path, index=False)

        last_date_str = str(df["date"].max().date())
        progress.mark_complete(ticker, last_date_str)
        if mode == "update":
            n_updated += 1
        else:
            n_downloaded += 1

        if (i + 1) % 10 == 0:
            total_downloaded = already_downloaded + n_downloaded
            msg = (
                f"Progress: {total_downloaded}/{total_tickers} downloaded "
                f"({100.0 * total_downloaded / total_tickers:.1f}%) | "
                f"+{n_downloaded} new, +{n_updated} updated, {n_skipped} skipped | {ticker} "
                f"[{key_rotator.available_count} keys]"
            )
            logger.info(msg)
            print(msg, flush=True)  # Force Jupyter output

    total_downloaded = already_downloaded + n_downloaded
    logger.info(
        "Price download complete: %d/%d downloaded (%.1f%%) | session: +%d new, +%d updated",
        total_downloaded, total_tickers, 100.0 * total_downloaded / total_tickers,
        n_downloaded, n_updated,
    )


def phase_merge(
    output_dir: str,
    permno_mgr: PermnoManager,
    min_price: float = 1.0,
    min_history_days: int = 504,
) -> None:
    """
    Phase 3: Merge per-ticker Parquet files into pipeline-ready format.

    No API calls needed — reads local files only.

    Data quality filters applied:
    - Rows with adj_price < min_price are removed (penny stock filter).
    - Stocks with fewer than min_history_days trading days are dropped entirely
      (ISD: min_listing_days = T = 504 ~ 2 years).

    :param output_dir (str): Base output directory
    :param permno_mgr (PermnoManager): Permno mapping
    :param min_price (float): Minimum adjusted price threshold (default $1.00).
        Rows below this are removed as penny stock data.
    :param min_history_days (int): Minimum number of trading days for a stock to be
        kept in the merged output (default 504 = ~2 years, matching ISD T).
    """
    raw_dir = os.path.join(output_dir, "tiingo_raw")
    meta_dir = os.path.join(output_dir, "tiingo_meta")
    output_path = os.path.join(output_dir, "tiingo_us_equities.parquet")

    if not os.path.isdir(raw_dir):
        logger.error("No raw data directory found at %s", raw_dir)
        return

    # Load exchange info from supported_tickers.csv
    meta_csv = os.path.join(meta_dir, "supported_tickers.csv")
    if os.path.exists(meta_csv):
        meta_df = pd.read_csv(meta_csv)
        exchange_map: dict[str, int] = {}
        for _, row in meta_df.iterrows():
            ticker = str(row["ticker"])
            exchange = str(row.get("exchange", ""))
            exchange_map[ticker] = EXCHANGE_CODE_MAP.get(exchange, 1)
    else:
        exchange_map = {}
        logger.warning("No supported_tickers.csv found; defaulting exchange_code=1")

    # Process each ticker file
    all_frames: list[pd.DataFrame] = []
    parquet_files = [f for f in os.listdir(raw_dir) if f.endswith(".parquet")]
    logger.info("Merging %d ticker files...", len(parquet_files))

    for filename in parquet_files:
        ticker = filename.replace(".parquet", "")
        filepath = os.path.join(raw_dir, filename)

        try:
            df = pd.read_parquet(filepath)
        except Exception as e:
            logger.warning("Failed to read %s: %s", filepath, e)
            continue

        if df.empty:
            continue

        permno = permno_mgr.get_or_create(ticker)

        # Map columns to internal schema
        n_rows = len(df)
        dates = pd.to_datetime(df["date"]).dt.tz_localize(None).values

        # Adjusted close price
        if "adjClose" in df.columns:
            adj_price = df["adjClose"].astype(float).values
        elif "close" in df.columns:
            adj_price = df["close"].astype(float).values
        else:
            logger.warning("Ticker %s has no close/adjClose column", ticker)
            continue

        # Volume
        if "volume" in df.columns:
            vol_numeric = pd.to_numeric(df["volume"], errors="coerce")
            volume = pd.Series(vol_numeric).fillna(0).astype(int).values
        else:
            volume = np.zeros(n_rows, dtype=int)

        # Market cap: estimate from close × volume as ranking proxy
        # Since Tiingo free tier lacks shares outstanding,
        # we use: close × average_volume × 20 as a rough market cap proxy.
        # This preserves relative ranking for universe construction.
        if "close" in df.columns:
            close_arr = np.asarray(df["close"].astype(float))
        else:
            close_arr = np.asarray(adj_price)

        vol_float = np.asarray(volume, dtype=float)
        rolling_avg_vol = np.asarray(pd.Series(vol_float).rolling(window=63, min_periods=1).mean())
        market_cap = close_arr * np.asarray(rolling_avg_vol) * 20.0

        mapped = pd.DataFrame({
            "permno": permno,
            "date": dates,
            "adj_price": adj_price,
            "volume": volume,
            "exchange_code": exchange_map.get(ticker, 1),
            "share_code": 10,
            "market_cap": market_cap,
            "delisting_return": np.nan,
        })

        all_frames.append(mapped)

    if not all_frames:
        logger.error("No data to merge.")
        return

    merged = pd.concat(all_frames, ignore_index=True)
    merged = merged.sort_values(["permno", "date"]).reset_index(drop=True)
    n_raw = len(merged)
    n_tickers_raw = merged["permno"].nunique()

    # --- Data quality filters ---

    # Filter 1: Remove rows with invalid prices (non-positive)
    merged = pd.DataFrame(merged[merged["adj_price"] > 0])

    # Filter 2: Remove penny stock rows (adj_price < min_price)
    if min_price > 0:
        n_before_price = len(merged)
        merged = pd.DataFrame(merged[merged["adj_price"] >= min_price])
        n_penny = n_before_price - len(merged)
        if n_penny > 0:
            logger.info(
                "Penny stock filter (adj_price < $%.2f): removed %d rows",
                min_price, n_penny,
            )

    # Filter 2b: Remove rows with extreme prices (Tiingo API bug)
    # Berkshire-A (~$600K) is the highest legitimate stock price
    MAX_REASONABLE_PRICE = 1_000_000.0
    n_before_extreme = len(merged)
    merged = pd.DataFrame(merged[merged["adj_price"] <= MAX_REASONABLE_PRICE])
    n_extreme = n_before_extreme - len(merged)
    if n_extreme > 0:
        logger.warning(
            "Extreme price filter (adj_price > $%.0f): removed %d rows "
            "(likely Tiingo API split adjustment bug)",
            MAX_REASONABLE_PRICE, n_extreme,
        )

    # Filter 3: Remove stocks with too few trading days
    if min_history_days > 0:
        days_per_stock: pd.Series = merged.groupby("permno")["date"].transform("count")  # type: ignore[assignment]
        thin_mask = days_per_stock < min_history_days
        n_thin_rows = int(thin_mask.sum())
        if n_thin_rows > 0:
            thin_permnos = merged.loc[thin_mask, "permno"].unique()  # type: ignore[union-attr]
            merged = pd.DataFrame(merged[~thin_mask])
            logger.info(
                "Min history filter (< %d days): removed %d stocks (%d rows)",
                min_history_days, len(thin_permnos), n_thin_rows,
            )

    merged = merged.reset_index(drop=True)

    merged.to_parquet(output_path, index=False)
    permno_mgr.save()

    n_tickers = merged["permno"].nunique()
    n_rows = len(merged)
    date_min = merged["date"].min()
    date_max = merged["date"].max()
    logger.info(
        "Merged %d rows for %d tickers (%s to %s) → %s "
        "(from %d raw rows, %d raw tickers)",
        n_rows, n_tickers, date_min.date(), date_max.date(), output_path,
        n_raw, n_tickers_raw,
    )


# ---------------------------------------------------------------------------
# High-level API (for notebook / programmatic use)
# ---------------------------------------------------------------------------


def run_download(
    api_keys: list[str],
    data_dir: str = "data/",
    max_tickers: int | None = None,
    start_date: str = "1995-01-01",
    end_date: str | None = None,
    wait_on_rate_limit: bool = False,
    sp500_first: bool = False,
    sp500_file: str | None = None,
    min_price: float = 1.0,
    min_history_days: int = 504,
    offline_mode: bool | None = None,
) -> bool:
    """
    Run the full Tiingo download pipeline (discovery, prices, merge).

    Convenience wrapper for notebook or programmatic usage.

    Offline mode (Colab / read-only environments): if ``tiingo_raw/`` is absent
    but ``tiingo_us_equities.parquet`` already exists, no download is attempted.
    This is auto-detected when ``offline_mode=None`` (default), or can be
    forced with ``offline_mode=True``.

    :param api_keys (list[str]): Tiingo API keys (ignored in offline mode)
    :param data_dir (str): Output directory
    :param max_tickers (int | None): Limit number of tickers (for testing)
    :param start_date (str): Earliest date to fetch
    :param end_date (str | None): Latest date to fetch (YYYY-MM-DD). None = today.
    :param wait_on_rate_limit (bool): If True, sleep ~62 min when all keys
        are exhausted instead of stopping. Default: False (stop and resume).
    :param sp500_first (bool): If True, download S&P 500 tickers first
        (priority), then remaining US equities.
    :param sp500_file (str | None): Optional local CSV with SP500 tickers
    :param min_price (float): Minimum adj_price (penny stock filter, default $1.00)
    :param min_history_days (int): Minimum trading days per stock (default 504)
    :param offline_mode (bool | None): If True, skip all downloads and use the
        existing merged parquet. If None (default), auto-detect: offline when
        tiingo_raw/ is absent and tiingo_us_equities.parquet exists.

    :return success (bool): True if download completed (or skipped in offline mode)
    """
    raw_dir = os.path.join(data_dir, "tiingo_raw")
    merged_path = os.path.join(data_dir, "tiingo_us_equities.parquet")

    # Auto-detect offline mode: raw dir absent but merged parquet present
    if offline_mode is None:
        offline_mode = not os.path.isdir(raw_dir) and os.path.exists(merged_path)

    if offline_mode:
        if not os.path.exists(merged_path):
            logger.error(
                "Offline mode active but %s not found. "
                "Cannot proceed without data.",
                merged_path,
            )
            return False
        logger.info(
            "Offline mode: tiingo_raw/ not found — using existing %s (no download).",
            merged_path,
        )
        return True

    meta_dir = os.path.join(data_dir, "tiingo_meta")
    os.makedirs(meta_dir, exist_ok=True)

    permno_mgr = PermnoManager(os.path.join(meta_dir, "ticker_to_permno.json"))
    progress = ProgressTracker(os.path.join(meta_dir, "download_progress.json"))
    key_rotator = KeyRotator(api_keys)

    logger.info("Validating %d API key(s)...", len(api_keys))
    n_valid = key_rotator.validate_all()
    logger.info("%d/%d keys valid.", n_valid, len(api_keys))

    if n_valid == 0:
        logger.error("No valid API keys. Cannot download.")
        return False

    # Phase 1: Discovery
    logger.info("=== Phase 1: Ticker Discovery ===")
    tickers_df = phase_discovery(
        data_dir, permno_mgr, sp500_first=sp500_first, sp500_file=sp500_file,
    )
    logger.info("Found %d US equity tickers", len(tickers_df))

    # Phase 2: Price download
    logger.info("=== Phase 2: Price Download (max_tickers=%s) ===", max_tickers)
    phase_prices(
        data_dir, tickers_df, key_rotator, progress,
        start_date=start_date, end_date=end_date,
        max_tickers=max_tickers,
        wait_on_rate_limit=wait_on_rate_limit,
    )

    # Phase 3: Merge
    logger.info("=== Phase 3: Merge to Pipeline Format ===")
    phase_merge(
        data_dir, permno_mgr,
        min_price=min_price,
        min_history_days=min_history_days,
    )

    logger.info("Done! Data saved to %s", data_dir)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    :return args (Namespace): Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Download US equity daily data from Tiingo API.",
    )
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["discovery", "prices", "merge", "all"],
        help="Execution phase (default: all)",
    )
    parser.add_argument(
        "--api-keys", type=str, default=None,
        help="Comma-separated API keys",
    )
    parser.add_argument(
        "--keys-file", type=str, default=None,
        help="Path to file with one API key per line",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/",
        help="Base output directory (default: data/)",
    )
    parser.add_argument(
        "--start-date", type=str, default="1995-01-01",
        help="Earliest date to fetch (default: 1995-01-01)",
    )
    parser.add_argument(
        "--max-tickers", type=int, default=None,
        help="Limit number of tickers to download (for testing)",
    )
    parser.add_argument(
        "--wait-on-rate-limit", action="store_true", default=False,
        help="Sleep ~62 min when all keys are exhausted instead of stopping. "
             "Default: stop and save progress (re-run to resume).",
    )
    parser.add_argument(
        "--sp500-first", action="store_true", default=False,
        help="Download S&P 500 tickers first (priority), then remaining US equities.",
    )
    parser.add_argument(
        "--sp500-file", type=str, default=None,
        help="Local CSV file with SP500 tickers (one per line or 'Symbol' column). "
             "If not provided, fetches from Wikipedia.",
    )
    parser.add_argument(
        "--min-price", type=float, default=1.0,
        help="Minimum adj_price to keep (penny stock filter). Default: $1.00",
    )
    parser.add_argument(
        "--min-history-days", type=int, default=504,
        help="Minimum trading days per stock (ISD T=504 ~2 years). Default: 504",
    )
    parser.add_argument(
        "--offline", action="store_true", default=False,
        help="Skip all downloads and use the existing tiingo_us_equities.parquet. "
             "Auto-detected when tiingo_raw/ is absent and the merged file exists.",
    )
    return parser.parse_args()


def load_api_keys(args: argparse.Namespace) -> list[str]:
    """
    Load API keys from CLI args, file, or environment variable.

    :param args (Namespace): Parsed arguments

    :return keys (list[str]): List of API keys
    """
    keys: list[str] = []

    # From CLI
    if args.api_keys:
        keys.extend(k.strip() for k in args.api_keys.split(",") if k.strip())

    # From file
    if args.keys_file:
        if not os.path.exists(args.keys_file):
            logger.error("Keys file not found: %s", args.keys_file)
        else:
            with open(args.keys_file) as f:
                keys.extend(line.strip() for line in f if line.strip())

    # From environment
    env_keys = os.environ.get("TIINGO_API_KEYS", "") or os.environ.get("TIINGO_API_KEY", "")
    if env_keys:
        keys.extend(k.strip() for k in env_keys.split(",") if k.strip())

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_keys: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique_keys.append(k)

    return unique_keys


def main() -> int:
    """
    Main entry point.

    :return exit_code (int): 0 on success, 1 on failure
    """
    args = parse_args()
    output_dir = args.output_dir
    phase = args.phase

    os.makedirs(output_dir, exist_ok=True)

    # --- Offline mode: skip all downloads, use existing merged parquet ---
    raw_dir = os.path.join(output_dir, "tiingo_raw")
    merged_path = os.path.join(output_dir, "tiingo_us_equities.parquet")

    offline = args.offline or (not os.path.isdir(raw_dir) and os.path.exists(merged_path))
    if offline:
        if not os.path.exists(merged_path):
            logger.error(
                "Offline mode active but %s not found. Cannot proceed.",
                merged_path,
            )
            return 1
        logger.info(
            "Offline mode: tiingo_raw/ not found — using existing %s (no download).",
            merged_path,
        )
        return 0

    meta_dir = os.path.join(output_dir, "tiingo_meta")
    os.makedirs(meta_dir, exist_ok=True)

    # Initialize managers
    permno_mgr = PermnoManager(os.path.join(meta_dir, "ticker_to_permno.json"))
    progress = ProgressTracker(os.path.join(meta_dir, "download_progress.json"))

    # Phases that need API keys
    needs_api = phase in ("prices", "all")

    if needs_api:
        keys = load_api_keys(args)
        if not keys:
            logger.error(
                "No API keys provided. Use --api-keys, --keys-file, "
                "or set TIINGO_API_KEYS environment variable."
            )
            return 1

        key_rotator = KeyRotator(keys)
        logger.info("Validating %d API key(s)...", len(keys))
        n_valid = key_rotator.validate_all()

        if n_valid == 0:
            logger.error("No valid API keys. Aborting.")
            return 1
        logger.info("%d/%d API keys valid.", n_valid, len(keys))
    else:
        key_rotator: KeyRotator | None = None

    # Phase 1: Discovery
    if phase in ("discovery", "all"):
        logger.info("=== Phase 1: Ticker Discovery ===")
        tickers_df = phase_discovery(
            output_dir, permno_mgr,
            sp500_first=args.sp500_first,
            sp500_file=args.sp500_file,
        )
    else:
        # Load existing tickers
        csv_path = os.path.join(meta_dir, "supported_tickers.csv")
        if os.path.exists(csv_path):
            tickers_df = pd.read_csv(csv_path)
        else:
            logger.error(
                "No supported_tickers.csv found. Run --phase discovery first."
            )
            return 1

    # Phase 2: Price download
    if phase in ("prices", "all") and key_rotator is not None:
        logger.info("=== Phase 2: Price Download ===")
        phase_prices(
            output_dir,
            tickers_df,
            key_rotator,
            progress,
            start_date=args.start_date,
            max_tickers=args.max_tickers,
            wait_on_rate_limit=args.wait_on_rate_limit,
        )

    # Phase 3: Merge
    if phase in ("merge", "all"):
        logger.info("=== Phase 3: Merge to Pipeline Format ===")
        phase_merge(
            output_dir, permno_mgr,
            min_price=args.min_price,
            min_history_days=args.min_history_days,
        )

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
