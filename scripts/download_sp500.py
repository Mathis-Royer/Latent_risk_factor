#!/usr/bin/env python
"""
Download S&P 500 index data for the SP500TotalReturn benchmark.

NOTE: This script is optional — the SP500 benchmark auto-downloads
on first use if the cache doesn't exist. Use this for:
  - Pre-downloading before offline runs
  - Forcing a refresh of stale data
  - Verifying data availability

Usage:
    python scripts/download_sp500.py
    python scripts/download_sp500.py --output-dir data/ --start-date 1995-01-01 --force
"""

import argparse
import logging
import sys

from src.benchmarks.sp500_index import download_sp500_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """
    Download S&P 500 index data to disk.

    :return exit_code (int): 0 on success, 1 on failure
    """
    parser = argparse.ArgumentParser(
        description="Download S&P 500 index data from Yahoo Finance"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/",
        help="Output directory (default: data/)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="1995-01-01",
        help="Start date YYYY-MM-DD (default: 1995-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cache is fresh",
    )
    args = parser.parse_args()

    logger.info("Downloading S&P 500 index data...")
    df = download_sp500_data(
        data_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        force=args.force,
    )
    if df is None:
        logger.error("Download failed")
        return 1

    logger.info(
        "Done! Saved %d days to %s/sp500_index.parquet",
        len(df),
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
