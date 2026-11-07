"""
Resolve Tiingo API keys without ever hardcoding them in tracked files.

Sources, first non-empty one wins:
    1. TIINGO_API_KEYS (or TIINGO_API_KEY) environment variable, comma-separated
    2. Colab Secrets entry "TIINGO_API_KEYS" (key icon in the Colab sidebar)
    3. A gitignored file, one key per line (default: keys.txt)

Keys are only needed to refresh the dataset: with data/tiingo_us_equities.parquet
present, the downloader runs offline and an empty list is fine.
"""

import logging
import os

logger = logging.getLogger(__name__)

ENV_VAR = "TIINGO_API_KEYS"


def _split_keys(raw: str, sep: str) -> list[str]:
    """
    Split a raw key string, strip whitespace, drop empties and duplicates.

    :param raw (str): Raw string containing keys
    :param sep (str): Separator ("," or newline)

    :return keys (list[str]): Unique keys in original order
    """
    keys = [k.strip() for k in raw.split(sep)]
    return list(dict.fromkeys(k for k in keys if k))


def _keys_from_colab() -> str:
    """
    Read the TIINGO_API_KEYS Colab secret, if running on Colab and it is set.

    :return raw (str): Raw secret value, or "" when unavailable
    """
    try:
        # Only importable inside a Colab runtime
        from google.colab import userdata  # type: ignore[import-not-found]  # pylint: disable=import-outside-toplevel
    except ImportError:
        return ""
    try:
        return userdata.get(ENV_VAR) or ""
    # SecretNotFoundError / NotebookAccessError are not importable outside Colab
    except Exception:  # pylint: disable=broad-exception-caught
        return ""


def load_tiingo_keys(keys_file: str = "keys.txt") -> list[str]:
    """
    Load Tiingo API keys from env var, Colab Secrets, then a gitignored file.

    :param keys_file (str): Path to a file with one key per line

    :return keys (list[str]): Unique API keys (empty if none configured)
    """
    raw_env = os.environ.get(ENV_VAR, "") or os.environ.get("TIINGO_API_KEY", "")
    keys = _split_keys(raw_env, ",")
    if keys:
        logger.info("Tiingo keys: %d from environment variable", len(keys))
        return keys

    keys = _split_keys(_keys_from_colab(), ",")
    if keys:
        logger.info("Tiingo keys: %d from Colab Secrets", len(keys))
        return keys

    if os.path.exists(keys_file):
        with open(keys_file, encoding="utf-8") as f:
            keys = _split_keys(f.read(), "\n")
        if keys:
            logger.info("Tiingo keys: %d from %s", len(keys), keys_file)
            return keys

    logger.info("No Tiingo API keys configured (offline mode only)")
    return []
