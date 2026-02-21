"""
Curriculum batch sampler for VAE training.

Phases 1-2 (λ_co > 0): SYNCHRONOUS + STRATIFIED batching.
Phase 3 (λ_co = 0): Standard RANDOM shuffling.

Reference: ISD Section MOD-005 — Sub-task 1.
"""

from collections.abc import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler


class CurriculumBatchSampler(Sampler[list[int]]):
    """
    Batch sampler that switches between synchronous+stratified (co-movement
    phases) and random shuffling (free refinement phase).

    Phases 1-2 (λ_co > 0):
      - Select a random time block (δ_sync = 21 days)
      - Sample B/S windows per stratum (pre-clustered stocks)
      - Guarantees temporal synchronization for co-movement loss

    Phase 3 (λ_co = 0):
      - Standard random shuffling across all windows

    Attributes:
        n_windows: int — total number of windows
        batch_size: int — windows per batch
        window_metadata: pd.DataFrame — stock_id, start_date, end_date per window
        strata: np.ndarray — stratum assignment per window (n_windows,)
        n_strata: int — number of strata
        delta_sync: int — max date gap for synchronization (days)
        synchronous: bool — True for phases 1-2, False for phase 3
    """

    def __init__(
        self,
        n_windows: int,
        batch_size: int,
        window_metadata: pd.DataFrame | None = None,
        strata: np.ndarray | None = None,
        n_strata: int = 15,
        delta_sync: int = 21,
        synchronous: bool = True,
        seed: int = 42,
    ) -> None:
        """
        :param n_windows (int): Total number of windows
        :param batch_size (int): Windows per batch
        :param window_metadata (pd.DataFrame | None): Metadata with end_date
        :param strata (np.ndarray | None): Stratum assignment per window
        :param n_strata (int): Number of strata for stratified sampling
        :param delta_sync (int): Maximum date gap for synchronization
        :param synchronous (bool): If True, use synchronous+stratified batching
        :param seed (int): Random seed
        """
        self.n_windows = n_windows
        self.batch_size = batch_size
        self.window_metadata = window_metadata
        self.n_strata = n_strata
        self.delta_sync = delta_sync
        self.synchronous = synchronous
        self.rng = np.random.RandomState(seed)

        # Build stratum → window index mapping
        if strata is not None:
            self.strata = strata
        else:
            # Default: assign strata randomly if not provided
            self.strata = self.rng.randint(0, n_strata, size=n_windows)

        # Build time-block → window index mapping for synchronous batching
        self._build_time_blocks()

    def _build_time_blocks(self) -> None:
        """Build mapping from time blocks to window indices."""
        self.time_blocks: dict[int, list[int]] = {}
        if self.window_metadata is not None and "end_date" in self.window_metadata.columns:
            # Group by end_date quantized to delta_sync blocks
            dates = pd.to_datetime(self.window_metadata["end_date"])
            min_date = dates.min()
            # Quantize to blocks of delta_sync days
            block_ids = ((dates - min_date).dt.days // self.delta_sync).values
            for idx, block_id in enumerate(block_ids):
                block_key = int(block_id)
                if block_key not in self.time_blocks:
                    self.time_blocks[block_key] = []
                self.time_blocks[block_key].append(idx)
        else:
            # No metadata: single block with all windows
            self.time_blocks[0] = list(range(self.n_windows))

    def set_synchronous(self, synchronous: bool) -> None:
        """
        Switch batching mode. Called when curriculum transitions.

        :param synchronous (bool): True for phases 1-2, False for phase 3
        """
        self.synchronous = synchronous

    def __iter__(self) -> Iterator[list[int]]:
        """Generate batches for one epoch."""
        if self.synchronous:
            yield from self._synchronous_batches()
        else:
            yield from self._random_batches()

    def _synchronous_batches(self) -> list[list[int]]:
        """
        Synchronous + stratified batching.

        For each batch:
        1. Pick a random time block
        2. Within that block, sample B/S windows per stratum
        """
        batches: list[list[int]] = []
        block_keys = list(self.time_blocks.keys())

        # Number of batches = ceil(n_windows / batch_size)
        n_batches = max(1, self.n_windows // self.batch_size)

        per_stratum = max(1, self.batch_size // self.n_strata)

        for _ in range(n_batches):
            # Pick a random time block
            block_key = self.rng.choice(block_keys)
            block_indices = np.array(self.time_blocks[block_key])

            if len(block_indices) == 0:
                continue

            # Stratify within the block
            block_strata = self.strata[block_indices]
            batch_indices: list[int] = []

            for s in range(self.n_strata):
                stratum_mask = block_strata == s
                stratum_indices = block_indices[stratum_mask]
                if len(stratum_indices) == 0:
                    continue
                n_sample = min(per_stratum, len(stratum_indices))
                sampled = self.rng.choice(stratum_indices, size=n_sample, replace=False)
                batch_indices.extend(sampled.tolist())

            # If not enough from strata, pad from the block
            if len(batch_indices) < self.batch_size and len(block_indices) > len(batch_indices):
                remaining = set(block_indices.tolist()) - set(batch_indices)
                n_pad = min(self.batch_size - len(batch_indices), len(remaining))
                padded = self.rng.choice(list(remaining), size=n_pad, replace=False)
                batch_indices.extend(padded.tolist())

            if batch_indices:
                final_batch = batch_indices[:self.batch_size]
                assert all(idx < self.n_windows for idx in final_batch), (
                    f"Batch index >= n_windows ({self.n_windows})"
                )
                batches.append(final_batch)

        return batches

    def _random_batches(self) -> list[list[int]]:
        """Standard random shuffling — all windows equally likely."""
        indices = self.rng.permutation(self.n_windows).tolist()
        batches: list[list[int]] = []
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start:start + self.batch_size]
            if batch:
                batches.append(batch)
        return batches

    def __len__(self) -> int:
        """Approximate number of batches per epoch."""
        return max(1, self.n_windows // self.batch_size)


def compute_strata(
    returns: pd.DataFrame,
    stock_ids: list[str],
    n_strata: int = 15,
    lookback: int = 63,
    seed: int = 42,
) -> np.ndarray:
    """
    Pre-cluster stocks into S strata using k-means on trailing returns.

    :param returns (pd.DataFrame): Returns data (dates × stocks)
    :param stock_ids (list[str]): Stock identifiers
    :param n_strata (int): Number of strata (10-20)
    :param lookback (int): Trailing days for clustering (63)
    :param seed (int): Random seed

    :return strata (np.ndarray): Stratum assignment per stock (n_stocks,)
    """
    from sklearn.cluster import KMeans

    # Use last `lookback` days of returns
    available_cols = [s for s in stock_ids if s in returns.columns]
    if len(available_cols) == 0:
        return np.zeros(len(stock_ids), dtype=int)

    tail_returns = returns[available_cols].iloc[-lookback:]
    # Fill NaN with 0 for clustering
    features = tail_returns.fillna(0).T.values  # (n_stocks, lookback)

    # Adjust n_strata if fewer stocks
    effective_strata = min(n_strata, len(available_cols))

    kmeans = KMeans(
        n_clusters=effective_strata,
        random_state=seed,
        n_init="auto",
    )
    labels = kmeans.fit_predict(features)

    # Map back to full stock_ids list (missing stocks get stratum 0)
    strata = np.zeros(len(stock_ids), dtype=int)
    for i, sid in enumerate(stock_ids):
        if sid in available_cols:
            idx = available_cols.index(sid)
            strata[i] = labels[idx]

    return strata
