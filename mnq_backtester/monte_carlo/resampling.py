"""
Monte Carlo Resampling Methods for Trade Series.
Supports: Trade Order Shuffling (Permutation), IID Bootstrap, Moving Block Bootstrap,
and Politis & Romano Stationary Bootstrap.
"""

from enum import Enum
from typing import List
import random
import numpy as np

class ResamplingMethod(Enum):
    SHUFFLE = "SHUFFLE"                       # Permutation without replacement (sequence risk)
    IID_BOOTSTRAP = "IID_BOOTSTRAP"           # Standard bootstrap with replacement
    BLOCK_BOOTSTRAP = "BLOCK_BOOTSTRAP"       # Moving Block bootstrap (fixed block length)
    STATIONARY_BOOTSTRAP = "STATIONARY"       # Politis-Romano Stationary Bootstrap (geometric block length)

class TradeResampler:
    """
    Applies rigorous stochastic resampling techniques to realized trade series.
    """

    @staticmethod
    def shuffle(pnls: List[float], seed: int = 42) -> List[float]:
        """Shuffles trade order without replacement to isolate path-dependency and sequence risk."""
        rng = random.Random(seed)
        shuffled = list(pnls)
        rng.shuffle(shuffled)
        return shuffled

    @staticmethod
    def iid_bootstrap(pnls: List[float], n_samples: int, seed: int = 42) -> List[float]:
        """Draws N trades with replacement."""
        rng = np.random.default_rng(seed)
        p_arr = np.array(pnls)
        indices = rng.integers(0, len(pnls), size=n_samples)
        return list(p_arr[indices])

    @staticmethod
    def moving_block_bootstrap(pnls: List[float], n_samples: int, block_size: int = 5, seed: int = 42) -> List[float]:
        """
        Moving Block Bootstrap: resamples continuous blocks of trades to preserve volatility clustering.
        """
        if len(pnls) <= block_size:
            return TradeResampler.iid_bootstrap(pnls, n_samples, seed)

        rng = np.random.default_rng(seed)
        n = len(pnls)
        resampled = []
        max_start = n - block_size

        while len(resampled) < n_samples:
            start = rng.integers(0, max_start + 1)
            block = pnls[start:start + block_size]
            resampled.extend(block)

        return resampled[:n_samples]

    @staticmethod
    def stationary_bootstrap(pnls: List[float], n_samples: int, avg_block_size: float = 5.0, seed: int = 42) -> List[float]:
        """
        Politis & Romano (1994) Stationary Bootstrap:
        Block lengths follow a Geometric distribution with parameter p = 1 / avg_block_size.
        Guarantees stationarity of the resampled sequence.
        """
        if not pnls:
            return []

        rng = np.random.default_rng(seed)
        n = len(pnls)
        p_geom = 1.0 / max(1.0, avg_block_size)

        resampled = []
        idx = rng.integers(0, n)

        while len(resampled) < n_samples:
            resampled.append(pnls[idx])
            # With probability p, jump to a new random start; otherwise advance to next item (circular)
            if rng.random() < p_geom:
                idx = rng.integers(0, n)
            else:
                idx = (idx + 1) % n

        return resampled
