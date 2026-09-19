"""
Partitions subpackage: Sample separation, Walk-Forward Cross Validation, and Crisis Regime isolation.
"""

from .crisis_regimes import CrisisRegimeLibrary, MarketRegimeWindow
from .splitters import DataSplitter, SplitResult
from .walk_forward import WalkForwardGenerator, WalkForwardFold

__all__ = [
    "CrisisRegimeLibrary",
    "MarketRegimeWindow",
    "DataSplitter",
    "SplitResult",
    "WalkForwardGenerator",
    "WalkForwardFold"
]
