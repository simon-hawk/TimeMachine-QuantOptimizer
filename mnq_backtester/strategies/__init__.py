"""
Strategies subpackage: Base strategy interface and reference intraday trading models for MNQ.
"""

from .base import BaseStrategy
from .example_orb import OpeningRangeBreakoutStrategy
from .example_fvg import FairValueGapStrategy
from .example_boswaves import BOSWavesGravityStrategy

__all__ = [
    "BaseStrategy",
    "OpeningRangeBreakoutStrategy",
    "FairValueGapStrategy",
    "BOSWavesGravityStrategy"
]
