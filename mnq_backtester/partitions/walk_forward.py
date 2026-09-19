"""
Walk-Forward Cross-Validation (WFCV) Generator.
Generates Rolling and Anchored Walk-Forward Folds with purge and embargo buffers.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Generator, Optional
from datetime import datetime

@dataclass
class WalkForwardFold:
    fold_index: int
    train_bars: List[Dict[str, Any]]
    test_bars: List[Dict[str, Any]]
    train_start: Optional[datetime] = None
    train_end: Optional[datetime] = None
    test_start: Optional[datetime] = None
    test_end: Optional[datetime] = None

class WalkForwardGenerator:
    """
    Generates realistic walk-forward cross-validation splits across trading days.
    """

    @staticmethod
    def _extract_time(bar: Dict[str, Any]) -> Optional[datetime]:
        raw = bar.get("timestamp") or bar.get("time_ny") or bar.get("date")
        if isinstance(raw, datetime):
            return raw
        elif isinstance(raw, str):
            try:
                return datetime.fromisoformat(raw)
            except Exception:
                return None
        return None

    @classmethod
    def generate_rolling_folds(
        cls,
        bars: List[Dict[str, Any]],
        train_bars_count: int,
        test_bars_count: int,
        step_bars_count: int,
        embargo_bars: int = 30
    ) -> List[WalkForwardFold]:
        """
        Generates rolling training windows of fixed length:
        [Train Window] -> [Embargo] -> [Test Window], shifting forward by step_bars_count.
        """
        folds = []
        total_bars = len(bars)
        fold_idx = 1
        start_idx = 0

        while True:
            train_end = start_idx + train_bars_count
            test_start = train_end + embargo_bars
            test_end = test_start + test_bars_count

            if test_end > total_bars:
                break

            train_slice = bars[start_idx:train_end]
            test_slice = bars[test_start:test_end]

            folds.append(WalkForwardFold(
                fold_index=fold_idx,
                train_bars=train_slice,
                test_bars=test_slice,
                train_start=cls._extract_time(train_slice[0]),
                train_end=cls._extract_time(train_slice[-1]),
                test_start=cls._extract_time(test_slice[0]),
                test_end=cls._extract_time(test_slice[-1])
            ))

            fold_idx += 1
            start_idx += step_bars_count

        return folds

    @classmethod
    def generate_anchored_folds(
        cls,
        bars: List[Dict[str, Any]],
        initial_train_bars: int,
        test_bars_count: int,
        step_bars_count: int,
        embargo_bars: int = 30
    ) -> List[WalkForwardFold]:
        """
        Generates expanding (anchored) training windows:
        [0 ... Expanding Train] -> [Embargo] -> [Test Window].
        """
        folds = []
        total_bars = len(bars)
        fold_idx = 1
        current_train_end = initial_train_bars

        while True:
            test_start = current_train_end + embargo_bars
            test_end = test_start + test_bars_count

            if test_end > total_bars:
                break

            train_slice = bars[0:current_train_end]
            test_slice = bars[test_start:test_end]

            folds.append(WalkForwardFold(
                fold_index=fold_idx,
                train_bars=train_slice,
                test_bars=test_slice,
                train_start=cls._extract_time(train_slice[0]),
                train_end=cls._extract_time(train_slice[-1]),
                test_start=cls._extract_time(test_slice[0]),
                test_end=cls._extract_time(test_slice[-1])
            ))

            fold_idx += 1
            current_train_end += step_bars_count

        return folds
