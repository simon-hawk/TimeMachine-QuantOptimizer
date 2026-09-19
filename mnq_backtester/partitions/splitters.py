"""
Data Splitters for In-Sample (IS), Out-of-Sample (OOS), and Train/Val/Test Holdout Partitioning.
Includes purge/embargo buffers to prevent indicator lookahead leakage.
"""

from dataclasses import dataclass
from datetime import datetime, date
from typing import List, Dict, Any, Optional

@dataclass
class SplitResult:
    train_bars: List[Dict[str, Any]]
    val_bars: List[Dict[str, Any]]
    test_bars: List[Dict[str, Any]]
    train_start: Optional[datetime] = None
    train_end: Optional[datetime] = None
    val_start: Optional[datetime] = None
    val_end: Optional[datetime] = None
    test_start: Optional[datetime] = None
    test_end: Optional[datetime] = None
    embargo_bars: int = 0

    @property
    def total_bars(self) -> int:
        return len(self.train_bars) + len(self.val_bars) + len(self.test_bars)

    def summary(self) -> str:
        t_len = len(self.train_bars)
        v_len = len(self.val_bars)
        ts_len = len(self.test_bars)
        tot = self.total_bars
        return (
            f"=== Partition Split Summary ===\n"
            f"Train (In-Sample): {t_len:,} bars ({t_len/tot*100:.1f}%) | {self.train_start} to {self.train_end}\n"
            f"Val   (Validation): {v_len:,} bars ({v_len/tot*100:.1f}%) | {self.val_start} to {self.val_end}\n"
            f"Test  (Out-of-Sample): {ts_len:,} bars ({ts_len/tot*100:.1f}%) | {self.test_start} to {self.test_end}\n"
            f"Embargo Buffer: {self.embargo_bars} bars"
        )


class DataSplitter:
    """
    Separates time series into clean, leak-free In-Sample and Out-of-Sample partitions.
    """

    @staticmethod
    def _extract_time(bar: Dict[str, Any]) -> Optional[datetime]:
        raw = bar.get("timestamp") or bar.get("time_ny") or bar.get("date")
        if isinstance(raw, datetime):
            return raw
        elif isinstance(raw, date):
            return datetime.combine(raw, datetime.min.time())
        elif isinstance(raw, str):
            try:
                return datetime.fromisoformat(raw)
            except Exception:
                return None
        return None

    @classmethod
    def train_test_split(
        cls,
        bars: List[Dict[str, Any]],
        train_ratio: float = 0.70,
        embargo_bars: int = 50
    ) -> SplitResult:
        """
        Splits data into In-Sample (Train) and Out-of-Sample (Test) with an embargo gap.
        """
        if not bars:
            return SplitResult([], [], [])

        n = len(bars)
        train_end_idx = int(n * train_ratio)
        test_start_idx = min(n, train_end_idx + embargo_bars)

        train_bars = bars[:train_end_idx]
        test_bars = bars[test_start_idx:]

        t_start = cls._extract_time(train_bars[0]) if train_bars else None
        t_end = cls._extract_time(train_bars[-1]) if train_bars else None
        ts_start = cls._extract_time(test_bars[0]) if test_bars else None
        ts_end = cls._extract_time(test_bars[-1]) if test_bars else None

        return SplitResult(
            train_bars=train_bars,
            val_bars=[],
            test_bars=test_bars,
            train_start=t_start,
            train_end=t_end,
            test_start=ts_start,
            test_end=ts_end,
            embargo_bars=embargo_bars
        )

    @classmethod
    def train_val_test_split(
        cls,
        bars: List[Dict[str, Any]],
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        embargo_bars: int = 50
    ) -> SplitResult:
        """
        Splits data into Train (60%), Validation (20%), and Test (20%) with embargo buffers.
        """
        if not bars:
            return SplitResult([], [], [])

        n = len(bars)
        train_end_idx = int(n * train_ratio)
        val_start_idx = min(n, train_end_idx + embargo_bars)
        val_end_idx = min(n, int(n * (train_ratio + val_ratio)))
        test_start_idx = min(n, val_end_idx + embargo_bars)

        train_bars = bars[:train_end_idx]
        val_bars = bars[val_start_idx:val_end_idx]
        test_bars = bars[test_start_idx:]

        return SplitResult(
            train_bars=train_bars,
            val_bars=val_bars,
            test_bars=test_bars,
            train_start=cls._extract_time(train_bars[0]) if train_bars else None,
            train_end=cls._extract_time(train_bars[-1]) if train_bars else None,
            val_start=cls._extract_time(val_bars[0]) if val_bars else None,
            val_end=cls._extract_time(val_bars[-1]) if val_bars else None,
            test_start=cls._extract_time(test_bars[0]) if test_bars else None,
            test_end=cls._extract_time(test_bars[-1]) if test_bars else None,
            embargo_bars=embargo_bars
        )

    @classmethod
    def date_boundary_split(
        cls,
        bars: List[Dict[str, Any]],
        split_date: date,
        embargo_bars: int = 50
    ) -> SplitResult:
        """
        Splits data based on a concrete cutoff date (e.g. In-Sample before 2024-01-01, OOS after).
        """
        train_bars = []
        test_candidate = []

        for b in bars:
            t = cls._extract_time(b)
            if t and t.date() < split_date:
                train_bars.append(b)
            else:
                test_candidate.append(b)

        # Apply embargo to test set start
        test_bars = test_candidate[embargo_bars:] if len(test_candidate) > embargo_bars else []

        return SplitResult(
            train_bars=train_bars,
            val_bars=[],
            test_bars=test_bars,
            train_start=cls._extract_time(train_bars[0]) if train_bars else None,
            train_end=cls._extract_time(train_bars[-1]) if train_bars else None,
            test_start=cls._extract_time(test_bars[0]) if test_bars else None,
            test_end=cls._extract_time(test_bars[-1]) if test_bars else None,
            embargo_bars=embargo_bars
        )
