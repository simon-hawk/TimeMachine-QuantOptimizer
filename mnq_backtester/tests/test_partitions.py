"""
Unit tests for Sample Separation, Walk-Forward splits, and Crisis Regimes.
"""

from datetime import datetime, date, timedelta
from mnq_backtester.partitions.splitters import DataSplitter
from mnq_backtester.partitions.walk_forward import WalkForwardGenerator
from mnq_backtester.partitions.crisis_regimes import CrisisRegimeLibrary

def test_train_test_split_embargo():
    now = datetime(2026, 1, 1, 9, 30)
    bars = [{"timestamp": now + timedelta(minutes=i), "close": 18000 + i} for i in range(1000)]

    split = DataSplitter.train_test_split(bars, train_ratio=0.70, embargo_bars=50)
    assert len(split.train_bars) == 700
    assert len(split.test_bars) == 250 # 1000 - 700 - 50 embargo = 250
    assert split.embargo_bars == 50

def test_walk_forward_rolling_folds():
    now = datetime(2026, 1, 1, 9, 30)
    bars = [{"timestamp": now + timedelta(minutes=i), "close": 18000 + i} for i in range(500)]

    folds = WalkForwardGenerator.generate_rolling_folds(
        bars,
        train_bars_count=100,
        test_bars_count=50,
        step_bars_count=50,
        embargo_bars=10
    )
    assert len(folds) >= 6
    assert len(folds[0].train_bars) == 100
    assert len(folds[0].test_bars) == 50

def test_crisis_regime_slicing():
    # Construct bars spanning 2020 COVID window and 2021
    bars = [
        {"timestamp": datetime(2020, 2, 20, 10, 0), "close": 9000.0},
        {"timestamp": datetime(2020, 3, 15, 10, 0), "close": 7500.0},
        {"timestamp": datetime(2021, 6, 1, 10, 0), "close": 14000.0},
    ]

    covid_slice = CrisisRegimeLibrary.slice_by_regime(bars, "COVID_CRASH_2020")
    assert len(covid_slice) == 2
    assert covid_slice[0]["close"] == 9000.0
    assert covid_slice[1]["close"] == 7500.0
