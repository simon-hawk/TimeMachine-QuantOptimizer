"""
Unit tests for Data Quality and Accuracy Validator.
"""

from datetime import datetime, timedelta
from mnq_backtester.data.validator import DataValidator

def test_clean_bars_pass_validation():
    validator = DataValidator(tick_size=0.25)
    now = datetime(2026, 1, 5, 9, 30)
    bars = []
    for i in range(100):
        t = now + timedelta(minutes=i)
        bars.append({
            "timestamp": t,
            "open": 18000.0,
            "high": 18010.50,
            "low": 17995.25,
            "close": 18005.75,
            "volume": 1000
        })

    report = validator.validate_bars(bars)
    assert report.is_valid is True
    assert report.total_bars == 100
    assert report.passed_bars == 100
    assert report.failed_bars == 0
    assert report.ohlc_invariant_errors == 0
    assert report.tick_quantization_errors == 0

def test_ohlc_invariant_failure():
    validator = DataValidator(tick_size=0.25)
    # High is less than Open (invalid)
    corrupt_bars = [{
        "timestamp": datetime(2026, 1, 5, 9, 30),
        "open": 18050.0,
        "high": 18020.0, # Corrupted: High < Open
        "low": 17990.0,
        "close": 18010.0,
        "volume": 500
    }]
    report = validator.validate_bars(corrupt_bars)
    assert report.is_valid is False
    assert report.ohlc_invariant_errors > 0

def test_tick_quantization_check():
    validator = DataValidator(tick_size=0.25)
    # Non-quantized price (e.g. 18000.133 not multiple of 0.25)
    unquantized_bars = [{
        "timestamp": datetime(2026, 1, 5, 9, 30),
        "open": 18000.133,
        "high": 18010.0,
        "low": 17995.0,
        "close": 18005.0,
        "volume": 500
    }]
    report = validator.validate_bars(unquantized_bars)
    assert report.tick_quantization_errors > 0

    # Test auto-clean and quantization
    cleaned = validator.clean_and_quantize(unquantized_bars)
    assert cleaned[0]["open"] == 18000.25
    clean_report = validator.validate_bars(cleaned)
    assert clean_report.tick_quantization_errors == 0
