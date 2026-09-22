"""
Unit tests for Dynamic PineScript Parser & Technical Indicator Engine.
"""

import math
from datetime import datetime, timedelta
from mnq_backtester.pinescript.indicators import PineTA
from mnq_backtester.pinescript.parser import PineScriptStrategy, PineScriptRunner
from mnq_backtester.strategies.library import PINE_STRATEGY_SUITE

def test_pine_ta_indicators():
    series = [100.0, 102.0, 101.0, 105.0, 108.0, 107.0, 110.0, 115.0, 112.0, 118.0]
    sma3 = PineTA.sma(series, 3)
    assert abs(sma3[2] - 101.0) < 1e-4 # (100+102+101)/3 = 101.0

    ema3 = PineTA.ema(series, 3)
    assert not any(math.isnan(x) for x in ema3[2:])

    rsi = PineTA.rsi(series, 5)
    assert len(rsi) == len(series)

    hi = PineTA.highest(series, 3)
    assert hi[-1] == 118.0

    lo = PineTA.lowest(series, 3)
    assert lo[-1] == 112.0

def test_pinescript_strategy_parser_and_run():
    # Construct synthetic bars
    now = datetime(2026, 1, 5, 9, 30)
    bars = []
    price = 18000.0
    for i in range(120):
        t = now + timedelta(minutes=i)
        bars.append({
            "timestamp": t,
            "open": price,
            "high": price + 5.0,
            "low": price - 5.0,
            "close": price + (2.0 if i % 2 == 0 else -1.0),
            "volume": 500
        })
        price = bars[-1]["close"]

    # Test running first available generic StockSharp PineScript
    strat_key = next(iter(PINE_STRATEGY_SUITE))
    strat_code = PINE_STRATEGY_SUITE[strat_key]["code"]
    res, strat = PineScriptRunner.run_pine_code(strat_code, bars)
    assert strat.name is not None
    assert res.total_bars_processed > 0
