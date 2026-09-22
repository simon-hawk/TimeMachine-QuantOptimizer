"""
Fast vector and loop-accelerated calculation utilities.
Provides accelerated rolling window statistics, EMA, and drawdown sequences.
"""

import math
from typing import List, Tuple

def fast_sma(series: List[float], period: int) -> List[float]:
    """Calculates Simple Moving Average with O(N) single-pass rolling window."""
    n = len(series)
    out = [float("nan")] * n
    if period <= 0 or n < period:
        return out
    
    window_sum = sum(series[:period])
    out[period - 1] = window_sum / period
    for i in range(period, n):
        window_sum += series[i] - series[i - period]
        out[i] = window_sum / period
    return out

def fast_ema(series: List[float], period: int) -> List[float]:
    """Calculates Exponential Moving Average with O(N) recursion."""
    n = len(series)
    out = [float("nan")] * n
    if period <= 0 or n < period:
        return out
    
    alpha = 2.0 / (period + 1.0)
    sma_seed = sum(series[:period]) / period
    out[period - 1] = sma_seed
    prev = sma_seed
    for i in range(period, n):
        curr = alpha * series[i] + (1.0 - alpha) * prev
        out[i] = curr
        prev = curr
    return out

def fast_drawdowns(equity_curve: List[float]) -> Tuple[float, List[float]]:
    """
    Computes maximum percentage drawdown and point-in-time drawdown series.
    Returns (max_drawdown_pct, drawdown_series).
    """
    if not equity_curve:
        return 0.0, []
    
    peak = equity_curve[0]
    dd_series = []
    max_dd = 0.0

    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (val - peak) / peak if peak > 0 else 0.0
        dd_series.append(dd)
        if dd < max_dd:
            max_dd = dd

    return abs(max_dd), dd_series
