"""
PineScript Technical Indicator Library (ta.* equivalent in Python).
Implements exact TradingView calculations for SMA, EMA, RSI, ATR, MACD, Highest/Lowest,
Crossover/Crossunder, Supertrend, Bollinger Bands, and VWAP.
"""

import math
from typing import List, Tuple, Union, Optional, Any
import numpy as np

class PineTA:
    @staticmethod
    def sma(series: List[float], length: int) -> List[float]:
        """Simple Moving Average."""
        if length <= 0 or not series:
            return [float('nan')] * len(series)
        result = [float('nan')] * len(series)
        for i in range(length - 1, len(series)):
            result[i] = sum(series[i - length + 1:i + 1]) / float(length)
        return result

    @staticmethod
    def ema(series: List[float], length: int) -> List[float]:
        """Exponential Moving Average (PineScript alpha = 2 / (length + 1))."""
        if length <= 0 or not series:
            return [float('nan')] * len(series)
        result = [float('nan')] * len(series)
        alpha = 2.0 / (length + 1.0)
        
        # First valid is SMA
        if len(series) >= length:
            init_sma = sum(series[:length]) / float(length)
            result[length - 1] = init_sma
            curr = init_sma
            for i in range(length, len(series)):
                curr = alpha * series[i] + (1.0 - alpha) * curr
                result[i] = curr
        return result

    @staticmethod
    def tr(high: List[float], low: List[float], close: List[float]) -> List[float]:
        """True Range."""
        n = len(high)
        if n == 0:
            return []
        result = [high[0] - low[0]]
        for i in range(1, n):
            h_l = high[i] - low[i]
            h_pc = abs(high[i] - close[i - 1])
            l_pc = abs(low[i] - close[i - 1])
            result.append(max(h_l, h_pc, l_pc))
        return result

    @staticmethod
    def atr(high: List[float], low: List[float], close: List[float], length: int = 14) -> List[float]:
        """Average True Range (RMA smoothed)."""
        tr_series = PineTA.tr(high, low, close)
        return PineTA.rma(tr_series, length)

    @staticmethod
    def rma(series: List[float], length: int) -> List[float]:
        """Running Moving Average (Wilder's smoothing: alpha = 1 / length)."""
        if length <= 0 or not series:
            return [float('nan')] * len(series)
        result = [float('nan')] * len(series)
        alpha = 1.0 / float(length)
        
        if len(series) >= length:
            init_sma = sum(series[:length]) / float(length)
            result[length - 1] = init_sma
            curr = init_sma
            for i in range(length, len(series)):
                curr = alpha * series[i] + (1.0 - alpha) * curr
                result[i] = curr
        return result

    @staticmethod
    def rsi(series: List[float], length: int = 14) -> List[float]:
        """Relative Strength Index."""
        if len(series) <= length:
            return [50.0] * len(series)
        
        gains = [0.0] * len(series)
        losses = [0.0] * len(series)
        for i in range(1, len(series)):
            diff = series[i] - series[i - 1]
            if diff > 0:
                gains[i] = diff
            else:
                losses[i] = abs(diff)

        avg_gain = PineTA.rma(gains[1:], length)
        avg_loss = PineTA.rma(losses[1:], length)
        
        result = [float('nan')] * len(series)
        for i in range(len(avg_gain)):
            idx = i + 1
            ag = avg_gain[i]
            al = avg_loss[i]
            if not math.isnan(ag) and not math.isnan(al):
                if al == 0:
                    result[idx] = 100.0
                else:
                    rs = ag / al
                    result[idx] = 100.0 - (100.0 / (1.0 + rs))
        return result

    @staticmethod
    def highest(series: List[float], length: int) -> List[float]:
        """Highest value over lookback length."""
        if length <= 0 or not series:
            return [float('nan')] * len(series)
        result = [float('nan')] * len(series)
        for i in range(length - 1, len(series)):
            result[i] = max(series[i - length + 1:i + 1])
        return result

    @staticmethod
    def lowest(series: List[float], length: int) -> List[float]:
        """Lowest value over lookback length."""
        if length <= 0 or not series:
            return [float('nan')] * len(series)
        result = [float('nan')] * len(series)
        for i in range(length - 1, len(series)):
            result[i] = min(series[i - length + 1:i + 1])
        return result

    @staticmethod
    def crossover(a: List[float], b: Union[List[float], float]) -> List[bool]:
        """True if series 'a' crosses above 'b' on current bar."""
        n = len(a)
        result = [False] * n
        is_b_scalar = isinstance(b, (int, float))
        for i in range(1, n):
            b_prev = b if is_b_scalar else b[i - 1]
            b_curr = b if is_b_scalar else b[i]
            if not (math.isnan(a[i - 1]) or math.isnan(a[i]) or math.isnan(b_prev) or math.isnan(b_curr)):
                if a[i - 1] <= b_prev and a[i] > b_curr:
                    result[i] = True
        return result

    @staticmethod
    def crossunder(a: List[float], b: Union[List[float], float]) -> List[bool]:
        """True if series 'a' crosses below 'b' on current bar."""
        n = len(a)
        result = [False] * n
        is_b_scalar = isinstance(b, (int, float))
        for i in range(1, n):
            b_prev = b if is_b_scalar else b[i - 1]
            b_curr = b if is_b_scalar else b[i]
            if not (math.isnan(a[i - 1]) or math.isnan(a[i]) or math.isnan(b_prev) or math.isnan(b_curr)):
                if a[i - 1] >= b_prev and a[i] < b_curr:
                    result[i] = True
        return result

    @staticmethod
    def macd(series: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float], List[float]]:
        """MACD line, Signal line, and Histogram."""
        fast_ema = PineTA.ema(series, fast)
        slow_ema = PineTA.ema(series, slow)
        macd_line = [f - s if not (math.isnan(f) or math.isnan(s)) else float('nan') for f, s in zip(fast_ema, slow_ema)]
        
        # Signal line
        valid_macd = [m for m in macd_line if not math.isnan(m)]
        sig_ema = PineTA.ema(valid_macd, signal)
        
        signal_line = [float('nan')] * len(series)
        hist = [float('nan')] * len(series)
        
        lead_nans = len(series) - len(valid_macd)
        for i, s_val in enumerate(sig_ema):
            idx = lead_nans + i
            signal_line[idx] = s_val
            if not math.isnan(macd_line[idx]) and not math.isnan(s_val):
                hist[idx] = macd_line[idx] - s_val
                
        return macd_line, signal_line, hist

    @staticmethod
    def vwap(high: List[float], low: List[float], close: List[float], volume: List[float], timestamps: List[Any]) -> List[float]:
        """Intraday Session VWAP (resets daily)."""
        n = len(close)
        result = [close[0]] * n
        cum_vol = 0.0
        cum_pv = 0.0
        cur_day = None
        
        for i in range(n):
            ts = timestamps[i]
            d = ts.date() if hasattr(ts, "date") else str(ts)[:10]
            if cur_day != d:
                cur_day = d
                cum_vol = 0.0
                cum_pv = 0.0
                
            typical = (high[i] + low[i] + close[i]) / 3.0
            v = volume[i] if volume[i] > 0 else 1.0
            cum_pv += typical * v
            cum_vol += v
            result[i] = (cum_pv / cum_vol) if cum_vol > 0 else close[i]
        return result

    @staticmethod
    def bollinger_bands(series: List[float], length: int = 20, mult: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
        """Bollinger Bands: Middle (SMA), Upper, Lower."""
        mid = PineTA.sma(series, length)
        upper = [float('nan')] * len(series)
        lower = [float('nan')] * len(series)
        
        for i in range(length - 1, len(series)):
            window = series[i - length + 1:i + 1]
            std = float(np.std(window))
            upper[i] = mid[i] + mult * std
            lower[i] = mid[i] - mult * std
        return mid, upper, lower

    @staticmethod
    def supertrend(high: List[float], low: List[float], close: List[float], factor: float = 3.0, atr_period: int = 10) -> Tuple[List[float], List[int]]:
        """SuperTrend line and direction (1 = Bullish, -1 = Bearish)."""
        atr_vals = PineTA.atr(high, low, close, atr_period)
        n = len(close)
        st = [close[0]] * n
        direction = [1] * n
        
        up = [(high[0] + low[0]) / 2.0 - factor * (atr_vals[0] if not math.isnan(atr_vals[0]) else 10.0)] * n
        dn = [(high[0] + low[0]) / 2.0 + factor * (atr_vals[0] if not math.isnan(atr_vals[0]) else 10.0)] * n

        for i in range(1, n):
            atr_v = atr_vals[i] if not math.isnan(atr_vals[i]) else 10.0
            hl2 = (high[i] + low[i]) / 2.0
            
            basic_up = hl2 - factor * atr_v
            basic_dn = hl2 + factor * atr_v
            
            up[i] = max(basic_up, up[i - 1]) if close[i - 1] > up[i - 1] else basic_up
            dn[i] = min(basic_dn, dn[i - 1]) if close[i - 1] < dn[i - 1] else basic_dn
            
            if direction[i - 1] == -1 and close[i] > dn[i - 1]:
                direction[i] = 1
            elif direction[i - 1] == 1 and close[i] < up[i - 1]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]
                
            st[i] = up[i] if direction[i] == 1 else dn[i]
            
        return st, direction

    @staticmethod
    def variance(series: List[float], length: int) -> List[float]:
        """Rolling sample variance (ta.variance in PineScript)."""
        if length <= 1 or not series:
            return [float('nan')] * len(series)
        result = [float('nan')] * len(series)
        for i in range(length - 1, len(series)):
            window = series[i - length + 1:i + 1]
            valid = [x for x in window if not math.isnan(x)]
            if len(valid) == length:
                result[i] = float(np.var(valid, ddof=1))
        return result

