"""
CME Trading Sessions and Bar Aggregation.
Handles RTH (09:30-16:00 ET), ETH (Overnight/Pre-market), and multi-timeframe bar resampling.
"""

from enum import Enum
from datetime import datetime, time
from typing import List, Dict, Any, Optional

class SessionType(Enum):
    ALL = "ALL"
    RTH = "RTH"               # Regular Trading Hours: 09:30 - 16:00 ET
    ETH = "ETH"               # Extended Trading Hours: 18:00 - 09:30 ET
    PRE_MARKET = "PRE_MARKET" # 07:00 - 09:30 ET
    CUSTOM = "CUSTOM"

class SessionManager:
    """
    Filters and labels intraday bars based on CME trading sessions.
    """

    @staticmethod
    def get_bar_session(bar_time: datetime) -> SessionType:
        """Determines session for a given bar timestamp in NY time."""
        t = bar_time.time()
        if time(9, 30) <= t < time(16, 0):
            return SessionType.RTH
        elif time(7, 0) <= t < time(9, 30):
            return SessionType.PRE_MARKET
        else:
            return SessionType.ETH

    @staticmethod
    def filter_session(bars: List[Dict[str, Any]], session_type: SessionType = SessionType.RTH) -> List[Dict[str, Any]]:
        """Filters a bar list down to only the requested session type."""
        if session_type == SessionType.ALL:
            return bars

        filtered = []
        for bar in bars:
            raw_time = bar.get("timestamp") or bar.get("time_ny") or bar.get("date")
            if not isinstance(raw_time, datetime):
                try:
                    bar_time = datetime.fromisoformat(str(raw_time))
                except Exception:
                    filtered.append(bar)
                    continue
            else:
                bar_time = raw_time

            b_session = SessionManager.get_bar_session(bar_time)
            if session_type == SessionType.RTH and b_session == SessionType.RTH:
                filtered.append(bar)
            elif session_type == SessionType.PRE_MARKET and b_session == SessionType.PRE_MARKET:
                filtered.append(bar)
            elif session_type == SessionType.ETH and b_session in (SessionType.ETH, SessionType.PRE_MARKET):
                filtered.append(bar)
        return filtered

    @staticmethod
    def aggregate_bars(bars_1m: List[Dict[str, Any]], timeframe_minutes: int = 5) -> List[Dict[str, Any]]:
        """
        Resamples 1-minute bars into higher timeframe bars (e.g. 5m, 15m, 1h).
        Preserves timestamps, High as max, Low as min, Volume as sum.
        """
        if timeframe_minutes <= 1:
            return bars_1m

        aggregated = []
        for i in range(0, len(bars_1m), timeframe_minutes):
            chunk = bars_1m[i:i + timeframe_minutes]
            if not chunk:
                continue

            first_bar = chunk[0]
            last_bar = chunk[-1]
            raw_time = first_bar.get("timestamp") or first_bar.get("time_ny") or first_bar.get("date")

            agg_bar = {
                "timestamp": raw_time,
                "time_ny": raw_time,
                "open": first_bar["open"],
                "high": max(b["high"] for b in chunk),
                "low": min(b["low"] for b in chunk),
                "close": last_bar["close"],
                "volume": sum(b.get("volume", 0) for b in chunk),
                "num_1m_bars": len(chunk)
            }
            aggregated.append(agg_bar)
        return aggregated
