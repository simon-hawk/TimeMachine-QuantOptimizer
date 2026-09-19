"""
Ilmanen Factor Strategy: Trend Following + Value/Overextension Cap + Dynamic Volatility Targeting.
Based on Antti Ilmanen's Master Treatise: 'Expected Returns: An Investor's Guide to Harvesting Market Rewards' (Wiley 2011).

Key Concepts Implemented from the 600-page Treatise:
1. Time-Series Trend / Momentum Filter: Captures persistent trends across medium/short horizons.
2. Value Overextension Guard (Mean-Reversion): Caps or fades extreme expansions (+2.5 sigma divergence)
   to protect against sudden momentum crashes.
3. Continuous Dynamic Volatility Targeting: Sizing is inversely proportional to realized volatility
   (ATR / rolling std), keeping portfolio risk budget constant across market regimes.
4. Intraday U-Shaped Liquidity Curve: Trades actively during Morning Open (09:30-11:30) and
   Power Hour (14:30-15:45), while suppressing breakout trades during Midday Mean-Reverting Chop (11:30-14:00).
5. Turn-of-the-Month (TOTM) Effect: Calendar tilt expanding risk budget during last trading day and
   first 3 days of the calendar month.
"""

from datetime import datetime, time
from typing import List, Dict, Any, Optional
import math
import numpy as np

from .base import BaseStrategy
from ..core.order import Order, OrderType, OrderSide, Position


class IlmanenFactorStrategy(BaseStrategy):
    """
    Event-driven multi-factor strategy implementing Ilmanen's Core Factor Premia:
    - Trend (Moving Average Envelopes / Dual Trend)
    - Value / Mean Reversion (Statistical Z-score Overextension Cap)
    - Volatility Targeting (Dynamic Risk-Parity Contract Sizing)
    - Intraday Volume/Volatility Regime Gating (Open & Power Hour vs Midday Chop)
    - Turn-of-the-Month (TOTM) Seasonal Risk Multiplier
    """

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 100,
        overextension_z_threshold: float = 2.5,
        target_vol_points: float = 25.0,  # Target risk budget in index points
        base_contracts: int = 1,
        max_contracts: int = 4,
        stop_atr_multiplier: float = 2.0,
        take_profit_atr_multiplier: float = 3.5,
        atr_period: int = 14,
        enable_totm_filter: bool = True,
        enable_intraday_timing: bool = True,
    ):
        super().__init__(
            name="Ilmanen_Trend_Value_Factor_v2",
            parameters={
                "fast_period": fast_period,
                "slow_period": slow_period,
                "overextension_z_threshold": overextension_z_threshold,
                "target_vol_points": target_vol_points,
                "base_contracts": base_contracts,
                "max_contracts": max_contracts,
                "stop_atr_multiplier": stop_atr_multiplier,
                "take_profit_atr_multiplier": take_profit_atr_multiplier,
                "atr_period": atr_period,
                "enable_totm_filter": enable_totm_filter,
                "enable_intraday_timing": enable_intraday_timing,
            },
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.z_threshold = overextension_z_threshold
        self.target_vol = target_vol_points
        self.base_contracts = base_contracts
        self.max_contracts = max_contracts
        self.stop_mult = stop_atr_multiplier
        self.tp_mult = take_profit_atr_multiplier
        self.atr_period = atr_period
        self.enable_totm = enable_totm_filter
        self.enable_intraday = enable_intraday_timing

        self.closes: List[float] = []
        self.highs: List[float] = []
        self.lows: List[float] = []
        self.current_day = None

    def _calculate_atr(self) -> float:
        """Computes true range and average true range."""
        if len(self.closes) < self.atr_period + 1:
            return 15.0  # Safe default for MNQ

        tr_list = []
        start_idx = len(self.closes) - self.atr_period
        for i in range(start_idx, len(self.closes)):
            h = self.highs[i]
            l = self.lows[i]
            prev_c = self.closes[i - 1]
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            tr_list.append(tr)
        return float(np.mean(tr_list)) if tr_list else 15.0

    def _is_turn_of_month(self, dt: datetime) -> bool:
        """
        Ilmanen Turn-of-the-Month (TOTM) Effect:
        Elevated institutional flows on the last day of the month and first 3 calendar days.
        """
        # Day 1, 2, 3 or late in month (28th+)
        return dt.day in (1, 2, 3) or dt.day >= 28

    def _calculate_vol_targeted_contracts(self, realized_atr: float, is_totm: bool) -> int:
        """
        Ilmanen Dynamic Volatility Targeting:
        Weight proportional to Target Vol / Realized Vol, boosted during TOTM.
        """
        if realized_atr <= 0:
            return self.base_contracts
        vol_scalar = self.target_vol / realized_atr
        # TOTM boost (+25% risk budget)
        if self.enable_totm and is_totm:
            vol_scalar *= 1.25

        sized_qty = int(round(self.base_contracts * vol_scalar))
        return max(1, min(self.max_contracts, sized_qty))

    def on_bar(
        self,
        bar_index: int,
        bars: List[Dict[str, Any]],
        active_position: Optional[Position],
    ) -> Optional[Order]:
        cur_bar = bars[bar_index]
        close_p = float(cur_bar["close"])
        high_p = float(cur_bar["high"])
        low_p = float(cur_bar["low"])

        self.closes.append(close_p)
        self.highs.append(high_p)
        self.lows.append(low_p)

        # Ensure adequate history for lookback
        if len(self.closes) < self.slow_period + 10:
            return None

        raw_t = cur_bar.get("timestamp") or cur_bar.get("time_ny") or cur_bar.get("date")
        if isinstance(raw_t, datetime):
            b_time = raw_t
        else:
            try:
                b_time = datetime.fromisoformat(str(raw_t))
            except Exception:
                return None

        t = b_time.time()
        # Ensure within RTH session
        if not (time(9, 30) <= t < time(15, 45)):
            return None

        # Intraday U-Shaped Volatility Timing:
        # Chapter 21: Morning Open (09:30-11:30) & Afternoon Close (14:00-15:45) favor Trend;
        # Midday (11:30-14:00) favors Mean-Reversion / Exhaustion Fades only.
        is_morning_open = time(9, 30) <= t < time(11, 30)
        is_power_hour = time(14, 0) <= t < time(15, 45)
        is_midday_chop = time(11, 30) <= t < time(14, 0)

        # 1. Trend Factor Calculation (Fast vs Slow Moving Averages)
        fast_ma = float(np.mean(self.closes[-self.fast_period:]))
        slow_ma = float(np.mean(self.closes[-self.slow_period:]))
        trend_bullish = fast_ma > slow_ma
        trend_bearish = fast_ma < slow_ma

        # 2. Value / Overextension Z-score
        slow_window = self.closes[-self.slow_period:]
        rolling_std = float(np.std(slow_window)) or 1.0
        z_score = (close_p - slow_ma) / rolling_std

        # 3. Volatility Sizing & TOTM Seasonality
        is_totm = self._is_turn_of_month(b_time)
        atr = self._calculate_atr()
        contract_qty = self._calculate_vol_targeted_contracts(atr, is_totm)
        stop_dist = max(5.0, atr * self.stop_mult)
        tp_dist = max(10.0, atr * self.tp_mult)

        if active_position is not None:
            return None

        def to_tick(p: float) -> float:
            return round(round(p / 0.25) * 0.25, 2)

        # A. Bullish Trend Entry: Active during morning or afternoon trend windows
        if (is_morning_open or is_power_hour) and trend_bullish and (0.2 <= z_score <= self.z_threshold):
            return Order(
                order_id=f"ILMANEN_TREND_BUY_{bar_index}",
                symbol="MNQ",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                contracts=contract_qty,
                stop_loss_price=to_tick(close_p - stop_dist),
                take_profit_price=to_tick(close_p + tp_dist),
                created_at=b_time,
                strategy_tag="Ilmanen_Trend_Long",
            )

        # B. Bearish Trend Entry: Active during morning or afternoon trend windows
        elif (is_morning_open or is_power_hour) and trend_bearish and (-self.z_threshold <= z_score <= -0.2):
            return Order(
                order_id=f"ILMANEN_TREND_SELL_{bar_index}",
                symbol="MNQ",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                contracts=contract_qty,
                stop_loss_price=to_tick(close_p + stop_dist),
                take_profit_price=to_tick(close_p - tp_dist),
                created_at=b_time,
                strategy_tag="Ilmanen_Trend_Short",
            )

        # C. Value Mean-Reversion Counter-Trade on Extreme Overextension
        # Especially potent during midday chop when momentum runs out of liquidity
        elif z_score > (self.z_threshold + 0.3):
            return Order(
                order_id=f"ILMANEN_FADE_SELL_{bar_index}",
                symbol="MNQ",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                contracts=max(1, contract_qty // 2),
                stop_loss_price=to_tick(close_p + (stop_dist * 0.75)),
                take_profit_price=to_tick(slow_ma),
                created_at=b_time,
                strategy_tag="Ilmanen_Value_Fade_Short",
            )

        elif z_score < -(self.z_threshold + 0.3):
            return Order(
                order_id=f"ILMANEN_DIP_BUY_{bar_index}",
                symbol="MNQ",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                contracts=max(1, contract_qty // 2),
                stop_loss_price=to_tick(close_p - (stop_dist * 0.75)),
                take_profit_price=to_tick(slow_ma),
                created_at=b_time,
                strategy_tag="Ilmanen_Value_Dip_Long",
            )

        return None
