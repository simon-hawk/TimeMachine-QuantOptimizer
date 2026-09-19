"""
Break of Structure (BOS) and Momentum Gravity Strategy for MNQ.
Identifies swing high/low structural breaks with EMA trend alignment and ATR volatility stops.
"""

from datetime import datetime, time
from typing import List, Dict, Any, Optional
import numpy as np
from .base import BaseStrategy
from ..core.order import Order, OrderType, OrderSide, Position

class BOSWavesGravityStrategy(BaseStrategy):
    """
    Break of Structure (BOS) with Momentum Trend Alignment and 2:1 R-Multiple.
    """

    def __init__(
        self,
        lookback_swing: int = 10,
        ema_period: int = 20,
        risk_reward: float = 2.0,
        stop_loss_points: float = 15.0,
        contracts: int = 1
    ):
        super().__init__(name="BOSWaves", parameters={
            "lookback_swing": lookback_swing,
            "ema_period": ema_period,
            "risk_reward": risk_reward,
            "stop_loss_points": stop_loss_points,
            "contracts": contracts
        })
        self.lookback_swing = lookback_swing
        self.ema_period = ema_period
        self.risk_reward = risk_reward
        self.stop_loss_points = stop_loss_points
        self.contracts = contracts

    def on_bar(
        self,
        bar_index: int,
        bars: List[Dict[str, Any]],
        active_position: Optional[Position]
    ) -> Optional[Order]:
        if bar_index < max(self.lookback_swing + 5, self.ema_period + 5) or active_position is not None:
            return None

        cur_bar = bars[bar_index]
        raw_t = cur_bar.get("timestamp") or cur_bar.get("time_ny") or cur_bar.get("date")
        if isinstance(raw_t, datetime):
            b_time = raw_t
        else:
            try:
                b_time = datetime.fromisoformat(str(raw_t))
            except Exception:
                return None

        t = b_time.time()
        if not (time(9, 45) <= t <= time(15, 30)):
            return None

        # Calculate EMA
        closes = [float(b["close"]) for b in bars[bar_index - self.ema_period:bar_index + 1]]
        ema_val = float(np.mean(closes)) # Fast approximation for window

        # Swing High & Swing Low over lookback
        swing_bars = bars[bar_index - self.lookback_swing:bar_index]
        swing_high = max(float(b["high"]) for b in swing_bars)
        swing_low = min(float(b["low"]) for b in swing_bars)

        cur_c = float(cur_bar["close"])

        # Bullish BOS Breakout above swing high + above EMA
        if cur_c > swing_high and cur_c > ema_val:
            sl = cur_c - self.stop_loss_points
            tp = cur_c + (self.stop_loss_points * self.risk_reward)
            return Order(
                order_id=f"BOS_BUY_{bar_index}",
                symbol="MNQ",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                contracts=self.contracts,
                stop_loss_price=round(round(sl / 0.25) * 0.25, 2),
                take_profit_price=round(round(tp / 0.25) * 0.25, 2),
                created_at=b_time,
                strategy_tag="BOS_LONG"
            )

        # Bearish BOS Breakdown below swing low + below EMA
        elif cur_c < swing_low and cur_c < ema_val:
            sl = cur_c + self.stop_loss_points
            tp = cur_c - (self.stop_loss_points * self.risk_reward)
            return Order(
                order_id=f"BOS_SELL_{bar_index}",
                symbol="MNQ",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                contracts=self.contracts,
                stop_loss_price=round(round(sl / 0.25) * 0.25, 2),
                take_profit_price=round(round(tp / 0.25) * 0.25, 2),
                created_at=b_time,
                strategy_tag="BOS_SHORT"
            )

        return None
