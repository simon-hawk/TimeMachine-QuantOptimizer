"""
Fair Value Gap (FVG) and Inverse FVG Strategy for MNQ.
Identifies 3-bar imbalance zones (gaps) created with strong displacement bodies (>= 60%)
and trades re-tests / inversions with defined risk.
"""

from datetime import datetime, time
from typing import List, Dict, Any, Optional
from .base import BaseStrategy
from ..core.order import Order, OrderType, OrderSide, Position

class FairValueGapStrategy(BaseStrategy):
    """
    ICT Fair Value Gap (FVG) / Inverse FVG intraday strategy on 5-minute bars.
    """

    def __init__(
        self,
        min_fvg_points: float = 3.0,
        displacement_body_pct: float = 0.60,
        risk_reward: float = 2.0,
        contracts: int = 1
    ):
        super().__init__(name="FVG_Model", parameters={
            "min_fvg_points": min_fvg_points,
            "displacement_body_pct": displacement_body_pct,
            "risk_reward": risk_reward,
            "contracts": contracts
        })
        self.min_fvg_points = min_fvg_points
        self.displacement_body_pct = displacement_body_pct
        self.risk_reward = risk_reward
        self.contracts = contracts

    def on_bar(
        self,
        bar_index: int,
        bars: List[Dict[str, Any]],
        active_position: Optional[Position]
    ) -> Optional[Order]:
        if bar_index < 5 or active_position is not None:
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
        # Restrict to primary liquid window (09:45 to 15:30 ET)
        if not (time(9, 45) <= t <= time(15, 30)):
            return None

        # Look at previous 3-bar pattern: b0, b1, b2
        b0 = bars[bar_index - 3]
        b1 = bars[bar_index - 2] # displacement bar
        b2 = bars[bar_index - 1] # completing bar

        b1_range = float(b1["high"]) - float(b1["low"])
        b1_body = abs(float(b1["close"]) - float(b1["open"]))
        disp_pct = (b1_body / b1_range) if b1_range > 0 else 0.0

        if disp_pct < self.displacement_body_pct:
            return None

        cur_c = float(cur_bar["close"])
        cur_l = float(cur_bar["low"])
        cur_h = float(cur_bar["high"])

        # Bullish FVG: b0.high < b2.low (Gap in price)
        if float(b0["high"]) < float(b2["low"]):
            fvg_top = float(b2["low"])
            fvg_bot = float(b0["high"])
            fvg_width = fvg_top - fvg_bot

            if fvg_width >= self.min_fvg_points:
                # Retest tap into zone on current bar
                if cur_l <= fvg_top and cur_c >= fvg_bot:
                    sl = fvg_bot - 2.0 # Stop slightly below FVG bottom
                    risk_pts = cur_c - sl
                    if 4.0 <= risk_pts <= 25.0:
                        tp = cur_c + (risk_pts * self.risk_reward)
                        return Order(
                            order_id=f"FVG_BUY_{bar_index}",
                            symbol="MNQ",
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            contracts=self.contracts,
                            stop_loss_price=round(round(sl / 0.25) * 0.25, 2),
                            take_profit_price=round(round(tp / 0.25) * 0.25, 2),
                            created_at=b_time,
                            strategy_tag="FVG_LONG"
                        )

        # Bearish FVG: b0.low > b2.high
        elif float(b0["low"]) > float(b2["high"]):
            fvg_bot = float(b2["high"])
            fvg_top = float(b0["low"])
            fvg_width = fvg_top - fvg_bot

            if fvg_width >= self.min_fvg_points:
                # Retest tap into zone
                if cur_h >= fvg_bot and cur_c <= fvg_top:
                    sl = fvg_top + 2.0
                    risk_pts = sl - cur_c
                    if 4.0 <= risk_pts <= 25.0:
                        tp = cur_c - (risk_pts * self.risk_reward)
                        return Order(
                            order_id=f"FVG_SELL_{bar_index}",
                            symbol="MNQ",
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            contracts=self.contracts,
                            stop_loss_price=round(round(sl / 0.25) * 0.25, 2),
                            take_profit_price=round(round(tp / 0.25) * 0.25, 2),
                            created_at=b_time,
                            strategy_tag="FVG_SHORT"
                        )

        return None
