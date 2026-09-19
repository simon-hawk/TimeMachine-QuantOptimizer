"""
Opening Range Breakout (ORB) Strategy for MNQ.
Identifies the high and low of the first 15 minutes of the RTH session (09:30 - 09:45 ET)
and enters on clean break with midpoint stop-loss and 2.0 RR target.
"""

from datetime import datetime, time
from typing import List, Dict, Any, Optional
from .base import BaseStrategy
from ..core.order import Order, OrderType, OrderSide, Position

class OpeningRangeBreakoutStrategy(BaseStrategy):
    """
    15-minute Opening Range Breakout Strategy with Midpoint Stop-Loss and 2:1 Reward-to-Risk.
    """

    def __init__(
        self,
        orb_minutes: int = 15,
        risk_reward: float = 2.0,
        contracts: int = 1,
        max_risk_points: float = 30.0
    ):
        super().__init__(name="15m_ORB", parameters={
            "orb_minutes": orb_minutes,
            "risk_reward": risk_reward,
            "contracts": contracts,
            "max_risk_points": max_risk_points
        })
        self.orb_minutes = orb_minutes
        self.risk_reward = risk_reward
        self.contracts = contracts
        self.max_risk_points = max_risk_points
        
        self.current_day = None
        self.orb_high = None
        self.orb_low = None
        self.orb_complete = False
        self.traded_today = False

    def on_bar(
        self,
        bar_index: int,
        bars: List[Dict[str, Any]],
        active_position: Optional[Position]
    ) -> Optional[Order]:
        cur_bar = bars[bar_index]
        raw_t = cur_bar.get("timestamp") or cur_bar.get("time_ny") or cur_bar.get("date")
        if isinstance(raw_t, datetime):
            b_time = raw_t
        else:
            try:
                b_time = datetime.fromisoformat(str(raw_t))
            except Exception:
                return None

        b_date = b_time.date()
        t = b_time.time()

        # Reset daily state on new trading day
        if self.current_day != b_date:
            self.current_day = b_date
            self.orb_high = None
            self.orb_low = None
            self.orb_complete = False
            self.traded_today = False

        # Build ORB high/low between 09:30 and 09:45
        if time(9, 30) <= t < time(9, 30 + self.orb_minutes):
            h = float(cur_bar["high"])
            l = float(cur_bar["low"])
            self.orb_high = h if self.orb_high is None else max(self.orb_high, h)
            self.orb_low = l if self.orb_low is None else min(self.orb_low, l)
            return None

        # At or after 09:45, mark ORB as established
        if t >= time(9, 30 + self.orb_minutes) and not self.orb_complete and self.orb_high is not None:
            self.orb_complete = True

        # Check for trade trigger during the morning window (09:45 to 11:30)
        if self.orb_complete and not self.traded_today and active_position is None:
            if time(9, 30 + self.orb_minutes) <= t <= time(11, 30):
                c = float(cur_bar["close"])
                midpoint = (self.orb_high + self.orb_low) / 2.0
                
                # Bullish Breakout
                if c > self.orb_high:
                    risk_pts = min(self.max_risk_points, c - midpoint)
                    if risk_pts > 4.0: # Minimum 4 points risk for valid setup
                        sl = c - risk_pts
                        tp = c + (risk_pts * self.risk_reward)
                        self.traded_today = True
                        return Order(
                            order_id=f"ORB_BUY_{bar_index}",
                            symbol="MNQ",
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            contracts=self.contracts,
                            stop_loss_price=round(round(sl / 0.25) * 0.25, 2),
                            take_profit_price=round(round(tp / 0.25) * 0.25, 2),
                            created_at=b_time,
                            strategy_tag="15m_ORB_LONG"
                        )

                # Bearish Breakdown
                elif c < self.orb_low:
                    risk_pts = min(self.max_risk_points, midpoint - c)
                    if risk_pts > 4.0:
                        sl = c + risk_pts
                        tp = c - (risk_pts * self.risk_reward)
                        self.traded_today = True
                        return Order(
                            order_id=f"ORB_SELL_{bar_index}",
                            symbol="MNQ",
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            contracts=self.contracts,
                            stop_loss_price=round(round(sl / 0.25) * 0.25, 2),
                            take_profit_price=round(round(tp / 0.25) * 0.25, 2),
                            created_at=b_time,
                            strategy_tag="15m_ORB_SHORT"
                        )

        return None
