"""
Orders, Positions, and High-Resolution Trade Records for MNQ Futures.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    BRACKET = "BRACKET"

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

@dataclass
class Order:
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    contracts: int = 1
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    trail_distance_points: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    strategy_tag: str = "DEFAULT"

@dataclass
class Position:
    position_id: str
    symbol: str
    side: OrderSide
    contracts: int
    entry_price: float
    entry_time: datetime
    entry_bar_index: int = 0
    initial_stop_loss: Optional[float] = None
    current_stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trail_distance_points: Optional[float] = None
    peak_favorable_price: float = 0.0
    worst_adverse_price: float = 0.0
    strategy_tag: str = "DEFAULT"
    bars_held: int = 0

    def __post_init__(self):
        if self.peak_favorable_price == 0.0:
            self.peak_favorable_price = self.entry_price
        if self.worst_adverse_price == 0.0:
            self.worst_adverse_price = self.entry_price

    @property
    def stop_loss(self) -> Optional[float]:
        return self.current_stop_loss

    def update_excursions(self, high: float, low: float):
        """Updates MFE (Peak Favorable) and MAE (Worst Adverse) excursions."""
        self.bars_held += 1
        if self.side == OrderSide.BUY:
            if high > self.peak_favorable_price:
                self.peak_favorable_price = high
            if low < self.worst_adverse_price:
                self.worst_adverse_price = low
        else: # SELL
            if low < self.peak_favorable_price:
                self.peak_favorable_price = low
            if high > self.worst_adverse_price:
                self.worst_adverse_price = high

@dataclass
class TradeRecord:
    trade_id: int
    symbol: str
    side: OrderSide
    contracts: int
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    exit_reason: str
    duration_minutes: float
    points_pnl: float
    gross_pnl_usd: float
    commission_usd: float
    net_pnl_usd: float
    mfe_points: float
    mfe_usd: float
    mae_points: float
    mae_usd: float
    r_multiple: float
    account_equity_after: float
    order_id: str = "ORDER_0"
    entry_bar_index: int = 0
    exit_bar_index: int = 0
    strategy_tag: str = "DEFAULT"
    duration_bars: int = 1
    drawdown_at_exit_pct: float = 0.0

    @property
    def is_win(self) -> bool:
        return self.net_pnl_usd > 0.0

    @property
    def is_loss(self) -> bool:
        return self.net_pnl_usd < 0.0

    @property
    def is_breakeven(self) -> bool:
        return self.net_pnl_usd == 0.0
