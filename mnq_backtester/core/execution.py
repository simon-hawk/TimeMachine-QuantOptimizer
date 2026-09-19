"""
Execution and Fill Simulator for MNQ Futures.
Applies CME tick quantization, realistic slippage, and exchange commissions.
"""

from typing import Tuple, Optional
from datetime import datetime
from .order import Order, OrderType, OrderSide, Position
from ..config import BacktestConfig

class ExecutionSimulator:
    """
    Handles fill pricing, slippage modeling, and commission deduction for MNQ.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.tick_size = config.contract.tick_size        # 0.25 pt
        self.point_value = config.contract.point_value    # $2.00 / pt
        self.commission_per_side = config.commission_per_side # $0.62 / side
        self.slippage_ticks = config.slippage_ticks      # default 1 tick

    def quantize_price(self, price: float) -> float:
        """Rounds price to nearest CME 0.25 tick."""
        return round(round(price / self.tick_size) * self.tick_size, 2)

    def calculate_slippage(self, side: OrderSide, is_entry: bool) -> float:
        """
        Calculates slippage in points.
        For BUY entry or SELL exit (buying): slippage raises price (+).
        For SELL entry or BUY exit (selling): slippage lowers price (-).
        """
        slip_points = self.slippage_ticks * self.tick_size
        if (side == OrderSide.BUY and is_entry) or (side == OrderSide.SELL and not is_entry):
            return slip_points
        else:
            return -slip_points

    def calculate_commission(self, contracts: int, is_round_trip: bool = False) -> float:
        """Calculates total commission in USD."""
        multiplier = 2.0 if is_round_trip else 1.0
        return contracts * self.commission_per_side * multiplier

    def simulate_fill_price(self, order: Order, base_price: float, is_entry: bool = True) -> float:
        """Calculates market/stop fill price including spread crossing and slippage."""
        slip = self.calculate_slippage(order.side, is_entry=is_entry)
        return self.quantize_price(base_price + slip)

    def execute_market_entry(
        self,
        order: Order,
        bar_open: float,
        timestamp: datetime
    ) -> Position:
        """Executes a market order on bar open with slippage."""
        fill_price = self.simulate_fill_price(order, bar_open, is_entry=True)

        return Position(
            position_id=f"POS_{order.order_id}",
            symbol=order.symbol,
            side=order.side,
            contracts=order.contracts,
            entry_price=fill_price,
            entry_time=timestamp,
            initial_stop_loss=order.stop_loss_price,
            current_stop_loss=order.stop_loss_price,
            take_profit=order.take_profit_price,
            trail_distance_points=order.trail_distance_points,
            strategy_tag=order.strategy_tag
        )

ExecutionEngine = ExecutionSimulator
