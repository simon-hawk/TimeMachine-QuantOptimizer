"""
Unit tests for MNQ Execution Engine, Point Multiplier, Slippage, and Fills.
"""

from datetime import datetime, time
from mnq_backtester.config import BacktestConfig
from mnq_backtester.core.engine import BacktestEngine
from mnq_backtester.core.order import Order, OrderType, OrderSide, Position

def test_mnq_point_multiplier_and_commission():
    config = BacktestConfig(
        initial_capital=50_000.0,
        commission_per_side=0.62, # $1.24 round trip
        slippage_ticks=0.0        # zero slippage for exact math test
    )
    engine = BacktestEngine(config=config)

    # 2-bar test: Buy on bar 0 at 18000.00, TP hit on bar 1 at 18010.00 (+10.0 pts)
    bars = [
        {"timestamp": datetime(2026, 1, 5, 9, 30), "open": 18000.0, "high": 18005.0, "low": 17998.0, "close": 18002.0},
        {"timestamp": datetime(2026, 1, 5, 9, 31), "open": 18002.0, "high": 18012.0, "low": 18001.0, "close": 18010.0}
    ]

    def simple_strategy(idx, bars_hist, active_pos):
        if idx == 0 and active_pos is None:
            return Order(
                order_id="BUY_1",
                symbol="MNQ",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                contracts=1,
                stop_loss_price=17990.0,
                take_profit_price=18010.0
            )
        return None

    res = engine.run(bars, simple_strategy)
    assert len(res.trades) == 1
    trade = res.trades[0]

    # Filled on Bar 1 Open at 18002.00, TP hit at 18010.00 (+8.0 pts)
    # +8.0 points * $2.00/point = $16.00 gross
    # Commission: $1.24
    # Net: $14.76
    assert trade.points_pnl == 8.0
    assert trade.gross_pnl_usd == 16.00
    assert trade.commission_usd == 1.24
    assert trade.net_pnl_usd == 14.76
    assert trade.exit_reason == "TAKE_PROFIT"
    assert engine.current_equity == 50_014.76
