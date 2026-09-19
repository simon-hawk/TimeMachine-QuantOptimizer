"""
Unit tests for Performance Metrics & Risk Statistics.
"""

from datetime import datetime
from mnq_backtester.core.order import TradeRecord, OrderSide
from mnq_backtester.analytics.metrics import PerformanceMetrics

def test_performance_metrics_math():
    now = datetime(2026, 1, 5, 10, 0)
    # 3 trades: 2 wins (+100, +200), 1 loss (-100)
    trades = [
        TradeRecord(
            trade_id=1, strategy_tag="S1", symbol="MNQ", side=OrderSide.BUY,
            contracts=1, entry_time=now, entry_price=18000.0, exit_time=now,
            exit_price=18050.0, exit_reason="TP", duration_bars=5, duration_minutes=5.0,
            points_pnl=50.0, gross_pnl_usd=100.0, commission_usd=0.0, net_pnl_usd=100.0,
            mfe_points=55.0, mfe_usd=110.0, mae_points=5.0, mae_usd=10.0, r_multiple=2.0,
            account_equity_after=50100.0
        ),
        TradeRecord(
            trade_id=2, strategy_tag="S1", symbol="MNQ", side=OrderSide.BUY,
            contracts=1, entry_time=now, entry_price=18000.0, exit_time=now,
            exit_price=18100.0, exit_reason="TP", duration_bars=10, duration_minutes=10.0,
            points_pnl=100.0, gross_pnl_usd=200.0, commission_usd=0.0, net_pnl_usd=200.0,
            mfe_points=105.0, mfe_usd=210.0, mae_points=5.0, mae_usd=10.0, r_multiple=4.0,
            account_equity_after=50300.0
        ),
        TradeRecord(
            trade_id=3, strategy_tag="S1", symbol="MNQ", side=OrderSide.BUY,
            contracts=1, entry_time=now, entry_price=18000.0, exit_time=now,
            exit_price=17950.0, exit_reason="SL", duration_bars=3, duration_minutes=3.0,
            points_pnl=-50.0, gross_pnl_usd=-100.0, commission_usd=0.0, net_pnl_usd=-100.0,
            mfe_points=5.0, mfe_usd=10.0, mae_points=50.0, mae_usd=100.0, r_multiple=-1.0,
            account_equity_after=50200.0
        )
    ]

    equity_curve = [
        {"timestamp": now, "equity": 50000.0},
        {"timestamp": now, "equity": 50100.0},
        {"timestamp": now, "equity": 50300.0},
        {"timestamp": now, "equity": 50200.0}
    ]

    stats = PerformanceMetrics.calculate(trades, equity_curve, initial_capital=50000.0)
    assert stats.total_trades == 3
    assert stats.winning_trades == 2
    assert stats.losing_trades == 1
    assert abs(stats.win_rate - 66.67) < 0.05
    assert stats.gross_profit_usd == 300.0
    assert stats.gross_loss_usd == 100.0
    assert stats.profit_factor == 3.0
    assert stats.net_profit_usd == 200.0
    assert stats.payoff_ratio == 1.5 # Avg win (150) / Avg loss (100)
    assert stats.max_drawdown_usd == 100.0
