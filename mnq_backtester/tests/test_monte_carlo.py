"""
Unit tests for Monte Carlo Stochastic Resampling & Stress Testing.
"""

from datetime import datetime
from mnq_backtester.core.order import TradeRecord, OrderSide
from mnq_backtester.monte_carlo.engine import MonteCarloEngine, ResamplingMethod
from mnq_backtester.monte_carlo.resampling import TradeResampler

def test_monte_carlo_resampling_methods():
    pnls = [100.0, -50.0, 150.0, -80.0, 200.0, -100.0]

    # Test Shuffle
    shuffled = TradeResampler.shuffle(pnls, seed=1)
    assert len(shuffled) == len(pnls)
    assert sorted(shuffled) == sorted(pnls)

    # Test Bootstrap
    bootstrapped = TradeResampler.iid_bootstrap(pnls, n_samples=20, seed=1)
    assert len(bootstrapped) == 20

    # Test Stationary Bootstrap
    stationary = TradeResampler.stationary_bootstrap(pnls, n_samples=20, avg_block_size=3.0, seed=1)
    assert len(stationary) == 20

def test_monte_carlo_orchestrator():
    now = datetime(2026, 1, 5, 10, 0)
    trades = []
    for i in range(20):
        pnl = 50.0 if i % 3 != 0 else -60.0
        trades.append(TradeRecord(
            trade_id=i + 1, strategy_tag="S1", symbol="MNQ", side=OrderSide.BUY,
            contracts=1, entry_time=now, entry_price=18000.0, exit_time=now,
            exit_price=18000.0 + pnl, exit_reason="TP", duration_bars=5, duration_minutes=5.0,
            points_pnl=pnl, gross_pnl_usd=pnl * 2.0, commission_usd=1.24, net_pnl_usd=pnl * 2.0 - 1.24,
            mfe_points=pnl + 5.0, mfe_usd=(pnl + 5.0) * 2.0, mae_points=5.0, mae_usd=10.0, r_multiple=1.5,
            account_equity_after=50000.0 + i * 50
        ))

    engine = MonteCarloEngine(iterations=200, seed=42)
    report = engine.run(trades, initial_capital=50000.0, method=ResamplingMethod.SHUFFLE)

    assert report.distribution.iterations == 200
    assert report.distribution.equity_p50 > 0
    assert report.distribution.max_dd_p95 >= report.distribution.max_dd_p50
    assert len(report.slippage_sensitivity) == 5
