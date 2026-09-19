"""
Slippage Sensitivity and Fragility Stress Tester.
Evaluates strategy robustness by incrementally degrading execution friction from 0 to 4 ticks per trade.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from ..core.order import TradeRecord
from ..analytics.metrics import PerformanceMetrics

@dataclass
class SlippageSensitivityResult:
    slippage_ticks: float
    slippage_cost_usd_per_trade: float
    net_profit_usd: float
    profit_factor: float
    win_rate: float
    max_drawdown_usd: float
    is_profitable: bool

class SlippageStressTester:
    """
    Simulates degrading execution quality across trade logs to determine strategy breaking point.
    """

    @staticmethod
    def evaluate_sensitivity(
        trades: List[TradeRecord],
        point_value: float = 2.00,
        tick_size: float = 0.25,
        initial_capital: float = 50_000.0,
        tested_ticks: tuple = (0.0, 1.0, 2.0, 3.0, 4.0)
    ) -> List[SlippageSensitivityResult]:
        if not trades:
            return []

        results = []
        for ticks in tested_ticks:
            # Additional slippage penalty relative to base (assumed 1.0 tick base)
            tick_diff = ticks - 1.0
            penalty_pts = tick_diff * tick_size * 2.0 # entry + exit = 2 sides
            penalty_usd = penalty_pts * point_value

            adjusted_trades = []
            for t in trades:
                adj_points = t.points_pnl - penalty_pts
                adj_net = t.net_pnl_usd - penalty_usd
                
                # Mock adjusted trade record
                adj_t = TradeRecord(
                    trade_id=t.trade_id,
                    strategy_tag=t.strategy_tag,
                    symbol=t.symbol,
                    side=t.side,
                    contracts=t.contracts,
                    entry_time=t.entry_time,
                    entry_price=t.entry_price,
                    exit_time=t.exit_time,
                    exit_price=t.exit_price,
                    exit_reason=t.exit_reason,
                    duration_bars=t.duration_bars,
                    duration_minutes=t.duration_minutes,
                    points_pnl=adj_points,
                    gross_pnl_usd=t.gross_pnl_usd - penalty_usd,
                    commission_usd=t.commission_usd,
                    net_pnl_usd=adj_net,
                    mfe_points=t.mfe_points,
                    mfe_usd=t.mfe_usd,
                    mae_points=t.mae_points,
                    mae_usd=t.mae_usd,
                    r_multiple=t.r_multiple,
                    account_equity_after=t.account_equity_after
                )
                adjusted_trades.append(adj_t)

            # Recompute summary
            stats = PerformanceMetrics.calculate(adjusted_trades, [], initial_capital=initial_capital)
            cost_per_trade = ticks * tick_size * 2.0 * point_value

            results.append(SlippageSensitivityResult(
                slippage_ticks=ticks,
                slippage_cost_usd_per_trade=round(cost_per_trade, 2),
                net_profit_usd=stats.net_profit_usd,
                profit_factor=stats.profit_factor,
                win_rate=stats.win_rate,
                max_drawdown_usd=stats.max_drawdown_usd,
                is_profitable=stats.net_profit_usd > 0
            ))

        return results
