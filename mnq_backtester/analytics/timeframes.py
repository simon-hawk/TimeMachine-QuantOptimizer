"""
Multi-Timeframe Performance Breakdown.
Computes granular window statistics for 7-Day, 1-Month, 3-Month, 1-Year, and 10-Year horizons.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from ..core.order import TradeRecord
from .metrics import PerformanceMetrics, PerformanceStats

@dataclass
class WindowMetric:
    window_label: str
    start_date: datetime
    end_date: datetime
    total_trades: int
    win_rate: float
    net_profit_usd: float
    profit_factor: float
    max_drawdown_usd: float
    max_drawdown_pct: float
    sharpe_ratio: float

@dataclass
class TimeframeReport:
    windows: List[WindowMetric]

    def summary_table(self) -> str:
        lines = [
            "┌────────────────────────────────────────────────────────────────────────────────────────────┐",
            "│                         MULTI-TIMEFRAME PERFORMANCE BREAKDOWN                              │",
            "├──────────┬────────┬──────────┬──────────────┬───────────────┬──────────────┬───────────────┤",
            "│ Window   │ Trades │ Win Rate │ Net P&L ($)  │ Profit Factor │ Max DD ($)   │ Max DD (%)    │",
            "├──────────┼────────┼──────────┼──────────────┼───────────────┼──────────────┼───────────────┤"
        ]
        for w in self.windows:
            lines.append(
                f"│ {w.window_label:<8} │ {w.total_trades:<6} │ {w.win_rate:>7.2f}% │ ${w.net_profit_usd:>11.2f} │ {w.profit_factor:>13.2f} │ ${w.max_drawdown_usd:>11.2f} │ {w.max_drawdown_pct:>12.2f}% │"
            )
        lines.append("└──────────┴────────┴──────────┴──────────────┴───────────────┴──────────────┴───────────────┘")
        return "\n".join(lines)


class MultiTimeframeAnalyzer:
    """
    Slices trades and equity curves across 7D, 1M, 3M, 1Y, and 10Y rolling and trailing windows.
    """

    WINDOWS = [
        ("7-Day", 7),
        ("1-Month", 30),
        ("3-Month", 90),
        ("1-Year", 365),
        ("10-Year", 3650)
    ]

    @classmethod
    def analyze(
        cls,
        trades: List[TradeRecord],
        equity_curve: List[Dict[str, Any]],
        initial_capital: float = 50_000.0
    ) -> TimeframeReport:
        if not trades or not equity_curve:
            return TimeframeReport(windows=[])

        # Reference end date is the last timestamp in equity curve
        last_t = equity_curve[-1]["timestamp"]
        if not isinstance(last_t, datetime):
            last_t = datetime.fromisoformat(str(last_t))

        first_t = equity_curve[0]["timestamp"]
        if not isinstance(first_t, datetime):
            first_t = datetime.fromisoformat(str(first_t))
        total_backtested_days = max(1, (last_t - first_t).days)

        window_metrics = []

        for label, days in cls.WINDOWS:
            # If the backtest history does not reach 1-Year or 10-Year, let the async long-timeframe loader populate them
            if days > total_backtested_days * 1.15 and days >= 365:
                continue

            cutoff_date = last_t - timedelta(days=days)
            
            # Filter trades in window
            w_trades = [t for t in trades if t.exit_time >= cutoff_date]
            # Filter equity in window
            w_equity = [p for p in equity_curve if (p["timestamp"] if isinstance(p["timestamp"], datetime) else datetime.fromisoformat(str(p["timestamp"]))) >= cutoff_date]

            if not w_trades:
                continue

            w_start_eq = w_equity[0]["equity"] if w_equity else initial_capital
            stats = PerformanceMetrics.calculate(w_trades, w_equity, initial_capital=w_start_eq)

            window_metrics.append(WindowMetric(
                window_label=label,
                start_date=cutoff_date,
                end_date=last_t,
                total_trades=stats.total_trades,
                win_rate=stats.win_rate,
                net_profit_usd=stats.net_profit_usd,
                profit_factor=stats.profit_factor,
                max_drawdown_usd=stats.max_drawdown_usd,
                max_drawdown_pct=stats.max_drawdown_pct,
                sharpe_ratio=stats.sharpe_ratio
            ))

        return TimeframeReport(windows=window_metrics)

