"""
Terminal Visualizer and ASCII Tearsheet Generator for MNQ Backtests.
"""

from typing import List, Dict, Any, Optional
from ..core.engine import BacktestResult
from ..analytics.metrics import PerformanceStats
from ..analytics.detailed_trades import TradeDistributionStats
from ..analytics.timeframes import TimeframeReport
from ..analytics.risk import VaRReport
from ..monte_carlo.engine import MonteCarloReport

class ReportVisualizer:
    """
    Renders clean, structured terminal tearsheets and reports.
    """

    @staticmethod
    def print_full_tearsheet(
        result: BacktestResult,
        stats: PerformanceStats,
        trade_dist: Optional[TradeDistributionStats] = None,
        tf_report: Optional[TimeframeReport] = None,
        var_report: Optional[VaRReport] = None,
        mc_report: Optional[MonteCarloReport] = None
    ) -> str:
        sections = []

        header = (
            "========================================================================================\n"
            "                      MNQ ADVANCED BACKTESTING & STRESS REPORT                          \n"
            "========================================================================================"
        )
        sections.append(header)

        # Performance table
        sections.append(stats.summary_table())

        # Trade excursion & distribution
        if trade_dist:
            sections.append(trade_dist.summary())

        # Multi-timeframe table (7D, 1M, 3M, 1Y, 10Y)
        if tf_report and tf_report.windows:
            sections.append(tf_report.summary_table())

        # Tail risk & Value at Risk
        if var_report:
            sections.append(var_report.summary())

        # Monte Carlo & Slippage Stress
        if mc_report:
            sections.append(mc_report.summary_table())

        footer = "========================================================================================"
        sections.append(footer)

        full_text = "\n\n".join(sections)
        return full_text
