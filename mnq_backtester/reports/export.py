"""
Export Utilities: JSON, CSV, and Markdown report generators.
"""

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Any
from ..core.engine import BacktestResult
from ..analytics.metrics import PerformanceStats

class ReportExporter:
    """
    Exports backtest results, trade logs, and metrics to files.
    """

    @staticmethod
    def export_trades_to_csv(trades: List[Any], file_path: str):
        """Exports detailed trade records to a CSV file."""
        if not trades:
            return

        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "trade_id", "strategy_tag", "symbol", "side", "contracts",
            "entry_time", "entry_price", "exit_time", "exit_price", "exit_reason",
            "duration_bars", "duration_minutes", "points_pnl", "gross_pnl_usd",
            "commission_usd", "net_pnl_usd", "mfe_points", "mfe_usd",
            "mae_points", "mae_usd", "r_multiple", "account_equity_after",
            "drawdown_at_exit_pct"
        ]

        with open(p, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for t in trades:
                row = {k: getattr(t, k, "") for k in fieldnames}
                # Format enums and datetimes
                if hasattr(row["side"], "value"):
                    row["side"] = row["side"].value
                writer.writerow(row)

    @staticmethod
    def export_summary_json(stats: PerformanceStats, file_path: str):
        """Exports statistical performance dictionary to JSON."""
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        with open(p, mode="w") as f:
            json.dump(asdict(stats), f, indent=2)
