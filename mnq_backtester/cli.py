"""
Command-Line Interface (CLI) for MNQ Advanced Backtesting and Stress Engine.
"""

import sys
import os
from pathlib import Path
import argparse
import logging
from datetime import datetime, timedelta

# Ensure workspace root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mnq_backtester.config import BacktestConfig
from mnq_backtester.data.loader import DataLoader
from mnq_backtester.data.validator import DataValidator
from mnq_backtester.data.sessions import SessionManager
from mnq_backtester.partitions.splitters import DataSplitter
from mnq_backtester.partitions.crisis_regimes import CrisisRegimeLibrary
from mnq_backtester.core.engine import BacktestEngine
from mnq_backtester.strategies.example_orb import OpeningRangeBreakoutStrategy
from mnq_backtester.strategies.example_fvg import FairValueGapStrategy
from mnq_backtester.strategies.example_boswaves import BOSWavesGravityStrategy
from mnq_backtester.analytics.metrics import PerformanceMetrics
from mnq_backtester.analytics.detailed_trades import DetailedTradeAnalyzer
from mnq_backtester.analytics.timeframes import MultiTimeframeAnalyzer
from mnq_backtester.analytics.risk import RiskMetricsCalculator
from mnq_backtester.monte_carlo.engine import MonteCarloEngine, ResamplingMethod
from mnq_backtester.reports.visualizer import ReportVisualizer
from mnq_backtester.reports.export import ReportExporter

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("MNQ_CLI")

def parse_args():
    parser = argparse.ArgumentParser(description="MNQ Advanced Backtesting & Stress-Testing Engine")
    parser.add_argument("--strategy", type=str, choices=["orb", "fvg", "boswaves"], default="orb", help="Strategy to evaluate")
    parser.add_argument("--days", type=int, default=365, help="Number of historical days to simulate")
    parser.add_argument("--capital", type=float, default=50_000.0, help="Initial account capital in USD")
    parser.add_argument("--split", type=str, choices=["none", "train_test", "train_val_test"], default="train_test", help="Sample separation mode")
    parser.add_argument("--regime", type=str, default="none", help="Crisis regime filter (e.g., COVID_CRASH_2020, FED_RATE_HIKE_BEAR_2022, GFC_2008)")
    parser.add_argument("--monte-carlo", type=int, default=1000, help="Number of Monte Carlo iterations (0 to skip)")
    parser.add_argument("--export-csv", type=str, default="", help="Path to export trade log CSV")
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"Initializing MNQ Backtest for Strategy: {args.strategy.upper()} ({args.days} Days)...")

    # 1. Load Data
    daily_candles = DataLoader.fetch_yahoo_daily("MNQ=F", days=args.days)
    logger.info(f"Loaded {len(daily_candles)} daily candles. Reconstructing 1m intraday bars...")

    all_1m_bars = []
    for day_c in daily_candles:
        day_bars = DataLoader.reconstruct_1m_bars(day_c)
        all_1m_bars.extend(day_bars)

    logger.info(f"Total reconstructed bars: {len(all_1m_bars):,}")

    # 2. Verify Data Quality
    validator = DataValidator(tick_size=0.25)
    val_report = validator.validate_bars(all_1m_bars)
    logger.info(f"Data Quality Status: {'PASSED' if val_report.is_valid else 'FLAGGED'} ({val_report.valid_percentage:.2f}% clean)")

    # 3. Crisis Regime Isolation (if requested)
    if args.regime.lower() != "none":
        logger.info(f"Filtering dataset for Crisis Regime: {args.regime.upper()}...")
        all_1m_bars = CrisisRegimeLibrary.slice_by_regime(all_1m_bars, args.regime)
        logger.info(f"Crisis slice extracted: {len(all_1m_bars):,} bars remaining.")

    # 4. Sample Separation (In-Sample / Out-of-Sample)
    if args.split == "train_test":
        split_res = DataSplitter.train_test_split(all_1m_bars, train_ratio=0.70, embargo_bars=30)
        logger.info(f"\n{split_res.summary()}\n")
        # By default, evaluate on Out-of-Sample (Test) bars
        eval_bars = split_res.test_bars
        eval_label = "OUT-OF-SAMPLE (TEST)"
    else:
        eval_bars = all_1m_bars
        eval_label = "FULL DATASET"

    if not eval_bars:
        logger.warning("No evaluation bars available after partition filtering. Using full dataset.")
        eval_bars = all_1m_bars
        eval_label = "FULL DATASET"

    logger.info(f"Running simulation on {len(eval_bars):,} bars [{eval_label}]...")

    # 5. Instantiate Strategy
    if args.strategy == "orb":
        strat = OpeningRangeBreakoutStrategy(orb_minutes=15, risk_reward=2.0)
    elif args.strategy == "fvg":
        strat = FairValueGapStrategy(min_fvg_points=3.0, risk_reward=2.0)
    elif args.strategy == "boswaves":
        strat = BOSWavesGravityStrategy(lookback_swing=10, risk_reward=2.0)
    else:
        strat = OpeningRangeBreakoutStrategy()

    # 6. Execute Backtest
    config = BacktestConfig(initial_capital=args.capital)
    engine = BacktestEngine(config=config)
    result = engine.run(eval_bars, strategy_fn=strat.on_bar)

    # 7. Compute Analytics
    stats = PerformanceMetrics.calculate(result.trades, result.equity_curve, initial_capital=args.capital)
    trade_dist = DetailedTradeAnalyzer.analyze(result.trades)
    tf_report = MultiTimeframeAnalyzer.analyze(result.trades, result.equity_curve, initial_capital=args.capital)
    var_report = RiskMetricsCalculator.calculate_var_cvar(result.trades, portfolio_value=args.capital)

    # 8. Monte Carlo Simulation
    mc_report = None
    if args.monte_carlo > 0 and result.trades:
        logger.info(f"Running {args.monte_carlo:,} Monte Carlo Permutations...")
        mc_engine = MonteCarloEngine(iterations=args.monte_carlo)
        mc_report = mc_engine.run(result.trades, initial_capital=args.capital, method=ResamplingMethod.SHUFFLE)

    # 9. Output Report
    tearsheet = ReportVisualizer.print_full_tearsheet(
        result=result,
        stats=stats,
        trade_dist=trade_dist,
        tf_report=tf_report,
        var_report=var_report,
        mc_report=mc_report
    )
    print("\n" + tearsheet + "\n")

    # 10. Export
    if args.export_csv:
        ReportExporter.export_trades_to_csv(result.trades, args.export_csv)
        logger.info(f"Trade log exported to: {args.export_csv}")

if __name__ == "__main__":
    main()
