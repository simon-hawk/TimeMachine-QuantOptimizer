"""
End-to-End Walkthrough Example for TimeMachine QuantOptimizer.
Runs an automated demonstration of strategy execution, metrics evaluation,
and Monte Carlo stress analysis in pure Python.
"""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from mnq_backtester.data.loader import DataLoader
from mnq_backtester.strategies.library import get_strategy_suite, PINE_STRATEGY_SUITE
from mnq_backtester.pinescript.parser import PineScriptRunner
from mnq_backtester.analytics.metrics import PerformanceMetrics
from mnq_backtester.monte_carlo.engine import MonteCarloEngine, ResamplingMethod

def main():
    print("=" * 70)
    print(" TIMEMACHINE QUANTOPTIMIZER - QUICKSTART DEMO")
    print("=" * 70)

    # 1. Inspect strategies
    suite = get_strategy_suite()
    print(f"\n[1] Strategy Library: {len(suite)} open-source algorithms available.")
    for s in suite[:5]:
        print(f"    - {s['name']} ({s['category']})")

    # 2. Load and construct bars
    print("\n[2] Loading historical market data (MNQ Continuous Futures)...")
    daily = DataLoader.fetch_yahoo_daily(symbol="MNQ=F", days=60)
    print(f"    Loaded {len(daily)} daily session candles.")

    intraday_bars = []
    for candle in daily[-10:]:
        intraday_bars.extend(DataLoader.reconstruct_1m_bars(candle))
    print(f"    Reconstructed {len(intraday_bars)} 1-minute microstructure bars.")

    # 3. Execute backtest
    chosen_id = next(iter(PINE_STRATEGY_SUITE))
    code = PINE_STRATEGY_SUITE[chosen_id]["code"]
    print(f"\n[3] Simulating strategy: '{chosen_id}'...")
    result, strat = PineScriptRunner.run_pine_code(code, intraday_bars)
    print(f"    Processed bars: {result.total_bars_processed}")
    print(f"    Total simulated trades: {len(result.trades)}")

    # 4. Metrics
    stats = PerformanceMetrics.calculate(
        trades=result.trades,
        equity_curve=result.equity_curve,
        initial_capital=50000.0
    )
    print("\n[4] Performance Analytics:")
    print(f"    Ending Capital:      ${stats.final_equity:,.2f}")
    print(f"    Total Net Profit:    ${stats.net_profit_usd:,.2f}")
    print(f"    Win Rate:            {stats.win_rate:.2f}%")
    print(f"    Max Drawdown Pct:    {stats.max_drawdown_pct:.2f}%")
    print(f"    Sharpe Ratio:        {stats.sharpe_ratio:.2f}")
    print(f"    Profit Factor:       {stats.profit_factor:.2f}")

    # 5. Monte Carlo stress test
    print("\n[5] Monte Carlo Resampling (500 iterations)...")
    mc = MonteCarloEngine(iterations=500, seed=42)
    mc_report = mc.run(result.trades, initial_capital=50000.0, method=ResamplingMethod.IID_BOOTSTRAP)
    dist = mc_report.distribution
    print(f"    5th  Percentile (Conservative): ${dist.equity_p5:,.2f}")
    print(f"    50th Percentile (Median):       ${dist.equity_p50:,.2f}")
    print(f"    95th Percentile (Optimistic):   ${dist.equity_p95:,.2f}")
    print(f"    Worst-Case 95% Max DD:          ${dist.max_dd_p95:,.2f}")
    print("\nSimulation completed successfully!")

if __name__ == "__main__":
    main()
