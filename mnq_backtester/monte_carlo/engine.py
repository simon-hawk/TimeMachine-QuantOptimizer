"""
Monte Carlo Simulation Orchestrator.
Executes high-speed stochastic simulation runs across trade logs.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

from ..core.order import TradeRecord
from .resampling import TradeResampler, ResamplingMethod
from .confidence import MonteCarloConfidence, MonteCarloDistribution
from .stress_test import SlippageStressTester, SlippageSensitivityResult

@dataclass
class MonteCarloReport:
    distribution: MonteCarloDistribution
    resampling_method: ResamplingMethod
    slippage_sensitivity: List[SlippageSensitivityResult]

    def summary_table(self) -> str:
        dist_table = self.distribution.summary_table()
        lines = [
            dist_table,
            "",
            "┌─────────────────────────────────────────────────────────────┐",
            "│              SLIPPAGE SENSITIVITY STRESS TEST               │",
            "├────────┬──────────────┬──────────────┬───────────────┬──────┤",
            "│ Ticks  │ Cost / Trade │ Net P&L ($)  │ Profit Factor │ Pass │",
            "├────────┼──────────────┼──────────────┼───────────────┼──────┤"
        ]
        for s in self.slippage_sensitivity:
            pass_str = "YES" if s.is_profitable else "NO"
            lines.append(
                f"│ {s.slippage_ticks:>4.1f}t  │ ${s.slippage_cost_usd_per_trade:>10.2f} │ ${s.net_profit_usd:>10.2f} │ {s.profit_factor:>13.2f} │ {pass_str:^4} │"
            )
        lines.append("└────────┴──────────────┴──────────────┴───────────────┴──────┘")
        return "\n".join(lines)


class MonteCarloEngine:
    """
    Simulates thousands of synthetic trade paths to quantify sequence risk and worst-case drawdowns.
    """

    def __init__(self, iterations: int = 1_000, seed: int = 42):
        self.iterations = iterations
        self.seed = seed

    def run(
        self,
        trades: List[TradeRecord],
        initial_capital: float = 50_000.0,
        method: ResamplingMethod = ResamplingMethod.SHUFFLE,
        dd_threshold: float = 2_500.0,
        profit_target: float = 3_000.0
    ) -> MonteCarloReport:
        if not trades:
            empty_dist = MonteCarloConfidence.evaluate_paths([], [], initial_capital, dd_threshold, profit_target)
            return MonteCarloReport(
                distribution=empty_dist,
                resampling_method=method,
                slippage_sensitivity=[]
            )

        pnls = [t.net_pnl_usd for t in trades]
        n_trades = len(pnls)

        simulated_equity_curves = []
        simulated_max_dds = []

        for i in range(self.iterations):
            run_seed = self.seed + i
            
            if method == ResamplingMethod.SHUFFLE:
                resampled_pnls = TradeResampler.shuffle(pnls, seed=run_seed)
            elif method == ResamplingMethod.IID_BOOTSTRAP:
                resampled_pnls = TradeResampler.iid_bootstrap(pnls, n_samples=n_trades, seed=run_seed)
            elif method == ResamplingMethod.BLOCK_BOOTSTRAP:
                resampled_pnls = TradeResampler.moving_block_bootstrap(pnls, n_samples=n_trades, block_size=5, seed=run_seed)
            elif method == ResamplingMethod.STATIONARY_BOOTSTRAP:
                resampled_pnls = TradeResampler.stationary_bootstrap(pnls, n_samples=n_trades, avg_block_size=5.0, seed=run_seed)
            else:
                resampled_pnls = TradeResampler.shuffle(pnls, seed=run_seed)

            # Build equity curve for this run
            eq = initial_capital
            peak = initial_capital
            max_dd = 0.0
            curve = [eq]

            for pnl in resampled_pnls:
                eq += pnl
                curve.append(eq)
                if eq > peak:
                    peak = eq
                dd = peak - eq
                if dd > max_dd:
                    max_dd = dd

            simulated_equity_curves.append(curve)
            simulated_max_dds.append(max_dd)

        distribution = MonteCarloConfidence.evaluate_paths(
            simulated_equity_curves=simulated_equity_curves,
            simulated_max_dds=simulated_max_dds,
            initial_capital=initial_capital,
            dd_threshold=dd_threshold,
            profit_target=profit_target
        )

        sensitivity = SlippageStressTester.evaluate_sensitivity(
            trades=trades,
            point_value=2.00,
            tick_size=0.25,
            initial_capital=initial_capital
        )

        return MonteCarloReport(
            distribution=distribution,
            resampling_method=method,
            slippage_sensitivity=sensitivity
        )
