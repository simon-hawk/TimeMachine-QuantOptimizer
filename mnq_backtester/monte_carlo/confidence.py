"""
Monte Carlo Confidence Intervals and Distribution Analytics.
Computes 5th, 25th, 50th, 75th, 95th percentile equity trajectories and drawdown probabilities.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

@dataclass
class MonteCarloDistribution:
    iterations: int
    initial_capital: float
    
    # Final Equity Percentiles
    equity_p5: float
    equity_p25: float
    equity_p50: float # Median
    equity_p75: float
    equity_p95: float
    
    # Max Drawdown Percentiles ($)
    max_dd_p5: float
    max_dd_p25: float
    max_dd_p50: float
    max_dd_p75: float
    max_dd_p95: float # 95% worst-case max drawdown
    
    # Max Drawdown Percentiles (%)
    max_dd_pct_p50: float
    max_dd_pct_p95: float
    
    # Ruin / Rule Breach Probability
    drawdown_breach_threshold: float
    drawdown_breach_probability_pct: float
    profit_target_probability_pct: float

    def summary_table(self) -> str:
        return (
            f"┌─────────────────────────────────────────────────────────────┐\n"
            f"│            MONTE CARLO SIMULATION ({self.iterations:,} RUNS)            │\n"
            f"├──────────────────────┬───────────────┬──────────────────────┤\n"
            f"│ Percentile           │ Final Equity  │ Max Drawdown ($)     │\n"
            f"├──────────────────────┼───────────────┼──────────────────────┤\n"
            f"│ 5th (Conservative)   │ ${self.equity_p5:<13.2f}│ ${self.max_dd_p95:<20.2f}│\n"
            f"│ 25th Percentile      │ ${self.equity_p25:<13.2f}│ ${self.max_dd_p75:<20.2f}│\n"
            f"│ 50th (Median)        │ ${self.equity_p50:<13.2f}│ ${self.max_dd_p50:<20.2f}│\n"
            f"│ 75th Percentile      │ ${self.equity_p75:<13.2f}│ ${self.max_dd_p25:<20.2f}│\n"
            f"│ 95th (Optimistic)    │ ${self.equity_p95:<13.2f}│ ${self.max_dd_p5:<20.2f}│\n"
            f"├──────────────────────┴───────────────┴──────────────────────┤\n"
            f"│ Prob of Trailing DD Breaching ${self.drawdown_breach_threshold:,.0f}: {self.drawdown_breach_probability_pct:>7.2f}%       │\n"
            f"│ Prob of Hitting Profit Target ($3,000):  {self.profit_target_probability_pct:>7.2f}%       │\n"
            f"└─────────────────────────────────────────────────────────────┘"
        )


class MonteCarloConfidence:
    """
    Processes matrices of Monte Carlo simulated paths into percentile distributions.
    """

    @staticmethod
    def evaluate_paths(
        simulated_equity_curves: List[List[float]],
        simulated_max_dds: List[float],
        initial_capital: float = 50_000.0,
        dd_threshold: float = 2_500.0,
        profit_target: float = 3_000.0
    ) -> MonteCarloDistribution:
        n_runs = len(simulated_equity_curves)
        if n_runs == 0:
            return MonteCarloDistribution(
                iterations=0, initial_capital=initial_capital,
                equity_p5=initial_capital, equity_p25=initial_capital,
                equity_p50=initial_capital, equity_p75=initial_capital,
                equity_p95=initial_capital, max_dd_p5=0.0, max_dd_p25=0.0,
                max_dd_p50=0.0, max_dd_p75=0.0, max_dd_p95=0.0,
                max_dd_pct_p50=0.0, max_dd_pct_p95=0.0,
                drawdown_breach_threshold=dd_threshold,
                drawdown_breach_probability_pct=0.0,
                profit_target_probability_pct=0.0
            )

        final_equities = np.array([curve[-1] for curve in simulated_equity_curves])
        max_dds = np.array(simulated_max_dds)

        # Calculate percentiles
        eq_p5 = float(np.percentile(final_equities, 5))
        eq_p25 = float(np.percentile(final_equities, 25))
        eq_p50 = float(np.percentile(final_equities, 50))
        eq_p75 = float(np.percentile(final_equities, 75))
        eq_p95 = float(np.percentile(final_equities, 95))

        dd_p5 = float(np.percentile(max_dds, 5))
        dd_p25 = float(np.percentile(max_dds, 25))
        dd_p50 = float(np.percentile(max_dds, 50))
        dd_p75 = float(np.percentile(max_dds, 75))
        dd_p95 = float(np.percentile(max_dds, 95))

        dd_pct_p50 = (dd_p50 / initial_capital) * 100.0
        dd_pct_p95 = (dd_p95 / initial_capital) * 100.0

        # Breaching probability
        breaches = np.sum(max_dds >= dd_threshold)
        breach_prob = (breaches / n_runs) * 100.0

        # Profit target probability
        target_hits = np.sum(final_equities >= (initial_capital + profit_target))
        target_prob = (target_hits / n_runs) * 100.0

        return MonteCarloDistribution(
            iterations=n_runs,
            initial_capital=initial_capital,
            equity_p5=round(eq_p5, 2),
            equity_p25=round(eq_p25, 2),
            equity_p50=round(eq_p50, 2),
            equity_p75=round(eq_p75, 2),
            equity_p95=round(eq_p95, 2),
            max_dd_p5=round(dd_p5, 2),
            max_dd_p25=round(dd_p25, 2),
            max_dd_p50=round(dd_p50, 2),
            max_dd_p75=round(dd_p75, 2),
            max_dd_p95=round(dd_p95, 2),
            max_dd_pct_p50=round(dd_pct_p50, 2),
            max_dd_pct_p95=round(dd_pct_p95, 2),
            drawdown_breach_threshold=dd_threshold,
            drawdown_breach_probability_pct=round(breach_prob, 2),
            profit_target_probability_pct=round(target_prob, 2)
        )
