"""
Risk Metrics and Tail Risk Analysis for MNQ Portfolios.
Implements Historical Value at Risk (VaR), Conditional VaR (Expected Shortfall), and Risk of Ruin.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import math
import numpy as np
from ..core.order import TradeRecord

@dataclass
class VaRReport:
    confidence_level: float
    var_dollar: float
    var_pct: float
    cvar_dollar: float
    cvar_pct: float
    risk_of_ruin_pct: float

    def summary(self) -> str:
        return (
            f"=== Tail Risk & Ruin Analysis ({self.confidence_level*100:.0f}% Confidence) ===\n"
            f"Value at Risk (VaR):        ${self.var_dollar:.2f} ({self.var_pct:.2f}% of portfolio)\n"
            f"Expected Shortfall (CVaR):  ${self.cvar_dollar:.2f} ({self.cvar_pct:.2f}% of portfolio)\n"
            f"Estimated Risk of Ruin:     {self.risk_of_ruin_pct:.2f}%"
        )


class RiskMetricsCalculator:
    """
    Computes tail-risk distributions, expected shortfall, and ruin probability.
    """

    @classmethod
    def calculate_var_cvar(
        cls,
        trades: List[TradeRecord],
        portfolio_value: float = 50_000.0,
        confidence: float = 0.95,
        ruin_loss_threshold: float = 2_500.0
    ) -> VaRReport:
        if not trades:
            return VaRReport(
                confidence_level=confidence,
                var_dollar=0.0, var_pct=0.0,
                cvar_dollar=0.0, cvar_pct=0.0,
                risk_of_ruin_pct=0.0
            )

        returns = np.array([t.net_pnl_usd / portfolio_value for t in trades])
        sorted_returns = np.sort(returns)

        # Index corresponding to tail
        tail_idx = max(0, int((1.0 - confidence) * len(sorted_returns)))
        
        var_pct = abs(float(sorted_returns[tail_idx])) if sorted_returns[tail_idx] < 0 else 0.0
        var_dollar = var_pct * portfolio_value

        tail = sorted_returns[:max(1, tail_idx + 1)]
        cvar_pct = abs(float(np.mean(tail))) if np.mean(tail) < 0 else 0.0
        cvar_dollar = cvar_pct * portfolio_value

        # Perry/Nauck Risk of Ruin approximation for finite capital:
        # P(Ruin) = ((1 - A) / (1 + A))^U where A = Expected PnL / Volatility, U = ruin units
        pnls = np.array([t.net_pnl_usd for t in trades])
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)

        if mean_pnl <= 0:
            ruin_pct = 100.0
        elif std_pnl < 1e-4:
            ruin_pct = 0.0
        else:
            # Number of risk units to ruin
            units = ruin_loss_threshold / std_pnl
            edge = mean_pnl / std_pnl
            # Classic Gambler's Ruin / Bachelier formula
            ruin_prob = math.exp(-2.0 * edge * units)
            ruin_pct = min(100.0, max(0.0, ruin_prob * 100.0))

        return VaRReport(
            confidence_level=confidence,
            var_dollar=round(var_dollar, 2),
            var_pct=round(var_pct * 100.0, 2),
            cvar_dollar=round(cvar_dollar, 2),
            cvar_pct=round(cvar_pct * 100.0, 2),
            risk_of_ruin_pct=round(ruin_pct, 2)
        )
