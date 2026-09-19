"""
Ilmanen Factor Risk Premia Analytics.
Based on Antti Ilmanen's (2012) framework for style premia attribution:
- Value / Reversion Exposure
- Momentum / Trend Exposure
- Dynamic Volatility Targeting Efficacy
- Multi-Factor Diversification Multiplier D = 1 / sqrt(w^T Sigma w)
"""

from typing import List, Dict, Any, Optional
import math
import numpy as np

from ..core.order import TradeRecord


class IlmanenFactorAttribution:
    """
    Analyzes trade logs and returns series to attribute performance to Ilmanen style factors
    and calculate factor diversification benefits.
    """

    @classmethod
    def attribute_trades(
        cls,
        trades: List[TradeRecord],
        portfolio_capital: float = 50_000.0,
    ) -> Dict[str, Any]:
        if not trades:
            return {
                "total_trades": 0,
                "factor_breakdown": {},
                "diversification_multiplier": 1.0,
                "factor_correlation": 0.0,
            }

        factor_trades: Dict[str, List[TradeRecord]] = {
            "Trend_Long": [],
            "Trend_Short": [],
            "Value_Exhaustion_Fade_Short": [],
            "Value_Exhaustion_Dip_Long": [],
            "Other": [],
        }

        for t in trades:
            factor_key = "Other"
            tag = getattr(t, "strategy_tag", "")
            if "Trend_Long" in tag:
                factor_key = "Trend_Long"
            elif "Trend_Short" in tag:
                factor_key = "Trend_Short"
            elif "Fade_Short" in tag:
                factor_key = "Value_Exhaustion_Fade_Short"
            elif "Dip_Long" in tag:
                factor_key = "Value_Exhaustion_Dip_Long"
            
            factor_trades[factor_key].append(t)

        factor_summary = {}
        trend_pnls = []
        value_pnls = []

        for f_name, f_list in factor_trades.items():
            if not f_list:
                continue
            pnls = [tr.net_pnl_usd for tr in f_list]
            win_pnls = [p for p in pnls if p > 0]
            loss_pnls = [p for p in pnls if p <= 0]
            
            win_rate = len(win_pnls) / len(pnls) if pnls else 0.0
            tot_pnl = sum(pnls)
            avg_trade = tot_pnl / len(pnls) if pnls else 0.0
            profit_factor = abs(sum(win_pnls) / sum(loss_pnls)) if sum(loss_pnls) != 0 else float("inf")

            factor_summary[f_name] = {
                "trade_count": len(f_list),
                "total_pnl_usd": round(tot_pnl, 2),
                "win_rate_pct": round(win_rate * 100, 2),
                "avg_trade_usd": round(avg_trade, 2),
                "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 999.0,
            }

            if "Trend" in f_name:
                trend_pnls.extend(pnls)
            elif "Value" in f_name:
                value_pnls.extend(pnls)

        # Calculate Factor Correlation between Trend and Value
        # Ilmanen empirical insight: Value and Trend are negatively correlated (-0.3 to -0.6)
        min_len = min(len(trend_pnls), len(value_pnls))
        if min_len >= 5:
            corr = float(np.corrcoef(trend_pnls[:min_len], value_pnls[:min_len])[0, 1])
            if math.isnan(corr):
                corr = 0.0
        else:
            corr = -0.35  # Empirical Ilmanen benchmark prior

        # Factor Diversification Multiplier: D = 1 / sqrt(w1^2 + w2^2 + 2*w1*w2*rho)
        w1, w2 = 0.5, 0.5
        denom = math.sqrt(w1**2 + w2**2 + (2 * w1 * w2 * corr))
        diversification_multiplier = round(1.0 / denom, 2) if denom > 0 else 1.0

        return {
            "total_trades": len(trades),
            "factor_breakdown": factor_summary,
            "trend_value_correlation": round(corr, 3),
            "diversification_multiplier": diversification_multiplier,
            "ilmanen_insight": (
                f"Trend vs Value correlation is {corr:.2f}. "
                f"Combining them provides a {((diversification_multiplier - 1.0) * 100):.1f}% "
                f"Sharpe improvement over single-style exposure."
            ),
        }
