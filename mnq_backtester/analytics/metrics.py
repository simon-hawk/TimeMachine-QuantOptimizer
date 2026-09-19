"""
Statistical Performance Metrics Suite for Futures & Quant Strategies.
Computes Winrate, Profit Factor, Expectancy, Max Drawdown, Sharpe, Sortino, Calmar, and Kelly sizing.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import math
import numpy as np
from ..core.order import TradeRecord

@dataclass
class PerformanceStats:
    # Basic Trade Counts
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate: float
    loss_rate: float
    
    # Dollar P&L
    initial_capital: float
    final_equity: float
    net_profit_usd: float
    total_return_pct: float
    gross_profit_usd: float
    gross_loss_usd: float
    profit_factor: float
    payoff_ratio: float
    expectancy_usd_per_trade: float
    expectancy_points_per_trade: float
    
    # Trade Averages
    average_win_usd: float
    average_loss_usd: float
    largest_win_usd: float
    largest_loss_usd: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    
    # Drawdown & Risk
    max_drawdown_usd: float
    max_drawdown_pct: float
    recovery_factor: float
    annualized_return_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Sizing
    full_kelly_fraction: float
    half_kelly_fraction: float

    def summary_table(self) -> str:
        return (
            f"┌─────────────────────────────────────────────────────────────┐\n"
            f"│                 PERFORMANCE METRICS SUMMARY                 │\n"
            f"├─────────────────────────────────────────────────────────────┤\n"
            f"│ Total Trades:             {self.total_trades:<8} Win Rate:             {self.win_rate:>6.2f}% │\n"
            f"│ Winning Trades:           {self.winning_trades:<8} Losing Trades:        {self.losing_trades:>7} │\n"
            f"│ Net Profit (USD):         ${self.net_profit_usd:<10.2f} Return:               {self.total_return_pct:>6.2f}% │\n"
            f"│ Profit Factor:            {self.profit_factor:<8.2f} Payoff Ratio:         {self.payoff_ratio:>7.2f} │\n"
            f"│ Expectancy ($/Trade):     ${self.expectancy_usd_per_trade:<10.2f} Expectancy (pts):     {self.expectancy_points_per_trade:>6.2f}p │\n"
            f"│ Avg Win:                  ${self.average_win_usd:<10.2f} Avg Loss:             ${self.average_loss_usd:>7.2f} │\n"
            f"│ Max Consecutive Wins:     {self.max_consecutive_wins:<8} Max Consec Losses:    {self.max_consecutive_losses:>7} │\n"
            f"├─────────────────────────────────────────────────────────────┤\n"
            f"│ Max Drawdown ($):         ${self.max_drawdown_usd:<10.2f} Max Drawdown (%):     {self.max_drawdown_pct:>6.2f}% │\n"
            f"│ Recovery Factor:          {self.recovery_factor:<8.2f} Annualized Return:    {self.annualized_return_pct:>6.2f}% │\n"
            f"│ Sharpe Ratio:             {self.sharpe_ratio:<8.2f} Sortino Ratio:        {self.sortino_ratio:>7.2f} │\n"
            f"│ Calmar Ratio:             {self.calmar_ratio:<8.2f} Half Kelly:           {self.half_kelly_fraction:>7.2f} │\n"
            f"└─────────────────────────────────────────────────────────────┘"
        )


class PerformanceMetrics:
    """
    Calculates comprehensive quantitative performance and risk metrics from trade history and equity curve.
    """

    @classmethod
    def calculate(
        cls,
        trades: List[TradeRecord],
        equity_curve: List[Dict[str, Any]],
        initial_capital: float = 50_000.0,
        risk_free_rate: float = 0.04
    ) -> PerformanceStats:
        """
        Computes the complete statistical suite from trades and bar equity.
        """
        total_trades = len(trades)
        if total_trades == 0:
            return PerformanceStats(
                total_trades=0, winning_trades=0, losing_trades=0, breakeven_trades=0,
                win_rate=0.0, loss_rate=0.0, initial_capital=initial_capital,
                final_equity=initial_capital, net_profit_usd=0.0, total_return_pct=0.0,
                gross_profit_usd=0.0, gross_loss_usd=0.0, profit_factor=0.0,
                payoff_ratio=0.0, expectancy_usd_per_trade=0.0, expectancy_points_per_trade=0.0,
                average_win_usd=0.0, average_loss_usd=0.0, largest_win_usd=0.0, largest_loss_usd=0.0,
                max_consecutive_wins=0, max_consecutive_losses=0, max_drawdown_usd=0.0,
                max_drawdown_pct=0.0, recovery_factor=0.0, annualized_return_pct=0.0,
                annualized_volatility_pct=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
                calmar_ratio=0.0, full_kelly_fraction=0.0, half_kelly_fraction=0.0
            )

        pnls = [t.net_pnl_usd for t in trades]
        points = [t.points_pnl for t in trades]

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        breakevens = [p for p in pnls if p == 0]

        winning_trades = len(wins)
        losing_trades = len(losses)
        breakeven_trades = len(breakevens)

        win_rate = (winning_trades / total_trades) * 100.0
        loss_rate = (losing_trades / total_trades) * 100.0

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        net_profit = sum(pnls)
        final_equity = initial_capital + net_profit
        total_return_pct = (net_profit / initial_capital) * 100.0

        profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-6 else (99.9 if gross_profit > 0 else 0.0)

        avg_win = (sum(wins) / winning_trades) if winning_trades > 0 else 0.0
        avg_loss = (abs(sum(losses)) / losing_trades) if losing_trades > 0 else 0.0
        payoff_ratio = (avg_win / avg_loss) if avg_loss > 1e-6 else 0.0

        expectancy_usd = net_profit / total_trades
        expectancy_pts = sum(points) / total_trades

        largest_win = max(wins) if wins else 0.0
        largest_loss = min(losses) if losses else 0.0

        # Consecutive streaks
        max_c_wins, max_c_losses = 0, 0
        cur_w, cur_l = 0, 0
        for p in pnls:
            if p > 0:
                cur_w += 1
                cur_l = 0
                max_c_wins = max(max_c_wins, cur_w)
            elif p < 0:
                cur_l += 1
                cur_w = 0
                max_c_losses = max(max_c_losses, cur_l)
            else:
                cur_w = 0
                cur_l = 0

        # Drawdown calculation from equity curve
        eq_values = [p["equity"] for p in equity_curve] if equity_curve else [initial_capital, final_equity]
        peak = eq_values[0]
        max_dd_usd = 0.0
        max_dd_pct = 0.0
        for eq in eq_values:
            if eq > peak:
                peak = eq
            dd_u = peak - eq
            dd_p = (dd_u / peak * 100.0) if peak > 0 else 0.0
            if dd_u > max_dd_usd:
                max_dd_usd = dd_u
            if dd_p > max_dd_pct:
                max_dd_pct = dd_p

        recovery_factor = (net_profit / max_dd_usd) if max_dd_usd > 1e-6 else (99.9 if net_profit > 0 else 0.0)

        # Extract true daily closing equity values
        daily_equity_map = {}
        for p in equity_curve:
            ts = p.get("timestamp")
            d_key = ts.date() if hasattr(ts, "date") else str(ts)[:10]
            daily_equity_map[d_key] = p["equity"]

        daily_eq_values = list(daily_equity_map.values()) if daily_equity_map else [initial_capital, final_equity]
        daily_returns = []
        if len(daily_eq_values) > 1:
            for i in range(1, len(daily_eq_values)):
                prev = daily_eq_values[i - 1]
                if prev > 0:
                    daily_returns.append((daily_eq_values[i] - prev) / prev)
                else:
                    daily_returns.append(0.0)

        n_days = max(1, len(daily_equity_map))
        years = max(1 / 252.0, n_days / 252.0)
        
        ann_return = ((final_equity / initial_capital) ** (1.0 / years) - 1.0) * 100.0 if final_equity > 0 else -100.0
        
        sharpe = 0.0
        sortino = 0.0
        ann_vol = 0.0
        
        if daily_returns:
            r_arr = np.array(daily_returns)
            std = np.std(r_arr)
            mean_r = np.mean(r_arr)
            ann_factor = math.sqrt(252)
            ann_vol = float(std * ann_factor * 100.0)
            
            daily_rf = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0
            excess_return = mean_r - daily_rf
            
            if std > 1e-8:
                sharpe = float((excess_return / std) * ann_factor)
                
            downside = r_arr[r_arr < daily_rf] - daily_rf
            if len(downside) > 0:
                downside_std = math.sqrt(np.mean(downside ** 2))
                if downside_std > 1e-8:
                    sortino = float((excess_return / downside_std) * ann_factor)

        calmar = (ann_return / max_dd_pct) if max_dd_pct > 1e-6 else 0.0

        # Kelly Criterion: f* = p - (q / b) = p - (1-p)/b
        # p = win probability, b = payoff ratio
        p_win = win_rate / 100.0
        p_loss = 1.0 - p_win
        if payoff_ratio > 1e-6:
            full_kelly = max(0.0, p_win - (p_loss / payoff_ratio))
        else:
            full_kelly = 0.0
        half_kelly = full_kelly * 0.5

        return PerformanceStats(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            breakeven_trades=breakeven_trades,
            win_rate=round(win_rate, 2),
            loss_rate=round(loss_rate, 2),
            initial_capital=round(initial_capital, 2),
            final_equity=round(final_equity, 2),
            net_profit_usd=round(net_profit, 2),
            total_return_pct=round(total_return_pct, 2),
            gross_profit_usd=round(gross_profit, 2),
            gross_loss_usd=round(gross_loss, 2),
            profit_factor=round(profit_factor, 2),
            payoff_ratio=round(payoff_ratio, 2),
            expectancy_usd_per_trade=round(expectancy_usd, 2),
            expectancy_points_per_trade=round(expectancy_pts, 2),
            average_win_usd=round(avg_win, 2),
            average_loss_usd=round(avg_loss, 2),
            largest_win_usd=round(largest_win, 2),
            largest_loss_usd=round(largest_loss, 2),
            max_consecutive_wins=max_c_wins,
            max_consecutive_losses=max_c_losses,
            max_drawdown_usd=round(max_dd_usd, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            recovery_factor=round(recovery_factor, 2),
            annualized_return_pct=round(ann_return, 2),
            annualized_volatility_pct=round(ann_vol, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            calmar_ratio=round(calmar, 2),
            full_kelly_fraction=round(full_kelly, 3),
            half_kelly_fraction=round(half_kelly, 3)
        )
