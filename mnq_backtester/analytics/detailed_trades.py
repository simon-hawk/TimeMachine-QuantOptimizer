"""
Detailed Trade Analyzer: Granular trade metrics, MAE/MFE excursions, duration profiles, and side breakdowns.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from ..core.order import TradeRecord, OrderSide

@dataclass
class TradeDistributionStats:
    total_trades: int
    long_trades: int
    short_trades: int
    long_win_rate: float
    short_win_rate: float
    
    # Excursions
    avg_mfe_points: float
    avg_mae_points: float
    avg_mfe_usd: float
    avg_mae_usd: float
    trade_efficiency_pct: float # (Realized Profit / Peak MFE)
    
    # Durations (minutes)
    avg_duration_minutes: float
    avg_win_duration_minutes: float
    avg_loss_duration_minutes: float
    
    # Exit Reason Breakdown
    take_profit_exits: int
    stop_loss_exits: int
    eod_exits: int
    signal_reversal_exits: int

    def summary(self) -> str:
        return (
            f"=== Trade Execution & Excursion Analysis ===\n"
            f"Long Trades:  {self.long_trades} (Win Rate: {self.long_win_rate:.1f}%)\n"
            f"Short Trades: {self.short_trades} (Win Rate: {self.short_win_rate:.1f}%)\n"
            f"Avg MFE (Peak Potential): {self.avg_mfe_points:.2f} pts (${self.avg_mfe_usd:.2f})\n"
            f"Avg MAE (Adverse Drawdown): {self.avg_mae_points:.2f} pts (${self.avg_mae_usd:.2f})\n"
            f"Trade Capture Efficiency: {self.trade_efficiency_pct:.1f}%\n"
            f"Avg Holding Duration: {self.avg_duration_minutes:.1f} mins (Wins: {self.avg_win_duration_minutes:.1f}m, Losses: {self.avg_loss_duration_minutes:.1f}m)\n"
            f"Exits Breakdown: TP={self.take_profit_exits}, SL={self.stop_loss_exits}, EOD={self.eod_exits}, Reversals={self.signal_reversal_exits}"
        )


class DetailedTradeAnalyzer:
    """
    Extracts deep microstructure and excursion metrics from trade logs.
    """

    @classmethod
    def analyze(cls, trades: List[TradeRecord]) -> TradeDistributionStats:
        if not trades:
            return TradeDistributionStats(
                total_trades=0, long_trades=0, short_trades=0, long_win_rate=0.0,
                short_win_rate=0.0, avg_mfe_points=0.0, avg_mae_points=0.0,
                avg_mfe_usd=0.0, avg_mae_usd=0.0, trade_efficiency_pct=0.0,
                avg_duration_minutes=0.0, avg_win_duration_minutes=0.0,
                avg_loss_duration_minutes=0.0, take_profit_exits=0,
                stop_loss_exits=0, eod_exits=0, signal_reversal_exits=0
            )

        longs = [t for t in trades if t.side == OrderSide.BUY]
        shorts = [t for t in trades if t.side == OrderSide.SELL]

        long_wins = [t for t in longs if t.is_win]
        short_wins = [t for t in shorts if t.is_win]

        l_wr = (len(long_wins) / len(longs) * 100.0) if longs else 0.0
        s_wr = (len(short_wins) / len(shorts) * 100.0) if shorts else 0.0

        mfes_pts = [t.mfe_points for t in trades]
        maes_pts = [t.mae_points for t in trades]
        mfes_usd = [t.mfe_usd for t in trades]
        maes_usd = [t.mae_usd for t in trades]

        avg_mfe_p = float(np.mean(mfes_pts)) if mfes_pts else 0.0
        avg_mae_p = float(np.mean(maes_pts)) if maes_pts else 0.0
        avg_mfe_u = float(np.mean(mfes_usd)) if mfes_usd else 0.0
        avg_mae_u = float(np.mean(maes_usd)) if maes_usd else 0.0

        # Efficiency on winning trades: how much of peak MFE was captured at exit
        win_trades = [t for t in trades if t.is_win]
        effs = []
        for wt in win_trades:
            if wt.mfe_points > 0:
                effs.append((wt.points_pnl / wt.mfe_points) * 100.0)
        avg_eff = float(np.mean(effs)) if effs else 0.0

        durations = [t.duration_minutes for t in trades]
        win_durs = [t.duration_minutes for t in trades if t.is_win]
        loss_durs = [t.duration_minutes for t in trades if t.is_loss]

        avg_dur = float(np.mean(durations)) if durations else 0.0
        avg_w_dur = float(np.mean(win_durs)) if win_durs else 0.0
        avg_l_dur = float(np.mean(loss_durs)) if loss_durs else 0.0

        tp_count = sum(1 for t in trades if t.exit_reason == "TAKE_PROFIT")
        sl_count = sum(1 for t in trades if t.exit_reason == "STOP_LOSS")
        eod_count = sum(1 for t in trades if "EOD" in t.exit_reason)
        rev_count = sum(1 for t in trades if t.exit_reason == "SIGNAL_REVERSAL")

        return TradeDistributionStats(
            total_trades=len(trades),
            long_trades=len(longs),
            short_trades=len(shorts),
            long_win_rate=round(l_wr, 2),
            short_win_rate=round(s_wr, 2),
            avg_mfe_points=round(avg_mfe_p, 2),
            avg_mae_points=round(avg_mae_p, 2),
            avg_mfe_usd=round(avg_mfe_u, 2),
            avg_mae_usd=round(avg_mae_u, 2),
            trade_efficiency_pct=round(avg_eff, 2),
            avg_duration_minutes=round(avg_dur, 1),
            avg_win_duration_minutes=round(avg_w_dur, 1),
            avg_loss_duration_minutes=round(avg_l_dur, 1),
            take_profit_exits=tp_count,
            stop_loss_exits=sl_count,
            eod_exits=eod_count,
            signal_reversal_exits=rev_count
        )
