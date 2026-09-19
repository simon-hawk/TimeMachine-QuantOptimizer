"""
High-Precision Event-Driven Backtesting Engine with 40-Tick Sub-Bar Detailization.
Executes bar-by-bar, queues signals for next-bar open fills (TradingView parity),
and simulates 40-tick auction paths intra-bar for realistic Stop Loss, Take Profit,
and Trailing Stop execution without lookahead bias.
"""

import math
import logging
from datetime import datetime, time, timedelta
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field

from ..config import BacktestConfig, DEFAULT_CONFIG
from .order import Order, OrderSide, OrderType, OrderStatus, Position, TradeRecord
from .execution import ExecutionSimulator
from .micro_ticks import TickDetailSimulator, MicroTick

logger = logging.getLogger("BacktestEngine")

@dataclass
class BacktestResult:
    """Complete output packet from backtest execution."""
    config: BacktestConfig
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    total_bars_processed: int = 0
    halted_early: bool = False
    halt_reason: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

class BacktestEngine:
    """
    Institutional Futures Backtester with 40-Tick intra-bar microstructure resolution.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.execution = ExecutionSimulator(self.config)
        self.reset()

    def reset(self):
        """Resets engine state for a clean simulation run."""
        self.current_equity = self.config.initial_capital
        self.peak_equity = self.config.initial_capital
        self.daily_start_equity = self.config.initial_capital
        self.current_day_date = None
        self.active_position: Optional[Position] = None
        self.pending_order: Optional[Order] = None
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self.is_halted = False
        self.halt_reason = ""
        self.trade_counter = 0

    def run(
        self,
        bars: List[Dict[str, Any]],
        strategy_fn: Callable[[int, List[Dict[str, Any]], Optional[Position]], Optional[Order]]
    ) -> BacktestResult:
        self.reset()
        if not bars:
            return BacktestResult(config=self.config)

        point_val = self.config.contract.point_value # $2.00 / pt
        max_dd_limit = self.config.max_trailing_drawdown
        daily_loss_limit = self.config.daily_loss_limit

        for idx, bar in enumerate(bars):
            if self.is_halted:
                break

            raw_time = bar.get("timestamp") or bar.get("time_ny") or bar.get("date")
            if isinstance(raw_time, datetime):
                bar_time = raw_time
            else:
                try:
                    bar_time = datetime.fromisoformat(str(raw_time))
                except Exception:
                    bar_time = datetime.now()

            bar_date = bar_time.date()
            o = float(bar["open"])
            h = float(bar["high"])
            l = float(bar["low"])
            c = float(bar["close"])

            # 1. Day boundary reset for daily loss limit
            if self.current_day_date != bar_date:
                self.current_day_date = bar_date
                self.daily_start_equity = self.current_equity

            # 2. Fill Pending Order from previous bar close at this bar's Open (TradingView parity)
            if self.pending_order is not None and self.active_position is None:
                self._fill_pending_order(self.pending_order, o, bar_time, idx)
                self.pending_order = None

            # 3. 40-Tick Intra-Bar Microstructure Resolution for Active Positions
            if self.active_position is not None:
                sub_ticks = TickDetailSimulator.generate_sub_ticks(bar, ticks_per_bar=40, seed=idx)
                exited = False

                for tick in sub_ticks:
                    tp = tick.price
                    self.active_position.update_excursions(tp, tp)

                    # EOD Liquidation check
                    if not self.config.allow_overnight_holds:
                        if tick.timestamp.time() >= self.config.eod_liquidation_time:
                            slip = self.execution.calculate_slippage(self.active_position.side, is_entry=False)
                            exit_price = self.execution.quantize_price(tp + slip)
                            self._close_position(self.active_position, exit_price, tick.timestamp, "EOD_LIQUIDATION", idx)
                            exited = True
                            break

                    # Check Stop Loss
                    sl = self.active_position.current_stop_loss
                    tp_target = self.active_position.take_profit

                    if self.active_position.side == OrderSide.BUY:
                        # Stop loss triggered
                        if sl is not None and tp <= sl:
                            slip = self.execution.calculate_slippage(OrderSide.BUY, is_entry=False)
                            exit_price = self.execution.quantize_price(sl + slip)
                            self._close_position(self.active_position, exit_price, tick.timestamp, "STOP_LOSS", idx)
                            exited = True
                            break
                        # Take profit triggered
                        elif tp_target is not None and tp >= tp_target:
                            exit_price = self.execution.quantize_price(tp_target)
                            self._close_position(self.active_position, exit_price, tick.timestamp, "TAKE_PROFIT", idx)
                            exited = True
                            break

                    elif self.active_position.side == OrderSide.SELL:
                        # Stop loss triggered
                        if sl is not None and tp >= sl:
                            slip = self.execution.calculate_slippage(OrderSide.SELL, is_entry=False)
                            exit_price = self.execution.quantize_price(sl + slip)
                            self._close_position(self.active_position, exit_price, tick.timestamp, "STOP_LOSS", idx)
                            exited = True
                            break
                        # Take profit triggered
                        elif tp_target is not None and tp <= tp_target:
                            exit_price = self.execution.quantize_price(tp_target)
                            self._close_position(self.active_position, exit_price, tick.timestamp, "TAKE_PROFIT", idx)
                            exited = True
                            break

            # 4. Prop firm risk rule checks
            current_drawdown = self.peak_equity - self.current_equity
            day_loss = self.daily_start_equity - self.current_equity

            if self.config.enable_prop_firm_rules:
                if current_drawdown >= max_dd_limit:
                    self.is_halted = True
                    self.halt_reason = f"Max Trailing Drawdown Breached (${current_drawdown:.2f} >= ${max_dd_limit:.2f})"
                    break
                if day_loss >= daily_loss_limit:
                    self.is_halted = True
                    self.halt_reason = f"Daily Loss Limit Breached (${day_loss:.2f} >= ${daily_loss_limit:.2f})"
                    break

            # 5. Generate strategy signals at Bar Close (Queued for Next-Bar Open)
            if not self.is_halted and self.active_position is None and self.pending_order is None:
                new_order = strategy_fn(idx, bars, self.active_position)
                if new_order is not None:
                    self.pending_order = new_order

            # 6. Record equity curve snapshot at bar close
            unrealized_pnl = 0.0
            if self.active_position is not None:
                if self.active_position.side == OrderSide.BUY:
                    unrealized_pnl = (c - self.active_position.entry_price) * point_val * self.active_position.contracts
                else:
                    unrealized_pnl = (self.active_position.entry_price - c) * point_val * self.active_position.contracts

            bar_equity = self.current_equity + unrealized_pnl
            if bar_equity > self.peak_equity:
                self.peak_equity = bar_equity

            dd_usd = self.peak_equity - bar_equity
            dd_pct = (dd_usd / self.peak_equity * 100.0) if self.peak_equity > 0 else 0.0

            self.equity_curve.append({
                "timestamp": bar_time,
                "equity": round(bar_equity, 2),
                "drawdown_usd": round(dd_usd, 2),
                "drawdown_pct": round(dd_pct, 2),
                "close_price": c,
                "unrealized_pnl": round(unrealized_pnl, 2)
            })

        return BacktestResult(
            config=self.config,
            trades=self.trades,
            equity_curve=self.equity_curve,
            total_bars_processed=len(bars),
            halted_early=self.is_halted,
            halt_reason=self.halt_reason,
            start_time=bars[0].get("timestamp") if bars else None,
            end_time=bars[-1].get("timestamp") if bars else None
        )

    def _fill_pending_order(self, order: Order, open_price: float, fill_time: datetime, bar_idx: int):
        """Fills a pending order at the bar Open with realistic bid-ask crossing and slippage."""
        fill_price = self.execution.simulate_fill_price(order, open_price, is_entry=True)
        order.status = OrderStatus.FILLED
        order.filled_price = fill_price
        order.filled_at = fill_time

        self.active_position = Position(
            position_id=f"POS_{order.order_id}",
            symbol=order.symbol,
            side=order.side,
            contracts=order.contracts,
            entry_price=fill_price,
            entry_time=fill_time,
            entry_bar_index=bar_idx,
            initial_stop_loss=order.stop_loss_price,
            current_stop_loss=order.stop_loss_price,
            take_profit=order.take_profit_price,
            trail_distance_points=order.trail_distance_points,
            peak_favorable_price=fill_price,
            worst_adverse_price=fill_price,
            strategy_tag=order.strategy_tag
        )

    def _close_position(
        self,
        pos: Position,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
        exit_bar_idx: int
    ):
        """Calculates trade P&L, commissions, excursions, and updates cash equity."""
        point_val = self.config.contract.point_value
        side_mult = 1.0 if pos.side == OrderSide.BUY else -1.0
        
        points_pnl = (exit_price - pos.entry_price) * side_mult
        gross_pnl = points_pnl * point_val * pos.contracts
        
        # Round-trip commission ($0.62 per contract side = $1.24 per round trip)
        total_comm = self.config.commission_per_side * 2 * pos.contracts
        net_pnl = gross_pnl - total_comm

        # Update cash equity
        self.current_equity += net_pnl
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

        # Excursion calculations
        if pos.side == OrderSide.BUY:
            mfe_pts = max(0.0, pos.peak_favorable_price - pos.entry_price)
            mae_pts = max(0.0, pos.entry_price - pos.worst_adverse_price)
        else:
            mfe_pts = max(0.0, pos.entry_price - pos.peak_favorable_price)
            mae_pts = max(0.0, pos.worst_adverse_price - pos.entry_price)

        mfe_usd = mfe_pts * point_val * pos.contracts
        mae_usd = mae_pts * point_val * pos.contracts

        initial_risk_pts = abs(pos.entry_price - (pos.initial_stop_loss or pos.entry_price))
        r_multiple = (points_pnl / initial_risk_pts) if initial_risk_pts > 0 else 0.0

        dur_minutes = (exit_time - pos.entry_time).total_seconds() / 60.0 if isinstance(exit_time, datetime) and isinstance(pos.entry_time, datetime) else 5.0

        self.trade_counter += 1
        record = TradeRecord(
            trade_id=self.trade_counter,
            order_id=pos.position_id,
            symbol=pos.symbol,
            side=pos.side,
            contracts=pos.contracts,
            entry_time=pos.entry_time,
            entry_price=pos.entry_price,
            entry_bar_index=pos.entry_bar_index,
            exit_time=exit_time,
            exit_price=exit_price,
            exit_bar_index=exit_bar_idx,
            exit_reason=exit_reason,
            duration_minutes=round(dur_minutes, 1),
            points_pnl=round(points_pnl, 2),
            gross_pnl_usd=round(gross_pnl, 2),
            commission_usd=round(total_comm, 2),
            net_pnl_usd=round(net_pnl, 2),
            mfe_points=round(mfe_pts, 2),
            mfe_usd=round(mfe_usd, 2),
            mae_points=round(mae_pts, 2),
            mae_usd=round(mae_usd, 2),
            r_multiple=round(r_multiple, 2),
            account_equity_after=round(self.current_equity, 2),
            strategy_tag=pos.strategy_tag
        )

        self.trades.append(record)
        self.active_position = None
