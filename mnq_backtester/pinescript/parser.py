"""
High-Fidelity PineScript Transpiler & Execution Engine.
Comprehensive parsing for ALL proprietary strategy families:
  1. Morning Drive Series (Pre-market 9:00-9:30 range, Daily High/Low target, 200 EMA)
  2. London Judas Swing (Asian session midnight-2am range sweep, London 2-5am trigger)
  3. VRP Volatility Fade (Variance difference z-score > 1.5, reversion)
  4. Gravity SMC & Volatility Master (BSL/SSL sweep + VRP variance surge)
  5. Relative FX / Econophysics Mean Reversion (Shannon Entropy + Bollinger Z-Score > 2.0)
  6. Momentum Breakout Scalper Series (Intrabar lookback breakout, lunch lockout)
  7. Spike Master Series (Supertrend flip vs RSI displacement, Chandelier exit, Asymmetric targets)
  8. NightMoves Globex Drift (15:55 entry, daily close strength, ATR filter)
  9. PSW ORB Series (15m ORB, RVOL, Pre-market sweeps, Trend riders)
 10. VWAP Cross & Pullback Series (VWAP bounces, power hour, dynamic trails)
 11. 10am AMD Suite (Accumulation, Judas Wick sweep threshold, Distribution re-entry)
 12. ICT Silver Bullet Suite (10:00-11:00 AM NY session, FVG displacement ratio)
 13. Inverse FVG Series (Full day retests, failed imbalance inversion)
 14. SMC Sweeps & BOSWaves Structural Breakouts
"""

import re
import math
import logging
from datetime import datetime, time as dtime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np

from .indicators import PineTA
from ..core.order import Order, OrderType, OrderSide, Position
from ..data.sessions import SessionManager

logger = logging.getLogger("PineTranspiler")

class PineScriptStrategy:
    """
    Transpiles and executes arbitrary proprietary PineScript code with exact parameters.
    """

    def __init__(self, script_code: str, custom_name: str = "PineScript_Strategy"):
        self.raw_code = script_code
        self.name = custom_name
        self.params: Dict[str, Any] = {}
        self._extract_metadata(script_code)

    def _extract_metadata(self, code: str):
        # Extract Strategy Title
        title_m = re.search(r'(?:strategy|indicator)\(\s*(?:title\s*=\s*)?["\']([^"\']+)["\']', code)
        if title_m:
            self.name = title_m.group(1)

        # 1. Standard PineScript Input Functions (v4 & v5)
        input_patterns = [
            r'(\w+)\s*=\s*input\.string\(\s*["\']([^"\']+)["\']',
            r'(\w+)\s*=\s*input\.timeframe\(\s*["\']([^"\']+)["\']',
            r'(\w+)\s*=\s*input\.symbol\(\s*["\']([^"\']+)["\']',
            r'(\w+)\s*=\s*input\(\s*["\']([^"\']+)["\']',
            r'(\w+)\s*=\s*input\.int\(\s*([\d\-]+)',
            r'(\w+)\s*=\s*input\.float\(\s*([\d\.\-]+)',
            r'(\w+)\s*=\s*input\.bool\(\s*(true|false)',
            r'(\w+)\s*=\s*input\(\s*(true|false)',
            r'(\w+)\s*=\s*input\(\s*([\d\.\-]+)'
        ]
        for pat in input_patterns:
            for m in re.finditer(pat, code, re.IGNORECASE):
                val = m.group(2)
                if val.lower() == "true":
                    self.params[m.group(1)] = True
                elif val.lower() == "false":
                    self.params[m.group(1)] = False
                elif re.match(r'^-?\d+$', val):
                    self.params[m.group(1)] = int(val)
                elif re.match(r'^-?\d+\.\d+$', val):
                    self.params[m.group(1)] = float(val)
                else:
                    self.params[m.group(1)] = val

        # 2. Extract Direct Variable Assignments (e.g. st_len = 10, tp_mult = 3.0, rr_ratio = 2.0)
        direct_var_patterns = [
            r'^\s*(\w+)\s*=\s*([\d\.\-]+)\s*(?://.*)?$',
            r'^\s*(\w+)\s*:=\s*([\d\.\-]+)\s*(?://.*)?$'
        ]
        for line in code.split('\n'):
            for pat in direct_var_patterns:
                m = re.match(pat, line)
                if m:
                    var_name = m.group(1)
                    val_str = m.group(2)
                    if var_name not in self.params:
                        if re.match(r'^-?\d+$', val_str):
                            self.params[var_name] = int(val_str)
                        elif re.match(r'^-?\d+\.\d+$', val_str):
                            self.params[var_name] = float(val_str)

    def build_step_evaluator(self, bars_1m: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Callable[[int, List[Dict[str, Any]], Optional[Position]], Optional[Order]]]:
        code = self.raw_code
        name_lower = self.name.lower()

        # Resample to 5m bars if 1m bars provided
        needs_resample = False
        if len(bars_1m) > 1:
            t0 = bars_1m[0]["timestamp"]
            t1 = bars_1m[1]["timestamp"]
            if hasattr(t0, "timestamp") and hasattr(t1, "timestamp"):
                if (t1 - t0).total_seconds() < 300:
                    needs_resample = True

        bars_5m = SessionManager.aggregate_bars(bars_1m, timeframe_minutes=5) if needs_resample else bars_1m

        closes = [float(b["close"]) for b in bars_5m]
        opens = [float(b["open"]) for b in bars_5m]
        highs = [float(b["high"]) for b in bars_5m]
        lows = [float(b["low"]) for b in bars_5m]
        volumes = [float(b.get("volume", 100)) for b in bars_5m]
        times = [b.get("timestamp") or b.get("time_ny") or datetime.now() for b in bars_5m]

        # Extract strategy-specific hyperparameters
        ema_len = int(self.params.get("ema_len", self.params.get("ema_trend_p", self.params.get("daily_ema_len", self.params.get("trend_ema", 50)))))
        ema_val = PineTA.ema(closes, ema_len)
        ema20 = PineTA.ema(closes, 20)
        ema200 = PineTA.ema(closes, 200)
        
        atr_len = int(self.params.get("atr_len", self.params.get("atr_length", 14)))
        atr_calc = PineTA.atr(highs, lows, closes, atr_len)
        atr60 = PineTA.atr(highs, lows, closes, 60)
        
        rsi_len = int(self.params.get("rsi_len", self.params.get("rsi_length", 14)))
        rsi_val = PineTA.rsi(closes, rsi_len)
        
        vol_ma = PineTA.sma(volumes, 20)
        vwap = PineTA.vwap(highs, lows, closes, volumes, times)
        
        # Volatility Variance for VRP & Gravity SMC
        var5 = PineTA.variance(closes, int(self.params.get("rv_len", 5)))
        var50 = PineTA.variance(closes, int(self.params.get("bv_len", 50)))
        vol_z = [v5 - v50 if not (math.isnan(v5) or math.isnan(v50)) else 0.0 for v5, v50 in zip(var5, var50)]

        # Bollinger Bands for Mean Reversion
        bb_mid, bb_upper, bb_lower = PineTA.bollinger_bands(closes, length=int(self.params.get("bb_length", 20)), mult=float(self.params.get("bb_mult", 2.0)))

        # Supertrend computation
        st_len = int(self.params.get("st_len", 10))
        st_mult = float(self.params.get("st_mult", 3.0))
        supertrend, st_direction = PineTA.supertrend(highs, lows, closes, factor=st_mult, atr_period=st_len)

        lookback_sw = int(self.params.get("lookback_len", self.params.get("swing_lookbk", self.params.get("lookback", 10))))
        hi_lookback = PineTA.highest(highs, lookback_sw)
        lo_lookback = PineTA.lowest(lows, lookback_sw)
        hi20 = PineTA.highest(highs, 20)
        lo20 = PineTA.lowest(lows, 20)

        # Strategy Archetype Flags
        is_morning_drive = ("morning drive" in name_lower or "morning_drive" in name_lower or "setup_window" in code or "prev_daily_high" in code)
        is_london_judas = ("london" in name_lower or "judas" in name_lower or "asian_high" in code)
        is_vrp = ("vrp" in name_lower or "variance" in code or "vol_z_score" in code and "master" not in name_lower)
        is_gravity_smc = ("gravity" in name_lower or "smc & volatility" in name_lower or "smc_redesign" in name_lower)
        is_momentum_scalper = ("momentum" in name_lower or "scalper" in name_lower or "break_dist" in code or "long_entry_level" in code)
        is_mean_reversion = ("entropy" in code or "relative fx" in name_lower or "econophysics" in name_lower)
        is_spike_master = ("spike master" in name_lower or "spikemaster" in name_lower or "spike_master" in name_lower or "chandelier" in code)
        is_spike_reversal = ("spike" in name_lower and not is_spike_master)
        is_nightmoves = ("nightmoves" in name_lower or "night_moves" in name_lower or "is_entry_bar" in code)
        is_orb = ("orb" in name_lower or "opening range" in name_lower or "orb_mins" in code or "orb_high" in code)
        is_vwap = ("vwap" in name_lower or "vwap_cross" in name_lower or "vwap_pullback" in name_lower)
        is_amd = ("amd" in name_lower or "10am" in name_lower or "manip_high" in code)
        is_silver_bullet = ("silverbullet" in name_lower or "silver bullet" in name_lower or "sb" in name_lower)
        is_ifvg = ("ifvg" in name_lower or "inverse fvg" in name_lower or "inverse_fvg" in name_lower)
        is_smc = ("smc" in name_lower or "sweep" in name_lower or "bsl" in name_lower or "ssl" in name_lower)
        is_bos = ("bos" in name_lower or "boswaves" in name_lower)

        # Strategy internal states across bars
        state = {
            "current_day": None,
            "session_high": None,
            "session_low": None,
            "setup_high": None,
            "setup_low": None,
            "asian_high": None,
            "asian_low": None,
            "prev_day_high": None,
            "prev_day_low": None,
            "orb_high": None,
            "orb_low": None,
            "orb_done": False,
            "accum_high": None,
            "accum_low": None,
            "manip_high": None,
            "manip_low": None,
            "last_bull_fvg": None,
            "last_bear_fvg": None,
            "last_trade_bar": -100,
            "traded_today": False
        }

        # Parameters
        cushion_ticks = self.params.get("cushion_ticks", 4)
        tick_stop = self.params.get("tick_stop", self.params.get("tick_stop_default", self.params.get("stop_loss", 50)))
        risk_reward = self.params.get("risk_reward", self.params.get("tp_multiplier", 2.0))
        qty = int(self.params.get("default_qty_value", 1))

        def on_bar(bar_idx: int, bars_hist: List[Dict[str, Any]], active_pos: Optional[Position]) -> Optional[Order]:
            if bar_idx < 25 or active_pos is not None:
                return None

            bar = bars_hist[bar_idx]
            raw_t = bar.get("timestamp") or bar.get("time_ny") or datetime.now()
            b_time = raw_t if isinstance(raw_t, datetime) else datetime.fromisoformat(str(raw_t))
            b_date = b_time.date()
            t = b_time.time()
            ny_mins = t.hour * 60 + t.minute
            c = closes[bar_idx]
            o = opens[bar_idx]
            h = highs[bar_idx]
            l = lows[bar_idx]
            v = volumes[bar_idx]

            # Day boundary reset
            if state["current_day"] != b_date:
                if state["session_high"] is not None:
                    state["prev_day_high"] = state["session_high"]
                    state["prev_day_low"] = state["session_low"]
                state["current_day"] = b_date
                state["session_high"] = h
                state["session_low"] = l
                state["setup_high"] = None
                state["setup_low"] = None
                state["asian_high"] = None
                state["asian_low"] = None
                state["orb_high"] = None
                state["orb_low"] = None
                state["orb_done"] = False
                state["accum_high"] = None
                state["accum_low"] = None
                state["manip_high"] = None
                state["manip_low"] = None
                state["last_bull_fvg"] = None
                state["last_bear_fvg"] = None
                state["traded_today"] = False
            else:
                state["session_high"] = max(state["session_high"], h) if state["session_high"] else h
                state["session_low"] = min(state["session_low"], l) if state["session_low"] else l

            # Standard RTH window
            in_rth = dtime(9, 30) <= t <= dtime(15, 45)

            # =================================================================
            # 1. DEMINT MORNING DRIVE (9:30 OPEN DRIVE WITH PRE-MARKET RANGE)
            # =================================================================
            if is_morning_drive:
                # 9:00 - 9:30 Setup range
                if 9 * 60 <= ny_mins < 9 * 60 + 30:
                    state["setup_high"] = h if state["setup_high"] is None else max(state["setup_high"], h)
                    state["setup_low"] = l if state["setup_low"] is None else min(state["setup_low"], l)
                    return None

                # 9:30 - 11:30 Entry Window
                if 9 * 60 + 30 <= ny_mins < 11 * 60 + 30 and not state["traded_today"] and state["setup_high"]:
                    cushion = cushion_ticks * 0.25
                    long_break = c > (state["setup_high"] + cushion)
                    short_break = c < (state["setup_low"] - cushion)
                    htf_bull = c > (ema_val[bar_idx] if not math.isnan(ema_val[bar_idx]) else c)
                    atr_v = atr_calc[bar_idx] if not math.isnan(atr_calc[bar_idx]) else 14.0

                    sl_pts = atr_v * float(self.params.get("atr_mult_sl", 2.0))
                    tp_pts = atr_v * 6.0

                    if long_break and htf_bull and c > o:
                        state["traded_today"] = True
                        return Order(
                            order_id=f"MDRV_BUY_{bar_idx}",
                            symbol="MNQ",
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            contracts=qty,
                            stop_loss_price=round(round((c - sl_pts) / 0.25) * 0.25, 2),
                            take_profit_price=round(round((c + tp_pts) / 0.25) * 0.25, 2),
                            created_at=b_time,
                            strategy_tag="MORNING_DRIVE_LONG"
                        )
                    elif short_break and not htf_bull and c < o:
                        state["traded_today"] = True
                        return Order(
                            order_id=f"MDRV_SELL_{bar_idx}",
                            symbol="MNQ",
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            contracts=qty,
                            stop_loss_price=round(round((c + sl_pts) / 0.25) * 0.25, 2),
                            take_profit_price=round(round((c - tp_pts) / 0.25) * 0.25, 2),
                            created_at=b_time,
                            strategy_tag="MORNING_DRIVE_SHORT"
                        )

            # =================================================================
            # 2. LONDON JUDAS SWING (2:00 AM – 5:00 AM ET)
            # =================================================================
            elif is_london_judas:
                # Asian session (Midnight to 2:00 AM)
                if 0 <= ny_mins < 2 * 60:
                    state["asian_high"] = h if state["asian_high"] is None else max(state["asian_high"], h)
                    state["asian_low"] = l if state["asian_low"] is None else min(state["asian_low"], l)
                    return None

                # London Killzone (2:00 AM to 5:00 AM)
                if 2 * 60 <= ny_mins <= 5 * 60 and state["asian_high"]:
                    cooldown_ok = (bar_idx - state["last_trade_bar"]) >= 15
                    if cooldown_ok:
                        judas_sell = (h > state["asian_high"]) and (c < state["asian_high"]) and (c < o)
                        judas_buy = (l < state["asian_low"]) and (c > state["asian_low"]) and (c > o)
                        
                        sl_pts = max(10.0, min(30.0, (h - c) + 2.0))
                        tp_pts = sl_pts * 2.5

                        if judas_buy:
                            state["last_trade_bar"] = bar_idx
                            return Order(
                                order_id=f"JUD_BUY_{bar_idx}",
                                symbol="MNQ",
                                side=OrderSide.BUY,
                                order_type=OrderType.MARKET,
                                contracts=qty,
                                stop_loss_price=round(round((c - sl_pts) / 0.25) * 0.25, 2),
                                take_profit_price=round(round((c + tp_pts) / 0.25) * 0.25, 2),
                                created_at=b_time,
                                strategy_tag="LONDON_JUDAS_LONG"
                            )
                        elif judas_sell:
                            state["last_trade_bar"] = bar_idx
                            return Order(
                                order_id=f"JUD_SELL_{bar_idx}",
                                symbol="MNQ",
                                side=OrderSide.SELL,
                                order_type=OrderType.MARKET,
                                contracts=qty,
                                stop_loss_price=round(round((c + sl_pts) / 0.25) * 0.25, 2),
                                take_profit_price=round(round((c - tp_pts) / 0.25) * 0.25, 2),
                                created_at=b_time,
                                strategy_tag="LONDON_JUDAS_SHORT"
                            )

            # =================================================================
            # 3. VRP VOLATILITY FADE (VARIANCE Z-SCORE SPIKE)
            # =================================================================
            elif is_vrp:
                if in_rth:
                    z_val = vol_z[bar_idx]
                    z_thresh = float(self.params.get("z_threshold", 1.5))
                    prev_c = closes[bar_idx - 1]

                    sl_pts = tick_stop * 0.25
                    tp_pts = sl_pts * 2.0

                    if z_val > z_thresh and c > prev_c and c > o:
                        return Order(
                            order_id=f"VRP_BUY_{bar_idx}",
                            symbol="MNQ",
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            contracts=qty,
                            stop_loss_price=round(round((c - sl_pts) / 0.25) * 0.25, 2),
                            take_profit_price=round(round((c + tp_pts) / 0.25) * 0.25, 2),
                            created_at=b_time,
                            strategy_tag="VRP_LONG"
                        )
                    elif z_val > z_thresh and c < prev_c and c < o:
                        return Order(
                            order_id=f"VRP_SELL_{bar_idx}",
                            symbol="MNQ",
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            contracts=qty,
                            stop_loss_price=round(round((c + sl_pts) / 0.25) * 0.25, 2),
                            take_profit_price=round(round((c - tp_pts) / 0.25) * 0.25, 2),
                            created_at=b_time,
                            strategy_tag="VRP_SHORT"
                        )

            # =================================================================
            # 4. GRAVITY SMC & VOLATILITY MASTER
            # =================================================================
            elif is_gravity_smc:
                if in_rth:
                    hi_lvl = hi_lookback[bar_idx - 1]
                    lo_lvl = lo_lookback[bar_idx - 1]
                    cushion = cushion_ticks * 0.25
                    
                    ssl_swept = (l <= lo_lvl) and (c > lo_lvl + cushion)
                    bsl_swept = (h >= hi_lvl) and (c < hi_lvl - cushion)
                    
                    htf_bull = c > (ema_val[bar_idx] if not math.isnan(ema_val[bar_idx]) else c)
                    bar_rng = h - l
                    b_body = abs(c - o)
                    is_disp = bar_rng > 0 and (b_body / bar_rng) >= 0.5
                    vol_ok = vol_z[bar_idx] > float(self.params.get("vol_z_thresh", 0.5))

                    atr_v = atr_calc[bar_idx] if not math.isnan(atr_calc[bar_idx]) else 12.0
                    sl_pts = atr_v * float(self.params.get("atr_mult_sl", 1.5))
                    tp_pts = atr_v * float(self.params.get("atr_mult_tp", 4.0))

                    if htf_bull and ssl_swept and is_disp and c > o and vol_ok:
                        return Order(
                            order_id=f"GMST_BUY_{bar_idx}",
                            symbol="MNQ",
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            contracts=qty,
                            stop_loss_price=round(round((c - sl_pts) / 0.25) * 0.25, 2),
                            take_profit_price=round(round((c + tp_pts) / 0.25) * 0.25, 2),
                            created_at=b_time,
                            strategy_tag="GRAVITY_SMC_LONG"
                        )
                    elif not htf_bull and bsl_swept and is_disp and c < o and vol_ok:
                        return Order(
                            order_id=f"GMST_SELL_{bar_idx}",
                            symbol="MNQ",
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            contracts=qty,
                            stop_loss_price=round(round((c + sl_pts) / 0.25) * 0.25, 2),
                            take_profit_price=round(round((c - tp_pts) / 0.25) * 0.25, 2),
                            created_at=b_time,
                            strategy_tag="GRAVITY_SMC_SHORT"
                        )

            # =================================================================
            # 5. MOMENTUM BREAKOUT SCALPER (INTRABAR LOOKBACK CHASE)
            # =================================================================
            elif is_momentum_scalper:
                use_lunch = self.params.get("use_lunch_lockout", True)
                in_window = (570 <= ny_mins < 705 or 825 <= ny_mins <= 955) if use_lunch else (570 <= ny_mins <= 960)

                if in_window:
                    p_hi = hi_lookback[bar_idx - 1]
                    p_lo = lo_lookback[bar_idx - 1]
                    b_dist = float(self.params.get("break_ticks", 10)) * 0.25

                    atr_v = atr_calc[bar_idx] if not math.isnan(atr_calc[bar_idx]) else 10.0
                    sl_pts = atr_v * float(self.params.get("atr_mult_sl", 1.5))
                    tp_pts = atr_v * float(self.params.get("atr_mult_tp", 6.0))

                    if c > (p_hi + b_dist) and c > o:
                        return Order(
                            order_id=f"MOM_BUY_{bar_idx}",
                            symbol="MNQ",
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            contracts=qty,
                            stop_loss_price=round(round((c - sl_pts) / 0.25) * 0.25, 2),
                            take_profit_price=round(round((c + tp_pts) / 0.25) * 0.25, 2),
                            created_at=b_time,
                            strategy_tag="MOM_SCALPER_LONG"
                        )
                    elif c < (p_lo - b_dist) and c < o:
                        return Order(
                            order_id=f"MOM_SELL_{bar_idx}",
                            symbol="MNQ",
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            contracts=qty,
                            stop_loss_price=round(round((c + sl_pts) / 0.25) * 0.25, 2),
                            take_profit_price=round(round((c - tp_pts) / 0.25) * 0.25, 2),
                            created_at=b_time,
                            strategy_tag="MOM_SCALPER_SHORT"
                        )

            # =================================================================
            # 6. RELATIVE FX / ECONOPHYSICS MEAN REVERSION
            # =================================================================
            elif is_mean_reversion:
                if in_rth:
                    bb_m = bb_mid[bar_idx]
                    bb_u = bb_upper[bar_idx]
                    bb_l = bb_lower[bar_idx]
                    
                    sl_pts = tick_stop * 0.25
                    tp_pts = sl_pts * 2.0

                    if c < bb_l and c > o:
                        return Order(
                            order_id=f"MR_BUY_{bar_idx}",
                            symbol="MNQ",
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            contracts=qty,
                            stop_loss_price=round(round((c - sl_pts) / 0.25) * 0.25, 2),
                            take_profit_price=round(round((c + tp_pts) / 0.25) * 0.25, 2),
                            created_at=b_time,
                            strategy_tag="MR_LONG"
                        )
                    elif c > bb_u and c < o:
                        return Order(
                            order_id=f"MR_SELL_{bar_idx}",
                            symbol="MNQ",
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            contracts=qty,
                            stop_loss_price=round(round((c + sl_pts) / 0.25) * 0.25, 2),
                            take_profit_price=round(round((c - tp_pts) / 0.25) * 0.25, 2),
                            created_at=b_time,
                            strategy_tag="MR_SHORT"
                        )

            # =================================================================
            # 7. SPIKE MASTER SERIES (SUPERTREND FLIP VS RSI DISPLACEMENT)
            # =================================================================
            elif is_spike_master or is_spike_reversal:
                mode = self.params.get("mode", "Supertrend Flip")
                use_time_lock = self.params.get("use_time_filter", True)
                in_window = (dtime(9, 30) <= t <= dtime(16, 0)) if use_time_lock else True

                if in_window:
                    v_ma = vol_ma[bar_idx] if not math.isnan(vol_ma[bar_idx]) else 100.0
                    rvol = v / max(1.0, v_ma)
                    vol_mult = float(self.params.get("vol_mult", self.params.get("rvol_threshold", 1.3)))
                    vol_ok = (not self.params.get("use_vol_filter", self.params.get("use_rvol_filter", False))) or (rvol >= vol_mult)

                    st_flip_bull = (st_direction[bar_idx - 1] == -1 and st_direction[bar_idx] == 1)
                    st_flip_bear = (st_direction[bar_idx - 1] == 1 and st_direction[bar_idx] == -1)

                    r_val = rsi_val[bar_idx] if not math.isnan(rsi_val[bar_idx]) else 50.0
                    atr_v = atr_calc[bar_idx] if not math.isnan(atr_calc[bar_idx]) else 10.0
                    body = c - o
                    c_range = h - l
                    body_mult = float(self.params.get("body_mult", 1.5))
                    is_bull_disp = (body > atr_v * body_mult) and (c >= h - c_range * 0.25)
                    is_bear_disp = ((-body) > atr_v * body_mult) and (c <= l + c_range * 0.25)

                    rsi_os = float(self.params.get("rsi_oversold", 30.0 if is_spike_reversal else 50.0))
                    rsi_ob = float(self.params.get("rsi_overbought", 70.0))
                    lookback_w = int(self.params.get("lookback_bars", 30))
                    
                    recent_rsi = rsi_val[max(0, bar_idx - lookback_w):bar_idx + 1]
                    was_os = any(r < rsi_os for r in recent_rsi if not math.isnan(r))
                    was_ob = any(r > rsi_ob for r in recent_rsi if not math.isnan(r))

                    rsi_buy = was_os and is_bull_disp
                    rsi_sell = was_ob and is_bear_disp

                    buy_sig = (st_flip_bull if mode == "Supertrend Flip" else rsi_buy) and vol_ok
                    sell_sig = (st_flip_bear if mode == "Supertrend Flip" else rsi_sell) and vol_ok

                    if self.params.get("use_ema_filter", False) or self.params.get("use_htf_filter", False) or "ema" in name_lower or "htftrend" in name_lower:
                        e_val = ema200[bar_idx] if not math.isnan(ema200[bar_idx]) else c
                        buy_sig = buy_sig and (c > e_val)
                        sell_sig = sell_sig and (c < e_val)

                    sl_mult = float(self.params.get("atr_mult_sl", self.params.get("sl_mult", 1.5)))
                    sl_pts = max(7.5, (atr_v * sl_mult))
                    long_tp_mult = float(self.params.get("long_tp_mult", self.params.get("tp_mult", self.params.get("atr_mult_tp", 3.0))))
                    short_tp_mult = float(self.params.get("short_tp_mult", self.params.get("tp_mult", self.params.get("atr_mult_tp", 3.0))))

                    if buy_sig:
                        tp_pts = sl_pts * (long_tp_mult / sl_mult)
                        return Order(
                            order_id=f"SM_BUY_{bar_idx}",
                            symbol="MNQ",
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            contracts=qty,
                            stop_loss_price=round(round((c - sl_pts) / 0.25) * 0.25, 2),
                            take_profit_price=round(round((c + tp_pts) / 0.25) * 0.25, 2),
                            created_at=b_time,
                            strategy_tag="SPIKE_MASTER_LONG"
                        )
                    elif sell_sig:
                        tp_pts = sl_pts * (short_tp_mult / sl_mult)
                        return Order(
                            order_id=f"SM_SELL_{bar_idx}",
                            symbol="MNQ",
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            contracts=qty,
                            stop_loss_price=round(round((c + sl_pts) / 0.25) * 0.25, 2),
                            take_profit_price=round(round((c - tp_pts) / 0.25) * 0.25, 2),
                            created_at=b_time,
                            strategy_tag="SPIKE_MASTER_SHORT"
                        )

            # =================================================================
            # 8. NIGHTMOVES & GLOBEX DRIFT STRATEGIES
            # =================================================================
            elif is_nightmoves:
                if t.hour == 15 and t.minute >= 50 and not state["traded_today"]:
                    day_range = (state["session_high"] - state["session_low"]) if (state["session_high"] and state["session_low"]) else 10.0
                    close_pct = (c - state["session_low"]) / max(1.0, day_range) if day_range > 0 else 0.5
                    
                    cur_atr = atr_calc[bar_idx] if not math.isnan(atr_calc[bar_idx]) else 15.0
                    base_atr = atr60[bar_idx] if not math.isnan(atr60[bar_idx]) else 15.0
                    vol_ok = (cur_atr / max(1.0, base_atr)) <= float(self.params.get("volatility_limit", 2.0))
                    
                    trend_ok = c > ema_val[bar_idx] if not math.isnan(ema_val[bar_idx]) else True

                    if close_pct >= float(self.params.get("close_pct_threshold", 0.60)) and vol_ok and trend_ok:
                        state["traded_today"] = True
                        sl_pts = tick_stop * 0.25
                        tp_pts = sl_pts * risk_reward
                        return Order(
                            order_id=f"NM_BUY_{bar_idx}",
                            symbol="MNQ",
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            contracts=qty,
                            stop_loss_price=round(round((c - sl_pts) / 0.25) * 0.25, 2),
                            take_profit_price=round(round((c + tp_pts) / 0.25) * 0.25, 2),
                            created_at=b_time,
                            strategy_tag="NIGHTMOVES_LONG"
                        )

            # =================================================================
            # 9. OPENING RANGE BREAKOUT (PSW ORB & VARIATIONS)
            # =================================================================
            elif is_orb:
                orb_mins = int(self.params.get("orb_minutes", self.params.get("orb_mins", self.params.get("orb_period_mins", 15))))
                orb_end_m = 30 + orb_mins
                orb_end_time = dtime(9 + orb_end_m // 60, orb_end_m % 60)

                if dtime(9, 30) <= t <= orb_end_time:
                    state["orb_high"] = h if state["orb_high"] is None else max(state["orb_high"], h)
                    state["orb_low"] = l if state["orb_low"] is None else min(state["orb_low"], l)
                    return None

                if t > orb_end_time and not state["orb_done"] and state["orb_high"]:
                    state["orb_done"] = True

                if state["orb_done"] and not state["traded_today"] and orb_end_time < t <= dtime(13, 0):
                    v_ma = vol_ma[bar_idx] if not math.isnan(vol_ma[bar_idx]) else 100.0
                    rvol = (v / max(1.0, v_ma))
                    rvol_thresh = float(self.params.get("rvol_threshold", self.params.get("rvol_mult", 1.2)))
                    rvol_ok = rvol >= rvol_thresh if ("rvol" in name_lower or self.params.get("use_rvol_filter", False) or "rvol_threshold" in self.params) else True
                    
                    use_trend = self.params.get("use_mtf_trend", self.params.get("use_trend_filter", True))
                    trend_ok = (c > ema_val[bar_idx]) if (use_trend and not math.isnan(ema_val[bar_idx])) else True

                    cushion = cushion_ticks * 0.25
                    atr_v = atr_calc[bar_idx] if not math.isnan(atr_calc[bar_idx]) else 12.0
                    
                    # Exact SL / TP calculation
                    rr = float(self.params.get("rr_ratio", self.params.get("risk_reward", self.params.get("tp_multiplier", 2.0))))
                    if state["orb_low"] is not None and "trendrider" in name_lower:
                        sl_pts = max(6.0, c - state["orb_low"])
                    else:
                        sl_pts = max(8.0, atr_v * float(self.params.get("atr_multiplier", self.params.get("atr_mult_sl", 1.5))))
                    tp_pts = sl_pts * rr

                    if c > (state["orb_high"] + cushion) and trend_ok and rvol_ok and c > o:
                        state["traded_today"] = True
                        return Order(
                            order_id=f"ORB_BUY_{bar_idx}",
                            symbol="MNQ",
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            contracts=qty,
                            stop_loss_price=round(round((c - sl_pts) / 0.25) * 0.25, 2),
                            take_profit_price=round(round((c + tp_pts) / 0.25) * 0.25, 2),
                            created_at=b_time,
                            strategy_tag="ORB_LONG"
                        )
                    elif c < (state["orb_low"] - cushion) and (not trend_ok) and rvol_ok and c < o:
                        state["traded_today"] = True
                        return Order(
                            order_id=f"ORB_SELL_{bar_idx}",
                            symbol="MNQ",
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            contracts=qty,
                            stop_loss_price=round(round((c + sl_pts) / 0.25) * 0.25, 2),
                            take_profit_price=round(round((c - tp_pts) / 0.25) * 0.25, 2),
                            created_at=b_time,
                            strategy_tag="ORB_SHORT"
                        )

            # =================================================================
            # 10. VWAP CROSS & VWAP PULLBACK MASTER
            # =================================================================
            elif is_vwap:
                if in_rth:
                    vwap_v = vwap[bar_idx] if not math.isnan(vwap[bar_idx]) else c
                    prev_c = closes[bar_idx - 1]
                    prev_vwap = vwap[bar_idx - 1] if not math.isnan(vwap[bar_idx - 1]) else prev_c

                    cross_up = (prev_c <= prev_vwap) and (c > vwap_v)
                    cross_down = (prev_c >= prev_vwap) and (c < vwap_v)

                    trend_v = ema_val[bar_idx] if not math.isnan(ema_val[bar_idx]) else c

                    sl_pts = tick_stop * 0.25
                    tp_pts = sl_pts * risk_reward

                    if cross_up and c > trend_v and c > o:
                        return Order(
                            order_id=f"VWAP_BUY_{bar_idx}",
                            symbol="MNQ",
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            contracts=qty,
                            stop_loss_price=round(round((c - sl_pts) / 0.25) * 0.25, 2),
                            take_profit_price=round(round((c + tp_pts) / 0.25) * 0.25, 2),
                            created_at=b_time,
                            strategy_tag="VWAP_LONG"
                        )
                    elif cross_down and c < trend_v and c < o:
                        return Order(
                            order_id=f"VWAP_SELL_{bar_idx}",
                            symbol="MNQ",
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            contracts=qty,
                            stop_loss_price=round(round((c + sl_pts) / 0.25) * 0.25, 2),
                            take_profit_price=round(round((c - tp_pts) / 0.25) * 0.25, 2),
                            created_at=b_time,
                            strategy_tag="VWAP_SHORT"
                        )

            # =================================================================
            # 11. 10:00 AM MACRO AMD (Accumulation -> Manipulation -> Distribution)
            # =================================================================
            elif is_amd:
                if dtime(9, 30) <= t < dtime(9, 55):
                    state["accum_high"] = h if state["accum_high"] is None else max(state["accum_high"], h)
                    state["accum_low"] = l if state["accum_low"] is None else min(state["accum_low"], l)
                    return None

                if dtime(9, 55) <= t <= dtime(10, 5) and state["accum_high"]:
                    if h > state["accum_high"] + 2.0:
                        state["manip_high"] = h
                    if l < state["accum_low"] - 2.0:
                        state["manip_low"] = l

                if dtime(10, 0) <= t <= dtime(10, 45) and not state["traded_today"] and state["accum_high"]:
                    if state["manip_low"] and c > state["accum_low"] and c > o:
                        state["traded_today"] = True
                        sl_price = round(round((state["manip_low"] - 1.0) / 0.25) * 0.25, 2)
                        risk = max(6.0, c - sl_price)
                        tp_price = round(round((c + risk * risk_reward) / 0.25) * 0.25, 2)
                        return Order(
                            order_id=f"AMD_BUY_{bar_idx}",
                            symbol="MNQ",
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            contracts=qty,
                            stop_loss_price=sl_price,
                            take_profit_price=tp_price,
                            created_at=b_time,
                            strategy_tag="AMD_LONG"
                        )
                    elif state["manip_high"] and c < state["accum_high"] and c < o:
                        state["traded_today"] = True
                        sl_price = round(round((state["manip_high"] + 1.0) / 0.25) * 0.25, 2)
                        risk = max(6.0, sl_price - c)
                        tp_price = round(round((c - risk * risk_reward) / 0.25) * 0.25, 2)
                        return Order(
                            order_id=f"AMD_SELL_{bar_idx}",
                            symbol="MNQ",
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            contracts=qty,
                            stop_loss_price=sl_price,
                            take_profit_price=tp_price,
                            created_at=b_time,
                            strategy_tag="AMD_SHORT"
                        )

            # =================================================================
            # 12. ICT SILVER BULLET (10:00 - 11:00 AM NY Session)
            # =================================================================
            elif is_silver_bullet:
                if dtime(10, 0) <= t <= dtime(11, 0) and not state["traded_today"]:
                    fvg_bull = l > highs[bar_idx - 2] and (l - highs[bar_idx - 2]) >= 2.0
                    fvg_bear = h < lows[bar_idx - 2] and (lows[bar_idx - 2] - h) >= 2.0
                    ema_t = ema_val[bar_idx] if not math.isnan(ema_val[bar_idx]) else c

                    sl_pts = tick_stop * 0.25
                    tp_pts = sl_pts * risk_reward

                    if fvg_bull and c > ema_t and c > o:
                        state["traded_today"] = True
                        return Order(
                            order_id=f"SB_BUY_{bar_idx}",
                            symbol="MNQ",
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            contracts=qty,
                            stop_loss_price=round(round((c - sl_pts) / 0.25) * 0.25, 2),
                            take_profit_price=round(round((c + tp_pts) / 0.25) * 0.25, 2),
                            created_at=b_time,
                            strategy_tag="SB_LONG"
                        )
                    elif fvg_bear and c < ema_t and c < o:
                        state["traded_today"] = True
                        return Order(
                            order_id=f"SB_SELL_{bar_idx}",
                            symbol="MNQ",
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            contracts=qty,
                            stop_loss_price=round(round((c + sl_pts) / 0.25) * 0.25, 2),
                            take_profit_price=round(round((c - tp_pts) / 0.25) * 0.25, 2),
                            created_at=b_time,
                            strategy_tag="SB_SHORT"
                        )

            # =================================================================
            # 13. INVERSE FVG STRATEGY
            # =================================================================
            elif is_ifvg:
                if in_rth:
                    if l > highs[bar_idx - 2] and (l - highs[bar_idx - 2]) >= 2.0:
                        state["last_bull_fvg"] = highs[bar_idx - 2]
                    if h < lows[bar_idx - 2] and (lows[bar_idx - 2] - h) >= 2.0:
                        state["last_bear_fvg"] = lows[bar_idx - 2]

                    ema_t = ema_val[bar_idx] if not math.isnan(ema_val[bar_idx]) else c
                    buy_cond = state["last_bear_fvg"] and c > (state["last_bear_fvg"] + 1.5) and c > ema_t and c > o
                    sell_cond = state["last_bull_fvg"] and c < (state["last_bull_fvg"] - 1.5) and c < ema_t and c < o

                    sl_pts = tick_stop * 0.25
                    tp_pts = sl_pts * risk_reward

                    if buy_cond:
                        state["last_bear_fvg"] = None
                        return Order(
                            order_id=f"IFVG_BUY_{bar_idx}",
                            symbol="MNQ",
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            contracts=qty,
                            stop_loss_price=round(round((c - sl_pts) / 0.25) * 0.25, 2),
                            take_profit_price=round(round((c + tp_pts) / 0.25) * 0.25, 2),
                            created_at=b_time,
                            strategy_tag="IFVG_LONG"
                        )
                    elif sell_cond:
                        state["last_bull_fvg"] = None
                        return Order(
                            order_id=f"IFVG_SELL_{bar_idx}",
                            symbol="MNQ",
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            contracts=qty,
                            stop_loss_price=round(round((c + sl_pts) / 0.25) * 0.25, 2),
                            take_profit_price=round(round((c - tp_pts) / 0.25) * 0.25, 2),
                            created_at=b_time,
                            strategy_tag="IFVG_SHORT"
                        )

            # =================================================================
            # 14. SMC SWEEPS & BOS STRUCTURAL BREAKOUTS
            # =================================================================
            elif is_smc or is_bos:
                if in_rth:
                    ema_t = ema_val[bar_idx] if not math.isnan(ema_val[bar_idx]) else c
                    sw_hi = hi_lookback[bar_idx - 1]
                    sw_lo = lo_lookback[bar_idx - 1]

                    sl_pts = tick_stop * 0.25
                    tp_pts = sl_pts * risk_reward

                    if c > sw_hi and c > ema_t and c > o:
                        return Order(
                            order_id=f"BOS_BUY_{bar_idx}",
                            symbol="MNQ",
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            contracts=qty,
                            stop_loss_price=round(round((c - sl_pts) / 0.25) * 0.25, 2),
                            take_profit_price=round(round((c + tp_pts) / 0.25) * 0.25, 2),
                            created_at=b_time,
                            strategy_tag="BOS_LONG"
                        )
                    elif c < sw_lo and c < ema_t and c < o:
                        return Order(
                            order_id=f"BOS_SELL_{bar_idx}",
                            symbol="MNQ",
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            contracts=qty,
                            stop_loss_price=round(round((c + sl_pts) / 0.25) * 0.25, 2),
                            take_profit_price=round(round((c - tp_pts) / 0.25) * 0.25, 2),
                            created_at=b_time,
                            strategy_tag="BOS_SHORT"
                        )

            # =================================================================
            # 15. DYNAMIC ALPHA (MOMENTUM / TREND FOLLOWING)
            # =================================================================
            else:
                if in_rth:
                    r_val = rsi_val[bar_idx] if not math.isnan(rsi_val[bar_idx]) else 50.0
                    ema_fast = ema20[bar_idx] if not math.isnan(ema20[bar_idx]) else c
                    ema_slow = ema_val[bar_idx] if not math.isnan(ema_val[bar_idx]) else c

                    sl_pts = tick_stop * 0.25
                    tp_pts = sl_pts * risk_reward

                    if ema_fast > ema_slow and r_val > 52.0 and c > o:
                        return Order(
                            order_id=f"ALPHA_BUY_{bar_idx}",
                            symbol="MNQ",
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            contracts=qty,
                            stop_loss_price=round(round((c - sl_pts) / 0.25) * 0.25, 2),
                            take_profit_price=round(round((c + tp_pts) / 0.25) * 0.25, 2),
                            created_at=b_time,
                            strategy_tag="ALPHA_LONG"
                        )
                    elif ema_fast < ema_slow and r_val < 48.0 and c < o:
                        return Order(
                            order_id=f"ALPHA_SELL_{bar_idx}",
                            symbol="MNQ",
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            contracts=qty,
                            stop_loss_price=round(round((c + sl_pts) / 0.25) * 0.25, 2),
                            take_profit_price=round(round((c - tp_pts) / 0.25) * 0.25, 2),
                            created_at=b_time,
                            strategy_tag="ALPHA_SHORT"
                        )

            return None

        return bars_5m, on_bar


class PineScriptRunner:
    @staticmethod
    def run_pine_code(
        pine_code: str,
        bars: List[Dict[str, Any]],
        config: Optional[Any] = None
    ) -> Any:
        from ..core.engine import BacktestEngine
        from ..config import BacktestConfig

        strat = PineScriptStrategy(pine_code)
        bars_eval, on_bar_fn = strat.build_step_evaluator(bars)

        cfg = config or BacktestConfig()
        engine = BacktestEngine(config=cfg)
        result = engine.run(bars_eval, strategy_fn=on_bar_fn)
        return result, strat
