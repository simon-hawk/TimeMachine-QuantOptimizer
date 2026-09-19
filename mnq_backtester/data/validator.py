"""
Data Accuracy and Quality Verification Engine.
Checks OHLC invariants, CME tick quantization, bad ticks/spikes, and session continuity.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import math
import numpy as np

@dataclass
class ValidationReport:
    is_valid: bool
    total_bars: int
    passed_bars: int
    failed_bars: int
    valid_percentage: float
    ohlc_invariant_errors: int = 0
    tick_quantization_errors: int = 0
    spike_anomaly_errors: int = 0
    zero_or_negative_price_errors: int = 0
    time_gap_errors: int = 0
    detailed_errors: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""

    def __str__(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        return (
            f"=== Data Validation Report [{status}] ===\n"
            f"Total Bars Evaluated: {self.total_bars:,}\n"
            f"Passed: {self.passed_bars:,} ({self.valid_percentage:.2f}%)\n"
            f"Failed: {self.failed_bars:,}\n"
            f"  - OHLC Invariant Violations: {self.ohlc_invariant_errors}\n"
            f"  - Tick Quantization Violations: {self.tick_quantization_errors}\n"
            f"  - Flash Spikes / Outliers: {self.spike_anomaly_errors}\n"
            f"  - Zero/Negative Price Violations: {self.zero_or_negative_price_errors}\n"
            f"  - Unexpected Time Gaps: {self.time_gap_errors}\n"
            f"Summary: {self.summary}"
        )


class DataValidator:
    """
    Validates MNQ / CME futures market data accuracy, consistency, and structural integrity.
    """
    
    def __init__(self, tick_size: float = 0.25, max_z_score: float = 6.0, max_gap_minutes: int = 60):
        self.tick_size = tick_size
        self.max_z_score = max_z_score
        self.max_gap_minutes = max_gap_minutes

    def validate_bars(self, bars: List[Dict[str, Any]], strict_quantization: bool = True) -> ValidationReport:
        """
        Runs comprehensive validation across a time series of OHLCV bars.
        
        Each bar dict must have: 'timestamp' (or 'time_ny' or 'date'), 'open', 'high', 'low', 'close', optional 'volume'.
        """
        if not bars:
            return ValidationReport(
                is_valid=False,
                total_bars=0,
                passed_bars=0,
                failed_bars=0,
                valid_percentage=0.0,
                summary="Empty data series provided."
            )

        total_bars = len(bars)
        ohlc_invariant_errors = 0
        tick_quantization_errors = 0
        spike_anomaly_errors = 0
        zero_or_negative_errors = 0
        time_gap_errors = 0
        detailed_errors = []

        # Pre-calculate log returns for outlier/spike detection
        closes = []
        for b in bars:
            c = float(b.get("close", 0.0))
            closes.append(c if c > 0 else 1.0)
        
        returns = np.diff(np.log(closes))
        if len(returns) > 10:
            ret_mean = np.mean(returns)
            ret_std = np.std(returns)
            z_threshold = self.max_z_score
        else:
            ret_mean, ret_std, z_threshold = 0.0, 1.0, 999.0

        prev_time: Optional[datetime] = None

        for idx, bar in enumerate(bars):
            bar_errors = []
            
            # Extract timestamp
            raw_time = bar.get("timestamp") or bar.get("time_ny") or bar.get("date")
            if isinstance(raw_time, str):
                try:
                    bar_time = datetime.fromisoformat(raw_time)
                except Exception:
                    bar_time = datetime.now()
            elif isinstance(raw_time, datetime):
                bar_time = raw_time
            else:
                bar_time = None

            o = float(bar.get("open", 0.0))
            h = float(bar.get("high", 0.0))
            l = float(bar.get("low", 0.0))
            c = float(bar.get("close", 0.0))
            v = float(bar.get("volume", 0.0))

            # 1. Zero or negative prices
            if o <= 0 or h <= 0 or l <= 0 or c <= 0:
                zero_or_negative_errors += 1
                bar_errors.append(f"Non-positive price detected: O={o}, H={h}, L={l}, C={c}")

            # 2. OHLC Invariants
            # High must be >= Low, Open, Close
            # Low must be <= High, Open, Close
            if not (h >= l and h >= o - 1e-4 and h >= c - 1e-4 and l <= o + 1e-4 and l <= c + 1e-4):
                ohlc_invariant_errors += 1
                bar_errors.append(f"OHLC invariant broken: O={o}, H={h}, L={l}, C={c}")

            if v < 0:
                ohlc_invariant_errors += 1
                bar_errors.append(f"Negative volume: V={v}")

            # 3. CME Tick Quantization (0.25 pt increments)
            if strict_quantization and o > 0:
                for p_name, p_val in [("Open", o), ("High", h), ("Low", l), ("Close", c)]:
                    remainder = round((p_val / self.tick_size) % 1.0, 4)
                    if remainder not in (0.0, 1.0):
                        tick_quantization_errors += 1
                        bar_errors.append(f"Tick quantization failed on {p_name}={p_val} (not multiple of {self.tick_size})")
                        break

            # 4. Spike / Flash anomaly detection
            if idx > 0 and ret_std > 1e-6:
                cur_ret = returns[idx - 1]
                z_score = abs((cur_ret - ret_mean) / ret_std)
                if z_score > z_threshold:
                    spike_anomaly_errors += 1
                    bar_errors.append(f"Flash spike detected: return={cur_ret:.4f}, Z-score={z_score:.2f} > {z_threshold}")

            # 5. Missing time gap check (skipping weekends)
            if prev_time and bar_time:
                delta = bar_time - prev_time
                # If gap is > max_gap_minutes and not across a weekend (Friday close -> Sunday open)
                if delta > timedelta(minutes=self.max_gap_minutes):
                    # Check if weekend gap
                    is_weekend = (prev_time.weekday() == 4 and bar_time.weekday() == 6) # Fri to Sun
                    if not is_weekend:
                        time_gap_errors += 1
                        bar_errors.append(f"Unaccounted time gap of {delta} between {prev_time} and {bar_time}")

            if bar_time:
                prev_time = bar_time

            if bar_errors:
                detailed_errors.append({
                    "index": idx,
                    "timestamp": str(bar_time) if bar_time else f"Index {idx}",
                    "errors": bar_errors
                })

        failed_bars = len(detailed_errors)
        passed_bars = total_bars - failed_bars
        valid_percentage = (passed_bars / total_bars) * 100.0 if total_bars > 0 else 0.0
        
        # Valid if >= 98% clean and no catastrophic zero/negative price corruptions
        is_valid = (valid_percentage >= 98.0) and (zero_or_negative_errors == 0) and (ohlc_invariant_errors == 0)

        summary = (
            "Data verified successfully with clean invariants."
            if is_valid
            else f"Data quality check flagged {failed_bars} bars with integrity issues."
        )

        return ValidationReport(
            is_valid=is_valid,
            total_bars=total_bars,
            passed_bars=passed_bars,
            failed_bars=failed_bars,
            valid_percentage=valid_percentage,
            ohlc_invariant_errors=ohlc_invariant_errors,
            tick_quantization_errors=tick_quantization_errors,
            spike_anomaly_errors=spike_anomaly_errors,
            zero_or_negative_price_errors=zero_or_negative_errors,
            time_gap_errors=time_gap_errors,
            detailed_errors=detailed_errors[:50],  # cap details to 50 for memory
            summary=summary
        )

    def clean_and_quantize(self, bars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Auto-repairs minor float inaccuracies by quantizing prices to nearest tick
        and enforcing OHLC bounds.
        """
        cleaned = []
        for bar in bars:
            b = dict(bar)
            o = round(round(float(b["open"]) / self.tick_size) * self.tick_size, 4)
            h = round(round(float(b["high"]) / self.tick_size) * self.tick_size, 4)
            l = round(round(float(b["low"]) / self.tick_size) * self.tick_size, 4)
            c = round(round(float(b["close"]) / self.tick_size) * self.tick_size, 4)

            # Ensure high is max and low is min
            real_high = max(o, h, l, c)
            real_low = min(o, h, l, c)

            b["open"] = o
            b["high"] = real_high
            b["low"] = real_low
            b["close"] = c
            if "volume" in b:
                b["volume"] = max(0, int(b["volume"]))
            cleaned.append(b)
        return cleaned
