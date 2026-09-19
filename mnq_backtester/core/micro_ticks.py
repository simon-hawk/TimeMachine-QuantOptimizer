"""
High-Detailization Microstructure Simulator (40 Ticks Per Bar).
Generates intra-bar auction paths from OHLC data to simulate realistic tick-by-tick
order fills, slippage, and stop-loss/take-profit triggering without hindsight bias.
"""

import math
import random
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class MicroTick:
    timestamp: datetime
    price: float
    volume: int
    tick_index: int

class TickDetailSimulator:
    """
    Simulates 40 discrete auction ticks within each OHLC bar.
    Realistic intraday paths:
      - Bullish Candle (Close >= Open): Open -> Dip to Low -> Rally to High -> Pullback to Close
      - Bearish Candle (Close < Open): Open -> Pop to High -> Drop to Low -> Bounce to Close
      - Includes Poisson/Gaussian microstructure noise and 0.25 CME tick quantization.
    """

    TICKS_PER_BAR: int = 40
    TICK_SIZE: float = 0.25

    @classmethod
    def generate_sub_ticks(
        cls,
        bar: Dict[str, Any],
        ticks_per_bar: int = 40,
        seed: int = 42
    ) -> List[MicroTick]:
        o = float(bar["open"])
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])
        vol = int(bar.get("volume", 100))
        t_start = bar.get("timestamp") or bar.get("time_ny") or datetime.now()

        # Ensure timestamp is datetime
        if not isinstance(t_start, datetime):
            try:
                t_start = datetime.fromisoformat(str(t_start))
            except Exception:
                t_start = datetime.now()

        duration = timedelta(minutes=5)
        tick_delta = duration / ticks_per_bar
        rng = random.Random(seed + int(t_start.timestamp()) % 100000)

        # Segment allocations
        n1 = ticks_per_bar // 4       # Open to 1st Extreme
        n2 = ticks_per_bar // 2       # 1st Extreme to 2nd Extreme
        n3 = ticks_per_bar - (n1 + n2)# 2nd Extreme to Close

        ticks: List[MicroTick] = []
        cur_price = o

        def quantize(p: float) -> float:
            return round(round(p / cls.TICK_SIZE) * cls.TICK_SIZE, 2)

        # Bullish Path: O -> L -> H -> C
        # Bearish Path: O -> H -> L -> C
        is_bullish = c >= o

        if is_bullish:
            # Segment 1: Open -> Low
            for i in range(n1):
                prog = (i + 1) / n1
                noise = rng.normalvariate(0, cls.TICK_SIZE * 0.5)
                p = o + (l - o) * prog + noise
                p = max(l, min(h, p))
                cur_price = quantize(p)
                ticks.append(MicroTick(
                    timestamp=t_start + tick_delta * len(ticks),
                    price=cur_price,
                    volume=max(1, vol // ticks_per_bar),
                    tick_index=len(ticks)
                ))

            # Segment 2: Low -> High
            for i in range(n2):
                prog = (i + 1) / n2
                noise = rng.normalvariate(0, cls.TICK_SIZE * 0.5)
                p = l + (h - l) * prog + noise
                p = max(l, min(h, p))
                cur_price = quantize(p)
                ticks.append(MicroTick(
                    timestamp=t_start + tick_delta * len(ticks),
                    price=cur_price,
                    volume=max(1, vol // ticks_per_bar),
                    tick_index=len(ticks)
                ))

            # Segment 3: High -> Close
            for i in range(n3):
                prog = (i + 1) / n3
                noise = rng.normalvariate(0, cls.TICK_SIZE * 0.5)
                p = h + (c - h) * prog + noise
                p = max(l, min(h, p))
                cur_price = quantize(p)
                ticks.append(MicroTick(
                    timestamp=t_start + tick_delta * len(ticks),
                    price=cur_price,
                    volume=max(1, vol // ticks_per_bar),
                    tick_index=len(ticks)
                ))

        else:
            # Segment 1: Open -> High
            for i in range(n1):
                prog = (i + 1) / n1
                noise = rng.normalvariate(0, cls.TICK_SIZE * 0.5)
                p = o + (h - o) * prog + noise
                p = max(l, min(h, p))
                cur_price = quantize(p)
                ticks.append(MicroTick(
                    timestamp=t_start + tick_delta * len(ticks),
                    price=cur_price,
                    volume=max(1, vol // ticks_per_bar),
                    tick_index=len(ticks)
                ))

            # Segment 2: High -> Low
            for i in range(n2):
                prog = (i + 1) / n2
                noise = rng.normalvariate(0, cls.TICK_SIZE * 0.5)
                p = h + (l - h) * prog + noise
                p = max(l, min(h, p))
                cur_price = quantize(p)
                ticks.append(MicroTick(
                    timestamp=t_start + tick_delta * len(ticks),
                    price=cur_price,
                    volume=max(1, vol // ticks_per_bar),
                    tick_index=len(ticks)
                ))

            # Segment 3: Low -> Close
            for i in range(n3):
                prog = (i + 1) / n3
                noise = rng.normalvariate(0, cls.TICK_SIZE * 0.5)
                p = l + (c - l) * prog + noise
                p = max(l, min(h, p))
                cur_price = quantize(p)
                ticks.append(MicroTick(
                    timestamp=t_start + tick_delta * len(ticks),
                    price=cur_price,
                    volume=max(1, vol // ticks_per_bar),
                    tick_index=len(ticks)
                ))

        # Ensure exact high and low are touched in the sub-tick sequence
        if not any(t.price == h for t in ticks):
            ticks[ticks_per_bar // 2].price = h
        if not any(t.price == l for t in ticks):
            ticks[ticks_per_bar // 4].price = l
        ticks[-1].price = c

        return ticks
