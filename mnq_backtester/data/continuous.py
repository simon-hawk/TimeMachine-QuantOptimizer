"""
Continuous Futures Contract Adjustment and Roll Stitcher.
Implements Panama (Additive) and Ratio (Multiplicative) back-adjustments
to eliminate artificial price gaps caused by quarterly contract expirations (H, M, U, Z).
"""

from typing import List, Dict, Any, Literal
import numpy as np

class ContinuousFuturesStitcher:
    """
    Stitches individual futures contract segments into a seamless continuous price series.
    """

    @staticmethod
    def adjust_panama_additive(contract_segments: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Applies Panama (Additive) back-adjustment.
        Preserves absolute point moves and dollar P&L ($2.00 per point for MNQ).
        
        Calculates roll shift = Next_Contract_Open - Front_Contract_Close at roll boundary,
        and adds this cumulative shift to all older contracts backwards.
        """
        if not contract_segments:
            return []
        if len(contract_segments) == 1:
            return contract_segments[0]

        # Work backwards from the newest contract (which remains unadjusted)
        adjusted_segments = [list(contract_segments[-1])]
        cumulative_shift = 0.0

        for i in range(len(contract_segments) - 2, -1, -1):
            front_segment = contract_segments[i]
            next_segment = contract_segments[i + 1]

            if not front_segment or not next_segment:
                continue

            # Roll difference at transition
            roll_gap = next_segment[0]["open"] - front_segment[-1]["close"]
            cumulative_shift += roll_gap

            # Adjust all bars in the front segment
            adjusted_front = []
            for bar in front_segment:
                b = dict(bar)
                b["open"] = round(b["open"] + cumulative_shift, 4)
                b["high"] = round(b["high"] + cumulative_shift, 4)
                b["low"] = round(b["low"] + cumulative_shift, 4)
                b["close"] = round(b["close"] + cumulative_shift, 4)
                b["_roll_adjustment"] = cumulative_shift
                adjusted_front.append(b)

            adjusted_segments.insert(0, adjusted_front)

        # Flatten list of segments
        continuous_series = []
        for segment in adjusted_segments:
            continuous_series.extend(segment)
        return continuous_series

    @staticmethod
    def adjust_ratio_multiplicative(contract_segments: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Applies Ratio (Multiplicative) back-adjustment.
        Preserves percentage returns across rolls.
        
        Calculates ratio factor = Next_Contract_Open / Front_Contract_Close,
        and multiplies all older contracts backwards.
        """
        if not contract_segments:
            return []
        if len(contract_segments) == 1:
            return contract_segments[0]

        adjusted_segments = [list(contract_segments[-1])]
        cumulative_ratio = 1.0

        for i in range(len(contract_segments) - 2, -1, -1):
            front_segment = contract_segments[i]
            next_segment = contract_segments[i + 1]

            if not front_segment or not next_segment:
                continue

            front_close = front_segment[-1]["close"]
            next_open = next_segment[0]["open"]

            if front_close > 0:
                ratio = next_open / front_close
            else:
                ratio = 1.0

            cumulative_ratio *= ratio

            adjusted_front = []
            for bar in front_segment:
                b = dict(bar)
                b["open"] = round(b["open"] * cumulative_ratio, 4)
                b["high"] = round(b["high"] * cumulative_ratio, 4)
                b["low"] = round(b["low"] * cumulative_ratio, 4)
                b["close"] = round(b["close"] * cumulative_ratio, 4)
                b["_roll_ratio"] = cumulative_ratio
                adjusted_front.append(b)

            adjusted_segments.insert(0, adjusted_front)

        continuous_series = []
        for segment in adjusted_segments:
            continuous_series.extend(segment)
        return continuous_series
