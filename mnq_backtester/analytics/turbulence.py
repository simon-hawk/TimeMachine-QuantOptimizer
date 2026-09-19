"""
Institutional Risk & Turbulence Diagnostics.
Implements Financial Turbulence Index (Mahalanobis Distance), Volatility Shock Detection,
and Drawdown Attribution analytics.
"""

from typing import List, Dict, Any, Optional
import numpy as np

class TurbulenceDiagnostics:
    """
    Computes statistical distance of daily returns from historical distribution
    to identify outlier stress events and regime transitions.
    """

    @staticmethod
    def calculate_turbulence(
        returns: List[float],
        lookback: int = 60,
        threshold_sigma: float = 2.5
    ) -> Dict[str, Any]:
        """
        Calculates rolling Mahalanobis-like statistical distance for 1D or multi-asset returns.
        """
        if len(returns) < lookback:
            return {
                "avg_turbulence": 1.0,
                "max_turbulence": 1.0,
                "turbulent_days_pct": 0.0,
                "high_turbulence_periods": 0,
                "turbulence_series": [1.0] * len(returns)
            }

        r_arr = np.array(returns)
        turb_series = [1.0] * len(returns)
        high_turb_count = 0

        for i in range(lookback, len(returns)):
            window = r_arr[i - lookback:i]
            mean_w = np.mean(window)
            var_w = np.var(window, ddof=1)
            if var_w > 1e-8:
                curr_val = r_arr[i]
                diff = curr_val - mean_w
                # 1D Mahalanobis distance = abs(diff) / std
                dist = abs(diff) / np.sqrt(var_w)
                turb_series[i] = round(float(dist), 2)
                if dist >= threshold_sigma:
                    high_turb_count += 1
            else:
                turb_series[i] = 1.0

        valid_series = turb_series[lookback:]
        avg_turb = float(np.mean(valid_series)) if valid_series else 1.0
        max_turb = float(np.max(valid_series)) if valid_series else 1.0
        pct_turb = float((high_turb_count / max(1, len(valid_series))) * 100.0)

        return {
            "avg_turbulence": round(avg_turb, 2),
            "max_turbulence": round(max_turb, 2),
            "turbulent_days_pct": round(pct_turb, 2),
            "high_turbulence_periods": high_turb_count,
            "turbulence_series": turb_series
        }
