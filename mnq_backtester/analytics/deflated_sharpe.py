"""
Deflated Sharpe Ratio (DSR), Probabilistic Sharpe Ratio (PSR), and Multiple Testing Correction.
Implements the methodology of Marcos López de Prado & David Bailey:
  - 'The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality' (Journal of Portfolio Management, 2014)
"""

import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy import stats

class DeflatedSharpeRatio:
    """
    Computes PSR and DSR to mathematically adjust Sharpe ratios for:
      1. Non-normality (Skewness and Kurtosis of returns)
      2. Track record length (Sample size T)
      3. Multiple testing (Number of strategy trials / variations tested)
    """

    @staticmethod
    def calculate_skewness_kurtosis(returns: List[float]) -> Tuple[float, float]:
        """Calculates sample skewness and kurtosis."""
        if len(returns) < 5:
            return 0.0, 3.0
        r = np.array(returns)
        mean_r = np.mean(r)
        std_r = np.std(r, ddof=1)
        if std_r < 1e-8:
            return 0.0, 3.0
        
        n = len(r)
        m3 = np.sum((r - mean_r) ** 3) / n
        m4 = np.sum((r - mean_r) ** 4) / n
        
        skewness = float(m3 / (std_r ** 3))
        kurtosis = float(m4 / (std_r ** 4))
        return skewness, kurtosis

    @staticmethod
    def probabilistic_sharpe_ratio(
        observed_sharpe: float,
        benchmark_sharpe: float = 0.0,
        n_observations: int = 252,
        skewness: float = 0.0,
        kurtosis: float = 3.0
    ) -> float:
        """
        Calculates the Probabilistic Sharpe Ratio (PSR).
        PSR(SR*) = Prob(SR > SR* | sample estimates)
        """
        if n_observations <= 2:
            return 0.5

        # Variance of the Sharpe ratio estimator
        term = 1.0 - skewness * observed_sharpe + ((kurtosis - 1.0) / 4.0) * (observed_sharpe ** 2)
        if term <= 0:
            term = 1e-6
            
        std_sr = math.sqrt(term / max(1, n_observations - 1))
        
        if std_sr < 1e-8:
            return 1.0 if observed_sharpe > benchmark_sharpe else 0.0
            
        z_stat = (observed_sharpe - benchmark_sharpe) / std_sr
        psr = float(stats.norm.cdf(z_stat))
        return psr

    @classmethod
    def expected_max_sharpe(
        cls,
        n_trials: int,
        var_trials: float = 0.5,
        benchmark_sharpe: float = 0.0
    ) -> float:
        """
        Calculates the expected maximum Sharpe ratio under the null hypothesis
        across N independent/correlated trials using Euler-Mascheroni approximation.
        """
        if n_trials <= 1:
            return benchmark_sharpe
        
        euler_mascheroni = 0.5772156649
        z_approx = (1.0 - euler_mascheroni) * stats.norm.ppf(1.0 - 1.0 / n_trials) + \
                   euler_mascheroni * stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        
        expected_max = benchmark_sharpe + math.sqrt(var_trials) * z_approx
        return float(expected_max)

    @classmethod
    def calculate_dsr(
        cls,
        observed_sharpe: float,
        returns: List[float],
        n_trials: int = 110,
        var_trials: float = 0.5,
        benchmark_sharpe: float = 0.0
    ) -> Dict[str, Any]:
        """
        Calculates the full Deflated Sharpe Ratio (DSR) metrics package.
        """
        n_obs = len(returns)
        if n_obs < 5:
            return {
                "observed_sharpe": observed_sharpe,
                "dsr_probability": 0.5,
                "psr_probability": 0.5,
                "expected_max_sharpe": benchmark_sharpe,
                "skewness": 0.0,
                "kurtosis": 3.0,
                "is_statistically_significant": False
            }

        skewness, kurtosis = cls.calculate_skewness_kurtosis(returns)
        
        # Expected max Sharpe given N trials
        sr_benchmark = cls.expected_max_sharpe(n_trials=n_trials, var_trials=var_trials, benchmark_sharpe=benchmark_sharpe)
        
        # Standard PSR vs 0
        psr = cls.probabilistic_sharpe_ratio(
            observed_sharpe=observed_sharpe,
            benchmark_sharpe=benchmark_sharpe,
            n_observations=n_obs,
            skewness=skewness,
            kurtosis=kurtosis
        )

        # DSR vs Expected Max Sharpe across trials
        dsr = cls.probabilistic_sharpe_ratio(
            observed_sharpe=observed_sharpe,
            benchmark_sharpe=sr_benchmark,
            n_observations=n_obs,
            skewness=skewness,
            kurtosis=kurtosis
        )

        is_significant = (dsr >= 0.95)

        return {
            "observed_sharpe": round(observed_sharpe, 2),
            "dsr_probability": round(dsr * 100.0, 2),
            "psr_probability": round(psr * 100.0, 2),
            "expected_max_sharpe": round(sr_benchmark, 2),
            "skewness": round(skewness, 3),
            "kurtosis": round(kurtosis, 3),
            "n_trials_corrected": n_trials,
            "is_statistically_significant": is_significant
        }
