"""
Institutional Portfolio Construction and Optimization Suite.
Implements Hierarchical Risk Parity (HRP) and Quasi-Diagonalization (Marcos López de Prado).
Calculates optimal contract allocations, joint portfolio equity curve, Sharpe, and correlation matrix.
"""

from typing import List, Dict, Any, Optional, Tuple
import math
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

class HierarchicalRiskParity:
    """
    Constructs an optimal risk-balanced portfolio from backtest strategy returns.
    Avoids the matrix inversion instabilities of Markowitz Mean-Variance Optimization.
    """

    @staticmethod
    def get_quasi_diag(link: np.ndarray) -> List[int]:
        """Sorts clustered items by hierarchical tree order."""
        link = link.astype(int)
        sort_ix = pd_series = [link[-1, 0], link[-1, 1]]
        num_items = link[-1, 3]
        while max(sort_ix) >= num_items:
            sort_ix0 = []
            for i in sort_ix:
                if i >= num_items:
                    row = i - num_items
                    sort_ix0.append(link[row, 0])
                    sort_ix0.append(link[row, 1])
                else:
                    sort_ix0.append(i)
            sort_ix = sort_ix0
        return sort_ix

    @staticmethod
    def get_cluster_var(cov: np.ndarray, c_items: List[int]) -> float:
        """Calculates variance of a cluster using inverse-variance weights."""
        cov_slice = cov[np.ix_(c_items, c_items)]
        ivp = 1.0 / np.diag(cov_slice)
        ivp /= ivp.sum()
        w = ivp.reshape(-1, 1)
        c_var = np.dot(np.dot(w.T, cov_slice), w)[0, 0]
        return float(c_var)

    @classmethod
    def get_rec_bipart(cls, cov: np.ndarray, sort_ix: List[int]) -> np.ndarray:
        """Recursive bisection to assign risk weights across clusters."""
        w = np.ones(len(sort_ix))
        c_items = [sort_ix]
        while len(c_items) > 0:
            c_items0 = []
            for i in c_items:
                if len(i) > 1:
                    bi = len(i) // 2
                    c1 = i[:bi]
                    c2 = i[bi:]
                    var1 = cls.get_cluster_var(cov, c1)
                    var2 = cls.get_cluster_var(cov, c2)
                    alpha = 1.0 - var1 / (var1 + var2)
                    w[c1] *= alpha
                    w[c2] *= (1.0 - alpha)
                    c_items0.append(c1)
                    c_items0.append(c2)
            c_items = c_items0
        return w

    @classmethod
    def allocate_hrp(
        cls,
        strategy_returns: Dict[str, List[float]],
        total_contracts: int = 10
    ) -> Dict[str, Any]:
        """
        Computes Hierarchical Risk Parity weights and discrete contract sizing.
        """
        names = list(strategy_returns.keys())
        n_assets = len(names)
        if n_assets < 2:
            return {
                "strategies": names,
                "weights": {n: 1.0 for n in names},
                "contracts": {n: total_contracts for n in names},
                "correlation_matrix": [[1.0]],
                "portfolio_sharpe": 0.0,
                "portfolio_net_profit": 0.0
            }

        # Align series lengths
        min_len = min(len(r) for r in strategy_returns.values())
        if min_len < 5:
            equal_w = 1.0 / n_assets
            return {
                "strategies": names,
                "weights": {n: round(equal_w, 3) for n in names},
                "contracts": {n: max(1, total_contracts // n_assets) for n in names},
                "correlation_matrix": np.eye(n_assets).tolist(),
                "portfolio_sharpe": 0.0,
                "portfolio_net_profit": 0.0
            }

        data_mat = np.array([strategy_returns[n][-min_len:] for n in names])
        cov = np.cov(data_mat)
        std_dev = np.sqrt(np.diag(cov))
        std_outer = np.outer(std_dev, std_dev)
        std_outer[std_outer == 0] = 1e-6
        corr = cov / std_outer
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)

        # Distance matrix D = sqrt(0.5 * (1 - corr))
        dist = np.sqrt(0.5 * np.maximum(0, 1.0 - corr))
        np.fill_diagonal(dist, 0.0)

        try:
            condensed_dist = squareform(dist, checks=False)
            link = linkage(condensed_dist, method='single')
            sort_ix = cls.get_quasi_diag(link)
            raw_weights = cls.get_rec_bipart(cov, sort_ix)
            raw_weights = raw_weights / np.sum(raw_weights)
        except Exception:
            # Fallback to inverse variance
            inv_diag = 1.0 / np.maximum(1e-6, np.diag(cov))
            raw_weights = inv_diag / np.sum(inv_diag)

        # Discrete contract allocations
        weights_dict = {names[i]: round(float(raw_weights[i]), 4) for i in range(n_assets)}
        contracts_dict = {}
        allocated = 0
        for i, name in enumerate(names):
            ct = max(1, int(round(raw_weights[i] * total_contracts)))
            contracts_dict[name] = ct
            allocated += ct

        # Combined portfolio equity path
        portfolio_pnl_path = []
        cum_pnl = 0.0
        portfolio_returns = []
        for t in range(min_len):
            step_dollar = sum(data_mat[i][t] * contracts_dict[names[i]] for i in range(n_assets))
            cum_pnl += step_dollar
            portfolio_pnl_path.append(round(cum_pnl, 2))
            portfolio_returns.append(step_dollar)

        p_std = np.std(portfolio_returns)
        p_mean = np.mean(portfolio_returns)
        p_sharpe = float((p_mean / p_std) * math.sqrt(252)) if p_std > 1e-6 else 0.0

        return {
            "strategies": names,
            "weights": weights_dict,
            "contracts": contracts_dict,
            "correlation_matrix": [[round(float(c), 3) for c in row] for row in corr],
            "portfolio_sharpe": round(p_sharpe, 2),
            "portfolio_net_profit": round(cum_pnl, 2),
            "portfolio_equity_curve": portfolio_pnl_path
        }
