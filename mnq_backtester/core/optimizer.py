"""
Parallel Grid Search and Multiprocessing Optimization Engine.
Distributes parameter combinations across CPU cores for high-throughput parameter exploration.
"""

import os
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Tuple, Optional

def _eval_param_worker(args: Tuple[Callable, Dict[str, Any], Any]) -> Dict[str, Any]:
    eval_fn, params, data = args
    try:
        score, metrics = eval_fn(params, data)
        return {
            "params": params,
            "score": score,
            "metrics": metrics,
            "status": "success"
        }
    except Exception as e:
        return {
            "params": params,
            "score": -9999.0,
            "metrics": {},
            "status": f"error: {str(e)}"
        }

class ParallelOptimizer:
    """
    Multiprocessing optimizer for systematic parameter sweeps.
    """

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or max(1, os.cpu_count() or 2)

    def grid_search(
        self,
        eval_fn: Callable[[Dict[str, Any], Any], Tuple[float, Dict[str, Any]]],
        param_grid: Dict[str, List[Any]],
        data: Any
    ) -> List[Dict[str, Any]]:
        """
        Executes parallel grid search over all cartesian combinations of param_grid.
        """
        keys = list(param_grid.keys())
        combinations = list(itertools.product(*[param_grid[k] for k in keys]))
        tasks = []
        for combo in combinations:
            p_dict = dict(zip(keys, combo))
            tasks.append((eval_fn, p_dict, data))

        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {executor.submit(_eval_param_worker, task): task for task in tasks}
            for future in as_completed(future_to_task):
                res = future.result()
                results.append(res)

        # Sort descending by optimization score
        results.sort(key=lambda r: r["score"], reverse=True)
        return results
