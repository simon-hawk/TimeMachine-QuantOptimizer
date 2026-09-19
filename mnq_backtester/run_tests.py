"""
Direct Test Runner for MNQ Backtesting Engine.
Executes all unit tests without third-party test runners.
"""

import sys
import os
from pathlib import Path
import traceback

# Ensure workspace root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def run_all():
    print("=" * 60)
    print("RUNNING MNQ BACKTESTER TEST SUITE")
    print("=" * 60)

    # 1. Validator tests
    from mnq_backtester.tests.test_validator import (
        test_clean_bars_pass_validation,
        test_ohlc_invariant_failure,
        test_tick_quantization_check
    )
    # 2. Partition tests
    from mnq_backtester.tests.test_partitions import (
        test_train_test_split_embargo,
        test_walk_forward_rolling_folds,
        test_crisis_regime_slicing
    )
    # 3. Engine tests
    from mnq_backtester.tests.test_engine import (
        test_mnq_point_multiplier_and_commission
    )
    # 4. Metrics tests
    from mnq_backtester.tests.test_metrics import (
        test_performance_metrics_math
    )
    # 5. Monte Carlo tests
    from mnq_backtester.tests.test_monte_carlo import (
        test_monte_carlo_resampling_methods,
        test_monte_carlo_orchestrator
    )
    # 6. PineScript tests
    from mnq_backtester.tests.test_pinescript import (
        test_pine_ta_indicators,
        test_pinescript_strategy_parser_and_run
    )

    tests = [
        ("test_clean_bars_pass_validation", test_clean_bars_pass_validation),
        ("test_ohlc_invariant_failure", test_ohlc_invariant_failure),
        ("test_tick_quantization_check", test_tick_quantization_check),
        ("test_train_test_split_embargo", test_train_test_split_embargo),
        ("test_walk_forward_rolling_folds", test_walk_forward_rolling_folds),
        ("test_crisis_regime_slicing", test_crisis_regime_slicing),
        ("test_mnq_point_multiplier_and_commission", test_mnq_point_multiplier_and_commission),
        ("test_performance_metrics_math", test_performance_metrics_math),
        ("test_monte_carlo_resampling_methods", test_monte_carlo_resampling_methods),
        ("test_monte_carlo_orchestrator", test_monte_carlo_orchestrator),
        ("test_pine_ta_indicators", test_pine_ta_indicators),
        ("test_pinescript_strategy_parser_and_run", test_pinescript_strategy_parser_and_run),
    ]

    passed = 0
    failed = 0

    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"RESULTS: {passed} PASSED, {failed} FAILED")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    run_all()
