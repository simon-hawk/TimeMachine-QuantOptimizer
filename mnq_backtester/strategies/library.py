"""
Strategy Suite Library & Auto-Loader.
Loads categorized StockSharp quantitative optimizations.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple

_LOADED_SUITE: Dict[str, Dict[str, Any]] = {}

def clean_strategy_name(fname: str) -> str:
    clean = re.sub(r'^\d+_\d+_', '', fname)
    clean = re.sub(r'_OPTIMIZED\.pine$', '', clean)
    clean = re.sub(r'\.pine$', '', clean)
    clean = clean.replace('_', ' ')
    return clean

def categorize_stocksharp_filename(fname: str) -> Tuple[str, str]:
    clean = clean_strategy_name(fname)
    f_lower = fname.lower()
    if any(k in f_lower for k in ['trend', 'supertrend', 'ma_', 'ema_', 'slope', 'breakout', 'ichimoku', 'parabolic', 'donchian']):
        category = "StockSharp: Trend & Momentum"
    elif any(k in f_lower for k in ['reversion', 'rsi', 'stochastic', 'cci', 'bollinger', 'mean_reversion', 'zscore', 'vwap']):
        category = "StockSharp: Mean Reversion & Oscillators"
    elif any(k in f_lower for k in ['volatility', 'atr', 'vix', 'squeeze', 'width', 'expansion']):
        category = "StockSharp: Volatility & Breakouts"
    elif any(k in f_lower for k in ['volume', 'obv', 'vwma', 'climax', 'spike', 'delta']):
        category = "StockSharp: Volume & Order Flow"
    elif any(k in f_lower for k in ['candle', 'engulfing', 'pinbar', 'star', 'harami', 'soldier', 'crow', 'reversal']):
        category = "StockSharp: Candlestick Patterns"
    elif any(k in f_lower for k in ['arbitrage', 'pairs', 'hurst', 'kalman', 'cointegration', 'seasonality']):
        category = "StockSharp: Quant & Statistical Arbitrage"
    else:
        category = "StockSharp: Technical Alpha Models"
    return clean, category

def load_all_strategies(force_reload: bool = False) -> Dict[str, Dict[str, Any]]:
    """Loads StockSharp library."""
    global _LOADED_SUITE
    if _LOADED_SUITE and not force_reload:
        return _LOADED_SUITE

    suite = {}

    ss_dirs = [
        Path(__file__).resolve().parent / "pine_library"
    ]
    for ss_dir in ss_dirs:
        if ss_dir.exists() and ss_dir.is_dir():
            try:
                filenames = sorted([f for f in os.listdir(str(ss_dir)) if f.endswith('.pine')])
                for fname in filenames:
                    p = ss_dir / fname
                    clean_title, category = categorize_stocksharp_filename(fname)
                    strategy_id = f"ss_{p.stem.lower()}"
                    try:
                        with open(str(p), "r", encoding="utf-8", errors="ignore") as f:
                            code_content = f.read()

                        suite[strategy_id] = {
                            "id": strategy_id,
                            "name": f"{clean_title}",
                            "category": category,
                            "description": f"StockSharp quantitative optimization model: {clean_title}",
                            "code": code_content.strip()
                        }
                    except Exception:
                        continue
            except Exception:
                pass

    _LOADED_SUITE = suite
    return _LOADED_SUITE

def get_strategy_suite(force_reload: bool = False) -> List[Dict[str, Any]]:
    """Returns metadata for all preset strategies."""
    suite = load_all_strategies(force_reload=force_reload)
    return list(suite.values())

PINE_STRATEGY_SUITE = load_all_strategies()
