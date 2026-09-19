"""
MNQ Contract Specifications and System Configuration.
"""

from dataclasses import dataclass
from datetime import time
from typing import Optional

@dataclass(frozen=True)
class ContractSpec:
    symbol: str = "MNQ"
    full_name: str = "Micro E-mini Nasdaq-100 Index Futures"
    exchange: str = "CME"
    point_value: float = 2.00          # $2.00 per 1.00 full index point
    tick_size: float = 0.25            # Minimum price increment: 0.25 index points
    tick_value: float = 0.50           # $0.50 per 1 tick (point_value * tick_size)
    default_commission_per_side: float = 0.62  # $0.62 per contract per side ($1.24 round trip)
    default_slippage_ticks: float = 1.0        # Default 1 tick slippage on market/stop fills

@dataclass
class BacktestConfig:
    # Account & Capital
    initial_capital: float = 50_000.0  # Standard prop firm account / fund size
    currency: str = "USD"
    
    # Contract specs
    contract: ContractSpec = ContractSpec()
    commission_per_side: float = 0.62
    slippage_ticks: float = 1.0        # in 0.25 tick units
    
    # Execution & Risk Limits
    enable_prop_firm_rules: bool = False
    max_trailing_drawdown: float = 2_500.0   # e.g. $2,500 max trailing drawdown limit
    daily_loss_limit: float = 1_000.0        # e.g. $1,000 max daily loss
    profit_target: Optional[float] = 3_000.0 # e.g. $3,000 prop evaluation target
    
    # Sessions (New York Time)
    rth_start: time = time(9, 30)
    rth_end: time = time(16, 0)
    eod_liquidation_time: time = time(15, 59)
    allow_overnight_holds: bool = False
    
    # Monte Carlo defaults
    mc_iterations: int = 1_000
    mc_confidence_levels: tuple = (0.05, 0.25, 0.50, 0.75, 0.95)

DEFAULT_CONFIG = BacktestConfig()
