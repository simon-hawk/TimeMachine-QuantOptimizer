"""
Base Strategy Interface for MNQ Quantitative Strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..core.order import Order, Position

class BaseStrategy(ABC):
    """
    Abstract base class for all MNQ algorithmic trading strategies.
    """

    def __init__(self, name: str = "BaseStrategy", parameters: Optional[Dict[str, Any]] = None):
        self.name = name
        self.parameters = parameters or {}

    @abstractmethod
    def on_bar(
        self,
        bar_index: int,
        bars: List[Dict[str, Any]],
        active_position: Optional[Position]
    ) -> Optional[Order]:
        """
        Called on every bar in sequential order.
        Returns an Order object if an entry signal is generated, or None.
        """
        pass
