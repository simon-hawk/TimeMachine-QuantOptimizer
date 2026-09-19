"""
Crisis and Recession Market Regimes Library.
Pre-configured historical market crash windows (2000-2026) for extreme stress-testing.
"""

from dataclasses import dataclass
from datetime import datetime, date
from typing import List, Dict, Any, Optional

@dataclass(frozen=True)
class MarketRegimeWindow:
    id: str
    name: str
    start_date: date
    end_date: date
    peak_drawdown_pct: float
    vix_peak: float
    description: str

class CrisisRegimeLibrary:
    """
    Catalog of historical market crisis, crash, and recession time periods
    calibrated for Nasdaq-100 / MNQ stress testing.
    """

    REGIMES: Dict[str, MarketRegimeWindow] = {
        "DOT_COM_CRASH": MarketRegimeWindow(
            id="DOT_COM_CRASH",
            name="2000-2002 Dot-Com Tech Bubble Burst",
            start_date=date(2000, 3, 10),
            end_date=date(2002, 10, 9),
            peak_drawdown_pct=-83.0,
            vix_peak=49.0,
            description="Massive tech deleveraging, multi-year bear market with violent bear rallies."
        ),
        "GFC_2008": MarketRegimeWindow(
            id="GFC_2008",
            name="2007-2009 Global Financial Crisis",
            start_date=date(2007, 10, 31),
            end_date=date(2009, 3, 9),
            peak_drawdown_pct=-54.0,
            vix_peak=89.5,
            description="Systemic banking failure, Lehman Brothers collapse, extreme liquidity freeze."
        ),
        "FLASH_CRASH_2010": MarketRegimeWindow(
            id="FLASH_CRASH_2010",
            name="May 2010 Intraday Flash Crash",
            start_date=date(2010, 5, 3),
            end_date=date(2010, 5, 28),
            peak_drawdown_pct=-10.0,
            vix_peak=48.0,
            description="Rapid intraday liquidity vacuum and algorithmic cascade."
        ),
        "DEBT_DOWNGRADE_2011": MarketRegimeWindow(
            id="DEBT_DOWNGRADE_2011",
            name="2011 US Sovereign Debt Downgrade",
            start_date=date(2011, 7, 22),
            end_date=date(2011, 10, 4),
            peak_drawdown_pct=-16.8,
            vix_peak=48.0,
            description="Standard & Poor's US debt downgrade, European debt contagion."
        ),
        "VOLMAGEDDON_2018": MarketRegimeWindow(
            id="VOLMAGEDDON_2018",
            name="Feb 2018 Volmageddon Shock",
            start_date=date(2018, 1, 26),
            end_date=date(2018, 3, 29),
            peak_drawdown_pct=-12.5,
            vix_peak=50.3,
            description="XIV inverse-volatility product termination, sudden explosive volatility spike."
        ),
        "TECH_SELLOFF_Q4_2018": MarketRegimeWindow(
            id="TECH_SELLOFF_Q4_2018",
            name="Q4 2018 Fed Tightening & Tech Selloff",
            start_date=date(2018, 10, 1),
            end_date=date(2018, 12, 24),
            peak_drawdown_pct=-23.8,
            vix_peak=36.0,
            description="Aggressive rate hikes and trade tensions leading to steep Christmas Eve selloff."
        ),
        "COVID_CRASH_2020": MarketRegimeWindow(
            id="COVID_CRASH_2020",
            name="Feb-Mar 2020 COVID-19 Global Pandemic Crash",
            start_date=date(2020, 2, 19),
            end_date=date(2020, 3, 23),
            peak_drawdown_pct=-30.1,
            vix_peak=85.47,
            description="Fastest 30% drop in market history, limit-down circuit breakers triggered multiple times."
        ),
        "FED_RATE_HIKE_BEAR_2022": MarketRegimeWindow(
            id="FED_RATE_HIKE_BEAR_2022",
            name="2022 Inflation & Rapid Rate Hike Bear Market",
            start_date=date(2022, 1, 3),
            end_date=date(2022, 12, 28),
            peak_drawdown_pct=-35.6,
            vix_peak=38.9,
            description="Persistent 12-month downtrend driven by 75bps rate hikes; breakout strategies failed frequently."
        ),
        "JAPAN_CARRY_UNWIND_2024": MarketRegimeWindow(
            id="JAPAN_CARRY_UNWIND_2024",
            name="Aug 2024 Yen Carry Trade Liquidation",
            start_date=date(2024, 7, 16),
            end_date=date(2024, 8, 8),
            peak_drawdown_pct=-15.8,
            vix_peak=65.7,
            description="Bank of Japan rate hike triggering global leverage unwind and single-day VIX explosion."
        )
    }

    @classmethod
    def get_all_regimes(cls) -> List[MarketRegimeWindow]:
        """Returns all pre-configured crisis regimes."""
        return list(cls.REGIMES.values())

    @classmethod
    def get_regime(cls, regime_id: str) -> Optional[MarketRegimeWindow]:
        """Retrieves a specific crisis window by key."""
        return cls.REGIMES.get(regime_id.upper())

    @classmethod
    def slice_by_regime(cls, bars: List[Dict[str, Any]], regime_id: str) -> List[Dict[str, Any]]:
        """
        Slices a dataset to only include bars falling within the designated crisis regime.
        """
        regime = cls.get_regime(regime_id)
        if not regime:
            raise ValueError(f"Unknown regime ID: {regime_id}. Valid IDs: {list(cls.REGIMES.keys())}")

        sliced = []
        for b in bars:
            raw_time = b.get("timestamp") or b.get("time_ny") or b.get("date")
            if isinstance(raw_time, datetime):
                b_date = raw_time.date()
            elif isinstance(raw_time, date):
                b_date = raw_time
            else:
                try:
                    b_date = datetime.fromisoformat(str(raw_time)).date()
                except Exception:
                    continue

            if regime.start_date <= b_date <= regime.end_date:
                sliced.append(b)
        return sliced
