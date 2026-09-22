"""
Free Market Data Fetcher & Provider for QuantOptimizer.
Provides zero-setup historical and real-time market data access
using Yahoo Finance, Stooq, and public REST endpoints without requiring private API keys.
"""

import os
import csv
import json
import logging
import urllib.request
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

logger = logging.getLogger("PublicDataFetcher")

class PublicDataFetcher:
    """
    Public data downloader supporting equities, index futures proxies,
    ETFs, and crypto benchmarks.
    """

    DEFAULT_SYMBOLS = {
        "MNQ": "MNQ=F",
        "NQ": "NQ=F",
        "ES": "ES=F",
        "QQQ": "QQQ",
        "SPY": "SPY",
        "BTC": "BTC-USD",
        "ETH": "ETH-USD"
    }

    @staticmethod
    def fetch_yahoo(
        symbol: str = "QQQ",
        period_days: int = 365,
        interval: str = "1d"
    ) -> List[Dict[str, Any]]:
        """
        Fetch OHLCV candles from Yahoo Finance query endpoint.
        """
        sym = PublicDataFetcher.DEFAULT_SYMBOLS.get(symbol.upper(), symbol)
        
        # Try yfinance package if installed
        try:
            import yfinance as yf
            ticker = yf.Ticker(sym)
            df = ticker.history(period=f"{min(period_days, 730)}d", interval=interval)
            if not df.empty:
                candles = []
                for idx, row in df.iterrows():
                    dt = idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else idx
                    candles.append({
                        "date": dt,
                        "timestamp": dt,
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": int(row.get("Volume", 0))
                    })
                logger.info("Successfully fetched %d bars for %s via yfinance", len(candles), sym)
                return candles
        except Exception as e:
            logger.debug("yfinance direct module fetch fallback: %s", e)

        # Fallback to direct Yahoo Finance CSV download
        end_time = int(datetime.now().timestamp())
        start_time = int((datetime.now() - timedelta(days=period_days)).timestamp())
        url = (
            f"https://query1.finance.yahoo.com/v7/finance/download/{sym}"
            f"?period1={start_time}&period2={end_time}&interval={interval}&events=history"
        )
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                lines = response.read().decode("utf-8").splitlines()
            
            reader = csv.DictReader(lines)
            candles = []
            for r in reader:
                if not r.get("Close") or r["Close"] in ("null", ""):
                    continue
                dt = datetime.strptime(r["Date"], "%Y-%m-%d")
                candles.append({
                    "date": dt,
                    "timestamp": dt,
                    "open": float(r["Open"]),
                    "high": float(r["High"]),
                    "low": float(r["Low"]),
                    "close": float(r["Close"]),
                    "volume": int(float(r["Volume"])) if r.get("Volume") not in ("null", "", None) else 0
                })
            return candles
        except Exception as ex:
            logger.warning("Direct download failed for %s: %s", sym, ex)
            return []

    @staticmethod
    def fetch_stooq(symbol: str = "qqq.us") -> List[Dict[str, Any]]:
        """
        Fetch daily bars from Stooq free historical quotes.
        """
        url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d"
        headers = {"User-Agent": "QuantOptimizer/1.0"}
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                lines = resp.read().decode("utf-8").splitlines()
            reader = csv.DictReader(lines)
            bars = []
            for r in reader:
                if not r.get("Close"):
                    continue
                dt = datetime.strptime(r["Date"], "%Y-%m-%d")
                bars.append({
                    "date": dt,
                    "timestamp": dt,
                    "open": float(r["Open"]),
                    "high": float(r["High"]),
                    "low": float(r["Low"]),
                    "close": float(r["Close"]),
                    "volume": int(float(r["Volume"])) if r.get("Volume") else 0
                })
            bars.sort(key=lambda b: b["timestamp"])
            return bars
        except Exception as e:
            logger.warning("Stooq download error for %s: %s", symbol, e)
            return []
