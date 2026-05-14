import asyncio
import logging
import time
from functools import partial

import httpx
import pandas as pd
import yfinance as yf

from scanner.config import settings

logger = logging.getLogger(__name__)

MEXC_CONTRACT_BASE = "https://contract.mexc.com"


async def fetch_mexc_ohlcv(
    symbol: str, interval: str = "Min60", limit: int = 100
) -> pd.DataFrame | None:
    """
    Fetch OHLCV data for a single MEXC Perpetual Futures symbol.
    Symbol format: BTC_USDT  (MEXC Contract API format)
    Returns DataFrame columns: timestamp, open, high, low, close, volume
    sorted oldest-first.
    """
    hours_back = limit + 5
    start_ts = int(time.time()) - hours_back * 3600

    url = f"{MEXC_CONTRACT_BASE}/api/v1/contract/kline/{symbol}"
    params = {"interval": interval, "start": start_ts}

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            body = resp.json()

            if not body.get("success"):
                logger.error(f"MEXC kline error for {symbol}: {body}")
                return None

            data = body.get("data", {})
            times  = data.get("time",  [])
            opens  = data.get("open",  [])
            highs  = data.get("high",  [])
            lows   = data.get("low",   [])
            closes = data.get("close", [])
            vols   = data.get("vol",   [])

            if not times:
                logger.warning(f"MEXC returned empty kline data for {symbol}")
                return None

            df = pd.DataFrame({
                "timestamp": pd.to_datetime(times, unit="s", utc=True),
                "open":   pd.Series(opens,  dtype=float),
                "high":   pd.Series(highs,  dtype=float),
                "low":    pd.Series(lows,   dtype=float),
                "close":  pd.Series(closes, dtype=float),
                "volume": pd.Series(vols,   dtype=float).fillna(0.0),
            })

            df = df.sort_values("timestamp").reset_index(drop=True)
            return df

        except Exception as e:
            logger.error(f"Failed to fetch MEXC OHLCV for {symbol}: {e}")
            return None


async def fetch_all_mexc(
    symbols: list[str], interval: str = "Min60", limit: int = 100
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for multiple MEXC perpetual futures symbols concurrently.
    Uses a semaphore to avoid hammering the public API.
    """
    semaphore = asyncio.Semaphore(5)
    results: dict[str, pd.DataFrame] = {}

    async def _fetch_one(sym: str) -> None:
        async with semaphore:
            df = await fetch_mexc_ohlcv(sym, interval, limit)
            if df is not None:
                results[sym] = df
            await asyncio.sleep(0.2)

    await asyncio.gather(*[_fetch_one(sym) for sym in symbols])
    return results


# Maps internal symbol format (EUR/USD) to Yahoo Finance ticker (EURUSD=X)
_YF_SYMBOL_MAP: dict[str, str] = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "XAU/USD": "GC=F",      # Gold Futures — XAUUSD=X is dead on Yahoo Finance
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X",
    "NZD/USD": "NZDUSD=X",
    "USD/CHF": "USDCHF=X",
}


def _yf_fetch_sync(yf_symbol: str, limit: int) -> pd.DataFrame | None:
    """Synchronous yfinance download — called via run_in_executor."""
    try:
        df = yf.download(
            yf_symbol,
            period="8d",
            interval="1h",
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            return None

        # Flatten MultiIndex columns produced by some yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df = df.rename(columns={
            "Datetime": "timestamp",
            "Open":     "open",
            "High":     "high",
            "Low":      "low",
            "Close":    "close",
            "Volume":   "volume",
        })

        # Ensure UTC-aware timestamps
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Drop the forming (current) candle — its close is not final yet
        now = pd.Timestamp.now(tz="UTC")
        df = df[df["timestamp"] + pd.Timedelta(hours=1) <= now]

        return df.tail(limit).reset_index(drop=True)

    except Exception as e:
        logger.error(f"yfinance sync fetch error for {yf_symbol}: {e}")
        return None


async def fetch_yfinance_ohlcv(symbol: str, limit: int = 100) -> pd.DataFrame | None:
    """
    Fetch H1 OHLCV from Yahoo Finance for a forex/commodity symbol.
    No API key required.
    """
    yf_symbol = _YF_SYMBOL_MAP.get(symbol, symbol.replace("/", "") + "=X")
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(_yf_fetch_sync, yf_symbol, limit))


async def fetch_all_yfinance(
    symbols: list[str],
    limit: int = 100,
) -> dict[str, pd.DataFrame]:
    """
    Fetch H1 OHLCV for multiple forex symbols from Yahoo Finance concurrently.
    No API key or rate-limit concerns.
    """
    results: dict[str, pd.DataFrame] = {}

    async def _fetch_one(sym: str) -> None:
        df = await fetch_yfinance_ohlcv(sym, limit)
        if df is not None:
            results[sym] = df

    await asyncio.gather(*[_fetch_one(sym) for sym in symbols])
    return results
