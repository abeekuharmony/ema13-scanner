import asyncio
import logging
import time
from functools import partial

import httpx
import pandas as pd
import yfinance as yf

from scanner.config import settings

logger = logging.getLogger(__name__)

MEXC_CONTRACT_BASE  = "https://contract.mexc.com"
OANDA_PRACTICE_BASE = "https://api-fxpractice.oanda.com"


async def fetch_mexc_ohlcv(
    symbol: str, interval: str = "Min60", limit: int = 100
) -> pd.DataFrame | None:
    """
    Fetch OHLCV data for a single MEXC Perpetual Futures symbol.
    Symbol format: BTC_USDT  (MEXC Contract API format)
    Returns DataFrame columns: timestamp, open, high, low, close, volume
    sorted oldest-first.
    """
    # Compute start timestamp: go back far enough for limit hourly candles
    hours_back = limit + 5  # small buffer
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


async def fetch_twelvedata_ohlcv(
    symbol: str, interval: str = "1h", outputsize: int = 100
) -> pd.DataFrame | None:
    """
    Fetch OHLCV data from Twelve Data REST API.
    Returns DataFrame columns: timestamp, open, high, low, close, volume
    sorted oldest-first.
    """
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol":     symbol,
        "interval":   interval,
        "outputsize": outputsize,
        "apikey":     settings.twelvedata_api_key,
        "format":     "JSON",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

            if "values" not in data:
                logger.error(
                    f"Twelve Data error for {symbol}: "
                    f"{data.get('message', 'unknown error')}"
                )
                return None

            df = pd.DataFrame(data["values"])
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(float)
            if "volume" in df.columns:
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
            else:
                df["volume"] = 0.0

            df["timestamp"] = pd.to_datetime(df["datetime"])
            df = df.drop(columns=["datetime"])
            # Twelve Data returns newest-first; reverse to oldest-first
            df = df.iloc[::-1].reset_index(drop=True)
            return df

        except Exception as e:
            logger.error(f"Failed to fetch Twelve Data for {symbol}: {e}")
            return None


async def fetch_all_twelvedata(
    symbols: list[str], interval: str = "1h", outputsize: int = 100
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for multiple Twelve Data symbols sequentially.
    Respects free-tier rate limit (8 req/min) with 8s delay between calls.
    """
    results: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols):
        df = await fetch_twelvedata_ohlcv(sym, interval, outputsize)
        if df is not None:
            results[sym] = df
        if i < len(symbols) - 1:
            await asyncio.sleep(8.0)
    return results


async def fetch_oanda_ohlcv(
    symbol: str,
    granularity: str = "H1",
    count: int = 200,
    api_key: str = "",
) -> pd.DataFrame | None:
    """
    Fetch OHLCV from OANDA practice REST API v20.
    Symbol format: EUR/USD → converted to EUR_USD internally.
    Uses midpoint (M) prices — same as TradingView's OANDA feed.
    Returns DataFrame sorted oldest-first, timestamps UTC-aware.
    """
    instrument = symbol.replace("/", "_")
    url = f"{OANDA_PRACTICE_BASE}/v3/instruments/{instrument}/candles"
    params = {
        "count":       count,
        "granularity": granularity,
        "price":       "M",   # midpoint — matches TradingView
    }
    headers = {
        "Authorization":          f"Bearer {api_key}",
        "Accept-Datetime-Format": "RFC3339",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            candles = data.get("candles", [])
            if not candles:
                logger.warning(f"OANDA returned empty candles for {symbol}")
                return None

            rows = []
            for c in candles:
                mid = c.get("mid", {})
                rows.append({
                    "timestamp": pd.to_datetime(c["time"], utc=True),
                    "open":   float(mid["o"]),
                    "high":   float(mid["h"]),
                    "low":    float(mid["l"]),
                    "close":  float(mid["c"]),
                    "volume": float(c.get("volume", 0)),
                })

            df = pd.DataFrame(rows)
            df = df.sort_values("timestamp").reset_index(drop=True)
            return df

        except Exception as e:
            logger.error(f"Failed to fetch OANDA OHLCV for {symbol}: {e}")
            return None


async def fetch_all_oanda(
    symbols: list[str],
    granularity: str = "H1",
    count: int = 200,
    api_key: str = "",
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for multiple OANDA forex symbols concurrently.
    OANDA practice API allows 120 req/s — no delay needed.
    """
    semaphore = asyncio.Semaphore(8)
    results: dict[str, pd.DataFrame] = {}

    async def _fetch_one(sym: str) -> None:
        async with semaphore:
            df = await fetch_oanda_ohlcv(sym, granularity, count, api_key)
            if df is not None:
                results[sym] = df

    await asyncio.gather(*[_fetch_one(sym) for sym in symbols])
    return results


# Maps internal symbol format (EUR/USD) to Yahoo Finance ticker (EURUSD=X)
_YF_SYMBOL_MAP: dict[str, str] = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "XAU/USD": "XAUUSD=X",
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

        # yf.download with a single ticker returns a MultiIndex column when
        # auto_adjust=True in some versions — flatten to single level.
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
    No API key required. Runs sync yfinance in a thread pool.
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
