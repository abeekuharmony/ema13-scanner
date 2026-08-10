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

# ── OANDA v20 (real forex/metals/indices feed — matches the user's chart) ──
OANDA_HOSTS = {
    "practice": "https://api-fxpractice.oanda.com",
    "live":     "https://api-fxtrade.oanda.com",
}


async def fetch_oanda_ohlcv(
    instrument: str, granularity: str = "H1", count: int = 200
) -> pd.DataFrame | None:
    """
    Fetch candles from the OANDA v20 API. granularity: "M15", "H1", etc.
    Only COMPLETE (closed) candles are returned — OANDA's `complete` flag is
    the exact, authoritative version of the repaint fix. `time` is the candle
    OPEN time (UTC). Auth token comes from OANDA_API_TOKEN (never hard-coded).
    """
    token = settings.oanda_api_token
    if not token:
        logger.error("OANDA_API_TOKEN is not set — add it to the environment.")
        return None

    host = OANDA_HOSTS.get(settings.oanda_env, OANDA_HOSTS["practice"])
    url = f"{host}/v3/instruments/{instrument}/candles"
    params = {"granularity": granularity, "count": count, "price": "M"}
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            candles = [c for c in resp.json().get("candles", []) if c.get("complete")]
            if not candles:
                return None
            df = pd.DataFrame({
                "timestamp": pd.to_datetime([c["time"] for c in candles], utc=True, format="ISO8601"),
                "open":   [float(c["mid"]["o"]) for c in candles],
                "high":   [float(c["mid"]["h"]) for c in candles],
                "low":    [float(c["mid"]["l"]) for c in candles],
                "close":  [float(c["mid"]["c"]) for c in candles],
                "volume": [float(c.get("volume", 0)) for c in candles],
            })
            return df.sort_values("timestamp").reset_index(drop=True)
        except Exception as e:
            logger.error(f"OANDA fetch failed for {instrument} [{granularity}]: {e}")
            return None


async def fetch_all_oanda(
    instruments: list[str], granularity: str = "H1", count: int = 200
) -> dict[str, pd.DataFrame]:
    """Fetch OANDA candles for multiple instruments concurrently (rate-limited)."""
    semaphore = asyncio.Semaphore(5)
    results: dict[str, pd.DataFrame] = {}

    async def _one(inst: str) -> None:
        async with semaphore:
            df = await fetch_oanda_ohlcv(inst, granularity, count)
            if df is not None:
                results[inst] = df
            await asyncio.sleep(0.1)

    await asyncio.gather(*[_one(i) for i in instruments])
    return results

# MEXC interval → seconds per candle (used to drop the still-forming candle)
_INTERVAL_SECONDS = {
    "Min1": 60, "Min5": 300, "Min15": 900, "Min30": 1800,
    "Min60": 3600, "Hour4": 14400, "Hour8": 28800,
    "Day1": 86400, "Week1": 604800,
}


def _drop_forming(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Remove the still-forming candle so detectors only ever see CLOSED candles.

    Acting on the forming candle was the repaint bug: its "close" is just the
    live price, so signals fired mid-candle (EMA cross, EMA13 body cross)
    could un-cross by the candle close — phantom alerts that never appear
    on the finished chart.
    """
    secs = _INTERVAL_SECONDS.get(interval, 3600)
    now = pd.Timestamp.now(tz="UTC")
    return df[df["timestamp"] + pd.Timedelta(seconds=secs) <= now].reset_index(drop=True)


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
            # Repaint fix: signals must only ever evaluate CLOSED candles
            df = _drop_forming(df, interval)
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
