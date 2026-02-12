import asyncio
import logging

import httpx
import pandas as pd

from scanner.config import settings

logger = logging.getLogger(__name__)

BINANCE_FUTURES_BASE = "https://fapi.binance.com"


async def get_top_binance_futures_symbols(n: int = 25) -> list[str]:
    """
    Fetch top N Binance USDS-M Futures pairs by 24h quote volume.
    Returns raw Binance symbols: ['BTCUSDT', 'ETHUSDT', ...]
    No API key required for public data.
    """
    url = f"{BINANCE_FUTURES_BASE}/fapi/v1/ticker/24hr"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        tickers = resp.json()

    # Filter for USDT-margined pairs and sort by quote volume
    usdt_tickers = [
        t for t in tickers
        if t["symbol"].endswith("USDT") and float(t.get("quoteVolume", 0)) > 0
    ]
    usdt_tickers.sort(key=lambda t: float(t["quoteVolume"]), reverse=True)

    return [t["symbol"] for t in usdt_tickers[:n]]


async def fetch_binance_ohlcv(
    symbol: str, interval: str = "1h", limit: int = 50
) -> pd.DataFrame | None:
    """
    Fetch OHLCV data for a single Binance USDS-M Futures symbol via REST.
    Returns DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    url = f"{BINANCE_FUTURES_BASE}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            raw = resp.json()

            if not raw:
                return None

            # Binance kline format: [open_time, open, high, low, close, volume, ...]
            df = pd.DataFrame(
                raw,
                columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "_ct", "_qav", "_nt", "_tbbav", "_tbqav", "_ignore",
                ],
            )
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch Binance OHLCV for {symbol}: {e}")
            return None


async def fetch_all_binance(
    symbols: list[str], interval: str = "1h", limit: int = 50
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for multiple Binance symbols concurrently.
    Uses a semaphore to limit concurrent requests.
    """
    semaphore = asyncio.Semaphore(5)
    results: dict[str, pd.DataFrame] = {}

    async def _fetch_one(sym: str) -> None:
        async with semaphore:
            df = await fetch_binance_ohlcv(sym, interval, limit)
            if df is not None:
                results[sym] = df
            await asyncio.sleep(0.1)

    await asyncio.gather(*[_fetch_one(sym) for sym in symbols])
    return results


async def fetch_twelvedata_ohlcv(
    symbol: str, interval: str = "1h", outputsize: int = 50
) -> pd.DataFrame | None:
    """
    Fetch OHLCV data from Twelve Data REST API.
    Returns DataFrame with columns: timestamp, open, high, low, close, volume
    sorted oldest-first.
    """
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": settings.twelvedata_api_key,
        "format": "JSON",
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
                df["volume"] = pd.to_numeric(
                    df["volume"], errors="coerce"
                ).fillna(0)
            else:
                df["volume"] = 0.0

            df["timestamp"] = pd.to_datetime(df["datetime"])
            df = df.drop(columns=["datetime"])
            # Twelve Data returns newest first; reverse to oldest first
            df = df.iloc[::-1].reset_index(drop=True)
            return df

        except Exception as e:
            logger.error(f"Failed to fetch Twelve Data for {symbol}: {e}")
            return None


async def fetch_all_twelvedata(
    symbols: list[str], interval: str = "1h", outputsize: int = 50
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for multiple Twelve Data symbols sequentially.
    Respects the free tier rate limit (8 req/min) with 10s delay.
    """
    results: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols):
        df = await fetch_twelvedata_ohlcv(sym, interval, outputsize)
        if df is not None:
            results[sym] = df
        # Rate limit delay (skip after last symbol)
        if i < len(symbols) - 1:
            await asyncio.sleep(10.0)
    return results
