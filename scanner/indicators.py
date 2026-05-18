from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Signal:
    symbol: str
    direction: str       # "BUY" / "SELL" for ema_cross  |  "GREEN" / "RED" for mt_flip
    source: str          # "mexc" or "yfinance"
    close_price: float
    ema5: float
    ema13: float
    ema62: float
    candle_ts: str = ""            # ISO timestamp of the signal candle, used for deduplication
    signal_type: str = "ema_cross" # "ema_cross" or "mt_flip"
    mt_bull: bool = True           # Megatrend state at signal time


def _heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Convert regular OHLCV to Heikin-Ashi OHLCV. Expects open/high/low/close columns."""
    n    = len(df)
    ha_c = ((df["open"] + df["high"] + df["low"] + df["close"]) / 4).values
    ha_o = np.empty(n)
    ha_o[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    for i in range(1, n):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2
    ha           = df.copy()
    ha["close"]  = ha_c
    ha["open"]   = ha_o
    ha["high"]   = np.maximum(df["high"].values, np.maximum(ha_o, ha_c))
    ha["low"]    = np.minimum(df["low"].values,  np.minimum(ha_o, ha_c))
    return ha


def calculate_emas(df: pd.DataFrame, fast: int = 5, mid: int = 13, slow: int = 62) -> pd.DataFrame:
    """Add e5, e13, e62 EMA columns to DataFrame. Expects a 'close' column."""
    df = df.copy()
    df["e5"]  = df["close"].ewm(span=fast, adjust=False).mean()
    df["e13"] = df["close"].ewm(span=mid,  adjust=False).mean()
    df["e62"] = df["close"].ewm(span=slow, adjust=False).mean()
    return df


def calculate_supertrend(
    df: pd.DataFrame,
    atr_len: int = 14,
    multiplier: float = 2.5,
    use_ha: bool = False,
    atr_type: str = "rma",
) -> pd.DataFrame:
    """
    Supertrend indicator with optional Heikin-Ashi smoothing and SMA ATR.

    use_ha=True + atr_type='sma' gives the best empirical match to the private
    Megatrend indicator timing (tested against BTCUSDT.P flip times on MEXC).

    mt_bull = True  → Megatrend green (bullish)
    mt_bull = False → Megatrend red   (bearish)
    """
    work  = _heikin_ashi(df) if use_ha else df.copy()
    high  = work["high"].values
    low   = work["low"].values
    close = work["close"].values
    n     = len(work)

    tr_vals = pd.concat([
        work["high"] - work["low"],
        (work["high"] - work["close"].shift(1)).abs(),
        (work["low"]  - work["close"].shift(1)).abs(),
    ], axis=1).max(axis=1).values

    if atr_type == "sma":
        atr_series = pd.Series(tr_vals).rolling(atr_len, min_periods=1).mean().values
    else:
        # Wilder's RMA (com = period - 1 → alpha = 1/period, matches ta.atr)
        atr_series = pd.Series(tr_vals).ewm(com=atr_len - 1, adjust=False).mean().values

    hlc3        = (high + low + close) / 3.0
    basic_upper = hlc3 + multiplier * atr_series
    basic_lower = hlc3 - multiplier * atr_series

    final_upper = [0.0] * n
    final_lower = [0.0] * n
    bull        = [True] * n

    for i in range(n):
        if i == 0:
            final_upper[i] = basic_upper[i]
            final_lower[i] = basic_lower[i]
            bull[i]        = True
            continue

        # Upper band ratchets down only; resets when price closes above it
        if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Lower band ratchets up only; resets when price closes below it
        if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Flip direction when price crosses the trailing stop
        if not bull[i - 1] and close[i] > final_upper[i]:
            bull[i] = True
        elif bull[i - 1] and close[i] < final_lower[i]:
            bull[i] = False
        else:
            bull[i] = bull[i - 1]

    out           = df.copy()
    out["mt_bull"] = bull
    out["mt_bear"] = [not b for b in bull]
    return out


def detect_signal(
    df: pd.DataFrame,
    symbol: str,
    source: str,
    fast: int = 5,
    mid: int = 13,
    slow: int = 62,
    atr_len: int = 14,
    multiplier: float = 2.5,
) -> Signal | None:
    """
    Evaluate the 5/13/62 EMA Cloud signal on the last candle.
    Fires when EMA5 crosses EMA13 with EMA62 as structural trend filter.
    Megatrend state is included as context (confirmed ✓ / early entry).
    """
    min_rows = slow + atr_len + 2
    if len(df) < min_rows:
        return None

    df = calculate_emas(df, fast, mid, slow)
    df = calculate_supertrend(df, atr_len, multiplier)

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    # PineScript: ta.crossover(e5, e13) → prev e5 <= e13 AND curr e5 > e13
    cross_up   = (prev["e5"] <= prev["e13"]) and (curr["e5"] > curr["e13"])
    cross_down = (prev["e5"] >= prev["e13"]) and (curr["e5"] < curr["e13"])

    ema_bull = (
        cross_up
        and curr["e62"] < curr["e5"]
        and curr["e62"] < curr["e13"]
        and curr["close"] > curr["e62"]
    )
    ema_bear = (
        cross_down
        and curr["e62"] > curr["e5"]
        and curr["e62"] > curr["e13"]
        and curr["close"] < curr["e62"]
    )

    mt_bull_state = bool(curr["mt_bull"])
    ts = str(curr["timestamp"]) if "timestamp" in curr.index else ""

    if ema_bull:
        return Signal(
            symbol=symbol, direction="BUY", source=source,
            close_price=float(curr["close"]),
            ema5=float(curr["e5"]), ema13=float(curr["e13"]), ema62=float(curr["e62"]),
            candle_ts=ts, signal_type="ema_cross", mt_bull=mt_bull_state,
        )

    if ema_bear:
        return Signal(
            symbol=symbol, direction="SELL", source=source,
            close_price=float(curr["close"]),
            ema5=float(curr["e5"]), ema13=float(curr["e13"]), ema62=float(curr["e62"]),
            candle_ts=ts, signal_type="ema_cross", mt_bull=mt_bull_state,
        )

    return None


def detect_mt_flip_signal(
    df: pd.DataFrame,
    symbol: str,
    source: str,
    atr_len: int = 14,
    multiplier: float = 2.5,
) -> Signal | None:
    """
    Detect a Megatrend (Supertrend) colour flip on the last two CLOSED candles.

    direction="GREEN" → Megatrend flipped from Red  to Green (bearish → bullish)
    direction="RED"   → Megatrend flipped from Green to Red   (bullish → bearish)

    Only closed candles are examined — the current forming candle is excluded
    because its close price changes every minute and can cause false flips.
    Deduplication prevents re-alerting the same flip on the same candle.
    """
    if len(df) < atr_len + 2:
        return None

    # HA candles + SMA ATR give the best empirical match to the Megatrend flip times
    df = calculate_supertrend(df, atr_len, multiplier, use_ha=True, atr_type="sma")

    # Drop the forming (current) candle — only trust closed candles
    now    = pd.Timestamp.now(tz="UTC")
    col    = df["timestamp"]
    if col.dt.tz is None:
        now = now.replace(tzinfo=None)
    closed = df[col + pd.Timedelta(hours=1) <= now]

    if len(closed) < 2:
        return None

    prev  = closed.iloc[-2]
    curr  = closed.iloc[-1]
    ts    = str(curr["timestamp"]) if "timestamp" in curr.index else ""
    close = float(curr["close"])

    # Red → Green flip
    if not prev["mt_bull"] and curr["mt_bull"]:
        return Signal(
            symbol=symbol, direction="GREEN", source=source,
            close_price=close, ema5=0.0, ema13=0.0, ema62=0.0,
            candle_ts=ts, signal_type="mt_flip", mt_bull=True,
        )

    # Green → Red flip
    if prev["mt_bull"] and not curr["mt_bull"]:
        return Signal(
            symbol=symbol, direction="RED", source=source,
            close_price=close, ema5=0.0, ema13=0.0, ema62=0.0,
            candle_ts=ts, signal_type="mt_flip", mt_bull=False,
        )

    return None
