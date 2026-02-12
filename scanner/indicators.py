from enum import Enum
from dataclasses import dataclass

import pandas as pd


class CrossType(str, Enum):
    BODY_CROSS_UP = "body_cross_up"
    BODY_CROSS_DOWN = "body_cross_down"
    WICK_CROSS_UP = "wick_cross_up"
    WICK_CROSS_DOWN = "wick_cross_down"


@dataclass
class CrossSignal:
    symbol: str
    cross_type: CrossType
    price_close: float
    ema_value: float
    timeframe: str
    source: str  # "binance" or "twelvedata"


def calculate_ema(df: pd.DataFrame, period: int = 13) -> pd.DataFrame:
    """Add EMA column to DataFrame. Expects a 'close' column."""
    df = df.copy()
    df["ema"] = df["close"].ewm(span=period, adjust=False).mean()
    return df


def detect_crosses(
    df: pd.DataFrame, symbol: str, source: str, timeframe: str = "1h"
) -> list[CrossSignal]:
    """
    Detect EMA13 crosses on the last two candles.

    Expects DataFrame with columns: open, high, low, close, ema
    sorted chronologically (oldest first). Only examines the last
    two rows (previous candle + current candle).

    Body cross takes priority over wick cross to avoid duplicates.
    """
    signals: list[CrossSignal] = []

    if len(df) < 2:
        return signals

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    prev_close = prev["close"]
    prev_ema = prev["ema"]
    curr_close = curr["close"]
    curr_ema = curr["ema"]
    curr_high = curr["high"]
    curr_low = curr["low"]

    # --- Body Cross (Close Cross) ---
    # Cross Up: previous close was at or below EMA, current close is above
    if prev_close <= prev_ema and curr_close > curr_ema:
        signals.append(
            CrossSignal(
                symbol=symbol,
                cross_type=CrossType.BODY_CROSS_UP,
                price_close=curr_close,
                ema_value=curr_ema,
                timeframe=timeframe,
                source=source,
            )
        )
        return signals

    # Cross Down: previous close was at or above EMA, current close is below
    if prev_close >= prev_ema and curr_close < curr_ema:
        signals.append(
            CrossSignal(
                symbol=symbol,
                cross_type=CrossType.BODY_CROSS_DOWN,
                price_close=curr_close,
                ema_value=curr_ema,
                timeframe=timeframe,
                source=source,
            )
        )
        return signals

    # --- Wick Cross (Pierce) ---
    # Current candle straddles EMA (high above, low below)
    # AND previous candle was entirely on one side (not already straddling)
    curr_straddles = curr_high > curr_ema and curr_low < curr_ema
    prev_was_below = prev_close <= prev_ema and prev["high"] <= prev_ema
    prev_was_above = prev_close >= prev_ema and prev["low"] >= prev_ema

    if curr_straddles:
        if prev_was_below:
            signals.append(
                CrossSignal(
                    symbol=symbol,
                    cross_type=CrossType.WICK_CROSS_UP,
                    price_close=curr_close,
                    ema_value=curr_ema,
                    timeframe=timeframe,
                    source=source,
                )
            )
        elif prev_was_above:
            signals.append(
                CrossSignal(
                    symbol=symbol,
                    cross_type=CrossType.WICK_CROSS_DOWN,
                    price_close=curr_close,
                    ema_value=curr_ema,
                    timeframe=timeframe,
                    source=source,
                )
            )

    return signals
