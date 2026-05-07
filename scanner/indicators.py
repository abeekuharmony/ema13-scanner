from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import pandas as pd


@dataclass
class Signal:
    symbol: str
    direction: str       # "BUY" or "SELL"
    source: str          # "mexc" or "twelvedata"
    close_price: float
    ema5: float
    ema13: float
    ema62: float
    candle_ts: str = ""             # ISO timestamp of the signal candle, used for deduplication
    signal_type: str = "ema_cross"  # "ema_cross" or "body_cross"
    mt_bull: bool = True            # actual Megatrend state at signal time (informational)


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
) -> pd.DataFrame:
    """
    Supertrend indicator — matches PineScript ta.supertrend(multiplier, atr_len).
    This is the 'Simple Supertrend' mode used by the real Megatrend (jaggedsoft/SharkCIA).

    The line acts as a trailing ATR stop:
      - Bullish: close is above the lower band (green Megatrend)
      - Bearish: close is below the upper band (red Megatrend)
    Flips direction whenever price crosses the trailing stop line.
    Always either bull or bear — never neutral — matching observed 3-5 flips/day/symbol.

    mt_bull = True  → Megatrend green (bullish)
    mt_bear = True  → Megatrend red   (bearish)
    """
    df = df.copy()
    high   = df["high"].values
    low    = df["low"].values
    close  = df["close"].values
    n      = len(df)

    # Wilder's ATR (com = period - 1  →  alpha = 1/period, matches ta.atr)
    prev_close = df["close"].shift(1).values
    tr_vals = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1).values
    atr_series = pd.Series(tr_vals).ewm(com=atr_len - 1, adjust=False).mean().values

    hlc3         = (high + low + close) / 3.0
    basic_upper  = hlc3 + multiplier * atr_series
    basic_lower  = hlc3 - multiplier * atr_series

    final_upper = [0.0] * n
    final_lower = [0.0] * n
    bull        = [True] * n  # True = bullish (green Megatrend)

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

    df["mt_bull"] = bull
    df["mt_bear"] = [not b for b in bull]
    return df


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
    Evaluate the 5/13/62 EMA Cloud + Supertrend (Megatrend) signal on the last candle.

    Candle layout after fetching at HH:00 (top of hour):
      df.iloc[-1]  — curr: the candle that just closed at HH:00
      df.iloc[-2]  — prev: the candle before that

    Returns a Signal (BUY or SELL) or None if no signal fires.
    """
    min_rows = slow + atr_len + 2
    if len(df) < min_rows:
        return None

    df = calculate_emas(df, fast, mid, slow)
    df = calculate_supertrend(df, atr_len, multiplier)

    if len(df) < 2:
        return None

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    # ── EMA Cloud ────────────────────────────────────────────────────────
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

    # ── Megatrend state (informational — not a hard filter) ─────────────
    mt_bull_state = bool(curr["mt_bull"])

    # ── Signal: EMA cross + EMA62 filter only ───────────────────────────
    # Megatrend is shown in the alert as context but does not block the signal.
    # The cross fires first; Megatrend confirmation typically follows 2-4 bars later.
    ts = str(curr["timestamp"]) if "timestamp" in curr.index else ""

    if ema_bull:
        return Signal(
            symbol=symbol,
            direction="BUY",
            source=source,
            close_price=float(curr["close"]),
            ema5=float(curr["e5"]),
            ema13=float(curr["e13"]),
            ema62=float(curr["e62"]),
            candle_ts=ts,
            mt_bull=mt_bull_state,
        )

    if ema_bear:
        return Signal(
            symbol=symbol,
            direction="SELL",
            source=source,
            close_price=float(curr["close"]),
            ema5=float(curr["e5"]),
            ema13=float(curr["e13"]),
            ema62=float(curr["e62"]),
            candle_ts=ts,
            mt_bull=mt_bull_state,
        )

    return None


def detect_body_cross_signal(
    df: pd.DataFrame,
    symbol: str,
    source: str,
    mid: int = 13,
    scan_time: datetime | None = None,
) -> Signal | None:
    """
    H1 body cross: the just-closed candle's body (open→close) straddles EMA13.
    The just-closed candle is found by timestamp (current_hour - 1h), not by
    index, so the result is correct regardless of whether the API has already
    returned the new forming candle in its response.
    No EMA62 or Megatrend conditions required.
    """
    if len(df) < mid + 2:
        return None

    if scan_time is None:
        scan_time = datetime.now(timezone.utc)

    # Timestamp of the just-closed 1H candle
    current_hour   = scan_time.replace(minute=0, second=0, microsecond=0)
    just_closed_ts = current_hour - timedelta(hours=1)

    df = calculate_emas(df, fast=5, mid=mid, slow=62)

    # Match timestamp regardless of tz-awareness in the column
    col = df["timestamp"]
    if col.dt.tz is not None:
        target = just_closed_ts
    else:
        target = just_closed_ts.replace(tzinfo=None)

    mask = col == target
    if not mask.any():
        return None

    candle    = df[mask].iloc[0]
    c_open    = float(candle["open"])
    c_close   = float(candle["close"])
    e13       = float(candle["e13"])
    ts        = str(candle["timestamp"])

    if c_open < e13 and c_close > e13:
        return Signal(
            symbol=symbol, direction="BUY", source=source,
            close_price=c_close,
            ema5=float(candle["e5"]), ema13=e13, ema62=float(candle["e62"]),
            candle_ts=ts, signal_type="body_cross",
        )

    if c_open > e13 and c_close < e13:
        return Signal(
            symbol=symbol, direction="SELL", source=source,
            close_price=c_close,
            ema5=float(candle["e5"]), ema13=e13, ema62=float(candle["e62"]),
            candle_ts=ts, signal_type="body_cross",
        )

    return None
