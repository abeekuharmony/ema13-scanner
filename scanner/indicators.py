from dataclasses import dataclass

import pandas as pd


@dataclass
class Signal:
    symbol: str
    direction: str      # "BUY" or "SELL"
    source: str         # "mexc" or "twelvedata"
    close_price: float
    ema5: float
    ema13: float
    ema62: float


def calculate_emas(df: pd.DataFrame, fast: int = 5, mid: int = 13, slow: int = 62) -> pd.DataFrame:
    """Add e5, e13, e62 EMA columns to DataFrame. Expects a 'close' column."""
    df = df.copy()
    df["e5"]  = df["close"].ewm(span=fast, adjust=False).mean()
    df["e13"] = df["close"].ewm(span=mid,  adjust=False).mean()
    df["e62"] = df["close"].ewm(span=slow, adjust=False).mean()
    return df


def calculate_megatrend(
    df: pd.DataFrame,
    atr_len: int = 14,
    smooth_len: int = 14,
    r_mult: float = 2.5,
    breakout_len: int = 2,
) -> pd.DataFrame:
    """
    Custom ATR Breakout Megatrend (replicates PineScript 'Custom approximation' mode).

    Source: hl2 (high + low) / 2
    Bands:  EMA(hl2, smooth_len)  ±  ATR(atr_len, Wilder) * r_mult
    mt_bull: close above upper band for ALL of the last breakout_len bars
    mt_bear: close below lower band for ALL of the last breakout_len bars

    PineScript ta.atr() uses Wilder's smoothing (com = period - 1).
    PineScript ta.ema() uses standard EMA (span = period).
    """
    df = df.copy()
    df["hl2"] = (df["high"] + df["low"]) / 2.0

    # True Range
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Wilder's ATR (matches PineScript ta.atr)
    df["_atr"] = tr.ewm(com=atr_len - 1, adjust=False).mean()

    # EMA of hl2 (standard EMA, matches PineScript ta.ema)
    df["_mt_smooth"] = df["hl2"].ewm(span=smooth_len, adjust=False).mean()

    df["_mt_upper"] = df["_mt_smooth"] + df["_atr"] * r_mult
    df["_mt_lower"] = df["_mt_smooth"] - df["_atr"] * r_mult

    above = (df["close"] > df["_mt_upper"]).astype(float)
    below = (df["close"] < df["_mt_lower"]).astype(float)

    # All of the last breakout_len bars must satisfy the condition
    df["mt_bull"] = above.rolling(breakout_len).min().fillna(0).astype(bool)
    df["mt_bear"] = below.rolling(breakout_len).min().fillna(0).astype(bool)

    return df.drop(columns=["_atr", "_mt_smooth", "_mt_upper", "_mt_lower"])


def detect_signal(
    df: pd.DataFrame,
    symbol: str,
    source: str,
    fast: int = 5,
    mid: int = 13,
    slow: int = 62,
    atr_len: int = 14,
    smooth_len: int = 14,
    r_mult: float = 2.5,
    breakout_len: int = 2,
) -> Signal | None:
    """
    Evaluate the 5/13/62 EMA Cloud + Megatrend signal on the last CLOSED candle.

    Candle layout after fetching at HH:00 (top of hour):
      df.iloc[-1]  — curr: the candle that just closed at HH:00
      df.iloc[-2]  — prev: the candle before that

    Returns a Signal (BUY or SELL) or None if no signal fires.
    """
    min_rows = slow + atr_len + breakout_len + 2
    if len(df) < min_rows:
        return None

    df = calculate_emas(df, fast, mid, slow)
    df = calculate_megatrend(df, atr_len, smooth_len, r_mult, breakout_len)

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

    # ── Megatrend ────────────────────────────────────────────────────────
    mt_bull = bool(curr["mt_bull"])
    mt_bear = bool(curr["mt_bear"])

    # ── Combined signal ──────────────────────────────────────────────────
    if ema_bull and mt_bull:
        return Signal(
            symbol=symbol,
            direction="BUY",
            source=source,
            close_price=float(curr["close"]),
            ema5=float(curr["e5"]),
            ema13=float(curr["e13"]),
            ema62=float(curr["e62"]),
        )

    if ema_bear and mt_bear:
        return Signal(
            symbol=symbol,
            direction="SELL",
            source=source,
            close_price=float(curr["close"]),
            ema5=float(curr["e5"]),
            ema13=float(curr["e13"]),
            ema62=float(curr["e62"]),
        )

    return None
