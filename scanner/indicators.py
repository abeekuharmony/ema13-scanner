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
    signal_type: str = "ema_cross" # "ema_cross" | "mt_flip" | "ema13_body_cross" | "retest_setup" | "retest_trigger" | "retest_cancel"
    mt_bull: bool = True           # Megatrend state at signal time
    open_price: float = 0.0        # candle open (used by ema13_body_cross alerts)
    entry: float = 0.0             # retest trade plan: limit entry at EMA13
    stop: float = 0.0              # retest trade plan: 1×ATR beyond entry
    tp1: float = 0.0               # retest trade plan: 1.5R target
    tp2: float = 0.0               # retest trade plan: 2R target
    atr: float = 0.0               # ATR(14) at signal time (for transposing to other feeds)


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


def calculate_jma(src: np.ndarray, length: int = 14, phase: int = 100, power: int = 1) -> np.ndarray:
    """
    Jurik Moving Average (standard public replication).

    Reverse-engineered from the SharkCIA "Megatrend Alerts" on TradingView:
    JMA(hlc3, length=14, phase=100, power=1) reproduced the indicator's
    plotted value to within 0.003 on OANDA:XAUUSD 1H (4,048.867 vs
    4,048.869 in the data window). The Megatrend line IS this JMA.
    """
    phase_ratio = 2.5 if phase > 100 else (0.5 if phase < -100 else phase / 100 + 1.5)
    beta  = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
    alpha = beta ** power
    n   = len(src)
    e0  = np.zeros(n)
    e1  = np.zeros(n)
    e2  = np.zeros(n)
    out = np.zeros(n)
    e0[0]  = src[0]
    out[0] = src[0]
    for i in range(1, n):
        e0[i]  = (1 - alpha) * src[i] + alpha * e0[i - 1]
        e1[i]  = (src[i] - e0[i]) * (1 - beta) + beta * e1[i - 1]
        e2[i]  = (e0[i] + phase_ratio * e1[i] - out[i - 1]) * (1 - alpha) ** 2 + alpha ** 2 * e2[i - 1]
        out[i] = e2[i] + out[i - 1]
    return out


def calculate_megatrend(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """
    Megatrend colour state per bar: mt_bull = JMA(hlc3) rising.

    Colour rule validated against the user's confirmed BTCUSDT.P flips
    (May 17 23:00 RED, May 18 12:00 GREEN, May 18 15:00 RED matched
    exactly; May 16 19:00 GREEN detected 1 bar early) and against the
    live OANDA:XAUUSD chart state. Flat JMA keeps the previous colour.
    """
    hlc3 = ((df["high"] + df["low"] + df["close"]) / 3).values
    j    = calculate_jma(hlc3, length)
    n    = len(j)
    bull = [True] * n
    for i in range(1, n):
        if j[i] > j[i - 1]:
            bull[i] = True
        elif j[i] < j[i - 1]:
            bull[i] = False
        else:
            bull[i] = bull[i - 1]
    out            = df.copy()
    out["jma"]     = j
    out["mt_bull"] = bull
    out["mt_bear"] = [not b for b in bull]
    return out


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
    # JMA-based Megatrend — the real formula behind the TradingView indicator,
    # same method as detect_mt_flip_signal so "confirmed ✓" agrees with the
    # flip alerts and the chart. (atr_len/multiplier kept for signature compat.)
    df = calculate_megatrend(df)

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
    # JMA needs ~40 bars to converge (len 14, alpha ≈ 0.745)
    if len(df) < 40:
        return None

    # Real Megatrend formula: JMA(hlc3, 14, phase=100) slope = colour.
    # (atr_len/multiplier params kept for signature compatibility — unused.)
    df = calculate_megatrend(df)

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


def detect_ema13_body_cross(
    df: pd.DataFrame,
    symbol: str,
    source: str,
    fast: int = 5,
    mid: int = 13,
    slow: int = 62,
) -> Signal | None:
    """
    Detect a candle whose BODY closed across the EMA13 on the last CLOSED candle.

    direction="BUY"  → candle OPENED below EMA13 and CLOSED above it (bullish)
    direction="SELL" → candle OPENED above EMA13 and CLOSED below it (bearish)

    This is distinct from the 5/13/62 EMA cross: it tracks price action (the
    candle body itself) piercing the EMA13, not EMA5 crossing EMA13.
    """
    if len(df) < mid + 2:
        return None

    df   = calculate_emas(df, fast, mid, slow)
    curr = df.iloc[-1]
    e13  = float(curr["e13"])
    o    = float(curr["open"])
    c    = float(curr["close"])
    ts   = str(curr["timestamp"]) if "timestamp" in curr.index else ""

    # Bullish body cross: opened below the EMA13, closed above it
    if o < e13 and c > e13:
        return Signal(
            symbol=symbol, direction="BUY", source=source,
            close_price=c, ema5=float(curr["e5"]), ema13=e13, ema62=float(curr["e62"]),
            candle_ts=ts, signal_type="ema13_body_cross", open_price=o,
        )

    # Bearish body cross: opened above the EMA13, closed below it
    if o > e13 and c < e13:
        return Signal(
            symbol=symbol, direction="SELL", source=source,
            close_price=c, ema5=float(curr["e5"]), ema13=e13, ema62=float(curr["e62"]),
            candle_ts=ts, signal_type="ema13_body_cross", open_price=o,
        )

    return None


def detect_ema13_retest(
    df: pd.DataFrame,
    symbol: str,
    source: str,
    fast: int = 5,
    mid: int = 13,
    slow: int = 62,
    k_retest: int = 12,
) -> Signal | None:
    """
    EMA13 retest strategy — backtested 12mo/4 symbols (filter decomposition):
      raw crosses            PF 1.01  (no edge)
      Megatrend gate only    PF 0.96  (LOSES — MT alone is the weakest filter)
      EMA62 gate only        PF 1.15  (the workhorse filter)
      EMA62 + decisive cross + no-weekend: PF 1.21, +159R/yr — shipped config.
    The Megatrend hard gate is REMOVED: once the decisive-cross filter exists
    it added zero per-trade quality and halved the trade count (74R vs 159R).
    MT state is still reported as context. Costs matter: edge survives only
    on cheap execution (forex/maker); crypto taker fees destroy it (PF 0.65).

    State machine per symbol, evaluated statelessly on the last CLOSED bar:
      retest_setup   — filtered body cross of the EMA13: body across +
                       close beyond EMA62 + decisive close (≥0.3×ATR beyond
                       the EMA13) + not a weekend bar.
                       Plan: limit at EMA13, stop 1×ATR, TP 1.5R / 2R.
      retest_trigger — first touch of the EMA13 within k_retest bars of the
                       most recent filtered cross (the validated entry).
      retest_cancel  — candle body closed back across the EMA13 before any
                       touch: setup is void.
    Weekend bars (Sat/Sun UTC) neither create setups nor fire triggers —
    weekend trades tested worse and forex is closed for the user anyway.
    Deduplication is per candle via the normal fingerprint mechanism.
    """
    if len(df) < 70:
        return None

    df  = calculate_emas(df, fast, mid, slow)
    df  = calculate_megatrend(df)
    tr  = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.ewm(com=13, adjust=False).mean()

    o   = df["open"].values
    c   = df["close"].values
    h   = df["high"].values
    l   = df["low"].values
    e13 = df["e13"].values
    e62 = df["e62"].values
    atr = df["atr14"].values
    mtb = df["mt_bull"].values
    dow = df["timestamp"].dt.dayofweek.values if "timestamp" in df.columns else None
    j   = len(df) - 1
    ts  = str(df["timestamp"].iloc[j]) if "timestamp" in df.columns else ""

    # Weekend bars neither create setups nor fire triggers/cancels
    if dow is not None and dow[j] >= 5:
        return None

    def filtered_cross(t: int) -> str | None:
        if dow is not None and dow[t] >= 5:
            return None
        if np.isnan(atr[t]) or atr[t] <= 0:
            return None
        strong = abs(c[t] - e13[t]) >= 0.3 * atr[t]  # decisive close beyond EMA13
        if not strong:
            return None
        if o[t] < e13[t] and c[t] > e13[t] and c[t] > e62[t]:
            return "BUY"
        if o[t] > e13[t] and c[t] < e13[t] and c[t] < e62[t]:
            return "SELL"
        return None

    def build(sig_type: str, direction: str, level: float, atr_v: float) -> Signal:
        sign = 1 if direction == "BUY" else -1
        dist = 1.0 * atr_v
        return Signal(
            symbol=symbol, direction=direction, source=source,
            close_price=float(c[j]), ema5=0.0, ema13=float(level), ema62=float(e62[j]),
            candle_ts=ts, signal_type=sig_type, mt_bull=bool(mtb[j]),
            entry=float(level), stop=float(level - sign * dist),
            tp1=float(level + sign * dist * 1.5), tp2=float(level + sign * dist * 2.0),
            atr=float(atr_v),
        )

    # 1) Is the last closed bar itself a fresh filtered cross? → SETUP
    d = filtered_cross(j)
    if d and atr[j] > 0:
        return build("retest_setup", d, e13[j], atr[j])

    # 2) Otherwise: is there an active setup whose first event lands on bar j?
    for i in range(j - 1, max(j - 1 - k_retest, 69), -1):
        d = filtered_cross(i)
        if d:
            break
    else:
        return None

    is_long = d == "BUY"
    for t in range(i + 1, j + 1):
        touched      = (l[t] <= e13[t]) if is_long else (h[t] >= e13[t])
        crossed_back = (c[t] < e13[t])  if is_long else (c[t] > e13[t])
        if touched:
            # first touch — only alert if it happened on the newest closed bar
            if t == j and atr[j] > 0:
                return build("retest_trigger", d, e13[j], atr[j])
            return None
        if crossed_back:
            if t == j:
                return build("retest_cancel", d, e13[j], atr[j])
            return None

    return None
