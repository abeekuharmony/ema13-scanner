"""
research_v7x.py — backtest v7 vs v7.1 (VWMA13) vs v7.2 (RVOL spike) vs v7.3 (fractal break) on XAUUSD.
Run with Exness MT5 open + logged in:   py research_v7x.py

Same yardstick as research_cross_filters.py: after the signal candle closes, does price move
+1.5 ATR in the signal direction BEFORE 1.0 ATR against? (win +1.5R, loss -1R, 24-candle timeout = 0)
Volume = MT5 tick volume (spot gold has no exchange volume; TradingView/OANDA also uses tick volume).
"""
import numpy as np, pandas as pd, MetaTrader5 as mt5

EXNESS_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
TP, SL, TIMEOUT = 1.5, 1.0, 24
TFS = {"M5": (mt5.TIMEFRAME_M5, 60000), "H1": (mt5.TIMEFRAME_H1, 6000)}

def ema(x, n): return pd.Series(x).ewm(span=n, adjust=False).mean().values
def atr(h, l, c, n=14):
    tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)), abs(l - np.roll(c, 1)))); tr[0] = h[0] - l[0]
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().values

def outcome(i, d, c, h, l, a):
    up = c[i] + d * TP * a[i]; dn = c[i] - d * SL * a[i]
    for j in range(i + 1, min(i + 1 + TIMEOUT, len(c))):
        if (l[j] <= dn) if d > 0 else (h[j] >= dn): return -1.0
        if (h[j] >= up) if d > 0 else (l[j] <= up): return TP
    return 0.0

def body_cross(o, c, line):
    return np.where((o < line) & (c > line), 1, np.where((o > line) & (c < line), -1, 0))

def run(label, df):
    o, h, l, c = (df[k].values.astype(float) for k in ("open", "high", "low", "close"))
    v = df["tick_volume"].values.astype(float)
    n = len(c); days = (df["time"].iloc[-1] - df["time"].iloc[0]) / 86400
    a = atr(h, l, c)

    # v7 / v7.2 / v7.3 signal line = EMA13 ; v7.1 = VWMA13
    e13  = ema(c, 13)
    vwma = (pd.Series(c * v).rolling(13).sum() / pd.Series(v).rolling(13).sum()).values
    d_ema, d_vw = body_cross(o, c, e13), body_cross(o, c, vwma)

    # v7.2: volume >= 1.5 x SMA(volume, 20) of the PREVIOUS 20 candles
    vavg = pd.Series(v).rolling(20).mean().shift(1).values
    vol_ok = v >= 1.5 * vavg

    # v7.3: most recent confirmed 5-candle fractal (2 each side), formed within last 10 candles
    S, LOOK = 2, 10
    lastPH = np.full(n, np.nan); lastPHbar = np.full(n, -10**9)
    lastPL = np.full(n, np.nan); lastPLbar = np.full(n, -10**9)
    curPH = np.nan; curPHb = -10**9; curPL = np.nan; curPLb = -10**9
    for i in range(n):
        p = i - S                                   # candidate pivot, confirmed at bar i
        if p - S >= 0:
            win_h = np.r_[h[p - S:p], h[p + 1:i + 1]]
            win_l = np.r_[l[p - S:p], l[p + 1:i + 1]]
            if h[p] > win_h.max(): curPH, curPHb = h[p], p
            if l[p] < win_l.min(): curPL, curPLb = l[p], p
        lastPH[i], lastPHbar[i], lastPL[i], lastPLbar[i] = curPH, curPHb, curPL, curPLb
    brk_up = ~np.isnan(lastPH) & (np.arange(n) - lastPHbar <= LOOK) & (c > np.nan_to_num(lastPH, nan=np.inf))
    brk_dn = ~np.isnan(lastPL) & (np.arange(n) - lastPLbar <= LOOK) & (c < np.nan_to_num(lastPL, nan=-np.inf))

    versions = {
        "v7   (plain EMA13 body cross)": d_ema,
        "v7.1 (VWMA13 body cross)":     d_vw,
        "v7.2 (EMA13 + RVOL 1.5x)":     np.where(vol_ok, d_ema, 0),
        "v7.3 (EMA13 + fractal break)": np.where(((d_ema > 0) & brk_up) | ((d_ema < 0) & brk_dn), d_ema, 0),
    }
    print(f"\n===== XAUUSD {label}  ({n} candles, {days:.0f} days) =====")
    print(f"{'version':32} {'sig/day':>7} {'win%':>6} {'R/sig':>7} {'PF':>5} {'signals':>8}")
    for name, d in versions.items():
        idx = [i for i in range(60, n - 1) if d[i] != 0]
        r = np.array([outcome(i, d[i], c, h, l, a) for i in idx])
        pf = r[r > 0].sum() / max(1e-9, -r[r < 0].sum())
        print(f"{name:32} {len(r) / days:7.1f} {100 * (r > 0).mean():6.1f} {r.mean():+7.3f} {pf:5.2f} {len(r):8d}")

if not mt5.initialize(path=EXNESS_PATH):
    raise SystemExit(f"MT5 init failed: {mt5.last_error()}")
sym = next((s.name for s in mt5.symbols_get() if s.name.upper().startswith("XAUUSD")), None)
mt5.symbol_select(sym, True)
for label, (tf, count) in TFS.items():
    rates = mt5.copy_rates_from_pos(sym, tf, 1, count)   # start at 1 = closed candles only
    if rates is None or len(rates) < 500: print(f"{label}: no data"); continue
    run(label, pd.DataFrame(rates))
mt5.shutdown()
