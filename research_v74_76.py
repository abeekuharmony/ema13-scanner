"""
research_v74_76.py — v7.2 vs v7.4 (volume-spike lookback) vs v7.5 (body expansion) vs v7.6 (both), XAUUSD.
Run with Exness MT5 open + logged in:   py research_v74_76.py
Yardstick (same as before): after the signal candle closes, +1.5 ATR before -1.0 ATR? (+1.5R / -1R / 0 after 24 candles)
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
def pf(r): return r[r > 0].sum() / max(1e-9, -r[r < 0].sum())

def run(label, df):
    o, h, l, c = (df[k].values.astype(float) for k in ("open", "high", "low", "close"))
    v = df["tick_volume"].values.astype(float); n = len(c)
    days = (df["time"].iloc[-1] - df["time"].iloc[0]) / 86400
    a = atr(h, l, c); m = ema(c, 13)
    d = np.where((o < m) & (c > m), 1, np.where((o > m) & (c < m), -1, 0))

    spike = v >= 1.5 * pd.Series(v).rolling(20).mean().shift(1).values          # v7.2 rule, per candle
    def spike_within(k):                                                          # spike on this or previous k-1 candles
        return pd.Series(spike.astype(float)).rolling(k).max().values > 0
    body = np.abs(c - o)
    avg_body = pd.Series(body).rolling(20).mean().shift(1).values
    body_avg = lambda x: body >= x * avg_body                                     # body >= x * avg body of prev 20
    body_atr = lambda x: body >= x * a                                            # body >= x * ATR(14)

    tests = {
        "v7   plain cross":                         np.ones(n, bool),
        "v7.2 RVOL 1.5x on signal candle":          spike,
        "v7.4 RVOL spike within last 3":            spike_within(3),
        "v7.4 RVOL spike within last 5":            spike_within(5),
        "v7.5 body >= 1.5x avg body(20)":           body_avg(1.5),
        "v7.5 body >= 2.0x avg body(20)":           body_avg(2.0),
        "v7.5 body >= 1.0x ATR":                    body_atr(1.0),
        "v7.5 body >= 1.5x ATR":                    body_atr(1.5),
        "v7.6 spike<=3 + body>=1.5x avg":           spike_within(3) & body_avg(1.5),
        "v7.6 spike<=5 + body>=1.5x avg":           spike_within(5) & body_avg(1.5),
        "v7.6 spike<=3 + body>=1.0x ATR":           spike_within(3) & body_atr(1.0),
    }
    res = {i: outcome(i, d[i], c, h, l, a) for i in range(60, n - 1) if d[i] != 0}
    print(f"\n===== XAUUSD {label}  ({n} candles, {days:.0f} days) =====")
    print(f"{'version':38} {'sig/day':>7} {'win%':>6} {'PF':>5} {'n':>6} | {'PF 1st half':>11} {'PF 2nd half':>11}")
    for name, ok in tests.items():
        idx = np.array([i for i in res if ok[i]]); r = np.array([res[i] for i in idx])
        if len(r) == 0: print(f"{name:38}  none"); continue
        first = idx < n // 2
        print(f"{name:38} {len(r) / days:7.2f} {100 * (r > 0).mean():6.1f} {pf(r):5.2f} {len(r):6d} | "
              f"{pf(r[first]):11.2f} {pf(r[~first]):11.2f}")

if not mt5.initialize(path=EXNESS_PATH):
    raise SystemExit(f"MT5 init failed: {mt5.last_error()}")
sym = next(s.name for s in mt5.symbols_get() if s.name.upper().startswith("XAUUSD"))
mt5.symbol_select(sym, True)
for label, (tf, count) in TFS.items():
    rates = mt5.copy_rates_from_pos(sym, tf, 1, count)
    if rates is None or len(rates) < 500: print(f"{label}: no data"); continue
    run(label, pd.DataFrame(rates))
mt5.shutdown()
