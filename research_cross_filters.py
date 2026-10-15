"""
research_cross_filters.py — which filter removes ranging/noisy EMA13 body-cross signals on GOLD?
Run with MT5 open + logged into Exness:   py research_cross_filters.py

Signal = candle body crosses EMA13 (open one side, close the other), closed candles only.
Quality test (exit-agnostic): after the signal candle closes, does price move +1.5 ATR in the
signal direction BEFORE it moves 1.0 ATR against? (win = +1.5R, loss = -1R, 24-candle timeout = 0)
Reported per filter: signals/day, win %, expectancy (R per signal), PF.
"""
import numpy as np, pandas as pd, MetaTrader5 as mt5

EXNESS_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
TP, SL, TIMEOUT = 1.5, 1.0, 24
TFS = {"M5": (mt5.TIMEFRAME_M5, 60000), "H1": (mt5.TIMEFRAME_H1, 6000)}

def ema(x, n):  return pd.Series(x).ewm(span=n, adjust=False).mean().values
def atr(h, l, c, n=14):
    tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)), abs(l - np.roll(c, 1)))); tr[0] = h[0] - l[0]
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().values

def outcome(i, d, c, h, l, a):
    entry = c[i]; up = entry + d * TP * a[i]; dn = entry - d * SL * a[i]
    for j in range(i + 1, min(i + 1 + TIMEOUT, len(c))):
        hit_tp = h[j] >= up if d > 0 else l[j] <= up
        hit_sl = l[j] <= dn if d > 0 else h[j] >= dn
        if hit_sl: return -1.0          # conservative: stop first if both touched
        if hit_tp: return TP
    return 0.0

def run(label, df):
    o, h, l, c = (df[k].values.astype(float) for k in ("open", "high", "low", "close"))
    m = ema(c, 13); a = atr(h, l, c); hr = pd.to_datetime(df["time"], unit="s").dt.hour.values
    n = len(c); days = (df["time"].iloc[-1] - df["time"].iloc[0]) / 86400
    up = (o < m) & (c > m); dn = (o > m) & (c < m)
    d = np.where(up, 1, np.where(dn, -1, 0))
    above = c > m
    idx = [i for i in range(60, n - 1) if d[i] != 0]

    def fresh(i, k):    # previous k closes all on the OTHER side of EMA13
        side = above[i - k:i]
        return (not side.any()) if d[i] > 0 else side.all()
    def rbreak(i, k):   # close beyond the prior k-candle high/low
        return c[i] > h[i - k:i].max() if d[i] > 0 else c[i] < l[i - k:i].min()
    def cool(i, k):     # no other cross in the previous k candles
        return not d[i - k:i].any()
    def atrk(i, k):     # close clears EMA13 by k*ATR
        return (c[i] - m[i]) * d[i] >= k * a[i]
    tight = lambda i: hr[i] < 7 or 12 <= hr[i] < 16
    loose = lambda i: hr[i] < 21

    filters = {"RAW (every cross)": lambda i: True}
    for k in (0.1, 0.2, 0.3): filters[f"ATR >= {k}"] = lambda i, k=k: atrk(i, k)
    for k in (2, 3, 5, 8):    filters[f"Fresh {k} (k closes other side)"] = lambda i, k=k: fresh(i, k)
    for k in (5, 10, 20):     filters[f"Range break {k}"] = lambda i, k=k: rbreak(i, k)
    for k in (3, 5, 10):      filters[f"Cooldown {k}"] = lambda i, k=k: cool(i, k)
    filters["Session Tight 00-07+12-16"] = tight
    filters["Session Loose 00-21"] = loose
    filters["Fresh 3 + Loose"] = lambda i: fresh(i, 3) and loose(i)
    filters["Fresh 5 + Loose"] = lambda i: fresh(i, 5) and loose(i)
    filters["Fresh 3 + Range break 10"] = lambda i: fresh(i, 3) and rbreak(i, 10)

    res = {i: outcome(i, d[i], c, h, l, a) for i in idx}
    print(f"\n===== XAUUSD {label}  ({n} candles, {days:.0f} days) =====")
    print(f"{'filter':34} {'sig/day':>7} {'win%':>6} {'R/sig':>7} {'PF':>5}")
    for name, f in filters.items():
        r = np.array([res[i] for i in idx if f(i)])
        if len(r) == 0: print(f"{name:34}  none"); continue
        pf = r[r > 0].sum() / max(1e-9, -r[r < 0].sum())
        print(f"{name:34} {len(r) / days:7.1f} {100 * (r > 0).mean():6.1f} {r.mean():+7.3f} {pf:5.2f}")

if not mt5.initialize(path=EXNESS_PATH):
    raise SystemExit(f"MT5 init failed: {mt5.last_error()}")
names = [s.name for s in mt5.symbols_get()]
sym = next((s for s in names if s.upper().startswith("XAUUSD")), None)
mt5.symbol_select(sym, True)
for label, (tf, count) in TFS.items():
    rates = mt5.copy_rates_from_pos(sym, tf, 1, count)   # start at 1 = closed candles only
    if rates is None or len(rates) < 500: print(f"{label}: no data"); continue
    run(label, pd.DataFrame(rates))
mt5.shutdown()
print("\nDone. Paste me the output.")
