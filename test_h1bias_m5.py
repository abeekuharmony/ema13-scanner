"""
test_h1bias_m5.py — READ-ONLY backtest (no orders) on real Exness XAUUSD M5.
Question: does an H1 bias filter improve the M5 EMA13 body-cross signal?
  H1 bias = M5 BUY only if the last CLOSED H1 candle closed above its H1 EMA13
            (SELL mirror). Uses only H1 candles already closed at signal time.
Trade model (same for every variant, so they're comparable):
  entry = signal candle close, SL = 10-bar swing -/+ 0.1*ATR, TP = 3R.
  If SL and TP are hit in the same candle, count the loss (conservative).
  Each signal is scored on its own (overlap allowed) — this compares filters.
"""
import numpy as np, pandas as pd, MetaTrader5 as mt5

EXNESS_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
M5_BARS, H1_BARS = 80000, 8000
SW, R_MULT, ATR_DEC = 10, 3.0, 0.3

def ema(x, n):  return pd.Series(x).ewm(span=n, adjust=False).mean().values
def atr14(h, l, c):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values

def get(sym, tf, n):
    r = mt5.copy_rates_from_pos(sym, tf, 1, n)          # start at 1 = closed bars only
    return pd.DataFrame(r)

if not mt5.initialize(path=EXNESS_PATH):
    raise SystemExit("MT5 init failed: %s" % (mt5.last_error(),))
names = [s.name for s in mt5.symbols_get()]
sym = next((x for x in names if x.upper() == "XAUUSDM"), None) or next(x for x in names if "XAUUSD" in x.upper())
mt5.symbol_select(sym, True)
m5, h1 = get(sym, mt5.TIMEFRAME_M5, M5_BARS), get(sym, mt5.TIMEFRAME_H1, H1_BARS)
mt5.shutdown()

t, o, h, l, c = (m5[k].values for k in ["time", "open", "high", "low", "close"])
o, h, l, c = (x.astype(float) for x in (o, h, l, c))
e13, a = ema(c, 13), atr14(h, l, c)
n = len(c)
hr = ((t // 3600) % 24).astype(int)

# H1 bias from the last H1 candle CLOSED by the M5 candle's close (t + 300)
h1t, h1c = h1["time"].values, h1["close"].values.astype(float)
h1e = ema(h1c, 13)
idx = np.searchsorted(h1t + 3600, t + 300, side="right") - 1      # last h1 with close <= m5 close
valid = idx >= 13
bull = np.where(valid, h1c[np.clip(idx, 0, None)] > h1e[np.clip(idx, 0, None)], False)
bear = np.where(valid, h1c[np.clip(idx, 0, None)] < h1e[np.clip(idx, 0, None)], False)

def in_sess(hh, mode):
    if mode == "none":  return True
    if mode == "tight": return hh < 7 or 12 <= hh < 16
    return hh < 21                                                # loose

def outcome(i, lng):
    sw = l[i-SW+1:i+1].min() if lng else h[i-SW+1:i+1].max()
    stop = sw - 0.1*a[i] if lng else sw + 0.1*a[i]
    R = abs(c[i] - stop)
    if R <= 0: return None
    tp = c[i] + (R_MULT*R if lng else -R_MULT*R)
    for b in range(i+1, min(n, i+2000)):
        if (l[b] <= stop) if lng else (h[b] >= stop): return -1.0
        if (h[b] >= tp)   if lng else (l[b] <= tp):   return R_MULT
    return None

up = (o < e13) & (c > e13)
dn = (o > e13) & (c < e13)
res = {}
for i in range(200, n-1):
    if not (up[i] or dn[i]): continue
    lng = bool(up[i])
    r = outcome(i, lng)
    if r is None: continue
    res[i] = r

weeks = (t[-1] - t[0]) / (7*86400)
start = pd.to_datetime(t[0], unit="s"); end = pd.to_datetime(t[-1], unit="s")
print(f"Real Exness {sym} M5: {n} bars, {start:%Y-%m-%d} -> {end:%Y-%m-%d} (~{weeks:.0f} weeks)")
print(f"Trade model: entry at signal close, swing SL, {R_MULT:.0f}R target\n")
print(f"{'session':6} {'decisive':8} {'H1 bias':7} | {'trades':>6} {'/week':>6} {'PF':>5} {'net R':>7} {'win%':>5} {'maxDD R':>8}")
for sess in ["none", "tight", "loose"]:
    for dec in [False, True]:
        for bias in [False, True]:
            rs = []
            for i, r in res.items():
                lng = bool(up[i])
                if not in_sess(hr[i], sess): continue
                if dec and ((c[i]-e13[i]) if lng else (e13[i]-c[i])) < ATR_DEC*a[i]: continue
                if bias and not (bull[i] if lng else bear[i]): continue
                rs.append(r)
            rs = np.array(rs)
            if len(rs) == 0: continue
            pf = rs[rs>0].sum() / -rs[rs<0].sum() if (rs<0).any() else 99
            eq = np.cumsum(rs); dd = (eq - np.maximum.accumulate(eq)).min()
            print(f"{sess:6} {('0.3 ATR' if dec else 'off'):8} {('ON' if bias else 'off'):7} | "
                  f"{len(rs):6d} {len(rs)/weeks:6.1f} {pf:5.2f} {rs.sum():+7.0f} {100*(rs>0).mean():5.1f} {dd:8.0f}")
