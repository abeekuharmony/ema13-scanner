"""
research_user_rules.py — backtest every indicator version with the USER'S OWN trade management. XAUUSD, M5 + H1.
Run with Exness MT5 open + logged in:   py research_user_rules.py

Trade rules (as the user trades manually):
  Entry : after the signal candle closes, wait up to 3 candles for price to touch EMA5 -> enter at EMA5.
          No pullback within 3 candles -> enter at the close of the 3rd candle.
          If the stop is hit before entry -> trade skipped.
  Stop  : most recent structural swing low (BUY) / high (SELL) = 5-candle fractal, confirmed before the signal.
  Exits : 3 equal positions. TP1 = 1R, TP2 = 2R, TP3 = 3R.
          After TP1 -> SL to entry. After TP2 -> SL to TP1.
          Outcomes per trade:  SL first = -1R | TP1 then BE = +0.33R | TP2 then back to TP1 = +1.33R | all 3 = +2R
  Same candle touches SL and a TP -> SL counted (worst case). One trade at a time. Max hold 200 candles.
  Spread: results shown with no cost and with a 0.25 spread per trade.
"""
import numpy as np, pandas as pd, MetaTrader5 as mt5

EXNESS_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
WAIT, MAXHOLD, SPREAD = 3, 200, 0.25
TFS = {"M5": (mt5.TIMEFRAME_M5, 60000), "H1": (mt5.TIMEFRAME_H1, 6000)}

def ema(x, n): return pd.Series(x).ewm(span=n, adjust=False).mean().values
def atr(h, l, c, n=14):
    tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)), abs(l - np.roll(c, 1)))); tr[0] = h[0] - l[0]
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().values

def fractals(h, l, s=2):
    """lists of (confirm_bar, pivot_bar, price) for fractal highs and lows"""
    n = len(h); fh, fl = [], []
    for p in range(s, n - s):
        if h[p] > max(h[p - s:p].max(), h[p + 1:p + s + 1].max()): fh.append((p + s, p, h[p]))
        if l[p] < min(l[p - s:p].min(), l[p + 1:p + s + 1].min()): fl.append((p + s, p, l[p]))
    return fh, fl

def signals(o, h, l, c, v):
    n = len(c); e13 = ema(c, 13)
    cross = lambda line: np.where((o < line) & (c > line), 1, np.where((o > line) & (c < line), -1, 0))
    d = cross(e13)
    vwma = (pd.Series(c * v).rolling(13).sum() / pd.Series(v).rolling(13).sum()).values
    spike = v >= 1.5 * pd.Series(v).rolling(20).mean().shift(1).values
    spike3 = pd.Series(spike.astype(float)).rolling(3).max().values > 0
    body = np.abs(c - o); body_ok = body >= 1.5 * pd.Series(body).rolling(20).mean().shift(1).values
    # v7.3 fractal break (most recent fractal within last 10 candles)
    fh, fl = fractals(h, l)
    lastH = np.full(n, np.nan); lastHb = np.full(n, -10**9); lastL = np.full(n, np.nan); lastLb = np.full(n, -10**9)
    k = 0
    for cb, pb, pr in fh:
        lastH[cb:] = pr; lastHb[cb:] = pb
    for cb, pb, pr in fl:
        lastL[cb:] = pr; lastLb[cb:] = pb
    ar = np.arange(n)
    bu = (ar - lastHb <= 10) & (c > np.nan_to_num(lastH, nan=np.inf))
    bd = (ar - lastLb <= 10) & (c < np.nan_to_num(lastL, nan=-np.inf))
    return {
        "v7   plain cross":          d,
        "v7.1 VWMA13":               cross(vwma),
        "v7.2 RVOL 1.5x":            np.where(spike, d, 0),
        "v7.3 fractal break":        np.where(((d > 0) & bu) | ((d < 0) & bd), d, 0),
        "v7.4 RVOL within 3":        np.where(spike3, d, 0),
        "v7.5 body 1.5x avg":        np.where(body_ok, d, 0),
        "v7.6 RVOL<=3 + body":       np.where(spike3 & body_ok, d, 0),
    }, fh, fl

def swing_stop(i, d, entry, fh, fl, h, l):
    """most recent confirmed fractal low below entry (BUY) / high above entry (SELL); fallback 10-candle extreme"""
    src = fl if d > 0 else fh
    for cb, pb, pr in reversed(src):
        if cb > i: continue
        if i - pb > 60: break
        if (d > 0 and pr < entry) or (d < 0 and pr > entry): return pr
    return l[i - 9:i + 1].min() if d > 0 else h[i - 9:i + 1].max()

def simulate(sig, o, h, l, c, e5, fh, fl):
    n = len(c); trades = []; i = 60
    while i < n - WAIT - 2:
        d = sig[i]
        if d == 0: i += 1; continue
        # ---- entry: EMA5 pullback within WAIT candles, else close of last wait candle
        stop0 = swing_stop(i, d, c[i], fh, fl, h, l)
        entry = None; eb = None; skipped = False
        touched = False
        for j in range(i + 1, i + 1 + WAIT):
            touched = (l[j] <= e5[j]) if d > 0 else (h[j] >= e5[j])
            if touched:   # filled at EMA5 (a stop hit in this same candle is then counted as a loss below)
                entry = min(o[j], e5[j]) if d > 0 else max(o[j], e5[j]); eb = j; break
            if (l[j] <= stop0) if d > 0 else (h[j] >= stop0): skipped = True; break
        if skipped: i += 1; continue
        if entry is None: eb = i + WAIT; entry = c[eb]
        R = (entry - stop0) * d
        if R <= 0: i += 1; continue
        tps = [entry + d * k * R for k in (1, 2, 3)]
        sl = stop0; hit = 0; legs = []; j = eb + 1
        end = min(n - 1, eb + MAXHOLD)
        # pullback fill happens inside candle eb: worst case, that same candle can still hit the stop
        if touched and ((l[eb] <= sl) if d > 0 else (h[eb] >= sl)):
            legs = [-1.0] * 3; hit = 3
        while j <= end and hit < 3:
            if (l[j] <= sl) if d > 0 else (h[j] >= sl):          # stop first (worst case)
                legs += [(sl - entry) * d / R] * (3 - hit); hit = 3; break
            while hit < 3 and ((h[j] >= tps[hit]) if d > 0 else (l[j] <= tps[hit])):
                legs.append(hit + 1.0); hit += 1
                sl = entry if hit == 1 else tps[hit - 2]               # TP1 -> BE, TP2 -> TP1
            j += 1
        if hit < 3:                                               # timed out: close the rest at market
            legs += [(c[min(j, n - 1)] - entry) * d / R] * (3 - hit)
        r = sum(legs) / 3
        trades.append((r, r - SPREAD / R, j))
        i = max(i + 1, j)                                         # one trade at a time
    return trades

def stats(rs):
    rs = np.array(rs); eq = np.cumsum(rs); dd = (eq - np.maximum.accumulate(eq)).min()
    pf = rs[rs > 0].sum() / max(1e-9, -rs[rs < 0].sum())
    return pf, rs.mean(), rs.sum(), dd, 100 * (rs > 0).mean()

def run(label, df):
    o, h, l, c = (df[k].values.astype(float) for k in ("open", "high", "low", "close"))
    v = df["tick_volume"].values.astype(float); n = len(c)
    days = (df["time"].iloc[-1] - df["time"].iloc[0]) / 86400
    e5 = ema(c, 5)
    sigs, fh, fl = signals(o, h, l, c, v)
    print(f"\n===== XAUUSD {label}  ({n} candles, {days:.0f} days) — your rules: EMA5 pullback entry, swing SL, 1R/2R/3R scale-out =====")
    print(f"{'version':22} {'trades':>6} {'/day':>5} {'win%':>6} | {'PF':>5} {'R/trade':>8} {'total R':>8} {'maxDD R':>8} | {'PF':>5} {'R/trade':>8} {'total R':>8}")
    print(f"{'':22} {'':>6} {'':>5} {'':>6} | {'---- no spread ----':^33} | {'-- 0.25 spread --':^24}")
    for name, s in sigs.items():
        t = simulate(s, o, h, l, c, e5, fh, fl)
        if not t: print(f"{name:22} none"); continue
        a = stats([x[0] for x in t]); b = stats([x[1] for x in t])
        print(f"{name:22} {len(t):6d} {len(t) / days:5.2f} {a[4]:6.1f} | {a[0]:5.2f} {a[1]:+8.3f} {a[2]:+8.1f} {a[3]:+8.1f} | {b[0]:5.2f} {b[1]:+8.3f} {b[2]:+8.1f}")

if __name__ == "__main__":
    if not mt5.initialize(path=EXNESS_PATH):
        raise SystemExit(f"MT5 init failed: {mt5.last_error()}")
    sym = next(s.name for s in mt5.symbols_get() if s.name.upper().startswith("XAUUSD"))
    mt5.symbol_select(sym, True)
    for label, (tf, count) in TFS.items():
        rates = mt5.copy_rates_from_pos(sym, tf, 1, count)
        if rates is None or len(rates) < 500: print(f"{label}: no data"); continue
        run(label, pd.DataFrame(rates))
    mt5.shutdown()
