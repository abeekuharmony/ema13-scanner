"""
research_tma.py — v7 vs v7.1 (v7 + TMA Centered Bands rules) on XAUUSD, user's own trade management.
Run with Exness MT5 open + logged in:   py research_tma.py

TMA Centered Bands (Brylator v0.42, HalfLength 12, Weighted price, ATR 100, mult 2):
  HINDSIGHT  = the solid line you see on history: centred on each candle, uses 12 FUTURE candles.
  LIVE       = what the indicator actually showed when the signal candle closed (right edge, past candles only).
Rules: 1) BUY needs mid line green, SELL red   2) invalid if candle range >= 75% of channel width
       3) BUY needs open below mid line, SELL open above it.
Trade rules: EMA5 pullback entry (wait 3, else market), fractal swing SL, 1R/2R/3R scale-out
(SL to BE after TP1, to TP1 after TP2), worst-case same-candle, 0.25 spread.
"""
import numpy as np, pandas as pd, MetaTrader5 as mt5
import research_user_rules as R

H = 12; MULT = 2.0; ATRP = 100

def trade(i, d, o, h, l, c, e5, fh, fl):
    n = len(c)
    stop0 = R.swing_stop(i, d, c[i], fh, fl, h, l)
    entry = None; eb = None; touched = False
    for j in range(i + 1, i + 1 + R.WAIT):
        touched = (l[j] <= e5[j]) if d > 0 else (h[j] >= e5[j])
        if touched:
            entry = min(o[j], e5[j]) if d > 0 else max(o[j], e5[j]); eb = j; break
        if (l[j] <= stop0) if d > 0 else (h[j] >= stop0): return None
    if entry is None: eb = i + R.WAIT; entry = c[eb]
    risk = (entry - stop0) * d
    if risk <= 0: return None
    tps = [entry + d * k * risk for k in (1, 2, 3)]
    sl = stop0; hit = 0; legs = []; j = eb + 1; end = min(n - 1, eb + R.MAXHOLD)
    if touched and ((l[eb] <= sl) if d > 0 else (h[eb] >= sl)): legs = [-1.0] * 3; hit = 3
    while j <= end and hit < 3:
        if (l[j] <= sl) if d > 0 else (h[j] >= sl):
            legs += [(sl - entry) * d / risk] * (3 - hit); hit = 3; break
        while hit < 3 and ((h[j] >= tps[hit]) if d > 0 else (l[j] <= tps[hit])):
            legs.append(hit + 1.0); hit += 1
            sl = entry if hit == 1 else tps[hit - 2]
        j += 1
    if hit < 3: legs += [(c[min(j, n - 1)] - entry) * d / risk] * (3 - hit)
    return sum(legs) / 3 - R.SPREAD / risk, j

def pf(x): x = np.asarray(x); return x[x > 0].sum() / max(1e-9, -x[x < 0].sum())

def tma(h, l, c):
    n = len(c); P = (h + l + 2 * c) / 4
    k = np.arange(H + 1, 0, -1).astype(float)                 # 13,12,...,1  (newest first)
    S1 = k.sum()
    back = np.convolve(P, k)[:n]                              # sum_{m=0..H} (H+1-m) * P[t-m]
    back[:H] = np.nan
    fwd = np.full(n, np.nan)                                  # sum_{j=1..H} (H+1-j) * P[t+j]
    kf = np.arange(H, 0, -1).astype(float)
    for t in range(n - H):
        fwd[t] = (kf * P[t + 1:t + H + 1]).sum()
    live_mid = back / S1
    asym1 = np.r_[np.nan, (back[:-1] + H * P[1:]) / (S1 + H)]   # value for t-1 as seen at t (1 future candle)
    cent_mid = (back + fwd) / (2 * S1 - (H + 1))
    tr2 = np.maximum(h, np.r_[np.nan, c[:-1]]) - np.minimum(l, np.r_[np.nan, c[:-1]])
    atrx = pd.Series(tr2).rolling(ATRP).mean().shift(10).values
    def colour(cur, prev):
        g = np.zeros(n, bool); state = True
        for t in range(n):
            if not (np.isnan(cur[t]) or np.isnan(prev[t])):
                if cur[t] > prev[t]: state = True
                elif cur[t] < prev[t]: state = False
            g[t] = state
        return g
    live_up = colour(live_mid, asym1)
    cent_up = colour(cent_mid, np.r_[np.nan, cent_mid[:-1]])
    return live_mid, live_up, cent_mid, cent_up, 2 * MULT * atrx

def evaluate(name, sig, o, h, l, c, e5, fh, fl, n, days):
    res = {}
    for i in range(150, n - R.WAIT - 2):
        if sig[i] != 0:
            t = trade(i, sig[i], o, h, l, c, e5, fh, fl)
            if t: res[i] = t
    if not res: print(f"{name:40} none"); return
    one = []; nxt = 0
    for i in sorted(res):
        if i >= nxt: one.append(res[i][0]); nxt = res[i][1]
    one = np.array(one); eq = np.cumsum(one); dd = (eq - np.maximum.accumulate(eq)).min()
    idx = np.array(sorted(res)); allr = np.array([res[i][0] for i in idx]); first = idx < n // 2
    print(f"{name:40} {len(one):5d} {len(one)/days:5.2f} {100*(one>0).mean():5.1f} {pf(one):5.2f} {one.sum():+7.1f} {dd:+6.1f} | "
          f"{len(allr):5d} {pf(allr):5.2f} {pf(allr[first]):5.2f} {pf(allr[~first]):5.2f}")

def run(label, df):
    o, h, l, c = (df[k].values.astype(float) for k in ("open", "high", "low", "close")); n = len(c)
    days = (df["time"].iloc[-1] - df["time"].iloc[0]) / 86400
    e5, e13 = R.ema(c, 5), R.ema(c, 13); fh, fl = R.fractals(h, l)
    d = np.where((o < e13) & (c > e13), 1, np.where((o > e13) & (c < e13), -1, 0))
    lm, lup, cm, cup, width = tma(h, l, c)
    rng_ok = (h - l) < 0.75 * width
    def rules(mid, up, r1=True, r2=True, r3=True):
        okb = np.ones(n, bool); oks = np.ones(n, bool)
        if r1: okb &= up;       oks &= ~up
        if r2: okb &= rng_ok;   oks &= rng_ok
        if r3: okb &= o < mid;  oks &= o > mid
        return np.where((d > 0) & okb, 1, np.where((d < 0) & oks, -1, 0))
    tests = {
        "v7 plain cross":                         d,
        "v7.1 LIVE (all 3 rules) <- real":        rules(lm, lup),
        "v7.1 HINDSIGHT (all 3) <- not tradable": rules(cm, cup),
        "LIVE rule 1 only (colour)":              rules(lm, lup, True, False, False),
        "LIVE rule 2 only (candle < 75% chan)":   rules(lm, lup, False, True, False),
        "LIVE rule 3 only (open vs mid)":         rules(lm, lup, False, False, True),
    }
    print(f"\n===== XAUUSD {label} ({days:.0f} days) — your trade rules, after 0.25 spread =====")
    print(f"{'version':40} {'------ one trade at a time ------':^37} | {'---- every signal ----':^24}")
    print(f"{'':40} {'n':>5} {'/day':>5} {'win%':>5} {'PF':>5} {'totR':>7} {'maxDD':>6} | {'n':>5} {'PF':>5} {'1st':>5} {'2nd':>5}")
    for name, s in tests.items():
        evaluate(name, s, o, h, l, c, e5, fh, fl, n, days)

if __name__ == "__main__":
    if not mt5.initialize(path=R.EXNESS_PATH):
        raise SystemExit(f"MT5 init failed: {mt5.last_error()}")
    sym = next(s.name for s in mt5.symbols_get() if s.name.upper().startswith("XAUUSD"))
    mt5.symbol_select(sym, True)
    for label, (tf, count) in R.TFS.items():
        run(label, pd.DataFrame(mt5.copy_rates_from_pos(sym, tf, 1, count)))
    mt5.shutdown()
