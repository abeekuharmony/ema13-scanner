"""
research_tma_confirmed.py — (1) how big are the stops, in dollars, and (2) can a SHORT-LOOKAHEAD TMA, confirmed
a few candles after the cross, be traded live?  XAUUSD, Exness, 0.25 spread, closed candles only.
Run with Exness MT5 open + logged in:   py research_tma_confirmed.py [H1] [M5]

Idea: the centred TMA with half-length K only needs K candles AFTER the signal candle.  If you wait K candles,
the line value for the signal candle is FINAL (it never repaints again) and the trade is taken at the close of
candle i+K.  K=12 is the real indicator; K=2..6 is the same indicator with a smaller "HalfLength" input.
Signal at i: EMA13 body cross.  BUY if the body midpoint is below the K-line (discount), SELL if above.
Entry: close of candle i+K, only if the cross is still alive (close of i+K still on the signal side of EMA13).
Stops tested: swing (5-candle fractal), candle (extreme of candles i..i+K), atr1 (1.0 x ATR14).
Exits: ema13 (first close back across EMA13), 2R, triple (1R/2R/3R thirds, BE after TP1).
"""
import sys, numpy as np, pandas as pd, MetaTrader5 as mt5
import research_user_rules as R
import research_strategy_search as S
from research_channel_replace import ema, hma
from research_strategy_robust import htf_rising

def kline(P, K):
    """centred triangular-weighted line with K candles each side; value for bar t is final at close of t+K"""
    n = len(P); k = np.r_[np.arange(1, K + 2), np.arange(K, 0, -1)].astype(float)
    conv = np.convolve(P, k)                       # conv[t+K] = sum_j k[j] * P[t-K+j]
    out = np.full(n, np.nan); out[K:n - K] = conv[2 * K:n] / k.sum()
    return out

def trade2(D, j, d, stop, xm):
    """enter at close of bar j with the given stop price"""
    h, l, c, n = D.h, D.l, D.c, D.n; entry = c[j]; risk = (entry - stop) * d
    if risk <= 0.05: return None
    cost = R.SPREAD / risk; end = min(n - 1, j + R.MAXHOLD)
    hs = lambda k, s: (l[k] <= s) if d > 0 else (h[k] >= s)
    ht = lambda k, p: (h[k] >= p) if d > 0 else (l[k] <= p)
    if xm == "triple":
        tps = [entry + d * q * risk for q in (1, 2, 3)]; sl = stop; hit = 0; legs = []; k = j + 1
        while k <= end and hit < 3:
            if hs(k, sl): legs += [(sl - entry) * d / risk] * (3 - hit); hit = 3; break
            while hit < 3 and ht(k, tps[hit]):
                legs.append(hit + 1.0); hit += 1; sl = entry if hit == 1 else tps[hit - 2]
            k += 1
        if hit < 3: legs += [(c[min(k, n - 1)] - entry) * d / risk] * (3 - hit)
        return sum(legs) / 3 - cost, min(k, n - 1), risk
    tp = entry + d * 2 * risk if xm == "2R" else None
    for k in range(j + 1, end + 1):
        if hs(k, stop): return -1.0 - cost, k, risk
        if tp is not None and ht(k, tp): return 2.0 - cost, k, risk
        if xm == "ema13" and ((c[k] < D.e13[k]) if d > 0 else (c[k] > D.e13[k])):
            return (c[k] - entry) * d / risk - cost, k, risk
    return (c[end] - entry) * d / risk - cost, end, risk

def build(D, K, tma_side, extra_ok=None, stop_mode="swing"):
    """list of (j, d, stop) trades. K=0 means no delay."""
    o, h, l, c, n = D.o, D.h, D.l, D.c, D.n; bm = (o + c) / 2
    P = (h + l + 2 * c) / 4; line = kline(P, K) if (K > 0 and tma_side) else None
    up = (o < D.e13) & (c > D.e13); dn = (o > D.e13) & (c < D.e13); out = []
    for i in np.flatnonzero(up | dn):
        d = 1 if up[i] else -1; j = i + K
        if i < 250 or j >= n - 3: continue
        if tma_side:
            lv = line[i] if K > 0 else np.nan
            if np.isnan(lv) or not ((bm[i] < lv) if d > 0 else (bm[i] > lv)): continue
        if K > 0 and not ((c[j] > D.e13[j]) if d > 0 else (c[j] < D.e13[j])): continue
        if extra_ok is not None and not (extra_ok[0][j] if d > 0 else extra_ok[1][j]): continue
        if stop_mode == "swing": stop = D.swing_stop(j, d)
        elif stop_mode == "candle": stop = l[i:j + 1].min() if d > 0 else h[i:j + 1].max()
        else: stop = c[j] - d * 1.0 * D.a[j]
        out.append((j, d, stop))
    return out

def evaluate(D, trades, xm, days):
    split = int(D.n * 0.6); res = {}
    for j, d, st in trades:
        r = trade2(D, j, d, st, xm)
        if r: res[j] = r
    if len(res) < 20: return None
    idx = np.array(sorted(res)); x = np.array([res[i][0] for i in idx]); rk = np.array([res[i][2] for i in idx]); f = idx < split
    one = []; nxt = 0
    for i in idx:
        if i >= nxt: one.append(res[i][0]); nxt = res[i][1] + 1
    one = np.array(one); eq = np.cumsum(one); dd = (eq - np.maximum.accumulate(eq)).min()
    return dict(n=len(x), pm=len(x) / days * 30.4, win=100 * (x > 0).mean(), pf=S.pf(x), exp=x.mean(),
                is_pf=S.pf(x[f]), is_exp=x[f].mean(), oos_pf=S.pf(x[~f]), oos_exp=x[~f].mean(),
                pm1=len(one) / days * 30.4, pf1=S.pf(one), tot=one.sum(), dd=dd,
                risk_med=np.median(rk), risk_p90=np.percentile(rk, 90), n_unused=0)

HDR = (S.HDR + f" | {'stop $ med':>10} {'p90':>6}")
def row(name, s):
    return S.row(name, s) + (f" | {s['risk_med']:10.1f} {s['risk_p90']:6.1f}" if s else "")

def run(label, df):
    D = S.Data(df); days = (D.t[-1] - D.t[0]) / 86400; c = D.c
    e100 = ema(c, 100); hu, hd = htf_rising(D, S.HTF[label][1], 55)
    trend = ((c > e100) & (c > D.h55) & hu, (c < e100) & (c < D.h55) & hd)         # H1 strategy from last time
    print(f"\n######## XAUUSD {label} ({days:.0f} days): stop sizes in $ per 1 oz (= per 0.01 lot on Exness XAUUSD) ########")
    print("stop $ med / p90 = median and 90th-percentile distance from entry to stop; ATR14 median = %.1f" % np.median(D.a[300:]))
    print(f"\n=== A. baseline rules (K=0, no TMA) ===\n{HDR}")
    for nm, ex in (("plain cross", None), ("HMA55+EMA100+H4 rising (last strategy)", trend)):
        for sm in ("swing", "atr1"):
            for xm in ("ema13", "triple"):
                print(row(f"{nm} | {sm}/{xm}", evaluate(D, build(D, 0, False, ex, sm), xm, days)))
    print(f"\n=== B. TMA with K future candles, trade at close of i+K (K=12 is the real indicator) ===")
    for sm, xm in (("swing", "triple"), ("candle", "triple"), ("atr1", "2R"), ("candle", "ema13"), ("swing", "ema13")):
        print(f"--- stop={sm}, exit={xm}\n{HDR}")
        for K in (1, 2, 3, 4, 6, 12):
            print(row(f"TMA K={K}", evaluate(D, build(D, K, True, None, sm), xm, days)))
    print(f"\n=== C. TMA K=3/4/6 + trend filter (EMA100 side, HMA55 side, H4/H12 rising) ===")
    for sm, xm in (("candle", "triple"), ("atr1", "2R"), ("candle", "ema13")):
        print(f"--- stop={sm}, exit={xm}\n{HDR}")
        for K in (3, 4, 6):
            print(row(f"TMA K={K} + trend", evaluate(D, build(D, K, True, trend, sm), xm, days)))

if __name__ == "__main__":
    tfs = sys.argv[1:] or ["H1", "M5"]
    if not mt5.initialize(path=R.EXNESS_PATH):
        raise SystemExit(f"MT5 init failed: {mt5.last_error()}")
    sym = next(s.name for s in mt5.symbols_get() if s.name.upper().startswith("XAUUSD"))
    mt5.symbol_select(sym, True)
    data = {lb: pd.DataFrame(mt5.copy_rates_from_pos(sym, R.TFS[lb][0], 1, R.TFS[lb][1])) for lb in tfs}
    mt5.shutdown()
    for lb in tfs: run(lb, data[lb])
