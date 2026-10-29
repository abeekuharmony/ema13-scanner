"""
research_false_signals.py — what separates WINNING EMA13 body crosses from FALSE ones, measured at the signal candle?
Base signal = user's rule: bullish body cross of EMA13 with the candle CLOSE above HMA55 (BUY), mirror for SELL.
For each signal we record real-time features (known when the candle closes) and the outcome, then show, per feature
quintile (thresholds fixed on the first 60%), the average R in the first 60% (IS) and the last 40% (OOS), and how often
the hindsight TMA would have approved the signal.  XAUUSD, Exness, 0.25 spread.
Run with Exness MT5 open + logged in:   py research_false_signals.py [H1] [M5]

Outcomes (every signal traded independently, entry at the signal close):
  R_2R   = 1.5 ATR stop, 3 ATR target (neutral fixed bracket)
  R_ride = fractal swing stop, exit on the first close back across EMA13
"""
import sys, numpy as np, pandas as pd, MetaTrader5 as mt5
import research_user_rules as R
import research_strategy_search as S
from research_channel_replace import ema, hma, donchian_mid
from research_tma_realtime import tma_estimates, H

def adx(h, l, c, n=14):
    up = np.diff(h, prepend=h[0]); dn = -np.diff(l, prepend=l[0])
    pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)), abs(l - np.roll(c, 1)))); tr[0] = h[0] - l[0]
    sm = lambda x: pd.Series(x).ewm(alpha=1 / n, adjust=False).mean().values
    atr_ = sm(tr); pdi = 100 * sm(pdm) / atr_; mdi = 100 * sm(mdm) / atr_
    dx = 100 * abs(pdi - mdi) / np.where(pdi + mdi == 0, np.nan, pdi + mdi)
    return sm(np.nan_to_num(dx))

def bars_since(cond):
    out = np.full(len(cond), 999.0); last = -10**9
    for i, x in enumerate(cond):
        if x: last = i
        out[i] = min(999, i - last)
    return out

def features(D, label):
    o, h, l, c, a, v, t = D.o, D.h, D.l, D.c, D.a, D.v, D.t; n = D.n
    h55 = D.h55; e100 = ema(c, 100)
    cross = ((o < D.e13) & (c > D.e13)) | ((o > D.e13) & (c < D.e13))
    flip = D.h55up != np.r_[D.h55up[0], D.h55up[:-1]]
    side = c > h55; side_flip = side != np.r_[side[0], side[:-1]]
    hi50 = pd.Series(h).rolling(50).max().values; lo50 = pd.Series(l).rolling(50).min().values
    sec = S.HTF[label][1]; g = t // sec
    cl = pd.DataFrame({"g": g, "c": c}).groupby("g")["c"].last()
    hm = pd.Series(hma(cl.values, 55), index=cl.index)
    hslope = ((hm - hm.shift(1)) / cl.rolling(14).std()).shift(1).reindex(g).values   # last completed HTF bar
    return {  # each is "raw" (direction-free) or signed later by trade direction d
        "HMA55 slope over 3 bars (ATR)":   ("signed", (h55 - np.r_[np.full(3, np.nan), h55[:-3]]) / a),
        "close distance from HMA55 (ATR)": ("signed", (c - h55) / a),
        "close distance from EMA100 (ATR)": ("signed", (c - e100) / a),
        "EMA13 slope over 3 bars (ATR)":   ("signed", (D.e13 - np.r_[np.full(3, np.nan), D.e13[:-3]]) / a),
        f"{S.HTF[label][0]} HMA55 slope (signed)": ("signed", hslope),
        "Donchian50 position (0=low,1=high)": ("signed01", (c - lo50) / np.where(hi50 - lo50 == 0, np.nan, hi50 - lo50)),
        "ADX 14":                          ("raw", adx(h, l, c)),
        "EMA13 body crosses in last 20":   ("raw", pd.Series(cross.astype(float)).rolling(20).sum().values),
        "bars since HMA55 turned":         ("raw", bars_since(flip)),
        "bars since close crossed HMA55":  ("raw", bars_since(side_flip)),
        "signal body / ATR":               ("raw", abs(c - o) / a),
        "signal range / ATR":              ("raw", (h - l) / a),
        "close location in candle (dir)":  ("signed01", (c - l) / np.where(h - l == 0, np.nan, h - l)),
        "relative volume (vs 20)":         ("raw", v / pd.Series(v).rolling(20).mean().shift(1).values),
        "ATR percentile (200 bars)":       ("raw", pd.Series(a).rolling(200).rank(pct=True).values),
        "hour (server time)":              ("raw", ((t // 3600) % 24).astype(float)),
    }

def run(label, df):
    D = S.Data(df); o, c = D.o, D.c; n = D.n; split = int(n * 0.6)
    up = (o < D.e13) & (c > D.e13) & (c > D.h55); dn = (o > D.e13) & (c < D.e13) & (c < D.h55)
    sig = np.where(up, 1, np.where(dn, -1, 0))
    final = tma_estimates(D.h, D.l, c)[0][H]; bm = (o + c) / 2
    rows = []
    for i in np.flatnonzero(sig):
        if i < 300 or i >= n - 5 or np.isnan(final[i]): continue
        d = int(sig[i]); a2 = S.trade(D, i, d, "close", "atr", "2R"); rr = S.trade(D, i, d, "close", "swing", "ema13")
        if a2 and rr: rows.append((i, d, a2[0], rr[0], (bm[i] < final[i]) if d > 0 else (bm[i] > final[i])))
    X = pd.DataFrame(rows, columns=["i", "d", "R_2R", "R_ride", "tma"]); X["is"] = X.i < split
    F = features(D, label)
    for name, (kind, arr) in F.items():
        x = arr[X.i.values]
        if kind == "signed": x = x * X.d.values
        elif kind == "signed01": x = np.where(X.d.values > 0, x, 1 - x)
        X[name] = x
    print(f"\n######## XAUUSD {label}: {len(X)} signals (EMA13 body cross + close on the HMA55 side) ########")
    for col in ("R_2R", "R_ride"):
        print(f"all signals  {col}: avgR IS {X[X['is']][col].mean():+.3f}  OOS {X[~X['is']][col].mean():+.3f}   "
              f"PF IS {S.pf(X[X['is']][col]):.2f} OOS {S.pf(X[~X['is']][col]):.2f}")
    print(f"TMA would approve {100 * X.tma.mean():.0f}% of these signals")
    print("\nPer quintile (Q1 = lowest value; signed features: higher = more in the trade direction). "
          "cells = avgR IS / avgR OOS, TMA% = share the hindsight TMA approves")
    summary = []
    for name in F:
        xs = X[name]; ok = xs.notna()
        if ok.sum() < 50: continue
        edges = np.unique(np.nanquantile(xs[X["is"] & ok], [0, .2, .4, .6, .8, 1]))
        if len(edges) < 3: continue
        q = np.clip(np.searchsorted(edges[1:-1], xs, side="right"), 0, len(edges) - 2)
        q = pd.Series(np.where(ok, q, -1), index=X.index)
        print(f"\n{name}")
        print(f"  {'quintile':>9} {'range':>17} {'n IS':>5} {'n OOS':>6} | {'2R: IS':>7} {'OOS':>7} | {'ride: IS':>8} {'OOS':>7} | {'TMA%':>5}")
        best = []
        for k in range(len(edges) - 1):
            m = q == k; mi = m & X["is"]; mo = m & ~X["is"]
            r = (X.R_2R[mi].mean(), X.R_2R[mo].mean(), X.R_ride[mi].mean(), X.R_ride[mo].mean())
            print(f"  {'Q' + str(k + 1):>9} {edges[k]:8.2f}..{edges[k + 1]:<8.2f} {mi.sum():5d} {mo.sum():6d} | "
                  f"{r[0]:+7.3f} {r[1]:+7.3f} | {r[2]:+8.3f} {r[3]:+7.3f} | {100 * X.tma[m].mean():5.0f}")
            best.append((k, r))
        summary.append((name, best, edges))
    return X, summary

if __name__ == "__main__":
    tfs = sys.argv[1:] or ["H1", "M5"]
    if not mt5.initialize(path=R.EXNESS_PATH):
        raise SystemExit(f"MT5 init failed: {mt5.last_error()}")
    sym = next(s.name for s in mt5.symbols_get() if s.name.upper().startswith("XAUUSD"))
    mt5.symbol_select(sym, True)
    data = {lb: pd.DataFrame(mt5.copy_rates_from_pos(sym, R.TFS[lb][0], 1, R.TFS[lb][1])) for lb in tfs}
    mt5.shutdown()
    for lb in tfs: run(lb, data[lb])
