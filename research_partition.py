"""
research_partition.py — which REAL-TIME indicator best replaces the TMA "partition" (which side of the
centre line the signal candle sits on)?  Any indicator type.  XAUUSD, Exness, user's trade rules, 0.25 spread.
Run with Exness MT5 open + logged in:   py research_partition.py [H1] [M5]

System: bullish EMA13 body cross = BUY candidate, bearish = SELL candidate.
Each partition decides if the candidate is "on the right side":
  BUY allowed when the indicator says price is LOW / cheap (like: body below the TMA centre line),
  SELL allowed when it says price is HIGH / expensive.   (Trend-follower rows are the opposite, as a control.)
All values use only closed/past data (higher-timeframe and daily levels use COMPLETED bars only).
"""
import sys, numpy as np, pandas as pd, MetaTrader5 as mt5
import research_user_rules as R
from research_tma_realtime import tma_estimates, H
from research_channel_replace import perf, ema, sma, wma

def rsi(c, n):
    d = np.diff(c, prepend=c[0]); up = pd.Series(np.maximum(d, 0)).ewm(alpha=1/n, adjust=False).mean()
    dn = pd.Series(np.maximum(-d, 0)).ewm(alpha=1/n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).values
def stoch(h, l, c, n):
    hh = pd.Series(h).rolling(n).max().values; ll = pd.Series(l).rolling(n).min().values
    return 100 * (c - ll) / np.where(hh - ll == 0, np.nan, hh - ll)
def cci(h, l, c, n):
    tp = (h + l + c) / 3; m = pd.Series(tp).rolling(n).mean()
    md = pd.Series(tp).rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return ((tp - m) / (0.015 * md)).values
def supertrend_up(h, l, c, n=10, mult=3.0):
    tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)), abs(l - np.roll(c, 1)))); tr[0] = h[0] - l[0]
    a = pd.Series(tr).ewm(alpha=1/n, adjust=False).mean().values; mid = (h + l) / 2
    ub = mid + mult * a; lb = mid - mult * a; fu = ub.copy(); fl = lb.copy(); up = np.ones(len(c), bool)
    for i in range(1, len(c)):
        fu[i] = ub[i] if (ub[i] < fu[i-1] or c[i-1] > fu[i-1]) else fu[i-1]
        fl[i] = lb[i] if (lb[i] > fl[i-1] or c[i-1] < fl[i-1]) else fl[i-1]
        up[i] = True if c[i] > fu[i-1] else (False if c[i] < fl[i-1] else up[i-1])
    return up

def htf_last_completed(t, o, h, l, c, sec):
    """map higher-timeframe bars (period sec) onto base bars, using only the last COMPLETED HTF bar"""
    g = t // sec
    df = pd.DataFrame({"g": g, "o": o, "h": h, "l": l, "c": c})
    agg = df.groupby("g").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
    return agg, g

def partitions(o, h, l, c, v, t, bm):
    P = {}
    lowhigh = lambda x, mid: (x < mid, x > mid)                    # (buy_ok, sell_ok) for "below = cheap"
    p4 = (h + l + 2 * c) / 4
    # ---- oscillators (cheap = below midpoint)
    for n in (7, 14, 21):
        r = rsi(c, n); P[f"RSI {n} < 50"] = (r < 50, r > 50)
    r = rsi(c, 14); P["RSI 14 < 45 (deeper)"] = (r < 45, r > 55)
    for n in (14, 34):
        s = stoch(h, l, c, n); P[f"Stochastic %K {n} < 50"] = (s < 50, s > 50)
    for n in (20, 50):
        x = cci(h, l, c, n); P[f"CCI {n} < 0"] = (x < 0, x > 0)
    for n in (12, 34):
        m = c - np.r_[np.full(n, np.nan), c[:-n]]; P[f"Momentum {n} < 0"] = (m < 0, m > 0)
    mh = ema(c, 12) - ema(c, 26); hist = mh - ema(mh, 9)
    P["MACD histogram < 0"] = (hist < 0, hist > 0)
    P["MACD line < 0"] = (mh < 0, mh > 0)
    # ---- value levels (cheap = body below level)
    for n in (34, 62, 100, 200):
        P[f"Body below EMA {n}"] = lowhigh(bm, ema(c, n))
    for n in (50, 100):
        z = (c - sma(c, n)) / pd.Series(c).rolling(n).std().values; P[f"Z-score {n} < 0"] = (z < 0, z > 0)
    for n in (50, 100):
        rv = (pd.Series(p4 * v).rolling(n).sum() / pd.Series(v).rolling(n).sum()).values
        P[f"Body below rolling VWAP {n}"] = lowhigh(bm, rv)
    # ---- range position
    for n in (34, 55, 100):
        hh = pd.Series(h).rolling(n).max().values; ll = pd.Series(l).rolling(n).min().values
        P[f"Donchian {n} position < 50%"] = lowhigh(bm, (hh + ll) / 2)
    kij = (pd.Series(h).rolling(26).max().values + pd.Series(l).rolling(26).min().values) / 2
    ten = (pd.Series(h).rolling(9).max().values + pd.Series(l).rolling(9).min().values) / 2
    spanA = np.r_[np.full(26, np.nan), ((ten + kij) / 2)[:-26]]
    spanB = np.r_[np.full(26, np.nan), ((pd.Series(h).rolling(52).max() + pd.Series(l).rolling(52).min()) / 2).values[:-26]]
    P["Ichimoku: body below Kijun"] = lowhigh(bm, kij)
    P["Ichimoku: body below cloud"] = (bm < np.fmin(spanA, spanB), bm > np.fmax(spanA, spanB))
    # last swing (fractal high / low) 50% level
    fh, fl = R.fractals(h, l); n = len(c)
    sh = np.full(n, np.nan); sl_ = np.full(n, np.nan)
    for cb, pb, pr in fh: sh[cb:] = pr
    for cb, pb, pr in fl: sl_[cb:] = pr
    P["Below 50% of last swing range"] = lowhigh(bm, (sh + sl_) / 2)
    # ---- daily / weekly reference levels (completed days only, except today's open which is known)
    day = t // 86400
    dfd = pd.DataFrame({"d": day, "o": o, "h": h, "l": l, "c": c}).groupby("d").agg(
        o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
    prev = dfd.shift(1)
    today_open = dfd["o"].reindex(day).values
    pmid = ((prev["h"] + prev["l"]) / 2).reindex(day).values
    piv = ((prev["h"] + prev["l"] + prev["c"]) / 3).reindex(day).values
    P["Body below today's open"] = lowhigh(bm, today_open)
    P["Body below previous-day midpoint"] = lowhigh(bm, pmid)
    P["Body below daily pivot"] = lowhigh(bm, piv)
    wk = (t + 3 * 86400) // (7 * 86400)                            # weeks starting Monday
    wopen = pd.DataFrame({"w": wk, "o": o}).groupby("w")["o"].first().reindex(wk).values
    P["Body below weekly open"] = lowhigh(bm, wopen)
    # ---- higher timeframe (H4), completed bars only
    agg, g = htf_last_completed(t, o, h, l, c, 4 * 3600)
    e = pd.Series(ema(agg["c"].values, 13), index=agg.index).shift(1)
    P["Body below H4 EMA13 (last closed H4)"] = lowhigh(bm, e.reindex(g).values)
    dmid = ((agg["h"].rolling(20).max() + agg["l"].rolling(20).min()) / 2).shift(1)
    P["Body below H4 Donchian 20 mid"] = lowhigh(bm, dmid.reindex(g).values)
    # ---- trend-followers (CONTROL: opposite logic — buy only in uptrend)
    st = supertrend_up(h, l, c)
    P["[control] Supertrend up for BUY"] = (st, ~st)
    e62 = ema(c, 62)
    P["[control] Body ABOVE EMA 62 for BUY"] = (bm > e62, bm < e62)
    return P

def run(label, df):
    o, h, l, c = (df[k].values.astype(float) for k in ("open", "high", "low", "close")); n = len(c)
    v = df["tick_volume"].values.astype(float); t = df["time"].values.astype(np.int64)
    days = (t[-1] - t[0]) / 86400
    e5, e13 = R.ema(c, 5), R.ema(c, 13); fh, fl = R.fractals(h, l)
    est, _, _, atrx = tma_estimates(h, l, c); final = est[H]
    bm = (o + c) / 2
    up = (o < e13) & (c > e13); dn = (o > e13) & (c < e13)
    base_ok = (up | dn) & ~np.isnan(final)
    truth = np.where(up, bm < final, bm > final)
    print(f"\n===== XAUUSD {label} ({days:.0f} days) — EMA13 body cross + partition =====")
    print(f"{'partition':40} {'same as':>8} | {'---- one trade at a time ----':^31} | {'---- every signal ----':^24}")
    print(f"{'':40} {'true TMA':>8} | {'n':>5} {'win%':>5} {'PF':>5} {'totR':>7} {'maxDD':>6} | {'n':>5} {'PF':>5} {'1st':>5} {'2nd':>5}")
    rows = [("TRUE TMA (hindsight, not tradable)", (bm < final, bm > final)),
            ("NO partition (plain v7)", (np.ones(n, bool), np.ones(n, bool))),
            ("Live TMA dots", (bm < est[0], bm > est[0]))] + list(partitions(o, h, l, c, v, t, bm).items())
    for name, (bok, sok) in rows:
        bok = np.nan_to_num(bok).astype(bool); sok = np.nan_to_num(sok).astype(bool)
        vt = np.where(up, bok, sok)
        same = (vt[base_ok] == truth[base_ok]).mean()
        sig = np.where(up & bok, 1, np.where(dn & sok, -1, 0))
        print(f"{name:40} {100 * same:7.1f}% | " + perf(sig, o, h, l, c, e5, fh, fl, n), flush=True)

if __name__ == "__main__":
    tfs = sys.argv[1:] or ["H1", "M5"]
    if not mt5.initialize(path=R.EXNESS_PATH):
        raise SystemExit(f"MT5 init failed: {mt5.last_error()}")
    sym = next(s.name for s in mt5.symbols_get() if s.name.upper().startswith("XAUUSD"))
    mt5.symbol_select(sym, True)
    for label in tfs:
        run(label, pd.DataFrame(mt5.copy_rates_from_pos(sym, R.TFS[label][0], 1, R.TFS[label][1])))
    mt5.shutdown()
