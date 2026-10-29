"""
research_strategy_search.py — which REAL-TIME filter + trade management makes the EMA13 body cross most profitable?
XAUUSD, Exness MT5 data, 0.25 spread, closed candles only, worst case when SL and TP hit in the same candle.
Run with Exness MT5 open + logged in:   py research_strategy_search.py [H1] [M5]

Signal: bullish candle body crosses EMA13 (open < EMA13 < close) = BUY candidate, bearish = SELL candidate.
Filters (BUY side shown, SELL is the mirror): HMA 55/89 (side + slope, incl. Gemini's suggestions), Donchian 50,
EMA 50/100/200, LinReg slope, LWMA 50 discount, "dip in an uptrend" combos, higher-timeframe trend (M5->H1, H1->H4),
v7.1 volume. TRUE TMA (hindsight) is shown only as the unreachable reference.
Trade management variants:
  pull5/swing/triple  = user's rules (EMA5 pullback within 3 candles, fractal SL, 1R/2R/3R scale-out, BE after TP1)
  pull5/swing/2R      = same entry + stop, one target at 2R
  pull5/swing/3R+BE   = one target at 3R, SL to entry once +1R is reached
  pull5/swing/ema13   = no target, exit on the first close back across EMA13 (trend ride)
  close/swing/2R, close/swing/ema13, close/atr/2R, close/atr/ema13 = enter at the signal close (atr = 1.5 ATR14 stop)
  close/swing/hma     = enter at close, exit when HMA 55 turns against the trade
One trade at a time. IS = first 60% of the data (used to choose), OOS = last 40% (the honest check).
"""
import sys, bisect, numpy as np, pandas as pd, MetaTrader5 as mt5
import research_user_rules as R
from research_channel_replace import wma, ema, hma, linreg_end, donchian_mid
from research_tma_realtime import tma_estimates, H

SPREAD, MAXHOLD, WAIT = R.SPREAD, R.MAXHOLD, R.WAIT
MGMT = [("pull5", "swing", "triple"), ("pull5", "swing", "2R"), ("pull5", "swing", "3R+BE"), ("pull5", "swing", "ema13"),
        ("close", "swing", "2R"), ("close", "swing", "ema13"), ("close", "atr", "2R"), ("close", "atr", "ema13"),
        ("close", "swing", "hma")]
HTF = {"M5": ("H1", 3600), "H1": ("H4", 4 * 3600)}
SESS = {"all hours": None, "Loose 00-21": lambda hr: hr < 21,
        "Tight 00-07+12-16": lambda hr: (hr < 7) | ((hr >= 12) & (hr < 16))}

class Data:
    def __init__(self, df):
        self.o, self.h, self.l, self.c = (df[k].values.astype(float) for k in ("open", "high", "low", "close"))
        self.v = df["tick_volume"].values.astype(float); self.t = df["time"].values.astype(np.int64)
        self.n = len(self.c)
        self.e5, self.e13 = R.ema(self.c, 5), R.ema(self.c, 13); self.a = R.atr(self.h, self.l, self.c)
        self.h55 = hma(self.c, 55); self.h55up = self.h55 > np.r_[np.nan, self.h55[:-1]]
        fh, fl = R.fractals(self.h, self.l)
        self.F = {1: fl, -1: fh}; self.Fcb = {1: [x[0] for x in fl], -1: [x[0] for x in fh]}
        self.stops = {}

    def swing_stop(self, i, d):            # same rule as research_user_rules.swing_stop, with a bisect start
        key = (i, d)
        if key in self.stops: return self.stops[key]
        src, cbs, entry = self.F[d], self.Fcb[d], self.c[i]; s = None
        for k in range(bisect.bisect_right(cbs, i) - 1, -1, -1):
            cb, pb, pr = src[k]
            if i - pb > 60: break
            if (d > 0 and pr < entry) or (d < 0 and pr > entry): s = pr; break
        if s is None: s = self.l[i - 9:i + 1].min() if d > 0 else self.h[i - 9:i + 1].max()
        self.stops[key] = s; return s

def trade(D, i, d, em, sm, xm):
    o, h, l, c, n = D.o, D.h, D.l, D.c, D.n
    stop = D.swing_stop(i, d) if sm == "swing" else c[i] - d * 1.5 * D.a[i]
    hit_sl = lambda j, s: (l[j] <= s) if d > 0 else (h[j] >= s)
    hit_tp = lambda j, p: (h[j] >= p) if d > 0 else (l[j] <= p)
    touched = False
    if em == "pull5":
        entry = None
        for j in range(i + 1, i + 1 + WAIT):
            lv = D.e5[j - 1]                   # live EMA5 is touched exactly when price reaches last close's EMA5
            touched = (l[j] <= lv) if d > 0 else (h[j] >= lv)
            if touched:
                entry = min(o[j], lv) if d > 0 else max(o[j], lv); eb = j; break
            if hit_sl(j, stop): return None
        if entry is None: eb = i + WAIT; entry = c[eb]
    else:
        eb, entry = i, c[i]
    risk = (entry - stop) * d
    if risk <= 0: return None
    cost = SPREAD / risk; end = min(n - 1, eb + MAXHOLD)
    if touched and hit_sl(eb, stop): return -1.0 - cost, eb
    if xm == "triple":
        tps = [entry + d * k * risk for k in (1, 2, 3)]; sl = stop; hit = 0; legs = []; j = eb + 1
        while j <= end and hit < 3:
            if hit_sl(j, sl): legs += [(sl - entry) * d / risk] * (3 - hit); hit = 3; break
            while hit < 3 and hit_tp(j, tps[hit]):
                legs.append(hit + 1.0); hit += 1; sl = entry if hit == 1 else tps[hit - 2]
            j += 1
        if hit < 3: legs += [(c[min(j, n - 1)] - entry) * d / risk] * (3 - hit)
        return sum(legs) / 3 - cost, min(j, n - 1)
    tgt = {"2R": 2.0, "3R+BE": 3.0}.get(xm); tp = entry + d * tgt * risk if tgt else None
    sl = stop; be = entry + d * risk
    for j in range(eb + 1, end + 1):
        if hit_sl(j, sl): return (sl - entry) * d / risk - cost, j
        if tp is not None and hit_tp(j, tp): return tgt - cost, j
        if xm == "3R+BE" and hit_tp(j, be): sl = entry
        if xm == "ema13" and ((c[j] < D.e13[j]) if d > 0 else (c[j] > D.e13[j])):
            return (c[j] - entry) * d / risk - cost, j
        if xm == "hma" and (not D.h55up[j] if d > 0 else D.h55up[j]):
            return (c[j] - entry) * d / risk - cost, j
    return (c[end] - entry) * d / risk - cost, end

def pf(x): x = np.asarray(x); return x[x > 0].sum() / max(1e-9, -x[x < 0].sum()) if len(x) else 0.0

def backtest(D, sig, mg, days):
    """every signal traded independently (path-independent stats) + one-trade-at-a-time equity (realistic R / DD)"""
    split = int(D.n * 0.6); res = {}
    for i in np.flatnonzero(sig):
        if i < 250 or i >= D.n - WAIT - 2: continue
        tr = trade(D, i, int(sig[i]), *mg)
        if tr: res[i] = tr
    if len(res) < 10: return None
    idx = np.array(sorted(res)); x = np.array([res[i][0] for i in idx]); f = idx < split
    one = []; nxt = 0
    for i in idx:
        if i >= nxt: one.append(res[i][0]); nxt = res[i][1] + 1
    one = np.array(one); eq = np.cumsum(one); dd = (eq - np.maximum.accumulate(eq)).min()
    return dict(n=len(x), pm=len(x) / days * 30.4, win=100 * (x > 0).mean(), pf=pf(x), exp=x.mean(),
                is_pf=pf(x[f]), is_exp=x[f].mean(), oos_pf=pf(x[~f]), oos_exp=x[~f].mean(),
                n1=len(one), pm1=len(one) / days * 30.4, pf1=pf(one), tot=one.sum(), dd=dd)

def htf_slope_up(D, sec):
    g = D.t // sec
    cl = pd.DataFrame({"g": g, "c": D.c}).groupby("g")["c"].last()
    hm = pd.Series(hma(cl.values, 55), index=cl.index)
    up = (hm > hm.shift(1)).astype(float).where(hm.shift(1).notna()).shift(1)    # last COMPLETED HTF bar
    dn = (hm < hm.shift(1)).astype(float).where(hm.shift(1).notna()).shift(1)
    return up.reindex(g).fillna(0).values > 0, dn.reindex(g).fillna(0).values > 0

def filters(D, label):
    o, h, l, c, a = D.o, D.h, D.l, D.c, D.a; bm = (o + c) / 2
    h55, h89 = D.h55, hma(c, 89)
    up55, up89 = D.h55up, h89 > np.r_[np.nan, h89[:-1]]; dn55, dn89 = h55 < np.r_[np.nan, h55[:-1]], h89 < np.r_[np.nan, h89[:-1]]
    don = donchian_mid(h, l, 50); e50, e100, e200 = ema(c, 50), ema(c, 100), ema(c, 200)
    lr = linreg_end(c, 50); lrup, lrdn = lr > np.r_[np.nan, lr[:-1]], lr < np.r_[np.nan, lr[:-1]]
    lw = wma(c, 50); volok = D.v >= 1.2 * pd.Series(D.v).rolling(20).mean().shift(1).values
    hu, hd = htf_slope_up(D, HTF[label][1]); htf = HTF[label][0]
    final = tma_estimates(h, l, c)[0][H]
    F = {
        "NO filter (plain body cross)":             (np.ones(D.n, bool), np.ones(D.n, bool)),
        "TRUE TMA discount (hindsight, reference)": (bm < final, bm > final),
        "HMA55 below body (your eyeball rule)":      (bm > h55, bm < h55),
        "HMA55 below whole candle":                 (l > h55, h < h55),
        "HMA55 rising":                             (up55, dn55),
        "HMA55 below body AND rising":              (up55 & (bm > h55), dn55 & (bm < h55)),
        "HMA89 below body":                         (bm > h89, bm < h89),
        "HMA89 rising":                             (up89, dn89),
        "HMA55>HMA89, body above both (Gemini)":    ((h55 > h89) & (bm > h55), (h55 < h89) & (bm < h55)),
        "HMA55 ABOVE body (opposite, discount)":    (bm < h55, bm > h55),
        "Donchian50: body above mid (Gemini)":      (bm > don, bm < don),
        "Donchian50: body below mid (discount)":    (bm < don, bm > don),
        "EMA50 below body (Gemini)":                (bm > e50, bm < e50),
        "EMA100 below body (Gemini)":               (bm > e100, bm < e100),
        "LinReg50 rising (Gemini)":                 (lrup, lrdn),
        "LWMA50 discount":                          (bm < lw, bm > lw),
        "LWMA50 discount, >=0.5 ATR deep":          (bm < lw - 0.5 * a, bm > lw + 0.5 * a),
        "Dip in uptrend: HMA89 rising + LWMA50 disc": (up89 & (bm < lw), dn89 & (bm > lw)),
        "Dip in uptrend: above EMA200 + LWMA50 disc": ((bm > e200) & (bm < lw), (bm < e200) & (bm > lw)),
        f"{htf} HMA55 rising (higher-TF trend)":      (hu, hd),
        f"{htf} HMA55 rising + LWMA50 discount":      (hu & (bm < lw), hd & (bm > lw)),
        f"{htf} HMA55 rising + HMA55 rising":         (hu & up55, hd & dn55),
        "HMA55 rising + volume 1.2x (v7.1 vol)":    (up55 & volok, dn55 & volok),
        "LWMA50 discount + volume 1.2x":            (volok & (bm < lw), volok & (bm > lw)),
    }
    up = (o < D.e13) & (c > D.e13); dn = (o > D.e13) & (c < D.e13)
    return {k: np.where(up & np.nan_to_num(b).astype(bool), 1, np.where(dn & np.nan_to_num(s).astype(bool), -1, 0))
            for k, (b, s) in F.items()}

HDR = (f"{'':46}|{'------- every signal -------':^29}|{' IS 60% ':^13}|{' OOS 40% ':^13}|{'-- one trade at a time --':^27}\n"
       f"{'':46}|{'n':>5} {'/mo':>5} {'win%':>5} {'PF':>5} {'avgR':>6} |{'PF':>5} {'avgR':>7}|{'PF':>5} {'avgR':>7}|"
       f"{'/mo':>5} {'PF':>5} {'totR':>7} {'maxDD':>7}")
def row(name, s):
    if s is None: return f"{name:46}| too few trades"
    return (f"{name:46}|{s['n']:5d} {s['pm']:5.1f} {s['win']:5.1f} {s['pf']:5.2f} {s['exp']:+6.3f} |"
            f"{s['is_pf']:5.2f} {s['is_exp']:+7.3f}|{s['oos_pf']:5.2f} {s['oos_exp']:+7.3f}|"
            f"{s['pm1']:5.1f} {s['pf1']:5.2f} {s['tot']:+7.1f} {s['dd']:+7.1f}")

def run(label, df, out_csv):
    D = Data(df); days = (D.t[-1] - D.t[0]) / 86400; sigs = filters(D, label); rows = []
    hr = (D.t // 3600) % 24
    print(f"\n######## XAUUSD {label} ({days:.0f} days, {D.n} candles) ########")
    for mg in MGMT:
        name_m = "/".join(mg)
        for fname, sig in sigs.items():
            s = backtest(D, sig, mg, days)
            if s: rows.append(dict(tf=label, mgmt=name_m, filter=fname, session="all hours", **s))
    # sessions (M5 only): every filter x management again, restricted to the session
    if label == "M5":
        for sname, fn in SESS.items():
            if fn is None: continue
            m = fn(hr)
            for mg in MGMT:
                for fname, sig in sigs.items():
                    if fname.startswith("TRUE"): continue
                    s = backtest(D, np.where(m, sig, 0), mg, days)
                    if s: rows.append(dict(tf=label, mgmt="/".join(mg), filter=fname, session=sname, **s))
    res = pd.DataFrame(rows); res.to_csv(out_csv, index=False)

    def table(title, sub):
        print(f"\n=== {label}: {title} ===\n{HDR}")
        for _, r in sub.iterrows(): print(row(r["label"], r))
    allh = res[res.session == "all hours"]
    t1 = allh[allh.mgmt == "pull5/swing/triple"].copy(); t1["label"] = t1["filter"]
    table("every filter, YOUR trade management (pull5/swing/triple)", t1)
    for f in ("NO filter (plain body cross)", "HMA55 below body (your eyeball rule)", "HMA55 rising", "LWMA50 discount"):
        t = allh[allh["filter"] == f].copy(); t["label"] = t["mgmt"]
        table(f"'{f}' with every trade management", t)
    tradable = res[~res["filter"].str.startswith("TRUE") & (res.n >= (60 if label == "H1" else 300))].copy()
    tradable["label"] = tradable["filter"].str.slice(0, 26) + " | " + tradable["mgmt"].str.replace("pull5/", "p5/") \
        + (" | " + tradable.session.str.slice(0, 5)).where(tradable.session != "all hours", "")
    tradable["is_score"] = tradable.is_exp * np.sqrt(tradable.n * 0.6)
    table("TOP 20 chosen ONLY on the first 60% (avgR x sqrt n) -> OOS columns = honest result",
          tradable.sort_values("is_score", ascending=False).head(20))
    rob = tradable[(tradable.is_pf >= 1.15) & (tradable.oos_pf >= 1.15)].copy()
    rob["score"] = rob.tot / rob.dd.abs().clip(lower=1)
    table("ROBUST (IS PF>=1.15 AND OOS PF>=1.15), ranked by one-at-a-time total R / max drawdown",
          rob.sort_values("score", ascending=False).head(20))

if __name__ == "__main__":
    tfs = sys.argv[1:] or ["H1", "M5"]
    if not mt5.initialize(path=R.EXNESS_PATH):
        raise SystemExit(f"MT5 init failed: {mt5.last_error()}")
    sym = next(s.name for s in mt5.symbols_get() if s.name.upper().startswith("XAUUSD"))
    mt5.symbol_select(sym, True)
    data = {lb: pd.DataFrame(mt5.copy_rates_from_pos(sym, R.TFS[lb][0], 1, R.TFS[lb][1])) for lb in tfs}
    mt5.shutdown()
    for lb in tfs:
        run(lb, data[lb], f"strategy_search_{lb}.csv")
