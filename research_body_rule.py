"""
research_body_rule.py — which body-expansion setting works best? XAUUSD, user's own trade management.
M5: v7.6 (RVOL spike within 3 + body rule) with different body settings, vs no body rule (= v7.4).
H1: v7.2 (RVOL 1.5x) vs v7.2 + body 1.2x avg14.
Rules (same as research_user_rules.py): EMA5 pullback entry (wait 3, else market), fractal swing stop,
1R/2R/3R scale-out with SL to BE after TP1 and to TP1 after TP2, worst-case same-candle, 0.25 spread.
"""
import numpy as np, pandas as pd, MetaTrader5 as mt5
import research_user_rules as R

def trade(i, d, o, h, l, c, e5, fh, fl):
    """one trade from signal candle i; returns (R after spread, exit bar) or None if skipped"""
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

def pf(x): x = np.array(x); return x[x > 0].sum() / max(1e-9, -x[x < 0].sum())

def evaluate(name, sig, o, h, l, c, e5, fh, fl, n, days):
    res = {}
    for i in range(60, n - R.WAIT - 2):
        if sig[i] != 0:
            t = trade(i, sig[i], o, h, l, c, e5, fh, fl)
            if t: res[i] = t
    # one trade at a time
    one = []; nxt = 0
    for i in sorted(res):
        if i >= nxt: one.append(res[i][0]); nxt = res[i][1]
    one = np.array(one); eq = np.cumsum(one); dd = (eq - np.maximum.accumulate(eq)).min() if len(one) else 0
    allr = np.array([res[i][0] for i in sorted(res)]); first = np.array(sorted(res)) < n // 2
    print(f"{name:34} {len(one):5d} {len(one)/days:5.2f} {pf(one):5.2f} {one.sum():+7.1f} {dd:+6.1f} | "
          f"{len(allr):5d} {pf(allr):5.2f} {pf(allr[first]):5.2f} {pf(allr[~first]):5.2f}")

def run(label, df):
    o, h, l, c = (df[k].values.astype(float) for k in ("open", "high", "low", "close"))
    v = df["tick_volume"].values.astype(float); n = len(c)
    days = (df["time"].iloc[-1] - df["time"].iloc[0]) / 86400
    e5, e13, a = R.ema(c, 5), R.ema(c, 13), R.atr(h, l, c)
    fh, fl = R.fractals(h, l)
    d = np.where((o < e13) & (c > e13), 1, np.where((o > e13) & (c < e13), -1, 0))
    spike = v >= 1.5 * pd.Series(v).rolling(20).mean().shift(1).values
    spike3 = pd.Series(spike.astype(float)).rolling(3).max().values > 0
    body = np.abs(c - o)
    avgb = lambda k: pd.Series(body).rolling(k).mean().shift(1).values
    if label == "M5":
        tests = {
            "v7.6 body >= 1.5x avg20 (current)": spike3 & (body >= 1.5 * avgb(20)),
            "v7.6 body >= 1.2x avg14":           spike3 & (body >= 1.2 * avgb(14)),
            "v7.6 body >= 1.0x avg14":           spike3 & (body >= 1.0 * avgb(14)),
            "v7.6 body >= 1.0x ATR":             spike3 & (body >= 1.0 * a),
            "v7.6 body >= 1.2x ATR":             spike3 & (body >= 1.2 * a),
            "no body rule (= v7.4)":             spike3,
        }
    else:
        tests = {
            "v7.2 RVOL 1.5x (current)":          spike,
            "v7.2 + body >= 1.2x avg14":         spike & (body >= 1.2 * avgb(14)),
            "v7.2 + body >= 1.5x avg20":         spike & (body >= 1.5 * avgb(20)),
        }
    print(f"\n===== XAUUSD {label} ({days:.0f} days) — after 0.25 spread =====")
    print(f"{'version':34} {'--- one trade at a time ---':^31} | {'--- every signal ---':^24}")
    print(f"{'':34} {'n':>5} {'/day':>5} {'PF':>5} {'totR':>7} {'maxDD':>6} | {'n':>5} {'PF':>5} {'1st':>5} {'2nd':>5}")
    for name, ok in tests.items():
        evaluate(name, np.where(ok, d, 0), o, h, l, c, e5, fh, fl, n, days)

if not mt5.initialize(path=R.EXNESS_PATH):
    raise SystemExit(f"MT5 init failed: {mt5.last_error()}")
sym = next(s.name for s in mt5.symbols_get() if s.name.upper().startswith("XAUUSD"))
mt5.symbol_select(sym, True)
for label, (tf, count) in R.TFS.items():
    run(label, pd.DataFrame(mt5.copy_rates_from_pos(sym, tf, 1, count)))
mt5.shutdown()
