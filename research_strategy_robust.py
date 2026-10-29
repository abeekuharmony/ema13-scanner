"""
research_strategy_robust.py — are the winners of research_strategy_search.py real or lucky settings?
Neighbouring parameters for the trend filter, the exit line, and a quarter-by-quarter breakdown.
Run with Exness MT5 open + logged in:   py research_strategy_robust.py [H1] [M5]
"""
import sys, numpy as np, pandas as pd, MetaTrader5 as mt5
import research_user_rules as R
import research_strategy_search as S
from research_channel_replace import ema, hma, donchian_mid

def htf_rising(D, sec, L):
    g = D.t // sec
    cl = pd.DataFrame({"g": g, "c": D.c}).groupby("g")["c"].last()
    hm = pd.Series(hma(cl.values, L), index=cl.index); prev = hm.shift(1)
    up = (hm > prev).astype(float).where(prev.notna()).shift(1).reindex(g).fillna(0).values > 0
    dn = (hm < prev).astype(float).where(prev.notna()).shift(1).reindex(g).fillna(0).values > 0
    return up, dn

def run(label, df):
    D = S.Data(df); days = (D.t[-1] - D.t[0]) / 86400; o, h, l, c = D.o, D.h, D.l, D.c; bm = (o + c) / 2
    up = (o < D.e13) & (c > D.e13); dn = (o > D.e13) & (c < D.e13)
    sig = lambda b, s: np.where(up & b, 1, np.where(dn & s, -1, 0))
    F = {"NO filter": sig(np.ones(D.n, bool), np.ones(D.n, bool))}
    for L in (50, 75, 100, 150, 200):
        e = ema(c, L); F[f"body above EMA{L}"] = sig(bm > e, bm < e)
    sec = S.HTF[label][1]; htf = S.HTF[label][0]
    for L in (34, 55, 89):
        u, d = htf_rising(D, sec, L); F[f"{htf} HMA{L} rising"] = sig(u, d)
    for L in (34, 50, 100):
        m = donchian_mid(h, l, L); F[f"body above Donchian{L} mid"] = sig(bm > m, bm < m)
    e100 = ema(c, 100); u, d = htf_rising(D, sec, 55)
    F[f"EMA100 + {htf} HMA55 rising"] = sig((bm > e100) & u, (bm < e100) & d)
    if label == "M5":
        hr = (D.t // 3600) % 24; tight = (hr < 7) | ((hr >= 12) & (hr < 16)); loose = hr < 21
        for k in list(F):
            if k != "NO filter": F[k + " | Tight"] = np.where(tight, F[k], 0); F[k + " | Loose"] = np.where(loose, F[k], 0)
    print(f"\n######## XAUUSD {label} ({days:.0f} days) — parameter neighbours, exit = first close back across EMA13 ########")
    print(S.HDR.replace("|------- every", "|------- every"))
    for mg in (("close", "swing", "ema13"), ("pull5", "swing", "ema13")):
        print(f"--- entry/stop/exit: {'/'.join(mg)}")
        for k, s_ in F.items(): print(S.row(k, S.backtest(D, s_, mg, days)))
    # exit line neighbours for the two main families
    e13 = D.e13.copy()
    print(f"--- exit line neighbours (close/swing/<exit EMA>)")
    for k in ("NO filter", "body above EMA100", f"{htf} HMA55 rising"):
        for L in (8, 13, 21, 34):
            D.e13 = ema(c, L); print(S.row(f"{k} | exit EMA{L}", S.backtest(D, F[k], ("close", "swing", "ema13"), days)))
    D.e13 = e13
    # quarter by quarter (every signal), close/swing/ema13
    q = pd.to_datetime(D.t, unit="s").to_period("Q").astype(str)
    print(f"--- quarter by quarter, every signal, close/swing/ema13: PF (n)")
    keys = ["NO filter", "body above EMA100", f"{htf} HMA55 rising", f"EMA100 + {htf} HMA55 rising"]
    if label == "M5": keys += [k + " | Tight" for k in keys[1:]]
    res = {k: {} for k in keys}
    for k in keys:
        for i in np.flatnonzero(F[k]):
            if 250 <= i < D.n - 5:
                tr = S.trade(D, i, int(F[k][i]), "close", "swing", "ema13")
                if tr: res[k].setdefault(q[i], []).append(tr[0])
    qs = sorted(set(q[250:]))
    print(f"{'':40}" + "".join(f"{x:>14}" for x in qs))
    for k in keys:
        print(f"{k:40}" + "".join(f"{S.pf(res[k].get(x, [])):7.2f} ({len(res[k].get(x, [])):4d})" for x in qs))

if __name__ == "__main__":
    tfs = sys.argv[1:] or ["H1", "M5"]
    if not mt5.initialize(path=R.EXNESS_PATH):
        raise SystemExit(f"MT5 init failed: {mt5.last_error()}")
    sym = next(s.name for s in mt5.symbols_get() if s.name.upper().startswith("XAUUSD"))
    mt5.symbol_select(sym, True)
    data = {lb: pd.DataFrame(mt5.copy_rates_from_pos(sym, R.TFS[lb][0], 1, R.TFS[lb][1])) for lb in tfs}
    mt5.shutdown()
    for lb in tfs: run(lb, data[lb])
