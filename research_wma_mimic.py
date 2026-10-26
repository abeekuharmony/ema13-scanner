"""
research_wma_mimic.py — can a plain TradingView "Moving Average Weighted" (WMA) stand in for the FINAL
(centred) TMA middle line?  XAUUSD M5 + H1, Exness data.
Run with Exness MT5 open + logged in:   py research_wma_mimic.py

Side rule (user): BUY = bullish body cross of EMA13 with most of the body BELOW the line
                  (body midpoint (open+close)/2 < line). SELL = bearish cross, body midpoint ABOVE the line.
For each WMA length/source we measure, over all EMA13 body-cross candles, how often its BUY/SELL verdict
matches the verdict of the final centred TMA line (which is only known 12 candles later).
Then the best settings are backtested with the user's trade rules (EMA5 pullback entry, swing SL,
1R/2R/3R scale-out, 0.25 spread) using v7.1 filters (volume >= 1.2x, candle < 75% of channel).
"""
import numpy as np, pandas as pd, MetaTrader5 as mt5
import research_user_rules as R
from research_tma import evaluate
from research_tma_realtime import tma_estimates, H

def wma(x, L):
    w = np.arange(1, L + 1).astype(float)
    out = np.convolve(x, w[::-1])[:len(x)] / w.sum()
    out[:L - 1] = np.nan
    return out

def run(label, df):
    o, h, l, c = (df[k].values.astype(float) for k in ("open", "high", "low", "close")); n = len(c)
    v = df["tick_volume"].values.astype(float)
    days = (df["time"].iloc[-1] - df["time"].iloc[0]) / 86400
    e5, e13 = R.ema(c, 5), R.ema(c, 13); fh, fl = R.fractals(h, l)
    est, ext, width, atrx = tma_estimates(h, l, c)
    final = est[H]; live = est[0]
    bm = (o + c) / 2
    up = (o < e13) & (c > e13); dn = (o > e13) & (c < e13)
    cross = (up | dn) & ~np.isnan(final) & ~np.isnan(atrx)
    truth = np.where(up, bm < final, bm > final)                  # verdict of the final line
    def verdict(line): return np.where(up, bm < line, bm > line)
    def agree(line):
        ok = cross & ~np.isnan(line)
        vt, tr = verdict(line)[ok], truth[ok]
        acc = (vt == tr).mean()
        prec = (vt & tr).sum() / max(1, vt.sum())                 # of signals it ALLOWS, % the final line also allows
        rec = (vt & tr).sum() / max(1, tr.sum())                  # of signals the final line allows, % it catches
        return acc, prec, rec
    srcs = {"close": c, "hl2": (h + l) / 2, "hlc3": (h + l + c) / 3, "ohlc4": (o + h + l + c) / 4,
            "hlcc4": (h + l + 2 * c) / 4}
    print(f"\n===== XAUUSD {label} ({days:.0f} days, {cross.sum()} EMA13 body crosses) =====")
    print("How well each line predicts the FINAL TMA verdict (agree% / precision% / recall%)")
    a = agree(live); print(f"  {'Brylator live end (what TMA dots show now)':44} {100*a[0]:5.1f} / {100*a[1]:5.1f} / {100*a[2]:5.1f}")
    a = agree(ext);  print(f"  {'linear-trend end estimate':44} {100*a[0]:5.1f} / {100*a[1]:5.1f} / {100*a[2]:5.1f}")
    results = []
    for sname, s in srcs.items():
        for L in range(2, 31):
            acc, p, r = agree(wma(s, L)); results.append((acc, p, r, sname, L))
    results.sort(reverse=True)
    print("  Best WMA settings:")
    for acc, p, r, sname, L in results[:8]:
        print(f"  {'WMA ' + str(L) + ' on ' + sname:44} {100*acc:5.1f} / {100*p:5.1f} / {100*r:5.1f}")
    for L in (5, 13, 20):
        acc, p, r = agree(wma(srcs["hlcc4"], L))
        print(f"  {'(ref) WMA ' + str(L) + ' on hlcc4':44} {100*acc:5.1f} / {100*p:5.1f} / {100*r:5.1f}")

    # ---- PF with user's trade rules, v7.1 filters
    filt = ((h - l) < 0.75 * width) & (v >= 1.2 * pd.Series(v).rolling(20).mean().shift(1).values)
    def sig(line):
        vt = verdict(line)
        return np.where(cross & filt & vt & up, 1, np.where(cross & filt & vt & dn, -1, 0))
    tests = [("HINDSIGHT final TMA (ceiling, not tradable)", final),
             ("Brylator live end (v7.1 style)", live)]
    for acc, p, r, sname, L in results[:3]:
        tests.append((f"WMA {L} on {sname}", wma(srcs[sname], L)))
    print(f"\n{'line used for the side rule':46} {'------ one trade at a time ------':^37} | {'---- every signal ----':^24}")
    print(f"{'':46} {'n':>5} {'/day':>5} {'win%':>5} {'PF':>5} {'totR':>7} {'maxDD':>6} | {'n':>5} {'PF':>5} {'1st':>5} {'2nd':>5}")
    for name, line in tests:
        evaluate(f"{name:46}"[:46], sig(line), o, h, l, c, e5, fh, fl, n, days)

if __name__ == "__main__":
    if not mt5.initialize(path=R.EXNESS_PATH):
        raise SystemExit(f"MT5 init failed: {mt5.last_error()}")
    sym = next(s.name for s in mt5.symbols_get() if s.name.upper().startswith("XAUUSD"))
    mt5.symbol_select(sym, True)
    for label, (tf, count) in [("H1", R.TFS["H1"]), ("M5", R.TFS["M5"])]:
        run(label, pd.DataFrame(mt5.copy_rates_from_pos(sym, tf, 1, count)))
    mt5.shutdown()
