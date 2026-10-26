"""
research_lwma_zones.py — can an LWMA (TradingView "Moving Average Weighted") mark PREMIUM / DISCOUNT zones
for the EMA13 body-cross system?  XAUUSD, Exness, user's trade rules, 0.25 spread.
Run with Exness MT5 open + logged in:   py research_lwma_zones.py [H1] [M5]

Discount = candle body midpoint BELOW the LWMA  -> BUY allowed (bullish EMA13 body cross)
Premium  = candle body midpoint ABOVE the LWMA  -> SELL allowed (bearish EMA13 body cross)
Tests: LWMA length x source; "deep zone" (body at least k x ATR beyond the LWMA);
       two-LWMA zone (beyond both a medium and a long LWMA).
"""
import sys, numpy as np, pandas as pd, MetaTrader5 as mt5
import research_user_rules as R
from research_channel_replace import perf, wma

def run(label, df):
    o, h, l, c = (df[k].values.astype(float) for k in ("open", "high", "low", "close")); n = len(c)
    t = df["time"].values; days = (t[-1] - t[0]) / 86400
    e5, e13 = R.ema(c, 5), R.ema(c, 13); fh, fl = R.fractals(h, l); a = R.atr(h, l, c)
    bm = (o + c) / 2
    up = (o < e13) & (c > e13); dn = (o > e13) & (c < e13)
    src = {"close": c, "hlcc4": (h + l + 2 * c) / 4}
    rows = [("NO zone filter (plain v7)", np.where(up, 1, np.where(dn, -1, 0)))]
    def zone(line, k=0.0):
        return np.where(up & (bm < line - k * a), 1, np.where(dn & (bm > line + k * a), -1, 0))
    for sname, s in src.items():
        for L in (13, 21, 34, 50, 62, 89, 100, 144, 200):
            rows.append((f"LWMA {L} {sname}", zone(wma(s, L))))
    p = src["close"]
    for L in (34, 50, 100):
        for k in (0.5, 1.0, 1.5):
            rows.append((f"LWMA {L} close, deep >= {k} ATR", zone(wma(p, L), k)))
    for m1, m2 in ((21, 100), (34, 100), (34, 200), (50, 200)):
        w1, w2 = wma(p, m1), wma(p, m2)
        rows.append((f"Beyond BOTH LWMA {m1} & {m2} close",
                     np.where(up & (bm < np.fmin(w1, w2)), 1, np.where(dn & (bm > np.fmax(w1, w2)), -1, 0))))
    print(f"\n===== XAUUSD {label} ({days:.0f} days) — EMA13 body cross + LWMA premium/discount =====")
    print(f"{'zone rule':40} | {'---- one trade at a time ----':^31} | {'---- every signal ----':^24}")
    print(f"{'':40} | {'n':>5} {'win%':>5} {'PF':>5} {'totR':>7} {'maxDD':>6} | {'n':>5} {'PF':>5} {'1st':>5} {'2nd':>5}")
    for name, sig in rows:
        print(f"{name:40} | " + perf(sig, o, h, l, c, e5, fh, fl, n), flush=True)

if __name__ == "__main__":
    tfs = sys.argv[1:] or ["H1", "M5"]
    if not mt5.initialize(path=R.EXNESS_PATH):
        raise SystemExit(f"MT5 init failed: {mt5.last_error()}")
    sym = next(s.name for s in mt5.symbols_get() if s.name.upper().startswith("XAUUSD"))
    mt5.symbol_select(sym, True)
    for label in tfs:
        run(label, pd.DataFrame(mt5.copy_rates_from_pos(sym, R.TFS[label][0], 1, R.TFS[label][1])))
    mt5.shutdown()
