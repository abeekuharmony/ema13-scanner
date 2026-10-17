"""
research_v71_rvol.py — v7.1 (EMA13 body cross + volume spike + TMA rules 2+3) at different volume thresholds.
XAUUSD M5 + H1, user's trade rules, 0.25 spread.   Run with Exness MT5 open:  py research_v71_rvol.py
"""
import numpy as np, pandas as pd, MetaTrader5 as mt5
import research_user_rules as R
from research_tma import tma, evaluate

def run(label, df):
    o, h, l, c = (df[k].values.astype(float) for k in ("open", "high", "low", "close")); n = len(c)
    v = df["tick_volume"].values.astype(float)
    days = (df["time"].iloc[-1] - df["time"].iloc[0]) / 86400
    e5, e13 = R.ema(c, 5), R.ema(c, 13); fh, fl = R.fractals(h, l)
    d = np.where((o < e13) & (c > e13), 1, np.where((o > e13) & (c < e13), -1, 0))
    lm, _, _, _, width = tma(h, l, c)
    rng_ok = (h - l) < 0.75 * width
    vavg = pd.Series(v).rolling(20).mean().shift(1).values
    def v71(k):
        ok = rng_ok & (v >= k * vavg) if k else rng_ok
        return np.where((d > 0) & ok & (o < lm), 1, np.where((d < 0) & ok & (o > lm), -1, 0))
    print(f"\n===== XAUUSD {label} ({days:.0f} days) — v7.1 rules, your trade rules, after 0.25 spread =====")
    print(f"{'volume threshold':40} {'------ one trade at a time ------':^37} | {'---- every signal ----':^24}")
    print(f"{'':40} {'n':>5} {'/day':>5} {'win%':>5} {'PF':>5} {'totR':>7} {'maxDD':>6} | {'n':>5} {'PF':>5} {'1st':>5} {'2nd':>5}")
    for k in (None, 1.0, 1.2, 1.5):
        evaluate("no volume rule" if k is None else f"volume >= {k}x average" + (" (current v7.1)" if k == 1.5 else ""),
                 v71(k), o, h, l, c, e5, fh, fl, n, days)

if __name__ == "__main__":
    if not mt5.initialize(path=R.EXNESS_PATH):
        raise SystemExit(f"MT5 init failed: {mt5.last_error()}")
    sym = next(s.name for s in mt5.symbols_get() if s.name.upper().startswith("XAUUSD"))
    mt5.symbol_select(sym, True)
    for label, (tf, count) in R.TFS.items():
        run(label, pd.DataFrame(mt5.copy_rates_from_pos(sym, tf, 1, count)))
    mt5.shutdown()
