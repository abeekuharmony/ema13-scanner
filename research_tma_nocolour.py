"""
research_tma_nocolour.py — v7.1 TMA WITHOUT the colour rule (rules 2 + 3 only), XAUUSD, user's trade rules.
Run with Exness MT5 open + logged in:   py research_tma_nocolour.py
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
    lm, lup, cm, cup, width = tma(h, l, c)
    rng_ok = (h - l) < 0.75 * width
    spike = v >= 1.5 * pd.Series(v).rolling(20).mean().shift(1).values
    def r23(mid, extra=True):
        return np.where((d > 0) & rng_ok & (o < mid) & extra, 1, np.where((d < 0) & rng_ok & (o > mid) & extra, -1, 0))
    tests = {
        "v7 plain cross":                          d,
        "v7.1 LIVE all 3 rules (last test)":       np.where(((d > 0) & lup | (d < 0) & ~lup), r23(lm), 0),
        "v7.1 LIVE rules 2+3, no colour <- real":  r23(lm),
        "v7.1 HINDSIGHT rules 2+3, no colour":     r23(cm),
        "v7.2 RVOL (reference)":                   np.where(spike, d, 0),
        "v7.2 RVOL + LIVE rules 2+3":              r23(lm, spike),
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
