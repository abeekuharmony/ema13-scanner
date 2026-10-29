"""
research_false_signal_rules.py — test the filters suggested by research_false_signals.py as complete rules (IS/OOS).
Base = EMA13 body cross with the close on the HMA55 side (user's HMA rule).  XAUUSD, Exness, 0.25 spread.
Run with Exness MT5 open + logged in:   py research_false_signal_rules.py [H1] [M5]
"""
import sys, numpy as np, pandas as pd, MetaTrader5 as mt5
import research_user_rules as R
import research_strategy_search as S
from research_channel_replace import ema
from research_strategy_robust import htf_rising

def run(label, df):
    D = S.Data(df); days = (D.t[-1] - D.t[0]) / 86400; o, h, l, c, a = D.o, D.h, D.l, D.c, D.a
    up = (o < D.e13) & (c > D.e13); dn = (o > D.e13) & (c < D.e13)
    hb, hs = c > D.h55, c < D.h55                                   # user's HMA55 rule
    e100 = ema(c, 100); tb, ts = c > e100, c < e100                 # trend side of EMA100
    hu, hd = htf_rising(D, S.HTF[label][1], 55); htf = S.HTF[label][0]
    quiet = (h - l) < 1.0 * a                                       # signal candle smaller than a normal candle
    rv = D.v / pd.Series(D.v).rolling(20).mean().shift(1).values; lowvol = rv < 1.0
    nearb = (c - D.h55) < 0.5 * a; nears = (D.h55 - c) < 0.5 * a     # close within 0.5 ATR of HMA55
    hr = (D.t // 3600) % 24; ny = (hr >= 13) & (hr < 18)
    sg = lambda b, s: np.where(up & b, 1, np.where(dn & s, -1, 0))
    rules = {
        "HMA55 side (your rule)":                         sg(hb, hs),
        "+ EMA100 side":                                  sg(hb & tb, hs & ts),
        f"+ EMA100 side + {htf} HMA55 rising":            sg(hb & tb & hu, hs & ts & hd),
        "+ quiet candle (<1 ATR)":                        sg(hb & quiet, hs & quiet),
        "+ volume below average":                         sg(hb & lowvol, hs & lowvol),
        "+ close within 0.5 ATR of HMA55":                sg(hb & nearb, hs & nears),
        "+ EMA100 side + quiet candle":                   sg(hb & tb & quiet, hs & ts & quiet),
        "+ EMA100 side + close near HMA55":               sg(hb & tb & nearb, hs & ts & nears),
        f"+ EMA100 + {htf} rising + quiet":               sg(hb & tb & hu & quiet, hs & ts & hd & quiet),
        "+ 13-18h server time":                           sg(hb & ny, hs & ny),
        "+ EMA100 side + 13-18h":                         sg(hb & tb & ny, hs & ts & ny),
    }
    print(f"\n######## XAUUSD {label} ({days:.0f} days) — false-signal filters on top of the HMA55 rule ########")
    for mg in (("close", "swing", "ema13"), ("close", "atr", "2R"), ("pull5", "swing", "triple")):
        print(f"--- {'/'.join(mg)}\n{S.HDR}")
        for k, s in rules.items(): print(S.row(k, S.backtest(D, s, mg, days)))

if __name__ == "__main__":
    tfs = sys.argv[1:] or ["H1", "M5"]
    if not mt5.initialize(path=R.EXNESS_PATH):
        raise SystemExit(f"MT5 init failed: {mt5.last_error()}")
    sym = next(s.name for s in mt5.symbols_get() if s.name.upper().startswith("XAUUSD"))
    mt5.symbol_select(sym, True)
    data = {lb: pd.DataFrame(mt5.copy_rates_from_pos(sym, R.TFS[lb][0], 1, R.TFS[lb][1])) for lb in tfs}
    mt5.shutdown()
    for lb in tfs: run(lb, data[lb])
