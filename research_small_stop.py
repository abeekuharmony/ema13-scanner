"""
research_small_stop.py — the trend-side EMA13 body-cross strategy (HMA55 side + EMA100 side + higher-TF HMA55 rising)
on M15 / M30 / H1 with tighter stops, so the dollar stop fits a small account.  XAUUSD, Exness, 0.25 spread.
Run with Exness MT5 open + logged in:   py research_small_stop.py
Entry at the signal candle close. Stop = k x ATR14 (k = 0.75, 1.0, 1.25, 1.5). Exit = first close back across EMA13.
"""
import numpy as np, pandas as pd, MetaTrader5 as mt5
import research_user_rules as R, research_strategy_search as S
from research_channel_replace import ema
from research_strategy_robust import htf_rising
import research_tma_confirmed as T

TF = {"M15": (mt5.TIMEFRAME_M15, 30000, 3600, "H1"), "M30": (mt5.TIMEFRAME_M30, 15000, 4 * 3600, "H4"),
      "H1": (mt5.TIMEFRAME_H1, 6000, 4 * 3600, "H4")}
mt5.initialize(path=R.EXNESS_PATH); sym = next(s.name for s in mt5.symbols_get() if s.name.upper().startswith("XAUUSD"))
mt5.symbol_select(sym, True)
data = {k: pd.DataFrame(mt5.copy_rates_from_pos(sym, v[0], 1, v[1])) for k, v in TF.items()}; mt5.shutdown()
for lb, (_, _, sec, htf) in TF.items():
    D = S.Data(data[lb]); days = (D.t[-1] - D.t[0]) / 86400; c = D.c; e100 = ema(c, 100); hu, hd = htf_rising(D, sec, 55)
    up = (D.o < D.e13) & (c > D.e13); dn = (D.o > D.e13) & (c < D.e13)
    flt = {"trend (EMA100+HMA55+%s rising)" % htf: ((c > e100) & (c > D.h55) & hu, (c < e100) & (c < D.h55) & hd),
           "trend, no higher-TF": ((c > e100) & (c > D.h55), (c < e100) & (c < D.h55))}
    print(f"\n######## {lb}: {days:.0f} days, ATR14 median ${np.median(D.a[300:]):.1f} ########\n{T.HDR}")
    for fn, (b, s) in flt.items():
        for k in (0.75, 1.0, 1.25, 1.5):
            tr = []
            for i in np.flatnonzero((up & b) | (dn & s)):
                if i < 250 or i >= D.n - 3: continue
                d = 1 if up[i] else -1; tr.append((i, d, c[i] - d * k * D.a[i]))
            print(T.row(f"{fn[:34]:34} stop {k} ATR", T.evaluate(D, tr, "ema13", days)))
