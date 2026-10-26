"""
research_tma_realtime.py — can we tell IN REAL TIME which side of the FINAL (centred) TMA middle line a
signal candle will end up on?  XAUUSD M5 + H1, Exness data, user's trade rules, 0.25 spread.
Run with Exness MT5 open + logged in:   py research_tma_realtime.py

Signal base = v7.1 filters: EMA13 body cross + volume >= 1.2x avg(20 prev) + candle range < 75% of channel.
Side rule  : BUY only if the candle opened BELOW the TMA middle line, SELL only if ABOVE.
Approaches to decide the side without seeing the future:
  LIVE      : Brylator's live (right-edge) value, optionally with a safety margin (x ATR)
  DELAY N   : wait N candles, use the line value for the signal candle once N future candles exist,
              signal fires at candle i+N (entry pullback window starts after that)
  EXTRAP    : end-corrected estimate — future prices filled in by a linear trend of the last 13 prices
HINDSIGHT  = final centred line (the ceiling, NOT tradable).
"""
import numpy as np, pandas as pd, MetaTrader5 as mt5
import research_user_rules as R
from research_tma import evaluate

H = 12; MULT = 2.0; ATRP = 100

def tma_estimates(h, l, c):
    n = len(c); P = (h + l + 2 * c) / 4
    w_back = np.arange(H + 1, 0, -1).astype(float)            # 13..1 for P[t], P[t-1], ... P[t-12]
    S1 = w_back.sum()
    back = np.convolve(P, w_back)[:n]; back[:H] = np.nan
    w_f = np.arange(H, 0, -1).astype(float)                   # 12..1 for P[t+1] ... P[t+12]
    # est[m][t] = line value for bar t when m future candles are known
    est = {}
    cum_num = back.copy(); cum_den = np.full(n, S1)
    est[0] = cum_num / cum_den
    for m in range(1, H + 1):
        fut = np.r_[P[m:], np.full(m, np.nan)]
        cum_num = cum_num + w_f[m - 1] * fut
        cum_den = cum_den + w_f[m - 1]
        est[m] = cum_num / cum_den
    # linear-trend extrapolation of future prices from the last 13 prices
    x = np.arange(-H, 1).astype(float); xm = x.mean()
    ext = np.full(n, np.nan)
    for t in range(H, n):
        y = P[t - H:t + 1]; b = ((x - xm) * (y - y.mean())).sum() / ((x - xm) ** 2).sum()
        a = y.mean() - b * xm
        futp = a + b * np.arange(1, H + 1)
        ext[t] = (back[t] + (w_f * futp).sum()) / (2 * S1 - (H + 1))
    tr2 = np.maximum(h, np.r_[np.nan, c[:-1]]) - np.minimum(l, np.r_[np.nan, c[:-1]])
    atrx = pd.Series(tr2).rolling(ATRP).mean().shift(10).values
    return est, ext, 2 * MULT * atrx, atrx

def run(label, df):
    o, h, l, c = (df[k].values.astype(float) for k in ("open", "high", "low", "close")); n = len(c)
    v = df["tick_volume"].values.astype(float)
    days = (df["time"].iloc[-1] - df["time"].iloc[0]) / 86400
    e5, e13 = R.ema(c, 5), R.ema(c, 13); fh, fl = R.fractals(h, l)
    est, ext, width, atrx = tma_estimates(h, l, c)
    final = est[H]
    base = (o < e13) & (c > e13), (o > e13) & (c < e13)
    filt = ((h - l) < 0.75 * width) & (v >= 1.2 * pd.Series(v).rolling(20).mean().shift(1).values)
    up = base[0] & filt; dn = base[1] & filt
    valid = ~np.isnan(final) & ~np.isnan(est[0]) & ~np.isnan(atrx)

    # ---- 1. how much does the line move after the candle closes? (ATR units)
    print(f"\n===== XAUUSD {label} ({days:.0f} days) =====")
    print("Line revision |final - estimate| in ATR units (median / 80% / 95%):")
    for m in (0, 1, 2, 3, 4, 6, 9):
        r = np.abs(final - est[m])[valid] / atrx[valid]
        print(f"   after {m:2d} candles: {np.median(r):.2f} / {np.percentile(r, 80):.2f} / {np.percentile(r, 95):.2f}")
    r = np.abs(final - ext)[valid] / atrx[valid]
    print(f"   linear-trend estimate (0 candles): {np.median(r):.2f} / {np.percentile(r, 80):.2f} / {np.percentile(r, 95):.2f}")

    # ---- 2. hit rate: of the signals an approach accepts, how many are on the right side of the FINAL line?
    final_buy = up & (o < final); final_sell = dn & (o > final)
    n_final = (final_buy | final_sell)[valid].sum()
    def make(est_line, margin=0.0, delay=0):
        b = up & (o < est_line - margin * atrx); s = dn & (o > est_line + margin * atrx)
        sig = np.zeros(n, int)
        idx_b = np.where(b & valid)[0]; idx_s = np.where(s & valid)[0]
        for i in idx_b:
            if i + delay < n: sig[i + delay] = 1
        for i in idx_s:
            if i + delay < n: sig[i + delay] = -1
        acc = ((b & final_buy) | (s & final_sell))[valid].sum(); tot = (b | s)[valid].sum()
        return sig, tot, acc
    tests = [("HINDSIGHT final line (ceiling, not tradable)", final, 0, 0),
             ("LIVE, no margin (= v7.1)", est[0], 0, 0)]
    tests += [(f"LIVE, margin {m} ATR", est[0], m, 0) for m in (0.25, 0.5, 1.0)]
    tests += [(f"DELAY {d} candles", est[d], 0, d) for d in (2, 3, 4, 6)]
    tests += [("EXTRAP linear-trend estimate", ext, 0, 0), ("EXTRAP + margin 0.25 ATR", ext, 0.25, 0)]
    print(f"\nSignals the FINAL line would allow: {n_final}")
    print(f"{'approach':46} {'signals':>7} {'same side as final':>18}")
    built = []
    for name, line, m, d in tests:
        sig, tot, acc = make(line, m, d)
        print(f"{name:46} {tot:7d} {100 * acc / max(1, tot):17.1f}%")
        built.append((name, sig))

    # ---- 3. profit with the user's trade rules
    print(f"\n{'approach':46} {'------ one trade at a time ------':^37} | {'---- every signal ----':^24}")
    print(f"{'':46} {'n':>5} {'/day':>5} {'win%':>5} {'PF':>5} {'totR':>7} {'maxDD':>6} | {'n':>5} {'PF':>5} {'1st':>5} {'2nd':>5}")
    for name, sig in built:
        evaluate(f"{name:46}"[:46], sig, o, h, l, c, e5, fh, fl, n, days)

if __name__ == "__main__":
    if not mt5.initialize(path=R.EXNESS_PATH):
        raise SystemExit(f"MT5 init failed: {mt5.last_error()}")
    sym = next(s.name for s in mt5.symbols_get() if s.name.upper().startswith("XAUUSD"))
    mt5.symbol_select(sym, True)
    for label, (tf, count) in [("H1", R.TFS["H1"]), ("M5", R.TFS["M5"])]:
        run(label, pd.DataFrame(mt5.copy_rates_from_pos(sym, tf, 1, count)))
    mt5.shutdown()
