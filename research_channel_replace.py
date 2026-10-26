"""
research_channel_replace.py — find a REAL-TIME (non-repainting) channel to replace the TMA Centered Bands
in the user's system.  XAUUSD, Exness data, user's trade rules, 0.25 spread.
Run with Exness MT5 open + logged in:   py research_channel_replace.py

System: BUY  = bullish candle body crosses EMA13 AND most of the body is BELOW the channel middle line
               (body midpoint < mid).  SELL = bearish body cross AND body midpoint ABOVE the mid line.
Each candidate mid line uses ONLY past/current candles (what you would see live).
Reported per candidate:
  - closeness to the true (hindsight) TMA line: median distance in ATR, and % same-side verdict
  - PF with the user's trade rules (one trade at a time + every signal, split by year halves)
"""
import sys, numpy as np, pandas as pd, MetaTrader5 as mt5
import research_user_rules as R
from research_tma import trade, pf
from research_tma_realtime import tma_estimates, H

def wma(x, L):
    w = np.arange(1, L + 1).astype(float)
    out = np.convolve(x, w[::-1])[:len(x)] / w.sum(); out[:L - 1] = np.nan; return out
def sma(x, L): return pd.Series(x).rolling(L).mean().values
def ema(x, L): return pd.Series(x).ewm(span=L, adjust=False).mean().values
def hma(x, L): return wma(2 * wma(x, L // 2) - wma(x, L), int(np.sqrt(L)))
def zlema(x, L):
    lag = (L - 1) // 2; return ema(x + (x - np.r_[np.full(lag, np.nan), x[:-lag]]), L)
def alma(x, L, off=0.85, sig=6.0):
    m = off * (L - 1); s = L / sig; w = np.exp(-((np.arange(L) - m) ** 2) / (2 * s * s)); w /= w.sum()
    out = np.convolve(x, w[::-1])[:len(x)]; out[:L - 1] = np.nan; return out
def linreg_end(x, L):
    out = np.full(len(x), np.nan); t = np.arange(L) - (L - 1) / 2; tt = (t ** 2).sum()
    for i in range(L - 1, len(x)):
        y = x[i - L + 1:i + 1]; out[i] = y.mean() + ((t * (y - y.mean())).sum() / tt) * t[-1]
    return out
def supersmoother(x, L):
    a = np.exp(-1.414 * np.pi / L); b = 2 * a * np.cos(1.414 * np.pi / L)
    c2, c3 = b, -a * a; c1 = 1 - c2 - c3; f = np.copy(x)
    for i in range(2, len(x)): f[i] = c1 * (x[i] + x[i - 1]) / 2 + c2 * f[i - 1] + c3 * f[i - 2]
    return f
def donchian_mid(h, l, L): return (pd.Series(h).rolling(L).max().values + pd.Series(l).rolling(L).min().values) / 2
def nw_end(x, bw=8.0, win=50):
    w = np.exp(-(np.arange(win) ** 2) / (2 * bw * bw)); w /= w.sum()     # newest gets most weight, past only
    out = np.convolve(x, w)[:len(x)]; out[:win - 1] = np.nan; return out
def session_vwap(t, p, v):
    day = (t // 86400); out = np.empty(len(p)); pv = vv = 0.0; d0 = None
    for i in range(len(p)):
        if day[i] != d0: pv = vv = 0.0; d0 = day[i]
        pv += p[i] * v[i]; vv += v[i]; out[i] = pv / vv if vv else p[i]
    return out

def candidates(o, h, l, c, v, t):
    p = (h + l + 2 * c) / 4
    return {
        "Live TMA end (WMA13 hlcc4) [baseline]": wma(p, 13),
        "WMA 25 hlcc4 (same span as TMA)":       wma(p, 25),
        "Keltner mid EMA20":                     ema(c, 20),
        "Keltner mid EMA34":                     ema(c, 34),
        "Bollinger mid SMA20":                   sma(c, 20),
        "Donchian mid 20":                       donchian_mid(h, l, 20),
        "Donchian mid 34":                       donchian_mid(h, l, 34),
        "Lin-reg channel end 25":                linreg_end(p, 25),
        "Lin-reg channel end 50":                linreg_end(p, 50),
        "Hull MA 25":                            hma(p, 25),
        "ALMA 25":                               alma(p, 25),
        "Ehlers SuperSmoother 20":               supersmoother(p, 20),
        "Zero-lag EMA 25":                       zlema(p, 25),
        "Nadaraya-Watson end (bw 8)":            nw_end(p, 8.0),
        "Session VWAP (daily)":                  session_vwap(t, p, v),
    }

def perf(sig, o, h, l, c, e5, fh, fl, n):
    res = {}
    for i in range(150, n - R.WAIT - 2):
        if sig[i] != 0:
            tr = trade(i, sig[i], o, h, l, c, e5, fh, fl)
            if tr: res[i] = tr
    if not res: return "none"
    one = []; nxt = 0
    for i in sorted(res):
        if i >= nxt: one.append(res[i][0]); nxt = res[i][1]
    one = np.array(one); eq = np.cumsum(one); dd = (eq - np.maximum.accumulate(eq)).min()
    idx = np.array(sorted(res)); a = np.array([res[i][0] for i in idx]); f = idx < n // 2
    return (f"{len(one):5d} {100*(one>0).mean():5.1f} {pf(one):5.2f} {one.sum():+7.1f} {dd:+6.1f} | "
            f"{len(a):5d} {pf(a):5.2f} {pf(a[f]):5.2f} {pf(a[~f]):5.2f}")

def run(label, df, names=None):
    o, h, l, c = (df[k].values.astype(float) for k in ("open", "high", "low", "close")); n = len(c)
    v = df["tick_volume"].values.astype(float); t = df["time"].values
    days = (t[-1] - t[0]) / 86400
    e5, e13 = R.ema(c, 5), R.ema(c, 13); fh, fl = R.fractals(h, l)
    est, _, _, atrx = tma_estimates(h, l, c); final = est[H]
    bm = (o + c) / 2
    up = (o < e13) & (c > e13); dn = (o > e13) & (c < e13)
    ok = (up | dn) & ~np.isnan(final) & ~np.isnan(atrx)
    truth = np.where(up, bm < final, bm > final)
    cands = candidates(o, h, l, c, v, t)
    if names: cands = {k: cands[k] for k in names if k in cands}
    print(f"\n===== XAUUSD {label} ({days:.0f} days) — system: EMA13 body cross + body midpoint on the right side of the mid line =====")
    print(f"{'mid line':40} {'dist to true TMA':>16} {'same verdict':>12} | {'---- one trade at a time ----':^31} | {'---- every signal ----':^24}")
    print(f"{'':40} {'(median ATR)':>16} {'as true TMA':>12} | {'n':>5} {'win%':>5} {'PF':>5} {'totR':>7} {'maxDD':>6} | {'n':>5} {'PF':>5} {'1st':>5} {'2nd':>5}")
    rows = [("TRUE TMA (hindsight, not tradable)", final)] + list(cands.items())
    for name, mid in rows:
        good = ok & ~np.isnan(mid)
        dist = np.median(np.abs(mid - final)[good] / atrx[good])
        vt = np.where(up, bm < mid, bm > mid)
        same = (vt[good] == truth[good]).mean()
        sig = np.where(up & vt & ~np.isnan(mid), 1, np.where(dn & vt & ~np.isnan(mid), -1, 0))
        print(f"{name:40} {dist:16.2f} {100 * same:11.1f}% | " + perf(sig, o, h, l, c, e5, fh, fl, n), flush=True)

if __name__ == "__main__":
    tfs = sys.argv[1:] or ["H1", "M5"]
    if not mt5.initialize(path=R.EXNESS_PATH):
        raise SystemExit(f"MT5 init failed: {mt5.last_error()}")
    sym = next(s.name for s in mt5.symbols_get() if s.name.upper().startswith("XAUUSD"))
    mt5.symbol_select(sym, True)
    for label in tfs:
        run(label, pd.DataFrame(mt5.copy_rates_from_pos(sym, R.TFS[label][0], 1, R.TFS[label][1])))
    mt5.shutdown()
