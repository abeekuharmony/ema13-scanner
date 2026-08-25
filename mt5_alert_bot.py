"""
mt5_alert_bot.py — FULL-STACK entry alerts on real Exness MT5 data.

Signal: session window -> EMA13 body cross -> recent same-direction Megatrend
(JMA slope) flip -> EMA5 pullback touch = entry. Swing-low/high stop, 3R target.
Two tiers: LIVE (validated pairs) and FORWARD-TEST (demo-only, tagged).
Optional news flag when a high-impact event is near (needs FMP_API_KEY).

Setup (PowerShell on the VPS):
    py -m pip install MetaTrader5 pandas numpy python-dotenv requests
.env next to this script:
    TELEGRAM_BOT_TOKEN=...
    TELEGRAM_CHAT_ID=...
    FMP_API_KEY=...        (optional — for the news flag)
Run:  py mt5_alert_bot.py
"""
import os
import sys
import time
import json
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np  # noqa: F401
import pandas as pd
import requests
import MetaTrader5 as mt5

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

EXNESS_PATH        = r"C:\Program Files\MetaTrader 5\terminal64.exe"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
FMP_API_KEY        = os.getenv("FMP_API_KEY", "")
SCAN_EVERY_MIN     = 15
CANDLES            = 260
K_RETEST           = 12          # bars allowed for the pullback
NEAR_FLIP          = 4           # (legacy) unused since colour is a state, not a flip
MT_MULT            = 1.5         # Megatrend ATR-band multiplier (colour stickiness; tune to chart)
EXIT_ALERTS        = True        # send a "colour flipped, consider exiting" alert on open trades
SWING_LB           = 10
NEWS_WINDOW_H      = 3.0         # flag events within +/- this many hours
DEDUP_FILE         = Path(__file__).with_name("dedup_state.json")
OPEN_FILE          = Path(__file__).with_name("open_trades.json")

# Show the "RISKY (against EMA62)" caution ONLY on pairs where the 1yr Exness
# test showed EMA62-alignment IMPROVED PF. On the others (BTC/DOGE/SOL/BNB/
# indices) counter-trend trades actually win, so the warning is silenced.
EMA62_WARN = {"XAUUSD", "XAGUSD", "ETHUSD", "XRPUSD", "LINKUSD", "ADAUSD", "HBARUSD"}

# base name -> (display, session windows UTC or None=all-hours)
LIVE = {
    "XAUUSD": ("XAU/USD", [(0, 21)]),
    "XAGUSD": ("XAG/USD", [(0, 21)]),
    "BTCUSD": ("BTC/USD", [(0, 7), (12, 16)]),
    "ETHUSD": ("ETH/USD", [(7, 21)]),
    "SOLUSD": ("SOL/USD", [(0, 7), (12, 24)]),
    "XRPUSD": ("XRP/USD", [(7, 21)]),
    "DOGEUSD": ("DOGE/USD", [(0, 16), (21, 24)]),
    "LINKUSD": ("LINK/USD", [(7, 21)]),
    "BNBUSD": ("BNB/USD", [(0, 7), (12, 24)]),
    "ADAUSD": ("ADA/USD", None),     # promoted from demo (1yr Exness PF 1.52)
    "HBARUSD": ("HBAR/USD", None),   # promoted from demo (1yr Exness PF 1.58)
}
DEMO = {
    "USTEC":  ("US100", None), "US500": ("US500", None),
    "XLMUSD": ("XLM/USD", None),
    "ALGOUSD": ("ALGO/USD", None),
    "XMRUSD": ("XMR/USD", None), "SLXUSD": ("SLX/USD", None),
}


def log(m):
    print("%s  %s" % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), m), flush=True)


def resolve(names, base):
    for n in names:
        if n.upper() == base:
            return n
    hits = [n for n in names if base in n.upper()]
    return sorted(set(hits), key=len)[0] if hits else None


def jma(src, length=14, phase=100, power=1):
    pr = 2.5
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
    a = beta ** power
    n = len(src); e0 = np.zeros(n); e1 = np.zeros(n); e2 = np.zeros(n); o = np.zeros(n)
    e0[0] = src[0]; o[0] = src[0]
    for i in range(1, n):
        e0[i] = (1 - a) * src[i] + a * e0[i - 1]
        e1[i] = (src[i] - e0[i]) * (1 - beta) + beta * e1[i - 1]
        e2[i] = (e0[i] + pr * e1[i] - o[i - 1]) * (1 - a) ** 2 + a ** 2 * e2[i - 1]
        o[i] = e2[i] + o[i - 1]
    return o


def in_window(win, hour):
    if not win:
        return True
    return any(a <= hour < b for a, b in win)


def session_name(hour):
    if 0 <= hour < 7:   return "Asia"
    if 7 <= hour < 12:  return "London"
    if 12 <= hour < 16: return "London/NY"
    if 16 <= hour < 21: return "NY-PM"
    return "Off-hours"


def fmt(v):
    if v >= 100:
        return "{:,.2f}".format(v)
    if v >= 1:
        return "{:.4f}".format(v)
    return "{:.6f}".format(v)


def load_dedup():
    if DEDUP_FILE.exists():
        try:
            return json.loads(DEDUP_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_dedup(d):
    if len(d) > 4000:
        d = dict(list(d.items())[-2000:])
    try:
        DEDUP_FILE.write_text(json.dumps(d), encoding="utf-8")
    except Exception as e:
        log("dedup save failed: %s" % e)
    return d


def load_open():
    if OPEN_FILE.exists():
        try:
            return json.loads(OPEN_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_open(o):
    try:
        OPEN_FILE.write_text(json.dumps(o[-500:]), encoding="utf-8")
    except Exception as e:
        log("open-trades save failed: %s" % e)


def send_telegram(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log("Telegram not configured — alert below:\n" + text); return
    try:
        r = requests.post("https://api.telegram.org/bot%s/sendMessage" % TELEGRAM_BOT_TOKEN,
                          json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML",
                                "disable_web_page_preview": True}, timeout=30)
        if not r.ok or not r.json().get("ok"):
            log("Telegram API error: %s" % r.text)
    except Exception as e:
        log("Telegram send failed: %s" % e)


_news_cache = {"ts": 0, "events": []}
def upcoming_news():
    """High-impact events within +/- NEWS_WINDOW_H hours. Optional (needs FMP key)."""
    if not FMP_API_KEY:
        return None
    now = time.time()
    if now - _news_cache["ts"] > 1800:                       # refresh every 30 min
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            tmr = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
            r = requests.get("https://financialmodelingprep.com/api/v3/economic_calendar",
                             params={"from": today, "to": tmr, "apikey": FMP_API_KEY}, timeout=20)
            ev = []
            for e in r.json():
                if str(e.get("impact", "")).lower() == "high":
                    try:
                        t = datetime.fromisoformat(e["date"].replace("Z", "+00:00"))
                        if t.tzinfo is None:
                            t = t.replace(tzinfo=timezone.utc)
                        ev.append((t.timestamp(), "%s %s" % (e.get("country", ""), e.get("event", ""))))
                    except Exception:
                        pass
            _news_cache["events"] = ev; _news_cache["ts"] = now
        except Exception as e:
            log("news fetch failed: %s" % e)
    for ts, name in _news_cache["events"]:
        if abs(ts - time.time()) <= NEWS_WINDOW_H * 3600:
            return name
    return None


def connect():
    if not mt5.initialize(path=EXNESS_PATH):
        log("MT5 init failed: %s" % (mt5.last_error(),)); return False
    acc = mt5.account_info()
    if acc:
        log("Connected -> %s | %s" % (acc.company, acc.server))
    return True


def detect_offset(symbols):
    """Server-vs-UTC hour offset, taken as the value MOST symbols agree on.
    A single symbol can be mid session-break (stale tick, e.g. gold ~21-22 UTC)
    and read 1h off; crypto trades 24/7, so the modal offset is the true one."""
    now = time.time()
    offs = []
    for sym in symbols:
        mt5.symbol_select(sym, True)
        tk = mt5.symbol_info_tick(sym)
        if tk and tk.time:
            offs.append(round((tk.time - now) / 3600))
    if not offs:
        return 0
    return Counter(offs).most_common(1)[0][0]


def check(sym, disp, win, is_demo, offset, dedup, news, opens, warn_ema62):
    mt5.symbol_select(sym, True)
    rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H1, 0, CANDLES)
    if rates is None or len(rates) < 80:
        return None, []
    df = pd.DataFrame(rates)
    server_now = time.time() + offset * 3600
    df = df[df["time"] + 3600 <= server_now].reset_index(drop=True)   # closed candles only
    n = len(df)
    if n < 75:
        return None, []
    o, h, l, c = (df[x].values for x in ["open", "high", "low", "close"])
    e5 = df["close"].ewm(span=5, adjust=False).mean().values
    e13 = df["close"].ewm(span=13, adjust=False).mean().values
    e62 = df["close"].ewm(span=62, adjust=False).mean().values
    tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(com=13, adjust=False).mean().values
    jm = jma(((df["high"] + df["low"] + df["close"]) / 3).values)
    # Megatrend colour via an ATR band on the JMA (Supertrend-style, sticky).
    # The old raw-slope colour flipped on trivial JMA wiggles (~0.1pt) and fired
    # false counter-trend signals; a band ignores wiggles small vs ATR, matching
    # the chart's steady colour that holds green/red through shallow pullbacks.
    ub = jm + MT_MULT * atr
    lb = jm - MT_MULT * atr
    fub = ub.copy(); flb = lb.copy(); bull = np.ones(n, bool)
    for i in range(1, n):
        if np.isnan(atr[i]):
            bull[i] = bull[i - 1]; fub[i] = fub[i - 1]; flb[i] = flb[i - 1]; continue
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        if c[i] > fub[i - 1]:
            bull[i] = True
        elif c[i] < flb[i - 1]:
            bull[i] = False
        else:
            bull[i] = bull[i - 1]
    utc = (df["time"].values - offset * 3600)
    hr = ((utc // 3600) % 24).astype(int)
    j = n - 1

    def qualifying_cross(t):
        if np.isnan(atr[t]) or atr[t] <= 0:
            return None
        if not in_window(win, int(hr[t])):
            return None
        if o[t] < e13[t] and c[t] > e13[t] and bull[t]:
            return "BUY"
        if o[t] > e13[t] and c[t] < e13[t] and not bull[t]:
            return "SELL"
        return None

    def build(kind, d, level):
        sign = 1 if d == "BUY" else -1
        sw = min(l[j - SWING_LB:j + 1]) if d == "BUY" else max(h[j - SWING_LB:j + 1])
        stop = sw - 0.1 * atr[j] if d == "BUY" else sw + 0.1 * atr[j]
        R = abs(level - stop)
        tp = level + sign * R * 3.0
        hh = int(hr[j]); ch = (hh + 1) % 24
        # RISKY = trade is against the higher-timeframe trend (EMA62). Not blocked —
        # flagged so the user can choose to skip. Backtests: counter-EMA62 trades
        # are ~half the losers.
        aligned = (((level > e62[j]) if d == "BUY" else (level < e62[j])) and
                   ((e62[j] > e62[j - 10]) if d == "BUY" else (e62[j] < e62[j - 10])))
        return {"kind": kind, "dir": d, "sym": sym, "disp": disp, "entry": level,
                "stop": stop, "R": R, "tp": tp, "close": c[j], "hour": hh, "ch": ch,
                "risky": (not aligned) and warn_ema62,
                "date": datetime.fromtimestamp(int(utc[j]), tz=timezone.utc).strftime("%Y-%m-%d"),
                "session": session_name(hh), "demo": is_demo, "news": news,
                "cts": datetime.fromtimestamp(int(utc[j]), tz=timezone.utc).strftime("%Y-%m-%d %H:00")}

    # ---- manage open trades for this symbol: exit when the Megatrend colour flips ----
    exits = []
    kept = []
    for tr_ in opens:
        if tr_.get("sym") != sym:
            kept.append(tr_); continue
        idx = np.where(utc == tr_["ts"])[0]
        if len(idx) == 0:
            if tr_["ts"] >= int(utc[0]):        # entry still ahead of window start -> keep
                kept.append(tr_)
            continue                             # else fell out of the 260-bar window -> drop
        ei = int(idx[0]); lng_ = tr_["dir"] == "BUY"; closed = False; flip_at_j = False
        for b in range(ei + 1, n):
            hit = ((l[b] <= tr_["stop"]) if lng_ else (h[b] >= tr_["stop"])) or \
                  ((h[b] >= tr_["tp"]) if lng_ else (l[b] <= tr_["tp"]))
            flipped = (not bull[b]) if lng_ else bull[b]
            if hit:                              # stop/tp reached first -> closed, no exit alert
                closed = True; break
            if flipped:                          # colour turned against the trade
                closed = True; flip_at_j = (b == j); break
        if not closed:
            kept.append(tr_); continue
        if flip_at_j:
            hh = int(hr[j])
            exits.append({"kind": "EXIT", "dir": tr_["dir"], "disp": disp, "demo": is_demo,
                          "close": c[j], "hour": hh, "ch": (hh + 1) % 24,
                          "date": datetime.fromtimestamp(int(utc[j]), tz=timezone.utc).strftime("%Y-%m-%d"),
                          "cts": datetime.fromtimestamp(int(utc[j]), tz=timezone.utc).strftime("%Y-%m-%d %H:00")})
        # closed by stop/tp or an earlier flip -> just drop (no alert)
    opens[:] = kept

    # ---- entry signal on the just-closed bar ----
    sig = None
    d = qualifying_cross(j)
    if d:
        sig = build("SETUP", d, e5[j])
    else:
        i = None
        for t in range(j - 1, max(j - 1 - K_RETEST, 74), -1):
            if qualifying_cross(t):
                i = t; d = qualifying_cross(t); break
        if i is not None:
            lng = d == "BUY"
            for t in range(i + 1, j + 1):
                colour_ok = bull[t] if lng else (not bull[t])
                touched = (l[t] <= e5[t]) if lng else (h[t] >= e5[t])
                crossed_back = (c[t] < e13[t]) if lng else (c[t] > e13[t])
                if not colour_ok:
                    if t == j: sig = build("CANCEL", d, e5[j])
                    break
                if touched:
                    if t == j: sig = build("TRIGGER", d, e5[j])
                    break
                if crossed_back:
                    if t == j: sig = build("CANCEL", d, e5[j])
                    break
    # register a new open position on a TRIGGER (once per candle)
    if sig is not None and sig["kind"] == "TRIGGER":
        if not any(x.get("sym") == sym and x.get("ts") == int(utc[j]) for x in opens):
            opens.append({"sym": sym, "dir": sig["dir"], "stop": sig["stop"],
                          "tp": sig["tp"], "ts": int(utc[j]), "disp": disp, "demo": is_demo})
    return sig, exits


def format_alert(a):
    is_buy = a["dir"] == "BUY"
    arrow = "\u25B2 BUY" if is_buy else "\u25BC SELL"
    demo = "\n    \u26A0\uFE0F FORWARD-TEST \u2014 demo account only" if a["demo"] else ""
    if a["kind"] == "EXIT":
        return ("\U0001F6AA <b>%s</b>  %s  [Megatrend FLIP \u2014 consider EXIT]%s\n"
                "    Colour flipped against your %s \u2014 the trend may be over\n"
                "    Close: %s") % (a["disp"], arrow, demo, a["dir"], fmt(a["close"]))
    news = ("\n    \U0001F4F0 News: %s \u2014 trade with caution" % a["news"]) if a["news"] else ""
    risky = ("\n    \u26A0\uFE0F RISKY \u2014 against the higher-trend (EMA62); trade with caution or skip"
             if a.get("risky") else "")
    e, s, t, R = fmt(a["entry"]), fmt(a["stop"]), fmt(a["tp"]), fmt(a["R"])
    cand = "Signal candle (CLOSED): %02d:00\u2192%02d:00 UTC, %s" % (a["hour"], a["ch"], a["date"])
    if a["kind"] == "SETUP":
        emoji = "\U0001F7E2" if is_buy else "\U0001F534"
        return ("%s\U0001F4CB <b>%s</b>  %s  [Full-stack SETUP \u00B7 1H \u00B7 %s]%s%s\n"
                "    Cross + Megatrend colour aligned \u2014 watch for EMA5 pullback\n"
                "    Place limit \u2248 %s  |  Stop %s (swing, R=%s)\n"
                "    TP (3R): %s\n    %s%s") % (emoji, a["disp"], arrow, a["session"], demo, risky, e, s, R, t, cand, news)
    if a["kind"] == "TRIGGER":
        return ("\U0001F3AF <b>%s</b>  %s  [Full-stack ENTRY \u00B7 1H]%s%s\n"
                "    EMA5 pullback tapped \u2014 entry zone\n"
                "    Entry %s  |  Stop %s (R=%s)  |  TP(3R) %s\n    %s%s") % (
                a["disp"], arrow, demo, risky, e, s, R, t, cand, news)
    return ("\u274C <b>%s</b>  %s  [Full-stack VOID]%s\n"
            "    Body closed back across EMA13 before the pullback\n    Close: %s") % (
            a["disp"], arrow, demo, fmt(a["close"]))


def main():
    log("Full-stack MT5 alert bot starting")
    if not connect():
        return
    names = [x.name for x in mt5.symbols_get()]
    pairs = []
    for base, (disp, win) in LIVE.items():
        r = resolve(names, base)
        if r: pairs.append((r, disp, win, False, base in EMA62_WARN))
        else: log("LIVE symbol not found: %s" % base)
    for base, (disp, win) in DEMO.items():
        r = resolve(names, base)
        if r: pairs.append((r, disp, win, True, base in EMA62_WARN))
        else: log("demo symbol not on this account (skipped): %s" % base)
    log("Scanning %d symbols (%d live, %d demo)" % (
        len(pairs), sum(1 for p in pairs if not p[3]), sum(1 for p in pairs if p[3])))
    offset = detect_offset([p[0] for p in pairs]) if pairs else 0
    log("Server offset vs UTC: %+d h" % offset)
    dedup = load_dedup()
    opens = load_open()
    while True:
        now = time.time(); step = SCAN_EVERY_MIN * 60
        time.sleep(max(5, (now // step + 1) * step + 20 - now))
        if not mt5.terminal_info():
            log("MT5 disconnected — reconnecting"); connect()
        news = upcoming_news()
        alerts = []
        for sym, disp, win, is_demo, warn in pairs:
            try:
                a, ex = check(sym, disp, win, is_demo, offset, dedup, news, opens, warn)
                for x in ex:
                    if not EXIT_ALERTS:
                        continue
                    key = "EXIT:%s:%s" % (sym, x["cts"])
                    if dedup.get(key) == x["dir"]:
                        continue
                    dedup[key] = x["dir"]; alerts.append(x)
                if a:
                    key = "%s:%s:%s" % (a["kind"], sym, a["cts"])
                    if dedup.get(key) != a["dir"]:
                        dedup[key] = a["dir"]; alerts.append(a)
            except Exception as e:
                log("error on %s: %s" % (sym, e))
        save_open(opens)
        if alerts:
            dedup = save_dedup(dedup)
            # exits first, then triggers, setups, cancels last
            order = {"EXIT": 0, "TRIGGER": 1, "SETUP": 2, "CANCEL": 3}
            alerts.sort(key=lambda a: order.get(a["kind"], 3))
            header = ("<b>Full-stack Alert</b>\n<i>%d signal(s) on 1H</i>\n" % len(alerts)) + "\u2500" * 25 + "\n\n"
            send_telegram(header + "\n\n".join(format_alert(a) for a in alerts))
            log("Sent %d alert(s)" % len(alerts))
        else:
            log("scan complete - no new signals")


if __name__ == "__main__":
    main()
