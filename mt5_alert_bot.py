"""
mt5_alert_bot.py — EMA13 body-cross alerts on REAL Exness MT5 data.

Runs 24/7 on a Windows VPS with MetaTrader 5 open and logged into Exness.
One signal: a candle whose BODY closes across the EMA13 on the last CLOSED
1H candle (open one side, close the other; wicks ignored). Each pair only
alerts during its best sessions (from the 12-month Exness backtest); gold
alerts 24h with dead-hour signals flagged low-probability.

Setup (PowerShell on the VPS):
    py -m pip install MetaTrader5 pandas numpy python-dotenv requests
Create a file named ".env" next to this script containing:
    TELEGRAM_BOT_TOKEN=your_token_here
    TELEGRAM_CHAT_ID=your_chat_id_here
Run:
    py mt5_alert_bot.py
"""
import os
import sys
import time
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np  # noqa: F401 (kept for parity / future use)
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

# ── Config ────────────────────────────────────────────────────────────
EXNESS_PATH        = r"C:\Program Files\MetaTrader 5\terminal64.exe"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
SCAN_EVERY_MIN     = 15
EMA_MID            = 13
CANDLES            = 200
DEDUP_FILE         = Path(__file__).with_name("dedup_state.json")

# Exness symbol -> display name (12 validated pairs)
SYMBOLS = {
    "XAUUSDm": "XAU/USD", "XAGUSDm": "XAG/USD",
    "BTCUSDm": "BTC/USD", "ETHUSDm": "ETH/USD", "SOLUSDm": "SOL/USD",
    "XRPUSDm": "XRP/USD", "DOGEUSDm": "DOGE/USD", "LINKUSDm": "LINK/USD",
    "BNBUSDm": "BNB/USD",
    "USTECm": "US100", "US500m": "US500", "USOILm": "USOIL",
}

# Best-session windows (UTC hour [start, end) on the candle's open hour),
# from the 12-month Exness backtest. UTC = user's Accra local time.
SESSION_WINDOWS = {
    "XAUUSDm":  [(0, 21)],
    "XAGUSDm":  [(0, 21)],
    "BTCUSDm":  [(0, 7), (12, 16)],
    "ETHUSDm":  [(7, 21)],
    "SOLUSDm":  [(0, 7), (12, 24)],
    "XRPUSDm":  [(7, 21)],
    "DOGEUSDm": [(0, 16), (21, 24)],
    "LINKUSDm": [(7, 21)],
    "BNBUSDm":  [(0, 7), (12, 24)],
    "USTECm":   [(12, 16)],
    "US500m":   [(12, 21)],
    "USOILm":   [(12, 21)],
}
# Symbols that alert in ALL hours; out-of-window signals flagged low-probability
ALERT_ALL = ["XAUUSDm"]


def log(msg):
    print("%s  %s" % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), msg), flush=True)


def in_window(sym, hour):
    w = SESSION_WINDOWS.get(sym)
    if not w:
        return True
    return any(a <= hour < b for a, b in w)


def session_name(hour):
    if 0 <= hour < 7:   return "Asia"
    if 7 <= hour < 12:  return "London"
    if 12 <= hour < 16: return "London/NY"
    if 16 <= hour < 21: return "NY-PM"
    return "Off-hours"


def load_dedup():
    if DEDUP_FILE.exists():
        try:
            return json.loads(DEDUP_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_dedup(d):
    # keep it from growing forever
    if len(d) > 3000:
        d = dict(list(d.items())[-1500:])
    try:
        DEDUP_FILE.write_text(json.dumps(d), encoding="utf-8")
    except Exception as e:
        log("dedup save failed: %s" % e)
    return d


def fmt(v):
    if v >= 100:
        return "{:,.2f}".format(v)
    if v >= 1:
        return "{:.4f}".format(v)
    return "{:.6f}".format(v)


def send_telegram(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log("Telegram not configured — alert below:\n" + text)
        return
    try:
        r = requests.post(
            "https://api.telegram.org/bot%s/sendMessage" % TELEGRAM_BOT_TOKEN,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML",
                  "disable_web_page_preview": True},
            timeout=30,
        )
        if not r.ok or not r.json().get("ok"):
            log("Telegram API error: %s" % r.text)
    except Exception as e:
        log("Telegram send failed: %s" % e)


def connect():
    if not mt5.initialize(path=EXNESS_PATH):
        log("MT5 initialize failed: %s" % (mt5.last_error(),))
        return False
    acc = mt5.account_info()
    if acc:
        log("Connected -> %s | %s" % (acc.company, acc.server))
    return True


def detect_offset():
    for s in SYMBOLS:
        mt5.symbol_select(s, True)
        tk = mt5.symbol_info_tick(s)
        if tk and tk.time:
            return round((tk.time - time.time()) / 3600)
    return 0


def check_symbol(sym, offset, dedup):
    mt5.symbol_select(sym, True)
    rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H1, 0, CANDLES)
    if rates is None or len(rates) < EMA_MID + 3:
        return None
    df = pd.DataFrame(rates)
    server_now = time.time() + offset * 3600
    df = df[df["time"] + 3600 <= server_now]            # closed candles only
    if len(df) < EMA_MID + 2:
        return None
    df["e13"] = df["close"].ewm(span=EMA_MID, adjust=False).mean()
    cur = df.iloc[-1]
    o, c, e13 = float(cur["open"]), float(cur["close"]), float(cur["e13"])
    if o < e13 and c > e13:
        direction = "BUY"
    elif o > e13 and c < e13:
        direction = "SELL"
    else:
        return None

    open_utc = int(cur["time"]) - offset * 3600         # candle open in UTC
    hour = (open_utc // 3600) % 24
    in_prime = in_window(sym, hour)
    if not in_prime and sym not in ALERT_ALL:
        return None

    candle_day = datetime.fromtimestamp(open_utc, tz=timezone.utc).strftime("%Y-%m-%d")
    key = "%s:%s %02d:00" % (sym, candle_day, hour)
    if dedup.get(key) == direction:
        return None
    dedup[key] = direction

    disp  = SYMBOLS.get(sym, sym)
    emoji = "\U0001F7E2" if direction == "BUY" else "\U0001F534"
    arrow = "\u25B2 BUY" if direction == "BUY" else "\u25BC SELL"
    move  = "opened below, closed above" if direction == "BUY" else "opened above, closed below"
    ch    = (hour + 1) % 24
    warn  = "\n    \u26A0\uFE0F LOW-PROBABILITY (dead hours) \u2014 trade cautiously" if not in_prime else ""
    return (
        "%s <b>%s</b>  %s  [EMA13 Body Cross \u00B7 1H \u00B7 %s]\n"
        "    Candle %s the EMA13\n"
        "    Signal candle (CLOSED): %02d:00\u2192%02d:00 UTC, %s\n"
        "    Open: %s  \u2192  Close: %s\n"
        "    EMA13: %s%s"
    ) % (emoji, disp, arrow, session_name(hour), move, hour, ch, candle_day,
         fmt(o), fmt(c), fmt(e13), warn)


def main():
    log("MT5 EMA13 body-cross alert bot starting (%d symbols)" % len(SYMBOLS))
    if not connect():
        return
    offset = detect_offset()
    log("Server offset vs UTC: %+d h" % offset)
    dedup = load_dedup()
    while True:
        now = time.time()
        step = SCAN_EVERY_MIN * 60
        nxt = (now // step + 1) * step + 20          # next :00/:15/:30/:45 + 20s
        time.sleep(max(5, nxt - now))
        if not mt5.terminal_info():
            log("MT5 disconnected — reconnecting")
            connect()
        alerts = []
        for sym in SYMBOLS:
            try:
                m = check_symbol(sym, offset, dedup)
                if m:
                    alerts.append(m)
            except Exception as e:
                log("error on %s: %s" % (sym, e))
        if alerts:
            dedup = save_dedup(dedup)
            header = ("<b>EMA13 Body-Cross Alert</b>\n<i>%d signal(s) on 1H</i>\n" % len(alerts)) + "\u2500" * 25 + "\n\n"
            send_telegram(header + "\n\n".join(alerts))
            log("Sent %d alert(s)" % len(alerts))
        else:
            log("scan complete - no new signals")


if __name__ == "__main__":
    main()
