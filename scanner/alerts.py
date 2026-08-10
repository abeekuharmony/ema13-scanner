import logging

import httpx

from scanner.config import settings
from scanner.indicators import Signal

logger = logging.getLogger(__name__)


def _fmt_symbol(symbol: str, source: str) -> str:
    """Convert internal symbol to a clean display name."""
    if source == "mexc":
        return symbol.replace("_", "/") + " (Perp)"
    return symbol  # yfinance symbols are already like EUR/USD


def _fmt_price(price: float) -> str:
    if price >= 100:
        return f"{price:,.2f}"
    elif price >= 1:
        return f"{price:.4f}"
    else:
        return f"{price:.6f}"


def format_signal(signal: Signal) -> str:
    """Format a single Signal into an HTML Telegram message block."""
    sym     = _fmt_symbol(signal.symbol, signal.source)
    close_s = _fmt_price(signal.close_price)

    if signal.signal_type == "ema_cross":
        e5_s  = _fmt_price(signal.ema5)
        e13_s = _fmt_price(signal.ema13)
        e62_s = _fmt_price(signal.ema62)

        # Confluence: both the EMA cross AND the Megatrend agree on direction.
        # Megatrend is only reliable for MEXC (same data feed as the scanner);
        # yfinance/forex prices differ from TradingView's OANDA feed, so no MT.
        confirmed = signal.source == "mexc" and (
            (signal.direction == "BUY"  and signal.mt_bull) or
            (signal.direction == "SELL" and not signal.mt_bull)
        )

        if confirmed:
            # Both SELL (or BUY) confirmations aligned → STRONG signal
            emoji = "🟢🟢" if signal.direction == "BUY" else "🔴🔴"
            label = "▲ STRONG BUY" if signal.direction == "BUY" else "▼ STRONG SELL"
            align = (
                "Megatrend GREEN + EMA5 crossed ↑ EMA13 (EMA62 below)"
                if signal.direction == "BUY" else
                "Megatrend RED + EMA5 crossed ↓ EMA13 (EMA62 above)"
            )
            mt_color = "Bull (Green)" if signal.mt_bull else "Bear (Red)"
            return (
                f"{emoji} <b>{sym}</b>  {label}  [EMA Cross + Megatrend]\n"
                f"    ✅ Both confirmations aligned: {align}\n"
                f"    Close: {close_s}\n"
                f"    EMA5: {e5_s}  |  EMA13: {e13_s}  |  EMA62: {e62_s}\n"
                f"    Megatrend: ● {mt_color} — confirmed ✓"
            )

        # Single confirmation — EMA cross only (Megatrend not yet aligned)
        emoji = "\U0001f7e2" if signal.direction == "BUY" else "\U0001f534"
        arrow = "▲ BUY" if signal.direction == "BUY" else "▼ SELL"
        base  = (
            f"{emoji} <b>{sym}</b>  {arrow}  [EMA Cross]\n"
            f"    Close: {close_s}\n"
            f"    EMA5: {e5_s}  |  EMA13: {e13_s}  |  EMA62: {e62_s}"
        )
        if signal.source == "mexc":
            mt_color = "Bull (Green)" if signal.mt_bull else "Bear (Red)"
            base += f"\n    Megatrend: ● {mt_color} — early entry"
        return base

    if signal.signal_type == "mt_flip":
        emoji      = "\U0001f7e2" if signal.direction == "GREEN" else "\U0001f534"
        color_text = "turned GREEN 🟢" if signal.direction == "GREEN" else "turned RED 🔴"
        return (
            f"{emoji} <b>{sym}</b>  [Megatrend Flip]\n"
            f"    Megatrend {color_text}\n"
            f"    Close: {close_s}"
        )

    if signal.signal_type in ("retest_setup", "retest_trigger", "retest_cancel"):
        is_buy  = signal.direction == "BUY"
        arrow   = "▲ BUY" if is_buy else "▼ SELL"
        entry_s = _fmt_price(signal.entry)
        stop_s  = _fmt_price(signal.stop)
        tp1_s   = _fmt_price(signal.tp1)
        tp2_s   = _fmt_price(signal.tp2)
        atr_s   = _fmt_price(signal.atr)

        if signal.signal_type == "retest_setup":
            emoji = "\U0001f7e2" if is_buy else "\U0001f534"
            side  = "above" if is_buy else "below"
            return (
                f"{emoji}📋 <b>{sym}</b>  {arrow}  [EMA5 Retest SETUP]\n"
                f"    Decisive body close {side} EMA13; trend aligned (EMA62 + Megatrend line)\n"
                f"    Watch for shallow pullback to EMA5 ≈ {entry_s}\n"
                f"    Plan: limit {entry_s}  |  Stop {stop_s} (1×ATR = {atr_s})\n"
                f"    TP: {tp1_s} (1.5R)  /  {tp2_s} (2R)\n"
                f"    <i>Level drifts with the EMA5 — TRIGGER fires on the touch. Void if a body closes back across EMA13.</i>"
            )

        if signal.signal_type == "retest_trigger":
            emoji = "\U0001f3af"
            return (
                f"{emoji} <b>{sym}</b>  {arrow}  [EMA5 Retest TRIGGERED]\n"
                f"    Shallow pullback tested the EMA5 — entry zone\n"
                f"    Entry: {entry_s}  |  Stop: {stop_s} (1×ATR = {atr_s})\n"
                f"    TP: {tp1_s} (1.5R)  /  {tp2_s} (2R)\n"
                f"    Close now: {close_s}"
            )

        return (
            f"❌ <b>{sym}</b>  {arrow}  [EMA5 Retest CANCELLED]\n"
            f"    Body closed back across the EMA13 before the EMA5 retest\n"
            f"    Close: {close_s}"
        )

    if signal.signal_type == "ema13_body_cross":
        emoji  = "\U0001f7e2" if signal.direction == "BUY" else "\U0001f534"
        arrow  = "▲ BUY" if signal.direction == "BUY" else "▼ SELL"
        move   = "opened below, closed above" if signal.direction == "BUY" else "opened above, closed below"
        open_s = _fmt_price(signal.open_price)
        e13_s  = _fmt_price(signal.ema13)
        tf     = f" · {signal.timeframe}" if signal.timeframe else ""
        sess   = f" · {signal.session}" if signal.session else ""
        # Exact candle this signal refers to — the candle that just CLOSED,
        # not the one forming now. Shown in UTC to end which-candle confusion.
        cand = ""
        if signal.candle_ts:
            d  = signal.candle_ts[:10]
            hh = signal.candle_ts[11:13]
            try:
                ch = f"{(int(hh) + 1) % 24:02d}"
            except ValueError:
                ch = "??"
            cand = f"\n    Signal candle (CLOSED): {hh}:00→{ch}:00 UTC, {d}"
        warn = "\n    ⚠️ LOW-PROBABILITY (dead hours) — trade cautiously" if signal.low_prob else ""
        return (
            f"{emoji} <b>{sym}</b>  {arrow}  [EMA13 Body Cross{tf}{sess}]\n"
            f"    Candle {move} the EMA13"
            f"{cand}\n"
            f"    Open: {open_s}  →  Close: {close_s}\n"
            f"    EMA13: {e13_s}"
            f"{warn}"
        )

    return f"<b>{sym}</b> — unknown signal type"


def build_alert_message(signals: list[Signal]) -> str:
    """One signal type now: EMA13 body cross on 15m / 1H."""
    if not signals:
        return ""

    buys  = [s for s in signals if s.direction == "BUY"]
    sells = [s for s in signals if s.direction == "SELL"]

    header = (
        "<b>EMA13 Body-Cross Alert</b>\n"
        f"<i>{len(buys)} BUY / {len(sells)} SELL — closed candle across EMA13</i>\n"
        + "─" * 25 + "\n\n"
    )
    ordered = buys + sells
    body = "\n\n".join(format_signal(s) for s in ordered)
    return header + body


async def send_telegram_message(text: str) -> bool:
    """Send a message via Telegram Bot API. Splits if > 4096 chars."""
    if not text or not settings.telegram_bot_token:
        return False

    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
    chunks = _split_message(text, max_length=4096)

    async with httpx.AsyncClient(timeout=30.0) as client:
        for chunk in chunks:
            payload = {
                "chat_id":                  settings.telegram_chat_id,
                "text":                     chunk,
                "parse_mode":               "HTML",
                "disable_web_page_preview": True,
            }
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                result = resp.json()
                if not result.get("ok"):
                    logger.error(f"Telegram API error: {result}")
                    return False
            except Exception as e:
                logger.error(f"Failed to send Telegram message: {e}")
                return False

    return True


def _split_message(text: str, max_length: int = 4096) -> list[str]:
    if len(text) <= max_length:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break
        split_pos = text.rfind("\n", 0, max_length)
        if split_pos == -1:
            split_pos = max_length
        chunks.append(text[:split_pos])
        text = text[split_pos:].lstrip("\n")
    return chunks


async def send_alerts(signals: list[Signal]) -> None:
    if not signals:
        logger.info("No signals detected this scan.")
        return

    message = build_alert_message(signals)
    success = await send_telegram_message(message)
    if success:
        logger.info(f"Sent {len(signals)} signal(s) to Telegram.")
    else:
        logger.error("Failed to deliver alerts to Telegram.")
