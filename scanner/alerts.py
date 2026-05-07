import logging

import httpx

from scanner.config import settings
from scanner.indicators import Signal

logger = logging.getLogger(__name__)


def _fmt_symbol(symbol: str, source: str) -> str:
    """Convert internal symbol to a clean display name."""
    if source == "mexc":
        return symbol.replace("_", "/") + " (Perp)"
    return symbol  # Twelve Data symbols are already like EUR/USD


def _fmt_price(price: float) -> str:
    if price >= 100:
        return f"{price:,.2f}"
    elif price >= 1:
        return f"{price:.4f}"
    else:
        return f"{price:.6f}"


def format_signal(signal: Signal) -> str:
    """Format a single Signal into an HTML Telegram message block."""
    emoji   = "\U0001f7e2" if signal.direction == "BUY" else "\U0001f534"
    arrow   = "▲ BUY" if signal.direction == "BUY" else "▼ SELL"
    sym     = _fmt_symbol(signal.symbol, signal.source)
    close_s = _fmt_price(signal.close_price)
    e13_s   = _fmt_price(signal.ema13)
    label   = "[EMA Cross]" if signal.signal_type == "ema_cross" else "[Body Cross]"

    if signal.signal_type == "ema_cross":
        e5_s      = _fmt_price(signal.ema5)
        e62_s     = _fmt_price(signal.ema62)
        confirmed = (signal.direction == "BUY" and signal.mt_bull) or \
                    (signal.direction == "SELL" and not signal.mt_bull)
        mt_color  = "Bull (Green)" if signal.mt_bull else "Bear (Red)"
        mt_status = "confirmed ✓" if confirmed else "early entry"
        mt_label  = f"● {mt_color} — {mt_status}"
        return (
            f"{emoji} <b>{sym}</b>  {arrow}  {label}\n"
            f"    Close: {close_s}\n"
            f"    EMA5: {e5_s}  |  EMA13: {e13_s}  |  EMA62: {e62_s}\n"
            f"    Megatrend: {mt_label}"
        )

    # body_cross — simpler format, no EMA62/Megatrend conditions used
    return (
        f"{emoji} <b>{sym}</b>  {arrow}  {label}\n"
        f"    Close: {close_s}  |  EMA13: {e13_s}"
    )


def build_alert_message(signals: list[Signal]) -> str:
    if not signals:
        return ""

    buys  = [s for s in signals if s.direction == "BUY"]
    sells = [s for s in signals if s.direction == "SELL"]

    header = (
        "<b>5/13/62 Signal Alert</b>\n"
        f"<i>{len(signals)} signal(s) on 1H timeframe"
        f" — {len(buys)} BUY / {len(sells)} SELL</i>\n"
        + "─" * 25 + "\n\n"
    )

    # Show BUYs first, then SELLs
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
