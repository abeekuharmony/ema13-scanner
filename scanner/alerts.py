import logging

import httpx

from scanner.config import settings
from scanner.indicators import CrossSignal, CrossType

logger = logging.getLogger(__name__)

CROSS_LABELS = {
    CrossType.BODY_CROSS_UP: "BULLISH BODY CROSS",
    CrossType.BODY_CROSS_DOWN: "BEARISH BODY CROSS",
    CrossType.WICK_CROSS_UP: "BULLISH WICK PIERCE",
    CrossType.WICK_CROSS_DOWN: "BEARISH WICK PIERCE",
}

CROSS_EMOJI = {
    CrossType.BODY_CROSS_UP: "\U0001f7e2",   # green circle
    CrossType.BODY_CROSS_DOWN: "\U0001f534",  # red circle
    CrossType.WICK_CROSS_UP: "\U0001f7e1",    # yellow circle
    CrossType.WICK_CROSS_DOWN: "\U0001f7e0",  # orange circle
}

DIRECTION_ARROW = {
    CrossType.BODY_CROSS_UP: "\u2191",    # up arrow
    CrossType.BODY_CROSS_DOWN: "\u2193",  # down arrow
    CrossType.WICK_CROSS_UP: "\u2191",
    CrossType.WICK_CROSS_DOWN: "\u2193",
}


def format_signal(signal: CrossSignal) -> str:
    """Format a single CrossSignal into an HTML message block."""
    emoji = CROSS_EMOJI.get(signal.cross_type, "")
    label = CROSS_LABELS.get(signal.cross_type, signal.cross_type.value)
    arrow = DIRECTION_ARROW.get(signal.cross_type, "")
    source_tag = signal.source.upper()

    # Format price with appropriate decimal places
    if signal.price_close >= 100:
        price_fmt = f"{signal.price_close:,.2f}"
        ema_fmt = f"{signal.ema_value:,.2f}"
    else:
        price_fmt = f"{signal.price_close:.4f}"
        ema_fmt = f"{signal.ema_value:.4f}"

    return (
        f"{emoji} <b>{signal.symbol}</b> [{source_tag}] {arrow}\n"
        f"    {label}\n"
        f"    Close: {price_fmt} | EMA{settings.ema_period}: {ema_fmt}"
    )


def build_alert_message(signals: list[CrossSignal]) -> str:
    """Build the full Telegram alert message from a list of signals."""
    if not signals:
        return ""

    header = (
        f"<b>EMA{settings.ema_period} Cross Alert</b>\n"
        f"<i>{len(signals)} signal(s) on 1H timeframe</i>\n"
        + "\u2500" * 25
        + "\n\n"
    )
    body = "\n\n".join(format_signal(s) for s in signals)
    return header + body


async def send_telegram_message(text: str) -> bool:
    """
    Send a message via Telegram Bot HTTP API.
    Handles messages > 4096 chars by splitting.
    Returns True on success.
    """
    if not text or not settings.telegram_bot_token:
        return False

    url = (
        f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
    )
    chunks = _split_message(text, max_length=4096)

    async with httpx.AsyncClient(timeout=30.0) as client:
        for chunk in chunks:
            payload = {
                "chat_id": settings.telegram_chat_id,
                "text": chunk,
                "parse_mode": "HTML",
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
    """Split a long message into chunks that fit Telegram's limit."""
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


async def send_alerts(signals: list[CrossSignal]) -> None:
    """High-level: format signals and send via Telegram."""
    if not signals:
        logger.info("No EMA cross signals detected this scan.")
        return

    message = build_alert_message(signals)
    success = await send_telegram_message(message)
    if success:
        logger.info(f"Sent {len(signals)} signal(s) to Telegram.")
    else:
        logger.error("Failed to deliver alerts to Telegram.")
