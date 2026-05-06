import asyncio
import logging
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from scanner.config import settings
from scanner.exchanges import fetch_all_mexc, fetch_all_twelvedata
from scanner.indicators import detect_signal, detect_body_cross_signal, Signal
from scanner.alerts import send_alerts

logger = logging.getLogger(__name__)

# Deduplication: maps "source:symbol" → candle_ts of the last alerted signal.
# Prevents sending the same 1H signal up to 4× when scanning every 15 minutes.
_last_alerted: dict[str, str] = {}


async def scan_job() -> None:
    """
    Main scan job — runs every 15 minutes.
    Each 1H signal is only sent once per closed candle (deduped by candle_ts).
    1. Fetch 1H OHLCV for 30 MEXC perpetual futures symbols
    2. Fetch 1H OHLCV for 8 Twelve Data forex symbols
    3. Run 5/13/62 EMA Cloud + Megatrend signal detection on each
    4. Send Telegram alert for new signals only
    """
    start_time = datetime.now(timezone.utc)
    logger.info(f"Scan started at {start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    new_signals: list[Signal] = []

    # Body cross only runs at the top of the hour (:00 scan).
    # minute < 3 allows for small scheduler/network delays.
    top_of_hour = start_time.minute < 3

    # ── MEXC Perpetual Futures ────────────────────────────────────────────
    try:
        logger.info(f"Scanning {len(settings.mexc_symbols)} MEXC Futures pairs")
        mexc_data = await fetch_all_mexc(
            settings.mexc_symbols,
            interval="Min60",
            limit=settings.candle_limit,
        )

        for sym, df in mexc_data.items():
            try:
                sig = detect_signal(
                    df, symbol=sym, source="mexc",
                    fast=settings.ema_fast, mid=settings.ema_mid, slow=settings.ema_slow,
                    atr_len=settings.mt_atr_len, multiplier=settings.mt_multiplier,
                )
                if sig and _is_new(sig):
                    new_signals.append(sig)

                if top_of_hour:
                    body_sig = detect_body_cross_signal(
                        df, symbol=sym, source="mexc", mid=settings.ema_mid,
                    )
                    if body_sig and _is_new(body_sig):
                        new_signals.append(body_sig)
            except Exception as e:
                logger.error(f"Error processing {sym}: {e}")

    except Exception as e:
        logger.error(f"MEXC scan failed: {e}")

    # ── Twelve Data (Forex) ───────────────────────────────────────────────
    if settings.twelvedata_api_key:
        try:
            logger.info(f"Scanning {len(settings.twelvedata_symbols)} Twelve Data symbols")
            td_data = await fetch_all_twelvedata(
                settings.twelvedata_symbols,
                interval="1h",
                outputsize=settings.candle_limit,
            )

            for sym, df in td_data.items():
                try:
                    sig = detect_signal(
                        df, symbol=sym, source="twelvedata",
                        fast=settings.ema_fast, mid=settings.ema_mid, slow=settings.ema_slow,
                        atr_len=settings.mt_atr_len, multiplier=settings.mt_multiplier,
                    )
                    if sig and _is_new(sig):
                        new_signals.append(sig)

                    if top_of_hour:
                        body_sig = detect_body_cross_signal(
                            df, symbol=sym, source="twelvedata", mid=settings.ema_mid,
                        )
                        if body_sig and _is_new(body_sig):
                            new_signals.append(body_sig)
                except Exception as e:
                    logger.error(f"Error processing {sym}: {e}")

        except Exception as e:
            logger.error(f"Twelve Data scan failed: {e}")
    else:
        logger.warning("TWELVEDATA_API_KEY not set — skipping forex")

    # ── Send Alerts ───────────────────────────────────────────────────────
    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    logger.info(
        f"Scan complete in {elapsed:.1f}s. "
        f"{len(new_signals)} new signal(s)."
    )
    await send_alerts(new_signals)


def _is_new(sig: Signal) -> bool:
    """
    Return True if this is a new alert.
    Key includes signal_type so EMA Cross and Body Cross track independently.
    Silent when: same candle AND same direction.
    Fires when: new candle OR direction reversed on same candle.
    """
    key = f"{sig.signal_type}:{sig.source}:{sig.symbol}"
    fingerprint = f"{sig.candle_ts}:{sig.direction}"
    if _last_alerted.get(key) == fingerprint:
        return False
    _last_alerted[key] = fingerprint
    return True


def create_scheduler() -> AsyncIOScheduler:
    """
    Create APScheduler instance that fires scan_job every 15 minutes.
    Scanning sub-hourly catches signals as soon as the candle closes.
    Deduplication in scan_job ensures each 1H signal is only sent once.
    """
    scheduler = AsyncIOScheduler(timezone="UTC")

    scheduler.add_job(
        scan_job,
        trigger=CronTrigger(minute="*/15", timezone="UTC"),
        id="513_62_scan",
        name="5/13/62 Signal Scan",
        max_instances=1,
        misfire_grace_time=300,
    )

    return scheduler
