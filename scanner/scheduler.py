import asyncio
import logging
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from scanner.config import settings
from scanner.exchanges import fetch_all_mexc, fetch_all_twelvedata
from scanner.indicators import detect_signal, Signal
from scanner.alerts import send_alerts

logger = logging.getLogger(__name__)


async def scan_job() -> None:
    """
    Main scan job — runs at HH:01 UTC (one minute after each hourly candle close).
    1. Fetch 1H OHLCV for 30 MEXC perpetual futures symbols
    2. Fetch 1H OHLCV for 10 Twelve Data forex/commodity symbols
    3. Run 5/13/62 EMA Cloud + Megatrend signal detection on each
    4. Send Telegram alert for any signals found
    """
    start_time = datetime.now(timezone.utc)
    logger.info(f"Scan started at {start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    all_signals: list[Signal] = []

    # ── MEXC Perpetual Futures ────────────────────────────────────────────
    try:
        logger.info(f"Scanning {len(settings.mexc_symbols)} MEXC Futures pairs")
        mexc_data = await fetch_all_mexc(
            settings.mexc_symbols,
            interval="Hour1",
            limit=settings.candle_limit,
        )

        for sym, df in mexc_data.items():
            try:
                sig = detect_signal(
                    df, symbol=sym, source="mexc",
                    fast=settings.ema_fast, mid=settings.ema_mid, slow=settings.ema_slow,
                    atr_len=settings.mt_atr_len, smooth_len=settings.mt_smooth_len,
                    r_mult=settings.mt_r_mult, breakout_len=settings.mt_breakout_len,
                )
                if sig:
                    all_signals.append(sig)
            except Exception as e:
                logger.error(f"Error processing {sym}: {e}")

    except Exception as e:
        logger.error(f"MEXC scan failed: {e}")

    # ── Twelve Data (Forex / Commodities) ────────────────────────────────
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
                        atr_len=settings.mt_atr_len, smooth_len=settings.mt_smooth_len,
                        r_mult=settings.mt_r_mult, breakout_len=settings.mt_breakout_len,
                    )
                    if sig:
                        all_signals.append(sig)
                except Exception as e:
                    logger.error(f"Error processing {sym}: {e}")

        except Exception as e:
            logger.error(f"Twelve Data scan failed: {e}")
    else:
        logger.warning("TWELVEDATA_API_KEY not set — skipping forex/commodities")

    # ── Send Alerts ───────────────────────────────────────────────────────
    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    logger.info(
        f"Scan complete in {elapsed:.1f}s. "
        f"{len(all_signals)} signal(s) found."
    )
    await send_alerts(all_signals)


def create_scheduler() -> AsyncIOScheduler:
    """
    Create APScheduler instance that fires scan_job at HH:01 UTC every hour.
    The 1-minute offset gives the exchange APIs time to finalise the HH:00 candle.
    """
    scheduler = AsyncIOScheduler(timezone="UTC")

    scheduler.add_job(
        scan_job,
        trigger=CronTrigger(minute=settings.scan_interval_minutes, timezone="UTC"),
        id="513_62_scan",
        name="5/13/62 Signal Scan",
        max_instances=1,
        misfire_grace_time=300,
    )

    return scheduler
