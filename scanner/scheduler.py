import asyncio
import logging
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from scanner.config import settings
from scanner.exchanges import (
    get_top_binance_futures_symbols,
    fetch_all_binance,
    fetch_all_twelvedata,
)
from scanner.indicators import calculate_ema, detect_crosses, CrossSignal
from scanner.alerts import send_alerts

logger = logging.getLogger(__name__)


async def scan_job() -> None:
    """
    Main scan job. Called every hour by the scheduler.
    1. Fetch top Binance USDS-M futures symbols
    2. Fetch OHLCV for all Binance symbols
    3. Fetch OHLCV for Twelve Data symbols (forex/commodities)
    4. Calculate EMA and detect crosses
    5. Send alerts for any crosses found
    """
    start_time = datetime.now(timezone.utc)
    logger.info(f"Scan started at {start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    all_signals: list[CrossSignal] = []

    # --- Binance Futures ---
    try:
        symbols = await get_top_binance_futures_symbols(settings.binance_top_n)
        logger.info(f"Scanning {len(symbols)} Binance Futures pairs")

        binance_data = await fetch_all_binance(
            symbols, interval="1h", limit=settings.candle_limit
        )

        for sym, df in binance_data.items():
            try:
                df = calculate_ema(df, period=settings.ema_period)
                signals = detect_crosses(df, symbol=sym, source="binance")
                all_signals.extend(signals)
            except Exception as e:
                logger.error(f"Error processing {sym}: {e}")

    except Exception as e:
        logger.error(f"Binance scan failed: {e}")

    # --- Twelve Data (Forex/Commodities) ---
    if settings.twelvedata_api_key:
        try:
            logger.info(
                f"Scanning {len(settings.twelvedata_symbols)} Twelve Data symbols"
            )
            td_data = await fetch_all_twelvedata(
                settings.twelvedata_symbols,
                interval="1h",
                outputsize=settings.candle_limit,
            )

            for sym, df in td_data.items():
                try:
                    df = calculate_ema(df, period=settings.ema_period)
                    signals = detect_crosses(df, symbol=sym, source="twelvedata")
                    all_signals.extend(signals)
                except Exception as e:
                    logger.error(f"Error processing {sym}: {e}")

        except Exception as e:
            logger.error(f"Twelve Data scan failed: {e}")
    else:
        logger.warning("Twelve Data API key not set, skipping forex/commodities")

    # --- Send Alerts ---
    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    logger.info(
        f"Scan complete in {elapsed:.1f}s. "
        f"{len(all_signals)} signal(s) found."
    )
    await send_alerts(all_signals)


def create_scheduler() -> AsyncIOScheduler:
    """
    Create APScheduler instance that runs scan_job at the configured
    minute past every hour (default: HH:01, giving 1-min buffer after
    candle close at HH:00).
    """
    scheduler = AsyncIOScheduler(timezone="UTC")

    trigger = CronTrigger(minute=settings.scan_interval_minutes, timezone="UTC")

    scheduler.add_job(
        scan_job,
        trigger=trigger,
        id="ema13_scan",
        name="EMA13 Cross Scan",
        max_instances=1,
        misfire_grace_time=300,
    )

    return scheduler
