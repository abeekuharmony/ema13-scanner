import asyncio
import logging
import signal
import sys

from scanner.config import settings
from scanner.scheduler import create_scheduler, scan_job
from scanner.alerts import send_telegram_message


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


async def main() -> None:
    setup_logging()
    logger = logging.getLogger("5-13-62-scanner")

    logger.info("5/13/62 Signal Scanner starting...")

    await send_telegram_message(
        "<b>5/13/62 Signal Scanner</b> started.\n"
        f"MEXC Futures: {len(settings.mexc_symbols)} pairs\n"
        f"Forex (Yahoo Finance): {len(settings.forex_symbols)} pairs\n"
        f"Strategy: EMA5/13/62 Cloud + Megatrend (ATR Breakout)\n"
        f"Timeframe: 1H  |  Schedule: every 15 min (:00, :15, :30, :45 UTC)"
    )

    logger.info("Running initial scan...")
    await scan_job()

    scheduler = create_scheduler()
    scheduler.start()
    logger.info(
        f"Scheduler started. Scans at :{settings.scan_interval_minutes:02d} every hour UTC."
    )

    stop_event = asyncio.Event()

    def handle_shutdown(*_):
        logger.info("Shutdown signal received...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_shutdown)
        except NotImplementedError:
            signal.signal(sig, handle_shutdown)

    try:
        await stop_event.wait()
    finally:
        scheduler.shutdown(wait=False)
        await send_telegram_message("<b>5/13/62 Signal Scanner</b> stopped.")
        logger.info("Scanner shut down.")


if __name__ == "__main__":
    asyncio.run(main())
