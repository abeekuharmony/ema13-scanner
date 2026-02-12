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
    logger = logging.getLogger("ema13-scanner")

    logger.info("EMA13 Cross Scanner starting...")

    # Send startup notification
    await send_telegram_message(
        f"<b>EMA{settings.ema_period} Scanner</b> started.\n"
        f"Binance Futures: top {settings.binance_top_n} pairs\n"
        f"Twelve Data: {', '.join(settings.twelvedata_symbols)}\n"
        f"Schedule: every hour at :{settings.scan_interval_minutes:02d} UTC"
    )

    # Run one immediate scan on startup
    logger.info("Running initial scan...")
    await scan_job()

    # Set up scheduled scans
    scheduler = create_scheduler()
    scheduler.start()
    logger.info(
        f"Scheduler started. Scans at minute "
        f":{settings.scan_interval_minutes:02d} every hour."
    )

    # Keep running until interrupted
    stop_event = asyncio.Event()

    def handle_shutdown(*_):
        logger.info("Shutdown signal received...")
        stop_event.set()

    # Register signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_shutdown)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            signal.signal(sig, handle_shutdown)

    try:
        await stop_event.wait()
    finally:
        scheduler.shutdown(wait=False)
        await send_telegram_message(
            f"<b>EMA{settings.ema_period} Scanner</b> stopped."
        )
        logger.info("Scanner shut down.")


if __name__ == "__main__":
    asyncio.run(main())
