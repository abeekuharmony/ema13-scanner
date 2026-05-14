import asyncio
import logging
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from scanner.config import settings
from scanner.exchanges import fetch_all_mexc, fetch_all_yfinance
from scanner.indicators import detect_signal, detect_mt_flip_signal, Signal
from scanner.alerts import send_alerts

logger = logging.getLogger(__name__)

# Deduplication: maps "signal_type:source:symbol" → "candle_ts:direction"
# Prevents sending the same signal multiple times per candle.
_last_alerted: dict[str, str] = {}


async def scan_job() -> None:
    """
    Main scan job — runs every 15 minutes (:00, :15, :30, :45).
    Each signal is only sent once per candle (deduped by candle_ts + direction).

    Per scan:
      1. Fetch 1H OHLCV for 30 MEXC perpetual futures symbols
      2. Fetch 1H OHLCV for 8 forex pairs via Yahoo Finance
      3. For each symbol:
           a. EMA Cross  — fires when EMA5 crosses EMA13 with EMA62 trend filter
           b. MT Flip    — fires when Megatrend changes colour (Green↔Red)
      4. Send Telegram alert for any new signals
    """
    start_time = datetime.now(timezone.utc)
    logger.info(f"Scan started at {start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    new_signals: list[Signal] = []

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
                # EMA cross
                sig = detect_signal(
                    df, symbol=sym, source="mexc",
                    fast=settings.ema_fast, mid=settings.ema_mid, slow=settings.ema_slow,
                    atr_len=settings.mt_atr_len, multiplier=settings.mt_multiplier,
                )
                if sig and _is_new(sig):
                    new_signals.append(sig)

                # Megatrend colour flip
                mt_sig = detect_mt_flip_signal(
                    df, symbol=sym, source="mexc",
                    atr_len=settings.mt_atr_len, multiplier=settings.mt_multiplier,
                )
                if mt_sig and _is_new(mt_sig):
                    new_signals.append(mt_sig)

            except Exception as e:
                logger.error(f"Error processing {sym}: {e}")

    except Exception as e:
        logger.error(f"MEXC scan failed: {e}")

    # ── Yahoo Finance Forex ───────────────────────────────────────────────
    try:
        logger.info(f"Scanning {len(settings.forex_symbols)} forex pairs via Yahoo Finance")
        yf_data = await fetch_all_yfinance(
            settings.forex_symbols,
            limit=settings.candle_limit,
        )

        for sym, df in yf_data.items():
            try:
                # EMA cross only — no Megatrend for yfinance
                # (yfinance prices differ from TradingView's OANDA feed;
                #  Supertrend on yfinance data would not match what you see on TradingView)
                sig = detect_signal(
                    df, symbol=sym, source="yfinance",
                    fast=settings.ema_fast, mid=settings.ema_mid, slow=settings.ema_slow,
                    atr_len=settings.mt_atr_len, multiplier=settings.mt_multiplier,
                )
                if sig and _is_new(sig):
                    new_signals.append(sig)

            except Exception as e:
                logger.error(f"Error processing {sym}: {e}")

    except Exception as e:
        logger.error(f"Yahoo Finance scan failed: {e}")

    # ── Send Alerts ───────────────────────────────────────────────────────
    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    logger.info(
        f"Scan complete in {elapsed:.1f}s. "
        f"{len(new_signals)} new signal(s)."
    )
    await send_alerts(new_signals)


def _is_new(sig: Signal) -> bool:
    """
    Return True if this signal has not been alerted yet for this candle.
    Keyed by signal_type + source + symbol.
    Fingerprint combines candle_ts + direction — fires again if direction flips
    on the same candle (e.g. Megatrend flips GREEN then RED within same hour).
    """
    key         = f"{sig.signal_type}:{sig.source}:{sig.symbol}"
    fingerprint = f"{sig.candle_ts}:{sig.direction}"
    if _last_alerted.get(key) == fingerprint:
        return False
    _last_alerted[key] = fingerprint
    return True


def create_scheduler() -> AsyncIOScheduler:
    """
    Create APScheduler instance that fires scan_job every 15 minutes.
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
