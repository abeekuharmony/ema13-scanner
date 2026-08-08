import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from scanner.config import settings
from scanner.exchanges import fetch_all_mexc
from scanner.indicators import detect_ema13_body_cross, Signal
from scanner.alerts import send_alerts

logger = logging.getLogger(__name__)

# Deduplication: maps "signal_type:source:symbol" → "candle_ts:direction"
# Persisted to disk so Railway process restarts don't re-send already-alerted signals.
_DEDUP_FILE = Path("dedup_state.json")
_last_alerted: dict[str, str] = {}


def _load_dedup() -> None:
    global _last_alerted
    if _DEDUP_FILE.exists():
        try:
            _last_alerted = json.loads(_DEDUP_FILE.read_text(encoding="utf-8"))
            logger.info(f"Loaded {len(_last_alerted)} dedup entries from {_DEDUP_FILE}")
        except Exception as exc:
            logger.warning(f"Could not load dedup state: {exc} — starting fresh")
            _last_alerted = {}


def _save_dedup() -> None:
    try:
        _DEDUP_FILE.write_text(json.dumps(_last_alerted), encoding="utf-8")
    except Exception as exc:
        logger.warning(f"Could not save dedup state: {exc}")


def _in_session(symbol: str, hour: int) -> bool:
    """True if the UTC hour is within one of the pair's best-session windows.
    Symbols with no window defined are allowed at all hours."""
    windows = settings.session_windows.get(symbol)
    if not windows:
        return True
    return any(start <= hour < end for start, end in windows)


def _session_name(hour: int) -> str:
    if 0 <= hour < 7:   return "Asia"
    if 7 <= hour < 12:  return "London"
    if 12 <= hour < 16: return "London/NY"
    if 16 <= hour < 21: return "NY-PM"
    return "Off-hours"


async def scan_job() -> None:
    """
    Main scan job — runs every 15 minutes (:00, :15, :30, :45).

    ONE signal, on TWO timeframes (15m + 1H):
      A candle whose BODY closed across the EMA13 on the last CLOSED candle
        - opened BELOW EMA13, closed ABOVE it → BUY
        - opened ABOVE EMA13, closed BELOW it → SELL
      Wicks that poke through but close back are ignored (open vs close only).

    Each scan checks the just-closed 15m candle; the 1H candle only changes on
    the hour, so dedup (per symbol + timeframe + candle_ts) prevents repeats.
    The forming candle is dropped in the data layer, so signals never repaint.
    """
    start_time = datetime.now(timezone.utc)
    logger.info(f"Scan started at {start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    new_signals: list[Signal] = []

    for interval, label in settings.scan_timeframes:
        try:
            logger.info(f"Scanning {len(settings.mexc_symbols)} pairs on {label}")
            mexc_data = await fetch_all_mexc(
                settings.mexc_symbols, interval=interval, limit=settings.candle_limit,
            )
            for sym, df in mexc_data.items():
                try:
                    sig = detect_ema13_body_cross(
                        df, symbol=sym, source="mexc",
                        fast=settings.ema_fast, mid=settings.ema_mid, slow=settings.ema_slow,
                    )
                    if sig:
                        # Session filter: only alert if the signal candle's
                        # (UTC = local) open hour is in this pair's best window.
                        hour = int(df["timestamp"].iloc[-1].hour)
                        if not _in_session(sym, hour):
                            continue
                        sig.timeframe = label
                        sig.session   = _session_name(hour)
                        if _is_new(sig):
                            new_signals.append(sig)
                except Exception as e:
                    logger.error(f"Error processing {sym} [{label}]: {e}")
        except Exception as e:
            logger.error(f"MEXC scan failed on {label}: {e}")

    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    logger.info(f"Scan complete in {elapsed:.1f}s. {len(new_signals)} new signal(s).")
    await send_alerts(new_signals)


def _is_new(sig: Signal) -> bool:
    """
    Return True if this signal has not been alerted yet for this candle.
    Keyed by signal_type + source + timeframe + symbol, so the same symbol can
    alert independently on 15m and 1H.
    Fingerprint combines candle_ts + direction — a new candle re-alerts.
    State is persisted to disk so process restarts don't re-alert old signals.
    """
    key         = f"{sig.signal_type}:{sig.source}:{sig.timeframe}:{sig.symbol}"
    fingerprint = f"{sig.candle_ts}:{sig.direction}"
    if _last_alerted.get(key) == fingerprint:
        return False
    _last_alerted[key] = fingerprint
    _save_dedup()
    return True


def create_scheduler() -> AsyncIOScheduler:
    """
    Create APScheduler instance that fires scan_job every 15 minutes.
    Loads persisted dedup state so restarts don't re-send already-alerted signals.
    """
    _load_dedup()
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
