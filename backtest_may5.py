"""
Backtest: 5/13/62 EMA Cloud + Megatrend
Date: 2026-05-05 (all 24 hourly bars, 00:00 - 23:00 UTC)
Symbols: 30 MEXC perps + 8 Twelve Data forex
"""
import asyncio
from datetime import datetime, timezone, timedelta
from scanner.config import settings
from scanner.exchanges import fetch_all_mexc, fetch_all_twelvedata
from scanner.indicators import detect_signal


MAY5 = datetime(2026, 5, 5, tzinfo=timezone.utc)
HOURS = 24


async def main():
    print("=" * 58)
    print("  BACKTEST: 5/13/62 + Megatrend  |  2026-05-05  |  1H")
    print("=" * 58)

    # Fetch MEXC (150h back covers May 5th + 100+ candle warmup)
    print("\nFetching MEXC perpetuals data...")
    mexc_data = await fetch_all_mexc(
        settings.mexc_symbols, interval="Min60", limit=150
    )
    print(f"  Got data for {len(mexc_data)}/{len(settings.mexc_symbols)} symbols")

    # Fetch Twelve Data forex
    print("Fetching Twelve Data forex (rate-limited, ~65s)...")
    td_data = await fetch_all_twelvedata(
        settings.twelvedata_symbols, interval="1h", outputsize=150
    )
    print(f"  Got data for {len(td_data)}/{len(settings.twelvedata_symbols)} symbols")

    # Make Twelve Data timestamps timezone-aware
    for sym, df in td_data.items():
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        td_data[sym] = df

    all_data = (
        [(sym, df, "mexc")       for sym, df in mexc_data.items()] +
        [(sym, df, "twelvedata") for sym, df in td_data.items()]
    )

    print(f"\nReplay: {len(all_data)} symbols x 24 hours\n")
    print(f"{'Time (UTC)':<12} {'Dir':<6} {'Symbol':<24} {'Close':>12}")
    print("-" * 58)

    all_signals = []

    for h in range(HOURS):
        scan_time = MAY5 + timedelta(hours=h)
        for sym, df, source in all_data:
            df_slice = df[df["timestamp"] < scan_time].copy()
            if len(df_slice) < 2:
                continue
            try:
                sig = detect_signal(
                    df_slice, symbol=sym, source=source,
                    fast=settings.ema_fast, mid=settings.ema_mid,
                    slow=settings.ema_slow, atr_len=settings.mt_atr_len,
                    multiplier=settings.mt_multiplier,
                )
                if sig:
                    label = sym.replace("_", "/") + (" (Perp)" if source == "mexc" else "")
                    price = (
                        f"{sig.close_price:,.2f}"
                        if sig.close_price >= 1
                        else f"{sig.close_price:.6f}"
                    )
                    direction = "BUY " if sig.direction == "BUY" else "SELL"
                    print(f"{scan_time.strftime('%H:%M UTC'):<12} {direction:<6} {label:<24} {price:>12}")
                    all_signals.append((scan_time, sig))
            except Exception as e:
                print(f"  Error on {sym}: {e}")

    print("-" * 58)
    buys  = sum(1 for _, s in all_signals if s.direction == "BUY")
    sells = sum(1 for _, s in all_signals if s.direction == "SELL")
    print(f"\nTotal signals on May 5th 2026: {len(all_signals)}")
    print(f"  BUY : {buys}")
    print(f"  SELL: {sells}")
    print("=" * 58)


if __name__ == "__main__":
    asyncio.run(main())
