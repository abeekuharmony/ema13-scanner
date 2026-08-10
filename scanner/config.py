from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # ── OANDA v20 data feed (real forex/metals/indices — matches TV chart) ──
    # Add OANDA_API_TOKEN to the environment (Railway variables). Never commit it.
    oanda_api_token: str = ""
    oanda_env: str = "practice"   # "practice" (demo token) or "live"

    # Scanner behavior
    scan_interval_minutes: int = 15
    log_level: str = "INFO"
    # 200 bars — only the EMA13 is needed now, but extra warmup is harmless.
    candle_limit: int = 200

    # Timeframes scanned each run: (OANDA granularity, display label).
    # 15m removed (too noisy) — 1H only. Re-add ("M15","15m") to restore.
    scan_timeframes: list[tuple[str, str]] = [("H1", "1H")]

    # Per-instrument BEST trading sessions (UTC hour windows [start, end), by
    # the candle's OPEN hour). From the 12-month session backtest (TEST A).
    # Accra/GMT = UTC+0, so these hours are the user's local time.
    #   Asia 0-7 | London 7-12 | London/NY overlap 12-16 | NY-PM 16-21 | off 21-24
    session_windows: dict[str, list[tuple[int, int]]] = {
        "XAU_USD":    [(0, 7), (12, 16)],   # gold: Asia + overlap (PF 1.76 / 1.64)
        "BTC_USD":    [(0, 16)],            # Asia+London+overlap all >1.5
        "ETH_USD":    [(7, 16)],            # London + overlap (PF 1.80 / 1.90)
        "XAG_USD":    [(7, 16)],            # silver: London + overlap (PF 2.60 / 3.69)
        "NAS100_USD": [(12, 16)],           # US100: overlap only (PF 2.71)
        "SPX500_USD": [(7, 16)],            # US500: London + overlap (PF 1.70 / 1.59)
        "WTICO_USD":  [(0, 7), (12, 21)],   # USOIL: Asia + overlap + NY-PM
    }

    # Instruments that alert in ALL hours: signals inside session_windows are
    # normal; signals OUTSIDE (dead hours) still fire but are flagged
    # "low-probability — trade cautiously". Others stay muted outside windows.
    session_alert_all: list[str] = ["XAU_USD"]  # gold: alert 24h, tag dead hours

    # ── EMA period (only the EMA13 is used for the body-cross signal) ──
    ema_fast: int = 5
    ema_mid: int = 13
    ema_slow: int = 62

    # ── ACTIVE instruments: OANDA instrument code → display name ──────
    # Real forex/metals/indices feed — matches the user's TradingView (OANDA)
    # chart and is ~$1-2 from Exness (vs the old MEXC token, which was ~$18 off).
    oanda_symbols: dict[str, str] = {
        "XAU_USD":    "XAU/USD",   # gold
        "XAG_USD":    "XAG/USD",   # silver
        "NAS100_USD": "US100",     # Nasdaq 100
        "SPX500_USD": "US500",     # S&P 500
        "WTICO_USD":  "USOIL",     # WTI crude oil
        "BTC_USD":    "BTC/USD",   # Bitcoin (OANDA CFD; region-dependent)
        "ETH_USD":    "ETH/USD",   # Ethereum (OANDA CFD; region-dependent)
    }

    # Extra OANDA instruments available to add later (move into oanda_symbols):
    #   EUR_USD, GBP_USD, USD_JPY, AUD_USD, USD_CAD, NZD_USD, USD_CHF (FX majors),
    #   BCO_USD (Brent), NATGAS_USD, DE30_EUR, UK100_GBP, JP225_USD, US30_USD


settings = Settings()
