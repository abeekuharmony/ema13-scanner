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

    # Scanner behavior
    scan_interval_minutes: int = 15
    log_level: str = "INFO"
    # 200 bars — only the EMA13 is needed now, but extra warmup is harmless.
    candle_limit: int = 200

    # Timeframes scanned each run: (MEXC interval code, display label).
    # 15m is checked every scan; 1H only changes on the hour (dedup handles it).
    scan_timeframes: list[tuple[str, str]] = [("Min15", "15m"), ("Min60", "1H")]

    # ── EMA period (only the EMA13 is used for the body-cross signal) ──
    ema_fast: int = 5
    ema_mid: int = 13
    ema_slow: int = 62

    # ── MEXC Perpetual Futures ───────────────────────────────────────
    # ACTIVE list — the ONLY symbols the scanner scans. High-volume assets
    # the user trades on Exness (data proxied via MEXC). To bring muted
    # assets back, move them into this list from all_mexc_symbols below.
    mexc_symbols: list[str] = [
        "BTC_USDT",     # Bitcoin
        "ETH_USDT",     # Ethereum
        "XAUT_USDT",    # Tether Gold — proxy for XAU/USD
        "SILVER_USDT",  # Tokenised silver — proxy for XAG/USD
        "NAS100_USDT",  # Nasdaq 100 — proxy for US100
        "SPX500_USDT",  # S&P 500 — proxy for US500
        "USOIL_USDT",   # WTI crude oil — proxy for USOIL
    ]

    # Full roster — currently MUTED (not scanned). Kept so any pair can be
    # restored later by moving it into mexc_symbols above.
    all_mexc_symbols: list[str] = [
        "BTC_USDT",    "ETH_USDT",    "SOL_USDT",    "XRP_USDT",    "DOGE_USDT",
        "BNB_USDT",    "ADA_USDT",    "AVAX_USDT",   "DOT_USDT",    "LINK_USDT",
        "POL_USDT",    "LTC_USDT",    "ATOM_USDT",   "NEAR_USDT",   "APT_USDT",
        "ARB_USDT",    "OP_USDT",     "SUI_USDT",    "ENA_USDT",    "INJ_USDT",
        "SEI_USDT",    "TIA_USDT",    "WIF_USDT",    "PEPE_USDT",   "S_USDT",
        "RENDER_USDT", "AAVE_USDT",   "SHIB_USDT",   "ORDI_USDT",   "JUP_USDT",
        "XAUT_USDT",   "PAXG_USDT",   "SILVER_USDT",
        # Indices / commodities (proxies for Exness CFDs):
        "NAS100_USDT", "SPX500_USDT", "USOIL_USDT",  "US30_USDT",
        "NGAS_USDT",   "JP225_USDT",  "UKOIL_USDT",
    ]

    # ── Forex pairs (Yahoo Finance) — currently MUTED ────────────────
    # Empty = not scanned. Full list preserved in all_forex_symbols; restore
    # by copying entries back into forex_symbols.
    forex_symbols: list[str] = []
    all_forex_symbols: list[str] = [
        "EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD",
        "AUD/USD", "USD/CAD", "NZD/USD", "USD/CHF",
    ]


settings = Settings()
