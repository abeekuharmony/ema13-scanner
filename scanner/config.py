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
    # Need 62+ candles for EMA62 warmup plus ATR/Megatrend warmup
    candle_limit: int = 100

    # ── EMA Cloud periods ─────────────────────────────
    ema_fast: int = 5
    ema_mid: int = 13
    ema_slow: int = 62

    # ── Megatrend = Supertrend parameters ────────────
    mt_atr_len: int = 14
    mt_multiplier: float = 0.8

    # ── MEXC Perpetual Futures ───────────────────────────────────────
    # ACTIVE list — the ONLY symbols the scanner scans. Focused on 4 high-
    # volume assets used as data proxies for FOREX trading (data is only on
    # MEXC). To bring muted assets back, move them into this list from
    # all_mexc_symbols below. Nothing is deleted.
    mexc_symbols: list[str] = [
        "BTC_USDT",     # Bitcoin
        "ETH_USDT",     # Ethereum
        "XAUT_USDT",    # Tether Gold — proxy for XAU/USD (deep liquidity vs PAXG)
        "SILVER_USDT",  # Tokenised silver — proxy for XAG/USD
    ]

    # Full roster — currently MUTED (not scanned). Kept so any pair can be
    # restored later by moving it into mexc_symbols above.
    # (Includes PAXG_USDT as the thin-volume alternative gold proxy.)
    all_mexc_symbols: list[str] = [
        "BTC_USDT",    "ETH_USDT",    "SOL_USDT",    "XRP_USDT",    "DOGE_USDT",
        "BNB_USDT",    "ADA_USDT",    "AVAX_USDT",   "DOT_USDT",    "LINK_USDT",
        "POL_USDT",    "LTC_USDT",    "ATOM_USDT",   "NEAR_USDT",   "APT_USDT",
        "ARB_USDT",    "OP_USDT",     "SUI_USDT",    "ENA_USDT",    "INJ_USDT",
        "SEI_USDT",    "TIA_USDT",    "WIF_USDT",    "PEPE_USDT",   "S_USDT",
        "RENDER_USDT", "AAVE_USDT",   "SHIB_USDT",   "ORDI_USDT",   "JUP_USDT",
        "XAUT_USDT",   "PAXG_USDT",   "SILVER_USDT",
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
