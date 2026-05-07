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
    mt_multiplier: float = 2.5

    # ── MEXC Perpetual Futures (30 crypto pairs) ─────
    mexc_symbols: list[str] = [
        "BTC_USDT",    "ETH_USDT",    "SOL_USDT",    "XRP_USDT",    "DOGE_USDT",
        "BNB_USDT",    "ADA_USDT",    "AVAX_USDT",   "DOT_USDT",    "LINK_USDT",
        "POL_USDT",    "LTC_USDT",    "ATOM_USDT",   "NEAR_USDT",   "APT_USDT",
        "ARB_USDT",    "OP_USDT",     "SUI_USDT",    "ENA_USDT",    "INJ_USDT",
        "SEI_USDT",    "TIA_USDT",    "WIF_USDT",    "PEPE_USDT",   "S_USDT",
        "RENDER_USDT", "AAVE_USDT",   "SHIB_USDT",   "ORDI_USDT",   "JUP_USDT",
    ]

    # ── Forex pairs (Yahoo Finance — no API key required) ────────────
    forex_symbols: list[str] = [
        "EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD",
        "AUD/USD", "USD/CAD", "NZD/USD", "USD/CHF",
    ]


settings = Settings()
