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

    # Twelve Data (Forex / Commodities)
    twelvedata_api_key: str = ""

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
    # Matches PineScript ta.supertrend(multiplier, atr_len) / 'Simple Supertrend' mode.
    # Green (bull) when close > trailing lower band. Red (bear) when close < trailing upper band.
    mt_atr_len: int = 14
    mt_multiplier: float = 2.5

    # ── MEXC Perpetual Futures (30 crypto pairs) ─────
    # Format: BASE_USDT  (matches MEXC Contract API symbol format)
    mexc_symbols: list[str] = [
        "BTC_USDT",    "ETH_USDT",    "SOL_USDT",    "XRP_USDT",    "DOGE_USDT",
        "BNB_USDT",    "ADA_USDT",    "AVAX_USDT",   "DOT_USDT",    "LINK_USDT",
        "POL_USDT",    "LTC_USDT",    "ATOM_USDT",   "NEAR_USDT",   "APT_USDT",
        "ARB_USDT",    "OP_USDT",     "SUI_USDT",    "FIL_USDT",    "INJ_USDT",
        "SEI_USDT",    "TIA_USDT",    "WIF_USDT",    "PEPE_USDT",   "S_USDT",
        "RENDER_USDT", "AAVE_USDT",   "TON_USDT",    "ORDI_USDT",   "JUP_USDT",
    ]

    # ── Twelve Data Forex (8 confirmed free-tier pairs) ─
    # XAG/USD (Silver) requires a paid plan — excluded.
    # US30/USD (Dow Jones) is not available on Twelve Data — excluded.
    twelvedata_symbols: list[str] = [
        "EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD",
        "AUD/USD", "USD/CAD", "NZD/USD", "USD/CHF",
    ]


settings = Settings()
