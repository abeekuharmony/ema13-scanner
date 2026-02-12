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

    # Twelve Data
    twelvedata_api_key: str = ""

    # Scanner behavior
    scan_interval_minutes: int = 1
    binance_top_n: int = 25
    log_level: str = "INFO"

    # Twelve Data symbols (forex + commodities + crypto)
    # Free tier supports: XAU/USD, EUR/USD, GBP/USD, USD/JPY, BTC/USD, ETH/USD
    # Paid tier adds: XAG/USD, SPX, and more
    twelvedata_symbols: list[str] = [
        "XAU/USD",
        "EUR/USD",
        "GBP/USD",
        "USD/JPY",
    ]

    # EMA settings
    ema_period: int = 13
    candle_limit: int = 50


settings = Settings()
