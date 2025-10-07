import os

class Settings:
    def __init__(self):
        # --- Database ---
        self.DB_USER = os.getenv("DB_USER", "zfeng")
        self.DB_PASS = os.getenv("DB_PASS", "")
        self.DB_HOST = os.getenv("DB_HOST", "localhost")
        self.DB_PORT = os.getenv("DB_PORT", "5432")
        self.DB_NAME = os.getenv("DB_NAME", "quant_backtest")

        # --- EODHD API ---
        self.EODHD_API_KEY = os.getenv("EODHD_API_KEY", "")

    @property
    def DATABASE_URL(self):
        """Generate PostgreSQL connection URL."""
        if self.DB_PASS:
            return f"postgresql://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        else:
            return f"postgresql://{self.DB_USER}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

settings = Settings()