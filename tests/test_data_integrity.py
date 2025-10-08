from config.db_config import SessionLocal
from models.candle import Candle
from sqlalchemy import func
import pandas as pd

def test_symbol_data_integrity():
    """Verify data quality for a sample symbol."""
    symbol = "SPY.US"
    session = SessionLocal()

    try:
        rows = (
            session.query(Candle)
            .filter(Candle.symbol == symbol)
            .order_by(Candle.datetime.asc())
            .limit(1000)
            .all()
        )
        assert rows, f"No data found for {symbol}"

        df = pd.DataFrame([{
            "datetime": r.datetime,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume,
        } for r in rows])

        # check NaN
        assert not df.isnull().values.any(), "Data contains null values"

        # check open ≤ high ≥ low ≤ close
        invalid_rows = df[
            (df["high"] < df["low"]) |
            (df["open"] > df["high"]) |
            (df["close"] > df["high"])
        ]
        assert invalid_rows.empty, "Detected invalid OHLC relationships"

        # check datetime ASC
        assert df["datetime"].is_monotonic_increasing, "Datetime not sorted ascending"

    finally:
        session.close()
