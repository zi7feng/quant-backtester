from sqlalchemy.orm import Session
from config.db_config import SessionLocal
from models.candle import Candle
from sqlalchemy import func
import pandas as pd


def verify_symbol(symbol: str):
    """Verify data and print recent daily opens & closes, plus detail for one day."""
    session = SessionLocal()

    try:
        # --- Basic info ---
        total = session.query(func.count(Candle.id)).filter(Candle.symbol == symbol).scalar()
        min_dt = session.query(func.min(Candle.datetime)).filter(Candle.symbol == symbol).scalar()
        max_dt = session.query(func.max(Candle.datetime)).filter(Candle.symbol == symbol).scalar()

        print(f"\nSymbol: {symbol}")
        print(f"Total rows: {total:,}")
        print(f"Time range: {min_dt} → {max_dt}")

        # --- Load last ~10 days of data (roughly 4000 bars) ---
        candles = (
            session.query(Candle)
            .filter(Candle.symbol == symbol)
            .order_by(Candle.datetime.desc())
            .limit(4000)
            .all()
        )

        if not candles:
            print("No data found.")
            return

        # --- Convert to DataFrame ---
        df = pd.DataFrame(
            [{
                "datetime": c.datetime,
                "open": c.open,
                "close": c.close,
            } for c in candles]
        )

        df = df.sort_values("datetime").reset_index(drop=True)
        df["date"] = df["datetime"].dt.date

        # --- Group by day ---
        daily_summary = (
            df.groupby("date")
            .agg(open=("open", "first"), close=("close", "last"))
            .reset_index()
        )

        print("\nRecent 10 trading days (Open/Close):")
        print(daily_summary.tail(10).to_string(index=False))

        # ------------------------------------------------------------------
        # 🔍 Additional: Show full details for a specific day (e.g. 2025-04-24)
        # ------------------------------------------------------------------
        target_date = "2025-04-24"

        day_candles = (
            session.query(Candle)
            .filter(
                Candle.symbol == symbol,
                func.date(Candle.datetime) == target_date
            )
            .order_by(Candle.datetime.asc())
            .all()
        )

        if not day_candles:
            print(f"\nNo records found for {target_date}.")
        else:
            df_day = pd.DataFrame(
                [{
                    "datetime": c.datetime,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume
                } for c in day_candles]
            )

            print(f"\n✅ Found {len(df_day)} records for {symbol} on {target_date}")

            print(f"\n🔹 First 5 records on {target_date}:")
            print(df_day.head(5).to_string(index=False))

            print(f"\n🔹 Last 5 records on {target_date}:")
            print(df_day.tail(5).to_string(index=False))

    finally:
        session.close()


if __name__ == "__main__":
    verify_symbol("SPY.US")
