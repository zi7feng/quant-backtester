# verify_data.py
from sqlalchemy.orm import Session
from config.db_config import SessionLocal
from models.candle import Candle
from sqlalchemy import func

def verify_symbol(symbol: str):
    """verify data and format"""
    session = SessionLocal()

    try:
        # count row
        total = session.query(func.count(Candle.id)).filter(Candle.symbol == symbol).scalar()

        # query earliest and latest data
        min_dt = session.query(func.min(Candle.datetime)).filter(Candle.symbol == symbol).scalar()
        max_dt = session.query(func.max(Candle.datetime)).filter(Candle.symbol == symbol).scalar()

        first5 = session.query(Candle).filter(Candle.symbol == symbol).order_by(Candle.datetime.asc()).limit(5).all()
        last5 = session.query(Candle).filter(Candle.symbol == symbol).order_by(Candle.datetime.desc()).limit(5).all()

        print(f"Symbol: {symbol}")
        print(f"Total rows: {total:,}")
        print(f"Time range: {min_dt} → {max_dt}")
        print("\nFirst 5 records:")
        for row in first5:
            print(f"  {row.datetime} | O:{row.open} H:{row.high} L:{row.low} C:{row.close} V:{row.volume}")

        print("\nLast 5 records:")
        for row in last5:
            print(f"  {row.datetime} | O:{row.open} H:{row.high} L:{row.low} C:{row.close} V:{row.volume}")

    finally:
        session.close()

if __name__ == "__main__":
    verify_symbol("SPY.US")
