# models/candle.py
from sqlalchemy import Column, Integer, String, Float, BigInteger, DateTime, func
from config.db_config import Base

class Candle(Base):
    __tablename__ = "candles"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    interval = Column(String(5), nullable=False)
    datetime = Column(DateTime(timezone=True), nullable=False, index=True)

    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(BigInteger)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
