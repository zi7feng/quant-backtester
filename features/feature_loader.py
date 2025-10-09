"""
Feature Loader
--------------
Unified entry point for feature (indicator) computation.
Supports both full and chunked loading from DB for large backtests.
Ensures datetime index with America/New_York timezone.
"""

import pandas as pd
import pytz
from datetime import timedelta
from config.db_config import SessionLocal
from models.candle import Candle

# === Indicator imports ===
from features.indicators.trend import sma, ema, macd
from features.indicators.momentum import rsi, roc, momentum
from features.indicators.volatility import atr, bollinger_bands
from features.indicators.mean_reversion import zscore
from features.feature_utils import get_max_window


# ======================================================
# ✅ 1. Full-data Loader (default for short runs)
# ======================================================
def load_raw_data(symbol: str, start_date=None, end_date=None, limit: int = None) -> pd.DataFrame:
    """
    Load OHLCV candles for a given symbol from the database.
    Ensures datetime index with America/New_York timezone.
    """
    session = SessionLocal()
    try:
        query = session.query(Candle).filter(Candle.symbol == symbol)

        # Time filters
        if start_date:
            query = query.filter(Candle.datetime >= start_date)
        if end_date:
            query = query.filter(Candle.datetime <= end_date)

        # Order and limit
        if not start_date and not end_date:
            query = query.order_by(Candle.datetime.desc())
            if limit:
                query = query.limit(limit)
        else:
            query = query.order_by(Candle.datetime.asc())

        rows = query.all()
    finally:
        session.close()

    if not rows:
        raise ValueError(f"No data found for {symbol} in the specified range.")

    # Convert to DataFrame
    df = pd.DataFrame(
        [
            {
                "datetime": r.datetime,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
            for r in rows
        ]
    )

    # Normalize datetime index
    if "datetime" not in df.columns:
        raise KeyError("Missing 'datetime' column in candle data")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df.set_index("datetime", inplace=True)

    ny_tz = pytz.timezone("America/New_York")
    if df.index.tz is None:
        df.index = df.index.tz_localize(ny_tz)
    else:
        df.index = df.index.tz_convert(ny_tz)

    df.sort_index(inplace=True)

    assert isinstance(df.index, pd.DatetimeIndex), "Index must be DatetimeIndex"
    assert df.index.tz.zone == "America/New_York", "Index timezone must be America/New_York"

    return df


# ======================================================
# ✅ 2. Chunked Loader (for large backtests)
# ======================================================
def load_raw_data_chunked(symbol: str, start_date, end_date, days_per_chunk: int = 5):
    """
    Load OHLCV data from DB in small date chunks (streaming generator).

    Parameters
    ----------
    symbol : str
        e.g. "SPY.US"
    start_date, end_date : datetime
        UTC or tz-aware datetimes
    days_per_chunk : int
        How many days to read each time (default=5)

    Yields
    ------
    pd.DataFrame
        One chunk of OHLCV data indexed by datetime.
    """
    session = SessionLocal()
    ny_tz = pytz.timezone("America/New_York")

    try:
        current = start_date
        while current <= end_date:
            next_end = min(current + timedelta(days=days_per_chunk - 1), end_date)

            query = (
                session.query(Candle)
                .filter(Candle.symbol == symbol)
                .filter(Candle.datetime >= current)
                .filter(Candle.datetime <= next_end)
                .order_by(Candle.datetime.asc())
            )

            rows = query.all()
            if not rows:
                print(f"No data for {symbol} ({current.date()}–{next_end.date()})")
                current = next_end + timedelta(days=1)
                continue

            df = pd.DataFrame(
                [
                    {
                        "datetime": r.datetime,
                        "open": r.open,
                        "high": r.high,
                        "low": r.low,
                        "close": r.close,
                        "volume": r.volume,
                    }
                    for r in rows
                ]
            )
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            df.set_index("datetime", inplace=True)
            df.index = df.index.tz_convert(ny_tz)
            df.sort_index(inplace=True)

            print(f"Loaded {len(df):,} rows ({current.date()} → {next_end.date()})")

            yield df  # ✅ 每次返回一小块

            current = next_end + timedelta(days=1)

    finally:
        session.close()


# ======================================================
# ✅ 3. Indicator Computation
# ======================================================
def compute_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute all features based on config dictionary."""
    result = df.copy()

    # --- Trend indicators ---
    trend = cfg.get("trend", {})
    if "SMA" in trend:
        for w in trend["SMA"]:
            result[f"SMA_{w}"] = sma(df, window=w)
    if "EMA" in trend:
        for s in trend["EMA"]:
            result[f"EMA_{s}"] = ema(df, span=s)
    if "MACD" in trend:
        p = trend["MACD"]
        macd_df = macd(df, short_span=p.get("short", 12),
                       long_span=p.get("long", 26),
                       signal_span=p.get("signal", 9))
        result = result.join(macd_df)

    # --- Momentum indicators ---
    mom = cfg.get("momentum", {})
    if "RSI" in mom:
        for w in mom["RSI"]:
            result[f"RSI_{w}"] = rsi(df, window=w)
    if "ROC" in mom:
        for w in mom["ROC"]:
            result[f"ROC_{w}"] = roc(df, window=w)
    if "Momentum" in mom:
        for w in mom["Momentum"]:
            result[f"Momentum_{w}"] = momentum(df, window=w)

    # --- Volatility indicators ---
    vol = cfg.get("volatility", {})
    if "ATR" in vol:
        for w in vol["ATR"]:
            result[f"ATR_{w}"] = atr(df, window=w)
    if "Bollinger" in vol:
        p = vol["Bollinger"]
        bb = bollinger_bands(df,
                             window=p.get("window", 20),
                             num_std=p.get("num_std", 2.0))
        result = result.join(bb)

    # --- Mean reversion indicators ---
    mr = cfg.get("mean_reversion", {})
    if "ZScore" in mr:
        for w in mr["ZScore"]:
            result[f"ZScore_{w}"] = zscore(df["close"], window=w)

    return result


# ======================================================
# ✅ 4. Unified Feature Loader
# ======================================================
def load_features(symbol: str, config=None, buffer: float = 0.2,
                  start_date=None, end_date=None, limit: int = None) -> pd.DataFrame:
    """
    Unified entry point for feature (indicator) computation.

    Parameters
    ----------
    symbol : str
        Asset symbol, e.g. "SPY.US"
    config : IndicatorConfig or dict
        Indicator configuration
    buffer : float
        Fractional buffer for lookback extension (e.g. 0.2 = +20%)
    start_date, end_date : datetime
        Date range for backtest
    limit : int
        Optional row limit (used when no dates provided)
    """
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    cfg = config or {}

    max_window = get_max_window(cfg)
    total_window = int(max_window * (1 + buffer))

    # Load raw OHLCV data
    if start_date or end_date:
        print(f"Loading {symbol}: date range [{start_date}, {end_date}]")
        df = load_raw_data(symbol, start_date=start_date, end_date=end_date)
    else:
        if limit is None:
            limit = total_window
        print(f"Loading {symbol}: max window={max_window}, buffer={buffer*100:.0f}%, total={limit} bars")
        df = load_raw_data(symbol, limit=limit)

    # Compute indicators
    df_features = compute_indicators(df, cfg)
    return df_features
