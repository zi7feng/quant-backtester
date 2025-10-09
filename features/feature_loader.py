"""
Feature Loader
--------------
Unified entry point for feature (indicator) computation.
Loads OHLCV data from DB, determines required lookback window
based on indicator config, applies buffer, and computes all features.

Usage:
    from features.indicator_config import IndicatorConfig
    from features.feature_loader import load_features

    cfg = IndicatorConfig()
    df_features = load_features("SPY.US", config=cfg, buffer=0.2)
"""

import pandas as pd
from config.db_config import SessionLocal
from models.candle import Candle

# Import all indicator groups
from features.indicators.trend import sma, ema, macd
from features.indicators.momentum import rsi, roc, momentum
from features.indicators.volatility import atr, bollinger_bands
from features.indicators.mean_reversion import zscore
from features.feature_utils import get_max_window


def load_raw_data(symbol: str, start_date=None, end_date=None, limit: int = None) -> pd.DataFrame:
    """
    Load OHLCV candles for a given symbol from the database.
    Supports both time range queries (start_date/end_date)
    and recent data queries (limit).

    Parameters
    ----------
    symbol : str
        Asset symbol, e.g. "SPY.US"
    start_date : datetime or None
        Start datetime (tz-aware, e.g. America/New_York). If None, no lower bound is applied.
    end_date : datetime or None
        End datetime (tz-aware). If None, no upper bound is applied.
    limit : int or None
        Number of most recent records to load when no time range is specified.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by datetime, containing columns:
        open, high, low, close, volume
    """
    session = SessionLocal()
    try:
        query = session.query(Candle).filter(Candle.symbol == symbol)

        # If a specific time range is provided, apply it directly (preferred for historical backtests)
        if start_date:
            query = query.filter(Candle.datetime >= start_date)
        if end_date:
            query = query.filter(Candle.datetime <= end_date)

        # If no explicit time range, pull the latest candles using LIMIT
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

    # Set datetime index for downstream indicator computation
    df.set_index("datetime", inplace=True)
    return df



# -----------------------------
# Compute indicators by config
# -----------------------------
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


# -----------------------------
# Main loader
# -----------------------------
def load_features(symbol: str, config=None, buffer: float = 0.2,
                  start_date=None, end_date=None, limit: int = None) -> pd.DataFrame:
    """
    Unified entry point for feature (indicator) computation.
    Supports both 'date range mode' and 'limit mode'.

    Parameters
    ----------
    symbol : str
        Asset symbol (e.g. "SPY.US")
    config : IndicatorConfig or dict
        Indicator configuration object.
    buffer : float
        Fractional buffer for lookback extension (e.g. 0.2 = +20%)
    start_date, end_date : datetime or None
        Date range mode (historical backtest)
    limit : int or None
        Number of recent bars (realtime or short-term analysis)
    """
    # Convert config to dict if necessary
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    cfg = config or {}

    # Detect largest required window
    max_window = get_max_window(cfg)
    total_window = int(max_window * (1 + buffer))

    # Choose loading mode
    if start_date or end_date:
        print(f"Loading {symbol}: using date range [{start_date}, {end_date}]")
        df = load_raw_data(symbol, start_date=start_date, end_date=end_date)
    else:
        # Use limit-based mode
        if limit is None:
            limit = total_window
        print(f"Loading {symbol}: max window={max_window}, buffer={buffer*100:.0f}%, total={limit} bars")
        df = load_raw_data(symbol, limit=limit)

    # Compute all features
    df_features = compute_indicators(df, cfg)
    return df_features
