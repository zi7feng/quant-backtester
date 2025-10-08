"""
Implements volatility-based indicators:
- ATR (Average True Range)
- StdDev (Rolling Standard Deviation)
- Bollinger Bands (Upper, Middle, Lower)
- Historical Volatility (HV)
- RSV (Relative Std. Volatility)

Each function takes a DataFrame with price columns (default: "close")
and returns a Series or DataFrame aligned with the input index.
"""

import pandas as pd
import numpy as np


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range (ATR)."""
    high, low, close = df["high"], df["low"], df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(window=window, min_periods=1).mean()


def stddev(df: pd.DataFrame, column: str = "close", window: int = 20) -> pd.Series:
    """Rolling Standard Deviation."""
    return df[column].rolling(window=window, min_periods=1).std()


def bollinger_bands(df: pd.DataFrame,
                    column: str = "close",
                    window: int = 20,
                    num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: mean ± n * std."""
    ma = df[column].rolling(window=window, min_periods=1).mean()
    std = df[column].rolling(window=window, min_periods=1).std()

    upper = ma + num_std * std
    lower = ma - num_std * std

    return pd.DataFrame({
        "Upper": upper,
        "Middle": ma,
        "Lower": lower
    })


def historical_volatility(df: pd.DataFrame,
                          column: str = "close",
                          window: int = 30,
                          trading_days: int = 252) -> pd.Series:
    """
    Historical Volatility (annualized).
    HV = std(log returns) * sqrt(trading_days)
    """
    log_ret = np.log(df[column] / df[column].shift(1))
    return log_ret.rolling(window=window, min_periods=1).std() * np.sqrt(trading_days)


def rsv(df: pd.DataFrame, column: str = "close", window: int = 20) -> pd.Series:
    """Relative Std. Volatility (σ / mean)."""
    rolling_std = df[column].rolling(window=window, min_periods=1).std()
    rolling_mean = df[column].rolling(window=window, min_periods=1).mean()
    return rolling_std / rolling_mean
