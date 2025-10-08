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
    """
    Average True Range (ATR) for price volatility.
    Args:
        df: DataFrame with 'high', 'low', 'close' columns.
        window: Lookback period (default: 14).
    Returns:
        Series of ATR values (non-negative).
    Raises:
        ValueError: If required columns missing, window <= 0, or df is empty.
    """
    required_cols = ["high", "low", "close"]
    if not all(col in df for col in required_cols):
        raise ValueError(f"DataFrame must contain {required_cols}")
    if window <= 0:
        raise ValueError("Window must be positive")
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    high = df["high"].ffill().bfill()
    low = df["low"].ffill().bfill()
    close = df["close"].ffill().bfill()
    if (high < 0).any() or (low < 0).any() or (close < 0).any():
        raise ValueError("Prices cannot be negative")
    if (low > high).any():
        raise ValueError("Low prices cannot exceed high prices")
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1).fillna(0)  # Handle shift-induced NaN
    return tr.rolling(window=window, min_periods=1).mean()


def stddev(df: pd.DataFrame, column: str = "close", window: int = 20) -> pd.Series:
    """
    Rolling Standard Deviation for price volatility.
    Args:
        df: DataFrame with price data.
        column: Price column name (default: "close").
        window: Lookback period (default: 20).
    Returns:
        Series of standard deviation values (non-negative).
    Raises:
        ValueError: If column not in df, window <= 0, or df is empty.
    """
    if column not in df:
        raise ValueError(f"Column {column} not found in DataFrame")
    if window <= 0:
        raise ValueError("Window must be positive")
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    prices = df[column].ffill().bfill()
    if (prices < 0).any():
        raise ValueError("Prices cannot be negative")
    return prices.rolling(window=window, min_periods=1).std(ddof=0).fillna(0)


def bollinger_bands(df: pd.DataFrame,
                    column: str = "close",
                    window: int = 20,
                    num_std: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands for volatility and reversal signals.
    Args:
        df: DataFrame with price data.
        column: Price column name (default: "close").
        window: Lookback period (default: 20).
        num_std: Number of standard deviations (default: 2.0).
    Returns:
        DataFrame with 'Upper', 'Middle', and 'Lower' bands.
    Raises:
        ValueError: If column not in df, window <= 0, num_std < 0, or df is empty.
    """
    if column not in df:
        raise ValueError(f"Column {column} not found in DataFrame")
    if window <= 0:
        raise ValueError("Window must be positive")
    if num_std < 0:
        raise ValueError("num_std cannot be negative")
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    prices = df[column].ffill().bfill()
    if (prices < 0).any():
        raise ValueError("Prices cannot be negative")
    ma = prices.rolling(window=window, min_periods=1).mean()
    std = prices.rolling(window=window, min_periods=1).std(ddof=0).fillna(0)
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
    Historical Volatility (annualized) for risk assessment.
    Args:
        df: DataFrame with price data.
        column: Price column name (default: "close").
        window: Lookback period (default: 30).
        trading_days: Annual trading days for scaling (default: 252).
    Returns:
        Series of annualized volatility values (non-negative).
    Raises:
        ValueError: If column not in df, window <= 0, trading_days <= 0, or df is empty.
    """
    if column not in df:
        raise ValueError(f"Column {column} not found in DataFrame")
    if window <= 0 or trading_days <= 0:
        raise ValueError("Window and trading_days must be positive")
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    prices = df[column].ffill().bfill()
    if (prices <= 0).any():
        raise ValueError("Prices must be positive")
    log_ret = np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan)
    return log_ret.rolling(window=window, min_periods=1).std(ddof=0) * np.sqrt(trading_days)


def rsv(df: pd.DataFrame, column: str = "close", window: int = 20) -> pd.Series:
    """
    Relative Std. Volatility (σ / mean) for volatility comparison.
    Args:
        df: DataFrame with price data.
        column: Price column name (default: "close").
        window: Lookback period (default: 20).
    Returns:
        Series of RSV values (non-negative).
    Raises:
        ValueError: If column not in df, window <= 0, or df is empty.
    """
    if column not in df:
        raise ValueError(f"Column {column} not found in DataFrame")
    if window <= 0:
        raise ValueError("Window must be positive")
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    prices = df[column].ffill().bfill()
    if (prices < 0).any():
        raise ValueError("Prices cannot be negative")
    rolling_std = prices.rolling(window=window, min_periods=1).std(ddof=0).fillna(0)
    rolling_mean = prices.rolling(window=window, min_periods=1).mean()
    rsv = rolling_std / (rolling_mean + 1e-10)
    return rsv.replace([np.inf, -np.inf], 0)  # Handle zero mean
