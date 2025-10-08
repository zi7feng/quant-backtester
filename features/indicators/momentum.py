"""
Implements momentum-based indicators:
- ROC (Rate of Change)
- Momentum
- RSI (Relative Strength Index)
- Stochastic Oscillator (%K, %D)
- CCI (Commodity Channel Index)
- Williams %R

Each function takes a DataFrame with price columns (default: "close")
and returns a Series or DataFrame aligned with the input index.
"""

import pandas as pd
import numpy as np


def roc(df: pd.DataFrame, column: str = "close", window: int = 12) -> pd.Series:
    """
    Rate of Change (ROC) as a percentage.
    Args:
        df: DataFrame with price data.
        column: Price column name (default: "close").
        window: Lookback period (default: 12).
    Returns:
        Series of ROC values in percentage.
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
    lagged = prices.shift(window)
    roc = ((prices / (lagged + 1e-10) - 1.0) * 100).fillna(0)
    return roc


def momentum(df: pd.DataFrame, column: str = "close", window: int = 10) -> pd.Series:
    """
    Momentum = current price - N-period price.
    Args:
        df: DataFrame with price data.
        column: Price column name (default: "close").
        window: Lookback period (default: 10).
    Returns:
        Series of momentum values.
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
    return (prices - prices.shift(window)).fillna(0)


def rsi(df: pd.DataFrame, column: str = "close", window: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI), a momentum oscillator.
    Args:
        df: DataFrame with price data.
        column: Price column name (default: "close").
        window: Lookback period (default: 14, typical for daily data).
    Returns:
        Series of RSI values (0 to 100, >70 overbought, <30 oversold).
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
    delta = prices.diff().fillna(0)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.clip(lower=0, upper=100)  # Ensure valid range


def stochastic_oscillator(df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
    """
    Stochastic Oscillator (%K, %D) for overbought/oversold signals.
    Args:
        df: DataFrame with 'high', 'low', 'close' columns.
        k_window: Lookback period for %K (default: 14).
        d_window: Smoothing period for %D (default: 3).
    Returns:
        DataFrame with '%K' and '%D' columns (0 to 100).
    Raises:
        ValueError: If required columns missing, windows <= 0, or df is empty.
    """
    required_cols = ["high", "low", "close"]
    if not all(col in df for col in required_cols):
        raise ValueError(f"DataFrame must contain {required_cols}")
    if k_window <= 0 or d_window <= 0:
        raise ValueError("k_window and d_window must be positive")
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    high = df["high"].ffill().bfill()
    low = df["low"].ffill().bfill()
    close = df["close"].ffill().bfill()
    if (high < 0).any() or (low < 0).any() or (close < 0).any():
        raise ValueError("Prices cannot be negative")
    if (low > high).any():
        raise ValueError("Low prices cannot exceed high prices")
    low_min = low.rolling(window=k_window, min_periods=1).min()
    high_max = high.rolling(window=k_window, min_periods=1).max()
    denominator = high_max - low_min + 1e-10
    percent_k = 100 * (close - low_min) / denominator
    percent_k = percent_k.where(high_max != low_min, 100.0).clip(lower=0, upper=100)
    percent_d = percent_k.rolling(window=d_window, min_periods=1).mean()
    return pd.DataFrame({
        "%K": percent_k,
        "%D": percent_d
    })

def cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Commodity Channel Index (CCI) for overbought/oversold signals.
    Args:
        df: DataFrame with 'high', 'low', 'close' columns.
        window: Lookback period (default: 20).
    Returns:
        Series of CCI values, clipped to [-1000, 1000] for stability.
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
    tp = (high + low + close) / 3
    ma = tp.rolling(window=window, min_periods=1).mean()
    md = (tp - ma).abs().rolling(window=window, min_periods=1).mean()
    cci = (tp - ma) / (0.015 * (md + 1e-10))
    return cci.clip(lower=-1000, upper=1000)  # Limit extreme values


def williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Williams %R for overbought/oversold signals.
    Args:
        df: DataFrame with 'high', 'low', 'close' columns.
        window: Lookback period (default: 14).
    Returns:
        Series of Williams %R values (-100 to 0).
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
    high_max = high.rolling(window=window, min_periods=1).max()
    low_min = low.rolling(window=window, min_periods=1).min()
    wr = -100 * (high_max - close) / (high_max - low_min + 1e-10)
    return wr.clip(lower=-100, upper=0)
