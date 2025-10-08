"""
Implements trend-based indicators:
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- WMA (Weighted Moving Average)
- MACD (Moving Average Convergence Divergence)
- ADX (Average Directional Index)
- Slope (Linear Regression Slope)

Each function takes a DataFrame with price columns (default: "close")
and returns a Series or DataFrame aligned with the input index.
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress


def sma(df: pd.DataFrame, column: str = "close", window: int = 20) -> pd.Series:
    """
    Simple Moving Average (SMA) for trend identification.
    Args:
        df: DataFrame with price data.
        column: Price column name (default: "close").
        window: Lookback period (default: 20).
    Returns:
        Series of SMA values.
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
    return prices.rolling(window=window, min_periods=1).mean()


def ema(df: pd.DataFrame, column: str = "close", span: int = 20) -> pd.Series:
    """
    Exponential Moving Average (EMA) for trend identification.
    Args:
        df: DataFrame with price data.
        column: Price column name (default: "close").
        span: Smoothing period (default: 20).
    Returns:
        Series of EMA values.
    Raises:
        ValueError: If column not in df, span <= 0, or df is empty.
    """
    if column not in df:
        raise ValueError(f"Column {column} not found in DataFrame")
    if span <= 0:
        raise ValueError("Span must be positive")
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    prices = df[column].ffill().bfill()
    if (prices < 0).any():
        raise ValueError("Prices cannot be negative")
    return prices.ewm(span=span, adjust=False).mean()


def wma(df: pd.DataFrame, column: str = "close", window: int = 20) -> pd.Series:
    """
    Weighted Moving Average (WMA) for trend identification.
    Args:
        df: DataFrame with price data.
        column: Price column name (default: "close").
        window: Lookback period (default: 20).
    Returns:
        Series of WMA values.
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
    weights = np.arange(1, window + 1, dtype=float)
    wma = prices.rolling(window=window, min_periods=1).apply(
        lambda x: np.sum(x * weights[-len(x):]) / np.sum(weights[-len(x):]), raw=True
    )
    return wma


def macd(df: pd.DataFrame,
         column: str = "close",
         short_span: int = 12,
         long_span: int = 26,
         signal_span: int = 9) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence (MACD) for trend signals.
    Args:
        df: DataFrame with price data.
        column: Price column name (default: "close").
        short_span: Short EMA period (default: 12).
        long_span: Long EMA period (default: 26).
        signal_span: Signal line period (default: 9).
    Returns:
        DataFrame with 'MACD', 'Signal', and 'Histogram' columns.
    Raises:
        ValueError: If column not in df, spans <= 0, or df is empty.
    """
    if column not in df:
        raise ValueError(f"Column {column} not found in DataFrame")
    if short_span <= 0 or long_span <= 0 or signal_span <= 0:
        raise ValueError("Spans must be positive")
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    prices = df[column].ffill().bfill()
    if (prices < 0).any():
        raise ValueError("Prices cannot be negative")
    ema_short = prices.ewm(span=short_span, adjust=False).mean()
    ema_long = prices.ewm(span=long_span, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "MACD": macd_line,
        "Signal": signal_line,
        "Histogram": histogram
    })


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average Directional Index (ADX) for trend strength.
    Args:
        df: DataFrame with 'high', 'low', 'close' columns.
        window: Lookback period (default: 14).
    Returns:
        Series of ADX values (0 to 100).
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
    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (low.diff() < 0), minus_dm, 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=1).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=window, min_periods=1).sum() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=window, min_periods=1).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    adx = dx.rolling(window=window, min_periods=1).mean()
    return adx.clip(lower=0, upper=100)


def slope(df: pd.DataFrame, column: str = "close", window: int = 20) -> pd.Series:
    """
    Linear Regression Slope for trend direction and strength.
    Args:
        df: DataFrame with price data.
        column: Price column name (default: "close").
        window: Lookback period (default: 20).
    Returns:
        Series of slope values.
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

    def calc_slope(x):
        if len(x) < 2 or np.any(np.isnan(x)):
            return np.nan
        x_vals = np.arange(len(x))
        try:
            slope, _, _, _, _ = linregress(x_vals, x)
            return slope
        except Exception:
            return np.nan

    return prices.rolling(window=window, min_periods=1).apply(calc_slope, raw=True)
