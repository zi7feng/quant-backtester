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
    """Simple Moving Average (SMA)."""
    return df[column].rolling(window=window, min_periods=1).mean()


def ema(df: pd.DataFrame, column: str = "close", span: int = 20) -> pd.Series:
    """Exponential Moving Average (EMA)."""
    return df[column].ewm(span=span, adjust=False).mean()


def wma(df: pd.DataFrame, column: str = "close", window: int = 20) -> pd.Series:
    """Weighted Moving Average (WMA)."""
    weights = pd.Series(range(1, window + 1), dtype=float)
    return df[column].rolling(window=window).apply(
        lambda x: (x * weights).sum() / weights.sum(), raw=True
    )


def macd(df: pd.DataFrame,
         column: str = "close",
         short_span: int = 12,
         long_span: int = 26,
         signal_span: int = 9) -> pd.DataFrame:
    """Moving Average Convergence Divergence (MACD)."""
    ema_short = ema(df, column=column, span=short_span)
    ema_long = ema(df, column=column, span=long_span)

    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame({
        "MACD": macd_line,
        "Signal": signal_line,
        "Histogram": histogram
    })


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average Directional Index (ADX)."""
    high, low, close = df["high"], df["low"], df["close"]

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
    plus_di = 100 * (pd.Series(plus_dm).rolling(window).sum() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100

    return dx.rolling(window=window).mean()


def slope(df: pd.DataFrame, column: str = "close", window: int = 20) -> pd.Series:
    """Linear Regression Slope."""
    slopes = []
    for i in range(len(df)):
        if i < window:
            slopes.append(np.nan)
            continue
        y = df[column].iloc[i - window:i]
        x = np.arange(window)
        slope_val, _, _, _, _ = linregress(x, y)
        slopes.append(slope_val)
    return pd.Series(slopes, index=df.index)
