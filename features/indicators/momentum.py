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
    """Rate of Change (ROC)."""
    return (df[column] / df[column].shift(window) - 1.0) * 100


def momentum(df: pd.DataFrame, column: str = "close", window: int = 10) -> pd.Series:
    """Momentum = current price - N-period price."""
    return df[column] - df[column].shift(window)


def rsi(df: pd.DataFrame, column: str = "close", window: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)."""
    delta = df[column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def stochastic_oscillator(df: pd.DataFrame,
                          k_window: int = 14,
                          d_window: int = 3) -> pd.DataFrame:
    """Stochastic Oscillator (%K, %D)."""
    low_min = df["low"].rolling(window=k_window, min_periods=1).min()
    high_max = df["high"].rolling(window=k_window, min_periods=1).max()

    percent_k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-10)
    percent_d = percent_k.rolling(window=d_window, min_periods=1).mean()

    return pd.DataFrame({
        "%K": percent_k,
        "%D": percent_d
    })


def cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Commodity Channel Index (CCI)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(window=window, min_periods=1).mean()
    md = tp.rolling(window=window, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)

    return (tp - ma) / (0.015 * (md + 1e-10))


def williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Williams %R."""
    high_max = df["high"].rolling(window=window, min_periods=1).max()
    low_min = df["low"].rolling(window=window, min_periods=1).min()

    return -100 * (high_max - df["close"]) / (high_max - low_min + 1e-10)
