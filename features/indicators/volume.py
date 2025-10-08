"""
Implements volume-based indicators:
- OBV (On-Balance Volume)
- Volume Moving Average (VMA)
- Volume Ratio (VR)
- Money Flow Index (MFI)

Each function takes a DataFrame with 'close' and 'volume' columns,
and returns a Series or DataFrame aligned with the input index.
"""

import pandas as pd
import numpy as np


def obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume (OBV) for trend confirmation.
    Args:
        df: DataFrame with 'close' and 'volume' columns.
    Returns:
        Series of cumulative OBV values.
    Raises:
        ValueError: If required columns missing or df is empty.
    """
    required_cols = ["close", "volume"]
    if not all(col in df for col in required_cols):
        raise ValueError(f"DataFrame must contain {required_cols}")
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    close = df["close"].ffill().bfill()
    volume = df["volume"].ffill().bfill()
    if (close < 0).any():
        raise ValueError("Close prices cannot be negative")
    if (volume < 0).any():
        raise ValueError("Volume cannot be negative")
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def volume_ma(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Volume Moving Average (VMA) for volume trend analysis.
    Args:
        df: DataFrame with 'volume' column.
        window: Lookback period (default: 20).
    Returns:
        Series of VMA values (non-negative).
    Raises:
        ValueError: If volume column missing, window <= 0, or df is empty.
    """
    if "volume" not in df:
        raise ValueError("DataFrame must contain 'volume' column")
    if window <= 0:
        raise ValueError("Window must be positive")
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    volume = df["volume"].ffill().bfill()
    if (volume < 0).any():
        raise ValueError("Volume cannot be negative")
    return volume.rolling(window=window, min_periods=1).mean()


def volume_ratio(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Volume Ratio (VR) = current volume / rolling average volume.
    Args:
        df: DataFrame with 'volume' column.
        window: Lookback period (default: 20).
    Returns:
        Series of VR values (non-negative).
    Raises:
        ValueError: If volume column missing, window <= 0, or df is empty.
    """
    if "volume" not in df:
        raise ValueError("DataFrame must contain 'volume' column")
    if window <= 0:
        raise ValueError("Window must be positive")
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    volume = df["volume"].ffill().bfill()
    if (volume < 0).any():
        raise ValueError("Volume cannot be negative")
    avg_vol = volume.rolling(window=window, min_periods=1).mean()
    vr = volume / (avg_vol + 1e-10)
    return vr.replace([np.inf, -np.inf], 0).clip(lower=0)


def mfi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Money Flow Index (MFI) for overbought/oversold signals.
    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume' columns.
        window: Lookback period (default: 14).
    Returns:
        Series of MFI values (0 to 100).
    Raises:
        ValueError: If required columns missing, window <= 0, or df is empty.
    """
    required_cols = ["high", "low", "close", "volume"]
    if not all(col in df for col in required_cols):
        raise ValueError(f"DataFrame must contain {required_cols}")
    if window <= 0:
        raise ValueError("Window must be positive")
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    high = df["high"].ffill().bfill()
    low = df["low"].ffill().bfill()
    close = df["close"].ffill().bfill()
    volume = df["volume"].ffill().bfill()
    if (high < 0).any() or (low < 0).any() or (close < 0).any():
        raise ValueError("Prices cannot be negative")
    if (volume < 0).any():
        raise ValueError("Volume cannot be negative")
    if (low > high).any():
        raise ValueError("Low prices cannot exceed high prices")
    tp = (high + low + close) / 3
    mf = tp * volume
    delta = tp.diff().fillna(0)
    pos_flow = np.where(delta > 0, mf, 0.0)
    neg_flow = np.where(delta < 0, mf, 0.0)
    pos_mf = pd.Series(pos_flow).rolling(window=window, min_periods=1).sum()
    neg_mf = pd.Series(neg_flow).rolling(window=window, min_periods=1).sum()
    mfr = pos_mf / (neg_mf + 1e-10)
    mfi = 100 - (100 / (1 + mfr))
    mfi = mfi.where((pos_mf != 0) | (neg_mf != 0), 50.0).clip(lower=0, upper=100)
    return mfi