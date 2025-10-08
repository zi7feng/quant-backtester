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
    """On-Balance Volume (OBV)."""
    direction = np.sign(df["close"].diff().fillna(0))
    return (direction * df["volume"]).cumsum()


def volume_ma(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Volume Moving Average (VMA)."""
    return df["volume"].rolling(window=window, min_periods=1).mean()


def volume_ratio(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Volume Ratio (VR) = current volume / rolling average volume."""
    avg_vol = df["volume"].rolling(window=window, min_periods=1).mean()
    return df["volume"] / (avg_vol + 1e-10)


def mfi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Money Flow Index (MFI)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]

    delta = tp.diff()
    pos_flow = np.where(delta > 0, mf, 0.0)
    neg_flow = np.where(delta < 0, mf, 0.0)

    pos_mf = pd.Series(pos_flow).rolling(window=window, min_periods=1).sum()
    neg_mf = pd.Series(neg_flow).rolling(window=window, min_periods=1).sum()

    mfr = pos_mf / (neg_mf + 1e-10)
    return 100 - (100 / (1 + mfr))
