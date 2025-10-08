"""
Implements mean-reversion metrics:
- Z-Score of Price/Spread
- Rolling Mean & Std
- Cointegration Test (Engle-Granger)
- Hurst Exponent

Each function takes a DataFrame or Series and returns statistical metrics
used for mean reversion and pair-trading strategies.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint


def rolling_mean_std(series: pd.Series, window: int = 30) -> pd.DataFrame:
    """
    Compute rolling mean and std of a series.
    Args:
        series: Input time series (e.g., price or returns).
        window: Rolling window size (default: 30).
    Returns:
        DataFrame with 'RollingMean' and 'RollingStd' columns.
    Raises:
        ValueError: If window <= 0 or series is empty.
    """
    if window <= 0:
        raise ValueError("Window must be positive")
    if series.empty:
        raise ValueError("Input series cannot be empty")
    # Handle NaN with forward/backward fill
    series = series.ffill().bfill()
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std(ddof=0).fillna(0)
    return pd.DataFrame({
        "RollingMean": rolling_mean,
        "RollingStd": rolling_std
    })


def zscore(series: pd.Series, window: int = 30) -> pd.Series:
    """
    Z-Score = (price - rolling mean) / rolling std.
    Args:
        series: Input time series (e.g., price or spread).
        window: Rolling window size (default: 30).
    Returns:
        Series of Z-scores, clipped to [-10, 10] for stability.
    Raises:
        ValueError: If window <= 0 or series is empty.
    """
    if window <= 0:
        raise ValueError("Window must be positive")
    if series.empty:
        raise ValueError("Input series cannot be empty")
    series = series.ffill().bfill()
    mean = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std(ddof=0).fillna(0)
    z = (series - mean) / (std + 1e-10)
    return z.clip(lower=-10, upper=10)


def cointegration_test(series_a: pd.Series, series_b: pd.Series) -> dict:
    """
    Engle-Granger Cointegration Test between two assets.
    Args:
        series_a: Price series of first asset.
        series_b: Price series of second asset.
    Returns:
        Dict with t-statistic, p-value, and critical values.
    Raises:
        ValueError: If series are empty, unequal length, or all NaN.
    """
    if series_a.empty or series_b.empty:
        raise ValueError("Series cannot be empty")
    if len(series_a) != len(series_b):
        raise ValueError("Series must have equal length")
    if series_a.isna().all() or series_b.isna().all():
        raise ValueError("Series cannot be all NaN")
    # Align indices and fill NaN using ffill and bfill
    series_a, series_b = series_a.align(series_b, join='inner')
    series_a = series_a.ffill().bfill()
    series_b = series_b.ffill().bfill()
    try:
        coint_result = coint(series_a, series_b)
        return {
            "t_stat": coint_result[0],
            "p_value": coint_result[1],
            "critical_values": {
                "1%": coint_result[2][0],
                "5%": coint_result[2][1],
                "10%": coint_result[2][2],
            }
        }
    except Exception as e:
        return {
            "t_stat": np.nan,
            "p_value": np.nan,
            "critical_values": {"1%": np.nan, "5%": np.nan, "10%": np.nan}
        }


def hurst_exponent(series: pd.Series, min_n: int = 16, min_segments: int = 10) -> float:
    """
    Estimate Hurst Exponent (H) using rescaled range (R/S) analysis.
    Args:
        series: Input time series (e.g., price or returns).
        min_n: Minimum segment size (default: 16).
        min_segments: Minimum number of segments (default: 10).
    Returns:
        Float H value (<0.5: mean-reverting, ~0.5: random walk, >0.5: trending).
        Returns np.nan if data is insufficient or invalid.
    Raises:
        ValueError: If min_n or min_segments <= 0, or series is empty/all NaN.
    """
    if min_n <= 0 or min_segments <= 0:
        raise ValueError("min_n and min_segments must be positive")
    if series.empty:
        raise ValueError("Input series cannot be empty")
    if series.isna().all():
        raise ValueError("Input series cannot be all NaN")
    series = series.ffill().bfill().values
    N = len(series)
    if N < 100:
        return np.nan
    max_n = N // min_segments
    n_list = []
    current = min_n
    while current <= max_n:
        n_list.append(current)
        current *= 2
    if len(n_list) < 2:
        return np.nan
    rs_list = []
    for n in n_list:
        num_segments = N // n
        if num_segments < min_segments:
            continue
        avg_rs = 0.0
        for k in range(num_segments):
            sub = series[k * n : (k + 1) * n]
            mean = np.mean(sub)
            if np.isnan(mean):
                continue
            deviate = sub - mean
            cum = np.cumsum(deviate)
            R = np.max(cum) - np.min(cum)
            S = np.std(sub, ddof=0) + 1e-10
            avg_rs += R / S
        avg_rs /= num_segments
        expected_rs = np.sqrt(n * np.pi / 2)
        rs_list.append(avg_rs / expected_rs)
    if len(rs_list) < 2:
        return np.nan
    log_n = np.log(n_list[:len(rs_list)])
    log_rs = np.log(rs_list)
    try:
        slope = np.polyfit(log_n, log_rs, 1)[0]
        return round(slope, 4)
    except Exception:
        return np.nan