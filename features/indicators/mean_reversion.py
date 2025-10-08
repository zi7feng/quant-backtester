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
    """Compute rolling mean and std of a series."""
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std()

    return pd.DataFrame({
        "RollingMean": rolling_mean,
        "RollingStd": rolling_std
    })


def zscore(series: pd.Series, window: int = 30) -> pd.Series:
    """Z-Score = (price - rolling mean) / rolling std."""
    mean = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std()
    return (series - mean) / (std + 1e-10)


def cointegration_test(series_a: pd.Series, series_b: pd.Series) -> dict:
    """
    Engle-Granger Cointegration Test between two assets.
    Returns p-value, t-statistic, and critical values.
    """
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


def hurst_exponent(series: pd.Series, lags: list = [2, 4, 8, 16, 32, 64]) -> float:
    """
    Estimate Hurst Exponent (H) to detect trend vs mean reversion.
    H < 0.5: mean-reverting
    H = 0.5: random walk
    H > 0.5: trending
    """
    tau = []
    lag_vals = []

    for lag in lags:
        diff = series.diff(lag).dropna()
        tau.append(np.sqrt(np.std(diff)))
        lag_vals.append(lag)

    reg = np.polyfit(np.log(lag_vals), np.log(tau), 1)
    hurst = reg[0] * 2.0
    return round(hurst, 4)
