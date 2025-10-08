"""
Unified import interface for all indicator categories.

You can import indicators either from a specific category:
    from features.indicators.trend import sma, ema, macd
or directly from this package:
    from features.indicators import sma, ema, macd, rsi, atr, zscore

Each submodule implements a family of indicators:
- trend.py          → SMA, EMA, MACD, ADX, Slope
- volatility.py     → ATR, StdDev, Bollinger Bands, HV, RSV
- momentum.py       → RSI, Momentum, ROC, CCI, Williams %R
- mean_reversion.py → Z-Score, Rolling Mean/Std, Cointegration, Hurst
- volume.py         → OBV, Volume MA, Volume Ratio, MFI
"""

# --- Trend-based indicators ---
from .trend import sma, ema, wma, macd, adx, slope

# --- Volatility-based indicators ---
from .volatility import atr, stddev, bollinger_bands, historical_volatility, rsv

# --- Momentum-based indicators ---
from .momentum import rsi, momentum, roc, stochastic_oscillator, cci, williams_r

# --- Mean-Reversion indicators ---
from .mean_reversion import zscore, rolling_mean_std, cointegration_test, hurst_exponent

# --- Volume-based indicators ---
from .volume import obv, volume_ma, volume_ratio, mfi

__all__ = [
    # Trend
    "sma", "ema", "wma", "macd", "adx", "slope",
    # Volatility
    "atr", "stddev", "bollinger_bands", "historical_volatility", "rsv",
    # Momentum
    "rsi", "momentum", "roc", "stochastic_oscillator", "cci", "williams_r",
    # Mean Reversion
    "zscore", "rolling_mean_std", "cointegration_test", "hurst_exponent",
    # Volume
    "obv", "volume_ma", "volume_ratio", "mfi"
]