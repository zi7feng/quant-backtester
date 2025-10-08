import pandas as pd
import numpy as np
from features.indicators.volatility import atr, stddev, bollinger_bands, historical_volatility, rsv


def test_atr_known_case():
    """
    Test ATR with a known 4-day dataset.
    True Range (TR) = max(high - low, abs(high - prev_close), abs(low - prev_close))
    ATR = rolling mean of TR.
    """
    df = pd.DataFrame({
        "high": [48.7, 48.72, 48.9, 48.87],
        "low": [47.79, 48.14, 48.39, 48.37],
        "close": [48.16, 48.61, 48.75, 48.63],
    })
    result = atr(df, window=3).round(3).iloc[-1]
    # Expected ATR should be around 0.4–0.6 according to TR examples from textbooks
    assert 0.4 <= result <= 0.6, f"ATR expected between 0.4–0.6, got {result}"


def test_stddev_exact_match():
    """
    Test rolling standard deviation against numpy std.
    The result should match np.std(x, ddof=0) for the same window.
    """
    df = pd.DataFrame({"close": [1, 2, 3, 4, 5]})
    result = stddev(df, "close", window=5).iloc[-1]
    expected = np.std([1, 2, 3, 4, 5], ddof=0)
    np.testing.assert_allclose(result, expected, rtol=1e-8)


def test_bollinger_bands_constant_data():
    """
    If prices are constant, all Bollinger Bands (upper, middle, lower) should be equal.
    """
    df = pd.DataFrame({"close": [100] * 10})
    bands = bollinger_bands(df, window=5)
    np.testing.assert_allclose(bands["Upper"], 100.0)
    np.testing.assert_allclose(bands["Middle"], 100.0)
    np.testing.assert_allclose(bands["Lower"], 100.0)


def test_bollinger_bands_expected_values():
    """
    Validate Bollinger Bands calculation for a simple increasing price series.
    The middle band should equal SMA, and the upper/lower should be SMA ± 2σ.
    """
    df = pd.DataFrame({"close": [1, 2, 3, 4, 5]})
    bands = bollinger_bands(df, window=5, num_std=2.0)
    sma = df["close"].rolling(window=5, min_periods=1).mean()
    std = df["close"].rolling(window=5, min_periods=1).std(ddof=0).fillna(0)  # Use ddof=0 and fill NaN
    expected_upper = sma + 2 * std
    expected_lower = sma - 2 * std

    np.testing.assert_allclose(bands["Upper"], expected_upper, rtol=1e-8)
    np.testing.assert_allclose(bands["Lower"], expected_lower, rtol=1e-8)
    np.testing.assert_allclose(bands["Middle"], sma, rtol=1e-8)


def test_historical_volatility_deterministic_case():
    """
    Test historical volatility on a deterministic geometric series (1, 2, 4, 8, 16).
    The log returns are constant, so HV should converge to 0.
    """
    df = pd.DataFrame({"close": [1, 2, 4, 8, 16]})
    hv = historical_volatility(df, window=5, trading_days=252)
    assert hv.iloc[-1] < 1e-10, f"Expected near-zero HV, got {hv.iloc[-1]}"


def test_rsv_constant_data():
    """
    Relative Std Volatility (σ / mean) for constant data should be 0.
    """
    df = pd.DataFrame({"close": [5.0] * 10})
    result = rsv(df, "close", window=5)
    np.testing.assert_allclose(result, 0.0)


def test_rsv_known_case():
    """
    RSV should equal std / mean over rolling window.
    Example: [1, 2, 3, 4, 5] window=5 → std=1.414, mean=3 → RSV ≈ 0.471.
    """
    df = pd.DataFrame({"close": [1, 2, 3, 4, 5]})
    result = rsv(df, "close", window=5).iloc[-1]
    expected = np.std([1, 2, 3, 4, 5], ddof=0) / np.mean([1, 2, 3, 4, 5])
    np.testing.assert_allclose(result, expected, rtol=1e-8)
