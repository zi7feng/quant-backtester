import pandas as pd
import numpy as np
from features.indicators.trend import sma, ema, wma, macd, adx, slope


def test_sma_exact():
    """SMA should match the manual rolling mean calculation."""
    df = pd.DataFrame({"close": [1, 2, 3, 4, 5]})
    result = sma(df, "close", window=3).round(3).tolist()
    expected = [1.0, 1.5, 2.0, 3.0, 4.0]
    assert result == expected, f"Expected {expected}, got {result}"


def test_ema_exact():
    """EMA(3) should match the known recursive formula."""
    df = pd.DataFrame({"close": [1, 2, 3, 4, 5]})
    # α = 2/(3+1) = 0.5
    # EMA = [1.0, 1.5, 2.25, 3.125, 4.0625]
    result = ema(df, "close", span=3).round(4).tolist()
    expected = [1.0, 1.5, 2.25, 3.125, 4.0625]
    np.testing.assert_allclose(result, expected, rtol=1e-4)


def test_wma_exact():
    """WMA should correctly apply linear weights to the rolling window."""
    df = pd.DataFrame({"close": [1, 2, 3, 4, 5]})
    result = wma(df, "close", window=3).round(3).tolist()
    # Expected weighted averages:
    # (1*1+2*2+3*3)/6=2.33, (2*1+3*2+4*3)/6=3.33, (3*1+4*2+5*3)/6=4.33
    expected = [np.nan, np.nan, 2.333, 3.333, 4.333]
    np.testing.assert_allclose(result[2:], expected[2:], rtol=1e-3, equal_nan=True)


def test_macd_structure_and_range():
    """MACD output must include MACD, Signal, and Histogram columns."""
    df = pd.DataFrame({"close": np.linspace(1, 10, 20)})
    macd_df = macd(df, short_span=3, long_span=6, signal_span=3)

    assert set(["MACD", "Signal", "Histogram"]).issubset(macd_df.columns), \
        "MACD output missing expected columns"
    assert macd_df["MACD"].notna().all(), "MACD contains NaN values"
    assert macd_df["Signal"].notna().all(), "Signal contains NaN values"


def test_adx_range():
    """ADX values should always lie between 0 and 100."""
    df = pd.DataFrame({
        "high": [10, 11, 12, 13, 14, 15],
        "low": [9, 9.5, 10, 11, 12, 13],
        "close": [9.5, 10, 11, 12, 13, 14],
    })
    result = adx(df, window=3)
    assert (0 <= result.fillna(0)).all() and (result.fillna(0) <= 100).all(), \
        "ADX out of valid [0, 100] range"


def test_slope_exact_linear():
    """Slope of a perfectly linear trend (y=x) should be 1.0 for any window."""
    df = pd.DataFrame({"close": np.arange(1, 11)})
    result = slope(df, "close", window=5).dropna().round(3)
    assert np.allclose(result, 1.0, atol=1e-3), \
        f"Expected slope ≈ 1.0, got {result.tolist()}"
