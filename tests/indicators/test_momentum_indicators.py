import pandas as pd
import numpy as np
from features.indicators.momentum import (
    roc,
    momentum,
    rsi,
    stochastic_oscillator,
    cci,
    williams_r,
)


def test_roc_known_case():
    """
    Test Rate of Change (ROC) with a small dataset.
    For [10, 12, 15] and window=1:
        ROC = [(12/10 - 1)*100, (15/12 - 1)*100] = [20%, 25%].
    """
    df = pd.DataFrame({"close": [10, 12, 15]})
    result = roc(df, window=1).round(2).tolist()
    expected = [0, 20.0, 25.0]
    np.testing.assert_allclose(result, expected, rtol=1e-8, equal_nan=True)


def test_momentum_known_case():
    """
    Test Momentum = current price - price N periods ago.
    For [10, 12, 15] with window=1:
        Momentum = [NaN, 2, 3].
    """
    df = pd.DataFrame({"close": [10, 12, 15]})
    result = momentum(df, window=1).tolist()
    expected = [0, 2.0, 3.0]
    np.testing.assert_allclose(result, expected, rtol=1e-8, equal_nan=True)


def test_rsi_known_case():
    """
    Validate RSI calculation using a classic example from textbooks.
    For gains only: RSI should approach 100.
    For losses only: RSI should approach 0.
    For alternating prices: RSI should hover around 50.
    """
    df_up = pd.DataFrame({"close": np.arange(1, 16)})   # strictly increasing
    df_down = pd.DataFrame({"close": np.arange(15, 0, -1)})  # strictly decreasing
    df_alt = pd.DataFrame({"close": [1, 2, 1, 2, 1, 2, 1, 2]})

    rsi_up = rsi(df_up, window=14).iloc[-1]
    rsi_down = rsi(df_down, window=14).iloc[-1]
    rsi_alt = rsi(df_alt, window=2).iloc[-1]

    assert rsi_up > 90, f"Expected RSI > 90 for uptrend, got {rsi_up}"
    assert rsi_down < 10, f"Expected RSI < 10 for downtrend, got {rsi_down}"
    assert 40 <= rsi_alt <= 60, f"Expected RSI ~50 for oscillating prices, got {rsi_alt}"


def test_stochastic_oscillator_known_case():
    """
    Stochastic Oscillator: %K = (Close - LowestLow) / (HighestHigh - LowestLow) * 100
    For prices [10, 12, 14] with highs = lows = close:
        %K = 100 for all, %D = 100 as well.
    """
    df = pd.DataFrame({
        "high": [10, 12, 14],
        "low": [10, 12, 14],
        "close": [10, 12, 14],
    })
    result = stochastic_oscillator(df, k_window=3, d_window=2)
    assert np.allclose(result["%K"], 100.0), f"Expected %K=100, got {result['%K'].tolist()}"
    assert np.allclose(result["%D"], 100.0), f"Expected %D=100, got {result['%D'].tolist()}"


def test_cci_constant_data():
    """
    When all prices are constant, CCI should be 0.
    """
    df = pd.DataFrame({
        "high": [10] * 20,
        "low": [10] * 20,
        "close": [10] * 20,
    })
    result = cci(df, window=10).fillna(0)
    assert np.allclose(result, 0.0), f"Expected CCI=0, got {result.tolist()}"


def test_cci_linear_increase():
    """
    For linearly increasing prices, CCI should be positive.
    """
    df = pd.DataFrame({
        "high": np.arange(10, 30),
        "low": np.arange(9, 29),
        "close": np.arange(9.5, 29.5),
    })
    result = cci(df, window=10)
    assert result.iloc[-1] > 0, f"Expected positive CCI for uptrend, got {result.iloc[-1]}"


def test_williams_r_known_case():
    """
    Williams %R = -100 * (HighestHigh - Close) / (HighestHigh - LowestLow)
    For increasing prices, %R should approach 0.
    For decreasing prices, %R should approach -100.
    """
    df_up = pd.DataFrame({
        "high": np.arange(10, 20),
        "low": np.arange(9, 19),
        "close": np.arange(9.5, 19.5),
    })
    df_down = pd.DataFrame({
        "high": np.arange(20, 10, -1),
        "low": np.arange(19, 9, -1),
        "close": np.arange(19.5, 9.5, -1),
    })

    wr_up = williams_r(df_up, window=5).iloc[-1]
    wr_down = williams_r(df_down, window=5).iloc[-1]

    assert wr_up > -20, f"Expected %R close to 0 for uptrend, got {wr_up}"
    assert wr_down < -80, f"Expected %R close to -100 for downtrend, got {wr_down}"
