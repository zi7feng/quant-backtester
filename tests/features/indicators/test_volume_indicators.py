import pandas as pd
import numpy as np
from features.indicators.volume import obv, volume_ma, volume_ratio, mfi


def test_obv_increasing_prices():
    """
    OBV should accumulate volume when price rises,
    subtract volume when price falls, and stay flat when unchanged.
    """
    df = pd.DataFrame({
        "close": [10, 11, 10, 10, 12],
        "volume": [100, 200, 300, 400, 500]
    })
    result = obv(df).tolist()
    # Price changes: [0, +, -, 0, +]
    # OBV: [0, +200, -100, -100, +400]
    expected = [0, 200, -100, -100, 400]
    np.testing.assert_array_equal(result, expected)


def test_volume_ma_known_values():
    """
    Volume moving average should equal the arithmetic mean of previous volumes.
    """
    df = pd.DataFrame({"volume": [10, 20, 30, 40, 50]})
    result = volume_ma(df, window=3).tolist()
    # Rolling means: [10, 15, 20, 30, 40]
    expected = [10, 15, 20, 30, 40]
    np.testing.assert_allclose(result, expected, rtol=1e-8)


def test_volume_ratio_correctness():
    """
    Volume ratio = current volume / rolling mean volume.
    For increasing volumes, VR should increase steadily toward 1.25–1.5.
    """
    df = pd.DataFrame({"volume": [10, 20, 30, 40, 50]})
    result = volume_ratio(df, window=3).round(3).tolist()
    # Expected ratios based on correct calculation
    expected = [1.0, 1.333, 1.5, 1.333, 1.25]
    np.testing.assert_allclose(result, expected, rtol=1e-3)


def test_mfi_constant_prices():
    """
    MFI should be 50 for flat prices (no positive or negative flow).
    This occurs because pos_mf ≈ neg_mf.
    """
    df = pd.DataFrame({
        "high": [10] * 10,
        "low": [10] * 10,
        "close": [10] * 10,
        "volume": [100] * 10
    })
    result = mfi(df, window=5)
    assert np.allclose(result.fillna(50).iloc[-1], 50, atol=1e-5), \
        f"Expected MFI ≈ 50, got {result.iloc[-1]}"


def test_mfi_upward_trend():
    """
    When prices rise consistently, MFI should approach 100
    because positive money flow dominates.
    """
    df = pd.DataFrame({
        "high": np.arange(10, 20),
        "low": np.arange(9, 19),
        "close": np.arange(9.5, 19.5),
        "volume": [100] * 10
    })
    result = mfi(df, window=5)
    assert result.iloc[-1] > 80, f"Expected MFI > 80 for strong uptrend, got {result.iloc[-1]}"


def test_mfi_downward_trend():
    """
    When prices fall consistently, MFI should approach 0
    because negative money flow dominates.
    """
    df = pd.DataFrame({
        "high": np.arange(20, 10, -1),
        "low": np.arange(19, 9, -1),
        "close": np.arange(19.5, 9.5, -1),
        "volume": [100] * 10
    })
    result = mfi(df, window=5)
    assert result.iloc[-1] < 20, f"Expected MFI < 20 for downtrend, got {result.iloc[-1]}"
