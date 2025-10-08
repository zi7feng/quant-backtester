import numpy as np
import pandas as pd
from features.indicators.mean_reversion import (
    rolling_mean_std,
    zscore,
    cointegration_test,
    hurst_exponent,
)


def test_rolling_mean_std_known_case():
    """
    For a simple series [1, 2, 3, 4, 5] with window=3:
    RollingMean = [1, 1.5, 2, 3, 4]
    RollingStd ≈ [0, 0.5, 0.816, 0.816, 0.816]
    """
    s = pd.Series([1, 2, 3, 4, 5])
    result = rolling_mean_std(s, window=3).round(3)

    expected_mean = [1.0, 1.5, 2.0, 3.0, 4.0]
    expected_std = [0.0, 0.5, 0.816, 0.816, 0.816]

    np.testing.assert_allclose(result["RollingMean"], expected_mean, rtol=1e-3)
    np.testing.assert_allclose(result["RollingStd"], expected_std, rtol=1e-3)


def test_zscore_behavior():
    """
    Validate Z-Score normalization:
    - For constant data, z-score should be 0.
    - For symmetric data, mean z-score should be ~0.
    """
    const = pd.Series([5] * 10)
    z_const = zscore(const, window=5)
    np.testing.assert_allclose(z_const.fillna(0), 0.0, atol=1e-8)

    # Use a stationary, symmetric series (e.g., sine wave)
    rand = pd.Series(np.sin(np.linspace(0, 10 * np.pi, 50)))
    z_rand = zscore(rand, window=10)
    assert abs(z_rand.mean()) < 1e-1, f"Z-score mean should be near 0, got {z_rand.mean()}"

def test_zscore_known_example():
    """
    For series [10, 20, 30] with window=3:
      Rolling z-scores = [0.0, 1.0, 1.22] (based on rolling mean and std with ddof=0).
    """
    s = pd.Series([10, 20, 30])
    z = zscore(s, window=3).round(2).tolist()
    expected = [0.0, 1.0, 1.22]
    np.testing.assert_allclose(z, expected, rtol=1e-8)


def test_cointegration_known_case():
    """
    If two series are perfectly correlated (y = 2x),
    the Engle-Granger test should give a very small p-value (<0.05).
    """
    np.random.seed(42)
    x = np.arange(1, 101)
    y = 2 * x + np.random.normal(0, 0.1, size=100)
    result = cointegration_test(pd.Series(x), pd.Series(y))
    assert result["p_value"] < 0.05, f"Expected strong cointegration, got p={result['p_value']:.4f}"


def test_cointegration_independent_series():
    """
    For independent random walk series, p-value should be large (>0.1).
    """
    np.random.seed(0)
    # Generate independent random walks (I(1) series)
    a = pd.Series(np.random.randn(100)).cumsum()
    b = pd.Series(np.random.randn(100)).cumsum()
    result = cointegration_test(a, b)
    assert result["p_value"] > 0.1, f"Expected non-cointegrated series, got p={result['p_value']:.4f}"


def test_hurst_exponent_random_walk():
    """
    For a pure random walk, H ≈ 0.5.
    """
    np.random.seed(1)
    rand_walk = np.cumsum(np.random.randn(1000))
    h = hurst_exponent(pd.Series(rand_walk))
    assert 0.4 < h < 0.6, f"Expected H ≈ 0.5 for random walk, got {h}"


def test_hurst_exponent_trending_series():
    """
    For a trending series (linear increase with noise), H > 0.5.
    """
    np.random.seed(0)
    trend = pd.Series(np.arange(1, 500) + np.random.randn(499) * 10)  # Linear trend with noise
    h = hurst_exponent(trend)
    assert h > 0.6, f"Expected H > 0.6 for trending series, got {h}"


def test_hurst_exponent_mean_reverting_series():
    """
    For a mean-reverting series (oscillating sinusoid), H < 0.5.
    """
    t = np.linspace(0, 50, 1000)
    s = pd.Series(np.sin(t))
    h = hurst_exponent(s)
    assert h < 0.5, f"Expected H < 0.5 for mean-reverting series, got {h}"
