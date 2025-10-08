from features.indicator_config import IndicatorConfig
from features.feature_loader import load_features
import pandas as pd

def test_feature_loader_integration():
    """Test full feature loader with database-backed data."""
    cfg = IndicatorConfig()
    df = load_features("SPY.US", config=cfg, buffer=0.1, limit=500)

    assert not df.empty, "Feature loader returned empty DataFrame"
    assert isinstance(df, pd.DataFrame), "Output should be a DataFrame"
    assert "SMA_20" in df.columns, "SMA_20 missing in output"
    assert "RSI_14" in df.columns, "RSI_14 missing in output"
    assert "ATR_14" in df.columns, "ATR_14 missing in output"
