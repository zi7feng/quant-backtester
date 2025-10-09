import pytest
import pandas as pd
import pytz
from datetime import datetime, timedelta

import features.feature_loader as fl


# ============================================================
# 1️⃣ Mock classes
# ============================================================

class DummyCandle:
    """模拟数据库 Candle 对象"""
    def __init__(self, dt, o, h, l, c, v, symbol="TEST.US"):
        self.datetime = dt
        self.open = o
        self.high = h
        self.low = l
        self.close = c
        self.volume = v
        self.symbol = symbol


class DummyQuery:
    """模拟 SQLAlchemy Query 链式调用"""
    def __init__(self, data):
        self._data = data

    def filter(self, *args, **kwargs):  # 返回自身即可支持链式调用
        return self

    def order_by(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def all(self):
        return self._data


class DummySession:
    """模拟 SessionLocal()"""
    def __init__(self, rows):
        self._rows = rows

    def query(self, *_args, **_kwargs):
        return DummyQuery(self._rows)

    def close(self):
        pass


# ============================================================
# 2️⃣ Fixtures
# ============================================================

@pytest.fixture
def dummy_data():
    tz = pytz.UTC
    now = datetime(2024, 1, 1, tzinfo=tz)
    return [
        DummyCandle(now + timedelta(minutes=i), 100 + i, 101 + i, 99 + i, 100.5 + i, 1000 + i)
        for i in range(5)
    ]


@pytest.fixture
def patch_session(monkeypatch, dummy_data):
    """用 DummySession 替换真实数据库会话"""
    def _fake_session():
        return DummySession(dummy_data)
    monkeypatch.setattr(fl, "SessionLocal", _fake_session)
    return _fake_session


# ============================================================
# 3️⃣ Tests for load_raw_data
# ============================================================

def test_load_raw_data_returns_dataframe(patch_session):
    """验证 load_raw_data 能返回正确的 DataFrame"""
    df = fl.load_raw_data("TEST.US")
    assert isinstance(df, pd.DataFrame)
    assert set(["open", "high", "low", "close", "volume"]).issubset(df.columns)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz.zone == "America/New_York"
    assert len(df) == 5


# ============================================================
# 4️⃣ Tests for load_raw_data_chunked
# ============================================================

def test_load_raw_data_chunked_yields_dataframe(monkeypatch, dummy_data):
    """验证 chunk 加载能正确迭代"""
    def _fake_session():
        return DummySession(dummy_data)
    monkeypatch.setattr(fl, "SessionLocal", _fake_session)

    start = dummy_data[0].datetime
    end = dummy_data[-1].datetime
    chunks = list(fl.load_raw_data_chunked("TEST.US", start, end, days_per_chunk=2))

    assert len(chunks) >= 1
    for df in chunks:
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz.zone == "America/New_York"


# ============================================================
# 5️⃣ Tests for compute_indicators
# ============================================================

def test_compute_indicators_adds_columns():
    """验证指标计算能正确添加列"""
    df = pd.DataFrame({
        "open": [100, 101, 102, 103, 104],
        "high": [101, 102, 103, 104, 105],
        "low": [99, 100, 101, 102, 103],
        "close": [100, 101, 102, 103, 104],
        "volume": [1000, 1001, 1002, 1003, 1004]
    }, index=pd.date_range("2024-01-01", periods=5, freq="min", tz="America/New_York"))

    cfg = {
        "trend": {"SMA": [2], "EMA": [2]},
        "momentum": {"RSI": [2]},
        "volatility": {"Bollinger": {"window": 2, "num_std": 2}},
        "mean_reversion": {"ZScore": [2]}
    }

    df_feat = fl.compute_indicators(df, cfg)
    # 确保添加了新列
    assert any("SMA_" in c for c in df_feat.columns)
    assert any("RSI_" in c for c in df_feat.columns)
    assert "ZScore_2" in df_feat.columns


# ============================================================
# 6️⃣ Tests for load_features (integration, isolated)
# ============================================================

def test_load_features_integrated(monkeypatch, patch_session):
    """集成测试：mock 掉 load_raw_data，验证整体能运行"""
    def fake_load_raw_data(symbol, start_date=None, end_date=None, limit=None):
        idx = pd.date_range("2024-01-01", periods=5, freq="min", tz="America/New_York")
        df = pd.DataFrame({
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100, 101, 102, 103, 104],
            "volume": [1000, 1001, 1002, 1003, 1004]
        }, index=idx)
        return df

    monkeypatch.setattr(fl, "load_raw_data", fake_load_raw_data)

    cfg = {
        "trend": {"SMA": [2]},
        "momentum": {"RSI": [2]},
    }
    df_feat = fl.load_features("TEST.US", config=cfg)
    assert isinstance(df_feat, pd.DataFrame)
    assert any("SMA_" in c for c in df_feat.columns)
