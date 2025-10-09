import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from pathlib import Path

# 导入正确的模块
import simulator.simulator as sim_mod
from simulator.config.sim_config import load_sim_config
from strategies.base_strategy import BaseStrategy


# ==============================
# 构造一个可控的 DummyStrategy
# ==============================
class DummyStrategy(BaseStrategy):
    """
    受控策略：
    - 在 Day1 09:32 首次触发时做断言：检查 visible_bars 的“可视窗口”是否正确
    - 在 Day1 09:32 返回 BUY 2 手
    - 在 Day1 09:35 返回 SELL 全部
    - 在 Day2 09:32 验证窗口重置（不含 Day1 数据）
    - 其它时刻 HOLD
    """

    def __init__(self, symbol, config=None, timezone="America/New_York"):
        super().__init__(symbol, config or {}, timezone)
        self._bought = False
        self._asserted_window_ok = False
        self._asserted_day2_reset = False

    def generate_signals(self, *args, **kwargs):
        """Dummy method for abstract interface — not used in this test."""
        return None

    def on_data(self, current_time, visible_bars: pd.DataFrame, account_state: dict):
        ny = pytz.timezone("America/New_York")
        probe_time_buy = ny.localize(pd.Timestamp("2024-01-02 09:32")).to_pydatetime()
        probe_time_sell = ny.localize(pd.Timestamp("2024-01-02 09:35")).to_pydatetime()
        day2_start = ny.localize(pd.Timestamp("2024-01-03 09:32")).to_pydatetime()

        # 动态获取 feed_delay（默认 2）
        feed_delay = self.config.get("env", {}).get("feed_delay_minutes", 2)

        # —— 可视窗口断言
        if not self._asserted_window_ok and current_time == probe_time_buy:
            assert isinstance(visible_bars.index, pd.DatetimeIndex)
            assert visible_bars.index.tz.zone == "America/New_York"
            assert visible_bars.index[-1] == current_time
            assert visible_bars.index.date.max() == current_time.date()

            tail = visible_bars.tail(feed_delay)
            head = visible_bars.iloc[:-feed_delay] if len(visible_bars) > feed_delay else pd.DataFrame()

            # ✅ 检查尾部 K 线：只有 open 有值
            if not tail.empty:
                assert tail["open"].notna().all(), "open must exist in delayed bars"
                assert tail["high"].isna().all(), "high must be NaN in delayed bars"
                assert tail["low"].isna().all(), "low must be NaN in delayed bars"
                assert tail["close"].isna().all(), "close must be NaN in delayed bars"
                assert tail["volume"].isna().all(), "volume must be NaN in delayed bars"

            # ✅ 检查前面完整部分：所有值完整
            if not head.empty:
                assert head[["open", "high", "low", "close", "volume"]].notna().all().all()

            self._asserted_window_ok = True

        # —— 验证 Day2 窗口重置
        if not self._asserted_day2_reset and current_time == day2_start:
            assert visible_bars.index.date.max() == current_time.date()
            self._asserted_day2_reset = True

        # —— 交易逻辑
        if (not self._bought) and current_time == probe_time_buy and account_state["position"] == 0:
            self._bought = True
            return {"signal": "BUY", "qty": 2}
        if self._bought and current_time == probe_time_sell and account_state["position"] > 0:
            return {"signal": "SELL", "qty": account_state["position"]}
        return {"signal": "HOLD", "qty": 0}


# ==============================
# 工具：构造两天的分钟线
# ==============================
def _make_two_days_df():
    tz = "America/New_York"
    day1 = pd.date_range("2024-01-02 09:30", periods=6, freq="min", tz=tz)
    day2 = pd.date_range("2024-01-03 09:30", periods=6, freq="min", tz=tz)
    idx = day1.append(day2)
    df = pd.DataFrame(index=idx)
    df["open"] = np.arange(100, 100 + len(idx)).astype(float)
    df["high"] = df["open"] + 1
    df["low"] = df["open"] - 1
    df["close"] = df["open"] + 0.5
    df["volume"] = 1000
    return df


# ==============================
# Fixtures & monkeypatch
# ==============================
@pytest.fixture
def patched_config(monkeypatch):
    cfg = {
        "account": {
            "initial_cash": 100000,
            "currency": "USD",
            "allow_short": False,
        },
        "broker": {
            "fee_type": "fixed",
            "fee_rate": 0.001,
            "fixed_fee": 0.0035,
            "min_fee": 1.0,
            "slippage": 0.0,
        },
        "env": {
            "feed_delay_minutes": 2,
            "latency_ms": 0,
            "volatility_factor": 0.0,
            "random_seed": 37,
        },
        "symbols": ["TEST.US"],
        "strategy": {"name": "DummyStrategy"},
    }
    # ✅ 正确的 monkeypatch 目标
    monkeypatch.setattr(sim_mod, "load_sim_config", lambda path=None: cfg)
    return cfg


@pytest.fixture
def patched_strategy_loader(monkeypatch):
    monkeypatch.setattr(sim_mod.Simulator, "_load_strategy_class", lambda self, name: DummyStrategy)


@pytest.fixture
def patched_chunk_loader(monkeypatch):
    df = _make_two_days_df()

    def _fake_gen(symbol, start_date, end_date, days_per_chunk=5):
        yield df

    monkeypatch.setattr(sim_mod, "load_raw_data_chunked", _fake_gen)


@pytest.fixture
def patched_perf(monkeypatch):
    def _fake_analyze(trades_df, equity_df):
        start = equity_df["Equity"].iloc[0] if not equity_df.empty else 100000.0
        end = equity_df["Equity"].iloc[-1] if not equity_df.empty else start
        return {
            "Symbol": "TEST.US",
            "StartEquity": round(start, 2),
            "EndEquity": round(end, 2),
            "TotalReturn(%)": round((end / start - 1) * 100, 4) if start else 0.0,
            "WinRate(%)": 0.0,
            "AvgWin": 0.0,
            "AvgLoss": 0.0,
            "ProfitFactor": 0.0,
            "SharpeRatio": 0.0,
            "MaxDrawdown(%)": 0.0,
            "TotalFees": float(trades_df["fee"].sum()) if not trades_df.empty else 0.0,
            "FeeAsPctOfPnL(%)": 0.0,
            "Trades": len(trades_df),
        }

    monkeypatch.setattr(sim_mod, "analyze_performance", _fake_analyze)


# ==============================
# 核心测试：端到端 + 可视窗口断言
# ==============================
def test_simulator_e2e_window_and_trades(
    patched_config, patched_strategy_loader, patched_chunk_loader, patched_perf, tmp_path
):
    sim = sim_mod.Simulator(
        start_date="2024-01-02 09:30:00",
        end_date="2024-01-03 09:35:00",
        config_path=None,
        days_per_chunk=5,
    )
    results = sim.run()
    assert "TEST.US" in results
    res = results["TEST.US"]
    account = res["account"]
    trades_df = res["trades"]
    equity_df = account.get_equity_df()

    # 验证交易
    assert len(trades_df) == 2
    buy_row = trades_df.iloc[0]
    sell_row = trades_df.iloc[1]
    assert buy_row["side"].upper() == "BUY"
    assert buy_row["qty"] == 2
    assert buy_row["timestamp"] == pd.Timestamp("2024-01-02 09:32", tz="America/New_York")
    assert np.allclose(
        buy_row["exec_price"],
        _make_two_days_df().loc[pd.Timestamp("2024-01-02 09:32", tz="America/New_York")]["open"],
    )
    assert sell_row["side"].upper() == "SELL"
    assert sell_row["qty"] == 2
    assert sell_row["timestamp"] == pd.Timestamp("2024-01-02 09:35", tz="America/New_York")
    assert np.allclose(
        sell_row["exec_price"],
        _make_two_days_df().loc[pd.Timestamp("2024-01-02 09:35", tz="America/New_York")]["open"],
    )
    assert trades_df["fee"].notna().all()
    assert all(trades_df["fee"] >= 1.0)
    assert account.position == 0

    # 验证 equity 曲线
    assert not equity_df.empty
    assert list(equity_df.columns) == ["Equity"]
    assert equity_df.index.isin(_make_two_days_df().index).all()
    assert np.allclose(equity_df["Equity"].iloc[0], 100000.0)

    # 验证报告
    rep = sim.report(output_dir=tmp_path)
    assert isinstance(rep, pd.DataFrame)
    assert "Symbol" in rep.columns
    assert "TotalReturn(%)" in rep.columns
    assert "TotalFees" in rep.columns
    assert rep.loc[rep["Symbol"] == "TEST.US", "Trades"].iloc[0] == 2
    summary_csv = tmp_path / "summary_report.csv"
    symbol_equity_csv = tmp_path / "TEST.US_equity.csv"
    assert summary_csv.exists()
    assert symbol_equity_csv.exists()


# ==============================
# 测试：多 chunk / 跨日重置
# ==============================
def test_simulator_daily_reset_with_multichunk(patched_config, patched_strategy_loader, monkeypatch):
    full = _make_two_days_df()
    day1_mask = full.index.date == pd.Timestamp("2024-01-02", tz="America/New_York").date()
    day2_mask = full.index.date == pd.Timestamp("2024-01-03", tz="America/New_York").date()
    df_day1 = full.loc[day1_mask]
    df_day2 = full.loc[day2_mask]

    def _gen(symbol, start_date, end_date, days_per_chunk=1):
        yield df_day1
        yield df_day2

    monkeypatch.setattr(sim_mod, "load_raw_data_chunked", _gen)

    sim = sim_mod.Simulator(
        start_date="2024-01-02 09:30:00",
        end_date="2024-01-03 09:35:00",
        config_path=None,
        days_per_chunk=1,
    )
    results = sim.run()
    assert "TEST.US" in results
    assert len(results["TEST.US"]["trades"]) == 2
    assert results["TEST.US"]["account"].position == 0


# ==============================
# 测试：空数据处理
# ==============================
def test_empty_data_handling(patched_config, patched_strategy_loader, monkeypatch):
    def _empty_gen(symbol, start_date, end_date, days_per_chunk=5):
        yield pd.DataFrame()

    monkeypatch.setattr(sim_mod, "load_raw_data_chunked", _empty_gen)

    sim = sim_mod.Simulator(
        start_date="2024-01-02 09:30:00",
        end_date="2024-01-03 09:35:00",
        config_path=None,
        days_per_chunk=5,
    )
    results = sim.run()
    assert "TEST.US" in results
    assert results["TEST.US"]["trades"].empty
    assert results["TEST.US"]["account"].position == 0
    assert np.allclose(results["TEST.US"]["account"].summary()["cash"], 100000.0)


# ==============================
# 测试：无效日期
# ==============================
def test_invalid_dates_raises_error(patched_config, patched_strategy_loader):
    with pytest.raises(ValueError):
        sim_mod.Simulator(
            start_date="2024-01-03 09:30:00",
            end_date="2024-01-02 09:30:00",
            config_path=None,
        )


# ==============================
# 测试：不同 feed_delay（有效值 1 和 2）
# ==============================
@pytest.mark.parametrize("feed_delay", [1, 2])
def test_feed_delay_variations(
    patched_config, patched_strategy_loader, patched_chunk_loader, patched_perf, tmp_path, feed_delay
):
    # 更新配置
    patched_config["env"]["feed_delay_minutes"] = feed_delay

    sim = sim_mod.Simulator(
        start_date="2024-01-02 09:30:00",
        end_date="2024-01-03 09:35:00",
        config_path=None,
        days_per_chunk=5,
    )

    results = sim.run()

    # 断言：symbol 正确
    assert "TEST.US" in results

    # 验证交易次数一致（买入一次、卖出一次）
    trades_df = results["TEST.US"]["trades"]
    assert len(trades_df) == 2
    assert set(trades_df["side"].str.upper()) == {"BUY", "SELL"}
    assert results["TEST.US"]["account"].position == 0


# ==============================
# 测试：多 symbol 支持
# ==============================
def test_multiple_symbols(patched_config, patched_strategy_loader, patched_chunk_loader, patched_perf, tmp_path):
    patched_config["symbols"] = ["TEST.US", "TEST2.US"]
    sim = sim_mod.Simulator(
        start_date="2024-01-02 09:30:00",
        end_date="2024-01-03 09:35:00",
        config_path=None,
        days_per_chunk=5,
    )
    results = sim.run()
    assert set(results.keys()) == {"TEST.US", "TEST2.US"}
    for sym in results:
        assert len(results[sym]["trades"]) == 2
        assert results[sym]["account"].position == 0
    rep = sim.report(output_dir=tmp_path)
    assert set(rep["Symbol"]) == {"TEST.US", "TEST2.US"}
