# pytest -v tests/simulator/test_broker.py
import pytest
import pandas as pd
from simulator.account import Account
from simulator.broker import Broker


@pytest.fixture
def account():
    """创建一个干净的测试账户"""
    return Account(initial_cash=100_000)


@pytest.fixture
def broker_rate(account):
    """按费率收费的券商"""
    return Broker(account, fee_type="rate", fee_rate=0.001, min_fee=1.0, slippage=0.0005)


@pytest.fixture
def broker_fixed(account):
    """按每股收费的券商"""
    return Broker(account, fee_type="fixed", fixed_fee=0.0035, min_fee=1.0, slippage=0.0005)


# ============================================================
# 1️⃣ 手续费计算 (fee_type="rate")
# ============================================================

def test_rate_fee_above_minimum(broker_rate):
    """测试 rate 模式手续费是否按比例计算且不低于 min_fee。"""
    fee = broker_rate._calc_fee(exec_price=100, qty=2000)  # 交易额 = 200,000
    expected = max(200_000 * 0.001, 1.0)
    assert fee == expected


def test_rate_fee_hits_minimum(broker_rate):
    """测试 rate 模式手续费触发最小值。"""
    fee = broker_rate._calc_fee(exec_price=10, qty=10)  # 交易额 = 100
    assert fee == 1.0   # 低于 min_fee 时应取 1.0


# ============================================================
# 2️⃣ 手续费计算 (fee_type="fixed")
# ============================================================

def test_fixed_fee_per_share(broker_fixed):
    """测试 fixed 模式按每股收费，且在上下限之间。"""
    # 每股 0.0035, 共 1000 股 = 3.5 USD
    fee = broker_fixed._calc_fee(exec_price=50, qty=1000)
    assert fee == pytest.approx(3.5, rel=1e-4)

def test_fixed_fee_minimum_applied(broker_fixed):
    """测试 fixed 模式触发最小收费。"""
    # 每股 0.0035, 共 100 股 = 0.35 < min_fee 1.0
    fee = broker_fixed._calc_fee(exec_price=50, qty=100)
    assert fee == 1.0

def test_fixed_fee_maximum_applied(broker_fixed):
    """测试 fixed 模式触发最高收费 (1% cap)。"""
    # 这里需要真正让每股费率 * 数量 > 1% cap
    # 比如价格低、股数极大时
    fee = broker_fixed._calc_fee(exec_price=1, qty=1_000_000)
    # trade_value = 1 * 1_000_000 = 1,000,000
    # per_share_fee = 1_000_000 * 0.0035 = 3,500
    # max_fee = 1% * 1,000,000 = 10,000 → 3,500 < 10,000 → 没触发
    # 我们要继续放大股数让 per_share_fee > max_fee：
    fee = broker_fixed._calc_fee(exec_price=1, qty=5_000_000)
    # trade_value = 5,000,000
    # per_share_fee = 17,500
    # max_fee = 50,000
    # 仍没超过 -> 再放大:
    fee = broker_fixed._calc_fee(exec_price=0.1, qty=2_000_000)
    # trade_value = 200,000
    # per_share_fee = 7,000
    # max_fee = 2,000 -> 触发 cap
    assert fee == pytest.approx(2_000.0, rel=1e-4)


# ============================================================
# 3️⃣ 滑点逻辑
# ============================================================

def test_apply_slippage_buy_and_sell(broker_rate):
    """测试买卖方向滑点是否正确应用。"""
    base_price = 100
    buy_price = broker_rate._apply_slippage("BUY", base_price)
    sell_price = broker_rate._apply_slippage("SELL", base_price)
    assert buy_price > base_price
    assert sell_price < base_price
    # 差值应约等于 base_price * slippage
    assert pytest.approx(buy_price - base_price, rel=1e-6) == base_price * broker_rate.slippage

def test_apply_slippage_invalid_side(broker_rate):
    """测试无效交易方向触发异常。"""
    with pytest.raises(ValueError):
        broker_rate._apply_slippage("HOLD", 100)


# ============================================================
# 4️⃣ 时区处理
# ============================================================

def test_to_ny_time_converts_naive_timestamp(broker_fixed):
    """测试时间戳被正确转为 America/New_York 时区。"""
    ts = broker_fixed._to_ny_time("2024-01-01 09:30")
    assert ts.tzinfo is not None
    assert ts.tz.zone in ["America/New_York", "US/Eastern"]

def test_to_ny_time_keeps_existing_tz(broker_fixed):
    """测试已有时区的时间戳被正确转换。"""
    ts = pd.Timestamp("2024-01-01 09:30", tz="UTC")
    converted = broker_fixed._to_ny_time(ts)
    assert converted.tz.zone == "America/New_York"


# ============================================================
# 5️⃣ execute_order 集成逻辑
# ============================================================

def test_execute_order_updates_account(broker_fixed, account):
    """测试 execute_order 是否调用 Account 并更新持仓。"""
    trade = broker_fixed.execute_order("2024-01-01 09:30", "BUY", 100, 100)
    assert "exec_price" in trade
    assert "fee" in trade
    assert account.position == 100
    assert account.cash < account.initial_cash  # 买入减少现金

def test_execute_order_sell(broker_fixed, account):
    """测试 SELL 正确更新持仓和现金。"""
    # 先买
    broker_fixed.execute_order("2024-01-01 09:30", "BUY", 100, 10)
    # 再卖
    trade = broker_fixed.execute_order("2024-01-01 10:00", "SELL", 110, 10)
    assert account.position == 0
    assert account.cash > account.initial_cash  # 卖出获利后现金增加
    assert trade["side"] == "SELL"

def test_execute_order_invalid_qty(broker_fixed):
    """测试下单数量非法时抛出异常。"""
    with pytest.raises(ValueError):
        broker_fixed.execute_order("2024-01-01 09:30", "BUY", 100, 0)
