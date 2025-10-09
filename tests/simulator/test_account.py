# tests/simulator/test_account
import pytest
import pandas as pd
from simulator.account import Account


@pytest.fixture
def account():
    """创建一个新的测试账户（默认初始资金 $100,000）。"""
    return Account(initial_cash=100_000)


# ============================================================
# 1️⃣ 基础买入逻辑
# ============================================================

def test_buy_reduces_cash_and_increases_position(account):
    """测试 BUY 交易是否正确减少现金、增加持仓并更新均价。"""
    account.record_trade("2024-01-01 09:30", "BUY", price=100, qty=10, fee=1.0)
    # 成本 = 100 * 10 + 1 = 1001
    assert pytest.approx(account.cash, rel=1e-4) == 100_000 - 1001
    assert account.position == 10
    assert account.avg_cost == 100.0
    assert account.realized_pnl == 0.0


# ============================================================
# 2️⃣ 连续买入（加仓）逻辑
# ============================================================

def test_weighted_avg_cost_after_multiple_buys(account):
    """测试连续买入时加权平均成本是否正确。"""
    account.record_trade("2024-01-01 09:30", "BUY", 100, 10, 1.0)   # 均价 = 100
    account.record_trade("2024-01-01 10:00", "BUY", 120, 10, 1.0)   # 均价应 = 110
    assert account.position == 20
    assert round(account.avg_cost, 2) == 110.00


# ============================================================
# 3️⃣ 卖出逻辑（部分平仓）
# ============================================================

def test_partial_sell_updates_pnl_and_cash(account):
    """测试部分卖出时现金、持仓和已实现盈亏的变化。"""
    account.record_trade("2024-01-01 09:30", "BUY", 100, 10, 1.0)
    account.record_trade("2024-01-01 10:00", "SELL", 110, 5, 1.0)
    # 每股盈利 10 * 5 = 50
    assert account.position == 5
    assert round(account.realized_pnl, 2) == 50.0
    # 现金增加：110*5 - 1 = 549
    expected_cash = 100_000 - 1001 + 549
    assert pytest.approx(account.cash, rel=1e-4) == expected_cash


# ============================================================
# 4️⃣ 清仓（全部卖出）
# ============================================================

def test_full_sell_resets_avg_cost(account):
    """测试全部卖出后均价是否归零。"""
    account.record_trade("2024-01-01 09:30", "BUY", 100, 10, 1.0)
    account.record_trade("2024-01-01 10:00", "SELL", 110, 10, 1.0)
    assert account.position == 0
    assert account.avg_cost == 0.0
    assert account.realized_pnl == 100.0
    # 现金恢复 + 盈利 - 手续费
    assert account.cash > account.initial_cash


# ============================================================
# 5️⃣ 手续费计算影响
# ============================================================

def test_fee_reduces_cash_and_pnl(account):
    """测试手续费对现金与盈亏的影响是否正确。"""
    account.record_trade("2024-01-01 09:30", "BUY", 100, 10, 5.0)
    account.record_trade("2024-01-01 10:00", "SELL", 105, 10, 5.0)
    # 总手续费 10，盈利 (105-100)*10=50
    # 净利润 40
    assert round(account.realized_pnl, 2) == 50.0
    total_fees = 10.0
    net_cash = account.cash - account.initial_cash
    # 检查总现金变化约等于净利润 - 手续费
    assert pytest.approx(net_cash, rel=1e-4) == 40.0


# ============================================================
# 6️⃣ 禁止做空
# ============================================================

def test_short_not_allowed_raises_error(account):
    """测试未启用 allow_short 时禁止做空。"""
    account.record_trade("2024-01-01 09:30", "BUY", 100, 10, 1.0)
    with pytest.raises(ValueError):
        account.record_trade("2024-01-01 10:00", "SELL", 110, 20, 1.0)  # 卖多于持仓


# ============================================================
# 7️⃣ 允许做空
# ============================================================

def test_allow_short_sell_passes():
    """测试允许做空时能正常建立负仓位。"""
    acc = Account(initial_cash=100_000, allow_short=True)
    acc.record_trade("2024-01-01 09:30", "SELL", 100, 10, 1.0)
    assert acc.position == -10
    assert acc.cash > acc.initial_cash  # 收到卖出资金


# ============================================================
# 8️⃣ update_equity 逻辑
# ============================================================

def test_update_equity_reflects_unrealized_pnl(account):
    """测试未实现盈亏是否正确计算。"""
    account.record_trade("2024-01-01 09:30", "BUY", 100, 10, 1.0)
    result = account.update_equity(current_price=105, timestamp="2024-01-01 10:00")
    assert round(result["unrealized"], 2) == 50.0
    assert "equity" in result
    # equity 应 ≈ 现金 + 1050
    expected_equity = account.cash + 1050
    assert pytest.approx(result["equity"], rel=1e-4) == expected_equity


# ============================================================
# 9️⃣ reset() 功能
# ============================================================

def test_reset_clears_all_positions(account):
    """测试 reset() 是否完全清空账户。"""
    account.record_trade("2024-01-01 09:30", "BUY", 100, 10, 1.0)
    account.update_equity(105, "2024-01-01 10:00")
    account.reset()

    assert account.cash == account.initial_cash
    assert account.position == 0
    assert account.avg_cost == 0.0
    assert account.realized_pnl == 0.0
    assert account.trades == []
    assert account.equity_curve == []
