import numpy as np
import pandas as pd

def analyze_performance(trades_df: pd.DataFrame, equity_df: pd.DataFrame):
    """
    Compute performance metrics from trade logs and equity curve.
    Returns a dict of metrics.
    """

    if trades_df.empty or equity_df.empty:
        return {}

    # --- 基础指标 ---
    start_equity = equity_df["Equity"].iloc[0]
    end_equity = equity_df["Equity"].iloc[-1]
    total_return = (end_equity / start_equity - 1) * 100

    # --- 计算逐日收益率 ---
    equity_df["Return"] = equity_df["Equity"].pct_change().fillna(0)
    mean_ret = equity_df["Return"].mean()
    std_ret = equity_df["Return"].std()
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret != 0 else 0  # 年化夏普比率

    # --- 最大回撤 ---
    cummax = equity_df["Equity"].cummax()
    drawdown = equity_df["Equity"] / cummax - 1
    max_dd = drawdown.min() * 100  # 百分比

    # --- 盈亏统计 ---
    closed_trades = trades_df.copy()
    # 用 realized pnl 或 cash 变化判断每笔盈亏
    pnl_series = closed_trades["realized_pnl"].diff().fillna(0)
    pnl_series = pnl_series[pnl_series != 0]

    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series < 0]
    win_rate = len(wins) / max(len(pnl_series), 1) * 100
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    profit_factor = abs(wins.sum() / losses.sum()) if len(losses) > 0 else np.inf

    # --- 手续费统计 ---
    total_fees = closed_trades["fee"].sum()
    gross_pnl = pnl_series.sum()
    fee_ratio = total_fees / abs(gross_pnl) * 100 if gross_pnl != 0 else 0

    return {
        "StartEquity": round(start_equity, 2),
        "EndEquity": round(end_equity, 2),
        "TotalReturn(%)": round(total_return, 2),
        "WinRate(%)": round(win_rate, 2),
        "AvgWin": round(avg_win, 2),
        "AvgLoss": round(avg_loss, 2),
        "ProfitFactor": round(profit_factor, 2),
        "SharpeRatio": round(sharpe, 3),
        "MaxDrawdown(%)": round(max_dd, 2),
        "TotalFees": round(total_fees, 2),
        "FeeAsPctOfPnL(%)": round(fee_ratio, 2),
        "Trades": len(trades_df)
    }
