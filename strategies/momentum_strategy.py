import pandas as pd
import numpy as np
from statistics import mean
from .base_strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """
    Intraday Momentum Breakout Strategy (with trading time control)
    ---------------------------------------------------------------
    • Trades only within the same day (no overnight data)
    • Waits for at least `min_bars` of same-day data before activation
    • Stops opening new positions after 15:30 (3:30 PM)
    • Uses rolling mean breakout logic to trigger entries/exits
    """

    def __init__(
        self,
        symbol: str,
        config=None,
        window: int = 30,
        min_bars: int = 20,
        buy_thresh: float = 0.01,
        sell_thresh: float = 0.008,
        take_profit: float = 0.012,
        stop_loss: float = 0.01,
    ):
        super().__init__(symbol, config or {}, timezone="America/New_York")
        self.window = window
        self.min_bars = min_bars
        self.buy_thresh = buy_thresh
        self.sell_thresh = sell_thresh
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.safety_buffer = 0.005
        self.cutoff_hour = 15      # 15:30 之后不再开新仓
        self.cutoff_minute = 30

    def generate_signals(self, *args, **kwargs):
        raise NotImplementedError("MomentumStrategy supports on_data() only.")

    def on_data(
        self,
        current_time: pd.Timestamp,
        visible_bars: pd.DataFrame,
        account_state: dict,
    ):
        """Main decision logic executed each bar."""
        if visible_bars.empty:
            return {"signal": "HOLD", "qty": 0}

        # --- ① Filter to same-day data ---
        current_date = current_time.date()
        today_bars = visible_bars[visible_bars.index.date == current_date]
        if len(today_bars) < self.min_bars:
            return {"signal": "HOLD", "qty": 0}

        position = account_state["position"]
        avg_cost = account_state["avg_cost"]
        cash = account_state["cash"]

        # --- ② Compute rolling mean from today's closes ---
        closes = today_bars["close"].dropna().to_list()[-self.window :]
        if len(closes) == 0:
            return {"signal": "HOLD", "qty": 0}

        avg_price = mean(closes)
        price = today_bars.iloc[-1]["open"]

        # --- ③ Time restriction: stop opening new positions after 15:30 ---
        after_cutoff = (
            current_time.hour > self.cutoff_hour
            or (current_time.hour == self.cutoff_hour and current_time.minute >= self.cutoff_minute)
        )

        # --- ④ Manage existing position ---
        if position > 0:
            pnl_pct = (price - avg_cost) / avg_cost

            # Take-profit / Stop-loss
            if pnl_pct >= self.take_profit or pnl_pct <= -self.stop_loss:
                return {"signal": "SELL", "qty": position}

            # Breakdown below rolling mean
            if price < avg_price * (1 - self.sell_thresh):
                return {"signal": "SELL", "qty": position}

            # Optional: force close all before 16:00
            if current_time.hour == 15 and current_time.minute >= 55:
                return {"signal": "SELL", "qty": position}

            return {"signal": "HOLD", "qty": 0}

        # --- ⑤ Entry logic when flat ---
        else:
            # 🚫 No new BUY after cutoff time
            if after_cutoff:
                return {"signal": "HOLD", "qty": 0}

            # Breakout above mean → BUY
            if price > avg_price * (1 + self.buy_thresh):
                max_price = price * (1 + self.safety_buffer)
                qty = int(cash // max_price)
                if qty > 0:
                    return {"signal": "BUY", "qty": qty}

        return {"signal": "HOLD", "qty": 0}
