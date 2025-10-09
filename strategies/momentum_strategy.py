import pandas as pd
import numpy as np
from statistics import mean
from .base_strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """
    Intraday Momentum Breakout Strategy (rolling-compounding version)
    -----------------------------------------------------------------
    • Trades only within the same day (no overnight data)
    • Waits for at least `min_bars` of same-day data before activation
    • Stops opening new positions after 15:30 (3:30 PM)
    • Uses rolling mean breakout logic to trigger entries/exits
    • Position size adjusts dynamically based on available cash (rolling compounding)
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
        stop_loss: float = 0.008,
        fee_per_trade: float = 1.00,      # fixed $1 per trade
        safety_buffer: float = 0.005,     # safety margin to avoid full use of cash
    ):
        super().__init__(symbol, config or {}, timezone="America/New_York")
        self.window = window
        self.min_bars = min_bars
        self.buy_thresh = buy_thresh
        self.sell_thresh = sell_thresh
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.fee_per_trade = fee_per_trade
        self.safety_buffer = safety_buffer

        self.cutoff_hour = 15      # no new entry after 15:30
        self.cutoff_minute = 30
        self.force_close_hour = 15
        self.force_close_minute = 55

    # Not used in streaming mode
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

        # --- ① Restrict to same-day bars ---
        current_date = current_time.date()
        # 修正：使用 .to_series().dt.date 提取 DatetimeIndex 的日期部分
        today_bars = visible_bars[visible_bars.index.to_series().dt.date == current_date]
        if len(today_bars) < self.min_bars:
            return {"signal": "HOLD", "qty": 0}

        position = account_state["position"]
        avg_cost = account_state["avg_cost"]
        cash = account_state["cash"]

        # --- ② Rolling mean calculation ---
        closes = today_bars["close"].dropna().to_list()[-self.window:]
        if len(closes) == 0:
            return {"signal": "HOLD", "qty": 0}

        avg_price = mean(closes)
        price = today_bars.iloc[-1]["open"]

        # --- ③ Time cutoff control ---
        after_cutoff = (
            current_time.hour > self.cutoff_hour
            or (current_time.hour == self.cutoff_hour and current_time.minute >= self.cutoff_minute)
        )

        # --- ④ Manage existing position ---
        if position > 0:
            pnl_pct = (price - avg_cost) / avg_cost

            # Stop-loss / Take-profit logic
            if pnl_pct >= self.take_profit:
                return {"signal": "SELL", "qty": position}
            if pnl_pct <= -self.stop_loss:
                return {"signal": "SELL", "qty": position}

            # Fall below rolling mean
            if price < avg_price * (1 - self.sell_thresh):
                return {"signal": "SELL", "qty": position}

            # Force close before 15:55
            if current_time.hour == self.force_close_hour and current_time.minute >= self.force_close_minute:
                return {"signal": "SELL", "qty": position}

            return {"signal": "HOLD", "qty": 0}

        # --- ⑤ Entry logic when flat ---
        else:
            # stop new entries after cutoff
            if after_cutoff:
                return {"signal": "HOLD", "qty": 0}

            # breakout above rolling mean → BUY
            if price > avg_price * (1 + self.buy_thresh):
                # available capital after deducting one-time fee
                effective_cash = max(cash - self.fee_per_trade, 0)
                if effective_cash <= 0:
                    return {"signal": "HOLD", "qty": 0}

                # apply safety buffer
                max_price = price * (1 + self.safety_buffer)
                qty = int(effective_cash // max_price)

                if qty > 0:
                    return {"signal": "BUY", "qty": qty}

        return {"signal": "HOLD", "qty": 0}