import pandas as pd
import numpy as np
from features.feature_loader import compute_indicators
from features.indicator_config import IndicatorConfig
from .base_strategy import BaseStrategy


class DemoStrategy(BaseStrategy):
    """
    Momentum-style example strategy
    --------------------------------
    Trading logic (executed at each minute's open):
      • At time t (the bar open):
          - We can see all fully closed bars up to t - feed_delay
          - Plus open prices of the last few bars (t-1, t)
      • Indicators (RSI, SMA) are computed on this augmented view
      • BUY when RSI < 30 and price > SMA
      • SELL when RSI > 70 or price < SMA
      • Only one position at a time (no pyramiding)
    """

    def __init__(self, symbol: str, config=None):
        self.safety_buffer = 0.002  # price cushion for execution
        config = config or IndicatorConfig(
            trend={"SMA": [50]},
            momentum={"RSI": [14]},
        )
        super().__init__(symbol, config, timezone="America/New_York")

    # --------------------------------------------------
    # Batch backtest (unused in streaming mode)
    # --------------------------------------------------
    def generate_signals(self, start_date=None, end_date=None, limit=None):
        raise NotImplementedError(
            "DemoStrategy is designed for on_data() simulation mode only."
        )

    # --------------------------------------------------
    # Streaming mode — one call per incoming bar
    # --------------------------------------------------
    def on_data(self, current_time, visible_bars: pd.DataFrame, account_state: dict):
        """
        Called once per simulated bar by the Simulator.

        Parameters
        ----------
        current_time : pd.Timestamp
            Current simulation timestamp.
        visible_bars : pd.DataFrame
            Market data visible to the strategy at this moment.
            Contains:
              - fully completed bars (t - feed_delay and earlier)
              - open-only records for recent bars (including current)
        account_state : dict
            {'cash', 'position', 'avg_cost', 'equity', ...}

        Returns
        -------
        dict
            Example: {"signal": "BUY"/"SELL"/"HOLD", "qty": int}
        """

        # --- Safety check: need enough lookback ---
        if len(visible_bars) < 50:
            return {"signal": "HOLD", "qty": 0}

        # --- Step 1: Compute technical indicators on visible data ---
        df = compute_indicators(visible_bars, self.config.to_dict())
        last_row = df.iloc[-1]

        # Extract latest indicator values
        rsi_val = last_row.get("RSI_14", np.nan)
        sma_val = last_row.get("SMA_50", np.nan)
        current_price = last_row.get("open", np.nan)

        cash = account_state["cash"]
        position = account_state["position"]

        # If indicators invalid (NaN) → skip
        if np.isnan(rsi_val) or np.isnan(sma_val) or np.isnan(current_price):
            return {"signal": "HOLD", "qty": 0}

        # --- Step 2: Decision logic ---
        # BUY condition
        if position == 0 and rsi_val < 30 and current_price > sma_val:
            # All-in with a safety buffer
            max_price = current_price * (1 + self.safety_buffer)
            qty = int(cash // max_price)
            if qty > 0:
                return {"signal": "BUY", "qty": qty}

        # SELL condition
        elif position > 0 and (rsi_val > 70 or current_price < sma_val):
            return {"signal": "SELL", "qty": position}

        # Otherwise HOLD
        return {"signal": "HOLD", "qty": 0}
