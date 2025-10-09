# strategies/base_strategy.py
import abc
import pytz
import pandas as pd


class BaseStrategy(abc.ABC):
    """
    Abstract base class for all trading strategies.
    ------------------------------------------------
    Defines the required interface for both:
      • batch backtesting (generate_signals)
      • real-time / simulated streaming (on_data)
    """

    def __init__(self, symbol: str, config, timezone: str = "America/New_York"):
        self.symbol = symbol
        self.config = config
        self.tz = pytz.timezone(timezone)

    # --------------------------------------------------
    # Utility: make timestamp timezone-aware
    # --------------------------------------------------
    @staticmethod
    def _to_tz_aware(date_str, tz):
        """Convert date string to timezone-aware Timestamp."""
        if date_str is None:
            return None
        ts = pd.Timestamp(date_str)
        if ts.tzinfo is None:
            ts = tz.localize(ts)
        else:
            ts = ts.tz_convert(tz)
        return ts


    @abc.abstractmethod
    def generate_signals(self, *args, **kwargs):
        """
        Batch mode — generate trading signals from full dataset.
        Typically used in historical backtests (non-incremental).
        """
        raise NotImplementedError("Subclasses must implement generate_signals()")

    @abc.abstractmethod
    def on_data(self, current_time, visible_bars: pd.DataFrame, account_state: dict):
        """
        Streaming mode — handle new market data in real time.
        Called once per new bar by the Simulator.

        Parameters
        ----------
        current_time : pd.Timestamp
            Current simulation timestamp (the bar being processed).
        visible_bars : pd.DataFrame
            All bars visible to the strategy at this time:
              - fully completed bars up to (t - feed_delay)
              - open-only data for the last few minutes
        account_state : dict
            Current account summary with fields like:
              {'cash', 'position', 'avg_cost', 'equity', ...}

        Returns
        -------
        dict
            Example: {"signal": "BUY"/"SELL"/"HOLD", "qty": int}
        """
        raise NotImplementedError("Subclasses must implement on_data()")
