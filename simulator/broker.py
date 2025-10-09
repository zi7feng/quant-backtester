import pandas as pd
import pytz

class Broker:
    """
    Simulated Broker
    ----------------
    Handles order execution and transaction costs.

    Features:
    - Supports 'rate' or 'fixed' fee models
    - Supports minimum fee (only for 'rate' mode)
    - Supports slippage (price deviation simulation)
    - Updates connected Account after execution
    - Auto-localizes timestamps to America/New_York (UTC-4)
    """

    def __init__(self, account,
                 fee_type: str = "rate",
                 fee_rate: float = 0.001,
                 fixed_fee: float = 1.0,
                 min_fee: float = 1.0,
                 slippage: float = 0.0005):
        self.account = account
        self.fee_type = fee_type.lower()
        self.fee_rate = fee_rate
        self.fixed_fee = fixed_fee
        self.min_fee = min_fee
        self.slippage = slippage
        self.tz = pytz.timezone("America/New_York")

    # --------------------------
    # Fee computation
    # --------------------------
    def _calc_fee(self, exec_price: float, qty: int) -> float:
        """Compute transaction fee based on configuration."""
        if self.fee_type == "rate":
            fee = exec_price * qty * self.fee_rate
            return max(fee, self.min_fee)
        elif self.fee_type == "fixed":
            trade_value = exec_price * qty
            per_share_fee = qty * self.fixed_fee
            min_fee = 1.00
            max_fee = trade_value * 0.01
            fee = max(min_fee, min(per_share_fee, max_fee))
            return fee
        else:
            raise ValueError(f"Invalid fee_type '{self.fee_type}' — must be 'rate' or 'fixed'.")

    # --------------------------
    # Slippage adjustment
    # --------------------------
    def _apply_slippage(self, side: str, price: float) -> float:
        """Adjust price based on slippage model."""
        side = side.upper()
        if side == "BUY":
            return price * (1 + self.slippage)
        elif side == "SELL":
            return price * (1 - self.slippage)
        else:
            raise ValueError(f"Invalid side '{side}' — must be BUY or SELL.")

    # --------------------------
    # Timezone normalization
    # --------------------------
    def _to_ny_time(self, ts):
        """Ensure timestamp is timezone-aware in America/New_York."""
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = self.tz.localize(ts)
        else:
            ts = ts.tz_convert(self.tz)
        return ts

    # --------------------------
    # Core execution
    # --------------------------
    def execute_order(self, timestamp, side: str, price: float, qty: int):
        """
        Execute a trade and update Account.
        Timestamps automatically normalized to America/New_York.
        """
        if qty <= 0:
            raise ValueError("Quantity must be positive.")

        # --- Normalize time ---
        timestamp = self._to_ny_time(timestamp)

        # --- Adjust price for slippage ---
        exec_price = self._apply_slippage(side, price)

        # --- Calculate fee ---
        fee = self._calc_fee(exec_price, qty)

        # --- Record trade in Account ---
        self.account.record_trade(
            timestamp=timestamp,
            side=side,
            price=exec_price,
            qty=qty,
            fee=fee
        )

        # --- Log ---
        print(f"[{timestamp}] {side:<4} {qty:>4} @ {exec_price:.2f} | Fee={fee:.2f}")

        return {
            "timestamp": timestamp,
            "side": side,
            "qty": qty,
            "exec_price": exec_price,
            "fee": fee
        }
