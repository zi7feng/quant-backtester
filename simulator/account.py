import pandas as pd
import pytz


class Account:
    """
    Simulated Trading Account (v2.0)
    --------------------------------
    Tracks:
        - cash balance
        - position (quantity)
        - average cost
        - realized & unrealized PnL
        - total equity (mark-to-market)
    """

    def __init__(self,
                 initial_cash: float = 100_000.0,
                 currency: str = "USD",
                 allow_short: bool = False):
        self.currency = currency
        self.allow_short = allow_short
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0
        self.avg_cost = 0.0
        self.realized_pnl = 0.0
        self.trades = []         # trade logs
        self.equity_curve = []   # [(timestamp, equity)]
        self.tz = pytz.timezone("America/New_York")

    # -----------------------------
    # Internal time normalization
    # -----------------------------
    def _to_ny_time(self, ts):
        """Ensure timestamp is timezone-aware in America/New_York."""
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = self.tz.localize(ts)
        else:
            ts = ts.tz_convert(self.tz)
        return ts

    # -----------------------------
    # Trade recording
    # -----------------------------
    def record_trade(self, timestamp, side: str, price: float, qty: int, fee: float):
        """
        Record a trade, update cash/position and realized PnL.

        Parameters
        ----------
        timestamp : datetime or str
        side : str ("BUY" or "SELL")
        price : float
        qty : int
        fee : float
        """
        timestamp = self._to_ny_time(timestamp)
        side = side.upper()

        if qty <= 0:
            raise ValueError("Trade quantity must be positive.")

        if side == "BUY":
            cost = price * qty + fee
            new_position = self.position + qty

            # Prevent margin violation if shorting not allowed
            if self.cash < cost and not self.allow_short:
                raise ValueError("Insufficient cash for BUY trade.")

            # Weighted average price
            self.avg_cost = (
                (self.avg_cost * self.position + price * qty) / new_position
                if new_position != 0 else 0.0
            )

            self.position = new_position
            self.cash -= cost

        elif side == "SELL":
            new_position = self.position - qty

            # Prevent short selling if not allowed
            if new_position < 0 and not self.allow_short:
                raise ValueError("Short selling not allowed.")

            proceeds = price * qty - fee
            realized = (price - self.avg_cost) * min(qty, self.position)
            self.realized_pnl += realized
            self.cash += proceeds
            self.position = new_position

            if self.position == 0:
                self.avg_cost = 0.0

        else:
            raise ValueError(f"Invalid trade side: {side}")

        self.trades.append({
            "timestamp": timestamp,
            "side": side,
            "price": round(price, 4),
            "qty": qty,
            "fee": round(fee, 4),
            "cash": round(self.cash, 2),
            "position": self.position,
            "avg_cost": round(self.avg_cost, 4),
            "realized_pnl": round(self.realized_pnl, 2)
        })

    # -----------------------------
    # Mark-to-market equity
    # -----------------------------
    def update_equity(self, current_price: float, timestamp=None):
        """Mark-to-market update: equity = cash + position * current_price."""
        timestamp = self._to_ny_time(timestamp) if timestamp else None
        unrealized = (current_price - self.avg_cost) * self.position
        total_equity = self.cash + self.position * current_price

        if timestamp is not None:
            self.equity_curve.append((timestamp, total_equity))

        return {
            "timestamp": timestamp,
            "equity": total_equity,
            "unrealized": unrealized,
            "realized": self.realized_pnl
        }

    # -----------------------------
    # Summary and reset
    # -----------------------------
    def summary(self):
        """Return a snapshot of account state."""
        current_equity = self.equity_curve[-1][1] if self.equity_curve else self.cash
        return {
            "cash": round(self.cash, 2),
            "position": self.position,
            "avg_cost": round(self.avg_cost, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "equity": round(current_equity, 2),
            "currency": self.currency
        }

    def reset(self):
        """Reset account to initial state."""
        self.cash = self.initial_cash
        self.position = 0
        self.avg_cost = 0.0
        self.realized_pnl = 0.0
        self.trades.clear()
        self.equity_curve.clear()

    # -----------------------------
    # Export helpers
    # -----------------------------
    def get_equity_df(self):
        """Return equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame(columns=["Datetime", "Equity"])
        return pd.DataFrame(self.equity_curve, columns=["Datetime", "Equity"]).set_index("Datetime")

    def trades_df(self):
        """Return trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame(columns=["timestamp", "side", "price", "qty", "fee", "cash", "position"])
        return pd.DataFrame(self.trades)
