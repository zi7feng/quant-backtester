import time
import random
import numpy as np
import pandas as pd
import pytz
import re
import importlib
from features.feature_loader import load_raw_data
from simulator.account import Account
from simulator.broker import Broker
from simulator.config.sim_config import load_sim_config


class Simulator:
    """
    Market Simulation Engine
    ------------------------
    Executes a strategy bar-by-bar with realistic visibility:
      • feed delay controls how many bars are fully visible
      • only open prices of recent bars are visible
      • optional volatility noise for execution
    """

    def __init__(self, start_date, end_date, config_path=None):
        cfg = load_sim_config(config_path)
        self.cfg = cfg
        self.start_date = start_date
        self.end_date = end_date

        # === General settings ===
        self.symbols = cfg.get("symbols", ["SPY.US"])
        self.strategy_name = cfg.get("strategy", {}).get("name", "DemoStrategy")
        self.feed_delay = cfg["env"].get("feed_delay_minutes", 1)
        self.latency_ms = cfg["env"].get("latency_ms", 200)
        self.volatility_factor = cfg["env"].get("volatility_factor", 0.1)
        self.random_seed = cfg["env"].get("random_seed", 42)

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.tz = pytz.timezone("America/New_York")
        self.results = {}

    # ------------------------------------------------------
    # Load strategy dynamically
    # ------------------------------------------------------
    def _load_strategy_class(self, strategy_name):
        module_name = re.sub(r"(?<!^)(?=[A-Z])", "_", strategy_name).lower()
        module = importlib.import_module(f"strategies.{module_name}")
        return getattr(module, strategy_name)

    # ------------------------------------------------------
    # Simulate execution price noise
    # ------------------------------------------------------
    def _simulate_exec_price(self, base_price: float) -> float:
        noise = np.random.randn() * self.volatility_factor / 100
        return base_price * (1 + noise)

    # ------------------------------------------------------
    # Run simulation for one symbol
    # ------------------------------------------------------
    def _run_single_symbol(self, symbol, strategy_cls):
        print(f"\n=== Running simulation for {symbol} ===")

        # --- Load raw data ---
        df = load_raw_data(symbol, start_date=self.start_date, end_date=self.end_date)
        assert df.index.is_monotonic_increasing, "Data index must be ascending!"
        print(f"Loaded {len(df):,} minute bars from DB.")
        print(f"Data range: {df.index[0]} → {df.index[-1]}")
        print(f"Feed delay = {self.feed_delay} min")

        # --- Initialize core components ---
        acc_cfg = self.cfg["account"]
        account = Account(
            initial_cash=acc_cfg["initial_cash"],
            currency=acc_cfg["currency"],
            allow_short=acc_cfg["allow_short"],
        )

        bro_cfg = self.cfg["broker"]
        broker = Broker(
            account,
            fee_type=bro_cfg["fee_type"],
            fee_rate=bro_cfg["fee_rate"],
            fixed_fee=bro_cfg["fixed_fee"],
            min_fee=bro_cfg["min_fee"],
            slippage=bro_cfg["slippage"],
        )

        strategy = strategy_cls(symbol)
        trades = []
        total_bars = len(df)
        delay = self.feed_delay

        print(f"\n🔍 Begin feeding bars sequentially...\n")

        # --- Main simulation loop ---
        for i in range(delay, total_bars):
            curr_bar = df.iloc[i]
            curr_time = df.index[i]

            # 1. Construct visible window (only full bars up to i - delay)
            visible_full = df.iloc[: i - delay + 1].copy()

            # 2. Append open-only data for the last (delay - 1) bars + current bar
            augmented = visible_full.copy()
            for j in range(delay - 1, -1, -1):
                idx = i - j
                if idx < 0:
                    continue
                ts = df.index[idx]
                open_price = df.iloc[idx]["open"]
                augmented.loc[ts] = {
                    "open": open_price,
                    "high": np.nan,
                    "low": np.nan,
                    "close": open_price,
                    "volume": np.nan,
                }

            augmented.sort_index(inplace=True)

            # 3. Debug print — show first few feed steps
            if i < delay + 5 or (i % 60 == 0):
                print("\n--------------------------------------------")
                print(f"FEED [{i}/{total_bars}] | Current time = {curr_time}")
                print(
                    f"Open={curr_bar['open']:.2f}, High={curr_bar['high']:.2f}, "
                    f"Low={curr_bar['low']:.2f}, Close={curr_bar['close']:.2f}"
                )
                print(
                    f"👁 Visible window: {augmented.index[0]} → {augmented.index[-1]} "
                    f"({len(augmented)} rows)"
                )
                print("👁 Last 5 visible opens:")
                print(augmented["open"].tail(5).to_string())

            # 4. Strategy decision based on visible data
            account_state = account.summary()
            signal_data = strategy.on_data(
                current_time=curr_time,
                visible_bars=augmented,
                account_state=account_state,
            )
            signal = signal_data.get("signal", "HOLD")
            qty = signal_data.get("qty", 0)

            # Debug: print signal
            if i < delay + 5 or signal != "HOLD":
                print(f"⚙️ Strategy output: {signal_data}")
                print(
                    f"💰 Account: cash={account_state['cash']:.2f}, "
                    f"pos={account_state['position']}, equity={account_state['equity']:.2f}"
                )

            # 5. Execute trade if needed
            if signal in ["BUY", "SELL"] and qty > 0:
                exec_price = self._simulate_exec_price(curr_bar["open"])
                trade = broker.execute_order(
                    timestamp=curr_time,
                    side=signal,
                    price=exec_price,
                    qty=qty,
                )
                if trade:
                    trades.append(trade)
                    print(
                        f" Trade executed at {curr_time}, signal={signal}, "
                        f"qty={qty}, price={exec_price:.2f}"
                    )

            # 6. Update account equity
            account.update_equity(curr_bar["close"], curr_time)

            # 7. Progress display
            if total_bars >= 100 and i % (total_bars // 10) == 0:
                pct = (i / total_bars) * 100
                print(f"  Progress: {pct:.1f}% ({i}/{total_bars})")

        # --- Simulation finished ---
        print(f"\nCompleted simulation for {symbol}")
        summary = account.summary()
        summary["trades"] = len(trades)
        print("Account summary:", summary)
        print("----------------------------------------------------")

        # Store result
        self.results[symbol] = {
            "account": account,
            "broker": broker,
            "strategy": strategy,
            "summary": summary,
            "trades": pd.DataFrame(trades),
        }

    # ------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------
    def run(self):
        strategy_cls = self._load_strategy_class(self.strategy_name)
        print(f"\nStarting simulation with strategy = {self.strategy_name}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Feed delay = {self.feed_delay} min, latency = {self.latency_ms} ms")

        for sym in self.symbols:
            self._run_single_symbol(sym, strategy_cls)

        print("\nAll simulations completed.")
        return self.results

    # ------------------------------------------------------
    # Combined report
    # ------------------------------------------------------
    def report(self):
        print("\n=== Simulation Summary ===")
        summaries = []
        for sym, res in self.results.items():
            s = res["summary"]
            print(
                f"{sym}: Equity={s['equity']}, PnL={s['realized_pnl']}, Trades={s['trades']}"
            )
            summaries.append(
                {
                    "Symbol": sym,
                    "Equity": s["equity"],
                    "PnL": s["realized_pnl"],
                    "Trades": s["trades"],
                }
            )
        return pd.DataFrame(summaries)
