import time
import random
import numpy as np
import pandas as pd
import pytz
import re
import importlib
import os
from pathlib import Path
from simulator.utils.perf_metrics import analyze_performance
from features.feature_loader import load_raw_data_chunked
from simulator.account import Account
from simulator.broker import Broker
from simulator.config.sim_config import load_sim_config
from datetime import datetime



class Simulator:
    """
    Market Simulation Engine (Optimized with Profiling + Chunk Loading)
    ------------------------------------------------------------------
    • Reads data in chunks to save memory
    • Profiles per-phase timing (data, window, strategy, broker, equity)
    • Resets window daily (no overnight data mix)
    • Uses slicing instead of copying for speed
    """

    def __init__(self, start_date, end_date, config_path=None, days_per_chunk: int = 5):
        cfg = load_sim_config(config_path)
        self.cfg = cfg

        # 🧭 统一处理 start_date 和 end_date 类型
        self.tz = pytz.timezone("America/New_York")

        def _to_dt(x):
            if isinstance(x, str):
                return self.tz.localize(datetime.fromisoformat(x))
            elif isinstance(x, pd.Timestamp):
                return x.tz_convert(self.tz)
            elif isinstance(x, datetime):
                return x if x.tzinfo else self.tz.localize(x)
            else:
                raise TypeError(f"Unsupported date type: {type(x)}")

        self.start_date = _to_dt(start_date)
        self.end_date = _to_dt(end_date)
        self.days_per_chunk = days_per_chunk

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
    def _load_strategy_class(self, strategy_name):
        module_name = re.sub(r"(?<!^)(?=[A-Z])", "_", strategy_name).lower()
        module = importlib.import_module(f"strategies.{module_name}")
        return getattr(module, strategy_name)

    # ------------------------------------------------------
    def _simulate_exec_price(self, base_price: float) -> float:
        """Simulate slight random slippage in execution price"""
        noise = np.random.randn() * self.volatility_factor / 100
        return base_price * (1 + noise)

    # ------------------------------------------------------
    def _run_single_symbol(self, symbol, strategy_cls):
        print(f"\n=== Running simulation for {symbol} ===")
        t0 = time.perf_counter()

        # --- Init components ---
        t_init_start = time.perf_counter()
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
        t_init_end = time.perf_counter()

        # --- Profiling accumulators ---
        total_rows = 0
        t_load_sum = 0.0
        t_window_sum = 0.0
        t_strategy_sum = 0.0
        t_broker_sum = 0.0
        t_equity_sum = 0.0

        print(f"\n🔍 Begin streaming chunks (each {self.days_per_chunk} days)...\n")

        # --- Process each chunk ---
        for df_chunk in load_raw_data_chunked(symbol, self.start_date, self.end_date, days_per_chunk=self.days_per_chunk):
            t_chunk_start = time.perf_counter()
            df = df_chunk
            total_rows += len(df)
            delay = self.feed_delay
            total_bars = len(df)

            # --- Loop within chunk ---
            for i in range(delay, total_bars):
                curr_bar = df.iloc[i]
                curr_time = df.index[i]
                curr_date = curr_time.date()

                # =============== Phase 1: Window construction ===============
                t1 = time.perf_counter()
                mask_today = df.index.date == curr_date
                df_today = df.loc[mask_today]
                i_today = df_today.index.get_loc(curr_time)
                if i_today < delay:
                    continue

                # slicing for visible window
                visible_full = df_today.iloc[: i_today - delay + 1]
                recent = df_today.iloc[i_today - delay + 1:i_today + 1].copy()
                recent[["high", "low", "close", "volume"]] = np.nan
                recent["close"] = recent["open"]
                augmented = pd.concat([visible_full, recent])
                t2 = time.perf_counter()
                t_window_sum += (t2 - t1)

                # =============== Phase 2: Strategy decision ===============
                t3 = time.perf_counter()
                account_state = account.summary()
                signal_data = strategy.on_data(
                    current_time=curr_time,
                    visible_bars=augmented,
                    account_state=account_state,
                )
                signal = signal_data.get("signal", "HOLD")
                qty = signal_data.get("qty", 0)
                t4 = time.perf_counter()
                t_strategy_sum += (t4 - t3)

                # =============== Phase 3: Trade execution ===============
                t5 = time.perf_counter()
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
                t6 = time.perf_counter()
                t_broker_sum += (t6 - t5)

                # =============== Phase 4: Equity update ===============
                t7 = time.perf_counter()
                account.update_equity(curr_bar["close"], curr_time)
                t8 = time.perf_counter()
                t_equity_sum += (t8 - t7)

                # Progress log every ~3000 bars
                if i % 3000 == 0:
                    elapsed = time.perf_counter() - t_chunk_start
                    print(f"  ⏱ {symbol} | {i}/{total_bars} in chunk | Elapsed {elapsed:.1f}s")

            t_chunk_end = time.perf_counter()
            t_load_sum += (t_chunk_end - t_chunk_start)
            print(f"✅ Processed chunk ({len(df)} rows) in {(t_chunk_end - t_chunk_start):.2f}s")

        # --- Summary ---
        total_time = time.perf_counter() - t0
        print(f"\n✅ Completed simulation for {symbol}")
        print(f"Total runtime: {total_time:.2f}s ({total_rows:,} rows total)")
        print("Breakdown:")
        print(f"  ⚙️  Init setup:   {(t_init_end - t_init_start):.3f}s")
        print(f"  🪟 Window build:  {t_window_sum:.3f}s")
        print(f"  🧠 Strategy:      {t_strategy_sum:.3f}s")
        print(f"  💸 Broker exec:   {t_broker_sum:.3f}s")
        print(f"  📈 Equity update: {t_equity_sum:.3f}s")
        print(f"  ⏩ Chunk total:    {t_load_sum:.3f}s")
        print(f"----------------------------------------------------")

        summary = account.summary()
        summary["trades"] = len(trades)
        self.results[symbol] = {
            "account": account,
            "broker": broker,
            "strategy": strategy,
            "summary": summary,
            "trades": pd.DataFrame(trades),
        }

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
    def report(self, output_dir="reports"):
        """
        Enhanced simulation report with full metrics.
        """
        Path(output_dir).mkdir(exist_ok=True)
        print("\n=== Simulation Summary ===")

        all_results = []
        for sym, res in self.results.items():
            account = res["account"]
            trades_df = account.trades_df()
            equity_df = account.get_equity_df()
            metrics = analyze_performance(trades_df, equity_df)
            metrics["Symbol"] = sym
            all_results.append(metrics)

            # 输出到控制台
            print(f"\n📈 {sym} Summary")
            for k, v in metrics.items():
                if k != "Symbol":
                    print(f"   {k:<20}: {v}")

            # 保存单个 equity 曲线
            equity_path = Path(output_dir) / f"{sym}_equity.csv"
            equity_df.to_csv(equity_path)
            print(f"   → Equity curve saved to {equity_path}")

        # 汇总所有 symbol 的指标
        df_report = pd.DataFrame(all_results)
        report_path = Path(output_dir) / "summary_report.csv"
        df_report.to_csv(report_path, index=False)
        print(f"\n✅ Summary report saved: {report_path}")

        return df_report
