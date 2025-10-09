# python -m scripts.run_strategy_demo


from strategies.demo_strategy import DemoStrategy

if __name__ == "__main__":
    strategy = DemoStrategy("SPY.US")
    signals = strategy.generate_signals(
        start_date="2025-04-01",
        end_date="2025-04-30"
    )
    print(signals.tail())
    print("Signal distribution:")
    print(signals["Signal"].value_counts())
    print("\nSample BUY/SELL signals:")
    print(signals[signals["Signal"] != "HOLD"].head(10))

