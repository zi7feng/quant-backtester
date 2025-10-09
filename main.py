from data.data_fetcher import fetch_and_store_symbol

def main():

    # Fetch SPY data from 2015 - 2025 with interval 1min
    symbol = "BMNR.US"
    interval = "1m"

    print("Starting data fetch pipeline...")
    fetch_and_store_symbol(
        symbol=symbol,
        interval=interval,
        from_date="2025-06-01",
        to_date="2025-10-01",
        force_refresh=True
    )
    print("Pipeline finished.")

if __name__ == "__main__":
    main()
