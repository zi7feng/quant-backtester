import os
import sys
import pandas as pd
import streamlit as st
from sqlalchemy import text

# ----------------------------------
# python -m scripts.price_chart
# ----------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings
from config.db_config import engine


# ----------------------------------
# Streamlit
# ----------------------------------
st.set_page_config(page_title="SPY Price Viewer", layout="wide")
st.title("📈 SPY Price Viewer (Daily first open from 1-min data)")
st.caption("Visualize the daily open price from database, restricted to 09:30–16:00 EST regular trading hours.")


# ----------------------------------
# Input
# ----------------------------------
symbol = st.text_input("Symbol:", "SPY.US")
start_date = st.date_input("Start Date", pd.Timestamp("2015-05-01"))
end_date = st.date_input("End Date", pd.Timestamp("2025-05-01"))

# ----------------------------------
# Query
# ----------------------------------
if st.button("Load Data"):
    with engine.connect() as conn:
        # Step 1: select data between 09:30–16:00
        # Step 2: first data as open
        query = text("""
            SELECT c.datetime, c.open
            FROM candles c
            JOIN (
                SELECT DATE(datetime AT TIME ZONE 'America/New_York') AS day,
                       MIN(datetime) AS first_dt
                FROM candles
                WHERE symbol = :symbol
                  AND datetime BETWEEN :start AND :end
                  AND (
                    (EXTRACT(HOUR FROM datetime AT TIME ZONE 'America/New_York') = 9
                     AND EXTRACT(MINUTE FROM datetime AT TIME ZONE 'America/New_York') >= 30)
                    OR EXTRACT(HOUR FROM datetime AT TIME ZONE 'America/New_York') BETWEEN 10 AND 15
                  )
                GROUP BY DATE(datetime AT TIME ZONE 'America/New_York')
            ) sub ON c.datetime = sub.first_dt
            WHERE c.symbol = :symbol
            ORDER BY c.datetime ASC
        """)

        df = pd.read_sql_query(
            query,
            conn,
            params={"symbol": symbol, "start": start_date, "end": end_date}
        )

    # ----------------------------------
    # show data
    # ----------------------------------
    if df.empty:
        st.warning("No data found in this range.")
    else:
        # convert to pandas datetime（remain UTC）
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime")

        # convert to newyork time
        df.index = df.index.tz_convert("America/New_York")

        st.subheader(f"Daily first open price for {symbol}")
        st.line_chart(df["open"])
        st.dataframe(df.tail(10))

        # show statistics info
        st.markdown(f"""
        **Total days:** {len(df)}  
        **First record:** {df.index[0]}  
        **Last record:** {df.index[-1]}
        """)

