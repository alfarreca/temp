import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Streamlit app title
st.title("📈 Weekly Price Tracker")

# File uploader
uploaded_file = st.file_uploader("Upload your Excel file", type="xlsx")

if uploaded_file:
    tickers_df = pd.read_excel(uploaded_file)

    if not all(col in tickers_df.columns for col in ["Symbol", "Exchange"]):
        st.error("Excel file must contain 'Symbol' and 'Exchange' columns.")
    else:
        symbols = tickers_df["Symbol"].tolist()

        # Calculate the most recent Monday to align weeks
        today = datetime.today()
        recent_monday = today - timedelta(days=today.weekday())

        # Fetch weekly data for the past 6 weeks
        start_date = recent_monday - timedelta(weeks=6)
        end_date = recent_monday

        data = {}
        for symbol in symbols:
            try:
                ticker_data = yf.download(symbol, start=start_date, end=end_date, interval="1wk")['Close']
                if len(ticker_data) < 6:
                    st.warning(f"Insufficient data for ticker {symbol}, skipping.")
                    continue
                data[symbol] = ticker_data[-6:].values
            except Exception as e:
                st.warning(f"Error fetching data for ticker {symbol}: {e}")

        # Ensure all arrays are 1-dimensional and have length 6
        valid_data = {k: v for k, v in data.items() if len(v) == 6}

        if valid_data:
            price_df = pd.DataFrame(valid_data).T
            price_df.columns = [(recent_monday - timedelta(weeks=i)).strftime('%Y-%m-%d') for i in reversed(range(6))]
            price_df.reset_index(inplace=True)
            price_df.rename(columns={'index': 'Symbol'}, inplace=True)

            st.subheader("Weekly Closing Prices")
            st.dataframe(price_df)
        else:
            st.error("No valid data fetched for the provided tickers.")
