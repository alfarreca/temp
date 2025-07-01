import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np

st.title("📈 Weekly Price Tracker")

uploaded_file = st.file_uploader("Upload your Excel file", type="xlsx")

def fetch_weekly_closes(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end, interval="1wk")['Close']
        closes = data.values
        closes = closes[~np.isnan(closes)]  # Remove NaN if any
        if closes.ndim != 1:
            return None  # Not 1D
        if len(closes) < 6:
            return None
        return closes[-6:]
    except Exception as e:
        return None

if uploaded_file:
    tickers_df = pd.read_excel(uploaded_file)

    if not all(col in tickers_df.columns for col in ["Symbol", "Exchange"]):
        st.error("Excel file must contain 'Symbol' and 'Exchange' columns.")
    else:
        symbols = tickers_df["Symbol"].tolist()

        today = datetime.today()
        recent_monday = today - timedelta(days=today.weekday())
        start_date = recent_monday - timedelta(weeks=6)
        end_date = recent_monday

        result = {}
        for symbol in symbols:
            closes = fetch_weekly_closes(symbol, start_date, end_date)
            if closes is not None and len(closes) == 6:
                result[symbol] = closes
            else:
                st.warning(f"Ticker {symbol}: Could not fetch 6 weeks of valid closing prices. Skipped.")

        if result:
            price_df = pd.DataFrame(result).T
            price_df.columns = [
                (recent_monday - timedelta(weeks=i)).strftime('%Y-%m-%d')
                for i in reversed(range(6))
            ]
            price_df.reset_index(inplace=True)
            price_df.rename(columns={'index': 'Symbol'}, inplace=True)
            st.subheader("Weekly Closing Prices")
            st.dataframe(price_df)
        else:
            st.error("No valid data fetched for the provided tickers.")
