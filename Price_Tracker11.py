import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

st.title("📈 Weekly Price Tracker")

uploaded_file = st.file_uploader("Upload your Excel file", type="xlsx")

def get_last_n_fridays(n):
    today = datetime.today()
    # Go back to the most recent Friday
    offset = (today.weekday() - 4) % 7
    last_friday = today - timedelta(days=offset)
    return [(last_friday - timedelta(weeks=i)).strftime('%Y-%m-%d') for i in reversed(range(n))]

def fetch_weekly_closes(symbol, n_weeks=6):
    today = datetime.today()
    # Go back to the most recent Friday
    offset = (today.weekday() - 4) % 7
    last_friday = today - timedelta(days=offset)
    start = last_friday - timedelta(weeks=n_weeks)
    end = last_friday + timedelta(days=1)  # So that we include last Friday in download
    try:
        data = yf.download(symbol, start=start, end=end, interval="1wk")['Close']
        closes = data.values
        closes = closes[~np.isnan(closes)]
        # If more than needed, just take the most recent n_weeks
        return closes[-n_weeks:] if len(closes) >= n_weeks else None
    except Exception as e:
        return None

if uploaded_file:
    tickers_df = pd.read_excel(uploaded_file)

    if not all(col in tickers_df.columns for col in ["Symbol", "Exchange"]):
        st.error("Excel file must contain 'Symbol' and 'Exchange' columns.")
    else:
        symbols = tickers_df["Symbol"].tolist()

        week_labels = get_last_n_fridays(6)

        result = {}
        for symbol in symbols:
            closes = fetch_weekly_closes(symbol, n_weeks=6)
            if closes is not None and len(closes) == 6:
                result[symbol] = closes
            else:
                st.warning(f"Ticker {symbol}: Could not fetch 6 weeks of valid Friday closing prices. Skipped.")

        if result:
            price_df = pd.DataFrame(result).T
            price_df.columns = week_labels
            price_df.reset_index(inplace=True)
            price_df.rename(columns={'index': 'Symbol'}, inplace=True)
            st.subheader("Weekly Closing Prices (Fridays)")
            st.dataframe(price_df)

            # Calculate % price change week over week
            pct_change_df = price_df.set_index("Symbol")[week_labels].pct_change(axis=1) * 100
            pct_change_df = pct_change_df.iloc[:, 1:]  # remove first week (no prior week)
            pct_change_df = pct_change_df.round(2)
            pct_change_df.reset_index(inplace=True)
            pct_change_df.columns = ["Symbol"] + [
                f"% Change {week_labels[i-1]} to {week_labels[i]}"
                for i in range(1, 6)
            ]
            st.subheader("Weekly % Price Change")
            st.dataframe(pct_change_df)

            # Chart: Line plot of weekly closes
            st.subheader("Price Trend Chart")
            ticker_options = price_df["Symbol"].tolist()
            tickers_to_plot = st.multiselect(
                "Select tickers to plot", ticker_options, default=ticker_options[:min(3, len(ticker_options))]
            )
            if tickers_to_plot:
                fig, ax = plt.subplots()
                for sym in tickers_to_plot:
                    row = price_df[price_df["Symbol"] == sym]
                    if not row.empty:
                        ax.plot(week_labels, row.iloc[0, 1:], marker='o', label=sym)
                ax.set_xlabel("Friday (End of Week)")
                ax.set_ylabel("Closing Price")
                ax.set_title("Weekly Closing Price Trend")
                ax.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig)

        else:
            st.error("No valid data fetched for the provided tickers.")
