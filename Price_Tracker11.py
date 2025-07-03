import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

st.title("📈 Weekly Price Tracker (Monday to Friday Weeks)")

uploaded_file = st.file_uploader("Upload your Excel file", type="xlsx")

def get_last_n_weeks(n):
    today = datetime.today()
    offset = (today.weekday() - 4) % 7  # Days since last Friday (4 = Friday)
    last_friday = today - timedelta(days=offset)
    weeks = []
    for i in reversed(range(n)):
        this_friday = last_friday - timedelta(weeks=i)
        this_monday = this_friday - timedelta(days=4)
        weeks.append((this_monday, this_friday))
    return weeks

def fetch_friday_closes(symbol, weeks):
    first_monday = weeks[0][0]
    last_friday = weeks[-1][1]
    data = yf.download(symbol, start=first_monday, end=last_friday + timedelta(days=1), interval="1d")
    if data.empty or "Close" not in data.columns:
        return None
    closes = []
    for monday, friday in weeks:
        week_data = data.loc[(data.index >= pd.Timestamp(monday)) & (data.index <= pd.Timestamp(friday))]
        friday_close = week_data.loc[week_data.index.weekday == 4, "Close"]
        if not friday_close.empty:
            closes.append(friday_close.iloc[-1])
        elif not week_data.empty:
            closes.append(week_data["Close"].iloc[-1])  # Last available day in week
        else:
            closes.append(np.nan)
    return closes if sum(np.isnan(closes)) == 0 else None

if uploaded_file:
    tickers_df = pd.read_excel(uploaded_file)

    if not all(col in tickers_df.columns for col in ["Symbol", "Exchange"]):
        st.error("Excel file must contain 'Symbol' and 'Exchange' columns.")
    else:
        symbols = tickers_df["Symbol"].tolist()
        weeks = get_last_n_weeks(6)
        week_labels = [f"{m.strftime('%Y-%m-%d')} to {f.strftime('%Y-%m-%d')}" for m, f in weeks]

        result = {}
        for symbol in symbols:
            closes = fetch_friday_closes(symbol, weeks)
            if closes is not None and len(closes) == 6:
                result[symbol] = closes
            else:
                st.warning(f"Ticker {symbol}: Could not fetch 6 weeks of valid closing prices. Skipped.")

        if result:
            price_df = pd.DataFrame(result).T
            price_df.columns = week_labels
            price_df.reset_index(inplace=True)
            price_df.rename(columns={'index': 'Symbol'}, inplace=True)
            st.subheader("Weekly Closing Prices (Monday–Friday Weeks)")
            st.dataframe(price_df)

            # Calculate % price change week over week
            pct_change_df = price_df.set_index("Symbol")[week_labels].pct_change(axis=1) * 100
            pct_change_df = pct_change_df.iloc[:, 1:]
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
                ax.set_xlabel("Week (Monday to Friday)")
                ax.set_ylabel("Closing Price")
                ax.set_title("Weekly Closing Price Trend (Mon–Fri Weeks)")
                ax.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig)
        else:
            st.error("No valid data fetched for the provided tickers.")
