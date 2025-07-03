import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

st.title("📈 Weekly Price Tracker (Fri–Fri Weeks + Current Week)")

uploaded_file = st.file_uploader("Upload your Excel file", type="xlsx")

def get_last_n_weeks(n):
    today = datetime.today()
    offset = (today.weekday() - 4) % 7  # 4 = Friday
    last_friday = today - timedelta(days=offset)
    weeks = []
    for i in reversed(range(n)):
        this_friday = last_friday - timedelta(weeks=i)
        this_monday = this_friday - timedelta(days=4)
        weeks.append((this_monday, this_friday))
    return weeks, last_friday

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
            closes.append(float(round(friday_close.iloc[-1], 3)))
        elif not week_data.empty:
            closes.append(float(round(week_data["Close"].iloc[-1], 3)))
        else:
            closes.append(np.nan)
    return closes if sum(np.isnan(closes)) == 0 else None

def fetch_current_week_close(symbol, current_week_start):
    today = datetime.today()
    data = yf.download(symbol, start=current_week_start, end=today + timedelta(days=1), interval="1d")
    if data.empty or "Close" not in data.columns:
        return np.nan
    # Use last available close in the range
    return float(round(data["Close"].iloc[-1], 3))

def get_friday(label):
    return label.split(" to ")[1]

if uploaded_file:
    tickers_df = pd.read_excel(uploaded_file)

    if not all(col in tickers_df.columns for col in ["Symbol", "Exchange"]):
        st.error("Excel file must contain 'Symbol' and 'Exchange' columns.")
    else:
        symbols = tickers_df["Symbol"].tolist()
        weeks, last_friday = get_last_n_weeks(6)
        week_labels = [f"{m.strftime('%Y-%m-%d')} to {f.strftime('%Y-%m-%d')}" for m, f in weeks]

        # Current week: from last Friday to latest close
        today = datetime.today()
        current_week_start = last_friday
        # Find the latest close (for label)
        last_data_close = (today if today.weekday() < 5 else today - timedelta(days=(today.weekday() - 4)))
        current_week_label = f"{current_week_start.strftime('%Y-%m-%d')} to {last_data_close.strftime('%Y-%m-%d')}"

        result = {}
        current_week_col = []
        for symbol in symbols:
            closes = fetch_friday_closes(symbol, weeks)
            current_close = fetch_current_week_close(symbol, current_week_start)
            if closes is not None and len(closes) == 6:
                result[symbol] = closes + [current_close]
            else:
                st.warning(f"Ticker {symbol}: Could not fetch 6 weeks of valid closing prices. Skipped.")
        if result:
            # Add the column label
            all_labels = week_labels + [current_week_label]
            # Weekly Closing Prices Table
            price_df = pd.DataFrame(result).T
            price_df.columns = all_labels
            price_df.reset_index(inplace=True)
            price_df.rename(columns={'index': 'Symbol'}, inplace=True)
            for col in all_labels:
                price_df[col] = pd.to_numeric(price_df[col], errors="coerce")
            st.subheader("Weekly Closing Prices (Fri–Fri Weeks + Current Week)")
            st.dataframe(price_df.style.format(precision=2))

            # --- Weekly % Change as percentage string ---
            try:
                pct_change_df = price_df.set_index("Symbol")[all_labels].astype(float).pct_change(axis=1) * 100
            except Exception as e:
                st.error(f"Error computing percent change: {e}")
            else:
                pct_change_df = pct_change_df.iloc[:, 1:]
                pct_change_df = pct_change_df.round(2)
                pct_change_str = pct_change_df.applymap(lambda x: "" if pd.isna(x) else f"{x:+.2f}%")
                pct_change_str.reset_index(inplace=True)
                # Improved headers: only show Friday dates, for current week use the last date in label
                pct_change_str.columns = ["Symbol"] + [
                    f"% Change {get_friday(all_labels[i-1])} to {get_friday(all_labels[i])}"
                    for i in range(1, len(all_labels))
                ]
                st.subheader("Weekly % Price Change")
                st.dataframe(pct_change_str)

            # --- Chart: Line plot of weekly closes ---
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
                        ax.plot(all_labels, row.iloc[0, 1:], marker='o', label=sym)
                ax.set_xlabel("Week (Friday to Friday, Current: Fri to latest close)")
                ax.set_ylabel("Closing Price")
                ax.set_title("Weekly Closing Price Trend (Fri–Fri Weeks + Current Week)")
                ax.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig)
        else:
            st.error("No valid data fetched for the provided tickers.")
