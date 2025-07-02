import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
import numpy as np
import matplotlib.pyplot as plt
import pytz
import os

st.title("📈 Weekly Price Tracker: Fridays, Current Week, Flags & History")

uploaded_file = st.file_uploader("Upload your Excel file", type="xlsx")

def fetch_weekly_and_current_closes(symbol, friday_dates, last_close_dt):
    try:
        # Fetch weekly data with a wider window to ensure we get enough data points
        weekly = yf.download(
            symbol,
            start=friday_dates[0] - timedelta(weeks=8),  # Wider window to ensure data
            end=last_close_dt + timedelta(days=1),
            interval="1wk",
            progress=False,
        )
        
        closes = []
        if not weekly.empty:
            # Find the closest dates to our target Fridays
            for target_date in friday_dates:
                # Find the closest weekly date <= target_date
                available_dates = weekly.index[weekly.index <= pd.to_datetime(target_date)]
                if len(available_dates) > 0:
                    closest_date = available_dates[-1]
                    row = weekly.loc[closest_date]
                    close_val = row.get("Close", row.get("Adj Close", np.nan))
                    if pd.notna(close_val):
                        closes.append(float(close_val))
                    else:
                        closes.append(np.nan)
                else:
                    closes.append(np.nan)
        
        # Fetch current week data
        current_week = yf.download(
            symbol,
            start=friday_dates[-1] + timedelta(days=1),
            end=last_close_dt + timedelta(days=1),
            interval="1d",
            progress=False,
        )
        
        last_close_val = np.nan
        if not current_week.empty:
            last_close_series = current_week.get('Close', current_week.get('Adj Close', pd.Series(dtype=float))).dropna()
            if not last_close_series.empty:
                last_close_val = last_close_series.iloc[-1]

        return closes, last_close_val
    
    except Exception as e:
        st.warning(f"Error fetching data for {symbol}: {str(e)}")
        return [np.nan] * len(friday_dates), np.nan

if uploaded_file:
    tickers_df = pd.read_excel(uploaded_file)

    if not all(col in tickers_df.columns for col in ["Symbol", "Exchange"]):
        st.error("Excel file must contain 'Symbol' and 'Exchange' columns.")
    else:
        symbols = tickers_df["Symbol"].tolist()

        today = datetime.today()
        tz = pytz.timezone('US/Eastern')
        now_est = datetime.now(tz)
        days_since_friday = (now_est.weekday() - 4) % 7  # Friday = 4
        last_friday = (now_est - timedelta(days=days_since_friday)).date()
        friday_dates = [last_friday - timedelta(weeks=i) for i in reversed(range(5))]

        # Find last trading day from a reliable index (SPY)
        spy_data = yf.download("SPY", period="7d", interval="1d", progress=False)
        if spy_data.empty:
            st.error("Could not fetch SPY data to align trading days.")
            st.stop()
        last_close_dt = spy_data.index[-1].to_pydatetime().date()

        week_labels = [d.strftime('%Y-%m-%d') for d in friday_dates]
        week_labels.append(f"Current Week ({friday_dates[-1].strftime('%Y-%m-%d')}–{last_close_dt.strftime('%Y-%m-%d')})")

        result = {}
        for symbol in symbols:
            closes, last_close = fetch_weekly_and_current_closes(symbol, friday_dates, last_close_dt)
            
            # Include data even if incomplete
            if len(closes) > 0:  # At least some data available
                result[symbol] = closes + [last_close]
            else:
                st.warning(f"Ticker {symbol}: No data available. Skipped.")

        if result:
            price_df = pd.DataFrame.from_dict(result, orient='index')
            price_df.columns = week_labels
            price_df = price_df.dropna(how='all')  # Remove rows with all NaN
            
            if not price_df.empty:
                price_df.reset_index(inplace=True)
                price_df.rename(columns={'index': 'Symbol'}, inplace=True)
                st.subheader("Weekly (Friday) + Current Week Closing Prices")
                st.dataframe(price_df)

                # Calculate percentage changes only for available data
                pct_change_df = price_df.set_index("Symbol")[week_labels].pct_change(axis=1, fill_method=None) * 100
                pct_change_df = pct_change_df.iloc[:, 1:].round(2)
                pct_change_df.reset_index(inplace=True)
                
                # Only include columns where we have data
                valid_week_pairs = []
                for i in range(1, len(week_labels)):
                    if i <= pct_change_df.shape[1]:  # Ensure column exists
                        valid_week_pairs.append((i-1, i))
                
                pct_change_df.columns = ["Symbol"] + [
                    f"% Change {week_labels[i-1]} to {week_labels[i]}" 
                    for i in range(1, len(week_labels))
                ]
                
                st.subheader("Weekly % Price Change")
                st.dataframe(pct_change_df)

                # Rest of your processing code...
                # [Keep all your existing scoring and flagging logic]
                
            else:
                st.error("No valid data available after filtering.")
        else:
            st.error("No data fetched for any tickers.")
