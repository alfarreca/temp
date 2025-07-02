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
    """
    Fetches the closing prices for the last 5 Fridays and the last available close
    for the current week for a given stock symbol.
    """
    
    # Ensure we request enough history for weekly data (5 Fridays + a buffer)
    weekly_start_date = friday_dates[0] - timedelta(weeks=8) 
    
    # Removed show_errors=False
    weekly = yf.download(
        symbol,
        start=weekly_start_date, 
        end=last_close_dt + timedelta(days=1),
        interval="1wk",
        progress=False
    )
    
    closes = []
    if not weekly.empty:
        # Convert friday_dates to pandas Timestamps for direct comparison
        target_fridays_ts = pd.to_datetime([str(d) for d in friday_dates])
        
        for target_date in target_fridays_ts:
            # Find the weekly bar whose end date (index) is closest to and before or on the target_date
            valid_weekly_rows = weekly[weekly.index <= target_date].tail(1)
            
            if not valid_weekly_rows.empty:
                close_val = valid_weekly_rows.iloc[0].get("Close", np.nan)
                if pd.isna(close_val):
                    close_val = valid_weekly_rows.iloc[0].get("Adj Close", np.nan)
                closes.append(close_val)
            else:
                closes.append(np.nan) # No data for this specific Friday
    
    # Filter out NaNs and check if we have enough weekly closes
    closes = [x for x in closes if not pd.isna(x)]
    if len(closes) < 5:
        return [], np.nan, "not_enough_weekly" # Indicate failure to get enough weekly data
        
    # Fetch current week daily data
    current_week_start = friday_dates[-1] + timedelta(days=1)
    # Removed show_errors=False
    current_week = yf.download(
        symbol,
        start=current_week_start,
        end=last_close_dt + timedelta(days=1),
        interval="1d",
        progress=False
    )
    
    last_close_val = np.nan
    if not current_week.empty:
        last_close_series = current_week.get('Close', current_week.get('Adj Close', pd.Series(dtype=float))).dropna()
        if isinstance(last_close_series, pd.Series):
            last_close_val = last_close_series.iloc[-1] if not last_close_series.empty else np.nan
        else:
            last_close_val = last_close_series
    
    if pd.isna(last_close_val):
        return closes, np.nan, "no_current_week_data" # Indicate failure to get current week's data

    return closes, last_close_val, "success"

if uploaded_file:
    try:
        tickers_df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading Excel file: {e}. Please ensure it's a valid .xlsx file.")
        st.stop()

    if not all(col in tickers_df.columns for col in ["Symbol", "Exchange"]):
        st.error("Excel file must contain 'Symbol' and 'Exchange' columns.")
    else:
        symbols = tickers_df["Symbol"].dropna().astype(str).unique().tolist() # Clean and unique symbols

        today = datetime.today()
        tz = pytz.timezone('US/Eastern')
        now_est = datetime.now(tz)
        days_since_friday = (now_est.weekday() - 4) % 7  # Friday = 4
        last_friday = (now_est - timedelta(days=days_since_friday)).date()
        friday_dates = [last_friday - timedelta(weeks=i) for i in reversed(range(5))]

        # Fetch data for a sample symbol to determine the actual last trading day
        # This helps align current week's data correctly even if today is not a trading day
        last_close_dt = None
        if symbols:
            # Iterate through symbols to find the first one that successfully fetches data
            for attempt_symbol in symbols:
                try:
                    # Removed show_errors=False
                    daily_data = yf.download(attempt_symbol, period="7d", interval="1d", progress=False)
                    if not daily_data.empty:
                        last_close_dt = daily_data.index[-1].to_pydatetime().date()
                        break # Found a valid sample, exit loop
                except Exception as e:
                    # Catch any exception during yf.download for the sample symbol
                    st.warning(f"Could not fetch sample data for {attempt_symbol} to align trading days due to an error: {e}")
                    continue # Try the next symbol

            if last_close_dt is None:
                st.error("Could not fetch data for any sample symbol to determine the last trading day. Please check your ticker list and internet connection.")
                st.stop()
        else:
            st.warning("No valid symbols found in the uploaded Excel file after cleaning.")
            st.stop()


        week_labels = [d.strftime('%Y-%m-%d') for d in friday_dates]
        week_labels.append(f"Current Week ({friday_dates[-1].strftime('%Y-%m-%d')}–{last_close_dt.strftime('%Y-%m-%d')})")

        result = {}
        for symbol in symbols:
            closes, last_close, status = fetch_weekly_and_current_closes(symbol, friday_dates, last_close_dt)
            
            if status == "not_enough_weekly":
                st.warning(f"Ticker {symbol}: Could not fetch 5 weeks of historical Friday closing data. Skipped.")
                continue
            elif status == "no_current_week_data":
                st.warning(f"Ticker {symbol}: Could not fetch current week's closing price. Skipped.")
                continue
            elif status == "success":
                try:
                    last_close_scalar = float(last_close)
                except Exception:
                    last_close_scalar = np.nan
                    st.warning(f"Ticker {symbol}: Current week's closing price could not be converted to a number. Skipped.")
                    continue

                if len(closes) == 5 and not pd.isna(last_close_scalar):
                    result[symbol] = closes + [last_close_scalar]
                else:
                    # This case should ideally be caught by specific status checks, but acts as a fallback
                    st.warning(f"Ticker {symbol}: Incomplete data after fetch. Check weekly/current closes. Skipped.")

        if result:
            price_df = pd.DataFrame(result).T
            price_df.columns = week_labels
            price_df.reset_index(inplace=True)
            price_df.rename(columns={'index': 'Symbol'}, inplace=True)
            st.subheader("Weekly (Friday) + Current Week Closing Prices")
            st.dataframe(price_df)

            pct_change_df = price_df.set_index("Symbol")[week_labels].pct_change(axis=1) * 100
            pct_change_df = pct_change_df.iloc[:, 1:].round(2)
            pct_change_df.reset_index(inplace=True)
            pct_change_df.columns = ["Symbol"] + [
                f"% Change {week_labels[i-1]} to {week_labels[i]}" for i in range(1, 6)
            ]
            st.subheader("Weekly % Price Change")
            st.dataframe(pct_change_df)

            pct_cols = pct_change_df.columns[1:]
            pct_change_df["Momentum Score"] = pct_change_df[pct_cols].sum(axis=1).round(2)
            mean_pct = pct_change_df[pct_cols].mean(axis=1)
            std_pct = pct_change_df[pct_cols].std(axis=1)
            pct_change_df["Volatility-Adj Score"] = (mean_pct / std_pct.replace(0, np.nan)).round(2)
            pct_change_df["Trend Consistency"] = (pct_change_df[pct_cols] > 0).sum(axis=1)
            pct_change_df["Last Week % Change"] = pct_change_df[pct_cols[-1]]

            price_first = price_df.set_index("Symbol")[week_labels[0]]
            price_last = price_df.set_index("Symbol")[week_labels[-1]]
            total_return = ((price_last - price_first) / price_first * 100).round(2)
            pct_change_df = pct_change_df.set_index("Symbol")
            pct_change_df["Total Return %"] = total_return
            pct_change_df.reset_index(inplace=True)

            # Flag Calculation
            top_momentum = set(pct_change_df.nlargest(3, "Momentum Score")["Symbol"])
            top_lastweek = set(pct_change_df.nlargest(3, "Last Week % Change")["Symbol"])
            top_breakout = top_momentum & top_lastweek # Green Circle

            cols_to_rank = ["Momentum Score", "Volatility-Adj Score", "Trend Consistency", "Last Week % Change", "Total Return %"]
            for col in cols_to_rank:
                # Rank handling for NaNs: 'top' puts NaNs at the end, 'average' rank method
                pct_change_df[f"{col} Rank"] = pct_change_df[col].rank(ascending=False, na_option='bottom', method='average')
            
            pct_change_df["All-Arounder Score"] = pct_change_df[[f"{col} Rank" for col in cols_to_rank]].sum(axis=1)
            top_all_arounder = set(pct_change_df.nsmallest(3, "All-Arounder Score")["Symbol"]) # Blue Square

            top_voladj = set(pct_change_df.nlargest(3, "Volatility-Adj Score")["Symbol"])
            top_mom_voladj = top_momentum & top_voladj # Green Checkmark

            def flag_cell(symbol):
                flags = []
                if symbol in top_breakout:
                    flags.append("🟢")
                if symbol in top_all_arounder:
                    flags.append("🟦")
                if symbol in top_mom_voladj:
                    flags.append("✅")
                return "".join(flags)

            pct_change_df["Flags"] = pct_change_df["Symbol"].apply(flag_cell)

            final_cols = ["Flags", "Symbol", "Momentum Score", "Volatility-Adj Score",
                          "Trend Consistency", "Last Week % Change", "Total Return %", "All-Arounder Score"]

            st.subheader("Analysis & Flags")
            st.write("🟢 Top Breakout (High Momentum & High Last Week % Change)")
            st.write("🟦 Top All-Arounder (Lowest combined rank across all metrics)")
            st.write("✅ Top Momentum & Volatility-Adjusted Score")
            
            st.dataframe(pct_change_df[final_cols].sort_values("All-Arounder Score"))

            # CSV History
            history_file = "flag_history.csv"
            today_str = date.today().isoformat()
            history = pct_change_df[final_cols].copy()
            history.insert(0, "Date", today_str)
            
            # Use st.session_state for temporary storage to prevent excessive file writes
            # and enable better persistence simulation in a single session.
            if 'history_df' not in st.session_state:
                if os.path.exists(history_file):
                    st.session_state.history_df = pd.read_csv(history_file)
                else:
                    st.session_state.history_df = pd.DataFrame(columns=["Date"] + final_cols)

            # Append only if not already added for today
            # Use tuple comparison for rows to avoid issues with floating point precision in full DF comparison
            current_day_history = st.session_state.history_df[st.session_state.history_df["Date"] == today_str]
            if not history.apply(tuple, axis=1).isin(current_day_history.apply(tuple, axis=1)).all():
                st.session_state.history_df = pd.concat([st.session_state.history_df, history], ignore_index=True)
                st.session_state.history_df.to_csv(history_file, index=False) # Write to CSV

            if st.checkbox("Show recent history"):
                st.subheader("Historical Flags & Scores")
                st.write(st.session_state.history_df.tail(30))

            # --- CHARTS WITH TABS ---
            st.subheader("Charts")
            ticker_options = price_df["Symbol"].tolist()
            tickers_to_plot = st.multiselect(
                "Select tickers to plot", ticker_options, default=ticker_options[:min(3, len(ticker_options))]
            )
            tab1, tab2 = st.tabs(["Price Trend", "Normalized Performance"])
            with tab1:
                st.markdown("**Raw weekly closing prices for each ticker.**")
                if tickers_to_plot:
                    fig, ax = plt.subplots(figsize=(10, 6)) # Added figsize for better readability
                    for sym in tickers_to_plot:
                        row = price_df[price_df["Symbol"] == sym]
                        if not row.empty:
                            ax.plot(week_labels, row.iloc[0, 1:], marker='o', label=sym)
                    ax.set_xlabel("Week")
                    ax.set_ylabel("Closing Price")
                    ax.set_title("Weekly Closing Price Trend")
                    ax.legend()
                    plt.xticks(rotation=45, ha='right') # Improved rotation and alignment
                    plt.tight_layout() # Adjust layout to prevent labels from overlapping
                    st.pyplot(fig)
                else:
                    st.info("Select at least one ticker to view the Price Trend chart.")
            with tab2:
                st.markdown("**Performance normalized to 100 at the start: compare pure relative gains/losses.**")
                if tickers_to_plot:
                    fig2, ax2 = plt.subplots(figsize=(10, 6)) # Added figsize
                    for sym in tickers_to_plot:
                        row = price_df[price_df["Symbol"] == sym]
                        if not row.empty:
                            prices = row.iloc[0, 1:].values.astype(float)
                            if len(prices) > 0 and prices[0] != 0: # Avoid division by zero
                                norm_prices = prices / prices[0] * 100
                                ax2.plot(week_labels, norm_prices, marker='o', label=sym)
                            else:
                                st.warning(f"Cannot normalize {sym}: Insufficient price data or starting price is zero.")
                    ax2.set_xlabel("Week")
                    ax2.set_ylabel("Normalized Price (Start=100)")
                    ax2.set_title("Normalized Weekly Performance")
                    ax2.legend()
                    plt.xticks(rotation=45, ha='right') # Improved rotation and alignment
                    plt.tight_layout() # Adjust layout
                    st.pyplot(fig2)
                else:
                    st.info("Select at least one ticker to view the Normalized Performance chart.")
        else:
            st.error("No valid data could be fetched for any of the symbols. Please check your ticker list and ensure they are valid and actively traded on Yahoo Finance.")
