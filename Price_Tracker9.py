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
        # Fetch weekly data with a wider window
        weekly = yf.download(
            symbol,
            start=friday_dates[0] - timedelta(weeks=8),
            end=last_close_dt + timedelta(days=1),
            interval="1wk",
            progress=False,
        )
        
        closes = []
        if not weekly.empty:
            for target_date in friday_dates:
                # Convert target_date to pandas Timestamp for comparison
                target_pd = pd.to_datetime(target_date)
                # Get all dates <= target date
                mask = weekly.index <= target_pd
                available_dates = weekly.index[mask]
                
                if len(available_dates) > 0:
                    closest_date = available_dates[-1]
                    row = weekly.loc[closest_date]
                    
                    # Safely get Close or Adj Close value
                    close_val = row['Close'] if 'Close' in row and pd.notna(row['Close']) else (
                        row['Adj Close'] if 'Adj Close' in row else np.nan
                    )
                    
                    # Ensure we get a scalar value
                    if pd.api.types.is_float(close_val):
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
            # Safely get last close value
            if 'Close' in current_week.columns:
                last_close_series = current_week['Close'].dropna()
            elif 'Adj Close' in current_week.columns:
                last_close_series = current_week['Adj Close'].dropna()
            else:
                last_close_series = pd.Series(dtype=float)
            
            if not last_close_series.empty:
                last_close_val = float(last_close_series.iloc[-1])

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

        # Get current date in EST
        tz = pytz.timezone('US/Eastern')
        now_est = datetime.now(tz)
        days_since_friday = (now_est.weekday() - 4) % 7  # Friday = 4
        last_friday = (now_est - timedelta(days=days_since_friday)).date()
        friday_dates = [last_friday - timedelta(weeks=i) for i in reversed(range(5))]

        # Find last trading day from SPY
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
            if any(pd.notna(x) for x in closes) or pd.notna(last_close):
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

                # Calculate percentage changes
                pct_change_df = price_df.set_index("Symbol")
                numeric_cols = pct_change_df.select_dtypes(include=[np.number]).columns
                
                # Calculate pct change only for numeric columns
                pct_change_values = pct_change_df[numeric_cols].pct_change(axis=1) * 100
                pct_change_df[numeric_cols[1:]] = pct_change_values.iloc[:, 1:].round(2)
                
                pct_change_df.reset_index(inplace=True)
                
                # Generate column labels for changes
                change_cols = []
                for i in range(1, len(week_labels)):
                    if i < len(week_labels):
                        change_cols.append(f"% Change {week_labels[i-1]} to {week_labels[i]}")
                
                pct_change_df.columns = ["Symbol"] + week_labels[:1] + change_cols
                
                st.subheader("Weekly % Price Change")
                st.dataframe(pct_change_df)

                # Calculate metrics
                change_columns = pct_change_df.columns[2:]  # Skip Symbol and first week
                
                if len(change_columns) > 0:
                    pct_change_df["Momentum Score"] = pct_change_df[change_columns].sum(axis=1, skipna=True).round(2)
                    mean_pct = pct_change_df[change_columns].mean(axis=1, skipna=True)
                    std_pct = pct_change_df[change_columns].std(axis=1, skipna=True)
                    pct_change_df["Volatility-Adj Score"] = (mean_pct / std_pct.replace(0, np.nan)).round(2)
                    pct_change_df["Trend Consistency"] = (pct_change_df[change_columns] > 0).sum(axis=1)
                    pct_change_df["Last Week % Change"] = pct_change_df[change_columns[-1]] if len(change_columns) > 0 else np.nan

                    # Calculate total return
                    price_first = price_df.set_index("Symbol")[week_labels[0]]
                    price_last = price_df.set_index("Symbol")[week_labels[-1]]
                    total_return = ((price_last - price_first) / price_first * 100).round(2)
                    pct_change_df["Total Return %"] = total_return

                    # Ranking system
                    cols_to_rank = ["Momentum Score", "Volatility-Adj Score", "Trend Consistency", 
                                   "Last Week % Change", "Total Return %"]
                    for col in cols_to_rank:
                        if col in pct_change_df.columns:
                            pct_change_df[f"{col} Rank"] = pct_change_df[col].rank(ascending=False, na_option='keep')
                    
                    if all(f"{col} Rank" in pct_change_df.columns for col in cols_to_rank):
                        pct_change_df["All-Arounder Score"] = pct_change_df[[f"{col} Rank" for col in cols_to_rank]].sum(axis=1)
                    
                        # Flagging system
                        top_momentum = set(pct_change_df.nlargest(3, "Momentum Score")["Symbol"])
                        top_lastweek = set(pct_change_df.nlargest(3, "Last Week % Change")["Symbol"])
                        top_breakout = top_momentum & top_lastweek

                        top_all_arounder = set(pct_change_df.nsmallest(3, "All-Arounder Score")["Symbol"])
                        top_voladj = set(pct_change_df.nlargest(3, "Volatility-Adj Score")["Symbol"])
                        top_mom_voladj = top_momentum & top_voladj

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

                        st.dataframe(pct_change_df[final_cols].sort_values("All-Arounder Score"))

                        # CSV History
                        history_file = "flag_history.csv"
                        today_str = date.today().isoformat()
                        history = pct_change_df[final_cols].copy()
                        history.insert(0, "Date", today_str)
                        if os.path.exists(history_file):
                            history.to_csv(history_file, mode='a', header=False, index=False)
                        else:
                            history.to_csv(history_file, mode='w', index=False)

                        if st.checkbox("Show recent history"):
                            if os.path.exists(history_file):
                                st.write(pd.read_csv(history_file).tail(30))
                            else:
                                st.write("No history file found.")

                        # Charts
                        st.subheader("Charts")
                        ticker_options = price_df["Symbol"].tolist()
                        tickers_to_plot = st.multiselect(
                            "Select tickers to plot", 
                            ticker_options, 
                            default=ticker_options[:min(3, len(ticker_options))]
                        
                        if tickers_to_plot:
                            tab1, tab2 = st.tabs(["Price Trend", "Normalized Performance"])
                            
                            with tab1:
                                fig, ax = plt.subplots()
                                for sym in tickers_to_plot:
                                    row = price_df[price_df["Symbol"] == sym]
                                    if not row.empty:
                                        prices = row.iloc[0, 1:].values
                                        valid_mask = ~pd.isna(prices)
                                        if any(valid_mask):
                                            ax.plot(
                                                np.array(week_labels)[valid_mask], 
                                                prices[valid_mask], 
                                                marker='o', 
                                                label=sym
                                            )
                                ax.set_xlabel("Week")
                                ax.set_ylabel("Closing Price")
                                ax.set_title("Weekly Closing Price Trend")
                                ax.legend()
                                plt.xticks(rotation=45)
                                st.pyplot(fig)
                            
                            with tab2:
                                fig2, ax2 = plt.subplots()
                                for sym in tickers_to_plot:
                                    row = price_df[price_df["Symbol"] == sym]
                                    if not row.empty:
                                        prices = row.iloc[0, 1:].values
                                        valid_mask = ~pd.isna(prices)
                                        if any(valid_mask) and pd.notna(prices[0]):
                                            norm_prices = prices / prices[0] * 100
                                            ax2.plot(
                                                np.array(week_labels)[valid_mask], 
                                                norm_prices[valid_mask], 
                                                marker='o', 
                                                label=sym
                                            )
                                ax2.set_xlabel("Week")
                                ax2.set_ylabel("Normalized Price (Start=100)")
                                ax2.set_title("Normalized Weekly Performance")
                                ax2.legend()
                                plt.xticks(rotation=45)
                                st.pyplot(fig2)
                else:
                    st.error("No valid data available after filtering.")
            else:
                st.error("No data fetched for any tickers.")
