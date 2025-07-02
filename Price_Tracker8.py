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
    weekly = yf.download(
        symbol,
        start=friday_dates[0] - timedelta(days=7),
        end=last_close_dt + timedelta(days=1),
        interval="1wk",
        progress=False,
    )
    closes = []
    if not weekly.empty:
        for i in range(-5, 0):
            if abs(i) <= len(weekly):
                row = weekly.iloc[i]
                close_val = row.get("Close", np.nan)
                if isinstance(close_val, pd.Series):
                    close_val = close_val.item() if len(close_val) > 0 else np.nan
                if not isinstance(close_val, float):
                    try:
                        close_val = float(close_val)
                    except:
                        close_val = np.nan
                if not pd.isna(close_val):
                    closes.append(close_val)
                else:
                    adj_close_val = row.get("Adj Close", np.nan)
                    if isinstance(adj_close_val, pd.Series):
                        adj_close_val = adj_close_val.item() if len(adj_close_val) > 0 else np.nan
                    if not isinstance(adj_close_val, float):
                        try:
                            adj_close_val = float(adj_close_val)
                        except:
                            adj_close_val = np.nan
                    closes.append(adj_close_val if not pd.isna(adj_close_val) else np.nan)
    closes = [x for x in closes if not pd.isna(x)]

    current_week = yf.download(
        symbol,
        start=friday_dates[-1] + timedelta(days=1),
        end=last_close_dt + timedelta(days=1),
        interval="1d",
        progress=False,
    )
    if not current_week.empty:
        last_close_series = current_week.get('Close', current_week.get('Adj Close', pd.Series(dtype=float))).dropna()
        if isinstance(last_close_series, pd.Series):
            last_close_val = last_close_series.iloc[-1] if not last_close_series.empty else np.nan
        else:
            last_close_val = last_close_series
    else:
        last_close_val = np.nan

    return closes, last_close_val

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

        sample_symbol = symbols[0]
        daily_data = yf.download(sample_symbol, period="7d", interval="1d", progress=False)
        if daily_data.empty:
            st.error(f"Could not fetch data for {sample_symbol} to align trading days.")
            st.stop()
        last_close_dt = daily_data.index[-1].to_pydatetime().date()

        week_labels = [d.strftime('%Y-%m-%d') for d in friday_dates]
        week_labels.append(f"Current Week ({friday_dates[-1].strftime('%Y-%m-%d')}–{last_close_dt.strftime('%Y-%m-%d')})")

        result = {}
        for symbol in symbols:
            closes, last_close = fetch_weekly_and_current_closes(symbol, friday_dates, last_close_dt)
            # Safe scalar conversion (no .item())
            try:
                last_close_scalar = float(last_close)
            except Exception:
                last_close_scalar = np.nan
            if len(closes) == 5 and not pd.isna(last_close_scalar):
                result[symbol] = closes + [last_close_scalar]
            else:
                st.warning(f"Ticker {symbol}: Could not fetch enough data for all 6 columns. Skipped.")

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

            top_momentum = set(pct_change_df.nlargest(3, "Momentum Score")["Symbol"])
            top_lastweek = set(pct_change_df.nlargest(3, "Last Week % Change")["Symbol"])
            top_breakout = top_momentum & top_lastweek

            cols_to_rank = ["Momentum Score", "Volatility-Adj Score", "Trend Consistency", "Last Week % Change", "Total Return %"]
            for col in cols_to_rank:
                pct_change_df[f"{col} Rank"] = pct_change_df[col].rank(ascending=False)
            pct_change_df["All-Arounder Score"] = pct_change_df[[f"{col} Rank" for col in cols_to_rank]].sum(axis=1)
            top_all_arounder = set(pct_change_df.nsmallest(3, "All-Arounder Score")["Symbol"])

            top_voladj = set(pct_change_df.nlargest(3, "Volatility-Adj Score")["Symbol"])
            top_mom_voladj = top_momentum & top_voladj

            def flag_cell(symbol):
                return ("🟢" if symbol in top_breakout else "") + \
                       ("🟦" if symbol in top_all_arounder else "") + \
                       ("✅" if symbol in top_mom_voladj else "")
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
                st.write(pd.read_csv(history_file).tail(30))

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
                    fig, ax = plt.subplots()
                    for sym in tickers_to_plot:
                        row = price_df[price_df["Symbol"] == sym]
                        if not row.empty:
                            ax.plot(week_labels, row.iloc[0, 1:], marker='o', label=sym)
                    ax.set_xlabel("Week")
                    ax.set_ylabel("Closing Price")
                    ax.set_title("Weekly Closing Price Trend")
                    ax.legend()
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
            with tab2:
                st.markdown("**Performance normalized to 100 at the start: compare pure relative gains/losses.**")
                if tickers_to_plot:
                    fig2, ax2 = plt.subplots()
                    for sym in tickers_to_plot:
                        row = price_df[price_df["Symbol"] == sym]
                        if not row.empty:
                            prices = row.iloc[0, 1:].values.astype(float)
                            norm_prices = prices / prices[0] * 100
                            ax2.plot(week_labels, norm_prices, marker='o', label=sym)
                    ax2.set_xlabel("Week")
                    ax2.set_ylabel("Normalized Price (Start=100)")
                    ax2.set_title("Normalized Weekly Performance")
                    ax2.legend()
                    plt.xticks(rotation=45)
                    st.pyplot(fig2)
        else:
            st.error("No valid data fetched.")
