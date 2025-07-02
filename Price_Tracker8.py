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
    # Weekly closes for last 5 Fridays
    weekly = yf.download(
        symbol,
        start=friday_dates[0] - timedelta(days=7),
        end=last_close_dt + timedelta(days=1),
        interval="1wk",
        progress=False,
    )
    closes = []
    if not weekly.empty:
        # yfinance returns weekly rows, each with 'Close' and maybe 'Adj Close'
        for i in range(-5, 0):
            if abs(i) <= len(weekly):
                row = weekly.iloc[i]
                if 'Close' in row and not pd.isna(row['Close']):
                    closes.append(row['Close'])
                elif 'Adj Close' in row and not pd.isna(row['Adj Close']):
                    closes.append(row['Adj Close'])
                else:
                    closes.append(np.nan)
    # Drop NaN to avoid bad tickers
    closes = [x for x in closes if not pd.isna(x)]

    # For current week: last available daily close after last Friday
    current_week = yf.download(
        symbol,
        start=friday_dates[-1] + timedelta(days=1),
        end=last_close_dt + timedelta(days=1),
        interval="1d",
        progress=False,
    )
    if not current_week.empty:
        if 'Close' in current_week:
            last_close_series = current_week['Close'].dropna()
        elif 'Adj Close' in current_week:
            last_close_series = current_week['Adj Close'].dropna()
        else:
            last_close_series = pd.Series(dtype=float)
        last_close_val = last_close_series[-1] if not last_close_series.empty else np.nan
    else:
        last_close_val = np.nan

    return closes, last_close_val

if uploaded_file:
    tickers_df = pd.read_excel(uploaded_file)

    if not all(col in tickers_df.columns for col in ["Symbol", "Exchange"]):
        st.error("Excel file must contain 'Symbol' and 'Exchange' columns.")
    else:
        symbols = tickers_df["Symbol"].tolist()

        # 1. Get last 5 Fridays (ending last full week)
        today = datetime.today()
        tz = pytz.timezone('US/Eastern')
        now_est = datetime.now(tz)
        days_since_friday = (now_est.weekday() - 4) % 7  # Friday = 4
        last_friday = (now_est - timedelta(days=days_since_friday)).date()
        friday_dates = [last_friday - timedelta(weeks=i) for i in reversed(range(5))]

        # 2. Get most recent available daily close for "current week"
        sample_symbol = symbols[0]
        daily_data = yf.download(sample_symbol, period="7d", interval="1d", progress=False)
        if daily_data.empty:
            st.error(f"Could not fetch data for {sample_symbol} to align trading days.")
            st.stop()
        last_close_dt = daily_data.index[-1].to_pydatetime().date()

        # Build labels
        week_labels = [d.strftime('%Y-%m-%d') for d in friday_dates]
        week_labels.append(f"Current Week ({friday_dates[-1].strftime('%Y-%m-%d')}–{last_close_dt.strftime('%Y-%m-%d')})")

        # --------- FETCH PRICES ---------
        result = {}
        for symbol in symbols:
            closes, last_close = fetch_weekly_and_current_closes(symbol, friday_dates, last_close_dt)
            if len(closes) == 5 and not np.isnan(last_close):
                result[symbol] = closes + [last_close]
            else:
                st.warning(f"Ticker {symbol}: Could not fetch enough data for all 6 columns. Skipped.")

        if result:
            price_df = pd.DataFrame(result).T
            price_df.columns = week_labels
            price_df.reset_index(inplace=True)
            price_df.rename(columns={'index': 'Symbol'}, inplace=True)
            st.subheader("Weekly (Friday) + Current Week Closing Prices")
            st.dataframe(price_df)

            # --- Weekly % price changes (vs previous week) ---
            pct_change_df = price_df.set_index("Symbol")[week_labels].pct_change(axis=1) * 100
            pct_change_df = pct_change_df.iloc[:, 1:]  # remove first col (no previous)
            pct_change_df = pct_change_df.round(2)
            pct_change_df.reset_index(inplace=True)
            pct_change_df.columns = ["Symbol"] + [
                f"% Change {week_labels[i-1]} to {week_labels[i]}" for i in range(1, 6)
            ]
            st.subheader("Weekly % Price Change")
            st.dataframe(pct_change_df)

            # --- SCORE STRATEGIES ---
            pct_cols = pct_change_df.columns[1:]

            # 1. Momentum Score: sum of 5 % changes
            pct_change_df["Momentum Score"] = pct_change_df[pct_cols].sum(axis=1).round(2)
            # 2. Volatility-Adj Score: mean/stddev
            mean_pct = pct_change_df[pct_cols].mean(axis=1)
            std_pct = pct_change_df[pct_cols].std(axis=1)
            pct_change_df["Volatility-Adj Score"] = (mean_pct / std_pct.replace(0, np.nan)).round(2)
            # 3. Trend Consistency: number of positive weeks
            pct_change_df["Trend Consistency"] = (pct_change_df[pct_cols] > 0).sum(axis=1)
            # 4. Last Week's % Change (for ranking)
            pct_change_df["Last Week % Change"] = pct_change_df[pct_cols[-1]]
            # 5. Total Return %: from first to last price (over 6 columns)
            price_first = price_df.set_index("Symbol")[week_labels[0]]
            price_last = price_df.set_index("Symbol")[week_labels[-1]]
            total_return = ((price_last - price_first) / price_first * 100).round(2)
            pct_change_df = pct_change_df.set_index("Symbol")
            pct_change_df["Total Return %"] = total_return
            pct_change_df.reset_index(inplace=True)

            # --- FLAGS ---
            # Top 3 Momentum + Last Week % Change = "Breakout" (🟢)
            top_momentum = set(pct_change_df.sort_values("Momentum Score", ascending=False).head(3)["Symbol"])
            top_lastweek = set(pct_change_df.sort_values("Last Week % Change", ascending=False).head(3)["Symbol"])
            top_breakout = top_momentum & top_lastweek
            # Top 3 All-Arounder = "All-Arounder" (🟦)
            cols_to_rank = [
                "Momentum Score", "Volatility-Adj Score", "Trend Consistency",
                "Last Week % Change", "Total Return %"
            ]
            for col in cols_to_rank:
                pct_change_df[f"{col} Rank"] = pct_change_df[col].rank(ascending=False, method='min')
            pct_change_df["All-Arounder Score"] = pct_change_df[[f"{col} Rank" for col in cols_to_rank]].sum(axis=1).astype(int)
            top_all_arounder = set(pct_change_df.sort_values("All-Arounder Score", ascending=True).head(3)["Symbol"])
            # Top 3 Momentum + Volatility-Adj = ✅
            top_voladj = set(pct_change_df.sort_values("Volatility-Adj Score", ascending=False).head(3)["Symbol"])
            top_mom_voladj = top_momentum & top_voladj

            # Flag column
            def flag_cell(symbol):
                flags = ""
                if symbol in top_breakout:
                    flags += "🟢"
                if symbol in top_all_arounder:
                    flags += "🟦"
                if symbol in top_mom_voladj:
                    flags += "✅"
                return flags
            pct_change_df["Flags"] = pct_change_df["Symbol"].apply(flag_cell)

            # --- FINAL COLUMNS ---
            score_cols = [
                "Flags",
                "Symbol",
                "Momentum Score",
                "Volatility-Adj Score",
                "Trend Consistency",
                "Last Week % Change",
                "Total Return %",
                "All-Arounder Score"
            ]
            st.subheader(
                "Ticker Scores (5 Strategies)\n🟢 Breakout (Momentum+LastWk) | 🟦 All-Arounder | ✅ Top Momentum+VolAdj"
            )
            st.dataframe(
                pct_change_df[score_cols].sort_values(
                    by=["Flags", "All-Arounder Score", "Momentum Score"], ascending=[False, True, False]
                )
            )

            # --- TOP PICK SUMMARIES ---
            if len(top_breakout):
                st.success(f"**🟢 Breakout candidates:** {', '.join(top_breakout)}")
            if len(top_all_arounder):
                st.info(f"**🟦 All-arounders:** {', '.join(top_all_arounder)}")
            if len(top_mom_voladj):
                st.warning(f"**✅ Momentum & Volatility-Adj top picks:** {', '.join(top_mom_voladj)}")
            if not (len(top_breakout) or len(top_all_arounder) or len(top_mom_voladj)):
                st.info("No tickers flagged in special categories this week.")

            # --- FLAG/SCORE HISTORY TRACKING ---
            history_cols = [
                "Symbol", "Flags", "Momentum Score", "Volatility-Adj Score", "Trend Consistency",
                "Last Week % Change", "Total Return %", "All-Arounder Score"
            ]
            today_str = date.today().isoformat()
            history_filename = "flag_history.csv"
            record_df = pct_change_df[history_cols].copy()
            record_df.insert(0, "Date", today_str)
            if not os.path.exists(history_filename):
                record_df.to_csv(history_filename, mode='w', index=False)
            else:
                record_df.to_csv(history_filename, mode='a', index=False, header=False)

            if st.checkbox("Show recent flag/score history"):
                hist_df = pd.read_csv(history_filename)
                st.write(hist_df.tail(30))  # Show last 30 records

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
            st.error("No valid data fetched for the provided tickers.")
