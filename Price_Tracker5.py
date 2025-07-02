import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

st.title("📈 Weekly Price Tracker: Scores, Charts & Top Picks")

uploaded_file = st.file_uploader("Upload your Excel file", type="xlsx")

def fetch_weekly_closes(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end, interval="1wk")['Close']
        closes = data.values
        closes = closes[~np.isnan(closes)]
        if closes.ndim != 1:
            return None
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
            week_labels = [
                (recent_monday - timedelta(weeks=i)).strftime('%Y-%m-%d')
                for i in reversed(range(6))
            ]
            price_df.columns = week_labels
            price_df.reset_index(inplace=True)
            price_df.rename(columns={'index': 'Symbol'}, inplace=True)
            st.subheader("Weekly Closing Prices")
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

            # ----------- SCORE STRATEGIES -----------
            # Extract % changes for calculation
            pct_cols = pct_change_df.columns[1:]  # the % change columns

            # 1. Momentum Score: sum of 5 % changes
            pct_change_df["Momentum Score"] = pct_change_df[pct_cols].sum(axis=1).round(2)
            
            # 2. Volatility-Adjusted Score: mean / std dev (handle division by zero)
            mean_pct = pct_change_df[pct_cols].mean(axis=1)
            std_pct = pct_change_df[pct_cols].std(axis=1)
            pct_change_df["Volatility-Adj Score"] = (mean_pct / std_pct.replace(0, np.nan)).round(2)
            
            # 3. Trend Consistency: number of positive weeks
            pct_change_df["Trend Consistency"] = (pct_change_df[pct_cols] > 0).sum(axis=1)
            
            # 4. Last Week's % Change (for ranking)
            pct_change_df["Last Week % Change"] = pct_change_df[pct_cols[-1]]
            
            # 5. Total Return %: from first to last price (over 6 weeks)
            price_first = price_df.set_index("Symbol")[week_labels[0]]
            price_last = price_df.set_index("Symbol")[week_labels[-1]]
            total_return = ((price_last - price_first) / price_first * 100).round(2)
            pct_change_df = pct_change_df.set_index("Symbol")
            pct_change_df["Total Return %"] = total_return
            pct_change_df.reset_index(inplace=True)

            # ----------- FLAG TOP BOTH -----------
            # Identify top 3 tickers in Momentum Score and Volatility-Adj Score
            top_momentum = set(
                pct_change_df.sort_values("Momentum Score", ascending=False).head(3)["Symbol"]
            )
            top_voladj = set(
                pct_change_df.sort_values("Volatility-Adj Score", ascending=False).head(3)["Symbol"]
            )

            # Flag tickers in both sets
            pct_change_df["Top Both"] = pct_change_df["Symbol"].apply(
                lambda x: "✅" if x in top_momentum and x in top_voladj else ""
            )

            # Put the flag as the first column for visibility
            score_cols = [
                "Top Both",  # new flag column
                "Symbol",
                "Momentum Score",
                "Volatility-Adj Score",
                "Trend Consistency",
                "Last Week % Change",
                "Total Return %"
            ]
            st.subheader("Ticker Scores (5 Strategies) – Top Picks Marked ✅")
            st.dataframe(
                pct_change_df[score_cols].sort_values(
                    by=["Top Both", "Momentum Score"], ascending=[False, False]
                )
            )

            # Show summary of top picks
            top_both_tickers = pct_change_df[pct_change_df["Top Both"] == "✅"]["Symbol"].tolist()
            if top_both_tickers:
                st.success(f"**Top short-term picks (top 3 in both Momentum & Volatility-Adj):** {', '.join(top_both_tickers)}")
            else:
                st.info("No ticker ranked in top 3 for both Momentum and Volatility-Adj Score this week.")

            # ----------- CHARTS WITH TABS -----------
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
                            norm_prices = prices / prices[0] * 100  # Normalize to 100 at start
                            ax2.plot(week_labels, norm_prices, marker='o', label=sym)
                    ax2.set_xlabel("Week")
                    ax2.set_ylabel("Normalized Price (Start=100)")
                    ax2.set_title("Normalized Weekly Performance")
                    ax2.legend()
                    plt.xticks(rotation=45)
                    st.pyplot(fig2)

        else:
            st.error("No valid data fetched for the provided tickers.")
