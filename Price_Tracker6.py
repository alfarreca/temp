import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

st.title("📈 Weekly Price Tracker: Breakouts, All-Arounders & Top Picks")

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

            # ----------- FLAG TOP STRATEGIES -----------
            # Top 3 Momentum + Last Week % Change = "Breakout" (🟢)
            top_momentum = set(
                pct_change_df.sort_values("Momentum Score", ascending=False).head(3)["Symbol"]
            )
            top_lastweek = set(
                pct_change_df.sort_values("Last Week % Change", ascending=False).head(3)["Symbol"]
            )
            top_breakout = top_momentum & top_lastweek

            # Top 3 All-Arounder = "All-Arounder" (🟦)
            cols_to_rank = [
                "Momentum Score",
                "Volatility-Adj Score",
                "Trend Consistency",
                "Last Week % Change",
                "Total Return %"
            ]
            for col in cols_to_rank:
                pct_change_df[f"{col} Rank"] = pct_change_df[col].rank(ascending=False, method='min')
            pct_change_df["All-Arounder Score"] = pct_change_df[[f"{col} Rank" for col in cols_to_rank]].sum(axis=1).astype(int)
            top_all_arounder = set(
                pct_change_df.sort_values("All-Arounder Score", ascending=True).head(3)["Symbol"]
            )

            # Top 3 Momentum + Volatility-Adj = ✅
            top_voladj = set(
                pct_change_df.sort_values("Volatility-Adj Score", ascending=False).head(3)["Symbol"]
            )
            top_mom_voladj = top_momentum & top_voladj

            # ----------- FLAG COLUMN (all can overlap) -----------
            def flag_cell(symbol):
                flags = ""
                if symbol in top_breakout:
                    flags += "🟢"  # Green circle = breakout
                if symbol in top_all_arounder:
                    flags += "🟦"  # Blue square = all-arounder
                if symbol in top_mom_voladj:
                    flags += "✅"  # Green check = top both
                return flags

            pct_change_df["Flags"] = pct_change_df["Symbol"].apply(flag_cell)

            # ----------- FINAL COLUMNS -----------
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

            # ----------- TOP PICK SUMMARIES -----------
            if len(top_breakout):
                st.success(f"**🟢 Breakout candidates:** {', '.join(top_breakout)}")
            if len(top_all_arounder):
                st.info(f"**🟦 All-arounders:** {', '.join(top_all_arounder)}")
            if len(top_mom_voladj):
                st.warning(f"**✅ Momentum & Volatility-Adj top picks:** {', '.join(top_mom_voladj)}")
            if not (len(top_breakout) or len(top_all_arounder) or len(top_mom_voladj)):
                st.info("No tickers flagged in special categories this week.")

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
