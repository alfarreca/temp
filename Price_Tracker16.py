import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

st.title("📈 Weekly Price Tracker (Fri–Fri Weeks + Current Week + Scoring)")

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
    return float(round(data["Close"].dropna().iloc[-1], 3))

def get_friday(label):
    return label.split(" to ")[1]

def calculate_cagr(start, end, periods_per_year, periods):
    if start <= 0 or end <= 0 or periods == 0:
        return np.nan
    return (end / start) ** (periods_per_year / periods) - 1

def calculate_max_drawdown(prices):
    if len(prices) < 2:
        return 0.0
    arr = np.array(prices, dtype=np.float64)
    running_max = np.maximum.accumulate(arr)
    drawdowns = (arr - running_max) / running_max
    return drawdowns.min() * 100  # As percentage

def calculate_strategy_scores(price_df, all_labels):
    scores = []
    for idx, row in price_df.iterrows():
        closes = row[all_labels].astype(float).values
        symbol = row["Symbol"]

        if np.isnan(closes).any():
            momentum_score = None
            vol_adj_score = None
            trend_consistency = None
            last_week_pct = None
            total_return = None
            all_around_score = 0
        else:
            # Momentum Score: % return over last week
            if len(closes) < 2 or closes[-2] == 0:
                momentum_score = 0
            else:
                momentum_score = ((closes[-1] - closes[-2]) / closes[-2]) * 100

            # Volatility-Adj Score: Sharpe-like, mean return/std
            returns = np.diff(closes) / closes[:-1]
            mean_return = np.nanmean(returns)
            std_return = np.nanstd(returns)
            vol_adj_score = mean_return / std_return * 100 if std_return > 0 else mean_return * 100

            # Trend Consistency: up weeks out of last 5
            trend_consistency = int(np.sum(returns[-5:] > 0))

            # Last Week % Change: % change in the last week
            last_week_pct = momentum_score

            # Total Return %: from first to last
            total_return = ((closes[-1] - closes[0]) / closes[0]) * 100 if closes[0] else 0

            # All-Arounder Score: custom sum (example: Trend*10 + VolAdj + Momentum)
            all_around_score = trend_consistency * 10 + vol_adj_score + momentum_score

        scores.append({
            "Symbol": symbol,
            "Momentum Score": None if momentum_score is None else round(momentum_score, 2),
            "Volatility-Adj Score": None if vol_adj_score is None else round(vol_adj_score, 2),
            "Trend Consistency": trend_consistency,
            "Last Week % Change": None if last_week_pct is None else round(last_week_pct, 2),
            "Total Return %": None if total_return is None else round(total_return, 2),
            "All-Arounder Score": int(np.nan_to_num(all_around_score)),
        })

    score_df = pd.DataFrame(scores)
    score_df = score_df.sort_values("All-Arounder Score", ascending=False).reset_index(drop=True)
    return score_df

# ---------- Multi-sheet XLSX Upload Block ----------
if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names
    sheet_choice = st.selectbox("Select sheet to analyze", sheet_names)
    tickers_df = pd.read_excel(xls, sheet_name=sheet_choice)

    if not all(col in tickers_df.columns for col in ["Symbol", "Exchange"]):
        st.error("Excel file must contain 'Symbol' and 'Exchange' columns.")
    else:
        symbols = tickers_df["Symbol"].tolist()
        weeks, last_friday = get_last_n_weeks(6)
        week_labels = [f"{m.strftime('%Y-%m-%d')} to {f.strftime('%Y-%m-%d')}" for m, f in weeks]

        # Current week: from last Friday to latest close
        today = datetime.today()
        current_week_start = last_friday
        last_data_close = today
        if today.weekday() >= 5:
            last_data_close = today - timedelta(days=(today.weekday() - 4))
        current_week_label = f"{current_week_start.strftime('%Y-%m-%d')} to {last_data_close.strftime('%Y-%m-%d')}"

        result = {}
        for symbol in symbols:
            closes = fetch_friday_closes(symbol, weeks)
            current_close = fetch_current_week_close(symbol, current_week_start)
            if closes is not None and len(closes) == 6:
                result[symbol] = closes + [current_close]
            else:
                st.warning(f"Ticker {symbol}: Could not fetch 6 weeks of valid closing prices. Skipped.")

        if result:
            all_labels = week_labels + [current_week_label]
            price_df = pd.DataFrame(result).T
            price_df.columns = all_labels
            price_df.reset_index(inplace=True)
            price_df.rename(columns={'index': 'Symbol'}, inplace=True)
            for col in all_labels:
                price_df[col] = pd.to_numeric(price_df[col], errors="coerce")
            st.subheader("Weekly Closing Prices (Fri–Fri Weeks + Current Week)")
            st.dataframe(price_df.style.format(precision=2))

            # Use only columns that are NOT all NaN for any row for scoring:
            non_nan_cols = [col for col in all_labels if not price_df[col].isna().all()]

            # --- Weekly % Change as percentage string (for the dedicated tab) ---
            pct_change_str = None
            try:
                pct_change_df = price_df.set_index("Symbol")[non_nan_cols].astype(float).pct_change(axis=1) * 100
            except Exception as e:
                st.error(f"Error computing percent change: {e}")
            else:
                pct_change_df = pct_change_df.iloc[:, 1:]
                pct_change_df = pct_change_df.round(2)
                pct_change_str = pct_change_df.applymap(lambda x: "" if pd.isna(x) else f"{x:+.2f}%")
                pct_change_str.reset_index(inplace=True)
                pct_change_str.columns = ["Symbol"] + [
                    f"% Change {get_friday(non_nan_cols[i-1])} to {get_friday(non_nan_cols[i])}"
                    for i in range(1, len(non_nan_cols))
                ]

            tab1, tab2, tab3, tab4 = st.tabs([
                "Price Trend",
                "Normalized Performance",
                "Weekly % Price Change",
                "Ticker Scores (5 Strategies)"
            ])

            with tab1:
                st.subheader("Weekly Closing Price Trend")
                ticker_options = price_df["Symbol"].tolist()
                tickers_to_plot = st.multiselect(
                    "Select tickers to plot", ticker_options, default=ticker_options[:min(3, len(ticker_options))], key="trend"
                )
                if tickers_to_plot:
                    fig, ax = plt.subplots()
                    for sym in tickers_to_plot:
                        row = price_df[price_df["Symbol"] == sym]
                        if not row.empty:
                            ax.plot(non_nan_cols, row[non_nan_cols].iloc[0], marker='o', label=sym)
                    ax.set_xlabel("Week (Friday to Friday, Current: Fri to latest close)")
                    ax.set_ylabel("Closing Price")
                    ax.set_title("Weekly Closing Price Trend")
                    ax.legend()
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

            with tab2:
                st.markdown("**Performance normalized to 100 at the start: compare pure relative gains/losses.**")
                ticker_options = price_df["Symbol"].tolist()

                col1, col2 = st.columns([4,1])
                with col1:
                    tickers_to_plot = st.multiselect(
                        "Select tickers to plot (normalized)", ticker_options, 
                        default=st.session_state.get('norm', ticker_options[:min(3, len(ticker_options))]), key="norm"
                    )
                with col2:
                    if st.button("Select all tickers to plot", key="norm_select_all_btn"):
                        st.session_state["norm"] = ticker_options
                        st.experimental_rerun()

                if tickers_to_plot:
                    fig, ax = plt.subplots()
                    cagr_dict = {}
                    mdd_dict = {}
                    for sym in tickers_to_plot:
                        row = price_df[price_df["Symbol"] == sym]
                        if not row.empty:
                            prices = row[non_nan_cols].astype(float)
                            # --- FIX for ambiguous truth value error ---
                            if len(prices.columns) == 0 or pd.isna(prices.iloc[0,0]) or prices.iloc[0,0] == 0:
                                norm_prices = prices.iloc[0]
                            else:
                                norm_prices = (prices.iloc[0] / prices.iloc[0,0]) * 100
                            ax.plot(non_nan_cols, norm_prices, marker='o', label=sym)
                            periods = len(norm_prices) - 1
                            periods_per_year = 52
                            cagr = calculate_cagr(norm_prices.iloc[0], norm_prices.iloc[-1], periods_per_year, periods)
                            cagr_dict[sym] = f"{cagr*100:.2f}%" if not np.isnan(cagr) else "n/a"
                            mdd = calculate_max_drawdown(norm_prices)
                            mdd_dict[sym] = f"{mdd:.2f}%" if not np.isnan(mdd) else "n/a"
                    ax.set_xlabel("Week")
                    ax.set_ylabel("Normalized Price (Start=100)")
                    ax.set_title("Normalized Weekly Performance")
                    ax.legend()
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    stats_df = pd.DataFrame({
                        "CAGR (Annualized)": cagr_dict,
                        "Max Drawdown": mdd_dict
                    })
                    st.subheader("Advanced Performance Metrics")
                    st.dataframe(stats_df)

            with tab3:
                st.subheader("Weekly % Price Change")
                if pct_change_str is not None:
                    st.dataframe(pct_change_str)

            with tab4:
                score_df = calculate_strategy_scores(price_df, non_nan_cols)
                score_df["Flags"] = ""
                if not score_df.empty:
                    if score_df["Momentum Score"].notna().any():
                        breakout_idx = score_df["Momentum Score"].idxmax()
                        if breakout_idx in score_df.index:
                            score_df.at[breakout_idx, "Flags"] += "🟢 "
                    all_around_idxs = score_df.nlargest(3, "All-Arounder Score").index
                    for i in all_around_idxs:
                        if i in score_df.index:
                            score_df.at[i, "Flags"] += "🔵 "
                score_df = score_df[["Flags", "Symbol", "Momentum Score", "Volatility-Adj Score", "Trend Consistency",
                                    "Last Week % Change", "Total Return %", "All-Arounder Score"]]
                st.markdown("### Ticker Scores (5 Strategies)")
                st.dataframe(score_df)
                breakout_candidates = score_df.loc[score_df["Flags"].str.contains("🟢"), "Symbol"].tolist()
                all_arounders = score_df.loc[score_df["Flags"].str.contains("🔵"), "Symbol"].tolist()
                if breakout_candidates:
                    st.success(f"🟢 **Breakout candidates:** {', '.join(breakout_candidates)}")
                if all_arounders:
                    st.info(f"🔵 **All-arounders:** {', '.join(all_arounders)}")

        else:
            st.error("No valid data fetched for the provided tickers.")
