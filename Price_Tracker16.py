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
    return drawdowns.min() * 100

def calculate_strategy_scores(price_df, all_labels):
    scores = []
    for idx, row in price_df.iterrows():
        closes = row[all_labels].astype(float).values
        symbol = row["Symbol"]
        if np.isnan(closes).any():
            scores.append({"Symbol": symbol, "All-Arounder Score": 0})
        else:
            returns = np.diff(closes) / closes[:-1]
            trend_consistency = int(np.sum(returns[-5:] > 0))
            vol_adj_score = (np.nanmean(returns) / np.nanstd(returns) * 100) if np.nanstd(returns) > 0 else 0
            momentum_score = ((closes[-1] - closes[-2]) / closes[-2]) * 100
            total_return = ((closes[-1] - closes[0]) / closes[0]) * 100
            all_around_score = trend_consistency * 10 + vol_adj_score + momentum_score
            scores.append({
                "Symbol": symbol,
                "Momentum Score": momentum_score,
                "Volatility-Adj Score": vol_adj_score,
                "Trend Consistency": trend_consistency,
                "Last Week % Change": momentum_score,
                "Total Return %": total_return,
                "All-Arounder Score": all_around_score
            })
    return pd.DataFrame(scores).sort_values("All-Arounder Score", ascending=False).reset_index(drop=True)

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet_choice = st.selectbox("Select sheet to analyze", xls.sheet_names)
    tickers_df = pd.read_excel(xls, sheet_name=sheet_choice)
    symbols = tickers_df["Symbol"].tolist()
    weeks, last_friday = get_last_n_weeks(6)
    week_labels = [f"{m.strftime('%Y-%m-%d')} to {f.strftime('%Y-%m-%d')}" for m, f in weeks]

    current_week_label = f"{last_friday.strftime('%Y-%m-%d')} to {datetime.today().strftime('%Y-%m-%d')}"
    result = {}
    for symbol in symbols:
        closes = fetch_friday_closes(symbol, weeks)
        if closes is not None:
            current_close = fetch_current_week_close(symbol, last_friday)
            result[symbol] = closes + [current_close]
        else:
            st.warning(f"Data not available for {symbol}")

    if result:
        all_labels = week_labels + [current_week_label]
        price_df = pd.DataFrame(result).T
        price_df.columns = all_labels
        price_df.reset_index(inplace=True)
        price_df.rename(columns={'index': 'Symbol'}, inplace=True)

        tab1, tab2 = st.tabs(["Price Trend", "Normalized Performance"])

        with tab2:
            ticker_options = price_df["Symbol"].tolist()
            if "norm_selected_tickers" not in st.session_state:
                st.session_state["norm_selected_tickers"] = ticker_options[:3]

            col1, col2 = st.columns([4, 1])
            with col1:
                tickers_to_plot = st.multiselect(
                    "Select tickers", ticker_options,
                    default=st.session_state["norm_selected_tickers"],
                    key="norm"
                )
            with col2:
                if st.button("Select all"):
                    st.session_state["norm_selected_tickers"] = ticker_options

            st.session_state["norm_selected_tickers"] = tickers_to_plot

            if tickers_to_plot:
                fig, ax = plt.subplots()
                for sym in tickers_to_plot:
                    prices = price_df[price_df["Symbol"] == sym][all_labels].iloc[0]
                    norm_prices = (prices / prices[0]) * 100
                    ax.plot(all_labels, norm_prices, label=sym)
                ax.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig)
