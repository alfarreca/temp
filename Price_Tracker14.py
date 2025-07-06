import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

st.title("📈 Weekly Price Tracker (Fri–Fri Weeks + Current Week + Scoring + ML Prediction)")

uploaded_file = st.file_uploader("Upload your Excel file", type="xlsx")

def get_last_n_weeks(n):
    today = datetime.today()
    offset = (today.weekday() - 4) % 7
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
    return float(round(data["Close"].iloc[-1], 3))

def get_friday(label):
    return label.split(" to ")[1]

def calculate_strategy_scores(price_df, all_labels):
    scores = []
    for idx, row in price_df.iterrows():
        closes = row[all_labels].astype(float).values
        symbol = row["Symbol"]
        momentum_score = ((closes[-1] - closes[-2]) / closes[-2]) * 100 if len(closes) > 1 and closes[-2] != 0 else 0
        returns = np.diff(closes) / closes[:-1]
        mean_return = np.nanmean(returns)
        std_return = np.nanstd(returns)
        vol_adj_score = mean_return / std_return * 100 if std_return > 0 else mean_return * 100
        trend_consistency = np.sum(returns[-5:] > 0)
        total_return = ((closes[-1] - closes[0]) / closes[0]) * 100 if closes[0] else 0
        all_around_score = trend_consistency * 10 + vol_adj_score + momentum_score
        scores.append({
            "Symbol": symbol,
            "Momentum Score": round(momentum_score, 2),
            "Volatility-Adj Score": round(vol_adj_score, 2),
            "Trend Consistency": int(trend_consistency),
            "Last Week % Change": round(momentum_score, 2),
            "Total Return %": round(total_return, 2),
            "All-Arounder Score": int(np.nan_to_num(all_around_score)),
        })
    score_df = pd.DataFrame(scores).sort_values("All-Arounder Score", ascending=False).reset_index(drop=True)
    return score_df

if uploaded_file:
    tickers_df = pd.read_excel(uploaded_file)
    if not all(col in tickers_df.columns for col in ["Symbol", "Exchange"]):
        st.error("Excel file must contain 'Symbol' and 'Exchange' columns.")
    else:
        symbols = tickers_df["Symbol"].tolist()
        weeks, last_friday = get_last_n_weeks(6)
        week_labels = [f"{m.strftime('%Y-%m-%d')} to {f.strftime('%Y-%m-%d')}" for m, f in weeks]
        current_week_start = last_friday
        current_week_label = f"{current_week_start.strftime('%Y-%m-%d')} to {datetime.today().strftime('%Y-%m-%d')}"
        result = {}
        for symbol in symbols:
            closes = fetch_friday_closes(symbol, weeks)
            current_close = fetch_current_week_close(symbol, current_week_start)
            if closes and len(closes) == 6:
                result[symbol] = closes + [current_close]
            else:
                st.warning(f"Ticker {symbol}: Could not fetch 6 weeks of valid closing prices. Skipped.")
        if result:
            all_labels = week_labels + [current_week_label]
            price_df = pd.DataFrame(result).T
            price_df.columns = all_labels
            price_df.reset_index(inplace=True)
            price_df.rename(columns={'index': 'Symbol'}, inplace=True)
            st.subheader("Weekly Closing Prices")
            st.dataframe(price_df)
            tab1, tab2, tab3, tab4 = st.tabs(["Price Trend", "Normalized Performance", "Ticker Scores", "ML Prediction"])
            with tab3:
                score_df = calculate_strategy_scores(price_df, all_labels)
                st.subheader("Ticker Scores")
                st.dataframe(score_df)
        else:
            st.error("No valid data fetched for the provided tickers.")
