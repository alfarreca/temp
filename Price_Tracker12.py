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
    return float(round(data["Close"].iloc[-1], 3))

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

        momentum_score = ((closes[-1] - closes[-2]) / closes[-2]) * 100 if len(closes) >= 2 and closes[-2] != 0 else 0
        returns = np.diff(closes) / closes[:-1]
        mean_return = np.nanmean(returns)
        std_return = np.nanstd(returns)
        vol_adj_score = mean_return / std_return * 100 if std_return > 0 else mean_return * 100
        trend_consistency = np.sum(returns[-5:] > 0)
        last_week_pct = momentum_score
        total_return = ((closes[-1] - closes[0]) / closes[0]) * 100 if closes[0] else 0
        all_around_score = trend_consistency * 10 + vol_adj_score + momentum_score

        scores.append({
            "Symbol": symbol,
            "Momentum Score": round(momentum_score, 2),
            "Volatility-Adj Score": round(vol_adj_score, 2),
            "Trend Consistency": int(trend_consistency) if not np.isnan(trend_consistency) else 0,
            "Last Week % Change": round(last_week_pct, 2),
            "Total Return %": round(total_return, 2),
            "All-Arounder Score": int(all_around_score) if not np.isnan(all_around_score) else 0,
        })

    score_df = pd.DataFrame(scores)
    score_df = score_df.sort_values("All-Arounder Score", ascending=False).reset_index(drop=True)
    return score_df

# Rest of the code remains unchanged (due to character limits)
# You should append the rest of the script as provided in the previous file from line 107 onwards.
