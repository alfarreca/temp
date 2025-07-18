import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objs as go
from pathlib import Path

st.set_page_config(layout="wide")
st.title("📈 Weekly Price Tracker (Fri–Fri Weeks + Current Week + Scoring)")

uploaded_file = st.file_uploader("Upload your Excel file", type="xlsx")

@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
def fetch_friday_closes(symbol, weeks):
    first_monday = weeks[0][0]
    last_friday = weeks[-1][1]
    data = yf.download(symbol, start=first_monday, end=last_friday + timedelta(days=1), interval="1d", progress=False)
    if data.empty or "Close" not in data.columns:
        return [np.nan] * len(weeks)

    closes = []
    for monday, friday in weeks:
        week_data = data.loc[(data.index >= pd.Timestamp(monday)) & (data.index <= pd.Timestamp(friday))]
        friday_close = week_data.loc[week_data.index.weekday == 4, "Close"]
        if not friday_close.empty:
            closes.append(float(round(friday_close.dropna().iloc[-1], 3)))
        elif not week_data.empty and "Close" in week_data.columns:
            non_nan_closes = week_data["Close"].dropna()
            closes.append(float(round(non_nan_closes.iloc[-1], 3)) if not non_nan_closes.empty else np.nan)
        else:
            closes.append(np.nan)
    return closes

def fetch_yesterday_close(symbol):
    yesterday = datetime.today() - timedelta(days=1)
    data = yf.download(symbol, start=yesterday, end=yesterday + timedelta(days=1), interval="1d", progress=False)
    closes = data["Close"].dropna()
    return float(round(closes.iloc[-1], 3)) if not closes.empty else np.nan

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

@st.cache_data(show_spinner=False)
def get_live_names_and_countries(symbols):
    names, countries = {}, {}
    for sym in symbols:
        try:
            info = yf.Ticker(sym).info
            names[sym] = info.get("longName") or info.get("shortName") or sym
            countries[sym] = info.get("country") or ""
        except Exception:
            names[sym] = sym
            countries[sym] = ""
    return names, countries

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    symbols = df["Symbol"].dropna().unique().tolist()

    weeks, last_friday = get_last_n_weeks(6)
    week_labels = [f"{start.date()} to {end.date()}" for start, end in weeks]
    current_week_start = last_friday + timedelta(days=1)
    current_week_label = f"{current_week_start.date()} to {datetime.today().date()}"

    price_data = []
    for sym in symbols:
        closes = fetch_friday_closes(sym, weeks)
        current_close = fetch_yesterday_close(sym)
        closes.append(current_close)
        price_data.append([sym] + closes)

    all_labels = week_labels + [current_week_label]
    price_df = pd.DataFrame(price_data, columns=["Symbol"] + all_labels)

    st.subheader("Weekly % Price Change")
    pct_df = price_df.copy()
    # Correct % change calculation with manual last-week-to-yesterday column
    pct_df[all_labels[:-1]] = pct_df[all_labels[:-1]].pct_change(axis=1) * 100
    last_week_label = all_labels[-2]
    current_label = all_labels[-1]
    last_friday_close = price_df[last_week_label]
    yesterday_close = price_df[current_label]
    pct_df[current_label] = ((yesterday_close - last_friday_close) / last_friday_close) * 100
    st.dataframe(pct_df.round(2), use_container_width=True)

    st.subheader("Normalized Price Performance (Start = 100)")
    norm_df = price_df.set_index("Symbol")
    norm_df = norm_df[all_labels].div(norm_df[all_labels[0]], axis=0) * 100
    norm_df = norm_df.dropna(thresh=3)  # allow some NaNs
    st.line_chart(norm_df.T)

    tabs = st.tabs(["Raw Prices", "% Change Table", "Normalized Chart", "Ticker Scores", "Drawdowns", "Volatility"])

    with tabs[0]:
        st.subheader("Raw Prices")
        st.dataframe(price_df, use_container_width=True)

    with tabs[1]:
        st.subheader("% Weekly Change Table")
        st.dataframe(pct_df.round(2), use_container_width=True)

    with tabs[2]:
        st.subheader("Normalized Chart")
        st.line_chart(norm_df.T)

    with tabs[3]:
        st.subheader("Ticker Scores (5 Strategies)")
        scores_df = pd.DataFrame(index=norm_df.index)
        scores_df["Momentum"] = norm_df.iloc[:, -1] - norm_df.iloc[:, -2]
        scores_df["Volatility"] = norm_df.std(axis=1)
        scores_df["Trend"] = norm_df.apply(lambda row: sum(row.diff() > 0), axis=1)
        scores_df["Total Return"] = norm_df.apply(lambda row: row.iloc[-1] - row.iloc[0], axis=1)
        scores_df["All-Around"] = scores_df.sum(axis=1)
        st.dataframe(scores_df.round(2), use_container_width=True)

    with tabs[4]:
        st.subheader("📉 Max Drawdown by Ticker")
        drawdowns = norm_df.apply(lambda row: calculate_max_drawdown(row.dropna()), axis=1)
        fig = go.Figure(go.Bar(
            x=drawdowns.index,
            y=drawdowns.values,
            marker_color='indianred'
        ))
        fig.update_layout(
            title="Max Drawdown (Normalized %)",
            xaxis_title="Ticker",
            yaxis_title="Drawdown (%)",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(drawdowns.rename("Max Drawdown (%)").round(2).reset_index(), use_container_width=True)

    with tabs[5]:
        st.subheader("📊 Volatility Table (Standard Deviation of % Weekly Change)")
        weekly_pct_change = price_df.set_index("Symbol")[all_labels].pct_change(axis=1) * 100
        volatility_table = weekly_pct_change.std(axis=1).rename("Volatility (%)")
        st.dataframe(volatility_table.round(2).reset_index(), use_container_width=True)
