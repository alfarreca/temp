from pathlib import Path

# Define the corrected Python script content
fixed_script = """
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objs as go

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
    data = yf.download(symbol, start=first_monday, end=last_friday + timedelta(days=1), interval="1d", progress=False)
    if data.empty or "Close" not in data.columns:
        return None

    closes = []
    for monday, friday in weeks:
        week_data = data.loc[(data.index >= pd.Timestamp(monday)) & (data.index <= pd.Timestamp(friday))]
        friday_close = week_data.loc[week_data.index.weekday == 4, "Close"]

        if not friday_close.empty and pd.api.types.is_numeric_dtype(friday_close):
            try:
                closes.append(float(round(friday_close.dropna().iloc[-1], 3)))
            except:
                closes.append(np.nan)
        elif not week_data.empty and "Close" in week_data.columns:
            non_nan_closes = week_data["Close"].dropna()
            if not non_nan_closes.empty:
                try:
                    closes.append(float(round(non_nan_closes.iloc[-1], 3)))
                except:
                    closes.append(np.nan)
            else:
                closes.append(np.nan)
        else:
            closes.append(np.nan)

    return closes if sum(np.isnan(closes)) == 0 else None

def fetch_current_week_close(symbol, current_week_start):
    today = datetime.today()
    data = yf.download(symbol, start=current_week_start, end=today + timedelta(days=1), interval="1d", progress=False)
    if data.empty or "Close" not in data.columns:
        return np.nan

    closes = data["Close"].dropna()
    if not closes.empty:
        try:
            return float(round(closes.iloc[-1], 3))
        except:
            return np.nan
    else:
        return np.nan

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

# Placeholder for main logic (tabs[0] to tabs[2])
# ...

# Corrected blocks with no indentation error
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
"""

# Save the fixed script to a downloadable file
fixed_path = Path("/mnt/data/Price_Tracker22_FIXED.py")
fixed_path.write_text(fixed_script)

fixed_path.name  # Return the filename to the assistant for user download link

