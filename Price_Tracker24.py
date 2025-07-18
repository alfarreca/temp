import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
import datetime
import copy

st.set_page_config(layout="wide")
st.title("📈 Weekly Price Tracker")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file:
    excel = pd.ExcelFile(uploaded_file)
    sheet_name = st.selectbox("Select sheet to analyze", excel.sheet_names)
    df = pd.read_excel(excel, sheet_name=sheet_name)
    df.columns = df.columns.str.strip()
    symbols = df["Symbol"].dropna().unique().tolist()

    today = datetime.date.today()
    weekday = today.weekday()
    last_friday = today - datetime.timedelta(days=(weekday - 4) % 7 + 7)
    weeks = [(last_friday - datetime.timedelta(days=7 * i)) for i in reversed(range(6))]

    @st.cache_data(show_spinner=False)
    def fetch_friday_closes(symbol, weeks):
        closes = []
        for week in weeks:
            start = week - datetime.timedelta(days=3)
            end = week + datetime.timedelta(days=3)
            try:
                data = yf.download(symbol, start=start, end=end, progress=False)
                if data.empty:
                    closes.append(np.nan)
                    continue
                week_data = data.copy()
                week_data.index = pd.to_datetime(week_data.index)
                friday_close = week_data.loc[week_data.index.weekday == 4, "Close"]
                if not friday_close.dropna().empty:
                    closes.append(float(round(friday_close.dropna().iloc[-1], 3)))
                elif not week_data["Close"].dropna().empty:
                    closes.append(float(round(week_data["Close"].dropna().iloc[-1], 3)))
                else:
                    closes.append(np.nan)
            except:
                closes.append(np.nan)
        return closes

    @st.cache_data(show_spinner=False)
    def fetch_yesterday_close(symbol):
        end_date = today
        start_date = today - datetime.timedelta(days=5)
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if data.empty:
                return np.nan
            return float(round(data["Close"].dropna().iloc[-1], 3))
        except:
            return np.nan

    price_data = []
    week_labels = [f"{week} to {week + datetime.timedelta(days=4)}" for week in weeks]
    current_week_label = None
    if weekday in [0, 1, 2, 3, 4]:
        current_week_label = f"{last_friday + datetime.timedelta(days=7)} to {last_friday + datetime.timedelta(days=11)}"

    all_labels = week_labels.copy()
    if current_week_label:
        all_labels.append(current_week_label)

    for sym in symbols:
        closes = fetch_friday_closes(sym, weeks)
        if current_week_label:
            current_close = fetch_yesterday_close(sym)
            closes.append(current_close)
        price_data.append([sym] + closes)

    price_df = pd.DataFrame(price_data, columns=["Symbol"] + all_labels)
    price_df.set_index("Symbol", inplace=True)

    with st.expander("📊 Show Raw Weekly Closes"):
        st.dataframe(price_df, use_container_width=True)

    norm_df = price_df.copy()
    norm_df = norm_df.div(norm_df.iloc[:, 0], axis=0) * 100
    norm_df = norm_df.dropna(how="all")

    st.subheader("Normalized Performance")
    fig = go.Figure()
    for symbol in norm_df.index:
        fig.add_trace(go.Scatter(
            x=norm_df.columns,
            y=norm_df.loc[symbol],
            mode='lines+markers',
            name=f"{symbol} ({(norm_df.loc[symbol].iloc[-1] - 100):+.1f}%)"
        ))
    fig.update_layout(
        title="Normalized Price Performance (Start = 100)",
        xaxis_title="Week",
        yaxis_title="Normalized Price",
        height=500,
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    tabs = st.tabs(["Performance Table", "Drawdowns", "Normalized Chart"])

    with tabs[0]:
        pct_change_df = price_df.pct_change(axis=1) * 100
        pct_change_df = pct_change_df.round(2)
        pct_change_df = pct_change_df.iloc[:, 1:]  # Skip first week (no previous)
        pct_change_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        st.dataframe(pct_change_df, use_container_width=True)

    with tabs[2]:
        st.subheader("Normalized Chart")
        st.plotly_chart(copy.deepcopy(fig), use_container_width=True)
