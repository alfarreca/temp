import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide")

st.title("📊 Weekly Price Tracker")

uploaded_file = st.file_uploader("Upload your Excel file", type=[".xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet_name = st.selectbox("Select sheet to analyze", xls.sheet_names)
    df = pd.read_excel(xls, sheet_name=sheet_name)

    if 'Symbol' not in df.columns:
        st.error("The selected sheet must contain a 'Symbol' column.")
    else:
        symbols = df['Symbol'].dropna().unique().tolist()

        @st.cache_data(show_spinner=False)
        def fetch_friday_closes(symbol, weeks=6):
            end = datetime.today()
            start = end - timedelta(weeks=weeks * 2)
            try:
                data = yf.download(symbol, start=start, end=end, progress=False)
                if data.empty:
                    return [np.nan] * weeks
                data = data.resample('D').ffill()
                closes = []
                for i in range(weeks):
                    week_end = end - timedelta(weeks=weeks - i - 1)
                    week_start = week_end - timedelta(days=6)
                    week_data = data[(data.index >= week_start) & (data.index <= week_end)]
                    friday_close = week_data.loc[week_data.index.weekday == 4, "Close"]
                    if not friday_close.dropna().empty:
                        closes.append(float(round(friday_close.dropna().iloc[-1], 3)))
                    elif not week_data["Close"].dropna().empty:
                        closes.append(float(round(week_data["Close"].dropna().iloc[-1], 3)))
                    else:
                        closes.append(np.nan)
                return closes
            except:
                return [np.nan] * weeks

        @st.cache_data(show_spinner=False)
        def fetch_current_price(symbol):
            try:
                data = yf.download(symbol, period="5d", interval="1d", progress=False)
                return float(round(data["Close"].dropna().iloc[-1], 3))
            except:
                return np.nan

        weeks = 6
        all_labels = []
        end = datetime.today()
        for i in range(weeks):
            week_end = end - timedelta(weeks=weeks - i - 1)
            week_start = week_end - timedelta(days=4)
            label = f"{week_start.date()} to {week_end.date()}"
            all_labels.append(label)

        price_data = []
        for sym in symbols:
            closes = fetch_friday_closes(sym, weeks)
            if all(np.isnan(closes)):
                continue
            last_price = fetch_current_price(sym)
            row = [sym] + closes + [last_price]
            price_data.append(row)

        price_df = pd.DataFrame(price_data, columns=["Symbol"] + all_labels + ["Current"])

        # Calculate percentage changes week over week
        for i in range(1, len(all_labels)):
            col_prev = all_labels[i - 1]
            col_curr = all_labels[i]
            if col_curr != all_labels[-1]:
                price_df[col_curr] = ((price_df[col_curr] - price_df[col_prev]) / price_df[col_prev] * 100).round(2)

        # Final column: last week close to current
        price_df["Last Week → Now"] = ((price_df["Current"] - price_df[all_labels[-1]]) / price_df[all_labels[-1]] * 100).round(2)

        # Normalize prices for chart
        norm_df = price_df.copy()
        norm_df = norm_df.drop(columns=["Current", "Last Week → Now"])
        norm_df.set_index("Symbol", inplace=True)
        norm_df = norm_df.div(norm_df.iloc[:, 0], axis=0) * 100

        # Drawdowns Calculation
        drawdowns = []
        for sym in norm_df.index:
            values = norm_df.loc[sym].values
            peak = values[0]
            max_draw = 0
            for v in values:
                if v > peak:
                    peak = v
                draw = (peak - v) / peak * 100
                if draw > max_draw:
                    max_draw = draw
            drawdowns.append(round(max_draw, 2))

        drawdown_df = pd.DataFrame({
            "Symbol": norm_df.index,
            "Max Drawdown %": drawdowns
        }).sort_values(by="Max Drawdown %", ascending=False).reset_index(drop=True)

        tabs = st.tabs(["Performance Table", "Drawdowns", "Normalized Chart"])

        with tabs[0]:
            st.subheader("Advanced Performance Metrics")
            st.dataframe(price_df.reset_index(drop=True), use_container_width=True)

        with tabs[1]:
            st.subheader("Maximum Drawdowns")
            st.dataframe(drawdown_df, use_container_width=True)

        with tabs[2]:
            st.subheader("Normalized Chart")
            fig = go.Figure()
            for sym in norm_df.index:
                percent_change = (norm_df.loc[sym].iloc[-1] - 100)
                fig.add_trace(go.Scatter(
                    x=norm_df.columns,
                    y=norm_df.loc[sym],
                    mode='lines+markers',
                    name=f"{sym} ({percent_change:+.1f}%)",
                    hovertemplate=f"%{{x}}<br>%{{y:.2f}}<extra>{sym}</extra>"
                ))
            fig.update_layout(
                title="Normalized Price Performance (Start = 100)",
                xaxis_title="Week",
                yaxis_title="Normalized Price",
                height=500,
                hovermode="closest"
            )
            st.plotly_chart(fig, use_container_width=True)
