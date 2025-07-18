import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta
import yfinance as yf
import calendar

st.set_page_config(layout="wide")
st.title("📈 Weekly Performance Tracker")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if not uploaded_file:
    st.stop()

xls = pd.ExcelFile(uploaded_file)
sheet = st.selectbox("Select sheet to analyze", xls.sheet_names)
df = pd.read_excel(uploaded_file, sheet_name=sheet)

if "Symbol" not in df.columns:
    st.error("Missing 'Symbol' column in the uploaded file.")
    st.stop()

symbols = df["Symbol"].dropna().unique().tolist()

@st.cache_data(show_spinner=False)
def fetch_weekly_closes(symbol: str, weeks: int):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=weeks * 2)
        data = yf.download(symbol, start=start_date, end=end_date)
        data = data[~data.index.duplicated(keep='last')]
        closes = []
        for i in range(weeks):
            ref_day = end_date - timedelta(days=end_date.weekday() + 3 + i * 7)
            ref_str = ref_day.strftime("%Y-%m-%d")
            weekly = data.loc[:ref_str]["Close"].dropna()
            if not weekly.empty:
                closes.append(round(weekly.iloc[-1], 2))
            else:
                closes.append(None)
        return closes[::-1]
    except Exception:
        return [None] * weeks

weeks_to_show = 6
all_prices = []
for sym in symbols:
    closes = fetch_weekly_closes(sym, weeks_to_show)
    all_prices.append([sym] + closes)

labels = []
now = datetime.now()
for i in range(weeks_to_show):
    start = now - timedelta(days=now.weekday() + 3 + (weeks_to_show - i - 1) * 7)
    end = start + timedelta(days=4)
    labels.append(f"{start.date()} to {end.date()}")

price_df = pd.DataFrame(all_prices, columns=["Symbol"] + labels)
price_df.set_index("Symbol", inplace=True)

norm_df = price_df.div(price_df.iloc[:, 0], axis=0) * 100
norm_df = norm_df.clip(lower=0, upper=10000)

fig = go.Figure()
for sym in norm_df.index:
    series = norm_df.loc[sym]
    percent = (series.iloc[-1] / series.iloc[0] - 1) * 100 if series.iloc[0] else 0
    if abs(percent) > 1000:
        continue
    fig.add_trace(go.Scatter(
        x=labels,
        y=series,
        mode='lines+markers',
        name=f"{sym} ({percent:+.1f}%)",
        hovertemplate=f"<b>{sym}</b><br>%{{x}}<br>%{{y:.2f}}",
    ))

fig.update_layout(
    title="Normalized Price Performance (Start = 100)",
    xaxis_title="Week",
    yaxis_title="Normalized Price",
    hovermode="closest",
    height=500,
    showlegend=True
)

st.subheader("Normalized Chart")
st.plotly_chart(fig, use_container_width=True)

# Tabs for layout
with st.container():
    tabs = st.tabs(["Performance Table", "Drawdowns", "Normalized Chart"])

with tabs[0]:
    st.dataframe(price_df.style.format("{:.2f}"), use_container_width=True)

with tabs[1]:
    drawdowns = norm_df.copy()
    for sym in drawdowns.index:
        peak = drawdowns.loc[sym].cummax()
        drawdowns.loc[sym] = (drawdowns.loc[sym] - peak) / peak * 100
    st.dataframe(drawdowns.style.format("{:.1f}"), use_container_width=True)

with tabs[2]:
    st.plotly_chart(fig, use_container_width=True)
