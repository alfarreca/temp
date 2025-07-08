import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from fredapi import Fred

# --- Config ---
st.set_page_config(page_title="Finance Dashboard", layout="wide")
st.title("📈 Stock & Economic Data Dashboard")

# --- API Keys (Replace with yours) ---
FRED_API_KEY = st.secrets.get("FRED_API_KEY", "your_fred_api_key_here")  # Get free key: https://fred.stlouisfed.org/docs/api/api_key.html

# --- Sidebar Inputs ---
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Stock Ticker (e.g., AAPL)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# --- Fetch Stock Data (Yahoo Finance) ---
@st.cache_data  # Cache to avoid reloading data
def get_stock_data(ticker, start, end):
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start, end=end)
    return stock, hist

try:
    stock, stock_data = get_stock_data(ticker, start_date, end_date)
    st.subheader(f"Stock Data: {ticker}")

    # Display metadata
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Price", f"${stock.info.get('currentPrice', 'N/A')}")
        st.metric("Market Cap", f"${stock.info.get('marketCap', 'N/A'):,}")
    with col2:
        st.metric("52W High", f"${stock.info.get('fiftyTwoWeekHigh', 'N/A')}")
        st.metric("P/E Ratio", stock.info.get('trailingPE', 'N/A'))

    # Plot stock price
    fig_stock = px.line(stock_data, x=stock_data.index, y="Close", title=f"{ticker} Closing Price")
    st.plotly_chart(fig_stock, use_container_width=True)

except Exception as e:
    st.error(f"Error fetching stock data: {e}")

# --- Fetch Economic Data (FRED API) ---
st.subheader("Economic Indicators (FRED)")
fred = Fred(api_key=FRED_API_KEY)

# Example FRED series (replace with your preferred series)
fred_series = {
    "GDP": "GDPC1",  # Real GDP
    "Unemployment": "UNRATE",  # Unemployment Rate
    "Inflation": "CPIAUCSL",  # CPI
}

selected_series = st.selectbox("Select Economic Indicator", list(fred_series.keys()))
series_id = fred_series[selected_series]

@st.cache_data
def get_fred_data(series_id):
    data = fred.get_series(series_id)
    return data

try:
    econ_data = get_fred_data(series_id)
    st.write(f"**{selected_series}**")
    fig_econ = px.line(econ_data, title=f"{selected_series} Over Time")
    st.plotly_chart(fig_econ, use_container_width=True)
except Exception as e:
    st.error(f"Error fetching FRED data: {e}")

# --- Notes ---
st.markdown("---")
st.caption("💡 Data Sources: Yahoo Finance (free) & FRED API (requires key)")
