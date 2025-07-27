import streamlit as st
import pandas as pd
import yfinance as yf
from io import BytesIO
import os

# ------------------------
# Settings
# ------------------------
st.set_page_config(page_title="📉 US Value Stock Screener", layout="wide")
st.title("📉 US Value Stock Screener (Low P/B, High ROE)")

# ------------------------
# Load universe (Russell 3000 Cleaned Excel)
# ------------------------
@st.cache_data
def load_universe():
    file_path = "Russell_3000_Cleaned.xlsx"
    if not os.path.exists(file_path):
        st.error(f"File '{file_path}' not found. Please upload it to your repo.")
        return pd.DataFrame()
    df = pd.read_excel(file_path)
    return df

universe_df = load_universe()
if universe_df.empty:
    st.stop()

# ------------------------
# Fetch financial data using yfinance
# ------------------------
@st.cache_data(show_spinner=True)
def fetch_metrics(tickers):
    data = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            row = {
                "Ticker": ticker,
                "Company": info.get("shortName"),
                "Sector": info.get("sector"),
                "P/B Ratio": info.get("priceToBook"),
                "ROE (TTM)": info.get("returnOnEquity"),
                "Debt/Equity": info.get("debtToEquity"),
                "Dividend Yield": info.get("dividendYield"),
                "Yahoo Finance": f"https://finance.yahoo.com/quote/{ticker}",
                "EDGAR Filings": f"https://www.sec.gov/edgar/browse/?CIK={ticker}"  
            }
            data.append(row)
        except:
            continue
    return pd.DataFrame(data)

# You can adjust the number of tickers here (500 for performance, or full universe)
sample_tickers = universe_df['Ticker'].unique().tolist()[:500]

with st.spinner("Fetching live financials..."):
    financials_df = fetch_metrics(sample_tickers)

# ------------------------
# Clean and filter
# ------------------------
financials_df = financials_df.dropna(subset=["P/B Ratio", "ROE (TTM)", "Debt/Equity"])
financials_df = financials_df[financials_df["P/B Ratio"] < 1.2]
financials_df = financials_df[financials_df["ROE (TTM)"] > 0]

# ------------------------
# Sidebar filters
# ------------------------
sectors = financials_df["Sector"].dropna().unique().tolist()
selected_sector = st.sidebar.multiselect("Filter by Sector", sectors, default=sectors)

filtered_df = financials_df[financials_df["Sector"].isin(selected_sector)]

# ------------------------
# Display table
# ------------------------
st.dataframe(
    filtered_df[["Ticker", "Company", "Sector", "P/B Ratio", "ROE (TTM)", "Debt/Equity", "Dividend Yield", "Yahoo Finance", "EDGAR Filings"]],
    use_container_width=True,
    hide_index=True
)

# ------------------------
# Download option (fixed)
# ------------------------
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

excel_data = convert_df_to_excel(filtered_df)

st.download_button(
    label="📥 Download Screener Data (XLSX)",
    data=excel_data,
    file_name="us_value_screener.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
) 
