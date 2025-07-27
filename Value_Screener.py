import streamlit as st
import pandas as pd
import yfinance as yf

# ------------------------
# Settings
# ------------------------
st.set_page_config(page_title="📉 US Value Stock Screener", layout="wide")
st.title("📉 US Value Stock Screener (Low P/B, High ROE)")

# ------------------------
# Load universe (S&P 1500 sample)
# ------------------------
@st.cache_data
def load_universe():
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    df = pd.read_csv(url)
    return df

universe_df = load_universe()
universe_df = universe_df.rename(columns={"Symbol": "Ticker", "Name": "Company"})

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

sample_tickers = universe_df['Ticker'].unique().tolist()[:100]  # Limit to first 100 for demo
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
# Download option
# ------------------------
st.download_button(
    label="📥 Download Screener Data (XLSX)",
    data=filtered_df.to_excel(index=False, engine='openpyxl'),
    file_name="us_value_screener.xlsx"
)
