import streamlit as st
import pandas as pd
import yfinance as yf
from io import BytesIO
import os

# ------------------------
# Settings
# ------------------------
st.set_page_config(page_title="📉 US Multi-Mode Stock Screener", layout="wide")
st.title("📉 US Multi-Mode Stock Screener")

# ------------------------
# Upload or load universe file
# ------------------------
@st.cache_data
def load_universe_from_file(uploaded_file):
    df = pd.read_excel(uploaded_file)
    return df

uploaded_file = st.sidebar.file_uploader("📁 Upload Russell 3000 Excel File", type=["xlsx"])

if uploaded_file:
    universe_df = load_universe_from_file(uploaded_file)
else:
    st.warning("Please upload your Russell_3000_Cleaned.xlsx file to begin.")
    st.stop()

# ------------------------
# Sidebar: Screener mode selection
# ------------------------
screener_mode = st.sidebar.selectbox("Select Screener Mode", ["Value", "Growth"])

# ------------------------
# Show raw data with search
# ------------------------
st.subheader("🗃️ Raw Ticker Universe (Loaded from Upload)")
search_term = st.text_input("Search company or ticker:").upper()
filtered_universe_df = universe_df[universe_df.apply(lambda row: search_term in row['Ticker'] or search_term in row['Company'].upper(), axis=1)] if search_term else universe_df
st.dataframe(filtered_universe_df, use_container_width=True)

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
                "Revenue Growth (YoY)": info.get("revenueGrowth"),
                "EPS Growth (YoY)": info.get("earningsQuarterlyGrowth"),
                "Yahoo Finance": f"https://finance.yahoo.com/quote/{ticker}",
                "EDGAR Filings": f"https://www.sec.gov/edgar/browse/?CIK={ticker}"  
            }
            data.append(row)
        except:
            continue
    return pd.DataFrame(data)

sample_tickers = universe_df['Ticker'].unique().tolist()[:500]
with st.spinner("Fetching live financials..."):
    financials_df = fetch_metrics(sample_tickers)

# ------------------------
# Clean and filter based on mode
# ------------------------
if screener_mode == "Value":
    financials_df = financials_df.dropna(subset=["P/B Ratio", "ROE (TTM)", "Debt/Equity"])
    financials_df = financials_df[financials_df["P/B Ratio"] < 1.2]
    financials_df = financials_df[financials_df["ROE (TTM)"] > 0]
    sectors = financials_df["Sector"].dropna().unique().tolist()
    selected_sector = st.sidebar.multiselect("Filter by Sector", sectors, default=sectors)
    filtered_df = financials_df[financials_df["Sector"].isin(selected_sector)]

    st.subheader("📊 Value Screener Results")
    with st.expander("ℹ️ Column Definitions - Value"):
        st.markdown("""
        - **P/B Ratio**: Price-to-Book ratio — how expensive the stock is vs. net asset value  
        - **ROE (TTM)**: Return on Equity over the trailing twelve months — a profitability measure  
        - **Debt/Equity**: A leverage ratio — how much debt the company uses to finance assets  
        - **Dividend Yield**: Annual dividend / current price — cash return to shareholders  
        - **Yahoo Finance**: Direct link to the company's stock page on Yahoo  
        - **EDGAR Filings**: Direct link to the company's SEC filings (10-Ks, 10-Qs, etc.)  
        """)

    st.dataframe(
        filtered_df[["Ticker", "Company", "Sector", "P/B Ratio", "ROE (TTM)", "Debt/Equity", "Dividend Yield", "Yahoo Finance", "EDGAR Filings"]],
        use_container_width=True,
        hide_index=True
    )

elif screener_mode == "Growth":
    financials_df = financials_df.dropna(subset=["Revenue Growth (YoY)", "EPS Growth (YoY)"])
    financials_df = financials_df[financials_df["Revenue Growth (YoY)"] > 0.1]
    financials_df = financials_df[financials_df["EPS Growth (YoY)"] > 0.1]
    sectors = financials_df["Sector"].dropna().unique().tolist()
    selected_sector = st.sidebar.multiselect("Filter by Sector", sectors, default=sectors)
    filtered_df = financials_df[financials_df["Sector"].isin(selected_sector)]

    st.subheader("📈 Growth Screener Results")
    with st.expander("ℹ️ Column Definitions - Growth"):
        st.markdown("""
        - **Revenue Growth (YoY)**: Year-over-year sales/revenue growth  
        - **EPS Growth (YoY)**: Year-over-year earnings-per-share growth  
        - **Yahoo Finance**: Direct link to the company's stock page on Yahoo  
        - **EDGAR Filings**: Direct link to the company's SEC filings (10-Ks, 10-Qs, etc.)  
        """)

    st.dataframe(
        filtered_df[["Ticker", "Company", "Sector", "Revenue Growth (YoY)", "EPS Growth (YoY)", "Yahoo Finance", "EDGAR Filings"]],
        use_container_width=True,
        hide_index=True
    )

# ------------------------
# Download option
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
    file_name=f"us_{screener_mode.lower()}_screener.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
) 
