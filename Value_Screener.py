import streamlit as st
import pandas as pd
import yfinance as yf
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="📉 US Multi-Mode Stock Screener", layout="wide")
st.title("📉 US Multi-Mode Stock Screener")

with st.expander("ℹ️ Screener Modes & Metrics Explained"):
    st.markdown("""
    ### 🧭 Screener Modes

    **1. Value Mode**  
    - P/B Ratio < 1.2 → undervalued relative to net assets  
    - ROE > 0 → profitable  

    **2. Growth Mode**  
    - Revenue Growth YoY > 10%  
    - EPS Growth YoY > 10%  

    **3. Dividend Mode**  
    - Dividend Yield > 3%  
    - ROE > 0 → profitable  
    - Payout Ratio < 70% or not available  

    **4. Value + Dividend Mode**  
    - Combines filters from Value and Dividend modes  

    ### 📊 Key Metrics

    | Metric | Description |
    |--------|-------------|
    | **P/B Ratio** | Price-to-Book ratio – how expensive the stock is vs. net asset value |
    | **ROE (TTM)** | Return on Equity – profitability measure |
    | **Debt/Equity** | Leverage ratio – debt used to finance assets |
    | **Dividend Yield** | Annual dividend ÷ price – cash return to shareholders |
    | **Payout Ratio** | Dividend ÷ earnings – sustainability of dividends |
    | **Revenue Growth** | YoY revenue expansion |
    | **EPS Growth** | YoY earnings expansion |
    | **Yahoo Finance** | Direct link to stock page |
    | **EDGAR Filings** | Direct link to SEC filings |
    """)

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

screener_mode = st.sidebar.selectbox("Select Screener Mode", ["Value", "Growth", "Dividend", "Value + Dividend"])

st.subheader("🗃️ Raw Ticker Universe (Loaded from Upload)")
search_term = st.text_input("Search company or ticker:").upper()
filtered_universe_df = universe_df[universe_df.apply(lambda row: search_term in row['Ticker'] or search_term in row['Company'].upper(), axis=1)] if search_term else universe_df
st.dataframe(filtered_universe_df, use_container_width=True)

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
                "Payout Ratio": info.get("payoutRatio"),
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

mode_logic = {
    "Value": lambda df: df.dropna(subset=["P/B Ratio", "ROE (TTM)", "Debt/Equity"]).query("`P/B Ratio` < 1.2 and `ROE (TTM)` > 0"),
    "Growth": lambda df: df.dropna(subset=["Revenue Growth (YoY)", "EPS Growth (YoY)"]).query("`Revenue Growth (YoY)` > 0.1 and `EPS Growth (YoY)` > 0.1"),
    "Dividend": lambda df: df.dropna(subset=["Dividend Yield", "ROE (TTM)"]).query("`Dividend Yield` > 0.03 and `ROE (TTM)` > 0 and (`Payout Ratio`.isnull() or `Payout Ratio` < 0.7)"),
    "Value + Dividend": lambda df: df.dropna(subset=["P/B Ratio", "ROE (TTM)", "Dividend Yield"]).query("`P/B Ratio` < 1.2 and `ROE (TTM)` > 0 and `Dividend Yield` > 0.03 and (`Payout Ratio`.isnull() or `Payout Ratio` < 0.7)")
}

financials_df = mode_logic[screener_mode](financials_df)
sectors = financials_df["Sector"].dropna().unique().tolist()
selected_sector = st.sidebar.multiselect("Filter by Sector", sectors, default=sectors)
filtered_df = financials_df[financials_df["Sector"].isin(selected_sector)]

st.subheader(f"📊 {screener_mode} Screener Results")

st.markdown("### 📌 Screener Summary")
st.write(f"**Total Stocks:** {len(filtered_df)}")
col1, col2, col3 = st.columns(3)
col1.metric("Avg. P/B Ratio", f"{filtered_df['P/B Ratio'].mean():.2f}" if 'P/B Ratio' in filtered_df else "N/A")
col2.metric("Avg. ROE", f"{filtered_df['ROE (TTM)'].mean():.2%}" if 'ROE (TTM)' in filtered_df else "N/A")
col3.metric("Avg. Dividend Yield", f"{filtered_df['Dividend Yield'].mean():.2%}" if 'Dividend Yield' in filtered_df else "N/A")

st.markdown("### 📈 Sector Breakdown")
if not filtered_df.empty:
    sector_counts = filtered_df['Sector'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.bar(sector_counts.index, sector_counts.values)
    ax1.set_ylabel("Count")
    ax1.set_title("Stocks by Sector")
    ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig1)

    if screener_mode in ["Value", "Dividend", "Value + Dividend"] and "P/B Ratio" in filtered_df and "ROE (TTM)" in filtered_df:
        st.markdown("### 📉 P/B Ratio vs ROE Scatter")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=filtered_df, x="P/B Ratio", y="ROE (TTM)", hue="Sector", ax=ax2)
        ax2.set_title("P/B Ratio vs ROE")
        st.pyplot(fig2)

    if "Dividend Yield" in filtered_df.columns:
        st.markdown("### 📊 Dividend Yield Histogram")
        fig3, ax3 = plt.subplots()
        ax3.hist(filtered_df["Dividend Yield"].dropna(), bins=20, color='green', alpha=0.7)
        ax3.set_title("Distribution of Dividend Yields")
        ax3.set_xlabel("Dividend Yield")
        ax3.set_ylabel("Frequency")
        st.pyplot(fig3)

st.markdown("### 🧾 Screener Table")
st.dataframe(filtered_df, use_container_width=True, hide_index=True)

def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

excel_data = convert_df_to_excel(filtered_df)

st.download_button(
    label="📥 Download Screener Data (XLSX)",
    data=excel_data,
    file_name=f"us_{screener_mode.lower().replace(' + ', '_')}_screener.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
) 
