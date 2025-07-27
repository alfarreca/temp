import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import time
from datetime import datetime

st.set_page_config(layout="wide")

st.title("📊 US Value & Dividend Stock Screener")

# Collapsible help
with st.expander("ℹ️ Screener Modes & Metrics Explained"):
    st.markdown("""
    ### Screener Modes:
    - **Value**: P/B < 1.2, ROE > 0
    - **Growth**: EPS and Revenue Growth > 10%
    - **Dividend**: Dividend Yield > 3%, Payout Ratio < 70%, ROE > 0
    - **Value + Dividend**: Combines Value + Dividend criteria

    ### Metrics:
    - **P/B Ratio**: Price-to-Book ratio — how expensive the stock is vs. net asset value
    - **ROE (TTM)**: Return on Equity over trailing twelve months — a profitability measure
    - **Debt/Equity**: How much debt the company uses to finance assets
    - **Dividend Yield**: Annual dividend / current price — cash return to shareholders
    - **Payout Ratio**: Portion of earnings paid out as dividends (lower = more sustainable)
    - **Yahoo Finance**: Direct link to Yahoo stock page
    - **EDGAR Filings**: Direct link to SEC 10-K/10-Q filings
    """)

# Upload Excel
uploaded_file = st.file_uploader("Upload Russell 3000 Excel File", type=["xlsx"])

@st.cache_data
def load_data(file):
    return pd.read_excel(file)

if uploaded_file:
    df = load_data(uploaded_file)
    screener_mode = st.selectbox("Select Screener Mode", ["Value", "Growth", "Dividend", "Value + Dividend"])

    # Standardize column names
    df = df.rename(columns={
        'Symbol': 'Ticker',
        'Exchange': 'Company'
    })

    # Clean up ticker names
    if 'Ticker' not in df.columns:
        st.error("❌ 'Symbol' column is required in uploaded file.")
        st.stop()
    df = df.dropna(subset=['Ticker'])
    df['Ticker'] = df['Ticker'].astype(str).str.upper()

    # Limit number of stocks to process (optional)
    max_stocks = st.slider("Max stocks to analyze (reduce for faster results)", 100, 3000, 500)
    df = df.head(max_stocks)

    # Fetch metrics from Yahoo Finance with rate limiting
    @st.cache_data
    def fetch_metrics(ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'P/B Ratio': info.get('priceToBook'),
                'ROE (TTM)': info.get('returnOnEquity'),
                'Debt/Equity': info.get('debtToEquity'),
                'Dividend Yield': info.get('dividendYield') * 100 if info.get('dividendYield') else None,
                'Payout Ratio': info.get('payoutRatio'),
                'Yahoo Finance': f'https://finance.yahoo.com/quote/{ticker}',
                'EDGAR Filings': f'https://www.sec.gov/edgar/browse/?CIK={ticker}',
                'EPS Growth': info.get('earningsGrowth', 0),
                'Revenue Growth': info.get('revenueGrowth', 0),
                'Market Cap': info.get('marketCap')
            }
        except Exception as e:
            st.warning(f"Failed to fetch {ticker}: {str(e)}")
            return None  # Return None to filter out failed requests later

    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics = []
    tickers = df['Ticker'].unique().tolist()
    
    for i, ticker in enumerate(tickers):
        status_text.text(f"Fetching data for {ticker} ({i+1}/{len(tickers)})")
        data = fetch_metrics(ticker)
        if data:  # Only append if data was successfully fetched
            metrics.append(data)
        progress_bar.progress((i + 1) / len(tickers))
        time.sleep(0.5)  # Add delay to avoid rate limiting

    # Filter out None values from failed requests
    metrics = [m for m in metrics if m is not None]
    df_metrics = pd.DataFrame(metrics)
    
    if len(df_metrics) == 0:
        st.error("Failed to fetch data for any stocks. Please try again later.")
        st.stop()

    final_df = pd.merge(df, df_metrics, on='Ticker', how='inner')

    # Filter logic
    if screener_mode == "Value":
        filtered = final_df[(final_df['P/B Ratio'] < 1.2) & (final_df['ROE (TTM)'] > 0)]
    elif screener_mode == "Growth":
        filtered = final_df[(final_df['EPS Growth'] > 0.1) & (final_df['Revenue Growth'] > 0.1)]
    elif screener_mode == "Dividend":
        filtered = final_df[(final_df['Dividend Yield'] > 3) & (final_df['ROE (TTM)'] > 0) &
                            ((final_df['Payout Ratio'].isna()) | (final_df['Payout Ratio'] < 0.7))]
    elif screener_mode == "Value + Dividend":
        filtered = final_df[(final_df['P/B Ratio'] < 1.2) & (final_df['Dividend Yield'] > 3) &
                            (final_df['ROE (TTM)'] > 0) &
                            ((final_df['Payout Ratio'].isna()) | (final_df['Payout Ratio'] < 0.7))]

    # Sector filter
    if 'Sector' in filtered.columns:
        sector_options = filtered['Sector'].dropna().unique()
        selected_sectors = st.multiselect("Filter by Sector", sector_options)
        if selected_sectors:
            filtered = filtered[filtered['Sector'].isin(selected_sectors)]

    # Summary box
    st.subheader("📋 Screener Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Stocks", len(filtered))
    col2.metric("Avg P/B", round(filtered['P/B Ratio'].mean(), 2))
    col3.metric("Avg ROE", round(filtered['ROE (TTM)'].mean(), 2))
    col4.metric("Avg Yield", round(filtered['Dividend Yield'].dropna().mean(), 2) if not filtered['Dividend Yield'].dropna().empty else "N/A")

    # Final table
    st.subheader("📌 Filtered Results")
    preferred_columns = ['Ticker', 'Company', 'Sector', 'Industry', 'P/B Ratio', 'ROE (TTM)', 'Debt/Equity',
                         'Dividend Yield', 'Payout Ratio', 'Market Cap', 'Yahoo Finance', 'EDGAR Filings']
    display_columns = [col for col in preferred_columns if col in filtered.columns]
    st.dataframe(filtered[display_columns])

    # Download button
    csv = filtered[display_columns].to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name=f"{screener_mode}_screener_results.csv",
        mime="text/csv"
    )

    # Charts
    if not filtered.empty:
        col1, col2 = st.columns(2)
        with col1:
            if 'Sector' in filtered.columns:
                st.write("📊 Sector Distribution")
                st.bar_chart(filtered['Sector'].value_counts())

        with col2:
            st.write("📈 P/B vs ROE")
            fig = px.scatter(filtered, x='P/B Ratio', y='ROE (TTM)', hover_name='Ticker', 
                            color='Sector' if 'Sector' in filtered.columns else None,
                            size='Market Cap' if 'Market Cap' in filtered.columns else None,
                            size_max=15)
            st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Please upload your Russell 3000 Excel file to begin.")
