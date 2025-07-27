import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import matplotlib.pyplot as plt
import io

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

    # Clean up ticker names
    df = df.dropna(subset=['Ticker'])
    df['Ticker'] = df['Ticker'].astype(str).str.upper()

    # Fetch metrics from Yahoo Finance
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
                'EDGAR Filings': f'https://www.sec.gov/edgar/browse/?CIK={ticker}'
            }
        except:
            return {}

    progress = st.progress(0)
    metrics = []
    tickers = df['Ticker'].unique().tolist()
    for i, ticker in enumerate(tickers):
        data = fetch_metrics(ticker)
        data['Ticker'] = ticker
        metrics.append(data)
        progress.progress((i + 1) / len(tickers))

    df_metrics = pd.DataFrame(metrics)
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

    # Raw data preview
    with st.expander("📄 Raw Data Preview"):
        st.dataframe(final_df)

    # Final table
    st.subheader("📌 Filtered Results")
    st.dataframe(filtered[['Ticker', 'Company', 'Sector', 'P/B Ratio', 'ROE (TTM)', 'Debt/Equity',
                           'Dividend Yield', 'Payout Ratio', 'Yahoo Finance', 'EDGAR Filings']])

    # Charts in 2-column layout
    col1, col2 = st.columns(2)
    with col1:
        st.write("📊 Sector Distribution")
        st.bar_chart(filtered['Sector'].value_counts())

    with col2:
        st.write("📈 P/B vs ROE")
        fig = px.scatter(filtered, x='P/B Ratio', y='ROE (TTM)', hover_name='Ticker', size_max=10)
        st.plotly_chart(fig, use_container_width=True)

    with col1:
        if "Dividend" in screener_mode:
            st.write("📉 Dividend Yield Histogram")
            fig2 = px.histogram(filtered, x='Dividend Yield', nbins=15)
            st.plotly_chart(fig2, use_container_width=True)

    with col2:
        if "Dividend" in screener_mode:
            st.write("🟠 Payout Ratio Histogram")
            fig3 = px.histogram(filtered, x='Payout Ratio', nbins=15)
            st.plotly_chart(fig3, use_container_width=True)
else:
    st.warning("Please upload your Russell 3000 Excel file to begin.")
