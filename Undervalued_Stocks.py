import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px

# App configuration
st.set_page_config(layout="wide")
st.title("📊 Undervalued Stock Screener")
st.subheader("Warren Buffett-inspired Value Investing Strategy")
st.write("""
This app helps identify potentially undervalued stocks based on fundamental analysis metrics.
You can analyze either a single stock or upload a file with multiple tickers.
""")

# Sidebar inputs
st.sidebar.header("Analysis Options")
analysis_type = st.sidebar.radio("Select analysis type:", 
                                ["Single Stock Analysis", "Multiple Stocks Analysis"])

# Single stock analysis
if analysis_type == "Single Stock Analysis":
    st.sidebar.header("Single Stock Parameters")
    single_ticker = st.sidebar.text_input(
        "Enter stock ticker:",
        help="Example: AAPL for Apple, MSFT for Microsoft"
    )
    exchange = st.sidebar.selectbox(
        "Select exchange (if international):",
        ["", "NYSE/NASDAQ", "TORONTO", "LONDON", "EURONEXT", "FRANKFURT", "HONG KONG", "SHANGHAI"]
    )

    if st.sidebar.button("Analyze Single Stock"):
        if single_ticker:
            df = pd.DataFrame({'Symbol': [single_ticker], 'Exchange': [exchange]})
            filtered_df, results_df = analyze_stocks(df)
            if results_df is not None:
                display_results(filtered_df, results_df, single_stock=True)
        else:
            st.warning("Please enter a stock ticker")

# Multiple stocks analysis
else:
    st.sidebar.header("File Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel file with stock tickers", 
        type=["xlsx"],
        help="File should contain 'Symbol' column (optional: 'Exchange')"
    )
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            if 'Symbol' not in df.columns:
                st.error("File must contain 'Symbol' column")
                st.stop()
            if 'Exchange' not in df.columns:
                df['Exchange'] = ''
            if st.button("Analyze Stocks"):
                filtered_df, results_df = analyze_stocks(df)
                if results_df is not None:
                    display_results(filtered_df, results_df)
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("""
        To begin analysis:
        1. Select analysis type (single stock or file upload)
        2. Enter stock ticker or upload file
        3. Adjust valuation parameters as needed
        4. Click analyze button
        """)
