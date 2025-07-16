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

# Valuation parameters
st.sidebar.header("Valuation Parameters")
with st.sidebar.expander("Adjust valuation thresholds"):
    pe_ratio_threshold = st.slider("Max P/E Ratio", 5, 50, 25)
    peg_ratio_threshold = st.slider("Max PEG Ratio", 0.1, 3.0, 1.5, step=0.1, format="%.1f")
    debt_to_equity_threshold = st.slider("Max Debt-to-Equity", 0.1, 5.0, 2.0, step=0.1, format="%.1f")
    current_ratio_threshold = st.slider("Min Current Ratio", 0.5, 3.0, 1.0, step=0.1, format="%.1f")
    roa_threshold = st.slider("Min Return on Assets (%)", 0, 20, 5)

def get_stock_data(ticker, exchange=""):
    try:
        exchange_map = {
            "TORONTO": ".TO", "LONDON": ".L", "EURONEXT": ".PA", "FRANKFURT": ".DE",
            "HONG KONG": ".HK", "SHANGHAI": ".SS"
        }
        full_ticker = ticker + exchange_map.get(exchange.upper(), "")
        stock = yf.Ticker(full_ticker)
        hist = stock.history(period="1d")

        if hist.empty:
            return None

        current_price = hist['Close'].iloc[-1]
        info = stock.info

        # Manual P/E fallback
        eps = info.get('trailingEps', np.nan)
        pe_ratio = info.get('trailingPE', np.nan)
        if (not pe_ratio or np.isnan(pe_ratio)) and eps and eps != 0:
            pe_ratio = current_price / eps
        if pe_ratio is not None and pe_ratio < 0:
            pe_ratio = None

        # Manual Forward P/E
        forward_eps = info.get('forwardEps', np.nan)
        forward_pe = info.get('forwardPE', np.nan)
        if (not forward_pe or np.isnan(forward_pe)) and forward_eps and forward_eps != 0:
            forward_pe = current_price / forward_eps
        if forward_pe is not None and forward_pe < 0:
            forward_pe = None

        # Manually calculate PEG ratio
        growth = info.get('earningsGrowth', np.nan)
        peg_ratio = None
        if pe_ratio is not None and pe_ratio > 0 and growth is not None and growth > 0:
            peg_ratio = pe_ratio / (growth * 100)

        # Manually calculate Dividend Yield
        dividend = info.get('dividendRate', np.nan)
        div_yield = None
        if dividend and dividend > 0 and current_price > 0:
            div_yield = (dividend / current_price) * 100

        metrics = {
            'Symbol': ticker,
            'Exchange': exchange,
            'CurrentPrice': current_price,
            'P/E': pe_ratio,
            'Forward P/E': forward_pe,
            'PEG': peg_ratio,
            'Debt/Equity': info.get('debtToEquity', np.nan),
            'CurrentRatio': info.get('currentRatio', np.nan),
            'ROA': info.get('returnOnAssets', np.nan) * 100 if info.get('returnOnAssets') else np.nan,
            'ROE': info.get('returnOnEquity', np.nan) * 100 if info.get('returnOnEquity') else np.nan,
            'ProfitMargin': info.get('profitMargins', np.nan) * 100 if info.get('profitMargins') else np.nan,
            'Price/Book': current_price / info.get('bookValue', np.nan) if info.get('bookValue') else np.nan,
            'DividendYield': div_yield,
            '52WeekLow': info.get('fiftyTwoWeekLow', np.nan),
            '52WeekHigh': info.get('fiftyTwoWeekHigh', np.nan),
            'DiscountFromHigh': (info.get('fiftyTwoWeekHigh', np.nan) - current_price) / info.get('fiftyTwoWeekHigh', 1) * 100 
                                if info.get('fiftyTwoWeekHigh') else np.nan,
            'Beta': info.get('beta', np.nan),
            'MarketCap': info.get('marketCap', np.nan)
        }
        return metrics

    except Exception as e:
        st.warning(f"Error fetching data for {ticker}: {str(e)}")
        return None

def analyze_stocks(tickers_df):
    st.write("## Analyzing Stocks...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []
    total_stocks = len(tickers_df)

    for i, row in tickers_df.iterrows():
        ticker = row['Symbol']
        exchange = row.get('Exchange', '')
        status_text.text(f"Processing {ticker} ({i+1}/{total_stocks})...")
        progress_bar.progress((i+1)/total_stocks)
        metrics = get_stock_data(ticker, exchange)
        if metrics:
            results.append(metrics)

    if not results:
        st.error("No valid stock data could be retrieved from Yahoo Finance.")
        return None, None

    results_df = pd.DataFrame(results)

    filtered_df = results_df[
        (results_df['P/E'].fillna(np.inf) <= pe_ratio_threshold) &
        (results_df['PEG'].fillna(np.inf) <= peg_ratio_threshold) &
        (results_df['Debt/Equity'].fillna(np.inf) <= debt_to_equity_threshold) &
        (results_df['CurrentRatio'].fillna(0) >= current_ratio_threshold) &
        (results_df['ROA'].fillna(0) >= roa_threshold)
    ].copy()

    weights = {'P/E': 0.3, 'PEG': 0.25, 'Debt/Equity': 0.2, 'CurrentRatio': 0.15, 'ROA': 0.1}

    filtered_df['Score'] = (
        (filtered_df['P/E'].fillna(0) / pe_ratio_threshold * weights['P/E']) +
        (filtered_df['PEG'].fillna(0) / peg_ratio_threshold * weights['PEG']) +
        (filtered_df['Debt/Equity'].fillna(0) / debt_to_equity_threshold * weights['Debt/Equity']) +
        (current_ratio_threshold / filtered_df['CurrentRatio'].fillna(np.inf) * weights['CurrentRatio']) +
        (roa_threshold / filtered_df['ROA'].fillna(np.inf) * weights['ROA'])
    )

    filtered_df = filtered_df.sort_values(by='Score', ascending=True)
    return filtered_df, results_df

def display_results(filtered_df, results_df, single_stock=False):
    st.write("---")
    st.write("## Raw Stock Data from Yahoo Finance")
    if results_df is not None and not results_df.empty:
        st.dataframe(results_df)
    if filtered_df is not None and not filtered_df.empty:
        st.write("---")
        st.success(f"Found {len(filtered_df)} potentially undervalued {'stock' if single_stock else 'stocks'}")
        st.write("## Filtered Results")
        st.dataframe(filtered_df)

# Sidebar inputs
st.sidebar.header("Analysis Options")
analysis_type = st.sidebar.radio("Select analysis type:", 
                                ["Single Stock Analysis", "Multiple Stocks Analysis"])

# Single stock analysis
if analysis_type == "Single Stock Analysis":
    st.sidebar.header("Single Stock Parameters")
    single_ticker = st.sidebar.text_input("Enter stock ticker:", help="Example: AAPL for Apple, MSFT for Microsoft")
    exchange = st.sidebar.selectbox("Select exchange (if international):",
                                    ["", "NYSE/NASDAQ", "TORONTO", "LONDON", "EURONEXT", "FRANKFURT", "HONG KONG", "SHANGHAI"])

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
    uploaded_file = st.sidebar.file_uploader("Upload Excel file with stock tickers", type=["xlsx"],
                                             help="File should contain 'Symbol' column (optional: 'Exchange')")
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
