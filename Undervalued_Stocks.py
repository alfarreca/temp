import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px

# App title and description
st.title("📊 Undervalued Stock Screener")
st.subheader("Warren Buffett-inspired Value Investing Strategy")
st.write("""
This app helps identify potentially undervalued stocks based on fundamental analysis metrics.
You can either upload an Excel file with multiple stocks or analyze a single ticker.
""")

# Initialize session state for button click
if 'analyze_clicked' not in st.session_state:
    st.session_state.analyze_clicked = False

# Function definitions moved to the top
def display_results(filtered_df, results_df):
    if len(filtered_df) > 0:
        st.success(f"Found {len(filtered_df)} potentially undervalued stocks out of {len(results_df)} analyzed")
        st.write("## Undervalued Stock Candidates")
        
        # Format display
        display_df = filtered_df[[
            'Symbol', 'Exchange', 'CurrentPrice', 'P/E', 'PEG', 'Price/Book',
            'Debt/Equity', 'CurrentRatio', 'ROA', 'ROE', 'ProfitMargin',
            'DividendYield', 'DiscountFromHigh', 'Score'
        ]].copy()
        
        st.dataframe(
            display_df.style.format({
                'CurrentPrice': '${:.2f}',
                'P/E': '{:.1f}',
                'PEG': '{:.1f}',
                'Price/Book': '{:.1f}',
                'Debt/Equity': '{:.1f}',
                'CurrentRatio': '{:.1f}',
                'ROA': '{:.1f}%',
                'ROE': '{:.1f}%',
                'ProfitMargin': '{:.1f}%',
                'DividendYield': '{:.1f}%',
                'DiscountFromHigh': '{:.1f}%',
                'Score': '{:.2f}'
            }),
            height=400
        )
        
        # Visualizations
        st.write("## Valuation Metrics Distribution")
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(
                results_df, 
                y='P/E',
                title='P/E Ratio Distribution',
                points="all"
            )
            fig.add_hline(y=pe_ratio_threshold, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                results_df, 
                y='Price/Book',
                title='Price-to-Book Distribution',
                points="all"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Download button
        st.download_button(
            label="Download Results as CSV",
            data=filtered_df.to_csv(index=False),
            file_name="undervalued_stocks.csv",
            mime="text/csv"
        )
    else:
        st.warning("""
        No stocks met all the valuation criteria. Try adjusting the parameters:
        - Increase P/E or PEG thresholds
        - Decrease Debt-to-Equity requirement
        - Lower ROA requirement
        """)

def display_single_ticker_results(filtered_df, results_df, ticker):
    if len(filtered_df) > 0:
        st.success(f"{ticker} appears to be undervalued based on the current criteria")
        st.write("## Analysis Results")
        
        # Format display for single stock
        display_df = filtered_df[[
            'Symbol', 'CurrentPrice', 'P/E', 'PEG', 'Price/Book',
            'Debt/Equity', 'CurrentRatio', 'ROA', 'ROE', 'ProfitMargin',
            'DividendYield', 'DiscountFromHigh', 'Score'
        ]].copy()
        
        st.dataframe(
            display_df.style.format({
                'CurrentPrice': '${:.2f}',
                'P/E': '{:.1f}',
                'PEG': '{:.1f}',
                'Price/Book': '{:.1f}',
                'Debt/Equity': '{:.1f}',
                'CurrentRatio': '{:.1f}',
                'ROA': '{:.1f}%',
                'ROE': '{:.1f}%',
                'ProfitMargin': '{:.1f}%',
                'DividendYield': '{:.1f}%',
                'DiscountFromHigh': '{:.1f}%',
                'Score': '{:.2f}'
            })
        )
        
        # Show all metrics for the single stock
        st.write("### Detailed Metrics")
        detailed_df = results_df.drop(columns=['Exchange']).T
        detailed_df.columns = ['Value']
        st.table(detailed_df.style.format({
            'Value': lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x
        }))
    else:
        st.info(f"{ticker} does not appear to be undervalued based on the current criteria")

def get_stock_data(ticker, exchange):
    try:
        # Add exchange suffix if needed
        full_ticker = ticker
        if exchange and exchange.upper() in ['TORONTO', 'TSX', 'CN']:
            full_ticker += '.TO'
        elif exchange and exchange.upper() in ['LONDON', 'LSE', 'UK']:
            full_ticker += '.L'
        elif exchange and exchange.upper() in ['EURONEXT', 'PARIS', 'FP']:
            full_ticker += '.PA'
        elif exchange and exchange.upper() in ['FRANKFURT', 'FRA', 'DE']:
            full_ticker += '.DE'
        
        stock = yf.Ticker(full_ticker)
        
        # Get current price
        hist = stock.history(period="1d")
        if hist.empty:
            return None
        current_price = hist['Close'].iloc[-1]
        
        # Get financials
        info = stock.info
        
        # Calculate metrics
        metrics = {
            'Symbol': ticker,
            'Exchange': exchange,
            'CurrentPrice': current_price,
            'P/E': info.get('trailingPE', np.nan),
            'PEG': info.get('pegRatio', np.nan),
            'Debt/Equity': info.get('debtToEquity', np.nan),
            'CurrentRatio': info.get('currentRatio', np.nan),
            'ROA': info.get('returnOnAssets', np.nan) * 100 if info.get('returnOnAssets') else np.nan,
            'ROE': info.get('returnOnEquity', np.nan) * 100 if info.get('returnOnEquity') else np.nan,
            'ProfitMargin': info.get('profitMargins', np.nan) * 100 if info.get('profitMargins') else np.nan,
            'MarketCap': info.get('marketCap', np.nan),
            'EPS': info.get('trailingEps', np.nan),
            'BookValue': info.get('bookValue', np.nan),
            'Price/Book': current_price / info['bookValue'] if (current_price and info.get('bookValue')) else np.nan,
            'DividendYield': info.get('dividendYield', np.nan) * 100 if info.get('dividendYield') else 0,
            '52WeekLow': info.get('fiftyTwoWeekLow', np.nan),
            '52WeekHigh': info.get('fiftyTwoWeekHigh', np.nan),
            'DiscountFromHigh': (info['fiftyTwoWeekHigh'] - current_price) / info['fiftyTwoWeekHigh'] * 100 
                               if (current_price and info.get('fiftyTwoWeekHigh')) else np.nan
        }
        return metrics
        
    except Exception as e:
        st.warning(f"Error fetching data for {ticker}: {str(e)}")
        return None

def analyze_stocks(df):
    try:
        st.write("## Analyzing Stocks...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        total_stocks = len(df)
        
        for i, row in df.iterrows():
            ticker = row['Symbol']
            exchange = row.get('Exchange', '')
            
            status_text.text(f"Processing {ticker} ({i+1}/{total_stocks})...")
            progress_bar.progress((i+1)/total_stocks)
            
            metrics = get_stock_data(ticker, exchange)
            if metrics:
                results.append(metrics)
        
        if not results:
            st.error("No valid stock data could be retrieved. Please check your tickers and try again.")
            return None, None
            
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Apply Buffett-style filters
        filtered_df = results_df[
            (results_df['P/E'] <= pe_ratio_threshold) &
            (results_df['PEG'] <= peg_ratio_threshold) &
            (results_df['Debt/Equity'] <= debt_to_equity_threshold) &
            (results_df['CurrentRatio'] >= current_ratio_threshold) &
            (results_df['ROA'] >= roa_threshold)
        ].copy()
        
        # Calculate composite score (lower is better)
        filtered_df['Score'] = (
            filtered_df['P/E'].fillna(0) / pe_ratio_threshold +
            filtered_df['PEG'].fillna(0) / peg_ratio_threshold +
            filtered_df['Debt/Equity'].fillna(0) / debt_to_equity_threshold +
            (current_ratio_threshold / filtered_df['CurrentRatio'].fillna(np.inf))
        ) / 4
        
        # Sort by score
        filtered_df = filtered_df.sort_values(by='Score', ascending=True)
        
        return filtered_df, results_df
        
    except Exception as e:
        st.error(f"Error processing stocks: {str(e)}")
        return None, None

# Sidebar for user inputs
st.sidebar.header("Valuation Parameters")

# Single ticker input
single_ticker = st.sidebar.text_input(
    "Enter a single stock ticker:",
    help="Example: AAPL for Apple, MSFT for Microsoft"
)

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Or upload Excel file with multiple tickers", 
    type=["xlsx"],
    help="File should contain 'Symbol' and 'Exchange' columns"
)

# Valuation parameters
pe_ratio_threshold = st.sidebar.slider(
    "Max P/E Ratio", 
    min_value=5, max_value=30, value=15, step=1
)

peg_ratio_threshold = st.sidebar.slider(
    "Max PEG Ratio", 
    min_value=0.5, max_value=2.0, value=1.0, step=0.1,
    format="%.1f"
)

debt_to_equity_threshold = st.sidebar.slider(
    "Max Debt-to-Equity", 
    min_value=0.1, max_value=2.0, value=0.5, step=0.1,
    format="%.1f"
)

current_ratio_threshold = st.sidebar.slider(
    "Min Current Ratio", 
    min_value=1.0, max_value=3.0, value=1.5, step=0.1,
    format="%.1f"
)

roa_threshold = st.sidebar.slider(
    "Min Return on Assets (%)", 
    min_value=1, max_value=20, value=5, step=1
)

# Main area button for single ticker analysis
if single_ticker and not uploaded_file:
    if st.button("Analyze Single Ticker"):
        st.session_state.analyze_clicked = True

# Main app logic
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        
        if 'Symbol' not in df.columns:
            st.error("The uploaded file must contain a 'Symbol' column")
            st.stop()
            
        if 'Exchange' not in df.columns:
            df['Exchange'] = ''
            
        if st.button("Analyze Stocks from File"):
            filtered_df, results_df = analyze_stocks(df)
            
            if filtered_df is not None:
                display_results(filtered_df, results_df)
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
elif st.session_state.analyze_clicked:
    # Create a dataframe with the single ticker
    df = pd.DataFrame({'Symbol': [single_ticker], 'Exchange': ''})
    filtered_df, results_df = analyze_stocks(df)
    
    if filtered_df is not None:
        display_single_ticker_results(filtered_df, results_df, single_ticker)
    st.session_state.analyze_clicked = False  # Reset after analysis
else:
    st.info("""
    To begin analysis:
    1. Either enter a single stock ticker above and click "Analyze Single Ticker" OR
    2. Upload an Excel file with stock symbols
    3. Adjust valuation parameters in the sidebar as needed
    """)

# Educational content
st.markdown("""
## Interpretation Guide

**Key Metrics**:
- **P/E Ratio**: Price-to-Earnings ratio (lower is better)
- **PEG Ratio**: P/E divided by growth rate (<1 suggests undervalued)
- **Debt/Equity**: Financial leverage (lower is safer)
- **Current Ratio**: Short-term liquidity (>1.5 is healthy)
- **ROA**: Return on Assets (>5% is good)

**Tips**:
- Start with broader parameters and narrow down
- Consider sector averages when evaluating metrics
- Combine with qualitative analysis
""")
