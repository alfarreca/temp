import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px

# App title and description
st.title("📈 Overvalued Stock Screener")
st.subheader("Identifying Potentially Overpriced Stocks")
st.write("""
This app helps identify potentially overvalued stocks based on fundamental analysis metrics.
You can either upload an Excel file with multiple stocks or analyze a single ticker.
""")

# Initialize session state for button click
if 'analyze_clicked' not in st.session_state:
    st.session_state.analyze_clicked = False

# Sidebar for user inputs
st.sidebar.header("Overvaluation Parameters")

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

# Overvaluation parameters
pe_ratio_threshold = st.sidebar.slider(
    "Min P/E Ratio to Consider Overvalued", 
    min_value=20, max_value=100, value=30, step=5
)

peg_ratio_threshold = st.sidebar.slider(
    "Min PEG Ratio to Consider Overvalued", 
    min_value=1.5, max_value=5.0, value=2.0, step=0.1,
    format="%.1f"
)

price_to_sales_threshold = st.sidebar.slider(
    "Min Price-to-Sales Ratio", 
    min_value=5, max_value=30, value=10, step=1
)

price_to_book_threshold = st.sidebar.slider(
    "Min Price-to-Book Ratio", 
    min_value=3, max_value=20, value=5, step=1
)

short_interest_threshold = st.sidebar.slider(
    "Min Short Interest (%)", 
    min_value=5, max_value=50, value=10, step=1
)

# Main area button for single ticker analysis
if single_ticker and not uploaded_file:
    if st.button("Analyze Single Ticker"):
        st.session_state.analyze_clicked = True

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
            'Price/Sales': info.get('priceToSalesTrailing12Months', np.nan),
            'Price/Book': current_price / info['bookValue'] if (current_price and info.get('bookValue')) else np.nan,
            'ShortInterest': info.get('shortPercentOfFloat', np.nan) * 100 if info.get('shortPercentOfFloat') else np.nan,
            'MarketCap': info.get('marketCap', np.nan),
            'EPS': info.get('trailingEps', np.nan),
            'BookValue': info.get('bookValue', np.nan),
            'RevenueGrowth': info.get('revenueGrowth', np.nan) * 100 if info.get('revenueGrowth') else np.nan,
            'ProfitMargin': info.get('profitMargins', np.nan) * 100 if info.get('profitMargins') else np.nan,
            '52WeekLow': info.get('fiftyTwoWeekLow', np.nan),
            '52WeekHigh': info.get('fiftyTwoWeekHigh', np.nan),
            'PremiumToHigh': (current_price - info['fiftyTwoWeekHigh']) / info['fiftyTwoWeekHigh'] * 100 
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
        
        # Apply overvaluation filters
        filtered_df = results_df[
            (results_df['P/E'] >= pe_ratio_threshold) &
            (results_df['PEG'] >= peg_ratio_threshold) &
            (results_df['Price/Sales'] >= price_to_sales_threshold) &
            (results_df['Price/Book'] >= price_to_book_threshold) &
            (results_df['ShortInterest'] >= short_interest_threshold)
        ].copy()
        
        # Calculate composite overvaluation score (higher is worse)
        filtered_df['OvervaluationScore'] = (
            (filtered_df['P/E'].fillna(0) / pe_ratio_threshold +
            filtered_df['PEG'].fillna(0) / peg_ratio_threshold +
            filtered_df['Price/Sales'].fillna(0) / price_to_sales_threshold +
            filtered_df['Price/Book'].fillna(0) / price_to_book_threshold +
            filtered_df['ShortInterest'].fillna(0) / short_interest_threshold)
        ) / 5
        
        # Sort by score
        filtered_df = filtered_df.sort_values(by='OvervaluationScore', ascending=False)
        
        return filtered_df, results_df
        
    except Exception as e:
        st.error(f"Error processing stocks: {str(e)}")
        return None, None

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
    3. Adjust overvaluation parameters in the sidebar as needed
    """)

def display_results(filtered_df, results_df):
    if len(filtered_df) > 0:
        st.warning(f"Found {len(filtered_df)} potentially overvalued stocks out of {len(results_df)} analyzed")
        st.write("## Overvalued Stock Candidates")
        
        # Format display
        display_df = filtered_df[[
            'Symbol', 'Exchange', 'CurrentPrice', 'P/E', 'PEG', 'Price/Sales', 'Price/Book',
            'ShortInterest', 'RevenueGrowth', 'ProfitMargin', 'PremiumToHigh', 'OvervaluationScore'
        ]].copy()
        
        st.dataframe(
            display_df.style.format({
                'CurrentPrice': '${:.2f}',
                'P/E': '{:.1f}',
                'PEG': '{:.1f}',
                'Price/Sales': '{:.1f}',
                'Price/Book': '{:.1f}',
                'ShortInterest': '{:.1f}%',
                'RevenueGrowth': '{:.1f}%',
                'ProfitMargin': '{:.1f}%',
                'PremiumToHigh': '{:.1f}%',
                'OvervaluationScore': '{:.2f}'
            }),
            height=400
        )
        
        # Visualizations
        st.write("## Overvaluation Metrics Distribution")
        
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
                y='Price/Sales',
                title='Price-to-Sales Distribution',
                points="all"
            )
            fig.add_hline(y=price_to_sales_threshold, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        # Download button
        st.download_button(
            label="Download Results as CSV",
            data=filtered_df.to_csv(index=False),
            file_name="overvalued_stocks.csv",
            mime="text/csv"
        )
    else:
        st.info("""
        No stocks met all the overvaluation criteria. Try adjusting the parameters:
        - Decrease P/E or PEG thresholds
        - Lower Price-to-Sales requirement
        - Reduce Price-to-Book requirement
        - Decrease Short Interest threshold
        """)

def display_single_ticker_results(filtered_df, results_df, ticker):
    if len(filtered_df) > 0:
        st.warning(f"{ticker} appears to be overvalued based on the current criteria")
        st.write("## Analysis Results")
        
        # Format display for single stock
        display_df = filtered_df[[
            'Symbol', 'CurrentPrice', 'P/E', 'PEG', 'Price/Sales', 'Price/Book',
            'ShortInterest', 'RevenueGrowth', 'ProfitMargin', 'PremiumToHigh', 'OvervaluationScore'
        ]].copy()
        
        st.dataframe(
            display_df.style.format({
                'CurrentPrice': '${:.2f}',
                'P/E': '{:.1f}',
                'PEG': '{:.1f}',
                'Price/Sales': '{:.1f}',
                'Price/Book': '{:.1f}',
                'ShortInterest': '{:.1f}%',
                'RevenueGrowth': '{:.1f}%',
                'ProfitMargin': '{:.1f}%',
                'PremiumToHigh': '{:.1f}%',
                'OvervaluationScore': '{:.2f}'
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
        st.success(f"{ticker} does not appear to be overvalued based on the current criteria")

# Educational content
st.markdown("""
## Interpretation Guide

**Key Overvaluation Metrics**:
- **High P/E Ratio**: Price-to-Earnings ratio (above 30 often considered expensive)
- **High PEG Ratio**: P/E divided by growth rate (>2 suggests overvaluation)
- **Price/Sales**: Revenue multiple (>10 often concerning)
- **Price/Book**: Asset value multiple (>5 may indicate overvaluation)
- **Short Interest**: % of float sold short (high % suggests skepticism)

**Additional Warning Signs**:
- Trading at significant premium to 52-week high
- High valuation despite slowing revenue growth
- Low or negative profit margins

**Tips**:
- Consider shorting opportunities for overvalued stocks
- Look for divergences between valuation and fundamentals
- Combine with technical analysis for timing
""")
