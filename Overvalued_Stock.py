import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px

# Initialize session state
if 'analyze_clicked' not in st.session_state:
    st.session_state.analyze_clicked = False

# App configuration
st.set_page_config(layout="wide")
st.title("📈 Overvalued Stock Screener")
st.subheader("Identifying Potentially Overpriced Stocks")
st.write("""
This app helps identify potentially overvalued stocks based on fundamental analysis metrics.
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
        help="Example: TSLA for Tesla, NVDA for Nvidia"
    )
    exchange = st.sidebar.selectbox(
        "Select exchange (if international):",
        ["", "NYSE/NASDAQ", "TORONTO", "LONDON", "EURONEXT", "FRANKFURT", "HONG KONG", "SHANGHAI"]
    )
    
    if st.sidebar.button("Analyze Single Stock"):
        st.session_state.analyze_clicked = True

# Multiple stocks analysis
else:
    st.sidebar.header("File Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel file with stock tickers", 
        type=["xlsx"],
        help="File should contain 'Symbol' column (optional: 'Exchange')"
    )

# Overvaluation parameters
st.sidebar.header("Overvaluation Parameters")
with st.sidebar.expander("Adjust overvaluation thresholds"):
    pe_ratio_threshold = st.slider(
        "Min P/E Ratio to Flag", 
        min_value=15, max_value=100, value=40, step=5,
        help="Lower values will flag fewer stocks as overvalued"
    )
    peg_ratio_threshold = st.slider(
        "Min PEG Ratio to Flag", 
        min_value=1.0, max_value=5.0, value=2.5, step=0.1,
        format="%.1f"
    )
    price_to_sales_threshold = st.slider(
        "Min Price/Sales to Flag", 
        min_value=3, max_value=30, value=8, step=1
    )
    price_to_book_threshold = st.slider(
        "Min Price/Book to Flag", 
        min_value=2, max_value=20, value=5, step=1
    )
    short_interest_threshold = st.slider(
        "Min Short Interest (%)", 
        min_value=3, max_value=50, value=10, step=1
    )

# Improved stock data fetcher
def get_stock_data(ticker, exchange=""):
    try:
        # Map exchanges to suffixes
        exchange_map = {
            "TORONTO": ".TO",
            "LONDON": ".L",
            "EURONEXT": ".PA",
            "FRANKFURT": ".DE",
            "HONG KONG": ".HK",
            "SHANGHAI": ".SS"
        }
        
        full_ticker = ticker + exchange_map.get(exchange.upper(), "")
        
        stock = yf.Ticker(full_ticker)
        hist = stock.history(period="1d")
        if hist.empty:
            return None
            
        current_price = hist['Close'].iloc[-1]
        info = stock.info
        
        # Calculate all metrics with fallback values
        metrics = {
            'Symbol': ticker,
            'Exchange': exchange,
            'CurrentPrice': current_price,
            'P/E': info.get('trailingPE', np.nan),
            'Forward P/E': info.get('forwardPE', np.nan),
            'PEG': info.get('pegRatio', np.nan),
            'Price/Sales': info.get('priceToSalesTrailing12Months', np.nan),
            'Price/Book': current_price / info.get('bookValue', np.nan) if info.get('bookValue') else np.nan,
            'ShortInterest': info.get('shortPercentOfFloat', 0) * 100,
            'Beta': info.get('beta', np.nan),
            'MarketCap': info.get('marketCap', np.nan),
            '52WeekLow': info.get('fiftyTwoWeekLow', np.nan),
            '52WeekHigh': info.get('fiftyTwoWeekHigh', np.nan),
            'PremiumToHigh': (current_price - info.get('fiftyTwoWeekHigh', np.nan)) / info.get('fiftyTwoWeekHigh', 1) * 100 
                           if info.get('fiftyTwoWeekHigh') else np.nan,
            'ProfitMargin': info.get('profitMargins', np.nan) * 100 if info.get('profitMargins') else np.nan,
            'RevenueGrowth': info.get('revenueGrowth', np.nan) * 100 if info.get('revenueGrowth') else np.nan
        }
        return metrics
        
    except Exception as e:
        st.warning(f"Error fetching data for {ticker}: {str(e)}")
        return None

# Analysis function
def analyze_stocks(tickers_df):
    try:
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
            st.error("No valid stock data could be retrieved. Please check your tickers and try again.")
            return None, None
            
        results_df = pd.DataFrame(results)
        
        # Apply filters with more flexible conditions
        filtered_df = results_df[
            (results_df['P/E'].fillna(0) >= pe_ratio_threshold) |
            (results_df['PEG'].fillna(0) >= peg_ratio_threshold) |
            (results_df['Price/Sales'].fillna(0) >= price_to_sales_threshold) |
            (results_df['Price/Book'].fillna(0) >= price_to_book_threshold) |
            (results_df['ShortInterest'].fillna(0) >= short_interest_threshold)
        ].copy()
        
        # Enhanced scoring
        filtered_df['OvervaluationScore'] = (
            (filtered_df['P/E'].fillna(0) / pe_ratio_threshold * 0.3) +
            (filtered_df['PEG'].fillna(0) / peg_ratio_threshold * 0.25) +
            (filtered_df['Price/Sales'].fillna(0) / price_to_sales_threshold * 0.2) +
            (filtered_df['Price/Book'].fillna(0) / price_to_book_threshold * 0.15) +
            (filtered_df['ShortInterest'].fillna(0) / short_interest_threshold * 0.1)
        )
        
        filtered_df = filtered_df.sort_values(by='OvervaluationScore', ascending=False)
        return filtered_df, results_df
        
    except Exception as e:
        st.error(f"Error processing stocks: {str(e)}")
        return None, None

# Display results
def display_results(filtered_df, results_df, single_stock=False):
    if len(filtered_df) > 0:
        st.warning(f"Found {len(filtered_df)} potentially overvalued {'stock' if single_stock else 'stocks'}")
        
        # Enhanced display for single stock
        if single_stock:
            st.write("## Detailed Analysis")
            detailed_metrics = filtered_df.iloc[0].to_dict()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current Price", f"${detailed_metrics['CurrentPrice']:,.2f}")
                st.metric("P/E Ratio", f"{detailed_metrics['P/E']:.1f}", 
                         delta=f"{detailed_metrics['P/E'] - pe_ratio_threshold:.1f} vs threshold")
                st.metric("PEG Ratio", f"{detailed_metrics['PEG']:.1f}",
                         delta=f"{detailed_metrics['PEG'] - peg_ratio_threshold:.1f} vs threshold")
                st.metric("Price/Sales", f"{detailed_metrics['Price/Sales']:.1f}")
                
            with col2:
                st.metric("Price/Book", f"{detailed_metrics['Price/Book']:.1f}")
                st.metric("Short Interest", f"{detailed_metrics['ShortInterest']:.1f}%")
                st.metric("Premium to High", f"{detailed_metrics['PremiumToHigh']:.1f}%")
                st.metric("Beta", f"{detailed_metrics['Beta']:.2f}")
            
            st.write("### All Metrics")
            st.dataframe(filtered_df.style.format({
                'CurrentPrice': '${:,.2f}',
                'MarketCap': '${:,.0f}',
                **{col: '{:.1f}' for col in ['P/E', 'PEG', 'Price/Sales', 'Price/Book', 'Beta']},
                **{col: '{:.1f}%' for col in ['ShortInterest', 'PremiumToHigh', 'ProfitMargin', 'RevenueGrowth']}
            }))
        
        # Display for multiple stocks
        else:
            st.write("## Overvalued Stock Candidates")
            display_cols = [
                'Symbol', 'CurrentPrice', 'P/E', 'PEG', 'Price/Sales', 'Price/Book',
                'ShortInterest', 'PremiumToHigh', 'OvervaluationScore'
            ]
            
            st.dataframe(
                filtered_df[display_cols].style.format({
                    'CurrentPrice': '${:.2f}',
                    'P/E': '{:.1f}',
                    'PEG': '{:.1f}',
                    'Price/Sales': '{:.1f}',
                    'Price/Book': '{:.1f}',
                    'ShortInterest': '{:.1f}%',
                    'PremiumToHigh': '{:.1f}%',
                    'OvervaluationScore': '{:.2f}'
                }).applymap(lambda x: 'color: red' if isinstance(x, (int, float)) and x > 1 else '', 
                          subset=['OvervaluationScore']),
                height=600,
                use_container_width=True
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
        No stocks met the overvaluation criteria. Try:
        - Decreasing P/E or PEG thresholds  
        - Lowering Price/Sales or Price/Book requirements
        - Reducing Short Interest threshold
        - Checking if your stocks have sufficient data
        """)

# Main logic
if analysis_type == "Single Stock Analysis" and st.session_state.analyze_clicked:
    if single_ticker:
        df = pd.DataFrame({'Symbol': [single_ticker], 'Exchange': [exchange]})
        filtered_df, results_df = analyze_stocks(df)
        
        if filtered_df is not None:
            display_results(filtered_df, results_df, single_stock=True)
        st.session_state.analyze_clicked = False
    else:
        st.warning("Please enter a stock ticker")

elif analysis_type == "Multiple Stocks Analysis" and uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        if 'Symbol' not in df.columns:
            st.error("File must contain 'Symbol' column")
            st.stop()
            
        if 'Exchange' not in df.columns:
            df['Exchange'] = ''
            
        if st.button("Analyze Stocks"):
            filtered_df, results_df = analyze_stocks(df)
            if filtered_df is not None:
                display_results(filtered_df, results_df)
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

else:
    st.info("""
    To begin analysis:
    1. Select analysis type (single stock or file upload)
    2. Enter stock ticker or upload file
    3. Adjust overvaluation parameters as needed
    4. Click analyze button
    """)

# Educational content
st.markdown("""
## Interpretation Guide

**Key Overvaluation Metrics**:
- **High P/E Ratio**: Price-to-Earnings (higher = more expensive)
- **High PEG Ratio**: P/E relative to growth (>2 = potentially overvalued)
- **Price/Sales**: Revenue multiple (>10 = potentially expensive)
- **Price/Book**: Asset value multiple (>5 = potentially overpriced)
- **Short Interest**: % of float sold short (>10% = bearish sentiment)

**Warning Signs**:
- Trading at premium to 52-week high
- High valuation with slowing growth
- Low or negative profit margins
- High beta (>1.5) indicating volatility

**Tips**:
- Different sectors have different valuation norms  
- Consider short interest trends over time
- Combine with technical analysis for timing
""")
