import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px

# App title and description
st.title("📊 Undervalued Stock Screener")
st.subheader("Warren Buffett-inspired Value Investing Strategy")
st.write("""
This app helps identify potentially undervalued stocks based on fundamental analysis metrics similar to those used by Warren Buffett.
Upload your Excel file with stock symbols to get started.
""")

# Sidebar for user inputs
st.sidebar.header("User Input Parameters")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload your Excel file with stock tickers", 
    type=["xlsx"],
    help="File should contain 'Symbol' and 'Exchange' columns"
)

# Valuation parameters
st.sidebar.subheader("Valuation Parameters")
pe_ratio_threshold = st.sidebar.slider(
    "Max P/E Ratio", 
    min_value=5, max_value=30, value=15, step=1,
    help="Warren Buffett typically looks for P/E ratios below 15-20"
)
peg_ratio_threshold = st.sidebar.slider(
    "Max PEG Ratio", 
    min_value=0.5, max_value=2.0, value=1.0, step=0.1,
    help="Price/Earnings-to-Growth ratio should ideally be < 1"
)
debt_to_equity_threshold = st.sidebar.slider(
    "Max Debt-to-Equity", 
    min_value=0.1, max_value=2.0, value=0.5, step=0.1,
    help="Lower debt-to-equity is preferred (typically < 0.5)"
)
current_ratio_threshold = st.sidebar.slider(
    "Min Current Ratio", 
    min_value=1.0, max_value=3.0, value=1.5, step=0.1,
    help="Measures short-term financial health (should be > 1.5)"
)
roa_threshold = st.sidebar.slider(
    "Min Return on Assets (%)", 
    min_value=1, max_value=20, value=5, step=1,
    help="Measures how efficient management is (typically > 5%)"
)

# Function to get financial data
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
            'DividendYield': info.get('dividendYield', np.nan) * 100 if info.get('dividendYield') else 0,
            '52WeekLow': info.get('fiftyTwoWeekLow', np.nan),
            '52WeekHigh': info.get('fiftyTwoWeekHigh', np.nan)
        }
        
        # Calculate price to book if we have both price and book value
        if current_price and info.get('bookValue'):
            metrics['Price/Book'] = current_price / info['bookValue']
        else:
            metrics['Price/Book'] = np.nan
            
        # Calculate discount from 52-week high
        if current_price and info.get('fiftyTwoWeekHigh'):
            metrics['DiscountFromHigh'] = (info['fiftyTwoWeekHigh'] - current_price) / info['fiftyTwoWeekHigh'] * 100
        else:
            metrics['DiscountFromHigh'] = np.nan
            
        return metrics
        
    except Exception as e:
        st.warning(f"Error fetching data for {ticker}: {str(e)}")
        return None

# Main app logic
if uploaded_file is not None:
    try:
        # Read the uploaded file
        df = pd.read_excel(uploaded_file)
        
        # Validate columns
        if 'Symbol' not in df.columns:
            st.error("The uploaded file must contain a 'Symbol' column")
            st.stop()
            
        if 'Exchange' not in df.columns:
            df['Exchange'] = ''  # Add empty Exchange column if not present
            
        st.success(f"Successfully loaded {len(df)} stocks from the uploaded file")
        
        # Show sample data
        with st.expander("Show uploaded data"):
            st.dataframe(df.head())
            
        # Analyze button
        if st.button("Analyze Stocks"):
            st.write("## Analyzing Stocks...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            total_stocks = len(df)
            
            for i, row in df.iterrows():
                ticker = row['Symbol']
                exchange = row['Exchange']
                
                status_text.text(f"Processing {ticker} ({i+1}/{total_stocks})...")
                progress_bar.progress((i+1)/total_stocks)
                
                metrics = get_stock_data(ticker, exchange)
                if metrics:
                    results.append(metrics)
            
            if not results:
                st.error("No valid stock data could be retrieved. Please check your tickers and try again.")
                st.stop()
                
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
            
            st.success(f"Found {len(filtered_df)} potentially undervalued stocks out of {len(results_df)} analyzed")
            
            # Show results
            st.write("## Undervalued Stock Candidates")
            
            # Format display
            display_df = filtered_df[[
                'Symbol', 'Exchange', 'CurrentPrice', 'P/E', 'PEG', 'Price/Book',
                'Debt/Equity', 'CurrentRatio', 'ROA', 'ROE', 'ProfitMargin',
                'DividendYield', 'DiscountFromHigh', 'Score'
            ]].copy()
            
            # Format percentages and decimals
            display_df['ROA'] = display_df['ROA'].apply(lambda x: f"{x:.1f}%" if not pd.isna(x) else "")
            display_df['ROE'] = display_df['ROE'].apply(lambda x: f"{x:.1f}%" if not pd.isna(x) else "")
            display_df['ProfitMargin'] = display_df['ProfitMargin'].apply(lambda x: f"{x:.1f}%" if not pd.isna(x) else "")
            display_df['DividendYield'] = display_df['DividendYield'].apply(lambda x: f"{x:.1f}%" if not pd.isna(x) else "")
            display_df['DiscountFromHigh'] = display_df['DiscountFromHigh'].apply(lambda x: f"{x:.1f}%" if not pd.isna(x) else "")
            display_df['CurrentPrice'] = display_df['CurrentPrice'].apply(lambda x: f"${x:.2f}" if not pd.isna(x) else "")
            
            st.dataframe(
                display_df,
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Score (lower is better)",
                        help="Composite valuation score (lower indicates more undervalued)",
                        min_value=0,
                        max_value=1.5,
                        format="%.2f"
                    )
                }
            )
            
            # Add some visualizations
            st.write("## Valuation Metrics Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    results_df, 
                    x='P/E', 
                    title='P/E Ratio Distribution',
                    nbins=20,
                    labels={'P/E': 'P/E Ratio'}
                )
                fig.add_vline(x=pe_ratio_threshold, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                fig = px.histogram(
                    results_df, 
                    x='PEG', 
                    title='PEG Ratio Distribution',
                    nbins=20,
                    labels={'PEG': 'PEG Ratio'}
                )
                fig.add_vline(x=peg_ratio_threshold, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
                
            # Download button
            st.download_button(
                label="Download Results as CSV",
                data=filtered_df.to_csv(index=False),
                file_name="undervalued_stocks.csv",
                mime="text/csv"
            )
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload an Excel file with stock tickers to begin analysis")

# Add some educational content
st.markdown("""
## Warren Buffett's Investment Principles

This app screens stocks based on several of Buffett's key investment criteria:

1. **Low P/E Ratio** - Stocks trading at reasonable earnings multiples
2. **Low PEG Ratio** - Growth considered relative to valuation
3. **Strong Balance Sheet** - Low debt and healthy current ratio
4. **Consistent Profitability** - Good return on assets and equity

Remember that quantitative screening is just the first step - qualitative analysis of the business is equally important!
""")
