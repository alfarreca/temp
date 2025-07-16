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
        st.session_state.analyze_clicked = True

# Multiple stocks analysis
else:
    st.sidebar.header("File Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel file with stock tickers", 
        type=["xlsx"],
        help="File should contain 'Symbol' column (optional: 'Exchange')"
    )

# Valuation parameters
st.sidebar.header("Valuation Parameters")
with st.sidebar.expander("Adjust valuation thresholds"):
    pe_ratio_threshold = st.slider(
        "Max P/E Ratio", 
        min_value=5, max_value=50, value=25, step=1,
        help="Higher values will include more stocks"
    )
    peg_ratio_threshold = st.slider(
        "Max PEG Ratio", 
        min_value=0.1, max_value=3.0, value=1.5, step=0.1,
        format="%.1f"
    )
    debt_to_equity_threshold = st.slider(
        "Max Debt-to-Equity", 
        min_value=0.1, max_value=5.0, value=2.0, step=0.1,
        format="%.1f"
    )
    current_ratio_threshold = st.slider(
        "Min Current Ratio", 
        min_value=0.5, max_value=3.0, value=1.0, step=0.1,
        format="%.1f"
    )
    roa_threshold = st.slider(
        "Min Return on Assets (%)", 
        min_value=0, max_value=20, value=5, step=1
    )

def get_stock_data(ticker, exchange=""):
    try:
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
        
        metrics = {
            'Symbol': ticker,
            'Exchange': exchange,
            'CurrentPrice': current_price,
            'P/E': info.get('trailingPE', np.nan),
            'Forward P/E': info.get('forwardPE', np.nan),
            'PEG': info.get('pegRatio', np.nan),
            'Debt/Equity': info.get('debtToEquity', np.nan),
            'CurrentRatio': info.get('currentRatio', np.nan),
            'ROA': info.get('returnOnAssets', np.nan) * 100 if info.get('returnOnAssets') else np.nan,
            'ROE': info.get('returnOnEquity', np.nan) * 100 if info.get('returnOnEquity') else np.nan,
            'ProfitMargin': info.get('profitMargins', np.nan) * 100 if info.get('profitMargins') else np.nan,
            'Price/Book': current_price / info.get('bookValue', np.nan) if info.get('bookValue') else np.nan,
            'DividendYield': info.get('dividendYield', 0) * 100,
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
        
        filtered_df = results_df[
            (results_df['P/E'].fillna(np.inf) <= pe_ratio_threshold) &
            (results_df['PEG'].fillna(np.inf) <= peg_ratio_threshold) &
            (results_df['Debt/Equity'].fillna(np.inf) <= debt_to_equity_threshold) &
            (results_df['CurrentRatio'].fillna(0) >= current_ratio_threshold) &
            (results_df['ROA'].fillna(0) >= roa_threshold)
        ].copy()
        
        weights = {
            'P/E': 0.3,
            'PEG': 0.25,
            'Debt/Equity': 0.2,
            'CurrentRatio': 0.15,
            'ROA': 0.1
        }
        
        filtered_df['Score'] = (
            (filtered_df['P/E'].fillna(0) / pe_ratio_threshold * weights['P/E']) +
            (filtered_df['PEG'].fillna(0) / peg_ratio_threshold * weights['PEG']) +
            (filtered_df['Debt/Equity'].fillna(0) / debt_to_equity_threshold * weights['Debt/Equity']) +
            (current_ratio_threshold / filtered_df['CurrentRatio'].fillna(np.inf) * weights['CurrentRatio']) +
            (roa_threshold / filtered_df['ROA'].fillna(np.inf) * weights['ROA'])
        )
        
        filtered_df = filtered_df.sort_values(by='Score', ascending=True)
        return filtered_df, results_df
        
    except Exception as e:
        st.error(f"Error processing stocks: {str(e)}")
        return None, None

def display_results(filtered_df, results_df, single_stock=False):
    if len(filtered_df) > 0:
        st.success(f"Found {len(filtered_df)} potentially undervalued {'stock' if single_stock else 'stocks'}")
        
        # Raw Data Section - Always Visible
        st.write("---")
        st.write("## Complete Stock Data")
        st.info("This table shows all analyzed stocks before filtering. Scroll to see all metrics.")
        st.dataframe(
            results_df.style.format({
                'CurrentPrice': '${:,.2f}',
                'MarketCap': '${:,.0f}',
                **{col: '{:.1f}' for col in ['P/E', 'PEG', 'Debt/Equity', 'CurrentRatio', 'Price/Book']},
                **{col: '{:.1f}%' for col in ['ROA', 'ROE', 'ProfitMargin', 'DividendYield', 'DiscountFromHigh']}
            }),
            height=400,
            use_container_width=True
        )
        
        # Filtered Results Section
        st.write("---")
        st.write("## Filtered Results")
        
        if single_stock:
            st.write("### Detailed Analysis")
            detailed_metrics = filtered_df.iloc[0].to_dict()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current Price", f"${detailed_metrics['CurrentPrice']:,.2f}")
                st.metric("P/E Ratio", f"{detailed_metrics['P/E']:.1f}")
                st.metric("PEG Ratio", f"{detailed_metrics['PEG']:.1f}")
                st.metric("Price/Book", f"{detailed_metrics['Price/Book']:.1f}")
                
            with col2:
                st.metric("Debt/Equity", f"{detailed_metrics['Debt/Equity']:.1f}")
                st.metric("Current Ratio", f"{detailed_metrics['CurrentRatio']:.1f}")
                st.metric("ROA", f"{detailed_metrics['ROA']:.1f}%")
                st.metric("Discount from High", f"{detailed_metrics['DiscountFromHigh']:.1f}%")
            
            st.write("### All Metrics")
            st.dataframe(filtered_df.style.format({
                'CurrentPrice': '${:,.2f}',
                'MarketCap': '${:,.0f}',
                **{col: '{:.1f}' for col in ['P/E', 'PEG', 'Debt/Equity', 'CurrentRatio', 'Price/Book']},
                **{col: '{:.1f}%' for col in ['ROA', 'ROE', 'ProfitMargin', 'DividendYield', 'DiscountFromHigh']}
            }))
        else:
            st.write("### Undervalued Stock Candidates")
            display_cols = [
                'Symbol', 'CurrentPrice', 'P/E', 'PEG', 'Price/Book',
                'Debt/Equity', 'CurrentRatio', 'ROA', 'DividendYield', 
                'DiscountFromHigh', 'Score'
            ]
            
            st.dataframe(
                filtered_df[display_cols].style.format({
                    'CurrentPrice': '${:.2f}',
                    'P/E': '{:.1f}',
                    'PEG': '{:.1f}',
                    'Price/Book': '{:.1f}',
                    'Debt/Equity': '{:.1f}',
                    'CurrentRatio': '{:.1f}',
                    'ROA': '{:.1f}%',
                    'DividendYield': '{:.1f}%',
                    'DiscountFromHigh': '{:.1f}%',
                    'Score': '{:.2f}'
                }),
                height=600,
                use_container_width=True
            )
            
            st.write("### Valuation Metrics Distribution")
            col1, col2 = st.columns(2)
            with col1:
                fig = px.box(results_df, y='P/E', title='P/E Ratio Distribution', points="all")
                fig.add_hline(y=pe_ratio_threshold, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(results_df, y='Price/Book', title='Price-to-Book Distribution', points="all")
                st.plotly_chart(fig, use_container_width=True)
            
            st.write("---")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download Filtered Results",
                    data=filtered_df.to_csv(index=False),
                    file_name="undervalued_stocks.csv",
                    mime="text/csv"
                )
            with col2:
                st.download_button(
                    label="Download Complete Data",
                    data=results_df.to_csv(index=False),
                    file_name="all_stock_data.csv",
                    mime="text/csv"
                )
    else:
        st.warning("No stocks met all the valuation criteria. Try adjusting your filters.")

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
    3. Adjust valuation parameters as needed
    4. Click analyze button
    """)

# Educational content
st.markdown("""
## Interpretation Guide

**Key Metrics**:
- **P/E Ratio**: Price-to-Earnings (lower = better value)
- **PEG Ratio**: P/E divided by growth (<1 = potentially undervalued)
- **Debt/Equity**: Financial leverage (lower = less risk)
- **Current Ratio**: Short-term liquidity (>1.5 = healthy)
- **ROA**: Return on Assets (>5% = good efficiency)

**Tips**:
- Different sectors have different typical valuations
- Combine with qualitative factors like moat, management
- Consider macroeconomic environment
- Start with broader filters and narrow down
""")
