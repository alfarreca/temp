import streamlit as st
import pandas as pd
import yfinance as yf
import io
import time
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Universal Stock Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache data loading to improve performance
@st.cache_data
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload an Excel (.xlsx) or CSV file.")
            return None
        
        required_cols = ['Symbol', 'Exchange', 'Sector', 'Industry', 'Theme', 'Name', 'Country', 'Notes', 'Asset_Type']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Fetch additional metrics with rate limiting
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yfinance_data(symbol, exchange, attempt=1):
    try:
        # Rate limiting - wait between requests
        if attempt > 1:
            time.sleep(2)  # Longer delay for retries
        
        ticker = f"{symbol}.{exchange}" if exchange else symbol
        stock = yf.Ticker(ticker)
        
        # Get info with timeout
        try:
            info = stock.info
            history = stock.history(period="1mo")
        except:
            if attempt <= 3:  # Max 3 retries
                return fetch_yfinance_data(symbol, exchange, attempt+1)
            raise
        
        metrics = {
            'P/E': info.get('trailingPE', None),
            'P/B': info.get('priceToBook', None),
            'ROE': info.get('returnOnEquity', None),
            'Current Price': info.get('currentPrice', None),
            'Price History': history['Close'] if not history.empty else None
        }
        
        return metrics
    except Exception as e:
        st.warning(f"Could not fetch data for {symbol}: API rate limit reached. Some metrics may be missing.")
        return None

# Main app function with improved data fetching
def main():
    st.title("📊 Universal Stock Screener")
    st.write("Upload your stock universe and apply different screening strategies")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload Stock Universe (Excel/CSV)", 
        type=["xlsx", "csv"]
    )
    
    if not uploaded_file:
        st.info("Please upload a stock universe file to begin screening")
        return
    
    df = load_data(uploaded_file)
    if df is None:
        return
    
    # Value Screener Configuration (simplified for demo)
    st.sidebar.subheader("Value Screener Filters")
    
    pb_range = st.sidebar.slider(
        "P/B Ratio Range",
        min_value=0.0,
        max_value=10.0,
        value=(0.0, 3.0),
        step=0.1
    )
    
    pe_range = st.sidebar.slider(
        "P/E Ratio Range",
        min_value=0.0,
        max_value=50.0,
        value=(0.0, 15.0),
        step=0.5
    )
    
    roe_range = st.sidebar.slider(
        "ROE Range (%)",
        min_value=0.0,
        max_value=100.0,
        value=(15.0, 100.0),
        step=1.0
    )
    
    sectors = st.sidebar.multiselect(
        "Sectors",
        options=sorted(df['Sector'].unique()),
        default=sorted(df['Sector'].unique())
    )
    
    if st.sidebar.button("Apply Screening", type="primary"):
        with st.spinner("Fetching stock data (this may take a while due to API limits)..."):
            screened_df = df.copy()
            screened_df = screened_df[screened_df['Sector'].isin(sectors)]
            
            # Initialize columns for metrics
            screened_df['P/E'] = None
            screened_df['P/B'] = None
            screened_df['ROE'] = None
            screened_df['Current Price'] = None
            screened_df['Price History'] = None
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, row in screened_df.iterrows():
                symbol = row['Symbol']
                exchange = row['Exchange']
                
                status_text.text(f"Fetching data for {symbol}...")
                
                metrics = fetch_yfinance_data(symbol, exchange)
                if metrics:
                    for metric_name, metric_value in metrics.items():
                        if metric_name in screened_df.columns:
                            screened_df.at[i, metric_name] = metric_value
                
                progress_bar.progress((i + 1) / len(screened_df))
                time.sleep(1)  # Rate limiting - 1 second between requests
            
            # Apply value filters
            screened_df = screened_df[
                (screened_df['P/B'].between(pb_range[0], pb_range[1])) &
                (screened_df['P/E'].between(pe_range[0], pe_range[1])) &
                (screened_df['ROE'].between(roe_range[0], roe_range[1]))
            ]
            
            # Display results
            st.subheader("Value Screener Results")
            st.write(f"Found {len(screened_df)} stocks matching your criteria")
            
            if not screened_df.empty:
                st.dataframe(
                    screened_df[['Symbol', 'Name', 'Sector', 'P/B', 'P/E', 'ROE', 'Current Price']],
                    use_container_width=True
                )
                
                # Download button
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    screened_df.to_excel(writer, index=False)
                st.download_button(
                    label="Download Results",
                    data=output.getvalue(),
                    file_name=f"value_screener_results_{datetime.now().date()}.xlsx"
                )
            else:
                st.warning("No stocks match your criteria")

if __name__ == "__main__":
    main()
