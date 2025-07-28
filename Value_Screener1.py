import streamlit as st
import pandas as pd
import yfinance as yf
import io
from datetime import datetime, timedelta

# App configuration
st.set_page_config(
    page_title="Value Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache data to improve performance
@st.cache_data(ttl=3600)
def load_data(uploaded_file):
    """Load and preprocess the Excel file"""
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            required_columns = ['Symbol', 'Exchange', 'Sector', 'Industry', 'Theme', 'Name', 'Country', 'Notes', 'Asset_Type']
            
            # Check if all required columns exist
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                st.warning(f"Missing columns in the uploaded file: {', '.join(missing_cols)}")
                return None
            
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    return None

@st.cache_data(ttl=3600)
def fetch_financials(ticker):
    """Fetch financial metrics using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        metrics = {
            'P/E': info.get('trailingPE', None),
            'P/B': info.get('priceToBook', None),
            'ROE': info.get('returnOnEquity', None),
            'Market Cap': info.get('marketCap', None),
            'Dividend Yield': info.get('dividendYield', None)
        }
        return metrics
    except:
        return None

@st.cache_data(ttl=3600)
def get_price_history(ticker, period='1y'):
    """Get price history for sparkline"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist['Close'].values.tolist()
    except:
        return None

def main():
    st.title("📊 Value Screener")
    st.markdown("""
    Screen for stocks with **low P/B**, **low P/E**, and **high ROE** - finding cheap but strong businesses.
    """)
    
    # File upload
    with st.sidebar:
        st.header("Data Input")
        uploaded_file = st.file_uploader(
            "Upload Excel File", 
            type=['xlsx', 'xls'],
            help="Excel file should contain Symbol, Exchange, Sector, Industry, Theme, Name, Country, Notes, Asset_Type columns"
        )
        
        if uploaded_file is not None:
            st.success("File uploaded successfully!")
    
    # Load data
    df = load_data(uploaded_file)
    
    if df is not None:
        # Filters
        with st.sidebar:
            st.header("Filters")
            
            # Asset Type filter
            asset_types = ['All'] + sorted(df['Asset_Type'].unique().tolist())
            selected_asset_type = st.selectbox(
                "Asset Type",
                asset_types,
                index=0
            )
            
            # Sector filter
            sectors = ['All'] + sorted(df['Sector'].dropna().unique().tolist())
            selected_sector = st.selectbox(
                "Sector",
                sectors,
                index=0
            )
            
            # Country filter
            countries = ['All'] + sorted(df['Country'].dropna().unique().tolist())
            selected_country = st.selectbox(
                "Country",
                countries,
                index=0
            )
            
            # Value metrics filters
            st.subheader("Value Metrics")
            pe_max = st.slider(
                "Max P/E Ratio", 
                min_value=0, 
                max_value=50, 
                value=15,
                help="Maximum allowed Price-to-Earnings ratio"
            )
            
            pb_max = st.slider(
                "Max P/B Ratio", 
                min_value=0.0, 
                max_value=5.0, 
                value=2.0,
                step=0.1,
                help="Maximum allowed Price-to-Book ratio"
            )
            
            roe_min = st.slider(
                "Min ROE (%)", 
                min_value=0, 
                max_value=50, 
                value=15,
                help="Minimum required Return on Equity"
            )
        
        # Apply filters
        filtered_df = df.copy()
        
        if selected_asset_type != 'All':
            filtered_df = filtered_df[filtered_df['Asset_Type'] == selected_asset_type]
        
        if selected_sector != 'All':
            filtered_df = filtered_df[filtered_df['Sector'] == selected_sector]
        
        if selected_country != 'All':
            filtered_df = filtered_df[filtered_df['Country'] == selected_country]
        
        # Fetch financial metrics for each symbol
        metrics_data = []
        symbols_to_remove = []
        
        with st.spinner("Fetching financial data..."):
            for symbol in filtered_df['Symbol'].unique():
                metrics = fetch_financials(symbol)
                if metrics:
                    metrics['Symbol'] = symbol
                    metrics_data.append(metrics)
                else:
                    symbols_to_remove.append(symbol)
            
            # Remove symbols where we couldn't fetch metrics
            filtered_df = filtered_df[~filtered_df['Symbol'].isin(symbols_to_remove)]
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                merged_df = pd.merge(filtered_df, metrics_df, on='Symbol', how='left')
                
                # Apply value filters
                value_filtered = merged_df[
                    (merged_df['P/E'] <= pe_max) & 
                    (merged_df['P/B'] <= pb_max) & 
                    (merged_df['ROE'] >= roe_min/100)
                ].sort_values(by=['P/E', 'P/B'])
                
                # Display results
                st.subheader(f"Filtered Results ({len(value_filtered)} companies)")
                
                if not value_filtered.empty:
                    # Add sparklines
                    with st.spinner("Loading price trends..."):
                        value_filtered['Price Trend'] = value_filtered['Symbol'].apply(
                            lambda x: get_price_history(x)
                        )
                    
                    # Display table
                    cols_to_show = [
                        'Symbol', 'Name', 'Sector', 'Industry', 'Country',
                        'P/E', 'P/B', 'ROE', 'Market Cap', 'Dividend Yield', 'Price Trend'
                    ]
                    
                    # Format the DataFrame for display
                    display_df = value_filtered[cols_to_show].copy()
                    display_df['ROE'] = (display_df['ROE'] * 100).round(1).astype(str) + '%'
                    display_df['Dividend Yield'] = (display_df['Dividend Yield'] * 100).round(2).astype(str) + '%' if 'Dividend Yield' in display_df.columns else 'N/A'
                    display_df['Market Cap'] = display_df['Market Cap'].apply(
                        lambda x: f"${x/1e9:.2f}B" if pd.notnull(x) else 'N/A'
                    )
                    display_df['P/E'] = display_df['P/E'].round(2)
                    display_df['P/B'] = display_df['P/B'].round(2)
                    
                    # Show the table
                    st.dataframe(
                        display_df,
                        column_config={
                            "Price Trend": st.column_config.LineChartColumn(
                                "Price Trend (1Y)",
                                width="medium",
                                help="1-year price trend",
                                y_min=0
                            )
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Show raw data option
                    if st.checkbox("Show raw data"):
                        st.write(value_filtered)
                    
                    # Download button
                    csv = value_filtered.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download filtered data as CSV",
                        data=csv,
                        file_name='value_screener_results.csv',
                        mime='text/csv'
                    )
                else:
                    st.warning("No companies match your criteria. Try adjusting your filters.")
            else:
                st.error("Could not fetch financial metrics for any symbols. Please check your data.")
    else:
        st.info("Please upload an Excel file to begin screening.")

if __name__ == "__main__":
    main()
