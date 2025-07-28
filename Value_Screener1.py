import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# App configuration
st.set_page_config(
    page_title="Value Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=3600)
def load_data(uploaded_file):
    """Load and preprocess the Excel file"""
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            required_columns = ['Symbol', 'Exchange', 'Sector', 'Industry', 
                              'Theme', 'Name', 'Country', 'Notes', 'Asset_Type']
            
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                return None
            
            # Fill missing values with 'Unknown'
            df['Asset_Type'] = df['Asset_Type'].fillna('Unknown')
            df['Sector'] = df['Sector'].fillna('Unknown')
            df['Country'] = df['Country'].fillna('Unknown')
            
            return df
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return None
    return None

@st.cache_data(ttl=3600)
def fetch_financials(ticker):
    """Fetch financial metrics using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get fundamental data with careful error handling
        pe = info.get('trailingPE', None)
        pb = info.get('priceToBook', None)
        roe = info.get('returnOnEquity', None)
        
        # Only return data if we have all required metrics
        if None not in [pe, pb, roe]:
            return {
                'P/E': pe,
                'P/B': pb,
                'ROE': roe,
                'Market Cap': info.get('marketCap', None),
                'Dividend Yield': info.get('dividendYield', 0)
            }
        return None
    except:
        return None

def highlight_undervalued(row):
    """Helper function for dataframe styling"""
    if row['P/B'] <= 1.0:
        return ['background-color: #e6ffe6']*len(row)
    elif row['P/B'] <= 1.2:
        return ['background-color: #ffffe6']*len(row)
    else:
        return ['']*len(row)

def main():
    st.title("📊 Value Screener")
    st.markdown("""
    **Screen for undervalued, profitable companies**  
    Default filters set to:  
    - P/B ≤ 1.2 (undervalued stocks)  
    - ROE > 0% (profitable companies)  
    - Only showing complete financial data
    """)
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel File", 
        type=['xlsx', 'xls'],
        help="Required columns: Symbol, Exchange, Sector, Industry, Theme, Name, Country, Notes, Asset_Type"
    )
    
    df = load_data(uploaded_file)
    
    if df is not None:
        # Filters
        with st.sidebar:
            st.header("Filters")
            
            # Asset type filter
            asset_types = ['All'] + sorted(df['Asset_Type'].dropna().unique().tolist())
            selected_asset_type = st.selectbox("Asset Type", asset_types)
            
            # Sector filter
            sectors = ['All'] + sorted(df['Sector'].dropna().unique().tolist())
            selected_sector = st.selectbox("Sector", sectors)
            
            # Country filter
            countries = ['All'] + sorted(df['Country'].dropna().unique().tolist())
            selected_country = st.selectbox("Country", countries)
            
            st.subheader("Value Metrics")
            # Updated default values as requested
            pe_max = st.slider("Max P/E Ratio", 0, 50, 20, 
                             help="Price-to-Earnings ratio ceiling")
            pb_max = st.slider("Max P/B Ratio", 0.0, 5.0, 1.2, 0.1,
                              help="Price-to-Book ratio ceiling (set to 1.2 for undervalued stocks)")
            roe_min = st.slider("Min ROE (%)", 0, 50, 0,
                              help="Minimum Return on Equity (set to 0% to filter for profitable companies)")
        
        # Apply basic filters
        filtered_df = df.copy()
        if selected_asset_type != 'All':
            filtered_df = filtered_df[filtered_df['Asset_Type'] == selected_asset_type]
        if selected_sector != 'All':
            filtered_df = filtered_df[filtered_df['Sector'] == selected_sector]
        if selected_country != 'All':
            filtered_df = filtered_df[filtered_df['Country'] == selected_country]
        
        # Fetch and filter financial data
        if not filtered_df.empty:
            with st.spinner("Fetching financial data (this may take a minute)..."):
                metrics_data = []
                valid_symbols = []
                
                for symbol in filtered_df['Symbol'].unique():
                    metrics = fetch_financials(symbol)
                    if metrics:  # Only include symbols with complete data
                        metrics['Symbol'] = symbol
                        metrics_data.append(metrics)
                        valid_symbols.append(symbol)
                
                if metrics_data:
                    # Merge with original data
                    metrics_df = pd.DataFrame(metrics_data)
                    merged_df = pd.merge(
                        filtered_df[filtered_df['Symbol'].isin(valid_symbols)],
                        metrics_df,
                        on='Symbol',
                        how='inner'  # Only keep rows with financial data
                    )
                    
                    # Apply value filters
                    value_df = merged_df[
                        (merged_df['P/E'] <= pe_max) & 
                        (merged_df['P/B'] <= pb_max) & 
                        (merged_df['ROE'] >= roe_min/100)
                    ].sort_values(by=['P/B', 'ROE'], ascending=[True, False])
                    
                    # Display results
                    st.subheader(f"🔍 Found {len(value_df)} qualifying companies")
                    
                    if not value_df.empty:
                        # Format display
                        display_df = value_df[[
                            'Symbol', 'Name', 'Sector', 'Country',
                            'P/E', 'P/B', 'ROE', 'Market Cap'
                        ]].copy()
                        
                        display_df['ROE'] = (display_df['ROE'] * 100).round(1).astype(str) + '%'
                        display_df['P/E'] = display_df['P/E'].round(1)
                        display_df['P/B'] = display_df['P/B'].round(2)
                        display_df['Market Cap'] = display_df['Market Cap'].apply(
                            lambda x: f"${x/1e9:.1f}B" if pd.notnull(x) else 'N/A'
                        )
                        
                        # Display styled dataframe
                        st.dataframe(
                            display_df.style.apply(highlight_undervalued, axis=1)),
                            hide_index=True,
                            use_container_width=True,
                            height=min(400, 35*(len(display_df)+1)
                        )
                        
                        # Download
                        csv = value_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Full Results",
                            csv,
                            "undervalued_stocks.csv",
                            "text/csv",
                            help="Download complete data including all columns"
                        )
                    else:
                        st.warning("No companies match all your criteria. Try relaxing your filters.")
                else:
                    st.error("Couldn't fetch complete financial data for any symbols. Check your data source.")
        
        else:
            st.warning("No companies match your initial filters. Try adjusting sector/country selections.")
    
    elif uploaded_file is None:
        st.info("""
        ℹ️ **How to use this screener:**
        1. Upload an Excel file with stock data
        2. Apply filters using the sidebar
        3. View undervalued stocks (P/B ≤ 1.2 by default)
        4. Download results for further analysis
        
        Required columns: Symbol, Exchange, Sector, Industry, Theme, Name, Country, Notes, Asset_Type
        """)

if __name__ == "__main__":
    main()
