import streamlit as st
import pandas as pd
import yfinance as yf

# App configuration
st.set_page_config(
    page_title="Value Screener",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data(uploaded_file):
    """Load and preprocess the Excel file"""
    try:
        df = pd.read_excel(uploaded_file)
        required_cols = ['Symbol', 'Exchange', 'Sector', 'Industry', 
                        'Theme', 'Name', 'Country', 'Notes', 'Asset_Type']
        
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
            return None
        
        # Handle missing values
        for col in ['Asset_Type', 'Sector', 'Country']:
            df[col] = df[col].fillna('Unknown')
            
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

@st.cache_data
def fetch_financials(ticker):
    """Fetch financial metrics"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'P/E': info.get('trailingPE'),
            'P/B': info.get('priceToBook'),
            'ROE': info.get('returnOnEquity'),
            'Market Cap': info.get('marketCap')
        }
    except:
        return None

def highlight_undervalued(row):
    """Highlight cells based on P/B ratio"""
    style = [''] * len(row)
    if row['P/B'] <= 1.0:
        style = ['background-color: #e6ffe6'] * len(row)
    elif row['P/B'] <= 1.2:
        style = ['background-color: #ffffe6'] * len(row)
    return style

def main():
    st.title("📊 Value Screener")
    st.markdown("""
    **Find undervalued stocks**  
    Default filters: P/B ≤ 1.2, ROE > 0%, complete financial data only
    """)
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel File", 
        type=['xlsx', 'xls'],
        help="Must contain Symbol, Exchange, Sector, Industry columns"
    )
    
    df = load_data(uploaded_file)
    
    if df is not None:
        # Filters
        with st.sidebar:
            st.header("Filters")
            
            asset_types = ['All'] + sorted(df['Asset_Type'].unique().tolist())
            selected_asset = st.selectbox("Asset Type", asset_types)
            
            sectors = ['All'] + sorted(df['Sector'].unique().tolist())
            selected_sector = st.selectbox("Sector", sectors)
            
            st.subheader("Value Metrics")
            pb_max = st.slider("Max P/B", 0.0, 5.0, 1.2, 0.1)
            roe_min = st.slider("Min ROE (%)", 0, 50, 0)
        
        # Apply filters
        filtered = df.copy()
        if selected_asset != 'All':
            filtered = filtered[filtered['Asset_Type'] == selected_asset]
        if selected_sector != 'All':
            filtered = filtered[filtered['Sector'] == selected_sector]
        
        # Get financial data
        with st.spinner("Loading financial data..."):
            results = []
            for symbol in filtered['Symbol'].unique():
                data = fetch_financials(symbol)
                if data and None not in data.values():
                    data['Symbol'] = symbol
                    results.append(data)
            
            if results:
                financials = pd.DataFrame(results)
                merged = pd.merge(filtered, financials, on='Symbol')
                
                # Apply value filters
                undervalued = merged[
                    (merged['P/B'] <= pb_max) & 
                    (merged['ROE'] >= roe_min/100)
                ].sort_values(['P/B', 'ROE'], ascending=[True, False])
                
                # Display results
                st.subheader(f"Found {len(undervalued)} matches")
                
                if not undervalued.empty:
                    # Format display
                    display = undervalued[[
                        'Symbol', 'Name', 'Sector', 
                        'P/B', 'ROE', 'Market Cap'
                    ]].copy()
                    
                    display['ROE'] = (display['ROE'] * 100).round(1).astype(str) + '%'
                    display['P/B'] = display['P/B'].round(2)
                    display['Market Cap'] = display['Market Cap'].apply(
                        lambda x: f"${x/1e9:.1f}B" if pd.notnull(x) else 'N/A'
                    )
                    
                    # Show styled dataframe - fixed indentation here
                    st.dataframe(
                        display.style.apply(highlight_undervalued, axis=1),
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Download button
                    st.download_button(
                        "Download Results",
                        undervalued.to_csv(index=False),
                        "undervalued_stocks.csv",
                        "text/csv"
                    )
                else:
                    st.warning("No matches found. Try adjusting filters.")
            else:
                st.error("Failed to load financial data. Check your symbols.")

if __name__ == "__main__":
    main()
