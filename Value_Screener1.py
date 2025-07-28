import streamlit as st
import pandas as pd
import yfinance as yf

# App configuration
st.set_page_config(
    page_title="Stock Screener",
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
        cashflow = stock.cashflow
        return {
            # Value metrics
            'P/E': info.get('trailingPE'),
            'P/B': info.get('priceToBook'),
            'ROE': info.get('returnOnEquity'),
            'Market Cap': info.get('marketCap'),
            
            # Dividend metrics
            'Dividend Yield': info.get('dividendYield', 0) * 100,  # Convert to percentage
            'Payout Ratio': info.get('payoutRatio'),
            'Free Cash Flow': cashflow.iloc[0].get('Free Cash Flow') if cashflow is not None else None
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

def highlight_dividend(row):
    """Highlight cells for dividend stocks"""
    style = [''] * len(row)
    if row['Dividend Yield'] >= 5:
        style = ['background-color: #e6f3ff'] * len(row)
    elif row['Dividend Yield'] >= 3:
        style = ['background-color: #f0f7ff'] * len(row)
    return style

def main():
    st.title("📊 Stock Screener")
    
    # Mode selection
    mode = st.sidebar.radio(
        "Screener Mode",
        ["Value Screener", "Dividend Screener"],
        index=0
    )
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel File", 
        type=['xlsx', 'xls'],
        help="Must contain Symbol, Exchange, Sector, Industry columns"
    )
    
    df = load_data(uploaded_file)
    
    if df is not None:
        # Common filters
        with st.sidebar:
            st.header("Basic Filters")
            
            asset_types = ['All'] + sorted(df['Asset_Type'].unique().tolist())
            selected_asset = st.selectbox("Asset Type", asset_types)
            
            sectors = ['All'] + sorted(df['Sector'].unique().tolist())
            selected_sector = st.selectbox("Sector", sectors)
            
            if mode == "Value Screener":
                st.header("Value Metrics")
                pb_max = st.slider("Max P/B", 0.0, 5.0, 1.2, 0.1)
                roe_min = st.slider("Min ROE (%)", 0, 50, 0)
            else:
                st.header("Dividend Metrics")
                div_min = st.slider("Min Dividend Yield (%)", 0.0, 10.0, 3.0, 0.1)
                payout_max = st.slider("Max Payout Ratio (%)", 0, 100, 70)
                fcf_positive = st.checkbox("Positive Free Cash Flow Only", True)
        
        # Apply basic filters
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
                if data and None not in [data.get('P/E'), data.get('P/B'), data.get('ROE')]:
                    data['Symbol'] = symbol
                    results.append(data)
            
            if results:
                financials = pd.DataFrame(results)
                merged = pd.merge(filtered, financials, on='Symbol')
                
                if mode == "Value Screener":
                    # Apply value filters
                    screened = merged[
                        (merged['P/B'] <= pb_max) & 
                        (merged['ROE'] >= roe_min/100)
                    ].sort_values(['P/B', 'ROE'], ascending=[True, False])
                    
                    # Display results
                    st.subheader("🔍 Value Stocks (P/B ≤ 1.2, ROE > 0%)")
                    
                    if not screened.empty:
                        # Format display
                        display = screened[[
                            'Symbol', 'Name', 'Sector', 
                            'P/B', 'ROE', 'Dividend Yield', 'Market Cap'
                        ]].copy()
                        
                        display['ROE'] = (display['ROE'] * 100).round(1).astype(str) + '%'
                        display['P/B'] = display['P/B'].round(2)
                        display['Dividend Yield'] = display['Dividend Yield'].round(2).astype(str) + '%'
                        display['Market Cap'] = display['Market Cap'].apply(
                            lambda x: f"${x/1e9:.1f}B" if pd.notnull(x) else 'N/A'
                        )
                        
                        # Show styled dataframe
                        st.dataframe(
                            display.style.apply(highlight_undervalued, axis=1),
                            hide_index=True,
                            use_container_width=True
                        )
                else:
                    # Apply dividend filters
                    dividend_filter = (
                        (merged['Dividend Yield'] >= div_min) &
                        (merged['Payout Ratio'] <= payout_max/100)
                    )
                    
                    if fcf_positive:
                        dividend_filter &= (merged['Free Cash Flow'] > 0)
                    
                    screened = merged[dividend_filter].sort_values(
                        ['Dividend Yield', 'Payout Ratio'], 
                        ascending=[False, True]
                    )
                    
                    # Display results
                    st.subheader(f"💰 Dividend Stocks (Yield ≥ {div_min}%, Payout ≤ {payout_max}%)")
                    
                    if not screened.empty:
                        # Format display
                        display = screened[[
                            'Symbol', 'Name', 'Sector',
                            'Dividend Yield', 'Payout Ratio', 'Free Cash Flow', 'Market Cap'
                        ]].copy()
                        
                        display['Dividend Yield'] = display['Dividend Yield'].round(2).astype(str) + '%'
                        display['Payout Ratio'] = (display['Payout Ratio'] * 100).round(1).astype(str) + '%'
                        display['Free Cash Flow'] = display['Free Cash Flow'].apply(
                            lambda x: f"${x/1e6:.1f}M" if pd.notnull(x) else 'N/A'
                        )
                        display['Market Cap'] = display['Market Cap'].apply(
                            lambda x: f"${x/1e9:.1f}B" if pd.notnull(x) else 'N/A'
                        )
                        
                        # Show styled dataframe
                        st.dataframe(
                            display.style.apply(highlight_dividend, axis=1),
                            hide_index=True,
                            use_container_width=True
                        )
                
                # Download button (for either mode)
                if not screened.empty:
                    st.download_button(
                        "Download Results",
                        screened.to_csv(index=False),
                        f"{mode.lower().replace(' ', '_')}_stocks.csv",
                        "text/csv"
                    )
                else:
                    st.warning("No matches found. Try adjusting filters.")
            else:
                st.error("Failed to load financial data. Check your symbols.")

if __name__ == "__main__":
    main()
