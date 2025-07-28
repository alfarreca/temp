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
        
        return {
            'P/E': info.get('trailingPE', None),
            'P/B': info.get('priceToBook', None),
            'ROE': info.get('returnOnEquity', None),
            'Market Cap': info.get('marketCap', None),
            'Dividend Yield': info.get('dividendYield', 0)
        }
    except:
        return None

def main():
    st.title("📊 Value Screener")
    st.markdown("Screen for stocks with **low P/B**, **low P/E**, and **high ROE**")
    
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
            
            # Get unique values safely
            asset_types = ['All'] + sorted(df['Asset_Type'].dropna().unique().tolist())
            selected_asset_type = st.selectbox("Asset Type", asset_types)
            
            sectors = ['All'] + sorted(df['Sector'].dropna().unique().tolist())
            selected_sector = st.selectbox("Sector", sectors)
            
            countries = ['All'] + sorted(df['Country'].dropna().unique().tolist())
            selected_country = st.selectbox("Country", countries)
            
            st.subheader("Value Metrics")
            pe_max = st.slider("Max P/E", 0, 50, 15)
            pb_max = st.slider("Max P/B", 0.0, 5.0, 2.0, 0.1)
            roe_min = st.slider("Min ROE (%)", 0, 50, 15)
        
        # Apply filters
        filtered_df = df.copy()
        if selected_asset_type != 'All':
            filtered_df = filtered_df[filtered_df['Asset_Type'] == selected_asset_type]
        if selected_sector != 'All':
            filtered_df = filtered_df[filtered_df['Sector'] == selected_sector]
        if selected_country != 'All':
            filtered_df = filtered_df[filtered_df['Country'] == selected_country]
        
        # Fetch financial data
        if not filtered_df.empty:
            with st.spinner("Fetching financial data..."):
                metrics_data = []
                for symbol in filtered_df['Symbol'].unique():
                    metrics = fetch_financials(symbol)
                    if metrics:
                        metrics['Symbol'] = symbol
                        metrics_data.append(metrics)
                
                if metrics_data:
                    metrics_df = pd.DataFrame(metrics_data)
                    merged_df = pd.merge(filtered_df, metrics_df, on='Symbol', how='left')
                    
                    # Apply value filters
                    value_df = merged_df[
                        (merged_df['P/E'].notna()) & 
                        (merged_df['P/B'].notna()) & 
                        (merged_df['ROE'].notna())
                    ].copy()
                    
                    value_df = value_df[
                        (value_df['P/E'] <= pe_max) & 
                        (value_df['P/B'] <= pb_max) & 
                        (value_df['ROE'] >= roe_min/100)
                    ].sort_values(by=['P/E', 'P/B'])
                    
                    # Display results
                    st.subheader(f"Results: {len(value_df)} companies")
                    
                    if not value_df.empty:
                        # Format display
                        display_df = value_df[[
                            'Symbol', 'Name', 'Sector', 'Industry', 'Country',
                            'P/E', 'P/B', 'ROE', 'Market Cap', 'Dividend Yield'
                        ]].copy()
                        
                        display_df['ROE'] = (display_df['ROE'] * 100).round(1).astype(str) + '%'
                        display_df['Dividend Yield'] = (display_df['Dividend Yield'] * 100).round(2).astype(str) + '%'
                        display_df['Market Cap'] = display_df['Market Cap'].apply(
                            lambda x: f"${x/1e9:.2f}B" if pd.notnull(x) else 'N/A'
                        )
                        
                        st.dataframe(display_df, hide_index=True)
                        
                        # Download
                        csv = value_df.to_csv(index=False)
                        st.download_button(
                            "Download Results",
                            csv,
                            "value_screener_results.csv",
                            "text/csv"
                        )
                    else:
                        st.warning("No companies match all filters")
                else:
                    st.error("Failed to fetch financial data")
        else:
            st.warning("No companies match your filters")
    
    elif uploaded_file is None:
        st.info("Please upload an Excel file to begin")

if __name__ == "__main__":
    main()
