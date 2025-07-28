import streamlit as st
import pandas as pd
import yfinance as yf
from io import BytesIO
import time

# App configuration
st.set_page_config(
    page_title="Dynamic Stock Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache data loading to improve performance
@st.cache_data(ttl=3600)
def load_data(uploaded_file):
    """Load and preprocess the Excel file"""
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload an Excel (.xlsx) or CSV file.")
            return None
        
        # Ensure required columns exist (case-insensitive)
        required_columns = ['symbol', 'exchange', 'sector', 'industry', 'theme', 
                           'name', 'country', 'notes', 'asset_type']
        
        # Convert column names to lowercase for comparison
        df.columns = df.columns.str.lower()
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Cache yfinance data to avoid repeated API calls
@st.cache_data(ttl=3600)
def get_yfinance_data(ticker, exchange):
    """Get additional metrics from yfinance"""
    try:
        # Handle exchanges - this is simplified and may need adjustment
        full_ticker = f"{ticker}.{exchange.lower()}" if exchange else ticker
        
        stock = yf.Ticker(full_ticker)
        info = stock.info
        
        metrics = {
            'pe_ratio': info.get('trailingPE', None),
            'pb_ratio': info.get('priceToBook', None),
            'roe': info.get('returnOnEquity', None),
            'market_cap': info.get('marketCap', None),
            'dividend_yield': info.get('dividendYield', None),
            '52_week_high': info.get('fiftyTwoWeekHigh', None),
            '52_week_low': info.get('fiftyTwoWeekLow', None)
        }
        
        # Get historical data for sparklines
        hist = stock.history(period="1y")
        if not hist.empty:
            metrics['price_trend'] = hist['Close'].values.tolist()
            metrics['volume_trend'] = hist['Volume'].values.tolist()
        
        return metrics
    except:
        return None

def value_screener_filters():
    """Filters for value screener"""
    st.sidebar.subheader("Value Screener Filters")
    
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        max_pe = st.number_input("Max P/E Ratio", min_value=0, max_value=100, value=15)
    with col2:
        max_pb = st.number_input("Max P/B Ratio", min_value=0, max_value=20, value=3, step=1)
    with col3:
        min_roe = st.number_input("Min ROE (%)", min_value=0, max_value=100, value=15)
    
    return {
        'max_pe': max_pe,
        'max_pb': max_pb,
        'min_roe': min_roe / 100  # Convert to decimal
    }

def apply_value_screener(df, filters):
    """Apply value screener logic"""
    filtered_df = df.copy()
    
    # Get additional metrics for each stock
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    metrics_data = []
    total_stocks = len(filtered_df)
    
    for i, (_, row) in enumerate(filtered_df.iterrows()):
        status_text.text(f"Fetching data for {row['symbol']} ({i+1}/{total_stocks})...")
        progress_bar.progress((i + 1) / total_stocks)
        
        metrics = get_yfinance_data(row['symbol'], row.get('exchange'))
        if metrics:
            metrics['symbol'] = row['symbol']
            metrics_data.append(metrics)
    
    progress_bar.empty()
    status_text.empty()
    
    if not metrics_data:
        st.warning("No financial metrics could be fetched for these stocks.")
        return pd.DataFrame()
    
    metrics_df = pd.DataFrame(metrics_data)
    filtered_df = filtered_df.merge(metrics_df, on='symbol', how='left')
    
    # Apply filters
    if filters['max_pe'] > 0:
        filtered_df = filtered_df[(filtered_df['pe_ratio'] <= filters['max_pe']) | 
                               (filtered_df['pe_ratio'].isna())]
    
    if filters['max_pb'] > 0:
        filtered_df = filtered_df[(filtered_df['pb_ratio'] <= filters['max_pb']) | 
                               (filtered_df['pb_ratio'].isna())]
    
    if filters['min_roe'] > 0:
        filtered_df = filtered_df[(filtered_df['roe'] >= filters['min_roe']) | 
                               (filtered_df['roe'].isna())]
    
    return filtered_df

def display_results(df, screener_type):
    """Display the filtered results with metrics"""
    if df.empty:
        st.warning("No stocks match your criteria.")
        return
    
    st.subheader(f"{screener_type} Results")
    st.write(f"Found {len(df)} matching stocks")
    
    # Display key metrics
    if screener_type == "Value Screener":
        cols_to_show = ['symbol', 'name', 'sector', 'industry', 'pe_ratio', 
                        'pb_ratio', 'roe', 'market_cap', 'dividend_yield']
        
        # Format the DataFrame for display
        display_df = df[cols_to_show].copy()
        display_df['market_cap'] = display_df['market_cap'].apply(
            lambda x: f"${x/1e9:.2f}B" if pd.notnull(x) else "N/A")
        display_df['dividend_yield'] = display_df['dividend_yield'].apply(
            lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
        
        st.dataframe(
            display_df.sort_values(by='pe_ratio', ascending=True),
            use_container_width=True,
            height=600
        )
    
    # Add sparkline charts if we have the data
    if 'price_trend' in df.columns:
        st.subheader("Price Trends")
        cols = st.columns(4)
        
        for i, (_, row) in enumerate(df.head(8).iterrows()):
            with cols[i % 4]:
                if isinstance(row['price_trend'], list):
                    st.line_chart(row['price_trend'], use_container_width=True)
                    st.caption(f"{row['symbol']} - 1Y Price Trend")
    
    # Add download button
    st.subheader("Export Results")
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Screener_Results')
    
    st.download_button(
        label="Download Excel",
        data=output.getvalue(),
        file_name=f"{screener_type.replace(' ', '_')}_Results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def main():
    """Main app function"""
    st.title("📊 Dynamic Stock Screener")
    st.write("Upload your stock universe and apply different screening strategies.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your stock universe (Excel or CSV)", 
        type=['xlsx', 'csv']
    )
    
    if not uploaded_file:
        st.info("Please upload a file to begin screening.")
        return
    
    # Load data
    df = load_data(uploaded_file)
    if df is None:
        return
    
    # Screener selection
    screener_type = st.sidebar.selectbox(
        "Select Screener Type",
        ["Value Screener", "Growth Screener", "Dividend Screener", "Technical Screener"],
        index=0
    )
    
    # Apply selected screener
    if screener_type == "Value Screener":
        filters = value_screener_filters()
        filtered_df = apply_value_screener(df, filters)
        display_results(filtered_df, screener_type)
    else:
        st.warning(f"{screener_type} is coming soon! Currently only Value Screener is implemented.")

if __name__ == "__main__":
    main()
