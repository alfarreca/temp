import streamlit as st
import pandas as pd
import yfinance as yf
from io import BytesIO
import time

# Check for required packages
try:
    import xlsxwriter
except ImportError:
    st.error("Missing required package: xlsxwriter. Please install it with `pip install xlsxwriter`")
    st.stop()

try:
    import openpyxl
except ImportError:
    st.error("Missing required package: openpyxl. Please install it with `pip install openpyxl`")
    st.stop()

# App configuration
st.set_page_config(
    page_title="Dynamic Stock Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=3600)
def load_data(uploaded_file):
    """Load and preprocess the Excel file"""
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
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
            
        # Clean data - convert all sector/country values to strings
        df['sector'] = df['sector'].astype(str)
        df['country'] = df['country'].astype(str)
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_yfinance_data(ticker, exchange):
    """Get additional metrics from yfinance"""
    try:
        full_ticker = f"{ticker}.{exchange.lower()}" if exchange else ticker
        stock = yf.Ticker(full_ticker)
        info = stock.info
        
        metrics = {
            'pe_ratio': float(info.get('trailingPE', 0)) if info.get('trailingPE') else None,
            'pb_ratio': float(info.get('priceToBook', 0)) if info.get('priceToBook') else None,
            'roe': float(info.get('returnOnEquity', 0)) if info.get('returnOnEquity') else None,
            'market_cap': float(info.get('marketCap', 0)) if info.get('marketCap') else None,
            'dividend_yield': float(info.get('dividendYield', 0)) if info.get('dividendYield') else None,
            '52_week_high': float(info.get('fiftyTwoWeekHigh', 0)) if info.get('fiftyTwoWeekHigh') else None,
            '52_week_low': float(info.get('fiftyTwoWeekLow', 0)) if info.get('fiftyTwoWeekLow') else None
        }
        
        hist = stock.history(period="1y")
        if not hist.empty:
            metrics['price_trend'] = hist['Close'].values.tolist()
            metrics['volume_trend'] = hist['Volume'].values.tolist()
        
        return metrics
    except Exception as e:
        st.warning(f"Couldn't fetch data for {ticker}: {str(e)}")
        return None

def value_screener_filters(available_sectors, available_countries):
    """Filters for value screener"""
    st.sidebar.subheader("Value Screener Filters")
    
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        max_pe = st.number_input("Max P/E Ratio", min_value=0, max_value=100, value=15)
    with col2:
        max_pb = st.number_input("Max P/B Ratio", min_value=0, max_value=20, value=3, step=1)
    with col3:
        min_roe = st.number_input("Min ROE (%)", min_value=0, max_value=100, value=15)
    
    st.sidebar.subheader("Additional Filters")
    sectors = st.sidebar.multiselect("Filter by Sector", options=available_sectors)
    countries = st.sidebar.multiselect("Filter by Country", options=available_countries)
    
    return {
        'max_pe': max_pe,
        'max_pb': max_pb,
        'min_roe': min_roe / 100,
        'sectors': sectors,
        'countries': countries
    }

def apply_value_screener(df, filters):
    """Apply value screener logic"""
    filtered_df = df.copy()
    
    if filters['sectors']:
        filtered_df = filtered_df[filtered_df['sector'].isin(filters['sectors'])]
    if filters['countries']:
        filtered_df = filtered_df[filtered_df['country'].isin(filters['countries'])]
    
    if filtered_df.empty:
        return filtered_df
    
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
        time.sleep(0.2)
    
    progress_bar.empty()
    status_text.empty()
    
    if not metrics_data:
        st.warning("No financial metrics could be fetched for these stocks.")
        return pd.DataFrame()
    
    metrics_df = pd.DataFrame(metrics_data)
    filtered_df = filtered_df.merge(metrics_df, on='symbol', how='left')
    
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
    
    cols_to_show = ['symbol', 'name', 'sector', 'industry', 'pe_ratio', 
                   'pb_ratio', 'roe', 'market_cap', 'dividend_yield', 'country']
    
    display_df = df[cols_to_show].copy()
    display_df['market_cap'] = display_df['market_cap'].apply(
        lambda x: f"${x/1e9:.2f}B" if pd.notnull(x) and x > 0 else "N/A")
    display_df['dividend_yield'] = display_df['dividend_yield'].apply(
        lambda x: f"{x*100:.2f}%" if pd.notnull(x) and x > 0 else "N/A")
    display_df['roe'] = display_df['roe'].apply(
        lambda x: f"{x*100:.2f}%" if pd.notnull(x) and x > 0 else "N/A")
    
    st.dataframe(
        display_df.sort_values(by='pe_ratio', ascending=True),
        use_container_width=True,
        height=600,
        column_config={
            "pe_ratio": st.column_config.NumberColumn(format="%.2f"),
            "pb_ratio": st.column_config.NumberColumn(format="%.2f")
        }
    )
    
    if 'price_trend' in df.columns and len(df) > 0:
        st.subheader("Price Trends (1 Year)")
        cols = st.columns(4)
        
        for i, (_, row) in enumerate(df.head(8).iterrows()):
            with cols[i % 4]:
                if isinstance(row['price_trend'], list) and len(row['price_trend']) > 0:
                    st.line_chart(row['price_trend'], use_container_width=True)
                    st.caption(f"{row['symbol']} - Current: {row['price_trend'][-1]:.2f}")
    
    st.subheader("Export Results")
    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Screener_Results')
        
        st.download_button(
            label="📥 Download Excel",
            data=output.getvalue(),
            file_name=f"{screener_type.replace(' ', '_')}_Results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Error creating Excel file: {str(e)}")

def main():
    """Main app function"""
    st.title("📊 Dynamic Stock Screener")
    st.write("Upload your stock universe and apply different screening strategies.")
    
    uploaded_file = st.file_uploader(
        "Upload your stock universe (Excel or CSV)", 
        type=['xlsx', 'csv'],
        key="file_uploader"
    )
    
    if not uploaded_file:
        st.info("Please upload a file to begin screening.")
        return
    
    with st.spinner("Loading data..."):
        df = load_data(uploaded_file)
    
    if df is None:
        return
    
    # Get unique sectors and countries
    available_sectors = sorted(df['sector'].unique().tolist())
    available_countries = sorted(df['country'].unique().tolist())
    
    screener_type = st.sidebar.selectbox(
        "Select Screener Type",
        ["Value Screener", "Growth Screener", "Dividend Screener", "Technical Screener"],
        index=0
    )
    
    if screener_type == "Value Screener":
        filters = value_screener_filters(available_sectors, available_countries)
        with st.spinner("Applying filters and fetching market data..."):
            filtered_df = apply_value_screener(df, filters)
        display_results(filtered_df, screener_type)
    else:
        st.warning(f"{screener_type} is coming soon! Currently only Value Screener is implemented.")

if __name__ == "__main__":
    main()
