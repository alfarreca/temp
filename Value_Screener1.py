import streamlit as st
import pandas as pd
import yfinance as yf
import io
from datetime import datetime, timedelta

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
        
        # Ensure required columns are present
        required_cols = ['Symbol', 'Exchange', 'Sector', 'Industry', 'Theme', 'Name', 'Country', 'Notes', 'Asset_Type']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Fetch additional metrics from Yahoo Finance
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_yfinance_data(symbol, exchange):
    try:
        # Handle symbols with exchange suffixes
        ticker = f"{symbol}.{exchange}" if exchange else symbol
        stock = yf.Ticker(ticker)
        
        info = stock.info
        history = stock.history(period="1mo")
        
        metrics = {
            'P/E': info.get('trailingPE', None),
            'P/B': info.get('priceToBook', None),
            'ROE': info.get('returnOnEquity', None),
            'Revenue Growth': info.get('revenueGrowth', None),
            'EPS Growth': info.get('earningsGrowth', None),
            'PEG Ratio': info.get('pegRatio', None),
            'Dividend Yield': info.get('dividendYield', None),
            'Payout Ratio': info.get('payoutRatio', None),
            'RSI': None,  # Would need calculation
            'Current Price': info.get('currentPrice', None),
            '52W High': info.get('fiftyTwoWeekHigh', None),
            '52W Low': info.get('fiftyTwoWeekLow', None),
            'Price History': history['Close'] if not history.empty else None
        }
        
        return metrics
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

# Define screener types and their filters
SCREENER_TYPES = {
    "Value Screener": {
        "description": "Finding cheap but strong businesses",
        "filters": {
            "P/B": {"type": "slider", "min": 0, "max": 10, "value": (0, 3), "step": 0.1},
            "P/E": {"type": "slider", "min": 0, "max": 50, "value": (0, 15), "step": 0.5},
            "ROE": {"type": "slider", "min": 0, "max": 100, "value": (15, 100), "step": 1},
            "Sector": {"type": "multiselect", "options": []},
            "Country": {"type": "multiselect", "options": []}
        }
    },
    "Growth Screener": {
        "description": "Identifying fast-growing companies",
        "filters": {
            "Revenue Growth": {"type": "slider", "min": -50, "max": 200, "value": (20, 200), "step": 1},
            "EPS Growth": {"type": "slider", "min": -100, "max": 200, "value": (20, 200), "step": 1},
            "PEG Ratio": {"type": "slider", "min": 0, "max": 5, "value": (0, 1.5), "step": 0.1},
            "Sector": {"type": "multiselect", "options": []},
            "Industry": {"type": "multiselect", "options": []}
        }
    },
    "Dividend Screener": {
        "description": "Income-focused investing",
        "filters": {
            "Dividend Yield": {"type": "slider", "min": 0, "max": 20, "value": (3, 20), "step": 0.1},
            "Payout Ratio": {"type": "slider", "min": 0, "max": 200, "value": (0, 80), "step": 1},
            "Sector": {"type": "multiselect", "options": []},
            "Asset_Type": {"type": "multiselect", "options": []}
        }
    },
    "Technical Screener": {
        "description": "Short-term trading setups",
        "filters": {
            "RSI": {"type": "slider", "min": 0, "max": 100, "value": (30, 70), "step": 1},
            "Price vs 50D MA": {"type": "select", "options": ["Above", "Below", "Any"]},
            "Price vs 200D MA": {"type": "select", "options": ["Above", "Below", "Any"]},
            "Volume Change": {"type": "slider", "min": -100, "max": 500, "value": (20, 500), "step": 5}
        }
    },
    "Thematic Screener": {
        "description": "Focused investing on trends/themes",
        "filters": {
            "Theme": {"type": "multiselect", "options": []},
            "Sector": {"type": "multiselect", "options": []},
            "Country": {"type": "multiselect", "options": []},
            "Asset_Type": {"type": "multiselect", "options": []}
        }
    }
}

# Main app function
def main():
    st.title("📊 Universal Stock Screener")
    st.write("Upload your stock universe and apply different screening strategies")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Stock Universe (Excel/CSV)", 
        type=["xlsx", "csv"],
        help="File should contain: Symbol, Exchange, Sector, Industry, Theme, Name, Country, Notes, Asset_Type"
    )
    
    if not uploaded_file:
        st.info("Please upload a stock universe file to begin screening")
        return
    
    # Load data
    df = load_data(uploaded_file)
    if df is None:
        return
    
    # Update filter options based on data
    for screener in SCREENER_TYPES.values():
        for filter_name, filter_config in screener["filters"].items():
            if filter_config["type"] == "multiselect":
                if filter_name in df.columns:
                    filter_config["options"] = sorted(df[filter_name].dropna().unique().tolist())
                else:
                    filter_config["options"] = []
    
    # Screener selection
    screener_type = st.sidebar.selectbox(
        "Select Screener Type",
        options=list(SCREENER_TYPES.keys()),
        format_func=lambda x: f"{x} - {SCREENER_TYPES[x]['description']}"
    )
    
    st.sidebar.markdown(f"**{screener_type}**")
    st.sidebar.caption(SCREENER_TYPES[screener_type]["description"])
    
    # Apply filters
    st.sidebar.subheader("Filters")
    filters = {}
    
    for filter_name, filter_config in SCREENER_TYPES[screener_type]["filters"].items():
        if filter_config["type"] == "slider":
            filters[filter_name] = st.sidebar.slider(
                filter_name,
                min_value=filter_config["min"],
                max_value=filter_config["max"],
                value=filter_config["value"],
                step=filter_config["step"]
            )
        elif filter_config["type"] == "multiselect" and filter_config["options"]:
            selected = st.sidebar.multiselect(
                filter_name,
                options=filter_config["options"],
                default=filter_config["options"]
            )
            filters[filter_name] = selected
        elif filter_config["type"] == "select":
            filters[filter_name] = st.sidebar.selectbox(
                filter_name,
                options=filter_config["options"]
            )
    
    # Fetch additional metrics for each stock
    if st.sidebar.button("Apply Screening", type="primary"):
        with st.spinner("Fetching stock data and applying filters..."):
            # Create a copy of the original dataframe
            screened_df = df.copy()
            
            # Add columns for metrics we'll fetch
            metrics_to_fetch = list(SCREENER_TYPES[screener_type]["filters"].keys())
            for metric in metrics_to_fetch:
                if metric not in screened_df.columns:
                    screened_df[metric] = None
            
            # Add price history column
            screened_df['Price History'] = None
            
            # Fetch data for each stock
            progress_bar = st.progress(0)
            for i, row in screened_df.iterrows():
                symbol = row['Symbol']
                exchange = row['Exchange']
                
                metrics = fetch_yfinance_data(symbol, exchange)
                if metrics:
                    for metric_name, metric_value in metrics.items():
                        if metric_name in screened_df.columns:
                            screened_df.at[i, metric_name] = metric_value
                
                progress_bar.progress((i + 1) / len(screened_df))
            
            # Apply filters
            for filter_name, filter_value in filters.items():
                if filter_name in screened_df.columns:
                    if isinstance(filter_value, tuple):  # Slider range
                        min_val, max_val = filter_value
                        screened_df = screened_df[
                            (screened_df[filter_name] >= min_val) & 
                            (screened_df[filter_name] <= max_val)
                        ]
                    elif isinstance(filter_value, list):  # Multiselect
                        if filter_value:  # Only filter if options are selected
                            screened_df = screened_df[screened_df[filter_name].isin(filter_value)]
                    elif isinstance(filter_value, str) and filter_value != "Any":  # Select box
                        if filter_name == "Price vs 50D MA":
                            # This would require actual calculation of moving averages
                            pass
                        elif filter_name == "Price vs 200D MA":
                            pass
                        else:
                            screened_df = screened_df[screened_df[filter_name] == filter_value]
            
            # Display results
            st.subheader(f"{screener_type} Results")
            st.write(f"Found {len(screened_df)} stocks matching your criteria")
            
            if not screened_df.empty:
                # Display important columns
                display_cols = ['Symbol', 'Name', 'Sector', 'Industry', 'Country'] + metrics_to_fetch
                display_cols = [col for col in display_cols if col in screened_df.columns]
                
                st.dataframe(
                    screened_df[display_cols],
                    use_container_width=True,
                    height=600
                )
                
                # Add download button
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    screened_df.to_excel(writer, index=False)
                st.download_button(
                    label="Download Screened Stocks",
                    data=output.getvalue(),
                    file_name=f"{screener_type.replace(' ', '_')}_Results_{datetime.now().date()}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                # Optional: Show charts for a few metrics
                if 'Price History' in screened_df.columns and not all(screened_df['Price History'].isna()):
                    st.subheader("Price Trends for Selected Stocks")
                    
                    # Display charts for first 5 stocks
                    for i, row in screened_df.head(5).iterrows():
                        if row['Price History'] is not None:
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.metric(label=row['Symbol'], value=f"${row.get('Current Price', 'N/A')}")
                                if 'Dividend Yield' in row:
                                    st.metric("Dividend Yield", f"{row['Dividend Yield']:.2%}" if pd.notna(row['Dividend Yield']) else "N/A")
                            
                            with col2:
                                st.line_chart(row['Price History'], use_container_width=True)

if __name__ == "__main__":
    main()
