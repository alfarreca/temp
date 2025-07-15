import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px

# App title and description
st.title("📊 Undervalued Stock Screener")
st.subheader("Warren Buffett-inspired Value Investing Strategy")
st.write("""
This app helps identify potentially undervalued stocks based on fundamental analysis metrics.
Upload your Excel file with stock symbols to get started.
""")

# Sidebar for user inputs
st.sidebar.header("Valuation Parameters")  # Fixed typo

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload your Excel file with stock tickers", 
    type=["xlsx"],
    help="File should contain 'Symbol' and 'Exchange' columns"
)

# Valuation parameters with better defaults and formatting
pe_ratio_threshold = st.sidebar.slider(
    "Max P/E Ratio", 
    min_value=5, max_value=30, value=15, step=1
)

peg_ratio_threshold = st.sidebar.slider(
    "Max PEG Ratio", 
    min_value=0.5, max_value=2.0, value=1.0, step=0.1,
    format="%.1f"  # Better decimal formatting
)

debt_to_equity_threshold = st.sidebar.slider(
    "Max Debt-to-Equity", 
    min_value=0.1, max_value=2.0, value=0.5, step=0.1,
    format="%.1f"
)

current_ratio_threshold = st.sidebar.slider(
    "Min Current Ratio", 
    min_value=1.0, max_value=3.0, value=1.5, step=0.1,
    format="%.1f"
)

roa_threshold = st.sidebar.slider(
    "Min Return on Assets (%)", 
    min_value=1, max_value=20, value=5, step=1
)

# Function to get financial data (same as before)
def get_stock_data(ticker, exchange):
    # ... (keep the same implementation as previous code)
    pass

# Main app logic with better empty state handling
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        
        if 'Symbol' not in df.columns:
            st.error("The uploaded file must contain a 'Symbol' column")
            st.stop()
            
        if 'Exchange' not in df.columns:
            df['Exchange'] = ''
            
        if st.button("Analyze Stocks"):
            # ... (keep the same analysis logic)
            
            if len(filtered_df) == 0:
                st.warning("""
                No stocks met all the valuation criteria. Try adjusting the parameters:
                - Increase P/E or PEG thresholds
                - Decrease Debt-to-Equity requirement
                - Lower ROA requirement
                """)
            else:
                # Show results
                st.write("## Undervalued Stock Candidates")
                
                # Format the display better
                st.dataframe(
                    filtered_df.style.format({
                        'CurrentPrice': '${:.2f}',
                        'P/E': '{:.1f}',
                        'PEG': '{:.1f}',
                        'ROA': '{:.1f}%',
                        'ROE': '{:.1f}%',
                        'DiscountFromHigh': '{:.1f}%'
                    }),
                    height=400
                )
                
                # Better visualizations
                st.write("## Valuation Metrics Distribution")
                
                if not results_df.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.box(
                            results_df, 
                            y='P/E',
                            title='P/E Ratio Distribution',
                            points="all"
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.box(
                            results_df, 
                            y='Price/Book',
                            title='Price-to-Book Distribution',
                            points="all"
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("""
    To begin analysis:
    1. Prepare an Excel file with stock symbols
    2. Upload using the file uploader in the sidebar
    3. Adjust valuation parameters as needed
    4. Click 'Analyze Stocks' button
    """)

# Educational content
st.markdown("""
## Interpretation Guide

**Key Metrics**:
- **P/E Ratio**: Price-to-Earnings ratio (lower is better)
- **PEG Ratio**: P/E divided by growth rate (<1 suggests undervalued)
- **Debt/Equity**: Financial leverage (lower is safer)
- **Current Ratio**: Short-term liquidity (>1.5 is healthy)
- **ROA**: Return on Assets (>5% is good)

**Tips**:
- Start with broader parameters and narrow down
- Consider sector averages when evaluating metrics
- Combine with qualitative analysis
""")
