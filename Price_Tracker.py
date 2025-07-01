import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import tempfile
import os

# App title
st.title('Ticker Price Change Tracker')

# Function to calculate weeks (FIXED - now returns proper date ranges)
def get_week_dates(weeks_back):
    end_date = datetime.now() - timedelta(weeks=weeks_back-1)
    start_date = end_date - timedelta(weeks=1)
    return start_date, end_date

# Improved price change function with better error handling
def get_price_change(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if len(data) > 0:
            start_price = data['Close'].iloc[0]
            end_price = data['Close'].iloc[-1]
            return (end_price - start_price) / start_price * 100
        return None
    except Exception as e:
        st.warning(f"Error fetching {ticker}: {str(e)}")
        return None

# File uploader
uploaded_file = st.file_uploader("Upload Excel file with tickers", type=['xlsx'])

if uploaded_file is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        df_tickers = pd.read_excel(tmp_file_path)
        os.unlink(tmp_file_path)
        
        if 'Symbol' not in df_tickers.columns:
            st.error("The Excel file must contain a 'Symbol' column")
            st.stop()
            
        exchange_col = df_tickers['Exchange'] if 'Exchange' in df_tickers.columns else None
        tickers = df_tickers['Symbol'].dropna().astype(str).unique()
        
        if len(tickers) > 0:
            st.success(f"Found {len(tickers)} tickers in the uploaded file")
            
            if st.button('Fetch Price Changes'):
                with st.spinner('Fetching data from Yahoo Finance...'):
                    results = pd.DataFrame(index=tickers)
                    failed_tickers = []
                    
                    for week in range(1, 7):
                        start_date, end_date = get_week_dates(week)
                        week_label = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                        
                        changes = []
                        for ticker in tickers:
                            full_ticker = f"{ticker}.{exchange_col[df_tickers['Symbol'] == ticker].iloc[0]}" if exchange_col is not None else ticker
                            change = get_price_change(full_ticker, start_date, end_date)
                            if change is None and week == 1:  # Only track failures on first attempt
                                failed_tickers.append(ticker)
                            changes.append(change)
                        
                        results[f'Week {week}'] = changes
                        results[f'Week {week} Dates'] = week_label
                    
                    # Display results
                    st.subheader('Weekly Price Changes (%)')
                    
                    display_df = results[[f'Week {i}' for i in range(1, 7)]].copy()
                    display_df.columns = [f'Week {i}' for i in range(1, 7)]
                    display_df.index = tickers
                    
                    def format_percent(x):
                        return f"{x:.2f}%" if pd.notnull(x) else "N/A"
                    
                    st.dataframe(display_df.applymap(format_percent))
                    
                    # Show date ranges
                    st.caption("Date ranges for each week:")
                    for i in range(1, 7):
                        st.caption(f"Week {i}: {results[f'Week {i} Dates'].iloc[0]}")
                    
                    # Show warnings for failed tickers
                    if failed_tickers:
                        st.warning(f"Could not fetch data for: {', '.join(set(failed_tickers))}. "
                                  "Please check if tickers need exchange suffixes (e.g. '.TO' for Toronto).")
                    
                    # Download button
                    csv = results.to_csv(index=True).encode('utf-8')
                    st.download_button(
                        "Download Results as CSV",
                        csv,
                        "ticker_price_changes.csv",
                        "text/csv",
                        key='download-csv'
                    )
        else:
            st.error("No tickers found in the uploaded file.")
            
    except Exception as e:
        st.error(f"Error reading the Excel file: {str(e)}")
else:
    st.info("Please upload an Excel file with at least a 'Symbol' column.")
