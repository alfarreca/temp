import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import tempfile
import os

# App title and description
st.title('📈 Ticker Price Change Tracker')
st.markdown("Track weekly price changes for your stock tickers")

# Function to get proper trading weeks (Monday to Friday)
def get_week_dates(weeks_back):
    today = datetime.now()
    last_friday = today - timedelta(days=(today.weekday() + 3) % 7)
    end_date = last_friday - timedelta(weeks=weeks_back-1)
    start_date = end_date - timedelta(days=4)  # Monday of that week
    return start_date, end_date

# Function to download ticker data with retries
def download_ticker(ticker, start, end, max_retries=3):
    for attempt in range(max_retries):
        try:
            data = yf.download(
                ticker,
                start=start,
                end=end + timedelta(days=1),  # Ensure we get the end date
                progress=False
            )
            if not data.empty:
                return data
        except Exception:
            if attempt == max_retries - 1:
                st.warning(f"Failed to fetch {ticker} after {max_retries} attempts")
            continue
    return None

# File uploader
uploaded_file = st.file_uploader("Upload Excel file with tickers", type=['xlsx'])

if uploaded_file is not None:
    try:
        # Read the Excel file
        df_tickers = pd.read_excel(uploaded_file)
        
        if 'Symbol' not in df_tickers.columns:
            st.error("The Excel file must contain a 'Symbol' column")
            st.stop()
            
        tickers = df_tickers['Symbol'].dropna().astype(str).str.upper().unique()
        
        if len(tickers) > 0:
            st.success(f"Found {len(tickers)} tickers in the uploaded file")
            
            # Add troubleshooting options
            with st.expander("⚙️ Advanced Options"):
                st.write("Try these if you're getting N/A results:")
                add_suffix = st.checkbox("Try adding common exchange suffixes", True)
                suffixes = st.multiselect(
                    "Select suffixes to try",
                    ['.TO', '.NS', '.L', '.DE', '.PA', '.AS', '.BR', '.AX', '.SI', '.KS'],
                    ['.TO', '.NS', '.L']
                )
            
            if st.button('🚀 Fetch Price Changes', type='primary'):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = pd.DataFrame(index=tickers)
                debug_info = []
                total_operations = len(tickers) * 6  # 6 weeks per ticker
                completed_operations = 0
                
                for week in range(1, 7):
                    start_date, end_date = get_week_dates(week)
                    week_label = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                    
                    changes = []
                    for ticker in tickers:
                        # Update progress
                        completed_operations += 1
                        progress = int(100 * completed_operations / total_operations)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {ticker} - Week {week}...")
                        
                        # Try different variations if enabled
                        ticker_variations = [ticker]
                        if add_suffix:
                            ticker_variations.extend([f"{ticker}{suffix}" for suffix in suffixes])
                        
                        found_data = False
                        for t in ticker_variations:
                            data = download_ticker(t, start_date, end_date)
                            if data is not None and not data.empty:
                                start_price = data['Close'].iloc[0]
                                end_price = data['Close'].iloc[-1]
                                change = (end_price - start_price) / start_price * 100
                                changes.append(change)
                                debug_info.append({
                                    'ticker': ticker,
                                    'variation': t,
                                    'week': week,
                                    'status': 'success'
                                })
                                found_data = True
                                break
                            else:
                                debug_info.append({
                                    'ticker': ticker,
                                    'variation': t,
                                    'week': week,
                                    'status': 'failed'
                                })
                        
                        if not found_data:
                            changes.append(None)
                    
                    results[f'Week {week}'] = changes
                    results[f'Week {week} Dates'] = week_label
                
                progress_bar.empty()
                status_text.empty()
                
                # Display results - FIXED FORMATTING HERE
                st.subheader('📅 Weekly Price Changes (%)')
                
                # Create display dataframe
                display_df = results[[f'Week {i}' for i in range(1, 7)]].copy()
                display_df.columns = [f'Week {i}' for i in range(1, 7)]
                display_df.index = tickers
                
                # Format percentages safely
                def format_percent(x):
                    try:
                        return f"{float(x):.2f}%" if pd.notnull(x) else "N/A"
                    except:
                        return "N/A"
                
                formatted_df = display_df.applymap(format_percent)
                
                # Color formatting
                def color_negative_red(val):
                    if val != "N/A":
                        try:
                            val_num = float(val.strip('%'))
                            color = 'red' if val_num < 0 else 'green'
                            return f'color: {color}; font-weight: bold'
                        except:
                            return ''
                    return ''
                
                # Apply styling
                styled_df = formatted_df.style.applymap(color_negative_red)
                st.dataframe(styled_df)
                
                # Show date ranges
                st.caption("📆 Trading weeks (Monday to Friday):")
                for i in range(1, 7):
                    st.caption(f"Week {i}: {results[f'Week {i} Dates'].iloc[0]}")
                
                # Show debug information
                debug_df = pd.DataFrame(debug_info)
                failed_tickers = debug_df[(debug_df['status'] == 'failed') & 
                                       (debug_df['week'] == 1)]['ticker'].unique()
                
                if len(failed_tickers) > 0:
                    st.warning(f"⚠️ Could not fetch data for: {', '.join(failed_tickers)}")
                    with st.expander("Debug details"):
                        st.write("Tried these variations:")
                        st.dataframe(debug_df.groupby(['ticker', 'variation'])['status'].value_counts())
                
                # Download options
                st.download_button(
                    "💾 Download Results as CSV",
                    results.to_csv(index=True).encode('utf-8'),
                    "ticker_price_changes.csv",
                    "text/csv"
                )
        else:
            st.error("No tickers found in the uploaded file.")
            
    except Exception as e:
        st.error(f"Error reading the Excel file: {str(e)}")
else:
    st.info("ℹ️ Please upload an Excel file with at least a 'Symbol' column")
