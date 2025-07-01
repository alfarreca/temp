import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Stock Price Tracker", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
        .main {padding-top: 2rem; padding-bottom: 2rem;}
        .stDataFrame {width: 100%;}
        .positive {color: green; font-weight: bold;}
        .negative {color: red; font-weight: bold;}
        .header {font-size: 1.5rem; margin-bottom: 1rem;}
        .info-text {color: #666; font-size: 0.9rem;}
    </style>
""", unsafe_allow_html=True)

def get_week_boundaries(weeks_back=6):
    """Get the start and end dates for the last N weeks (Monday to Friday)"""
    today = datetime.today()
    week_boundaries = []
    
    for i in range(weeks_back):
        days_to_last_monday = today.weekday()  # Monday is 0, Sunday is 6
        end_date = today - timedelta(days=days_to_last_monday + (6 - today.weekday()))
        start_date = end_date - timedelta(days=4)  # Go back to Monday
        
        # Adjust if we're in the current week (only show up to today)
        if i == 0 and end_date > today:
            end_date = today
        
        week_boundaries.append((start_date, end_date))
        today = start_date - timedelta(days=1)  # Move to previous week
    
    return list(reversed(week_boundaries))  # Return in chronological order

def get_weekly_prices(ticker, week_boundaries):
    """Get weekly closing prices for a ticker"""
    prices = []
    for start_date, end_date in week_boundaries:
        try:
            data = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1))
            if not data.empty:
                # Get the last available price in the week
                weekly_close = data['Adj Close'].iloc[-1]
                prices.append(weekly_close)
            else:
                prices.append(None)
        except:
            prices.append(None)
    return prices

def calculate_weekly_changes(df):
    """Calculate weekly price changes and percentages"""
    for i in range(1, len(df.columns)):
        prev_col = df.columns[i-1]
        current_col = df.columns[i]
        
        # Calculate absolute change
        df[f'{current_col}_change'] = df[current_col] - df[prev_col]
        
        # Calculate percentage change
        df[f'{current_col}_pct'] = (df[current_col] - df[prev_col]) / df[prev_col] * 100
    
    return df

def style_dataframe(val):
    """Style DataFrame cells based on value"""
    if isinstance(val, (float, int)):
        if val > 0:
            return 'color: green'
        elif val < 0:
            return 'color: red'
    return ''

def main():
    st.title("📈 Weekly Stock Price Tracker")
    st.markdown("Upload an Excel file with stock symbols to track their weekly price changes.")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx'], help="Excel file should have 'Symbol' and 'Exchange' columns")
    
    if uploaded_file is not None:
        try:
            # Read the Excel file
            df = pd.read_excel(uploaded_file)
            
            # Validate the file structure
            if not all(col in df.columns for col in ['Symbol', 'Exchange']):
                st.error("The Excel file must contain 'Symbol' and 'Exchange' columns")
                return
            
            # Get week boundaries (last 6 weeks)
            week_boundaries = get_week_boundaries(weeks_back=6)
            week_labels = [f"{start.strftime('%b %d')} - {end.strftime('%b %d')}" 
                          for start, end in week_boundaries]
            
            # Fetch data for all tickers
            progress_bar = st.progress(0)
            ticker_data = []
            
            for idx, row in df.iterrows():
                ticker = f"{row['Symbol']}.{row['Exchange']}" if pd.notna(row['Exchange']) else row['Symbol']
                prices = get_weekly_prices(ticker, week_boundaries)
                ticker_data.append([ticker] + prices)
                progress_bar.progress((idx + 1) / len(df))
            
            # Create DataFrame with weekly prices
            price_df = pd.DataFrame(ticker_data, columns=['Ticker'] + week_labels)
            
            # Calculate weekly changes
            price_df = calculate_weekly_changes(price_df)
            
            # Display the main price table
            st.subheader("Weekly Price Changes")
            
            # Create tabs for different views
            tab1, tab2 = st.tabs(["Price Table", "Change Analysis"])
            
            with tab1:
                # Display prices with conditional formatting
                st.dataframe(
                    price_df.style.applymap(style_dataframe, subset=week_labels[1:]),
                    use_container_width=True
                )
                
                # Download button
                st.download_button(
                    label="Download Data as CSV",
                    data=price_df.to_csv(index=False),
                    file_name="weekly_stock_prices.csv",
                    mime="text/csv"
                )
            
            with tab2:
                # Show percentage changes heatmap
                st.subheader("Weekly Percentage Changes")
                
                # Prepare data for heatmap
                pct_cols = [col for col in price_df.columns if '_pct' in col]
                heatmap_data = price_df.set_index('Ticker')[pct_cols]
                heatmap_data.columns = [col.replace('_pct', '') for col in heatmap_data.columns]
                
                # Plot heatmap
                fig, ax = plt.subplots(figsize=(12, max(6, len(heatmap_data) * 0.5)))
                sns.heatmap(
                    heatmap_data,
                    annot=True,
                    fmt=".1f",
                    cmap="RdYlGn",
                    center=0,
                    linewidths=0.5,
                    ax=ax
                )
                ax.set_title("Weekly Price Change Percentage (%)")
                st.pyplot(fig)
                
                # Show best/worst performers
                st.subheader("Performance Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Top Performers (Last Week)**")
                    last_week_pct = pct_cols[-1]
                    top_performers = price_df.nlargest(5, last_week_pct)[['Ticker', last_week_pct]]
                    st.dataframe(top_performers.style.format({last_week_pct: "{:.1f}%"}))
                
                with col2:
                    st.markdown("**Worst Performers (Last Week)**")
                    worst_performers = price_df.nsmallest(5, last_week_pct)[['Ticker', last_week_pct]]
                    st.dataframe(worst_performers.style.format({last_week_pct: "{:.1f}%"}))
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
