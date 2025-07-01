import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import openpyxl

# Enhanced import handling with user feedback
try:
    import yfinance as yf
    yf.pdr_override()  # Ensure proper yfinance functionality
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    st.error("❌ yfinance package not installed. Please run: pip install yfinance")
    st.stop()

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False
    st.warning("⚠️ Visualization features disabled - matplotlib or seaborn not installed")

# Set page config
st.set_page_config(
    page_title="Stock Price Tracker", 
    layout="wide",
    page_icon="📈"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .main {padding: 2rem;}
        .stDataFrame {width: 100%;}
        .positive {color: #4CAF50; font-weight: bold;}
        .negative {color: #F44336; font-weight: bold;}
        .header {font-size: 1.5rem; margin-bottom: 1rem;}
        .info-text {color: #666; font-size: 0.9rem;}
        .small-font {font-size: 0.8rem;}
        .ticker-header {background-color: #f0f2f6; border-radius: 5px; padding: 0.5rem;}
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
    """Get weekly closing prices for a ticker with error handling"""
    prices = []
    for start_date, end_date in week_boundaries:
        try:
            data = yf.download(
                ticker, 
                start=start_date, 
                end=end_date + timedelta(days=1),
                progress=False
            )
            if not data.empty:
                weekly_close = data['Adj Close'].iloc[-1]
                prices.append(round(weekly_close, 2))
            else:
                prices.append(None)
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
            prices.append(None)
    return prices

def calculate_weekly_changes(df, week_labels):
    """Calculate weekly price changes and percentages"""
    for i in range(1, len(week_labels)):
        prev_col = week_labels[i-1]
        current_col = week_labels[i]
        
        # Calculate absolute change
        df[f'{current_col}_change'] = df[current_col] - df[prev_col]
        
        # Calculate percentage change
        df[f'{current_col}_pct'] = (df[current_col] - df[prev_col]) / df[prev_col] * 100
    
    return df

def style_dataframe(val):
    """Style DataFrame cells based on value"""
    if isinstance(val, (float, int)):
        if val > 0:
            return 'color: #4CAF50'
        elif val < 0:
            return 'color: #F44336'
    return ''

def display_visualizations(price_df, week_labels):
    """Display visualizations if packages are available"""
    if not VIZ_AVAILABLE:
        st.warning("Visualizations unavailable - required packages not installed")
        return
    
    with st.expander("📊 Advanced Visualizations", expanded=True):
        tab1, tab2 = st.tabs(["Weekly Changes Heatmap", "Performance Analysis"])
        
        with tab1:
            st.subheader("Weekly Percentage Changes Heatmap")
            
            # Prepare data for heatmap
            pct_cols = [f"{col}_pct" for col in week_labels[1:]]
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
            st.pyplot(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Performance Analysis")
            
            # Calculate overall performance
            price_df['Total Change %'] = (
                (price_df[week_labels[-1]] - price_df[week_labels[0]]) / 
                price_df[week_labels[0]] * 100
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top Performers (Overall Period)**")
                top_performers = price_df.nlargest(
                    5, 'Total Change %')[['Ticker', 'Total Change %']]
                st.dataframe(
                    top_performers.style.format({'Total Change %': "{:.1f}%"}),
                    hide_index=True
                )
            
            with col2:
                st.markdown("**Worst Performers (Overall Period)**")
                worst_performers = price_df.nsmallest(
                    5, 'Total Change %')[['Ticker', 'Total Change %']]
                st.dataframe(
                    worst_performers.style.format({'Total Change %': "{:.1f}%"}),
                    hide_index=True
                )

def main():
    st.title("📈 Weekly Stock Price Tracker")
    st.markdown("""
        <div class="info-text">
        Upload an Excel file with stock symbols to track their weekly price changes.<br>
        File should contain columns: <strong>Symbol</strong> and <strong>Exchange</strong> (e.g., AAPL for NYSE or 005930.KS for Korean exchange)
        </div>
    """, unsafe_allow_html=True)
    
    # File uploader with example
    with st.expander("📁 Example File Format", expanded=False):
        example_df = pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', '005930'],
            'Exchange': [None, None, 'KS']
        })
        st.dataframe(example_df, hide_index=True)
        st.download_button(
            label="Download Example File",
            data=example_df.to_excel("example_tickers.xlsx", index=False),
            file_name="example_tickers.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    uploaded_file = st.file_uploader(
        "Upload Excel File", 
        type=['xlsx'],
        help="Excel file should have 'Symbol' and 'Exchange' columns"
    )
    
    if uploaded_file is not None:
        try:
            # Read the Excel file
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            # Validate the file structure
            if not all(col in df.columns for col in ['Symbol', 'Exchange']):
                st.error("❌ The Excel file must contain 'Symbol' and 'Exchange' columns")
                return
            
            # Clean data
            df['Exchange'] = df['Exchange'].fillna('')
            
            # Get week boundaries (last 6 weeks)
            week_boundaries = get_week_boundaries(weeks_back=6)
            week_labels = [f"{start.strftime('%b %d')} - {end.strftime('%b %d')}" 
                          for start, end in week_boundaries]
            
            # Fetch data for all tickers
            st.subheader("⏳ Fetching Market Data...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            ticker_data = []
            
            for idx, row in df.iterrows():
                ticker = f"{row['Symbol']}.{row['Exchange']}" if row['Exchange'] else row['Symbol']
                status_text.text(f"Fetching data for {ticker} ({idx+1}/{len(df)})...")
                prices = get_weekly_prices(ticker, week_boundaries)
                ticker_data.append([ticker] + prices)
                progress_bar.progress((idx + 1) / len(df))
            
            # Create DataFrame with weekly prices
            price_df = pd.DataFrame(ticker_data, columns=['Ticker'] + week_labels)
            
            # Calculate weekly changes
            price_df = calculate_weekly_changes(price_df, week_labels)
            
            # Display the main price table
            st.subheader("📊 Weekly Price Changes")
            st.dataframe(
                price_df.style.applymap(style_dataframe, subset=week_labels[1:]),
                use_container_width=True,
                hide_index=True
            )
            
            # Download button
            st.download_button(
                label="💾 Download Data as CSV",
                data=price_df.to_csv(index=False),
                file_name="weekly_stock_prices.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Display visualizations if available
            display_visualizations(price_df, week_labels)
        
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.error("Please check your file format and try again")

if __name__ == "__main__":
    main()
