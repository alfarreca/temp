import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import openpyxl  # For Excel file handling

# Function to get the start and end dates for the last N weeks
def get_week_dates(weeks_back=6):
    today = datetime.today()
    current_weekday = today.weekday()  # Monday=0, Sunday=6
    
    # Find the most recent Sunday (end of week)
    last_sunday = today - timedelta(days=current_weekday + 1 - 7 if current_weekday != 6 else 0)
    if current_weekday == 6 and today.hour < 17:  # If today is Sunday before market close
        last_sunday = today - timedelta(weeks=1)
    
    end_dates = [last_sunday - timedelta(weeks=i) for i in range(weeks_back)]
    start_dates = [end - timedelta(days=6) for end in end_dates]
    
    # Adjust for current week (if we're in the middle of the week)
    if current_weekday != 6:
        end_dates[0] = today
        start_dates[0] = last_sunday - timedelta(days=6)
    
    return list(zip(start_dates, end_dates))[::-1]  # Reverse to go from oldest to newest

# Function to fetch weekly prices for a symbol
def get_weekly_prices(symbol, exchange, week_ranges):
    full_symbol = f"{symbol}.{exchange}" if exchange else symbol
    try:
        data = yf.download(full_symbol, start=week_ranges[0][0], end=week_ranges[-1][1] + timedelta(days=1))
        if data.empty:
            return None
        
        weekly_prices = []
        for start, end in week_ranges:
            week_data = data.loc[start:end]
            if not week_data.empty:
                # Get the last available closing price in the week
                closing_price = week_data['Close'][-1]
                weekly_prices.append(closing_price)
            else:
                weekly_prices.append(None)
        
        return weekly_prices
    except Exception as e:
        st.error(f"Error fetching data for {full_symbol}: {str(e)}")
        return None

# Streamlit app
def main():
    st.title("Weekly Price Tracker")
    st.write("Upload an Excel file with 'Symbol' and 'Exchange' columns to track weekly price changes.")
    
    # File upload
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Read the Excel file
            df = pd.read_excel(uploaded_file)
            
            # Validate columns
            if not all(col in df.columns for col in ['Symbol', 'Exchange']):
                st.error("Excel file must contain 'Symbol' and 'Exchange' columns")
                return
            
            # Get the week ranges
            week_ranges = get_week_dates(6)
            
            # Display week headers
            week_headers = []
            for i, (start, end) in enumerate(week_ranges):
                if i == len(week_ranges) - 1:
                    week_headers.append(f"Current Week\n{start.strftime('%m/%d')}-{end.strftime('%m/%d')}")
                else:
                    week_headers.append(f"Week -{len(week_ranges)-1-i}\n{start.strftime('%m/%d')}-{end.strftime('%m/%d')}")
            
            # Process each symbol
            results = []
            for _, row in df.iterrows():
                symbol = row['Symbol']
                exchange = row['Exchange']
                
                prices = get_weekly_prices(symbol, exchange, week_ranges)
                if prices is not None:
                    # Calculate percentage changes
                    changes = []
                    for i in range(1, len(prices)):
                        if prices[i-1] is not None and prices[i] is not None and prices[i-1] != 0:
                            change = (prices[i] - prices[i-1]) / prices[i-1] * 100
                            changes.append(f"{change:.2f}%")
                        else:
                            changes.append("N/A")
                    
                    # Add to results
                    result_row = {
                        'Symbol': symbol,
                        'Exchange': exchange,
                        **{f'Price {i+1}': f"${prices[i]:.2f}" if prices[i] is not None else "N/A" for i in range(len(prices))},
                        **{f'Change {i+1}': changes[i-1] if i > 0 else "N/A" for i in range(len(prices)))}
                    results.append(result_row)
            
            if results:
                # Create results DataFrame
                results_df = pd.DataFrame(results)
                
                # Reorder columns to show price and change side by side
                ordered_columns = ['Symbol', 'Exchange']
                for i in range(len(week_ranges)):
                    ordered_columns.extend([
                        f'Price {i+1}',
                        f'Change {i+1}' if i > 0 else 'Change 1'
                    ])
                
                results_df = results_df[ordered_columns]
                
                # Rename columns for display
                display_columns = ['Symbol', 'Exchange']
                for i in range(len(week_ranges)):
                    display_columns.extend([
                        week_headers[i],
                        f'Change vs Prev' if i > 0 else 'Change'
                    ])
                
                # Display the table
                st.dataframe(results_df.style.format(None, subset=['Symbol', 'Exchange']), 
                             column_config={col: display_columns[i] for i, col in enumerate(results_df.columns)})
                
                # Add download button
                st.download_button(
                    label="Download Results as Excel",
                    data=results_df.to_excel(index=False),
                    file_name="weekly_price_changes.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("No valid data could be retrieved for the provided symbols.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
