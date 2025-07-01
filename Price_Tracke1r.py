import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

st.title('📈 Ticker Price Change Tracker')
st.markdown("Track weekly price changes for your stock tickers (upload an Excel file with a column named 'Symbol')")

def get_week_dates(weeks_back):
    today = datetime.now()
    last_friday = today - timedelta(days=(today.weekday() + 3) % 7)
    end_date = last_friday - timedelta(weeks=weeks_back-1)
    start_date = end_date - timedelta(days=4)
    return start_date, end_date

def download_ticker(ticker, start, end, max_retries=3):
    for attempt in range(max_retries):
        try:
            data = yf.download(
                ticker,
                start=start,
                end=end + timedelta(days=1),
                progress=False
            )
            if not data.empty:
                return data
        except Exception:
            continue
    return None

uploaded_file = st.file_uploader("Upload Excel file with tickers", type=['xlsx'])

if uploaded_file is not None:
    try:
        df_tickers = pd.read_excel(uploaded_file)
        st.write(f"File columns detected: {list(df_tickers.columns)}")
        columns = [str(c).strip() for c in df_tickers.columns]
        if 'Symbol' not in columns:
            st.error("The Excel file must contain a column named exactly 'Symbol' (case sensitive).")
            st.stop()

        # Get the actual column name in case of whitespace
        symbol_col = [c for c in df_tickers.columns if c.strip() == 'Symbol'][0]
        tickers = df_tickers[symbol_col].dropna().astype(str).str.upper().unique()

        if len(tickers) > 0:
            st.success(f"Found {len(tickers)} tickers in the uploaded file")
            with st.expander("⚙️ Advanced Options"):
                add_suffix = st.checkbox("Try adding common exchange suffixes", True)
                suffixes = st.multiselect(
                    "Select suffixes to try",
                    ['.TO', '.NS', '.L', '.DE', '.PA', '.AS', '.BR', '.AX', '.SI', '.KS'],
                    ['.TO', '.NS', '.L']
                )

            if st.button('🚀 Fetch Price Changes', type='primary'):
                progress_bar = st.progress(0)
                status_text = st.empty()

                data_dict = {}
                date_dict = {}
                total_operations = len(tickers) * 6
                completed_operations = 0
                debug_info = []

                for week in range(1, 7):
                    start_date, end_date = get_week_dates(week)
                    week_label = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                    date_dict[f'Week {week}'] = week_label
                    for idx, ticker in enumerate(tickers):
                        completed_operations += 1
                        progress = int(100 * completed_operations / total_operations)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {ticker} - Week {week}...")

                        ticker_variations = [ticker]
                        if add_suffix:
                            ticker_variations.extend([f"{ticker}{suffix}" for suffix in suffixes])
                        found = False
                        for t in ticker_variations:
                            data = download_ticker(t, start_date, end_date)
                            if data is not None and not data.empty:
                                start_price = data['Close'].iloc[0]
                                end_price = data['Close'].iloc[-1]
                                pct_change = (end_price - start_price) / start_price * 100
                                data_dict.setdefault(ticker, {})[f'Week {week}'] = pct_change
                                found = True
                                debug_info.append({'ticker': ticker, 'variation': t, 'week': week, 'status': 'success'})
                                break
                            else:
                                debug_info.append({'ticker': ticker, 'variation': t, 'week': week, 'status': 'failed'})
                        if not found:
                            data_dict.setdefault(ticker, {})[f'Week {week}'] = None

                progress_bar.empty()
                status_text.empty()

                # Build display DataFrame
                results = pd.DataFrame(data_dict).T
                results = results[[f'Week {i}' for i in range(1, 7)]]

                # Formatting for display
                def format_percent(x):
                    return f"{x:.2f}%" if pd.notnull(x) else "N/A"

                formatted_df = results.applymap(format_percent)
                st.subheader('📅 Weekly Price Changes (%)')
                st.dataframe(formatted_df, use_container_width=True)

                # Show week date ranges
                st.caption("📆 Trading weeks (Monday to Friday):")
                for i in range(1, 7):
                    st.caption(f"Week {i}: {date_dict[f'Week {i}']}")

                # Show failed tickers
                debug_df = pd.DataFrame(debug_info)
                failed_tickers = debug_df[(debug_df['status'] == 'failed') & 
                                       (debug_df['week'] == 1)]['ticker'].unique()
                if len(failed_tickers) > 0:
                    st.warning(f"⚠️ Could not fetch data for: {', '.join(failed_tickers)}")
                    with st.expander("Debug details"):
                        st.dataframe(debug_df)

                # Download options
                output = results.copy()
                output.index.name = 'Ticker'
                st.download_button(
                    "💾 Download Results as CSV",
                    output.to_csv().encode('utf-8'),
                    "ticker_price_changes.csv",
                    "text/csv"
                )

        else:
            st.error("No tickers found in the uploaded file.")

    except Exception as e:
        st.error(f"Error reading the Excel file: {str(e)}")
else:
    st.info("ℹ️ Please upload an Excel file with at least a 'Symbol' column")
