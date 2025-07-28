import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime

# App configuration
st.set_page_config(
    page_title="Stock Screener Pro",
    page_icon="📈",
    layout="wide"
)

@st.cache_data
def load_data(uploaded_file):
    """Load and preprocess the Excel file with error handling"""
    try:
        df = pd.read_excel(uploaded_file)
        required_cols = ['Symbol', 'Exchange', 'Sector', 'Industry',
                         'Theme', 'Name', 'Country', 'Notes', 'Asset_Type']

        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            return None

        # Handle missing values
        for col in ['Asset_Type', 'Sector', 'Country']:
            df[col] = df[col].fillna('Unknown')

        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def fetch_financials(ticker):
    """Fetch financial metrics with robust error handling"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        cashflow = stock.cashflow

        # Get dividend data with fallbacks
        dividend_yield = info.get('dividendYield', 0) * 100  # Convert to percentage
        payout_ratio = info.get('payoutRatio')

        # Get free cash flow with error handling
        try:
            fcf = cashflow.iloc[0].get('Free Cash Flow') if cashflow is not None else None
        except:
            fcf = None

        return {
            # Value metrics
            'P/E': info.get('trailingPE'),
            'P/B': info.get('priceToBook'),
            'ROE': info.get('returnOnEquity'),
            'Market Cap': info.get('marketCap'),

            # Dividend metrics
            'Dividend Yield': dividend_yield,
            'Payout Ratio': payout_ratio,
            'Free Cash Flow': fcf,
            'Last Updated': datetime.now().strftime("%Y-%m-%d %H:%M")
        }
    except Exception as e:
        st.warning(f"Couldn't fetch data for {ticker}: {str(e)}")
        return None

def main():
    st.title("📈 Stock Screener Pro")

    # Mode selection
    mode = st.sidebar.radio(
        "Screener Mode",
        ["Value Screener", "Dividend Screener"],
        index=0,
        help="Switch between value stocks and dividend stocks screening"
    )

    # File upload section
    st.sidebar.header("Data Input")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Stock Universe Excel File",
        type=['xlsx', 'xls'],
        help="File must contain: Symbol, Exchange, Sector, Industry, Name, Country"
    )

    if not uploaded_file:
        st.info("ℹ️ Please upload an Excel file to begin screening")
        st.markdown("""
        ### Expected File Format:
        The Excel file should contain these columns:
        - Symbol (e.g., AAPL)
        - Exchange (e.g., NASDAQ)
        - Sector (e.g., Technology)
        - Industry (e.g., Consumer Electronics)
        - Name (e.g., Apple Inc.)
        - Country (e.g., United States)
        """)
        return

    df = load_data(uploaded_file)
    if df is None:
        return

    # Common filters
    with st.sidebar:
        st.header("Basic Filters")
        selected_sector = st.selectbox(
            "Sector",
            ['All'] + sorted(df['Sector'].unique().tolist())
        )
        selected_country = st.selectbox(
            "Country",
            ['All'] + sorted(df['Country'].unique().tolist())
        )

        if mode == "Value Screener":
            st.header("Value Parameters")
            pb_max = st.slider(
                "Max Price-to-Book",
                0.0, 5.0, 1.2, 0.1,
                help="Lower values indicate more undervalued stocks"
            )
            roe_min = st.slider(
                "Min Return on Equity (%)",
                0, 50, 15,
                help="Higher values indicate more profitable companies"
            )
        else:
            st.header("Dividend Parameters")
            div_min = st.slider(
                "Min Dividend Yield (%)",
                0.0, 10.0, 3.0, 0.1,
                help="Higher values screen for higher-yielding stocks"
            )
            payout_max = st.slider(
                "Max Payout Ratio (%)",
                0, 100, 70,
                help="Lower values indicate more sustainable dividends"
            )
            fcf_positive = st.checkbox(
                "Require Positive Free Cash Flow",
                True,
                help="Ensures dividend is supported by cash generation"
            )

    # Apply basic filters
    filtered = df.copy()
    if selected_sector != 'All':
        filtered = filtered[filtered['Sector'] == selected_sector]
    if selected_country != 'All':
        filtered = filtered[filtered['Country'] == selected_country]

    # Fetch financial data with progress
    st.subheader("🔍 Screening Results")
    with st.spinner(f"Analyzing {len(filtered)} stocks..."):
        progress_bar = st.progress(0)
        results = []

        for i, symbol in enumerate(filtered['Symbol'].unique()):
            data = fetch_financials(symbol)
            if data:
                data['Symbol'] = symbol
                results.append(data)
            progress_bar.progress((i + 1) / len(filtered['Symbol'].unique()))

        if not results:
            st.error("No financial data could be retrieved. Please check your symbols and try again later.")
            return

        financials = pd.DataFrame(results)
        merged = pd.merge(filtered, financials, on='Symbol')

        if mode == "Value Screener":
            # Apply value filters
            screened = merged[
                (merged['P/B'].notna()) &
                (merged['P/B'] <= pb_max) &
                (merged['ROE'].notna()) &
                (merged['ROE'] >= roe_min / 100)
            ].sort_values(['P/B', 'ROE'], ascending=[True, False])

            # Display results
            if not screened.empty:
                display_cols = [
                    'Symbol', 'Name', 'Sector', 'Country',
                    'P/B', 'ROE', 'Dividend Yield', 'Market Cap'
                ]
                display_df = screened[display_cols].copy()

                # Formatting
                display_df['ROE'] = (display_df['ROE'] * 100).round(1).astype(str) + '%'
                display_df['Dividend Yield'] = display_df['Dividend Yield'].round(2).astype(str) + '%'
                display_df['P/B'] = display_df['P/B'].round(2)
                display_df['Market Cap'] = display_df['Market Cap'].apply(
                    lambda x: f"${x/1e9:.1f}B" if pd.notnull(x) else 'N/A'
                )

                st.dataframe(
                    display_df.style.apply(
                        lambda x: ['background: #e6ffe6' if x['P/B'] <= 1.0 else
                                   'background: #ffffe6' if x['P/B'] <= 1.2 else ''
                                   for _ in x],
                        axis=1
                    ),
                    hide_index=True,
                    use_container_width=True,
                    height=min(600, 35 * (len(display_df) + 1))
                )
            else:
                st.warning("No value stocks found matching your criteria")
        else:
            # Apply dividend filters
            dividend_filter = (
                (merged['Dividend Yield'].notna()) &
                (merged['Dividend Yield'] >= div_min) &
                (merged['Payout Ratio'].notna()) &
                (merged['Payout Ratio'] <= payout_max / 100)
            )

            if fcf_positive:
                dividend_filter &= (merged['Free Cash Flow'].notna()) & (merged['Free Cash Flow'] > 0)

            screened = merged[dividend_filter].sort_values(
                ['Dividend Yield', 'Payout Ratio'],
                ascending=[False, True]
            )

            # Display results
            if not screened.empty:
                display_cols = [
                    'Symbol', 'Name', 'Sector', 'Country',
                    'Dividend Yield', 'Payout Ratio', 'Free Cash Flow', 'Market Cap'
                ]
                display_df = screened[display_cols].copy()

                # Formatting
                display_df['Dividend Yield'] = display_df['Dividend Yield'].round(2).astype(str) + '%'
                display_df['Payout Ratio'] = (display_df['Payout Ratio'] * 100).round(1).astype(str) + '%'
                display_df['Free Cash Flow'] = display_df['Free Cash Flow'].apply(
                    lambda x: f"${x/1e6:.1f}M" if pd.notnull(x) else 'N/A'
                )
                display_df['Market Cap'] = display_df['Market Cap'].apply(
                    lambda x: f"${x/1e9:.1f}B" if pd.notnull(x) else 'N/A'
                )

                st.dataframe(
                    display_df.style.apply(
                        lambda x: ['background: #e6f3ff' if float(x['Dividend Yield'].replace('%', '')) >= 5 else
                                   'background: #f0f7ff' if float(x['Dividend Yield'].replace('%', '')) >= 3 else ''
                                   for _ in x],
                        axis=1
                    ),
                    hide_index=True,
                    use_container_width=True,
                    height=min(600, 35 * (len(display_df) + 1))
                )
            else:
                st.warning("No dividend stocks found matching your criteria")

        # Download button
        if not screened.empty:
            csv = screened.to_csv(index=False)
            st.download_button(
                "💾 Download Full Results",
                csv,
                f"{mode.lower().replace(' ', '_')}_stocks.csv",
                "text/csv",
                help="Download complete screening results"
            )

if __name__ == "__main__":
    main()
