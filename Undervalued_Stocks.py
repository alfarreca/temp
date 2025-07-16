import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="Undervalued Stock Screener", layout="wide")

st.title("📊 Undervalued Stock Screener")
st.markdown("""
**Warren Buffett-inspired Value Investing Strategy**

This app helps identify potentially undervalued stocks based on fundamental analysis metrics. You can analyze either a single stock or upload a file with multiple tickers.
""")

# ------------------------------
# Core Functions
# ------------------------------

def get_stock_data(ticker, exchange=""):
    try:
        if exchange:
            full_ticker = f"{ticker}.{exchange}"
        else:
            full_ticker = ticker

        stock = yf.Ticker(full_ticker)
        info = stock.info

        # Manual calculations
        current_price = info.get("currentPrice")
        forward_pe = info.get("forwardPE")
        earnings_growth = info.get("earningsQuarterlyGrowth")
        dividend_yield = (info.get("dividendRate") or 0) / current_price * 100 if current_price else None

        # Calculate P/E if not present
        trailing_eps = info.get("trailingEps")
        pe_ratio = current_price / trailing_eps if trailing_eps else None

        # Calculate PEG if forward PE and growth available
        if forward_pe and earnings_growth:
            try:
                peg_ratio = forward_pe / (earnings_growth * 100)
            except ZeroDivisionError:
                peg_ratio = None
        else:
            peg_ratio = None

        # Calculate Debt/Equity manually if needed
        balance_sheet = stock.balance_sheet
        if not balance_sheet.empty:
            try:
                total_liabilities = balance_sheet.loc["Total Liab"].iloc[0]
                total_equity = balance_sheet.loc["Total Stockholder Equity"].iloc[0]
                debt_to_equity = total_liabilities / total_equity if total_equity else None
            except:
                debt_to_equity = info.get("debtToEquity")
        else:
            debt_to_equity = info.get("debtToEquity")

        return {
            "Symbol": ticker,
            "Exchange": exchange,
            "CurrentPrice": current_price,
            "P/E": pe_ratio,
            "Forward P/E": forward_pe,
            "PEG": peg_ratio,
            "Debt/Equity": debt_to_equity,
            "CurrentRatio": info.get("currentRatio"),
            "ROA": info.get("returnOnAssets", 0) * 100 if info.get("returnOnAssets") else None,
            "ROE": info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else None,
            "ProfitMargin": info.get("profitMargins", 0) * 100 if info.get("profitMargins") else None,
            "Price/Book": info.get("priceToBook"),
            "DividendYield": dividend_yield,
            "52WeekLow": info.get("fiftyTwoWeekLow"),
            "52WeekHigh": info.get("fiftyTwoWeekHigh"),
            "DiscountFromHigh": (1 - current_price / info.get("fiftyTwoWeekHigh")) * 100 if current_price and info.get("fiftyTwoWeekHigh") else None,
            "Beta": info.get("beta"),
            "MarketCap": info.get("marketCap"),
        }
    except Exception as e:
        st.warning(f"⚠️ Failed to fetch data for {ticker}: {e}")
        return {}

def analyze_stocks(df):
    results = []
    for _, row in df.iterrows():
        data = get_stock_data(row['Symbol'], row.get("Exchange", ""))
        if data:
            results.append(data)
    results_df = pd.DataFrame(results)

    # Filter based on sliders
    filtered_df = results_df[
        (results_df["P/E"] <= st.session_state.max_pe) &
        (results_df["PEG"] <= st.session_state.max_peg) &
        (results_df["Debt/Equity"] <= st.session_state.max_de) &
        (results_df["CurrentRatio"] >= st.session_state.min_cr) &
        (results_df["ROA"] >= st.session_state.min_roa)
    ]

    return filtered_df, results_df

def display_results(df):
    if df.empty:
        st.warning("No data matched your filters or ticker was invalid.")
    else:
        st.subheader("Raw Stock Data from Yahoo Finance")
        st.dataframe(df)
        st.success(f"✅ Finished analyzing {len(df)} stock(s).")

# ------------------------------
# UI Controls
# ------------------------------

st.sidebar.header("Analysis Options")
analysis_type = st.sidebar.radio("Select analysis type:", ["Single Stock Analysis", "Multiple Stocks Analysis"])

if "max_pe" not in st.session_state:
    st.session_state.max_pe = 25
    st.session_state.max_peg = 1.5
    st.session_state.max_de = 2.0
    st.session_state.min_cr = 1.0
    st.session_state.min_roa = 5

with st.sidebar.expander("Adjust valuation thresholds"):
    st.session_state.max_pe = st.slider("Max P/E Ratio", 5, 50, st.session_state.max_pe)
    st.session_state.max_peg = st.slider("Max PEG Ratio", 0.1, 3.0, st.session_state.max_peg)
    st.session_state.max_de = st.slider("Max Debt-to-Equity", 0.1, 5.0, st.session_state.max_de)
    st.session_state.min_cr = st.slider("Min Current Ratio", 0.5, 3.0, st.session_state.min_cr)
    st.session_state.min_roa = st.slider("Min Return on Assets (%)", 0, 20, st.session_state.min_roa)

# ------------------------------
# Analysis Section
# ------------------------------

if analysis_type == "Single Stock Analysis":
    st.sidebar.subheader("Single Stock Parameters")
    symbol = st.sidebar.text_input("Enter stock ticker:")
    exchange = st.sidebar.selectbox("Select exchange (if international):", ["", "TO", "L", "AX", "HK"], index=0)

    if st.sidebar.button("Analyze Single Stock"):
        st.info(f"Processing {symbol} (1/1)...")
        df = pd.DataFrame([{"Symbol": symbol, "Exchange": exchange}])
        _, results_df = analyze_stocks(df)
        display_results(results_df)

elif analysis_type == "Multiple Stocks Analysis":
    st.sidebar.subheader("Upload Stock List")
    uploaded_file = st.sidebar.file_uploader("Upload CSV with 'Symbol' column", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.info(f"Processing {len(df)} tickers...")
        filtered_df, results_df = analyze_stocks(df)

        display_results(results_df)

        st.download_button("📥 Download Raw Data", results_df.to_csv(index=False), file_name="stock_data.csv")
        st.download_button("📉 Download Filtered Results", filtered_df.to_csv(index=False), file_name="filtered_data.csv")
