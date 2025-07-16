import streamlit as st
import pandas as pd
import yfinance as yf
import datetime

st.set_page_config(page_title="Undervalued Stock Screener", layout="wide")
st.title("📉 Undervalued Stock Screener")
st.subheader("Warren Buffett-inspired Value Investing Strategy")
st.write("This app helps identify potentially undervalued stocks based on fundamental analysis metrics. You can analyze either a single stock or upload a file with multiple tickers.")

analysis_type = st.sidebar.radio("Select analysis type:", ["Single Stock Analysis", "Multiple Stocks Analysis"])

max_pe = st.sidebar.slider("Max P/E Ratio", 5, 50, 25)
max_peg = st.sidebar.slider("Max PEG Ratio", 0.1, 3.0, 1.5)
max_debt_equity = st.sidebar.slider("Max Debt-to-Equity", 0.1, 5.0, 2.0)
min_current_ratio = st.sidebar.slider("Min Current Ratio", 0.5, 3.0, 1.0)
min_roa = st.sidebar.slider("Min Return on Assets (%)", 0, 20, 5)

def fetch_stock_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info

    try:
        current_price = info.get("currentPrice")
        pe_ratio = info.get("trailingPE")
        forward_pe = info.get("forwardPE")
        peg_ratio = info.get("pegRatio")
        roa = info.get("returnOnAssets", 0) * 100 if info.get("returnOnAssets") is not None else None
        roe = info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") is not None else None
        profit_margin = info.get("profitMargins", 0) * 100 if info.get("profitMargins") is not None else None
        pb_ratio = info.get("priceToBook")
        div_yield = info.get("dividendYield", 0) * 100 if info.get("dividendYield") is not None else None
        current_ratio = info.get("currentRatio")

        # Calculate D/E manually from balance sheet
        try:
            balance = ticker.balance_sheet
            liabilities = balance.loc["Total Liab"][0]
            equity = balance.loc["Total Stockholder Equity"][0]
            if equity and abs(equity) > 1e-6:
                de_ratio = liabilities / equity
                print(f"Debt/Equity (Calculated): {de_ratio}")
            else:
                de_ratio = None
        except Exception as e:
            print(f"Error calculating D/E manually: {e}")
            de_ratio = None

        fifty_two_week_high = info.get("fiftyTwoWeekHigh")
        fifty_two_week_low = info.get("fiftyTwoWeekLow")
        discount_from_high = (1 - current_price / fifty_two_week_high) * 100 if fifty_two_week_high else None
        beta = info.get("beta")
        market_cap = info.get("marketCap")

        return {
            "Symbol": ticker_symbol,
            "Exchange": info.get("exchange"),
            "CurrentPrice": current_price,
            "P/E": pe_ratio,
            "Forward P/E": forward_pe,
            "PEG": peg_ratio if forward_pe and forward_pe > 0 else None,
            "Debt/Equity": de_ratio,
            "CurrentRatio": current_ratio,
            "ROA": roa,
            "ROE": roe,
            "ProfitMargin": profit_margin,
            "Price/Book": pb_ratio,
            "DividendYield": div_yield,
            "52WeekLow": fifty_two_week_low,
            "52WeekHigh": fifty_two_week_high,
            "DiscountFromHigh": discount_from_high,
            "Beta": beta,
            "MarketCap": market_cap
        }

    except Exception as e:
        st.error(f"Error fetching data for {ticker_symbol}: {e}")
        return None

def analyze_stocks(tickers):
    results = []
    for i, symbol in enumerate(tickers):
        st.write(f"Processing {symbol} ({i+1}/{len(tickers)})...")
        data = fetch_stock_data(symbol)
        if data:
            results.append(data)

    df = pd.DataFrame(results)
    filtered_df = df[
        (df["P/E"].fillna(9999) < max_pe) &
        (df["PEG"].fillna(9999) < max_peg) &
        (df["Debt/Equity"].fillna(9999) < max_debt_equity) &
        (df["CurrentRatio"].fillna(0) > min_current_ratio) &
        (df["ROA"].fillna(0) > min_roa)
    ]

    return filtered_df, df

if analysis_type == "Single Stock Analysis":
    single_symbol = st.text_input("Enter stock ticker:", value="el")
    exchange = st.selectbox("Select exchange (if international):", ["", "LON", "FRA", "AS", "TO", "AX", "HK", "TWO"])

    if st.button("Analyze Single Stock"):
        if exchange:
            single_symbol = f"{single_symbol}.{exchange}"
        filtered_df, results_df = analyze_stocks([single_symbol])

        if results_df.empty:
            st.warning("No data matched your filters or ticker was invalid.")
        else:
            st.subheader("Raw Stock Data from Yahoo Finance")
            st.dataframe(results_df)
            st.success(f"✅ Finished analyzing {len(results_df)} stock(s).")

elif analysis_type == "Multiple Stocks Analysis":
    uploaded_file = st.file_uploader("Upload CSV with tickers", type="csv")
    if uploaded_file is not None:
        tickers_df = pd.read_csv(uploaded_file)
        if "Symbol" not in tickers_df.columns:
            st.error("CSV must contain a 'Symbol' column.")
        else:
            tickers = tickers_df["Symbol"].dropna().tolist()
            filtered_df, results_df = analyze_stocks(tickers)

            if results_df.empty:
                st.warning("No data matched your filters.")
            else:
                st.subheader("Raw Stock Data from Yahoo Finance")
                st.dataframe(results_df)
                st.success(f"✅ Finished analyzing {len(results_df)} stock(s).")
