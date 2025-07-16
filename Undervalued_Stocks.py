# Updated full script with manually calculated Debt/Equity and fallback PEG logic
# Also handles cases where yfinance lacks data and avoids showing "None"

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("📊 Undervalued Stock Screener")
st.subheader("Warren Buffett-inspired Value Investing Strategy")

st.markdown("""
This app helps identify potentially undervalued stocks based on fundamental analysis metrics. 
You can analyze either a single stock or upload a file with multiple tickers.
""")

# Sidebar controls
st.sidebar.header("Adjust valuation thresholds")
max_pe = st.sidebar.slider("Max P/E Ratio", 5, 50, 25)
max_peg = st.sidebar.slider("Max PEG Ratio", 0.1, 3.0, 1.5)
max_de = st.sidebar.slider("Max Debt-to-Equity", 0.1, 5.0, 2.0)
min_cr = st.sidebar.slider("Min Current Ratio", 0.5, 3.0, 1.0)
min_roa = st.sidebar.slider("Min Return on Assets (%)", 0, 20, 5)

# Ticker input
analysis_type = st.radio("Select analysis type:", ["Single Stock Analysis", "Multiple Stocks Analysis"])
ticker_list = []

if analysis_type == "Single Stock Analysis":
    single_ticker = st.text_input("Enter stock ticker:", value="aapl")
    exchange = st.selectbox("Select exchange (if international):", ["", "TO", "AX", "L", "HK", "NS", "SS", "SZ", "TWO", "SA"])
    if st.button("Analyze Single Stock"):
        symbol = f"{single_ticker.strip()}.{exchange}" if exchange else single_ticker.strip()
        ticker_list.append(symbol)
elif analysis_type == "Multiple Stocks Analysis":
    uploaded_file = st.file_uploader("Upload a CSV or TXT file with tickers", type=["csv", "txt"])
    if uploaded_file:
        if uploaded_file.name.endswith("csv"):
            df_uploaded = pd.read_csv(uploaded_file)
        else:
            df_uploaded = pd.read_csv(uploaded_file, header=None)
        ticker_list = df_uploaded.iloc[:, 0].dropna().unique().tolist()

# Main stock analyzer
def analyze_stocks(tickers):
    raw_data = []
    for i, ticker in enumerate(tickers):
        st.info(f"Processing {ticker} ({i+1}/{len(tickers)})...")
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            bs = stock.balance_sheet
            current_price = info.get("currentPrice", np.nan)
            pe = info.get("trailingPE", np.nan)
            fpe = info.get("forwardPE", np.nan)
            pb = info.get("priceToBook", np.nan)
            div_yield = info.get("dividendYield", np.nan)
            peg = info.get("pegRatio", np.nan)

            # PEG fallback: P/E divided by 5Y growth estimate
            if np.isnan(peg) and not np.isnan(pe):
                growth = info.get("earningsQuarterlyGrowth") or info.get("earningsGrowth")
                if growth and growth > 0:
                    peg = pe / (growth * 100)

            # Manually calculate Debt/Equity
            try:
                total_liab = bs.loc["Total Liab"][0]
                total_equity = bs.loc["Total Stockholder Equity"][0]
                calc_de = total_liab / total_equity if total_equity != 0 else np.nan
            except Exception:
                calc_de = np.nan

            # Other financials
            current_ratio = info.get("currentRatio", np.nan)
            roa = info.get("returnOnAssets", np.nan)
            roe = info.get("returnOnEquity", np.nan)
            profit_margin = info.get("profitMargins", np.nan)
            fifty_two_wk_low = info.get("fiftyTwoWeekLow", np.nan)
            fifty_two_wk_high = info.get("fiftyTwoWeekHigh", np.nan)
            discount = 100 * (fifty_two_wk_high - current_price) / fifty_two_wk_high if fifty_two_wk_high else np.nan
            beta = info.get("beta", np.nan)
            market_cap = info.get("marketCap", np.nan)

            raw_data.append({
                "Symbol": ticker,
                "Exchange": info.get("exchange", ""),
                "CurrentPrice": current_price,
                "P/E": pe,
                "Forward P/E": fpe,
                "PEG": peg,
                "Debt/Equity": round(calc_de, 3) if pd.notnull(calc_de) else np.nan,
                "CurrentRatio": current_ratio,
                "ROA": roa * 100 if pd.notnull(roa) else np.nan,
                "ROE": roe * 100 if pd.notnull(roe) else np.nan,
                "ProfitMargin": profit_margin * 100 if pd.notnull(profit_margin) else np.nan,
                "Price/Book": pb,
                "DividendYield": div_yield * 100 if pd.notnull(div_yield) else np.nan,
                "52WeekLow": fifty_two_wk_low,
                "52WeekHigh": fifty_two_wk_high,
                "DiscountFromHigh": round(discount, 4) if pd.notnull(discount) else np.nan,
                "Beta": beta,
                "MarketCap": market_cap
            })
        except Exception as e:
            st.warning(f"Error processing {ticker}: {e}")
    df = pd.DataFrame(raw_data)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

# Run analysis
if ticker_list:
    df = analyze_stocks(ticker_list)
    if not df.empty:
        st.subheader("Raw Stock Data from Yahoo Finance")
        st.dataframe(df, use_container_width=True)

        filtered_df = df[
            (df["P/E"] < max_pe) &
            (df["PEG"] < max_peg) &
            (df["Debt/Equity"] < max_de) &
            (df["CurrentRatio"] > min_cr) &
            (df["ROA"] > min_roa)
        ]
        st.subheader("📈 Filtered Potentially Undervalued Stocks")
        st.dataframe(filtered_df, use_container_width=True)
    else:
        st.error("No data to display. Check your tickers or try again.")
