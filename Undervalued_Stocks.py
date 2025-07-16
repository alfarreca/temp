import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px

# App configuration
st.set_page_config(layout="wide")
st.title("📊 Undervalued Stock Screener")
st.subheader("Warren Buffett-inspired Value Investing Strategy")
st.write("""
This app helps identify potentially undervalued stocks based on fundamental analysis metrics.
You can analyze either a single stock or upload a file with multiple tickers.
""")

# Valuation parameters
st.sidebar.header("Valuation Parameters")
with st.sidebar.expander("Adjust valuation thresholds"):
    pe_ratio_threshold = st.slider("Max P/E Ratio", 5, 50, 25)
    peg_ratio_threshold = st.slider("Max PEG Ratio", 0.1, 3.0, 1.5, step=0.1, format="%.1f")
    debt_to_equity_threshold = st.slider("Max Debt-to-Equity", 0.1, 5.0, 2.0, step=0.1, format="%.1f")
    current_ratio_threshold = st.slider("Min Current Ratio", 0.5, 3.0, 1.0, step=0.1, format="%.1f")
    roa_threshold = st.slider("Min Return on Assets (%)", 0, 20, 5)

def get_stock_data(ticker, exchange=""):
    try:
        exchange_map = {
            "TORONTO": ".TO", "LONDON": ".L", "EURONEXT": ".PA", "FRANKFURT": ".DE",
            "HONG KONG": ".HK", "SHANGHAI": ".SS"
        }
        full_ticker = ticker + exchange_map.get(exchange.upper(), "")
        stock = yf.Ticker(full_ticker)
        hist = stock.history(period="1d")

        if hist.empty:
            return None

        current_price = hist['Close'].iloc[-1]
        info = stock.info

        # Manual P/E fallback
        eps = info.get('trailingEps', np.nan)
        pe_ratio = info.get('trailingPE', np.nan)
        if (not pe_ratio or np.isnan(pe_ratio)) and eps and eps != 0:
            pe_ratio = current_price / eps
        if pe_ratio is not None and pe_ratio < 0:
            pe_ratio = None

        # Manual Forward P/E
        forward_eps = info.get('forwardEps', np.nan)
        forward_pe = info.get('forwardPE', np.nan)
        if (not forward_pe or np.isnan(forward_pe)) and forward_eps and forward_eps != 0:
            forward_pe = current_price / forward_eps
        if forward_pe is not None and forward_pe < 0:
            forward_pe = None

        # Manually calculate PEG ratio
        growth = info.get('earningsGrowth', np.nan)
        peg_ratio = None
        if pe_ratio is not None and pe_ratio > 0 and growth is not None and growth > 0:
            peg_ratio = pe_ratio / (growth * 100)

        # Manually calculate Dividend Yield
        dividend = info.get('dividendRate', np.nan)
        div_yield = None
        if dividend and dividend > 0 and current_price > 0:
            div_yield = (dividend / current_price) * 100

        # Manually calculate Debt/Equity if missing
        de_ratio = info.get("debtToEquity")
        if de_ratio is None:
            try:
                bs = stock.balance_sheet
                total_liab = bs.loc["Total Liab"][0]
                equity = bs.loc["Total Stockholder Equity"][0]
                if equity != 0:
                    de_ratio = total_liab / equity
                else:
                    de_ratio = None
            except Exception:
                de_ratio = None

        metrics = {
            'Symbol': ticker,
            'Exchange': exchange,
            'CurrentPrice': current_price,
            'P/E': pe_ratio,
            'Forward P/E': forward_pe,
            'PEG': peg_ratio,
            'Debt/Equity': round(de_ratio, 3) if de_ratio is not None else None,
            'CurrentRatio': info.get('currentRatio', np.nan),
            'ROA': info.get('returnOnAssets', np.nan) * 100 if info.get('returnOnAssets') else np.nan,
            'ROE': info.get('returnOnEquity', np.nan) * 100 if info.get('returnOnEquity') else np.nan,
            'ProfitMargin': info.get('profitMargins', np.nan) * 100 if info.get('profitMargins') else np.nan,
            'Price/Book': current_price / info.get('bookValue', np.nan) if info.get('bookValue') else np.nan,
            'DividendYield': div_yield,
            '52WeekLow': info.get('fiftyTwoWeekLow', np.nan),
            '52WeekHigh': info.get('fiftyTwoWeekHigh', np.nan),
            'DiscountFromHigh': (info.get('fiftyTwoWeekHigh', np.nan) - current_price) / info.get('fiftyTwoWeekHigh', 1) * 100 
                                if info.get('fiftyTwoWeekHigh') else np.nan,
            'Beta': info.get('beta', np.nan),
            'MarketCap': info.get('marketCap', np.nan)
        }
        return metrics

    except Exception as e:
        st.warning(f"Error fetching data for {ticker}: {str(e)}")
        return None
