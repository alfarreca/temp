import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objs as go

st.title("📈 Weekly Price Tracker (Fri–Fri Weeks + Current Week + Scoring)")

uploaded_file = st.file_uploader("Upload your Excel file", type="xlsx")

def get_last_n_weeks(n):
    today = datetime.today()
    offset = (today.weekday() - 4) % 7  # 4 = Friday
    last_friday = today - timedelta(days=offset)
    weeks = []
    for i in reversed(range(n)):
        this_friday = last_friday - timedelta(weeks=i)
        this_monday = this_friday - timedelta(days=4)
        weeks.append((this_monday, this_friday))
    return weeks, last_friday

def fetch_friday_closes(symbol, weeks):
    first_monday = weeks[0][0]
    last_friday = weeks[-1][1]
    data = yf.download(symbol, start=first_monday, end=last_friday + timedelta(days=1), interval="1d", progress=False)
    if data.empty or "Close" not in data.columns:
        return None
    closes = []
    for monday, friday in weeks:
        week_data = data.loc[(data.index >= pd.Timestamp(monday)) & (data.index <= pd.Timestamp(friday))]
        friday_close = week_data.loc[week_data.index.weekday == 4, "Close"]
        if not friday_close.empty:
            closes.append(float(round(friday_close.iloc[-1], 3)))
        elif not week_data.empty:
            closes.append(float(round(week_data["Close"].iloc[-1], 3)))
        else:
            closes.append(np.nan)
    return closes if sum(np.isnan(closes)) == 0 else None

def fetch_current_week_close(symbol, current_week_start):
    today = datetime.today()
    data = yf.download(symbol, start=current_week_start, end=today + timedelta(days=1), interval="1d", progress=False)
    if data.empty or "Close" not in data.columns:
        return np.nan
    return float(round(data["Close"].dropna().iloc[-1], 3))

def calculate_cagr(start, end, periods_per_year, periods):
    if start <= 0 or end <= 0 or periods == 0:
        return np.nan
    return (end / start) ** (periods_per_year / periods) - 1

def calculate_max_drawdown(prices):
    if len(prices) < 2:
        return 0.0
    arr = np.array(prices, dtype=np.float64)
    running_max = np.maximum.accumulate(arr)
    drawdowns = (arr - running_max) / running_max
    return drawdowns.min() * 100  # As percentage

# Fetch official name and country for each symbol (with caching for speed)
@st.cache_data(show_spinner=False)
def get_live_names_and_countries(symbols):
    names, countries = {}, {}
    for sym in symbols:
        try:
            info = yf.Ticker(sym).info
            names[sym] = info.get("longName") or info.get("shortName") or sym
            countries[sym] = info.get("country") or ""
        except Exception:
            names[sym] = sym
            countries[sym] = ""
    return names, countries

# ---------- Main App Logic ----------
if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names
    sheet_choice = st.selectbox("Select sheet to analyze", [""] + sheet_names, index=0)
    if sheet_choice and sheet_choice in sheet_names:
        tickers_df = pd.read_excel(xls, sheet_name=sheet_choice)
        if not all(col in tickers_df.columns for col in ["Symbol", "Exchange"]):
            st.error("Excel file must contain 'Symbol' and 'Exchange' columns.")
        else:
            symbols = tickers_df["Symbol"].tolist()
            weeks, last_friday = get_last_n_weeks(6)
            week_labels = [f"{m.strftime('%Y-%m-%d')} to {f.strftime('%Y-%m-%d')}" for m, f in weeks]

            # Current week: from last Friday to latest close
            today = datetime.today()
            current_week_start = last_friday
            last_data_close = today
            if today.weekday() >= 5:
                last_data_close = today - timedelta(days=(today.weekday() - 4))
            current_week_label = f"{current_week_start.strftime('%Y-%m-%d')} to {last_data_close.strftime('%Y-%m-%d')}"

            result = {}
            for symbol in symbols:
                closes = fetch_friday_closes(symbol, weeks)
                current_close = fetch_current_week_close(symbol, current_week_start)
                if closes is not None and len(closes) == 6:
                    result[symbol] = closes + [current_close]
                else:
                    st.warning(f"Ticker {symbol}: Could not fetch 6 weeks of valid closing prices. Skipped.")

            if result:
                all_labels = week_labels + [current_week_label]
                price_df = pd.DataFrame(result).T
                price_df.columns = all_labels
                price_df.index.name = "Symbol"
                price_df.reset_index(inplace=True)
                for col in all_labels:
                    price_df[col] = pd.to_numeric(price_df[col], errors="coerce")
                st.subheader("Weekly Closing Prices (Fri–Fri Weeks + Current Week)")
                st.dataframe(price_df.style.format(precision=2), use_container_width=True)

                # --- Normalized Performance Tab ---
                st.subheader("Normalized Performance")

                norm_df = price_df.set_index("Symbol")[all_labels]
                normed = norm_df.div(norm_df.iloc[:, 0], axis=0) * 100

                fig = go.Figure()
                for symbol in normed.index:
                    fig.add_trace(go.Scatter(
                        x=all_labels,
                        y=normed.loc[symbol],
                        mode="lines+markers",
                        name=symbol
                    ))
                fig.update_layout(
                    title="Normalized Price Performance (Start = 100)",
                    xaxis_title="Week",
                    yaxis_title="Normalized Price",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

                # --- Advanced Performance Metrics Table ---
                stats_data = []
                for symbol in norm_df.index:
                    series = norm_df.loc[symbol].dropna()
                    if len(series) >= 2:
                        start, end = series.iloc[0], series.iloc[-1]
                        cagr = calculate_cagr(start, end, periods_per_year=52, periods=len(series))
                        mdd = calculate_max_drawdown(series)
                        stats_data.append({
                            "Symbol": symbol,
                            "CAGR (Annualized)": f"{cagr * 100:.2f}%",
                            "Max Drawdown": f"{mdd:.2f}%"
                        })
                    else:
                        stats_data.append({
                            "Symbol": symbol,
                            "CAGR (Annualized)": "N/A",
                            "Max Drawdown": "N/A"
                        })
                stats_df = pd.DataFrame(stats_data).set_index("Symbol")

                # Add Ticker, Name, Country columns
                names_dict, countries_dict = get_live_names_and_countries(stats_df.index.tolist())
                stats_df["Ticker"] = stats_df.index
                stats_df["Name"] = stats_df.index.map(names_dict)
                stats_df["Country"] = stats_df.index.map(countries_dict)
                # Order columns as: Ticker, Name, Country, CAGR, Max Drawdown
                show_cols = ["Ticker", "Name", "Country"] + [col for col in stats_df.columns if col not in ["Ticker", "Name", "Country"]]
                stats_df = stats_df[show_cols]

                st.subheader("Advanced Performance Metrics")
                st.dataframe(stats_df, use_container_width=True)

else:
    st.info("Please upload an Excel file with at least 'Symbol' and 'Exchange' columns.")
