# Streamlit Screener App: Multi-Mode Stock Screener

import streamlit as st
import pandas as pd
type_filters = ["Value Screener", "Growth Screener", "Dividend Screener", "Technical Screener", "Thematic Screener"]

st.set_page_config(page_title="Multi-Screener App", layout="wide")
st.title("📊 Stock Screener App")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

@st.cache_data
def load_excel(file):
    return pd.read_excel(file)

if uploaded_file:
    df = load_excel(uploaded_file)
    st.subheader("Raw Data Preview")
    st.dataframe(df, use_container_width=True)

    screener_type = st.sidebar.selectbox("Select Screener Type", type_filters)

    filtered_df = df.copy()

    if screener_type == "Value Screener":
        st.sidebar.subheader("Value Filters")
        pb_threshold = st.sidebar.slider("Max P/B Ratio", 0.0, 5.0, 1.5)
        pe_threshold = st.sidebar.slider("Max P/E Ratio", 0.0, 50.0, 20.0)
        roe_threshold = st.sidebar.slider("Min ROE %", 0.0, 30.0, 10.0)

        for col in ["P/B Ratio", "P/E Ratio", "ROE (TTM)"]:
            if col not in filtered_df.columns:
                st.warning(f"Column '{col}' not found. Please include it in your Excel.")

        filtered_df = filtered_df[(filtered_df["P/B Ratio"] <= pb_threshold) &
                                  (filtered_df["P/E Ratio"] <= pe_threshold) &
                                  (filtered_df["ROE (TTM)"] >= roe_threshold)]

    elif screener_type == "Growth Screener":
        st.sidebar.subheader("Growth Filters")
        eps_growth = st.sidebar.slider("Min EPS Growth %", 0, 100, 10)
        rev_growth = st.sidebar.slider("Min Revenue Growth %", 0, 100, 10)

        for col in ["EPS YoY %", "Revenue YoY %"]:
            if col not in filtered_df.columns:
                st.warning(f"Column '{col}' not found. Please include it in your Excel.")

        filtered_df = filtered_df[(filtered_df["EPS YoY %"] >= eps_growth) &
                                  (filtered_df["Revenue YoY %"] >= rev_growth)]

    elif screener_type == "Dividend Screener":
        st.sidebar.subheader("Dividend Filters")
        min_yield = st.sidebar.slider("Min Dividend Yield %", 0.0, 10.0, 3.0)
        max_payout = st.sidebar.slider("Max Payout Ratio %", 0.0, 100.0, 70.0)

        for col in ["Dividend Yield", "Payout Ratio"]:
            if col not in filtered_df.columns:
                st.warning(f"Column '{col}' not found. Please include it in your Excel.")

        filtered_df = filtered_df[(filtered_df["Dividend Yield"] >= min_yield) &
                                  (filtered_df["Payout Ratio"] <= max_payout)]

    elif screener_type == "Technical Screener":
        st.sidebar.subheader("Technical Filters")
        rsi_max = st.sidebar.slider("Max RSI", 0, 100, 70)
        macd_signal = st.sidebar.selectbox("MACD Signal", ["Any", "Bullish", "Bearish"])

        if "RSI" not in filtered_df.columns or "MACD Signal" not in filtered_df.columns:
            st.warning("Please include 'RSI' and 'MACD Signal' columns in your Excel.")
        else:
            filtered_df = filtered_df[filtered_df["RSI"] <= rsi_max]
            if macd_signal != "Any":
                filtered_df = filtered_df[filtered_df["MACD Signal"] == macd_signal]

    elif screener_type == "Thematic Screener":
        st.sidebar.subheader("Thematic Filters")
        selected_sector = st.sidebar.multiselect("Select Sector(s)", options=df["Sector"].dropna().unique())
        selected_theme = st.sidebar.multiselect("Select Theme(s)", options=df["Theme"].dropna().unique())
        selected_country = st.sidebar.multiselect("Select Country(s)", options=df["Country"].dropna().unique())

        if selected_sector:
            filtered_df = filtered_df[filtered_df["Sector"].isin(selected_sector)]
        if selected_theme:
            filtered_df = filtered_df[filtered_df["Theme"].isin(selected_theme)]
        if selected_country:
            filtered_df = filtered_df[filtered_df["Country"].isin(selected_country)]

    st.subheader(f"🎯 Results: {len(filtered_df)} stocks found")
    st.dataframe(filtered_df, use_container_width=True)

    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Screener Results", csv, "screener_results.csv", "text/csv")

else:
    st.info("Please upload an Excel file with your stock universe.")
