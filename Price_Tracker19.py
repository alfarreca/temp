import streamlit as st
import pandas as pd
import yfinance as yf

# Example performance data (replace with your own calculation)
data = {
    "Ticker": ["THD", "EPHE", "VNM", "EPOL", "EWH", "CEZ.PR", "RSX", "MCHI", "EWT", "EWY"],
    "CAGR (Annualized)": ["-12.81%", "-4.39%", "76.42%", "115.90%", "85.67%", "43.29%", "0.00%", "57.81%", "129.87%", "347.82%"],
    "Max Drawdown": ["-6.55%", "-4.06%", "-4.55%", "-3.00%", "-1.71%", "-0.57%", "0.00%", "-1.62%", "-2.99%", "-0.85%"],
}
metrics_df = pd.DataFrame(data)

@st.cache_data(show_spinner=False)
def get_live_names(tickers):
    names = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            # Try longName, then shortName, then fallback to ticker
            names[t] = info.get("longName") or info.get("shortName") or t
        except Exception:
            names[t] = t
    return names

# Fetch live names
with st.spinner("Fetching official ticker names..."):
    names_dict = get_live_names(metrics_df["Ticker"].tolist())

metrics_df["Name"] = metrics_df["Ticker"].map(names_dict)
metrics_df = metrics_df[["Ticker", "Name", "CAGR (Annualized)", "Max Drawdown"]]

st.subheader("Advanced Performance Metrics")
st.dataframe(metrics_df, use_container_width=True)
