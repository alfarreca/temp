import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import openpyxl

# Enhanced import handling with user feedback
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    st.error("❌ yfinance package not installed. Please run: pip install yfinance")
    st.stop()

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False
    st.warning("⚠️ Visualization features disabled - matplotlib or seaborn not installed")

# Set page config
st.set_page_config(
    page_title="Stock Price Tracker", 
    layout="wide",
    page_icon="📈"
)

# [Rest of your existing code remains exactly the same...]
# [Keep all the function definitions and main() implementation]
# [Only the problematic yf.pdr_override() line has been removed]

if __name__ == "__main__":
    main()
