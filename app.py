
def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return (macd_line, macd_signal, macd_histogram)."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import requests
from PIL import Image

# App configuration
st.set_page_config(
    page_title="Gold CFD Day Trading Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #FFD700; font-weight: bold;}
    .sub-header {font-size: 1.5rem; color: #FFD700; border-bottom: 2px solid #FFD700; padding-bottom: 0.2rem;}
    .gold-text {color: #FFD700;}
    .important {background-color: #1a1a1a; padding: 15px; border-radius: 10px; border-left: 5px solid #FFD700;}
    .profit {color: #00FF00;}
    .loss {color: #FF0000;}
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<h1 class="main-header">Gold CFD Day Trading Assistant for eToro</h1>', unsafe_allow_html=True)
st.write("A comprehensive tool to help day traders analyze and track spot gold CFD trades on the eToro platform.")

# Sidebar
with st.sidebar:
    st.header("Account Settings")
    
    # Account balance input
    account_balance = st.number_input(
        "Account Balance (USD)", 
        min_value=100.0, 
        value=10000.0, 
        step=100.0
    )
    
    # Risk management settings
    st.subheader("Risk Management")
    risk_per_trade = st.slider("Risk per Trade (%)", 0.1, 5.0, 1.0, 0.1)
    stop_loss_default = st.number_input("Default Stop Loss (pips)", 10, 100, 20, 5)
    take_profit_default = st.number_input("Default Take Profit (pips)", 10, 200, 40, 5)
    
    # Trading preferences
    st.subheader("Trading Preferences")
    lot_size = st.selectbox("Lot Size", [0.01, 0.1, 0.5, 1.0, 2.0, 5.0], index=2)
    leverage = st.slider("Leverage", 1, 100, 10, 1)
    
    # News preferences
    st.subheader("News Preferences")
    news_keywords = st.text_input("Keywords to monitor", "gold, fed, inflation, USD, nonfarm payroll")
    
    # Display account metrics
    st.divider()
    st.subheader("Account Metrics")
    st.metric("Risk per Trade (USD)", f"${account_balance * risk_per_trade / 100:.2f}")
    st.metric("Available Margin", f"${account_balance * leverage:.2f}")

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Market Overview", 
    "Technical Analysis", 
    "Trade Calculator", 
    "Trade Journal", 
    "News & Analysis"
])

# Tab 1: Market Overview
with tab1:
    st.markdown('<h2 class="sub-header">Gold Market Overview</h2>', unsafe_allow_html=True)
    
    # Create mock price data (in a real app, this would come from an API)
    dates = pd.date_range(end=datetime.datetime.now(), periods=100, freq='D')
    prices = np.random.normal(1800, 50, 100).cumsum()
    
    # Display current price
    current_price = prices[-1]
    previous_price = prices[-2]
    price_change = current_price - previous_price
    change_percent = (price_change / previous_price) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2f} ({change_percent:.2f}%)")
    with col2:
        st.metric("Daily High", f"${current_price + 15:.2f}")
    with col3:
        st.metric("Daily Low", f"${current_price - 12:.2f}")
    
    # Price chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=dates[-30:],
        open=prices[-30:] - np.random.randint(5, 15, 30),
        high=prices[-30:] + np.random.randint(5, 15, 30),
        low=prices[-30:] - np.random.randint(5, 15, 30),
        close=prices[-30:],
        name="Gold Price"
    ))
    
    fig.update_layout(
        title="Gold Price (Last 30 Days)",
        yaxis_title="Price (USD)",
        xaxis_title="Date",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Market sentiment
    st.subheader("Market Sentiment")
    sentiment = np.random.choice(['Bullish', 'Bearish', 'Neutral'], p=[0.6, 0.2, 0.2])
    sentiment_color = "#00FF00" if sentiment == 'Bullish' else ("#FF0000" if sentiment == 'Bearish' else "#FFFF00")
    
    st.markdown(f"""
    <div class="important">
        <h3 style="color: {sentiment_color}; margin-top: 0;">Current Sentiment: {sentiment}</h3>
        <p>Based on technical indicators and recent price action, the market is showing {sentiment.lower()} tendencies.</p>
    </div>
    """, unsafe_allow_html=True)

# Tab 2: Technical Analysis
with tab2:
    st.markdown('<h2 class="sub-header">Technical Analysis</h2>', unsafe_allow_html=True)
    
    # Create technical indicators (mock data)
    prices_series = pd.Series(prices, index=dates)
    
    # Calculate indicators
    sma_20 = prices_series.rolling(window=20).mean()
    sma_50 = prices_series.rolling(window=50).mean()
    
    # RSI calculation
    delta = prices_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Create chart with subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price with Moving Averages', 'RSI'),
        vertical_spacing=0.1,
        row_width=[0.7, 0.3]
    )
    
    # Price and moving averages
    fig.add_trace(go.Scatter(x=dates, y=prices, name='Gold Price', line=dict(color='#FFD700')), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=sma_20, name='SMA 20', line=dict(color='#00FF00')), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=sma_50, name='SMA 50', line=dict(color='#FF0000')), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=dates, y=rsi, name='RSI', line=dict(color='#FFFFFF')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.update_layout(height=700, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical signals
    st.subheader("Technical Signals")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Moving Averages**: Golden Cross detected")
    
    with col2:
        if rsi.iloc[-1] > 70:
            st.error("**RSI**: Overbought (>70)")
        elif rsi.iloc[-1] < 30:
            st.success("**RSI**: Oversold (<30)")
        else:
            st.info("**RSI**: Neutral")
    
    with col3:
        st.warning("**Trend**: Strong Uptrend")

# Tab 3: Trade Calculator
with tab3:
    st.markdown('<h2 class="sub-header">Trade Calculator</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Position Sizing")
        
        entry_price = st.number_input("Entry Price", value=current_price, min_value=0.01, step=0.1)
        stop_loss = st.number_input("Stop Loss (pips)", value=stop_loss_default, min_value=1, step=1)
        risk_amount = st.number_input("Risk Amount (USD)", value=account_balance * risk_per_trade / 100, min_value=1.0, step=1.0)
        
        # Calculate position size
        pip_value = 10  # For gold, 1 pip is typically $10 per standard lot
        risk_in_pips = stop_loss
        position_size = risk_amount / (risk_in_pips * pip_value * 0.01)  # Simplified calculation
        
        st.metric("Position Size (Lots)", f"{position_size:.2f}")
        st.metric("Margin Required", f"${position_size * 1000 * entry_price / leverage:.2f}")
    
    with col2:
        st.subheader("Profit/Loss Calculator")
        
        take_profit = st.number_input("Take Profit (pips)", value=take_profit_default, min_value=1, step=1)
        units = st.number_input("Units", value=position_size * 1000, min_value=1.0, step=1.0)
        
        # Calculate potential outcomes
        potential_loss = risk_in_pips * pip_value * (units / 1000)
        potential_profit = take_profit * pip_value * (units / 1000)
        risk_reward_ratio = potential_profit / potential_loss
        
        st.metric("Potential Loss", f"-${potential_loss:.2f}")
        st.metric("Potential Profit", f"${potential_profit:.2f}")
        st.metric("Risk/Reward Ratio", f"{risk_reward_ratio:.2f}:1")
        
        if risk_reward_ratio >= 1.5:
            st.success("Good risk/reward ratio")
        else:
            st.warning("Consider improving your risk/reward ratio")

# Tab 4: Trade Journal
with tab4:
    st.markdown('<h2 class="sub-header">Trade Journal</h2>', unsafe_allow_html=True)
    
    # Initialize session state for trades if not exists
    if 'trades' not in st.session_state:
        st.session_state.trades = []
    
    with st.expander("Add New Trade"):
        col1, col2 = st.columns(2)
        
        with col1:
            trade_date = st.date_input("Trade Date", value=datetime.date.today())
            direction = st.selectbox("Direction", ["Long", "Short"])
            entry_price = st.number_input("Entry Price", min_value=0.01, value=current_price, step=0.1)
            position_size = st.number_input("Position Size", min_value=0.01, value=1.0, step=0.1)
        
        with col2:
            exit_price = st.number_input("Exit Price", min_value=0.01, value=current_price + 5, step=0.1)
            stop_loss = st.number_input("Stop Loss", min_value=0.01, value=current_price - 10, step=0.1)
            take_profit = st.number_input("Take Profit", min_value=0.01, value=current_price + 15, step=0.1)
            reason = st.text_area("Trade Reason")
        
        if st.button("Save Trade"):
            # Calculate P/L
            if direction == "Long":
                pips = (exit_price - entry_price) * 100
                profit = (exit_price - entry_price) * position_size * 1000
            else:
                pips = (entry_price - exit_price) * 100
                profit = (entry_price - exit_price) * position_size * 1000
            
            # Add trade to journal
            trade = {
                "date": trade_date,
                "direction": direction,
                "entry": entry_price,
                "exit": exit_price,
                "size": position_size,
                "pips": pips,
                "profit": profit,
                "reason": reason
            }
            
            st.session_state.trades.append(trade)
            st.success("Trade saved successfully!")
    
    # Display trade history
    if st.session_state.trades:
        st.subheader("Trade History")
        
        # Convert to DataFrame for display
        trades_df = pd.DataFrame(st.session_state.trades)
        
        # Format profit for display
        trades_df["profit_display"] = trades_df["profit"].apply(
            lambda x: f'<span class="profit">${x:.2f}</span>' if x >= 0 else f'<span class="loss">-${abs(x):.2f}</span>'
        )
        
        # Display table
        st.write(trades_df[["date", "direction", "entry", "exit", "size", "pips", "profit_display"]].to_html(escape=False), unsafe_allow_html=True)
        
        # Performance metrics
        st.subheader("Performance Metrics")
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df["profit"] > 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        total_profit = trades_df["profit"].sum()
        avg_profit = trades_df["profit"].mean() if total_trades > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trades", total_trades)
        col2.metric("Win Rate", f"{win_rate:.1f}%")
        col3.metric("Total Profit", f"${total_profit:.2f}")
        col4.metric("Average Profit", f"${avg_profit:.2f}")
        
        # Profit chart
        profit_cumulative = trades_df["profit"].cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trades_df["date"],
            y=profit_cumulative,
            mode='lines+markers',
            name='Equity Curve',
            line=dict(color='#FFD700')
        ))
        
        fig.update_layout(
            title="Equity Curve",
            yaxis_title="Profit (USD)",
            xaxis_title="Date",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trades recorded yet. Add your first trade to begin tracking your performance.")

# Tab 5: News & Analysis
with tab5:
    st.markdown('<h2 class="sub-header">Market News & Analysis</h2>', unsafe_allow_html=True)
    
    # Mock news data (in a real app, this would come from a news API)
    news_items = [
        {
            "title": "Fed Holds Rates Steady, Gold Prices Rally",
            "source": "Financial Times",
            "date": "2 hours ago",
            "impact": "High"
        },
        {
            "title": "Dollar Weakens as Inflation Cools More Than Expected",
            "source": "Bloomberg",
            "date": "5 hours ago",
            "impact": "Medium"
        },
        {
            "title": "Geopolitical Tensions Support Safe-Haven Demand for Gold",
            "source": "Reuters",
            "date": "1 day ago",
            "impact": "High"
        },
        {
            "title": "Central Banks Continue Gold Buying Spree",
            "source": "Kitco News",
            "date": "2 days ago",
            "impact": "Medium"
        }
    ]
    
    for news in news_items:
        impact_color = "#FF0000" if news["impact"] == "High" else ("#FFA500" if news["impact"] == "Medium" else "#00FF00")
        
        st.markdown(f"""
        <div style="padding: 15px; border-radius: 10px; background-color: #1a1a1a; margin-bottom: 15px;">
            <h3 style="margin-top: 0;">{news['title']}</h3>
            <p style="margin-bottom: 0;">
                <strong>Source:</strong> {news['source']} | 
                <strong>Date:</strong> {news['date']} | 
                <strong>Impact:</strong> <span style="color: {impact_color};">{news['impact']}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Economic calendar
    st.subheader("Economic Calendar")
    
    # Mock economic events
    events = [
        {"date": "Tomorrow, 8:30 AM EST", "event": "Nonfarm Payrolls", "currency": "USD", "impact": "High"},
        {"date": "Tomorrow, 10:00 AM EST", "event": "ISM Manufacturing PMI", "currency": "USD", "impact": "Medium"},
        {"date": "Next Tuesday, 2:00 PM EST", "event": "Fed Chair Speech", "currency": "USD", "impact": "High"},
        {"date": "Next Wednesday, 8:30 AM EST", "event": "CPI Data", "currency": "USD", "impact": "High"}
    ]
    
    for event in events:
        impact_color = "#FF0000" if event["impact"] == "High" else ("#FFA500" if event["impact"] == "Medium" else "#00FF00")
        
        st.markdown(f"""
        <div style="padding: 10px; border-radius: 5px; background-color: #1a1a1a; margin-bottom: 10px;">
            <strong>{event['date']}</strong>: {event['event']} ({event['currency']}) - 
            <span style="color: {impact_color};">{event['impact']} Impact</span>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>This application is for educational purposes only. Trading CFDs involves significant risk of loss.</p>
    <p>Always practice proper risk management and consider seeking advice from a financial professional.</p>
</div>
""", unsafe_allow_html=True)