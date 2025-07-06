import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

st.title("📈 Weekly Price Tracker (Fri–Fri Weeks + Current Week + Scoring + ML Prediction)")

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
    data = yf.download(symbol, start=first_monday, end=last_friday + timedelta(days=1), interval="1d")
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
    data = yf.download(symbol, start=current_week_start, end=today + timedelta(days=1), interval="1d")
    if data.empty or "Close" not in data.columns:
        return np.nan
    return float(round(data["Close"].iloc[-1], 3))

def get_friday(label):
    return label.split(" to ")[1]

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

def calculate_strategy_scores(price_df, all_labels):
    scores = []
    for idx, row in price_df.iterrows():
        closes = row[all_labels].astype(float).values
        symbol = row["Symbol"]

        # Momentum Score: % return over last week
        if len(closes) < 2 or closes[-2] == 0 or np.isnan(closes[-2]) or np.isnan(closes[-1]):
            momentum_score = np.nan
        else:
            momentum_score = ((closes[-1] - closes[-2]) / closes[-2]) * 100

        # Volatility-Adj Score: Sharpe-like, mean return/std
        returns = np.diff(closes) / closes[:-1]
        mean_return = np.nanmean(returns)
        std_return = np.nanstd(returns)
        if std_return > 0:
            vol_adj_score = mean_return / std_return * 100
        else:
            vol_adj_score = mean_return * 100 if not np.isnan(mean_return) else np.nan

        # Trend Consistency: up weeks out of last 5
        trend_consistency = np.sum(returns[-5:] > 0) if returns.shape[0] >= 5 else np.nan

        # Last Week % Change: % change in the last week
        last_week_pct = momentum_score

        # Total Return %: from first to last
        if closes[0] and not np.isnan(closes[0]) and not np.isnan(closes[-1]):
            total_return = ((closes[-1] - closes[0]) / closes[0]) * 100
        else:
            total_return = np.nan

        # All-Arounder Score: custom sum (example: Trend*10 + VolAdj + Momentum)
        if not any(np.isnan([trend_consistency, vol_adj_score, momentum_score])):
            all_around_score = trend_consistency * 10 + vol_adj_score + momentum_score
        else:
            all_around_score = np.nan

        scores.append({
            "Symbol": symbol,
            "Momentum Score": round(momentum_score, 2) if not np.isnan(momentum_score) else None,
            "Volatility-Adj Score": round(vol_adj_score, 2) if not np.isnan(vol_adj_score) else None,
            "Trend Consistency": int(trend_consistency) if not np.isnan(trend_consistency) else None,
            "Last Week % Change": round(last_week_pct, 2) if not np.isnan(last_week_pct) else None,
            "Total Return %": round(total_return, 2) if not np.isnan(total_return) else None,
            "All-Arounder Score": int(all_around_score) if not np.isnan(all_around_score) else None,
        })

    score_df = pd.DataFrame(scores)
    # Sort by All-Arounder
    score_df = score_df.sort_values("All-Arounder Score", ascending=False).reset_index(drop=True)
    return score_df

if uploaded_file:
    tickers_df = pd.read_excel(uploaded_file)

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
            # Add the column label
            all_labels = week_labels + [current_week_label]
            price_df = pd.DataFrame(result).T
            price_df.columns = all_labels
            price_df.reset_index(inplace=True)
            price_df.rename(columns={'index': 'Symbol'}, inplace=True)
            for col in all_labels:
                price_df[col] = pd.to_numeric(price_df[col], errors="coerce")
            st.subheader("Weekly Closing Prices (Fri–Fri Weeks + Current Week)")
            st.dataframe(price_df.style.format(precision=2))

            # --- Weekly % Change as percentage string ---
            try:
                pct_change_df = price_df.set_index("Symbol")[all_labels].astype(float).pct_change(axis=1) * 100
            except Exception as e:
                st.error(f"Error computing percent change: {e}")
            else:
                pct_change_df = pct_change_df.iloc[:, 1:]
                pct_change_df = pct_change_df.round(2)
                pct_change_str = pct_change_df.applymap(lambda x: "" if pd.isna(x) else f"{x:+.2f}%")
                pct_change_str.reset_index(inplace=True)
                pct_change_str.columns = ["Symbol"] + [
                    f"% Change {get_friday(all_labels[i-1])} to {get_friday(all_labels[i])}"
                    for i in range(1, len(all_labels))
                ]
                st.subheader("Weekly % Price Change")
                st.dataframe(pct_change_str)

            # --- Chart Tabs: Price Trend & Normalized Performance & Ticker Scores & ML ---
            tab1, tab2, tab3, tab4 = st.tabs([
                "Price Trend",
                "Normalized Performance",
                "Ticker Scores (5 Strategies)",
                "ML Next Week Prediction"
            ])

            with tab1:
                st.subheader("Weekly Closing Price Trend")
                ticker_options = price_df["Symbol"].tolist()
                tickers_to_plot = st.multiselect(
                    "Select tickers to plot", ticker_options, default=ticker_options[:min(3, len(ticker_options))], key="trend"
                )
                if tickers_to_plot:
                    fig, ax = plt.subplots()
                    for sym in tickers_to_plot:
                        row = price_df[price_df["Symbol"] == sym]
                        if not row.empty:
                            ax.plot(all_labels, row.iloc[0, 1:], marker='o', label=sym)
                    ax.set_xlabel("Week (Friday to Friday, Current: Fri to latest close)")
                    ax.set_ylabel("Closing Price")
                    ax.set_title("Weekly Closing Price Trend")
                    ax.legend()
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

            with tab2:
                st.markdown("**Performance normalized to 100 at the start: compare pure relative gains/losses.**")
                ticker_options = price_df["Symbol"].tolist()
                tickers_to_plot = st.multiselect(
                    "Select tickers to plot (normalized)", ticker_options, default=ticker_options[:min(3, len(ticker_options))], key="norm"
                )
                if tickers_to_plot:
                    fig, ax = plt.subplots()
                    cagr_dict = {}
                    mdd_dict = {}
                    for sym in tickers_to_plot:
                        row = price_df[price_df["Symbol"] == sym]
                        if not row.empty:
                            prices = row.iloc[0, 1:].astype(float)
                            norm_prices = (prices / prices.iloc[0]) * 100 if prices.iloc[0] != 0 else prices
                            ax.plot(all_labels, norm_prices, marker='o', label=sym)
                            periods = len(norm_prices) - 1
                            periods_per_year = 52
                            cagr = calculate_cagr(norm_prices.iloc[0], norm_prices.iloc[-1], periods_per_year, periods)
                            cagr_dict[sym] = f"{cagr*100:.2f}%" if not np.isnan(cagr) else "n/a"
                            mdd = calculate_max_drawdown(norm_prices)
                            mdd_dict[sym] = f"{mdd:.2f}%" if not np.isnan(mdd) else "n/a"
                    ax.set_xlabel("Week")
                    ax.set_ylabel("Normalized Price (Start=100)")
                    ax.set_title("Normalized Weekly Performance")
                    ax.legend()
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    stats_df = pd.DataFrame({
                        "CAGR (Annualized)": cagr_dict,
                        "Max Drawdown": mdd_dict
                    })
                    st.subheader("Advanced Performance Metrics")
                    st.dataframe(stats_df)

            with tab3:
                score_df = calculate_strategy_scores(price_df, all_labels)
                score_df["Flags"] = ""
                breakout_idx = score_df["Momentum Score"].idxmax()
                score_df.at[breakout_idx, "Flags"] += "🟢 "
                all_around_idxs = score_df.nlargest(3, "All-Arounder Score").index
                for i in all_around_idxs:
                    score_df.at[i, "Flags"] += "🔵 "
                score_df = score_df[["Flags", "Symbol", "Momentum Score", "Volatility-Adj Score", "Trend Consistency",
                                    "Last Week % Change", "Total Return %", "All-Arounder Score"]]
                st.markdown("### Ticker Scores (5 Strategies)")
                st.dataframe(score_df)
                breakout_candidates = score_df.loc[score_df["Flags"].str.contains("🟢"), "Symbol"].tolist()
                all_arounders = score_df.loc[score_df["Flags"].str.contains("🔵"), "Symbol"].tolist()
                if breakout_candidates:
                    st.success(f"🟢 **Breakout candidates:** {', '.join(breakout_candidates)}")
                if all_arounders:
                    st.info(f"🔵 **All-arounders:** {', '.join(all_arounders)}")

            with tab4:
                st.markdown("### ML Next Week Prediction")
                # --- Simulate rolling historical feature table for all tickers ---
                history = []
                for idx, row in price_df.iterrows():
                    closes = row[all_labels].astype(float).values
                    symbol = row["Symbol"]
                    for i in range(2, len(closes)-1):
                        last_week_return = (closes[i] - closes[i-1]) / closes[i-1] * 100 if closes[i-1] != 0 else 0
                        prev_week_return = (closes[i-1] - closes[i-2]) / closes[i-2] * 100 if closes[i-2] != 0 else 0
                        trend = np.sum(np.diff(closes[:i+1]) > 0)
                        volatility = np.std(np.diff(closes[:i+1]) / closes[:i]) if np.std(closes[:i+1]) > 0 else 0
                        total_return = (closes[i] - closes[0]) / closes[0] * 100 if closes[0] != 0 else 0
                        next_week_return = (closes[i+1] - closes[i]) / closes[i] * 100 if closes[i] != 0 else 0
                        history.append({
                            "Symbol": symbol,
                            "Momentum": last_week_return,
                            "Prev Momentum": prev_week_return,
                            "Trend Consistency": trend,
                            "Volatility": volatility,
                            "Total Return": total_return,
                            "Next_Week_Return": next_week_return,
                        })
                hist_df = pd.DataFrame(history)
                if not hist_df.empty and hist_df["Momentum"].notna().any():
                    X = hist_df[["Momentum", "Prev Momentum", "Trend Consistency", "Volatility", "Total Return"]]
                    y = hist_df["Next_Week_Return"]
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                    model.fit(X_scaled, y)
                    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                    st.subheader("ML Model Feature Importances")
                    st.bar_chart(importances)
                    # Predict for current tickers (using their latest features)
                    latest_features = []
                    symbols_pred = []
                    for idx, row in price_df.iterrows():
                        closes = row[all_labels].astype(float).values
                        symbol = row["Symbol"]
                        last_week_return = (closes[-1] - closes[-2]) / closes[-2] * 100 if closes[-2] != 0 else 0
                        prev_week_return = (closes[-2] - closes[-3]) / closes[-3] * 100 if closes[-3] != 0 else 0
                        trend = np.sum(np.diff(closes) > 0)
                        volatility = np.std(np.diff(closes) / closes[:-1]) if np.std(closes) > 0 else 0
                        total_return = (closes[-1] - closes[0]) / closes[0] * 100 if closes[0] != 0 else 0
                        features = [last_week_return, prev_week_return, trend, volatility, total_return]
                        latest_features.append(features)
                        symbols_pred.append(symbol)
                    latest_X = scaler.transform(latest_features)
                    predicted_returns = model.predict(latest_X)
                    predicted_df = pd.DataFrame({
                        "Symbol": symbols_pred,
                        "Predicted_Next_Week_%": predicted_returns
                    }).sort_values("Predicted_Next_Week_%", ascending=False).reset_index(drop=True)
                    st.subheader("ML Model: Predicted Top Next Week Winners")
                    st.dataframe(predicted_df.head(10))
                else:
                    st.info("Not enough historical data to train ML model. Collect more weekly data for stronger predictions.")

        else:
            st.error("No valid data fetched for the provided tickers.")
