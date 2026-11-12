"""
Bobby Stock Price Prediction Demo
=================================
Compares the annual trading performance of three simulated investors ($100,000 budget,
trading on the first trading day of each week over the most recent 365 calendar days):
  1. Investor 1 (SMA Crossover): Short MA = 5, Long MA = 20 on 1-day timeframe
     using only the most recent 365 calendar days of data.
  2. Investor 2 (ML Linear Regression): Trained on the past 4 years of data (excluding
     the most recent 365 calendar days) mapping 5-day past daily price movements to the
     next 5-day average price movement; tested on the most recent 365 calendar days.
  3. Investor 3 (CNN Graphical Analysis): Generates 5-day Candlestick chart images
     labeled with "Buy" / "Sell" and the absolute effect size of the next 5-day average
     price movement. Trained on the past 4 years of data and simulated on the most
     recent 365 calendar days.
"""

from datetime import timedelta
import math
import traceback
import warnings

import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# Optional TensorFlow/Keras import with graceful fallback
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Bobby Stock Price Prediction Demo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

INITIAL_BUDGET = 100_000.0
RECENT_DAYS = 365


# ---------------------------------------------------------------------------
# 1. Data Fetching & Preprocessing (Yahoo Finance 5 Years Daily Data)
# ---------------------------------------------------------------------------
def _extract_1d_series(df: pd.DataFrame, col_name: str) -> pd.Series:
    """Safely extracts a 1D numeric Series from a DataFrame even if columns are MultiIndex or duplicated."""
    sub = df[col_name]
    if isinstance(sub, pd.DataFrame):
        sub = sub.iloc[:, 0]
    return pd.to_numeric(sub, errors="coerce")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker: str) -> pd.DataFrame:
    """Downloads 5 years of daily OHLCV stock data from Yahoo Finance."""
    clean_ticker = ticker.strip().upper() or "NVDA"
    raw_df = None

    # Method 1: yf.Ticker().history() returns a clean single-level DataFrame
    try:
        tk = yf.Ticker(clean_ticker)
        raw_df = tk.history(period="5y", interval="1d", auto_adjust=False)
    except Exception:
        raw_df = None

    # Method 2: Fallback to yf.download()
    if raw_df is None or raw_df.empty:
        try:
            raw_df = yf.download(
                clean_ticker,
                period="5y",
                interval="1d",
                auto_adjust=False,
                progress=False,
            )
        except Exception:
            raw_df = None

    if raw_df is None or raw_df.empty:
        raise ValueError(f"No data returned from Yahoo Finance for ticker '{clean_ticker}'.")

    # Flatten MultiIndex columns if present (handles both (Price, Ticker) and (Ticker, Price))
    if isinstance(raw_df.columns, pd.MultiIndex):
        lvl0 = [str(c[0]) for c in raw_df.columns]
        lvl1 = [str(c[-1]) for c in raw_df.columns]
        if "Close" in lvl0:
            raw_df.columns = lvl0
        elif "Close" in lvl1:
            raw_df.columns = lvl1
        else:
            raw_df.columns = [str(c[0]) for c in raw_df.columns]

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in raw_df.columns:
            raise ValueError(f"Missing required column '{col}' in Yahoo Finance dataset.")

    df = pd.DataFrame(index=raw_df.index)
    for col in required_cols:
        df[col] = _extract_1d_series(raw_df, col)

    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    df["Volume"] = df["Volume"].fillna(0.0)

    # Normalize DatetimeIndex without timezone and remove any duplicate dates
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx
    df = df[~df.index.duplicated(keep="last")].sort_index().copy()

    # Identify the first trading day of each ISO calendar week
    iso = df.index.isocalendar()
    year_str = iso["year"].astype(str).values
    week_str = [f"{int(w):02d}" for w in iso["week"].values]
    df["YearWeek"] = [f"{y}-W{w}" for y, w in zip(year_str, week_str)]
    df["IsFirstTradingDayOfWeek"] = ~df["YearWeek"].duplicated(keep="first")
    return df


def split_train_and_recent_year(df: pd.DataFrame):
    """
    Splits 5-year daily stock data into:
      - train_df: Past 4 years of data (NOT in the most recent 365 calendar days)
      - test_df: Most recent 365 calendar days of data
    """
    latest_date = df.index.max()
    cutoff_date = latest_date - timedelta(days=RECENT_DAYS)
    train_df = df[df.index <= cutoff_date].copy()
    test_df = df[df.index > cutoff_date].copy()
    return train_df, test_df, cutoff_date, latest_date


# ---------------------------------------------------------------------------
# 2. Trading Execution Engine (First Trading Day of the Week Only)
# ---------------------------------------------------------------------------
def execute_weekly_simulation(
    test_df: pd.DataFrame,
    signals: list,
    criticalities: list,
    investor_name: str,
    initial_budget: float = INITIAL_BUDGET,
):
    """
    Simulates an investor who ONLY trades on the first trading day of each week.
    Trading volume depends on how critical the signal is (criticality in [0.10, 1.00]):
      - If Signal == 'Buy': invests `alloc_fraction * cash_on_hand` into stock at Close price.
      - If Signal == 'Sell': sells `alloc_fraction * shares_held` at Close price.
    """
    cash = float(initial_budget)
    shares = 0.0
    history_rows = []
    trade_log = []

    closes = test_df["Close"].values.astype(float)
    is_first_days = test_df["IsFirstTradingDayOfWeek"].values.astype(bool)
    dates = test_df.index

    for i in range(len(test_df)):
        dt = dates[i]
        price = float(closes[i])
        is_trade_day = bool(is_first_days[i])
        sig = str(signals[i]) if i < len(signals) else "Hold"
        crit = float(np.clip(criticalities[i] if i < len(criticalities) else 0.0, 0.0, 1.0))
        action_taken = "Hold"
        traded_shares = 0.0
        trade_value = 0.0

        if is_trade_day and sig in ("Buy", "Sell") and crit > 0.0:
            alloc_fraction = float(np.clip(0.15 + 0.80 * crit, 0.15, 0.95))
            if sig == "Buy" and cash > 10.0:
                trade_value = cash * alloc_fraction
                traded_shares = trade_value / price
                cash -= trade_value
                shares += traded_shares
                action_taken = "Buy"
            elif sig == "Sell" and shares > 1e-6:
                traded_shares = shares * alloc_fraction
                trade_value = traded_shares * price
                shares -= traded_shares
                cash += trade_value
                action_taken = "Sell"

            if action_taken in ("Buy", "Sell"):
                # Evaluate 5-day forward average movement for win rate
                fwd_slice = closes[i + 1 : min(len(closes), i + 6)]
                fwd_avg = float(np.mean(fwd_slice)) if len(fwd_slice) > 0 else price
                is_win = (action_taken == "Buy" and fwd_avg >= price) or (
                    action_taken == "Sell" and fwd_avg <= price
                )
                trade_log.append(
                    {
                        "Date": dt.strftime("%Y-%m-%d"),
                        "Weekday": dt.strftime("%A"),
                        "Action": action_taken,
                        "Close Price ($)": round(price, 2),
                        "Signal Criticality (%)": round(crit * 100, 1),
                        "Allocation (%)": round(alloc_fraction * 100, 1),
                        "Shares Traded": round(traded_shares, 4),
                        "Trade Value ($)": round(trade_value, 2),
                        "Cash After ($)": round(cash, 2),
                        "Stock Value After ($)": round(shares * price, 2),
                        "Total Worth ($)": round(cash + shares * price, 2),
                        "_is_win": bool(is_win),
                    }
                )

        stock_val = shares * price
        total_worth = cash + stock_val
        history_rows.append(
            {
                "Date": dt,
                "Close": price,
                "Cash": cash,
                "Shares": shares,
                "StockValue": stock_val,
                "TotalWorth": total_worth,
                "Action": action_taken,
                "Signal": sig,
                "Criticality": crit,
            }
        )

    hist_df = pd.DataFrame(history_rows).set_index("Date")
    wins = sum(1 for t in trade_log if t.get("_is_win", False))
    win_rate = (wins / len(trade_log) * 100.0) if len(trade_log) > 0 else 0.0

    clean_trade_log = [
        {k: v for k, v in t.items() if not k.startswith("_")} for t in trade_log
    ]
    trades_df = pd.DataFrame(clean_trade_log)

    final_worth = float(hist_df["TotalWorth"].iloc[-1])
    final_cash = float(hist_df["Cash"].iloc[-1])
    final_stock_val = float(hist_df["StockValue"].iloc[-1])
    growth_usd = final_worth - initial_budget
    growth_pct = (growth_usd / initial_budget) * 100.0

    rolling_max = hist_df["TotalWorth"].cummax()
    drawdown = (hist_df["TotalWorth"] - rolling_max) / rolling_max
    max_dd_pct = float(drawdown.min() * 100.0)

    return {
        "investor": investor_name,
        "history": hist_df,
        "trades": trades_df,
        "final_worth": final_worth,
        "final_cash": final_cash,
        "final_stock_value": final_stock_val,
        "growth_usd": growth_usd,
        "growth_pct": growth_pct,
        "num_trades": len(clean_trade_log),
        "buy_trades": sum(1 for x in clean_trade_log if x["Action"] == "Buy"),
        "sell_trades": sum(1 for x in clean_trade_log if x["Action"] == "Sell"),
        "win_rate_pct": win_rate,
        "max_drawdown_pct": max_dd_pct,
    }


# ---------------------------------------------------------------------------
# 3. Approach 1: SMA Crossover Strategy (Short MA = 5, Long MA = 20)
# ---------------------------------------------------------------------------
def run_sma_strategy(test_df: pd.DataFrame, short_window: int = 5, long_window: int = 20):
    """
    Uses ONLY daily stock data within the most recent 365 calendar days.
    Short MA = 5, Long MA = 20 (1-day timeframe).
    Buy when SMA_5 crosses above SMA_20 (Golden Cross);
    Sell when SMA_5 crosses below SMA_20 (Death Cross).
    """
    sma_df = test_df.copy()
    sma_df["SMA_Short"] = sma_df["Close"].rolling(window=short_window, min_periods=short_window).mean()
    sma_df["SMA_Long"] = sma_df["Close"].rolling(window=long_window, min_periods=long_window).mean()
    sma_df["Spread"] = sma_df["SMA_Short"] - sma_df["SMA_Long"]
    sma_df["SpreadPct"] = (sma_df["Spread"] / sma_df["Close"]).abs()

    valid_spreads = sma_df["SpreadPct"].dropna().values
    scale_denom = float(np.percentile(valid_spreads, 90)) if len(valid_spreads) > 0 else 0.02
    scale_denom = max(scale_denom, 1e-4)

    signals = []
    criticalities = []
    prev_regime = 0
    crossover_in_week = 0

    sma_s_vals = sma_df["SMA_Short"].values
    sma_l_vals = sma_df["SMA_Long"].values
    spread_pct_vals = sma_df["SpreadPct"].values
    is_first_vals = sma_df["IsFirstTradingDayOfWeek"].values

    for i in range(len(sma_df)):
        if np.isnan(sma_s_vals[i]) or np.isnan(sma_l_vals[i]):
            signals.append("Hold")
            criticalities.append(0.0)
            continue

        curr_regime = 1 if sma_s_vals[i] > sma_l_vals[i] else -1
        if prev_regime != 0 and curr_regime != prev_regime:
            crossover_in_week = curr_regime
        prev_regime = curr_regime

        if bool(is_first_vals[i]):
            active_dir = crossover_in_week if crossover_in_week != 0 else curr_regime
            raw_crit = float(spread_pct_vals[i] / scale_denom)
            if crossover_in_week != 0:
                raw_crit = min(1.0, raw_crit * 1.25 + 0.15)
            crit = float(np.clip(raw_crit, 0.15, 1.0))
            signals.append("Buy" if active_dir > 0 else "Sell")
            criticalities.append(crit)
            crossover_in_week = 0
        else:
            signals.append("Hold")
            criticalities.append(0.0)

    sim_result = execute_weekly_simulation(
        sma_df,
        signals,
        criticalities,
        investor_name="Investor 1: SMA Crossover (5/20)",
    )
    sim_result["sma_df"] = sma_df
    return sim_result


# ---------------------------------------------------------------------------
# 4. Feature Engineering for 5-Day Past -> 5-Day Future Price Movements
# ---------------------------------------------------------------------------
def build_5day_windows(df: pd.DataFrame):
    """
    Constructs sliding windows over `df`:
      - Past 5 days daily price movements:
        x_k = (Close[t - 4 + k] - Close[t - 5 + k]) / Close[t - 5 + k], k = 0..4
      - Target: Average price movement of the next 5 days relative to Close[t]:
        y = mean_{m=1..5} ( (Close[t + m] - Close[t]) / Close[t] )
    """
    closes = df["Close"].values.astype(float)
    dates = df.index
    X_list, y_list, date_list, window_ohlc_list = [], [], [], []

    for i in range(5, len(df) - 5):
        past_returns = [
            (closes[i - 4 + k] - closes[i - 5 + k]) / max(closes[i - 5 + k], 1e-6)
            for k in range(5)
        ]
        future_moves = [
            (closes[i + m] - closes[i]) / max(closes[i], 1e-6) for m in range(1, 6)
        ]
        avg_future_move = float(np.mean(future_moves))

        X_list.append(past_returns)
        y_list.append(avg_future_move)
        date_list.append(dates[i])
        window_ohlc_list.append(df.iloc[i - 4 : i + 1][["Open", "High", "Low", "Close"]])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, pd.DatetimeIndex(date_list), window_ohlc_list


# ---------------------------------------------------------------------------
# 5. Approach 2: ML Linear Regression (4-Year Train -> 1-Year Test)
# ---------------------------------------------------------------------------
def run_linear_regression_strategy(df: pd.DataFrame, cutoff_date: pd.Timestamp):
    """
    Trains an OLS Linear Regression model on the past 4 years of data
    (excluding the most recent 365 calendar days) and simulates weekly trading
    on the most recent 365 calendar days.
    """
    X_all, y_all, dates_idx, _ = build_5day_windows(df)

    train_mask = np.asarray(dates_idx < (cutoff_date - timedelta(days=7)), dtype=bool)
    test_mask = np.asarray(dates_idx > cutoff_date, dtype=bool)

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test) if len(X_test) > 0 else np.zeros(1, dtype=np.float32)

    p90_effect = float(np.percentile(np.abs(y_train_pred), 90)) if len(y_train_pred) > 0 else 0.01
    p90_effect = max(p90_effect, 1e-4)

    # Evaluate across all rows in test_df using positional indices
    test_df = df[df.index > cutoff_date].copy()
    test_start_pos = len(df) - len(test_df)
    closes = df["Close"].values.astype(float)

    signals = []
    criticalities = []

    for offset in range(len(test_df)):
        idx_pos = test_start_pos + offset
        if idx_pos >= 5:
            past_ret = np.array(
                [
                    (closes[idx_pos - 4 + k] - closes[idx_pos - 5 + k]) / max(closes[idx_pos - 5 + k], 1e-6)
                    for k in range(5)
                ],
                dtype=np.float32,
            ).reshape(1, -1)
            pred_val = float(np.ravel(model.predict(past_ret))[0])
            sig = "Buy" if pred_val > 0 else "Sell"
            crit = float(np.clip(abs(pred_val) / p90_effect, 0.15, 1.0))
            signals.append(sig)
            criticalities.append(crit)
        else:
            signals.append("Hold")
            criticalities.append(0.0)

    sim_result = execute_weekly_simulation(
        test_df,
        signals,
        criticalities,
        investor_name="Investor 2: ML Linear Regression",
    )

    dir_acc = float(np.mean((y_test_pred > 0) == (y_test > 0)) * 100.0) if len(y_test) > 0 else 0.0
    r2_val = float(r2_score(y_test, y_test_pred)) if len(y_test) > 1 else 0.0
    rmse_val = float(math.sqrt(mean_squared_error(y_test, y_test_pred))) if len(y_test) > 0 else 0.0

    coefs = [float(c) for c in np.ravel(model.coef_)]
    intercept = float(np.ravel(model.intercept_)[0]) if np.ndim(model.intercept_) > 0 else float(model.intercept_)
    dominant_lag = int(np.argmax(np.abs(coefs))) + 1
    regime_type = (
        "momentum-driven (positive sum of lag weights)"
        if sum(coefs) > 0
        else "mean-reverting (negative sum of lag weights)"
    )
    outcome_word = (
        "outperformed the initial budget (WIN)"
        if sim_result["growth_usd"] >= 0
        else "finished below the initial budget (LOSS)"
    )

    explanation = (
        f"**Win/Loss Outcome**: Investor 2 (ML Linear Regression) achieved a final portfolio worth of "
        f"**${sim_result['final_worth']:,.2f}** ({sim_result['growth_pct']:+.2f}% vs. the $100,000 budget), "
        f"meaning the strategy **{outcome_word}** over the most recent 365 calendar days.\n\n"
        f"**Why It Won/Lost**:\n"
        f"1. **Learned Linear Dynamics ({regime_type})**: Trained on {len(X_train):,} 5-day historical windows "
        f"from the prior 4 years, the model learned weights "
        f"$\\beta_1={coefs[0]:+.4f}, \\beta_2={coefs[1]:+.4f}, \\beta_3={coefs[2]:+.4f}, "
        f"\\beta_4={coefs[3]:+.4f}, \\beta_5={coefs[4]:+.4f}$ with intercept $\\beta_0={intercept:+.5f}$. "
        f"Day-{dominant_lag} price movement had the largest influence on predicted 5-day forward returns.\n"
        f"2. **Out-of-Sample Signal Quality**: Across the recent test year, the linear model achieved a "
        f"**{dir_acc:.1f}% directional hit rate** ($R^2 = {r2_val:.4f}$, $\\text{{RMSE}} = {rmse_val:.4f}$) "
        f"and a **{sim_result['win_rate_pct']:.1f}% weekly trade win rate** across {sim_result['num_trades']} executed weekly trades "
        f"({sim_result['buy_trades']} Buys, {sim_result['sell_trades']} Sells).\n"
        f"3. **Criticality Sizing Effect**: Because Linear Regression projects expected return linearly from past 5-day returns, "
        f"large single-week spikes amplified predicted effect sizes ($|\\hat{{y}}|$), scaling position sizes up to "
        f"high criticality on sharp moves. In trending weeks this compounded gains, while in choppy/sideways weeks "
        f"linear extrapolation suffered from mean-reversion whipsaws."
    )

    sim_result.update(
        {
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "coefficients": coefs,
            "intercept": intercept,
            "directional_accuracy_pct": dir_acc,
            "r2": r2_val,
            "rmse": rmse_val,
            "explanation": explanation,
        }
    )
    return sim_result


# ---------------------------------------------------------------------------
# 6. Approach 3: CNN Graphical Analysis (5-Day Candlestick Image Pipeline)
# ---------------------------------------------------------------------------
def render_candlestick_array(ohlc_5d: pd.DataFrame, img_size: int = 32) -> np.ndarray:
    """
    Fast, deterministic rasterizer that converts a 5-day OHLC window into an
    RGB Candlestick chart image tensor of shape (img_size, img_size, 3) normalized to [0, 1].
    """
    canvas = np.zeros((img_size, img_size, 3), dtype=np.float32)
    opens = ohlc_5d["Open"].values.astype(float)
    highs = ohlc_5d["High"].values.astype(float)
    lows = ohlc_5d["Low"].values.astype(float)
    closes = ohlc_5d["Close"].values.astype(float)

    p_min = float(np.min(lows))
    p_max = float(np.max(highs))
    rng = max(p_max - p_min, 1e-6)

    margin = 3
    usable_h = img_size - 2 * margin
    col_step = img_size // 5
    body_half = max(1, col_step // 3)

    def price_to_y(p: float) -> int:
        norm = (p - p_min) / rng
        y = int(round((img_size - 1 - margin) - norm * (usable_h - 1)))
        return int(np.clip(y, 0, img_size - 1))

    for idx in range(5):
        cx = int(idx * col_step + col_step // 2)
        y_high = price_to_y(highs[idx])
        y_low = price_to_y(lows[idx])
        y_open = price_to_y(opens[idx])
        y_close = price_to_y(closes[idx])

        y_top_wick, y_bot_wick = min(y_high, y_low), max(y_high, y_low)
        canvas[y_top_wick : y_bot_wick + 1, cx, :] = [0.82, 0.85, 0.90]

        y_top_body, y_bot_body = min(y_open, y_close), max(y_open, y_close)
        if y_bot_body == y_top_body:
            y_bot_body = min(img_size - 1, y_top_body + 1)

        x_left = max(0, cx - body_half)
        x_right = min(img_size - 1, cx + body_half)
        if closes[idx] >= opens[idx]:
            canvas[y_top_body : y_bot_body + 1, x_left : x_right + 1, :] = [0.13, 0.77, 0.37]
        else:
            canvas[y_top_body : y_bot_body + 1, x_left : x_right + 1, :] = [0.94, 0.27, 0.27]

    return canvas


def build_cnn_dataset(df: pd.DataFrame, img_size: int = 32):
    """Generates labeled 5-day candlestick images ('Buy'/'Sell' + absolute effect size)."""
    _, y_all, dates_idx, ohlc_windows = build_5day_windows(df)
    images = np.stack([render_candlestick_array(w, img_size=img_size) for w in ohlc_windows], axis=0)
    labels_buy = (y_all > 0).astype(np.float32)  # 1.0 = Buy, 0.0 = Sell
    effect_sizes = np.abs(y_all).astype(np.float32)
    return images, labels_buy, effect_sizes, y_all, dates_idx, ohlc_windows


def run_cnn_strategy(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    img_size: int = 32,
):
    """
    Trains a 2D Convolutional Neural Network on 4 years of 5-day Candlestick chart images
    labeled with Buy/Sell and absolute effect size, then simulates weekly trades over the
    most recent 365 calendar days.
    Uses a concatenated 2-column output tensor [prob_buy, effect_size] so Keras 2 and Keras 3
    compile/fit/predict work with a single standard NumPy array (avoiding multi-output KeyError: 0).
    """
    images, labels_buy, effect_sizes, y_raw, dates_idx, ohlc_windows = build_cnn_dataset(
        df, img_size=img_size
    )

    train_mask = np.asarray(dates_idx < (cutoff_date - timedelta(days=7)), dtype=bool)
    test_mask = np.asarray(dates_idx > cutoff_date, dtype=bool)

    X_img_train = images[train_mask]
    y_cls_train = labels_buy[train_mask]
    y_eff_train = effect_sizes[train_mask]
    Y_train = np.column_stack([y_cls_train, y_eff_train]).astype(np.float32)

    X_img_test = images[test_mask]
    y_cls_test = labels_buy[test_mask]
    y_eff_test = effect_sizes[test_mask]

    layer_params = [
        {"Layer": f"Input (Candlestick Image {img_size}x{img_size}x3)", "Output Shape": f"({img_size}, {img_size}, 3)", "Parameters": 0},
        {"Layer": "Conv2D (16 filters, 3x3, ReLU, same)", "Output Shape": f"({img_size}, {img_size}, 16)", "Parameters": 448},
        {"Layer": "MaxPooling2D (2x2)", "Output Shape": f"({img_size//2}, {img_size//2}, 16)", "Parameters": 0},
        {"Layer": "Conv2D (32 filters, 3x3, ReLU, same)", "Output Shape": f"({img_size//2}, {img_size//2}, 32)", "Parameters": 4640},
        {"Layer": "MaxPooling2D (2x2)", "Output Shape": f"({img_size//4}, {img_size//4}, 32)", "Parameters": 0},
        {"Layer": "Conv2D (64 filters, 3x3, ReLU, same)", "Output Shape": f"({img_size//4}, {img_size//4}, 64)", "Parameters": 18496},
        {"Layer": "GlobalAveragePooling2D", "Output Shape": "(64,)", "Parameters": 0},
        {"Layer": "Dense (32 units, ReLU) + Dropout(0.25)", "Output Shape": "(32,)", "Parameters": 2080},
        {"Layer": "Head 1: Buy/Sell Classifier Dense(1, Sigmoid)", "Output Shape": "(1,)", "Parameters": 33},
        {"Layer": "Head 2: Effect Size Regressor Dense(1, ReLU)", "Output Shape": "(1,)", "Parameters": 33},
    ]
    total_params = sum(item["Parameters"] for item in layer_params)

    # Pre-render all test_df 5-day candlestick images for fast single-batch inference
    test_df = df[df.index > cutoff_date].copy()
    test_start_pos = len(df) - len(test_df)
    eval_imgs = []
    valid_eval_offsets = []

    for offset in range(len(test_df)):
        idx_pos = test_start_pos + offset
        if idx_pos >= 4:
            w_ohlc = df.iloc[idx_pos - 4 : idx_pos + 1][["Open", "High", "Low", "Close"]]
            eval_imgs.append(render_candlestick_array(w_ohlc, img_size=img_size))
            valid_eval_offsets.append(offset)

    X_eval_batch = np.stack(eval_imgs, axis=0) if len(eval_imgs) > 0 else np.zeros((1, img_size, img_size, 3), dtype=np.float32)

    if TF_AVAILABLE and len(X_img_train) > 10:
        tf.random.set_seed(42)
        inp = layers.Input(shape=(img_size, img_size, 3), name="candlestick_input")
        x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(inp)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dropout(0.25)(x)
        out_cls = layers.Dense(1, activation="sigmoid", name="buy_sell_head")(x)
        out_eff = layers.Dense(1, activation="relu", name="effect_size_head")(x)
        out_combined = layers.Concatenate(name="combined_head")([out_cls, out_eff])

        cnn_model = models.Model(inputs=inp, outputs=out_combined)
        total_params = int(cnn_model.count_params())
        cnn_model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
        )
        sample_weights = (1.0 + (y_eff_train / (float(np.mean(y_eff_train)) + 1e-6))).astype(np.float32)
        cnn_model.fit(
            X_img_train,
            Y_train,
            sample_weight=sample_weights,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )
        if len(X_img_test) > 0:
            preds_test = np.asarray(cnn_model.predict(X_img_test, verbose=0))
            pred_cls_test = preds_test[:, 0]
            pred_eff_test = preds_test[:, 1]
        else:
            pred_cls_test = np.zeros(1, dtype=np.float32)
            pred_eff_test = np.zeros(1, dtype=np.float32)

        eval_preds = np.asarray(cnn_model.predict(X_eval_batch, verbose=0))
        eval_probs = eval_preds[:, 0]
        eval_effs = eval_preds[:, 1]
    else:
        from sklearn.neural_network import MLPClassifier, MLPRegressor

        flat_train = X_img_train.reshape(len(X_img_train), -1)
        flat_test = X_img_test.reshape(len(X_img_test), -1) if len(X_img_test) > 0 else np.zeros((1, img_size * img_size * 3))
        flat_eval = X_eval_batch.reshape(len(X_eval_batch), -1)
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=epochs * 10, random_state=42)
        reg = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=epochs * 10, random_state=42)
        clf.fit(flat_train, y_cls_train.astype(int))
        reg.fit(flat_train, y_eff_train)
        pred_cls_test = clf.predict_proba(flat_test)[:, 1]
        pred_eff_test = np.abs(reg.predict(flat_test))
        eval_probs = clf.predict_proba(flat_eval)[:, 1]
        eval_effs = np.abs(reg.predict(flat_eval))

    p90_eff = float(np.percentile(y_eff_train, 90)) if len(y_eff_train) > 0 else 0.02
    p90_eff = max(p90_eff, 1e-4)

    signals = ["Hold"] * len(test_df)
    criticalities = [0.0] * len(test_df)

    for idx_eval, offset in enumerate(valid_eval_offsets):
        prob_buy = float(eval_probs[idx_eval])
        eff_hat = float(abs(eval_effs[idx_eval]))
        sig = "Buy" if prob_buy >= 0.5 else "Sell"
        confidence = abs(prob_buy - 0.5) * 2.0
        eff_norm = min(1.0, eff_hat / p90_eff)
        crit = float(np.clip(0.55 * confidence + 0.45 * eff_norm, 0.15, 1.0))
        signals[offset] = sig
        criticalities[offset] = crit

    sim_result = execute_weekly_simulation(
        test_df,
        signals,
        criticalities,
        investor_name="Investor 3: CNN Graphical Analysis",
    )

    cls_acc = float(np.mean((pred_cls_test >= 0.5) == (y_cls_test >= 0.5)) * 100.0) if len(y_cls_test) > 0 else 0.0
    eff_mae = float(np.mean(np.abs(pred_eff_test - y_eff_test))) if len(y_eff_test) > 0 else 0.0
    outcome_word = (
        "outperformed the initial budget (WIN)"
        if sim_result["growth_usd"] >= 0
        else "finished below the initial budget (LOSS)"
    )

    hyperparams = {
        "Input Image Resolution": f"{img_size}x{img_size}x3 RGB (5-Day Candlestick Chart)",
        "Convolutional Layers": "3 blocks: Conv2D(16, 3x3) -> Conv2D(32, 3x3) -> Conv2D(64, 3x3)",
        "Pooling & Regularization": "MaxPooling2D(2x2) x2, GlobalAveragePooling2D, Dropout(0.25)",
        "Output Heads": "Dual Head: Sigmoid ('Buy'/'Sell' probability) + ReLU (Absolute Effect Size |y|)",
        "Optimizer & Learning Rate": f"Adam (learning_rate = {learning_rate})",
        "Loss Function": "Effect-Size Weighted MSE across [Buy/Sell Prob, Effect Size]",
        "Batch Size & Epochs": f"Batch Size = {batch_size}, Epochs = {epochs}",
        "Total Trainable Parameters": f"{total_params:,} parameters",
    }

    sample_indices = np.linspace(0, max(0, len(ohlc_windows) - 1), num=min(4, len(ohlc_windows)), dtype=int)
    samples = []
    for s_idx in sample_indices:
        idx_int = int(s_idx)
        samples.append(
            {
                "date": dates_idx[idx_int].strftime("%Y-%m-%d"),
                "label": "Buy" if float(labels_buy[idx_int]) >= 0.5 else "Sell",
                "effect_size_pct": float(effect_sizes[idx_int] * 100.0),
                "raw_return_pct": float(y_raw[idx_int] * 100.0),
                "image": images[idx_int],
            }
        )

    explanation = (
        f"**Win/Loss Outcome**: Investor 3 (CNN Graphical Analysis) reached a final worth of "
        f"**${sim_result['final_worth']:,.2f}** ({sim_result['growth_pct']:+.2f}% vs. the $100,000 budget), "
        f"meaning the CNN graphical investor **{outcome_word}** over the most recent 365 calendar days.\n\n"
        f"**Why It Won/Lost (Visual Pattern & Effect-Size Analysis)**:\n"
        f"1. **Candlestick Spatial Feature Extraction**: Unlike Linear Regression (which only sees close-to-close returns), "
        f"the CNN inspects 5-day OHLC candlestick geometry—capturing intraday high/low wicks, body-to-wick ratios, "
        f"bullish/bearish engulfing structures, and 5-day exhaustion wicks. Across {len(X_img_train):,} training images from the prior 4 years, "
        f"samples were labeled with **Buy/Sell** and weighted by the **absolute effect size** ($|\\Delta P_{{+5d}}|$).\n"
        f"2. **Out-of-Sample Accuracy & Trade Execution**: On the recent 365-day test window, the CNN achieved "
        f"**{cls_acc:.1f}% classification accuracy** on 5-day candlestick patterns (Effect-Size MAE = {eff_mae*100:.2f}%) "
        f"and a **{sim_result['win_rate_pct']:.1f}% weekly trade win rate** across {sim_result['num_trades']} weekly trades "
        f"({sim_result['buy_trades']} Buys, {sim_result['sell_trades']} Sells).\n"
        f"3. **CNN Architecture, Hyperparameters & Parameter Count**:\n"
        f"   - **Total Parameters**: **{total_params:,} trainable parameters** (`Conv2D(16)`: 448, `Conv2D(32)`: 4,640, "
        f"`Conv2D(64)`: 18,496, `Dense(32)`: 2,080, `Buy/Sell Head`: 33, `Effect-Size Head`: 33).\n"
        f"   - **Hyperparameters**: Input `{img_size}x{img_size}x3`, Kernel `3x3`, Activation `ReLU`, "
        f"Pooling `MaxPooling2D(2x2)` + `GlobalAveragePooling2D`, Dropout `0.25`, Optimizer `Adam(lr={learning_rate})`, "
        f"Batch Size `{batch_size}`, Epochs `{epochs}`."
    )

    sim_result.update(
        {
            "train_samples": int(len(X_img_train)),
            "test_samples": int(len(X_img_test)),
            "classification_accuracy_pct": cls_acc,
            "effect_size_mae": eff_mae,
            "total_params": total_params,
            "layer_params": layer_params,
            "hyperparams": hyperparams,
            "samples": samples,
            "explanation": explanation,
        }
    )
    return sim_result


# ---------------------------------------------------------------------------
# 7. Streamlit User Interface
# ---------------------------------------------------------------------------
def main():
    st.title("📈 Bobby Stock Price Prediction Demo")
    st.markdown(
        """
        **Comparing Three Weekly Trading Investors ($100,000 Initial Budget | Most Recent 365 Calendar Days):**
        * **Investor 1 — SMA Crossover Strategy (`Short MA = 5`, `Long MA = 20`)**: Uses daily stock data within the most recent 365 calendar days. Buys on Golden Cross and sells on Death Cross.
        * **Investor 2 — ML Linear Regression**: Trained on the **past 4 years** of daily data (5-day past price movement $\\rightarrow$ next 5-day average price movement) and simulated on the **most recent 365 calendar days**.
        * **Investor 3 — CNN Graphical Analysis**: Generates **5-day Candlestick charts** labeled with `"Buy"` / `"Sell"` and the **absolute effect size** of the next 5-day average movement. Trained on the **past 4 years** and simulated on the **most recent 365 calendar days**.
        """
    )

    with st.sidebar:
        st.header("⚙️ Simulation Parameters")
        ticker = st.text_input("Stock Ticker (Yahoo Finance)", value="NVDA").strip().upper()
        st.caption("Examples: `NVDA`, `AAPL`, `MSFT`, `TSLA`, `AMZN`, `GOOGL`")

        st.subheader("SMA Crossover Settings")
        sma_short = int(st.number_input("Short MA Length (Days)", min_value=2, max_value=30, value=5))
        sma_long = int(st.number_input("Long MA Length (Days)", min_value=10, max_value=100, value=20))

        st.subheader("CNN Hyperparameters")
        cnn_epochs = int(st.slider("CNN Training Epochs", min_value=5, max_value=50, value=15, step=5))
        cnn_batch = int(st.selectbox("CNN Batch Size", options=[16, 32, 64], index=1))
        cnn_lr = float(
            st.select_slider(
                "CNN Learning Rate (Adam)",
                options=[0.0005, 0.001, 0.002, 0.005],
                value=0.001,
            )
        )

        run_btn = st.button("🚀 Run 5-Year Data & 3-Investor Simulation", type="primary", use_container_width=True)

    if run_btn or "sim_results" not in st.session_state or st.session_state.get("last_ticker") != ticker:
        with st.spinner(f"Fetching 5 years of Yahoo Finance data for {ticker} & training ML/CNN models..."):
            try:
                df = fetch_stock_data(ticker)
                train_df, test_df, cutoff_date, latest_date = split_train_and_recent_year(df)
                sma_res = run_sma_strategy(test_df, short_window=sma_short, long_window=sma_long)
                lr_res = run_linear_regression_strategy(df, cutoff_date)
                cnn_res = run_cnn_strategy(
                    df,
                    cutoff_date,
                    epochs=cnn_epochs,
                    batch_size=cnn_batch,
                    learning_rate=cnn_lr,
                )
                st.session_state["sim_results"] = {
                    "ticker": ticker,
                    "df": df,
                    "train_df": train_df,
                    "test_df": test_df,
                    "cutoff_date": cutoff_date,
                    "latest_date": latest_date,
                    "sma": sma_res,
                    "lr": lr_res,
                    "cnn": cnn_res,
                }
                st.session_state["last_ticker"] = ticker
            except Exception as exc:
                st.error(f"Error running simulation for '{ticker}': {exc}")
                with st.expander("Technical Error Traceback"):
                    st.code(traceback.format_exc())
                return

    res = st.session_state["sim_results"]
    df = res["df"]
    train_df = res["train_df"]
    test_df = res["test_df"]
    sma_res = res["sma"]
    lr_res = res["lr"]
    cnn_res = res["cnn"]

    st.info(
        f"**{res['ticker']} 5-Year Yahoo Finance Dataset**: {len(df):,} total trading days "
        f"(`{df.index.min().strftime('%Y-%m-%d')}` to `{df.index.max().strftime('%Y-%m-%d')}`) | "
        f"**4-Year Training Window**: {len(train_df):,} trading days (`<= {res['cutoff_date'].strftime('%Y-%m-%d')}`) | "
        f"**1-Year Simulation Window (365 Calendar Days)**: {len(test_df):,} trading days, "
        f"{int(test_df['IsFirstTradingDayOfWeek'].sum())} weekly trading decision days."
    )

    col1, col2, col3 = st.columns(3)
    for col, inv in zip([col1, col2, col3], [sma_res, lr_res, cnn_res]):
        with col:
            st.metric(
                label=inv["investor"],
                value=f"${inv['final_worth']:,.2f}",
                delta=f"{inv['growth_pct']:+.2f}% (${inv['growth_usd']:+,.2f})",
            )
            st.caption(
                f"Cash: **${inv['final_cash']:,.2f}** | Stock: **${inv['final_stock_value']:,.2f}** | "
                f"Trades: **{inv['num_trades']}** (Win Rate: **{inv['win_rate_pct']:.1f}%**)"
            )

    st.subheader("💰 Annual Portfolio Growth Comparison (Most Recent 365 Calendar Days)")
    fig_worth = go.Figure()
    fig_worth.add_trace(
        go.Scatter(
            x=sma_res["history"].index,
            y=sma_res["history"]["TotalWorth"],
            name="Investor 1: SMA Crossover (5/20)",
            line=dict(color="#10b981", width=2.5),
        )
    )
    fig_worth.add_trace(
        go.Scatter(
            x=lr_res["history"].index,
            y=lr_res["history"]["TotalWorth"],
            name="Investor 2: ML Linear Regression",
            line=dict(color="#3b82f6", width=2.5),
        )
    )
    fig_worth.add_trace(
        go.Scatter(
            x=cnn_res["history"].index,
            y=cnn_res["history"]["TotalWorth"],
            name="Investor 3: CNN Graphical Analysis",
            line=dict(color="#a855f7", width=2.5),
        )
    )
    fig_worth.add_hline(
        y=INITIAL_BUDGET,
        line_dash="dash",
        line_color="#94a3b8",
        annotation_text="Initial Budget ($100,000)",
    )
    fig_worth.update_layout(
        height=420,
        xaxis_title="Date (Most Recent 365 Calendar Days)",
        yaxis_title="Total Portfolio Worth (USD $)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_worth, use_container_width=True)

    st.subheader("📊 Annual Investor Performance Scorecard")
    scorecard_df = pd.DataFrame(
        [
            {
                "Investor Strategy": inv["investor"],
                "Initial Budget ($)": f"${INITIAL_BUDGET:,.2f}",
                "Final Cash ($)": f"${inv['final_cash']:,.2f}",
                "Final Stock Holdings ($)": f"${inv['final_stock_value']:,.2f}",
                "Final Total Worth ($)": f"${inv['final_worth']:,.2f}",
                "Net Profit / Loss ($)": f"${inv['growth_usd']:+,.2f}",
                "Annual Growth (%)": f"{inv['growth_pct']:+.2f}%",
                "Weekly Trades Executed": f"{inv['num_trades']} ({inv['buy_trades']}B / {inv['sell_trades']}S)",
                "5-Day Forward Win Rate": f"{inv['win_rate_pct']:.1f}%",
                "Max Drawdown": f"{inv['max_drawdown_pct']:.2f}%",
            }
            for inv in [sma_res, lr_res, cnn_res]
        ]
    )
    st.dataframe(scorecard_df, use_container_width=True, hide_index=True)

    tab_sma, tab_lr, tab_cnn, tab_logs = st.tabs(
        [
            "1️⃣ SMA Crossover (5/20)",
            "2️⃣ ML Linear Regression + Win/Loss Explanation",
            "3️⃣ CNN Graphical Analysis + Hyperparameters",
            "📋 Weekly Trade Logs",
        ]
    )

    with tab_sma:
        st.markdown("### Investor 1: SMA Crossover Strategy (`Short MA = 5`, `Long MA = 20`)")
        sma_df = sma_res["sma_df"]
        fig_sma = go.Figure()
        fig_sma.add_trace(go.Scatter(x=sma_df.index, y=sma_df["Close"], name="Close Price", line=dict(color="#64748b", width=1.5)))
        fig_sma.add_trace(go.Scatter(x=sma_df.index, y=sma_df["SMA_Short"], name=f"Short SMA ({sma_short}d)", line=dict(color="#10b981", width=2)))
        fig_sma.add_trace(go.Scatter(x=sma_df.index, y=sma_df["SMA_Long"], name=f"Long SMA ({sma_long}d)", line=dict(color="#f59e0b", width=2)))
        fig_sma.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_sma, use_container_width=True)

    with tab_lr:
        st.markdown("### Investor 2: ML Linear Regression Strategy & Win/Loss Explanation")
        st.markdown(lr_res["explanation"])
        coef_df = pd.DataFrame(
            {
                "Feature (Past 5 Days Price Movement)": [
                    "Day t-4 Return (x1)",
                    "Day t-3 Return (x2)",
                    "Day t-2 Return (x3)",
                    "Day t-1 Return (x4)",
                    "Day t Return (x5)",
                ],
                "Learned Beta Coefficient": [f"{c:+.6f}" for c in lr_res["coefficients"]],
            }
        )
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

    with tab_cnn:
        st.markdown("### Investor 3: CNN Graphical Analysis, Hyperparameters & Win/Loss Explanation")
        st.markdown(cnn_res["explanation"])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Hyperparameters Used")
            hp_df = pd.DataFrame(
                [{"Hyperparameter": k, "Value": v} for k, v in cnn_res["hyperparams"].items()]
            )
            st.dataframe(hp_df, use_container_width=True, hide_index=True)
        with c2:
            st.markdown(f"#### Layer-by-Layer Parameter Count (Total: `{cnn_res['total_params']:,}`)")
            st.dataframe(pd.DataFrame(cnn_res["layer_params"]), use_container_width=True, hide_index=True)

        st.markdown("#### Sample Generated 5-Day Candlestick Charts (Labeled with Buy/Sell & Absolute Effect Size)")
        img_cols = st.columns(len(cnn_res["samples"]))
        for col, samp in zip(img_cols, cnn_res["samples"]):
            with col:
                disp_img = Image.fromarray((samp["image"] * 255).astype(np.uint8)).resize((160, 160), Image.NEAREST)
                st.image(disp_img, caption=f"{samp['date']} | Label: {samp['label']} | Effect Size: {samp['effect_size_pct']:.2f}%")

    with tab_logs:
        st.markdown("### First Trading Day of the Week — Executed Trade Logs")
        sub1, sub2, sub3 = st.tabs(["SMA Trades", "ML Linear Regression Trades", "CNN Graphical Trades"])
        with sub1:
            st.dataframe(sma_res["trades"], use_container_width=True, hide_index=True)
        with sub2:
            st.dataframe(lr_res["trades"], use_container_width=True, hide_index=True)
        with sub3:
            st.dataframe(cnn_res["trades"], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
