import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import io
import base64

# Path to ticker data
TICKER_PATH = "INDIAN_STOCK_TICKERS.csv"

def predict_stock(user_input, forecast_days=30):
    """
    Predicts stock prices for the given user input symbol or company name.
    Returns a base64 image tag (plot) and forecasted DataFrame.
    """
    try:
        tickers_df = pd.read_csv(TICKER_PATH)
    except Exception as e:
        return f"❌ Could not load ticker list: {e}"

    match = tickers_df[
        (tickers_df["SYMBOL"].astype(str).str.upper() == user_input.upper()) |
        (tickers_df["COMPANY_NAME"].str.contains(user_input, case=False, na=False))
    ]

    if match.empty:
        return "❌ Stock not found."

    ticker = match.iloc[0]["YF_TICKER"]
    name = match.iloc[0]["COMPANY_NAME"]

    # Fetch last 5 years of historical data
    df = yf.download(ticker, period="5y")
    if df.empty:
        return "⚠️ No data found."

    # Compute log returns for smoother training
    prices = df['Close'].values.reshape(-1, 1)
    log_returns = np.log(prices[1:] / prices[:-1])

    # Normalize data
    scaler = MinMaxScaler((0, 1))
    scaled = scaler.fit_transform(log_returns)

    # Create training sequences
    seq_len = 40
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Build and train LSTM model
    model = Sequential([
        LSTM(50, input_shape=(X.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=3, batch_size=64, verbose=0)

    # Forecast future values
    last_seq = scaled[-seq_len:]
    preds = []
    for _ in range(forecast_days):
        inp = last_seq.reshape(1, seq_len, 1)
        pred = model.predict(inp, verbose=0)[0, 0]
        preds.append(pred)
        last_seq = np.append(last_seq[1:], pred)

    # Convert back from log returns to prices
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    last_price = prices[-1, 0]
    forecast_prices = [last_price * np.exp(r) for r in preds]
    forecast_dates = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.Series(forecast_prices, index=forecast_dates, name="Predicted Price")

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(df['Close'], label="History")
    plt.plot(forecast_df, label="Forecast", color='red')
    plt.legend()
    plt.title(f"{name} ({ticker}) Prediction")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    img_tag = f'<img src="data:image/png;base64,{encoded}"/>'

    return img_tag, forecast_df


# ---- Load tickers when the module is imported ----
try:
    tickers_df = pd.read_csv(TICKER_PATH)
    TICKERS = tickers_df["SYMBOL"].astype(str).tolist()
    print(f"✅ Loaded {len(TICKERS)} stock tickers.")
except Exception as e:
    print("⚠️ Could not load tickers:", e)
    TICKERS = []
