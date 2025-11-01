import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU (Render doesn’t have one)

import pandas as pd
import numpy as np
import yfinance as yf
import io
import base64
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Path to ticker list CSV
TICKER_PATH = "INDIAN_STOCK_TICKERS.csv"

# ✅ Load tickers once
try:
    tickers_df = pd.read_csv(TICKER_PATH)
    TICKERS = tickers_df["SYMBOL"].dropna().unique().tolist()
    print(f"✅ Loaded {len(TICKERS)} tickers successfully.")
except Exception as e:
    print("⚠️ Could not load ticker list:", e)
    TICKERS = []


def predict_stock(user_input, forecast_days=30):
    """
    Predicts stock price trends using a lightweight regression model.
    Returns HTML <img> tag and forecast dataframe.
    """

    # Match ticker
    match = tickers_df[
        (tickers_df["SYMBOL"].astype(str).str.upper() == user_input.upper()) |
        (tickers_df["COMPANY_NAME"].str.contains(user_input, case=False, na=False))
    ]

    if match.empty:
        return "❌ Stock not found.", None

    ticker = match["SYMBOL"].iloc[0]

    # ✅ Fetch 5 years of data
    df = yf.download(ticker, period="5y")
    if df.empty:
        return f"❌ No data found for {ticker}.", None

    df = df[["Close"]].dropna().reset_index()

    # Prepare data for regression
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["Close"].values.reshape(-1, 1)

    # ✅ Train lightweight model
    model = LinearRegression()
    model.fit(X, y)

    # Predict next N days
    future_X = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
    preds = model.predict(future_X)

    forecast_df = pd.DataFrame({
        "Date": pd.date_range(df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days),
        "Predicted Close": preds.flatten()
    })

    # ✅ Plot historical + forecast
    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], df["Close"], label="Historical", linewidth=2)
    plt.plot(forecast_df["Date"], forecast_df["Predicted Close"], label="Forecast", linestyle="dashed", color="orange")
    plt.title(f"{ticker} — 30-Day Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price (₹)")
    plt.legend()
    plt.grid(True)

    # Convert plot to base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    img_html = f'<img src="data:image/png;base64,{img_base64}" alt="Stock forecast" width="700"/>'

    return img_html, forecast_df
