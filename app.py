from flask import Flask, render_template, request
from stock_predictor import predict_stock, TICKERS

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", tickers=TICKERS[:100])  # show 100 for dropdown

@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.form.get("ticker", "").strip()
    if not user_input:
        return render_template_string(HTML_TEMPLATE, result="❌ Please enter a stock name or symbol.")
    
    try:
        result, forecast_df = predict_stock(user_input)

        # ✅ if forecast_df is None, show a friendly message instead of crashing
        if forecast_df is None or forecast_df.empty:
            result += "<br>⚠️ Could not generate forecast — maybe the stock is delisted or data unavailable."
            return render_template_string(HTML_TEMPLATE, result=result)

        return render_template_string(HTML_TEMPLATE, result=result, forecast=forecast_df)

    except Exception as e:
        return render_template_string(HTML_TEMPLATE, result=f"❌ Error: {str(e)}")
