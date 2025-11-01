from flask import Flask, render_template, request
from stock_predictor import predict_stock, TICKERS

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.form.get("ticker", "").strip()
    if not user_input:
        return render_template_string(HTML_TEMPLATE, result="❌ Please enter a stock name or symbol.")

    try:
        result, forecast_df = predict_stock(user_input)
        if result is None or forecast_df is None or forecast_df.empty:
            return render_template_string(
                HTML_TEMPLATE,
                result=f"⚠️ Could not fetch or predict data for '{user_input}'. It may be invalid or too large."
            )
        return render_template_string(HTML_TEMPLATE, result=result, forecast=forecast_df)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, result=f"❌ Server error: {str(e)}")
