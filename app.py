from flask import Flask, render_template, request, jsonify
from stock_predictor import predict_stock, TICKERS

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", tickers=TICKERS)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        symbol = request.form.get("symbol")
        days = int(request.form.get("days", 30))
        result_img, forecast_df = predict_stock(symbol, days)
        return jsonify({"symbol": symbol, "image": result_img})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
