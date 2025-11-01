from flask import Flask, request, render_template
from stock_predictor import predict_stock

app = Flask(__name__)

@app.route("/")
def home():
    return '''
        <h2>📈 Stock Predictor</h2>
        <form method="post" action="/predict">
            <input type="text" name="symbol" placeholder="Enter stock name or symbol">
            <input type="number" name="days" placeholder="Forecast days" value="30">
            <button type="submit">Predict</button>
        </form>
    '''

@app.route("/predict", methods=["POST"])
def predict():
    symbol = request.form["symbol"]
    days = int(request.form["days"])
    result = predict_stock(symbol, days)
    if isinstance(result, str):
        return f"<p>{result}</p><a href='/'>Back</a>"
    else:
        img_tag, forecast_df = result
        return f"<h3>Forecast Complete!</h3>{img_tag}<br><a href='/'>Back</a>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
