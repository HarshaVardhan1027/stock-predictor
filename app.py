from flask import Flask, render_template, request
from stock_predictor import predict_stock, TICKERS

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", tickers=TICKERS[:100])  # show 100 for dropdown

@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.form.get("ticker")
    result = predict_stock(user_input)

    if isinstance(result, tuple):
        img_tag, forecast_df = result
        forecast_table = forecast_df.to_html(classes="table table-striped", border=0)
        return render_template("result.html", img_tag=img_tag, table=forecast_table, name=user_input)
    else:
        return render_template("error.html", message=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
