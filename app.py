from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model/scam_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]

    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]
    confidence = round(max(probabilities) * 100, 2)

    return render_template(
        "result.html",
        message=message,
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)