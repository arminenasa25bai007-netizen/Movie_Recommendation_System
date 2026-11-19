from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("models/model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_web", methods=["POST"])
def predict_web():
    features = pd.DataFrame([{
        "imdb_rating": float(request.form["imdb_rating"]),
        "watch_time_mins": float(request.form["watch_time_mins"]),
        "genre_match_score": float(request.form["genre_match_score"])
    }])
    prediction = model.predict(features)[0]
    confidence = max(model.predict_proba(features)[0]) * 100
    return render_template(
        "index.html",
        prediction_text=f"Prediction: {prediction} ({confidence:.1f}% Confidence)"
    )

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    features = pd.DataFrame([{
        "imdb_rating": float(data["imdb_rating"]),
        "watch_time_mins": float(data["watch_time_mins"]),
        "genre_match_score": float(data["genre_match_score"])
    }])
    prediction = model.predict(features)[0]
    confidence = round(float(max(model.predict_proba(features)[0]) * 100), 2)
    return jsonify({
        "predicted_recommendation": prediction,
        "confidence_percent": confidence
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)