import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

sample_movies = pd.DataFrame([
    {"imdb_rating": 8.4, "watch_time_mins": 115, "genre_match_score": 85},
    {"imdb_rating": 4.2, "watch_time_mins": 30, "genre_match_score": 25}
])

predictions = model.predict(sample_movies)
probabilities = model.predict_proba(sample_movies)

for i, row in sample_movies.iterrows():
    conf = max(probabilities[i]) * 100
    print(f"Movie {i+1} (Rating: {row['imdb_rating']}, Watch: {row['watch_time_mins']}m, Match: {row['genre_match_score']}%) "
          f"-> {predictions[i]} ({conf:.1f}% confidence)")