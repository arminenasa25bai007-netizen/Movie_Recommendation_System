import pandas as pd

df = pd.read_csv("Dataset/movie.csv")
df = df.dropna(subset=["recommendation"])
df["imdb_rating"] = df["imdb_rating"].fillna(df["imdb_rating"].mean().round(1))
df["watch_time_mins"] = df["watch_time_mins"].fillna(df["watch_time_mins"].mean().round(0))
df["genre_match_score"] = df["genre_match_score"].fillna(df["genre_match_score"].mean().round(0))

df.to_csv("Dataset/movies_clean.csv", index=False)
print("Cleaned dataset saved.")