import requests

url = "http://127.0.0.1:5001/predict"

payload = {
    "imdb_rating": 8.4,
    "watch_time_mins": 115,
    "genre_match_score": 85
}

response = requests.post(url, json=payload)
print("Status Code:", response.status_code)
print("API Response:", response.json())