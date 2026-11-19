import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Movie_Recommendation_Production")

df = pd.read_csv("Dataset/movies_clean.csv")
X = df[["imdb_rating", "watch_time_mins", "genre_match_score"]]
y = df["recommendation"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="Production_RandomForest"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")

    joblib.dump(model, "models/model.pkl")
    print(f"Model trained (Accuracy: {acc * 100:.2f}%) and saved to models/model.pkl")