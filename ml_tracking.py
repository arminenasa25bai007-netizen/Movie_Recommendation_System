import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Point MLflow to a local SQLite database file (mlflow.db)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Movie_Recommendation_Experiments")

df = pd.read_csv("Dataset/movies_clean.csv")
X = df[["imdb_rating", "watch_time_mins", "genre_match_score"]]
y = df["recommendation"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for n_trees in [50, 100]:
    with mlflow.start_run(run_name=f"RandomForest_{n_trees}_trees"):
        model = RandomForestClassifier(n_estimators=n_trees, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_param("n_estimators", n_trees)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
        print(f"Logged run with {n_trees} trees | Accuracy: {acc * 100:.2f}%")