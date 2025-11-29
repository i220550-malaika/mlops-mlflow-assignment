from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import json

def train_model(X_train, y_train, model_path="models/random_forest_model.pkl"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    return model_path

def evaluate_model(model_path, X_test, y_test, metrics_path="metrics/metrics.json"):
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    metrics = {"MSE": mean_squared_error(y_test, y_pred),
               "R2": r2_score(y_test, y_pred)}
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    return metrics
