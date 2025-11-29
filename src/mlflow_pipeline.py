# src/mlflow_pipeline.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import json

# -------------------
# Step 1: Data Extraction
# -------------------
def extract_data(local_path="data/raw_data.csv"):
    print(f"Loading data from {local_path}...")
    df = pd.read_csv(local_path)
    print(f"Data loaded, shape: {df.shape}")
    return df

# -------------------
# Step 2: Data Preprocessing
# -------------------
def preprocess_data(df, target_col="MEDV"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Preprocessing done.")
    return X_train, X_test, y_train, y_test

# -------------------
# Step 3: Model Training
# -------------------
def train_model(X_train, y_train, model_path="models/random_forest_model.pkl"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print(f"Model trained and saved to {model_path}")
    return model_path

# -------------------
# Step 4: Model Evaluation
# -------------------
def evaluate_model(model_path, X_test, y_test, metrics_path="metrics/metrics.json"):
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    metrics = {"MSE": mean_squared_error(y_test, y_pred), "R2": r2_score(y_test, y_pred)}
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    print(f"Metrics saved to {metrics_path}")
    return metrics

# -------------------
# Run the full pipeline
# -------------------
def run_pipeline():
    print("Running MLflow pipeline...")

    df = extract_data("data/raw_data.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df, target_col="MEDV")
    model_path = train_model(X_train, y_train)
    metrics = evaluate_model(model_path, X_test, y_test)

    print("Pipeline finished!")
    print("Evaluation Metrics:", metrics)

if __name__ == "__main__":
    run_pipeline()
