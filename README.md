1. Project Overview

This project demonstrates a full MLOps lifecycle:

Data extraction and preprocessing

Training a Random Forest model

Tracking metrics & artifacts with MLflow

Versioning data using DVC

CI automation via GitHub Workflows
mlops-mlflow-assignment/
│
├── data/
│   └── raw_data.csv
│
├── models/
│   └── random_forest_model.pkl
│
├── matrics
     └── matrics.json
├── src/
│   ├── fetch_boston.py
│   ├── preprocess.py
│   ├── train_model.py
│   └── mlflow_pipeline.py
│
├── .github/workflows/
│   └── ci.yml
│
└── README.md
