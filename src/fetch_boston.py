from sklearn.datasets import fetch_openml
import pandas as pd
import os

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Download Boston Housing dataset
boston = fetch_openml(name="boston", version=1, as_frame=True)
df = boston.frame

# Save dataset to CSV
df.to_csv("data/raw_data.csv", index=False)

print("Saved data/raw_data.csv")
