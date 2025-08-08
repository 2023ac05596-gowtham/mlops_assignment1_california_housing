# src/preprocess_data.py

import pandas as pd
import os


def preprocess():
    raw_path = "data/raw/california_housing.csv"
    df = pd.read_csv(raw_path)

    # Example preprocessing
    df = df.dropna()
    df["MedHouseVal"] = df["MedHouseVal"] * 100000  # example transformation

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/california_housing.csv", index=False)


if __name__ == "__main__":
    preprocess()
