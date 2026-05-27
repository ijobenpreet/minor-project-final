import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Paths
INPUT_PATH = "data/processed/nbaiot_combined.csv"
OUTPUT_PATH = "data/processed/nbaiot_clean.csv"
SCALER_PATH = "models/scaler.pkl"


def load_data(path):
    print("[INFO] Loading dataset...")
    df = pd.read_csv(path, low_memory=False)
    print(f"[INFO] Loaded shape: {df.shape}")
    return df


def clean_data(df):
    print("[INFO] Cleaning data...")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Replace Inf values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop NaN rows
    df.dropna(inplace=True)

    print(f"[INFO] Shape after cleaning: {df.shape}")
    return df


def split_features_labels(df):
    print("[INFO] Splitting features and labels...")

    # Drop non-numeric columns
    if "device" in df.columns:
        df = df.drop(columns=["device"])

    X = df.drop("label", axis=1)
    y = df["label"]

    return X, y


def normalize_data(X):
    print("[INFO] Normalizing features...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    print("[INFO] Scaler saved.")

    return X_scaled


def save_clean_data(X, y):
    print("[INFO] Saving processed dataset...")

    df_final = pd.DataFrame(X)
    df_final["label"] = y.values

    df_final.to_csv(OUTPUT_PATH, index=False)

    print(f"[INFO] Saved to {OUTPUT_PATH}")


def main():
    df = load_data(INPUT_PATH)

    df = clean_data(df)

    X, y = split_features_labels(df)

    X_scaled = normalize_data(X)

    save_clean_data(X_scaled, y)


if __name__ == "__main__":
    main()
