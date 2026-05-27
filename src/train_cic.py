import pandas as pd
import numpy as np
import glob
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Paths
CIC_FOLDER = "data/raw/CIC/MERGED_CSV/*.csv"
MODEL_PATH = "models/cic_model.pkl"
SCALER_PATH = "models/cic_scaler.pkl"


def load_cic_data():
    print("[INFO] Loading CIC dataset...")

    files = glob.glob(CIC_FOLDER)
    df_list = [pd.read_csv(f, low_memory=False) for f in files]

    df = pd.concat(df_list, ignore_index=True)

    df = df.sample(n=200000, random_state=42)


    print(f"[INFO] Dataset shape: {df.shape}")
    return df


def clean_data(df):
    print("[INFO] Cleaning data...")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    print(f"[INFO] Shape after cleaning: {df.shape}")
    return df


def prepare_data(df):
    print("[INFO] Preparing features...")

    label_col = None
    for col in ["label", "Label", "attack"]:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        raise Exception("Label column not found")

    y = df[label_col].apply(lambda x: 0 if "benign" in str(x).lower() else 1)

    X = df.drop(columns=[label_col])
    X = X.select_dtypes(include=[np.number])

    return X, y


def scale_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, SCALER_PATH)

    return X_scaled


def train_model(X, y):
    print("[INFO] Training model...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n===== RESULTS =====")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    return model


def main():
    df = load_cic_data()

    df = clean_data(df)

    X, y = prepare_data(df)

    X_scaled = scale_data(X)

    model = train_model(X_scaled, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"[INFO] Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
