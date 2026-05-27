import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Paths
DATA_PATH = "data/processed/nbaiot_clean.csv"
MODEL_PATH = "models/rf_model.pkl"


def load_data():
    print("[INFO] Loading processed dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Dataset shape: {df.shape}")
    return df


def split_data(df):
    print("[INFO] Splitting features and labels...")

    X = df.drop("label", axis=1)
    y = df["label"]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    print("[INFO] Training Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("[INFO] Training completed.")
    return model


def evaluate_model(model, X_test, y_test):
    print("[INFO] Evaluating model...")

    y_pred = model.predict(X_test)

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_test, y_pred))

    print("\n===== CONFUSION MATRIX =====")
    print(confusion_matrix(y_test, y_pred))


def save_model(model):
    print("[INFO] Saving model...")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"[INFO] Model saved at {MODEL_PATH}")


def main():
    df = load_data()

    X_train, X_test, y_train, y_test = split_data(df)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model)


if __name__ == "__main__":
    main()
