import pandas as pd
import numpy as np
import joblib
import glob

from sklearn.metrics import classification_report, confusion_matrix

# Paths
CIC_FOLDER = "data/raw/CIC/MERGED_CSV/*.csv"
MODEL_PATH = "models/rf_model.pkl"
SCALER_PATH = "models/scaler.pkl"


def load_cic_data():
    print("[INFO] Loading CIC dataset...")

    files = glob.glob(CIC_FOLDER)

    df_list = []
    for file in files:
        try:
            temp = pd.read_csv(file, low_memory=False)
            df_list.append(temp)
        except:
            print(f"[WARNING] Skipping {file}")

    df = pd.concat(df_list, ignore_index=True)

    print(f"[INFO] CIC dataset shape: {df.shape}")
    return df


def clean_data(df):
    print("[INFO] Cleaning CIC data...")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    print(f"[INFO] Shape after cleaning: {df.shape}")
    return df


def prepare_features(df):
    print("[INFO] Preparing features...")

    # Try to find label column automatically
    possible_labels = ["label", "Label", "attack", "Attack"]

    label_col = None
    for col in possible_labels:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        raise Exception("No label column found in CIC dataset")

    y = df[label_col]

    # Convert labels to binary
    y = y.apply(lambda x: 0 if "benign" in str(x).lower() else 1)

    # Drop non-numeric + label
    X = df.drop(columns=[label_col])
    X = X.select_dtypes(include=[np.number])

    return X, y


def align_features(X, model):
    print("[INFO] Aligning features with training data...")

    expected_features = model.n_features_in_

    if X.shape[1] > expected_features:
        X = X.iloc[:, :expected_features]
    elif X.shape[1] < expected_features:
        diff = expected_features - X.shape[1]
        for i in range(diff):
            X[f"missing_{i}"] = 0

    return X


def scale_data(X):
    print("[INFO] Scaling data...")

    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X)

    return X_scaled


def main():
    # Load model
    print("[INFO] Loading trained model...")
    model = joblib.load(MODEL_PATH)

    # Load CIC data
    df = load_cic_data()

    df = clean_data(df)

    X, y = prepare_features(df)

    X = align_features(X, model)

    X_scaled = scale_data(X)

    print("[INFO] Running predictions...")

    y_pred = model.predict(X_scaled)

    print("\n===== CROSS DATASET RESULTS =====")
    print(classification_report(y, y_pred))

    print("\n===== CONFUSION MATRIX =====")
    print(confusion_matrix(y, y_pred))


if __name__ == "__main__":
    main()
