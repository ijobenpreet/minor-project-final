import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ---------------- PATHS ----------------
DATA_PATH = "../data/processed/nbaiot_combined.csv"
MODEL_PATH = "../models/cic_model.pkl"

# ---------------- LOAD DATA ----------------
print("[INFO] Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("[INFO] Columns:", df.columns)

# ---------------- PREPROCESS ----------------
df = df.select_dtypes(include=[np.number])

# Detect label column
if "label" in df.columns:
    y = df["label"]
    X = df.drop("label", axis=1)
elif "Label" in df.columns:
    y = df["Label"]
    X = df.drop("Label", axis=1)
elif "attack" in df.columns:
    y = df["attack"]
    X = df.drop("attack", axis=1)
else:
    print("[WARNING] Using last column as label")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

print("[INFO] Features shape:", X.shape)

# ---------------- CLASS DISTRIBUTION ----------------
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.savefig("class_distribution.png")
print("[INFO] Saved class_distribution.png")

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- LOAD MODEL ----------------
print("[INFO] Loading model...")
model = joblib.load(MODEL_PATH)

print("[INFO] Model type:", type(model))

# ---------------- ALIGN FEATURES ----------------
X_test_array = X_test.values
expected = model.n_features_in_

if X_test_array.shape[1] != expected:
    print("[WARNING] Feature mismatch, resizing...")
    X_test_array = np.resize(X_test_array, (X_test_array.shape[0], expected))

# ---------------- PREDICT ----------------
print("[INFO] Running predictions...")
y_pred = model.predict(X_test_array)

# ---------------- METRICS ----------------
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")
print("[INFO] Saved confusion_matrix.png")

# ---------------- FEATURE IMPORTANCE ----------------
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    feature_names = X.columns

    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10,5))
    sns.barplot(
        x=importances[indices][:10],
        y=feature_names[indices][:10]
    )

    plt.title("Top 10 Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")

    plt.savefig("feature_importance.png")
    print("[INFO] Saved feature_importance.png")

else:
    print("[WARNING] Model does not support feature importance")

plt.show()
