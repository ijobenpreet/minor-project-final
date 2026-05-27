import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

# ---------------- LOAD DATASET ----------------

df = pd.read_csv('../data/processed/nbaiot_clean.csv')

TARGET_COLUMN = 'label'

# Features and labels
X = df.drop(TARGET_COLUMN, axis=1)

# Match model feature count
X = X.iloc[:, :39]

y = df[TARGET_COLUMN]

# ---------------- SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- LOAD MODEL ----------------

model = joblib.load('../models/cic_model.pkl')

# ---------------- PREDICT ----------------

preds = model.predict(X_test.values)

# ---------------- CONFUSION MATRIX ----------------

cm = confusion_matrix(y_test, preds)

# ---------------- PLOT ----------------

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Benign', 'Attack'],
    yticklabels=['Benign', 'Attack']
)

plt.title('Confusion Matrix — Random Forest')

plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Save figure
plt.savefig('confusion_matrix_rf.png', dpi=300)

plt.show()

print("Confusion Matrix saved as confusion_matrix_rf.png")