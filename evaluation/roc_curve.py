import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import joblib

# ---------------- LOAD DATASET ----------------

df = pd.read_csv('../data/processed/nbaiot_clean.csv')

TARGET_COLUMN = 'label'

# Features and labels
X = df.drop(TARGET_COLUMN, axis=1)

# Match model expected feature count
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

# ---------------- PREDICT PROBABILITY ----------------

probs = model.predict_proba(X_test.values)[:, 1]

# ---------------- ROC CURVE ----------------

fpr, tpr, thresholds = roc_curve(y_test, probs)

roc_auc = auc(fpr, tpr)

# ---------------- PLOT ----------------

plt.figure(figsize=(10, 7))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f'Random Forest (AUC = {roc_auc:.3f})'
)

# Random line
plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('ROC Curve — Random Forest')

plt.legend(loc='lower right')

plt.grid(True)

# Save figure
plt.savefig('roc_comparison.png', dpi=300)

plt.show()

print("ROC Curve saved as roc_comparison.png")