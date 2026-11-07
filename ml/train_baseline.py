import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    roc_auc_score
)

DATA_DIR = Path(__file__).parent / "data"

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "val.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

FEATURES = [
    "amount",
    "step",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

X_train = train_df[FEATURES]
y_train = train_df["isFraud"]

X_test = test_df[FEATURES]
y_test = test_df["isFraud"]

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest...")

model.fit(X_train, y_train)

# Fraud probabilities
probabilities = model.predict_proba(X_test)[:, 1]

# Default classification threshold
predictions = (probabilities >= 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("PR-AUC:", average_precision_score(y_test, probabilities))
print("ROC-AUC:", roc_auc_score(y_test, probabilities))

joblib.dump(
    model,
    DATA_DIR / "random_forest_model.pkl"
)

print("\nRandom Forest model saved.")