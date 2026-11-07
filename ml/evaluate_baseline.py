import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "ml" / "data" / "test.csv"
MODEL_PATH = ROOT / "ml" / "data" / "random_forest_model.pkl"
OUTPUT_PATH = ROOT / "ml" / "data" / "baseline_metrics.json"

FEATURES = [
    "amount",
    "step",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

test = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

X_test = test[FEATURES]
y_test = test["isFraud"]
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

metrics = {
    "model": "Random Forest",
    "threshold": 0.5,
    "test_rows": int(len(test)),
    "fraud_cases": int(y_test.sum()),
    "pr_auc": float(average_precision_score(y_test, probabilities)),
    "roc_auc": float(roc_auc_score(y_test, probabilities)),
    "fraud_precision": float(precision_score(y_test, predictions, zero_division=0)),
    "fraud_recall": float(recall_score(y_test, predictions, zero_division=0)),
    "fraud_f1": float(f1_score(y_test, predictions, zero_division=0)),
}

OUTPUT_PATH.write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
print(f"\nMetrics saved to: {OUTPUT_PATH}")