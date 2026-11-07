from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

results = pd.DataFrame([
    {
        "Model": "Random Forest",
        "Fraud Precision": 0.91,
        "Fraud Recall": 0.56,
        "Fraud F1": 0.70,
        "PR-AUC": 0.696495,
        "ROC-AUC": 0.955524,
    },
    {
        "Model": "GraphSAGE",
        "Fraud Precision": 0.75,
        "Fraud Recall": 0.07,
        "Fraud F1": 0.13,
        "PR-AUC": 0.099245,
        "ROC-AUC": 0.569340,
    },
])

results.to_csv(DATA_DIR / "model_comparison.csv", index=False)
print(results.to_string(index=False))
print(f"\nSaved: {DATA_DIR / 'model_comparison.csv'}")