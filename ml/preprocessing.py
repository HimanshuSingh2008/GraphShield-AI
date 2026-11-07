import pandas as pd
from pathlib import Path

# Project paths
DATA_DIR = Path(__file__).parent / "data"

INPUT_FILE = DATA_DIR / "PS_20174392719_1491204439457_log.csv"
OUTPUT_FILE = DATA_DIR / "processed_transactions.csv"

def preprocess_data():

    print("Loading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print("Original dataset shape:", df.shape)

    # Keep a smaller dataset during development.
    # We will use the full dataset later.
    df = df.sample(
        n=min(200_000, len(df)),
        random_state=42
    )

    # Remove unnecessary columns
    df = df.drop(
        columns=["isFlaggedFraud"],
        errors="ignore"
    )

    # Remove rows with missing values
    df = df.dropna()

    # Ensure amount is numeric
    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce"
    )

    df = df.dropna(subset=["amount"])

    # Keep transaction labels as integers
    df["isFraud"] = df["isFraud"].astype(int)

    # Reset index
    df = df.reset_index(drop=True)

    # Save processed data
    df.to_csv(OUTPUT_FILE, index=False)

    print("Processed dataset shape:", df.shape)
    print("Fraudulent transactions:", df["isFraud"].sum())
    print("Legitimate transactions:", (df["isFraud"] == 0).sum())

    print(f"Saved dataset to: {OUTPUT_FILE}")

if __name__ == "__main__":
    preprocess_data()