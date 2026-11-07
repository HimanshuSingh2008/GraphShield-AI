import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

INPUT_FILE = DATA_DIR / "processed_transactions.csv"

df = pd.read_csv(INPUT_FILE)

# Sort transactions chronologically
df = df.sort_values("step").reset_index(drop=True)

# Define temporal boundaries
n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

# Save splits
train_df.to_csv(DATA_DIR / "train.csv", index=False)
val_df.to_csv(DATA_DIR / "val.csv", index=False)
test_df.to_csv(DATA_DIR / "test.csv", index=False)

print("Training transactions:", len(train_df))
print("Validation transactions:", len(val_df))
print("Testing transactions:", len(test_df))

print("\nFraud distribution:")

print("Train:", train_df["isFraud"].mean())
print("Validation:", val_df["isFraud"].mean())
print("Test:", test_df["isFraud"].mean())