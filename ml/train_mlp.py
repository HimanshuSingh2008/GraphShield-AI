import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import classification_report

DATA_DIR = Path(__file__).parent / "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURES = [
    "amount", "step", "oldbalanceOrg",
    "newbalanceOrig", "oldbalanceDest", "newbalanceDest"
]

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "val.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[FEATURES])
X_val = scaler.transform(val_df[FEATURES])
X_test = scaler.transform(test_df[FEATURES])

def tensor(x):
    return torch.tensor(x, dtype=torch.float32, device=DEVICE)

X_train = tensor(X_train)
X_val = tensor(X_val)
X_test = tensor(X_test)

y_train = torch.tensor(train_df["isFraud"].values, dtype=torch.long, device=DEVICE)
y_val = torch.tensor(val_df["isFraud"].values, dtype=torch.long, device=DEVICE)
y_test = torch.tensor(test_df["isFraud"].values, dtype=torch.long, device=DEVICE)

model = nn.Sequential(
    nn.Linear(6, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
).to(DEVICE)

counts = torch.bincount(y_train, minlength=2).float()
fraud_weight = torch.sqrt(counts[0] / counts[1].clamp(min=1)).clamp(max=20)
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor([1.0, fraud_weight.item()], device=DEVICE)
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_ap = -1
best_path = DATA_DIR / "mlp_model.pth"

for epoch in range(1, 31):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train)
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_prob = torch.softmax(model(X_val), dim=1)[:, 1].cpu().numpy()

    val_ap = average_precision_score(y_val.cpu().numpy(), val_prob)

    if val_ap > best_ap:
        best_ap = val_ap
        torch.save(model.state_dict(), best_path)

    if epoch % 5 == 0:
        print(f"Epoch {epoch} | Loss {loss.item():.4f} | Val PR-AUC {val_ap:.4f}")

model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
model.eval()

with torch.no_grad():
    test_prob = torch.softmax(model(X_test), dim=1)[:, 1].cpu().numpy()

print("Test PR-AUC:", average_precision_score(y_test.cpu().numpy(), test_prob))
print("Test ROC-AUC:", roc_auc_score(y_test.cpu().numpy(), test_prob))
print(classification_report(
    y_test.cpu().numpy(),
    (test_prob >= 0.5).astype(int),
    zero_division=0
))