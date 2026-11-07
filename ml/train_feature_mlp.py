"""Train a transaction-feature-only neural baseline on the shared data split."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    recall_score,
)
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path(__file__).parent / "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURES = [
    "amount",
    "step",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


class TransactionMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, features):
        return self.classifier(features)


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    scaler = StandardScaler()
    x_train = torch.tensor(
        scaler.fit_transform(train_df[FEATURES]), dtype=torch.float32, device=DEVICE
    )
    x_val = torch.tensor(
        scaler.transform(val_df[FEATURES]), dtype=torch.float32, device=DEVICE
    )
    x_test = torch.tensor(
        scaler.transform(test_df[FEATURES]), dtype=torch.float32, device=DEVICE
    )

    y_train = torch.tensor(
        train_df["isFraud"].to_numpy(), dtype=torch.long, device=DEVICE
    )
    y_val = val_df["isFraud"].to_numpy()
    y_test = test_df["isFraud"].to_numpy()

    class_counts = torch.bincount(y_train, minlength=2).float()
    fraud_weight = torch.sqrt(
        class_counts[0] / class_counts[1].clamp(min=1)
    ).clamp(max=20.0)
    class_weights = torch.tensor(
        [1.0, fraud_weight.item()], dtype=torch.float32, device=DEVICE
    )

    model = TransactionMLP(len(FEATURES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_pr_auc = -1.0
    best_state = None
    for epoch in range(1, 21):
        model.train()
        optimizer.zero_grad()
        train_output = model(x_train)
        loss = criterion(train_output, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_probabilities = torch.softmax(model(x_val), dim=1)[:, 1].cpu().numpy()

        val_pr_auc = average_precision_score(y_val, val_probabilities)
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_state = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }

        if epoch % 5 == 0:
            print(
                f"Epoch {epoch} | Loss: {loss.item():.4f} | "
                f"Validation PR-AUC: {val_pr_auc:.4f}"
            )

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_probabilities = torch.softmax(model(x_val), dim=1)[:, 1].cpu().numpy()
        test_probabilities = torch.softmax(model(x_test), dim=1)[:, 1].cpu().numpy()

    precision, recall, thresholds = precision_recall_curve(y_val, val_probabilities)
    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-8)
    )
    best_threshold = float(thresholds[int(np.argmax(f1_scores))])

    print(f"\nDevice: {DEVICE}")
    print(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"Test fraud cases: {int(y_test.sum())}")
    print(f"Test PR-AUC: {average_precision_score(y_test, test_probabilities):.4f}")
    print(
        "Test recall at 0.5: "
        f"{recall_score(y_test, test_probabilities >= 0.5, zero_division=0):.4f}"
    )
    print(f"Validation-selected threshold: {best_threshold:.6f}")
    print(
        "Test recall at validation-selected threshold: "
        f"{recall_score(y_test, test_probabilities >= best_threshold, zero_division=0):.4f}"
    )


if __name__ == "__main__":
    main()
