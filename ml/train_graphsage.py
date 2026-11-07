import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve
)
from torch_geometric.nn import SAGEConv

DATA_DIR = Path(__file__).parent / "data"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# ----------------------------------------
# 1. Load the shared chronological splits
# ----------------------------------------

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "val.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))

print("Fraud counts:")
print("Train:", int(train_df["isFraud"].sum()))
print("Validation:", int(val_df["isFraud"].sum()))
print("Test:", int(test_df["isFraud"].sum()))

# Use all splits only to assign consistent IDs to accounts.
df = pd.concat([train_df, val_df, test_df], ignore_index=True)


# ----------------------------------------
# 2. Create account IDs
# ----------------------------------------

accounts = pd.concat([
    df["nameOrig"],
    df["nameDest"]
]).unique()

account_to_id = {
    account: idx
    for idx, account in enumerate(accounts)
}

num_nodes = len(accounts)

for frame in [train_df, val_df, test_df]:

    frame["source_id"] = (
        frame["nameOrig"].map(account_to_id)
    )

    frame["target_id"] = (
        frame["nameDest"].map(account_to_id)
    )


# ----------------------------------------
# 3. Build training graph
# ----------------------------------------

edge_index = torch.tensor(
    [
        train_df["source_id"].values,
        train_df["target_id"].values
    ],
    dtype=torch.long
)

# Node features from training transactions only
outgoing_count = (
    train_df.groupby("source_id").size()
)

incoming_count = (
    train_df.groupby("target_id").size()
)

outgoing_amount = (
    train_df.groupby("source_id")["amount"].sum()
)

incoming_amount = (
    train_df.groupby("target_id")["amount"].sum()
)

node_features = pd.DataFrame(
    index=range(num_nodes)
)

node_features["outgoing_count"] = outgoing_count
node_features["incoming_count"] = incoming_count

node_features["outgoing_amount"] = outgoing_amount
node_features["incoming_amount"] = incoming_amount

node_features = node_features.fillna(0)

node_features = np.log1p(node_features)

node_scaler = StandardScaler()

node_features = node_scaler.fit_transform(
    node_features
)

x = torch.tensor(
    node_features,
    dtype=torch.float
)


# ----------------------------------------
# 4. Prepare transaction features
# ----------------------------------------

FEATURES = [
    "amount",
    "step",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

scaler = StandardScaler()

X_train = scaler.fit_transform(
    train_df[FEATURES]
)

X_val = scaler.transform(
    val_df[FEATURES]
)

X_test = scaler.transform(
    test_df[FEATURES]
)

X_train = torch.tensor(
    X_train,
    dtype=torch.float
)

X_val = torch.tensor(
    X_val,
    dtype=torch.float
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float
)

y_train = torch.tensor(
    train_df["isFraud"].values,
    dtype=torch.long
)

y_val = torch.tensor(
    val_df["isFraud"].values,
    dtype=torch.long
)

y_test = torch.tensor(
    test_df["isFraud"].values,
    dtype=torch.long
)

train_edges = torch.tensor(
    [
        train_df["source_id"].values,
        train_df["target_id"].values
    ],
    dtype=torch.long
)

val_edges = torch.tensor(
    [
        val_df["source_id"].values,
        val_df["target_id"].values
    ],
    dtype=torch.long
)

test_edges = torch.tensor(
    [
        test_df["source_id"].values,
        test_df["target_id"].values
    ],
    dtype=torch.long
)


# ----------------------------------------
# 5. Define GraphSAGE model
# ----------------------------------------

class GraphSAGE(nn.Module):

    def __init__(self, input_dim, hidden_dim=64):

        super().__init__()

        self.conv1 = SAGEConv(
            input_dim,
            hidden_dim
        )

        self.conv2 = SAGEConv(
            hidden_dim,
            hidden_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 6, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(
        self,
        x,
        edge_index,
        transaction_edges,
        transaction_features
    ):

        h = self.conv1(x, edge_index)
        h = torch.relu(h)

        h = self.conv2(h, edge_index)
        h = torch.relu(h)

        source = transaction_edges[0]
        target = transaction_edges[1]

        source_embedding = h[source]
        target_embedding = h[target]

        edge_embedding = torch.cat(
            [
                source_embedding,
                target_embedding,
                transaction_features
            ],
            dim=1
        )

        return self.classifier(edge_embedding)


# ----------------------------------------
# 6. Train model
# ----------------------------------------

model = GraphSAGE(
    input_dim=x.shape[1]
).to(DEVICE)

x = x.to(DEVICE)
edge_index = edge_index.to(DEVICE)

train_edges = train_edges.to(DEVICE)
val_edges = val_edges.to(DEVICE)
test_edges = test_edges.to(DEVICE)

X_train = X_train.to(DEVICE)
X_val = X_val.to(DEVICE)
X_test = X_test.to(DEVICE)

y_train = y_train.to(DEVICE)
y_val = y_val.to(DEVICE)
y_test = y_test.to(DEVICE)

class_counts = torch.bincount(y_train, minlength=2).float()

# Use a capped fraud weight to avoid overwhelming the loss
fraud_weight = torch.sqrt(
    class_counts[0] / class_counts[1].clamp(min=1)
).clamp(max=20.0)

class_weights = torch.tensor(
    [1.0, fraud_weight.item()],
    dtype=torch.float,
    device=DEVICE
)

criterion = nn.CrossEntropyLoss(weight=class_weights)

print("Class weights:", class_weights.tolist())

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print("Training GraphSAGE...")

best_val_pr_auc = 0

for epoch in range(1, 51):

    model.train()

    optimizer.zero_grad()

    output = model(
        x,
        edge_index,
        train_edges,
        X_train
    )

    loss = criterion(
        output,
        y_train
    )

    loss.backward()

    optimizer.step()

    # Validation
    model.eval()

    with torch.no_grad():

        val_output = model(
            x,
            edge_index,
            val_edges,
            X_val
        )

        val_probabilities = torch.softmax(
            val_output,
            dim=1
        )[:, 1].cpu().numpy()

        val_pr_auc = average_precision_score(
            y_val.cpu().numpy(),
            val_probabilities
        )

    if val_pr_auc > best_val_pr_auc:

        best_val_pr_auc = val_pr_auc

        torch.save(
            model.state_dict(),
            DATA_DIR / "graphsage_model.pth"
        )

    if epoch % 5 == 0:

        print(
            f"Epoch {epoch} | "
            f"Loss: {loss.item():.4f} | "
            f"Validation PR-AUC: {val_pr_auc:.4f}"
        )


# ----------------------------------------
# 7. Evaluate using a validation-selected threshold
# ----------------------------------------

model.load_state_dict(
    torch.load(
        DATA_DIR / "graphsage_model.pth",
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

with torch.no_grad():

    test_output = model(
        x,
        edge_index,
        test_edges,
        X_test
    )

    test_probabilities = torch.softmax(
        test_output,
        dim=1
    )[:, 1].cpu().numpy()

# Choose threshold using validation data only
model.eval()

with torch.no_grad():
    val_output = model(x, edge_index, val_edges, X_val)
    val_probabilities = torch.softmax(
        val_output,
        dim=1
    )[:, 1].cpu().numpy()

val_labels = y_val.cpu().numpy()
test_labels = y_test.cpu().numpy()

# Try thresholds and select the one with best validation F1
precision, recall, thresholds = precision_recall_curve(
    val_labels,
    val_probabilities
)

f1_scores = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-8)
)

best_idx = int(np.argmax(f1_scores))
best_threshold = float(thresholds[best_idx])

print(f"Best validation threshold: {best_threshold:.6f}")
print(f"Validation F1 at threshold: {f1_scores[best_idx]:.4f}")

predictions = (test_probabilities >= best_threshold).astype(int)

print("\nGraphSAGE Classification Report")
print(
    classification_report(
        test_labels,
        predictions,
        zero_division=0
    )
)

print(
    "PR-AUC:",
    average_precision_score(
        test_labels,
        test_probabilities
    )
)

print(
    "ROC-AUC:",
    roc_auc_score(
        test_labels,
        test_probabilities
    )
)

print("\nGraphSAGE model saved.")
