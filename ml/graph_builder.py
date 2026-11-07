import pandas as pd
import torch
import numpy as np

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

DATA_DIR = Path(__file__).parent / "data"

INPUT_FILE = DATA_DIR / "processed_transactions.csv"
OUTPUT_FILE = DATA_DIR / "transaction_graph.pt"


def build_graph():

    print("Loading processed transactions...")

    df = pd.read_csv(INPUT_FILE)

    # --------------------------------
    # 1. Create account IDs
    # --------------------------------

    accounts = pd.concat([
        df["nameOrig"],
        df["nameDest"]
    ]).unique()

    account_to_id = {
        account: idx
        for idx, account in enumerate(accounts)
    }

    df["source_id"] = df["nameOrig"].map(account_to_id)
    df["target_id"] = df["nameDest"].map(account_to_id)

    num_nodes = len(accounts)

    print("Number of accounts:", num_nodes)
    print("Number of transactions:", len(df))

    # --------------------------------
    # 2. Create graph edges
    # --------------------------------

    edge_index = torch.tensor(
        [
            df["source_id"].values,
            df["target_id"].values
        ],
        dtype=torch.long
    )

    # --------------------------------
    # 3. Create transaction features
    # --------------------------------

    transaction_types = pd.get_dummies(
        df["type"],
        dtype=float
    )

    edge_features = pd.concat([
        df[["amount", "step"]].reset_index(drop=True),
        transaction_types.reset_index(drop=True)
    ], axis=1)

    # Log-transform amount to reduce skew
    edge_features["amount"] = np.log1p(
        edge_features["amount"]
    )

    # Standardize numeric features
    scaler = StandardScaler()

    edge_features[["amount", "step"]] = scaler.fit_transform(
        edge_features[["amount", "step"]]
    )

    edge_attr = torch.tensor(
        edge_features.values,
        dtype=torch.float
    )

    # --------------------------------
    # 4. Create node features
    # --------------------------------

    # Number of outgoing transactions
    outgoing_count = df.groupby("source_id").size()

    # Number of incoming transactions
    incoming_count = df.groupby("target_id").size()

    # Total amount sent
    outgoing_amount = df.groupby("source_id")["amount"].sum()

    # Total amount received
    incoming_amount = df.groupby("target_id")["amount"].sum()

    node_features = pd.DataFrame(
        index=range(num_nodes)
    )

    node_features["outgoing_count"] = outgoing_count
    node_features["incoming_count"] = incoming_count

    node_features["outgoing_amount"] = outgoing_amount
    node_features["incoming_amount"] = incoming_amount

    node_features = node_features.fillna(0)

    # Log-transform counts and amounts
    node_features = np.log1p(node_features)

    node_scaler = StandardScaler()

    node_features = node_scaler.fit_transform(
        node_features
    )

    x = torch.tensor(
        node_features,
        dtype=torch.float
    )

    # --------------------------------
    # 5. Create fraud labels
    # --------------------------------

    y = torch.tensor(
        df["isFraud"].values,
        dtype=torch.long
    )

    # --------------------------------
    # 6. Create PyTorch Geometric Data
    # --------------------------------

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y
    )

    graph.num_nodes = num_nodes

    print(graph)

    torch.save(graph, OUTPUT_FILE)

    print(f"Graph saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    build_graph()