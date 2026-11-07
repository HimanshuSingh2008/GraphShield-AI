import torch
import torch.nn.functional as F

from pathlib import Path
from torch_geometric.nn import SAGEConv

DATA_DIR = Path(__file__).parent / "data"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


class GraphSAGE(torch.nn.Module):

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

        self.classifier = torch.nn.Linear(
            hidden_dim * 2,
            2
        )

    def forward(self, x, edge_index):

        # First graph convolution
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Second graph convolution
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Edge representations
        source = edge_index[0]
        target = edge_index[1]

        source_embedding = x[source]
        target_embedding = x[target]

        edge_embedding = torch.cat(
            [
                source_embedding,
                target_embedding
            ],
            dim=1
        )

        return self.classifier(edge_embedding)


def main():

    print("Loading graph...")

    graph = torch.load(
        DATA_DIR / "transaction_graph.pt",
        map_location=DEVICE,
        weights_only=False
    )

    graph = graph.to(DEVICE)

    model = GraphSAGE(
        input_dim=graph.x.shape[1]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # Class imbalance handling
    class_counts = torch.bincount(graph.y)

    class_weights = (
        class_counts.sum() / class_counts
    ).float()

    class_weights = (
        class_weights / class_weights.sum()
    )

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights
    )

    print("Training GraphSAGE...")

    for epoch in range(1, 21):

        model.train()

        optimizer.zero_grad()

        output = model(
            graph.x,
            graph.edge_index
        )

        loss = criterion(
            output,
            graph.y
        )

        loss.backward()

        optimizer.step()

        if epoch % 5 == 0:

            print(
                f"Epoch {epoch}, "
                f"Loss: {loss.item():.4f}"
            )

    torch.save(
        model.state_dict(),
        DATA_DIR / "graphsage_model.pth"
    )

    print("GraphSAGE model saved.")


if __name__ == "__main__":
    main()