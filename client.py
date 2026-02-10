import argparse
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import MNISTCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    return [v.detach().cpu().numpy() for _, v in model.state_dict().items()]

def set_parameters(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    keys = list(model.state_dict().keys())
    state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    for k, v in zip(keys, parameters):
        t = torch.from_numpy(v)
        state_dict[k] = t
    model.load_state_dict(state_dict, strict=True)

def load_mnist_partition(cid: int, num_clients: int, batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose([transforms.ToTensor()])

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    # IID split: contiguous shards
    n = len(train_ds)
    shard = n // num_clients
    start = cid * shard
    end = (cid + 1) * shard if cid < num_clients - 1 else n
    train_subset = Subset(train_ds, list(range(start, end)))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader

def train(model, trainloader, epochs=1):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        for xb, yb in trainloader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

def evaluate(model, testloader) -> Tuple[float, float]:
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for xb, yb in testloader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            total_loss += float(loss.item()) * xb.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == yb).sum().item())
            total += xb.size(0)

    return total_loss / max(1, total), correct / max(1, total)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: int, num_clients: int, epochs: int):
        self.cid = cid
        self.num_clients = num_clients
        self.epochs = epochs
        self.model = MNISTCNN().to(DEVICE)
        self.trainloader, self.testloader = load_mnist_partition(cid, num_clients)

    def get_parameters(self, config: Dict):
        return get_parameters(self.model)

    def fit(self, parameters, config: Dict):
        set_parameters(self.model, parameters)
        train(self.model, self.trainloader, epochs=self.epochs)
        new_params = get_parameters(self.model)
        num_examples = len(self.trainloader.dataset)
        return new_params, num_examples, {"cid": self.cid}

    def evaluate(self, parameters, config: Dict):
        set_parameters(self.model, parameters)
        loss, acc = evaluate(self.model, self.testloader)
        num_examples = len(self.testloader.dataset)
        return float(loss), num_examples, {"accuracy": float(acc)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid", type=int, required=True)
    ap.add_argument("--num-clients", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--server", required=True, help="server host:port, e.g. 127.0.0.1:8080")
    args = ap.parse_args()

    fl.client.start_numpy_client(
        server_address=args.server,
        client=FlowerClient(args.cid, args.num_clients, args.epochs),
    )

if __name__ == "__main__":
    main()
