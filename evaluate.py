import torch
import numpy as np
from torchvision import datasets, transforms

from model import MNISTCNN
model = MNISTCNN()

# Load aggregated weights
def load_npz_into_model(model, npz_path):
    data = np.load(npz_path)
    state_dict = model.state_dict()

    for k, v in zip(state_dict.keys(), data.files):
        state_dict[k] = torch.tensor(data[v])

    model.load_state_dict(state_dict)

# Evaluation
def main():
    model = MNISTCNN()
    model.eval()

    # CHANGE round number if needed
    npz_path = "./global_round_5.npz"
    load_npz_into_model(model, npz_path)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data",
            train=False,
            transform=transforms.ToTensor(),
            download=True,
        ),
        batch_size=64,
        shuffle=False,
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total * 100
    print(f"\nGlobal Model Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()
