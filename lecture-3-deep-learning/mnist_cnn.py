"""
MNIST digit classifier using a small CNN.

Two conv layers, max pooling after each, then a single FC layer.
Trains on 60k examples, tests on 10k it's never seen.

The lecture specifically mentioned MNIST as the go-to project for
understanding deep learning — this is exactly that.

Requires: pip install torch torchvision
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 1 input channel (grayscale), 32 filters, 3x3 kernel
        # output: 32 feature maps of size 26x26
        self.conv1 = nn.Conv2d(1, 32, 3)

        # halves spatial dims: 26x26 → 13x13
        self.pool = nn.MaxPool2d(2)

        # 32 → 64 filters, 13x13 → 11x11 → 5x5 after pooling
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.fc = nn.Linear(64 * 5 * 5, 10)

        # zero out half the activations during training
        # if the model only passes training but not test, this is the fix
        self.dropout = nn.Dropout(0.5)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # these normalization values are computed from the MNIST dataset itself
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST("data", train=True,  download=True, transform=t)
    test_data  = datasets.MNIST("data", train=False, download=True, transform=t)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=64)

    model     = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"Training on {device}\n")
    print(f"{'Epoch':<8} {'Loss':<12} {'Train Acc':<14} {'Test Acc'}")
    print("-" * 46)

    for epoch in range(1, 11):
        loss       = train(model, train_loader, optimizer, criterion, device)
        train_acc  = accuracy(model, train_loader, device)
        test_acc   = accuracy(model, test_loader,  device)

        # flag when train and test diverge — that's overfitting starting
        warn = "  ← overfitting?" if train_acc - test_acc > 0.03 else ""
        print(f"{epoch:<8} {loss:<12.4f} {train_acc*100:<13.1f}% {test_acc*100:.1f}%{warn}")

    print(f"\nFinal test accuracy: {accuracy(model, test_loader, device)*100:.2f}%")


if __name__ == "__main__":
    main()
