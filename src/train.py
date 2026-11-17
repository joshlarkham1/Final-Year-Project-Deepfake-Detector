import torch
from torch import nn
import matplotlib.pyplot as plt

from dataset import getData, MEAN, STD
from models import CNN

batch_size = 32
epochs = 10
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def show_samples(data, n=25):

    images, labels = next(iter(data.train_loader()))
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)

    plt.figure(figsize=(10, 10))
    for i in range(min(n, len(images))):
        plt.subplot(5, 5, i + 1)
        img = (images[i] * std + mean).clamp(0, 1)
        plt.imshow(img.permute(1, 2, 0))
        plt.title(data.classes[labels[i]])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
    plt.tight_layout()
    plt.show()


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def main():
    data = getData(batch_size=batch_size)
    print(f"Classes: {data.classes}")
    print(f"Train: {len(data.train_set)}  Valid: {len(data.valid_set)}  Test: {len(data.test_set)}")
    print(f"Device: {device}")

    show_samples(data)

    model = CNN(num_classes=len(data.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, data.train_loader(), criterion, optimizer)
        val_loss, val_acc = evaluate(model, data.valid_loader(), criterion)
        print(f"Epoch {epoch}/{epochs}  "
              f"train loss {train_loss:.4f} acc {train_acc:.3f}  "
              f"val loss {val_loss:.4f} acc {val_acc:.3f}")

    torch.save(model.state_dict(), "cnn.pt")


if __name__ == "__main__":
    main()
