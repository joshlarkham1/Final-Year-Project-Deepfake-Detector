import torch
from torch import nn


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CNN(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),     # 224 -> 112
            conv_block(32, 64),    # 112 -> 56
            conv_block(64, 128),   # 56 -> 28
            conv_block(128, 256),  # 28 -> 14
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # 14x14 -> 1x1
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CNN(num_classes=2)
    x = torch.randn(4, 3, 224, 224)
    print(f"input:      {tuple(x.shape)}")
    for name in ["features", "pool", "classifier"]:
        x = getattr(model, name)(x)
        print(f"{name:<11} {tuple(x.shape)}")
    print(f"parameters: {count_parameters(model):,}")
