from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).resolve().parent.parent / "dataset"

def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=64, num_workers=2)
    n, mean, sq = 0, torch.zeros(3), torch.zeros(3)
    for x, _ in loader:
        n += x.numel() // 3
        mean += x.sum(dim=(0, 2, 3))
        sq += (x ** 2).sum(dim=(0, 2, 3))
    mean /= n
    std = (sq / n - mean ** 2).clamp(min=0).sqrt()
    return mean, std


# Computed once from the train split (Resize + ToTensor only) with compute_mean_std
MEAN = [0.5211308002471924, 0.42512014508247375, 0.3794030547142029]
STD = [0.27673235535621643, 0.25088441371917725, 0.2526720464229584]


class getData:

    def __init__(self, data_dir=DATA_DIR, batch_size=32, image_size=224):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        self.train_set = self._load("train")
        self.valid_set = self._load("valid")
        self.test_set = self._load("test")
        self.classes = self.train_set.classes  # ['fake', 'real']

    def _load(self, split):
        return datasets.ImageFolder(self.data_dir / split, transform=self.transform)

    def train_loader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def valid_loader(self):
        return DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False)

    def test_loader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
