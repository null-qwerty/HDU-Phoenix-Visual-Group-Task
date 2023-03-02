import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms as tr
from torch.utils.data import DataLoader


def dataLoader(Train=True, Batch_size=64, Size=28):
    transform = tr.Compose([
        tr.Resize(size=Size),
        tr.ToTensor(),
        tr.Normalize((0.1307,), (0.3081,))
    ])
    Dataset = FashionMNIST('./DL/fashionMNIST/Dataset', train=Train, download=True, transform=transform)
    Data_loader = DataLoader(Dataset, batch_size=Batch_size, shuffle=True)
    return Data_loader
