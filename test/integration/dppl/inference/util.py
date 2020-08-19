
from torchvision import transforms
from torchvision.datasets import MNIST
import torch.utils.data.dataloader as dataloader
import os

def loadData(batch_size):
    train = MNIST(os.environ.get("DATA_DIR", '.') + "/data", train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),  # ToTensor does min-max normalization.
    ]), )
    test = MNIST(os.environ.get("DATA_DIR", '.') + "/data", train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),  # ToTensor does min-max normalization.
    ]), )
    dataloader_args = dict(shuffle=True, batch_size=batch_size,
                        num_workers=3, pin_memory=False)
    train_loader = dataloader.DataLoader(train, **dataloader_args)
    test_loader = dataloader.DataLoader(test, **dataloader_args)
    return train_loader, test_loader