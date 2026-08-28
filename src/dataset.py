import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def compute_mean_std(data_dir:str) -> tuple[list[float], list[float]]:
    raw_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
        )
    loader = DataLoader(raw_dataset, batch_size=512, shuffle=False, num_workers=0)     
    channel_sum = 0.0
    channel_sum_square = 0.0
    numOfPixelsPerChannel = 0

    for images, _ in loader:
        numOfPixelsPerChannel += images.numel() / images.size(1)
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sum_square += (images ** 2).sum(dim=[0, 2, 3])       

    mean = channel_sum / numOfPixelsPerChannel
    std = (channel_sum_square / numOfPixelsPerChannel - mean ** 2).sqrt()

    return mean.tolist(), std.tolist()

def get_transforms(train: bool, mean: list[float], std: list[float]) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
            ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
        mean=mean,
        std=std,
        ),
        ])

def get_dataloaders(
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 2,
    ) -> tuple[DataLoader, DataLoader, list[float], list[float]]:
    mean, std = compute_mean_std(data_dir)
    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(train=True, mean=mean, std=std),
    )
    val_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=get_transforms(train=False, mean=mean, std=std),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, mean, std