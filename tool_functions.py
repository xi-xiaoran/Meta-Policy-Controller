import argparse
import json
import os.path as osp
import pathlib
import pickle
import time
import os
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torchvision.transforms as transforms
from models.Backbone import ResNet18, ResNet34, ResNet50
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10, CIFAR100
from torchvision.datasets.svhn import SVHN

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_cifar10_lt(root, imb_factor=0.01, train=True, download=True):
    """
    Generate the CIFAR-10-LT dataset.

    Args:
        root (str): Root directory for dataset storage.
        imb_factor (float, optional): Imbalance factor (minimum class size / maximum class size). Defaults to 0.01.
        train (bool, optional): Whether to use the training set (long-tailed distribution is only applied to the training set). Defaults to True.
        download (bool, optional): Whether to download the dataset if not found. Defaults to True.

    Returns:
        torch.utils.data.Subset: Subset of CIFAR-10 with long-tailed distribution.
    """
    # Load the original CIFAR-10 dataset
    full_dataset = CIFAR10(root=root, train=train, download=download)
    targets = np.array(full_dataset.targets)

    # Collect indices for each class
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)

    # Calculate the number of samples for each class (exponential decay)
    class_counts = np.array([len(indices) for indices in class_indices.values()])
    max_num = max(class_counts)
    min_num = int(max_num * imb_factor)

    # Generate sample counts for each class
    img_num_per_cls = []
    for cls_idx in range(len(class_indices)):
        num = int(max_num * (imb_factor ** (cls_idx / (len(class_indices) - 1.0))))
        img_num_per_cls.append(num)

    # Randomly sample the specified number of samples from each class
    selected_indices = []
    for cls_idx, indices in class_indices.items():
        np.random.shuffle(indices)
        selected = indices[:img_num_per_cls[cls_idx]]
        selected_indices.extend(selected)

    # Create a subset dataset
    subset_dataset = torch.utils.data.Subset(full_dataset, selected_indices)
    return subset_dataset

# Apply data augmentation (note: SubsetDataset requires manual application of Transform)
class ApplyTransform(Dataset):
    def __init__(self, subset, transform=None):
        """
        Apply a transform to a subset dataset.

        Args:
            subset (torch.utils.data.Subset): Subset dataset.
            transform (callable, optional): Transform to apply. Defaults to None.
        """
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)

def get_dataloader(dataset_name: str, root: str = './data', train: bool = True, download: bool = True, batch_size: int = 64, shuffle: bool = True):
    """
    Return the DataLoader for the specified dataset.

    Args:
        dataset_name (str): Name of the dataset ('MNIST', 'CIFAR10', 'CIFAR100', 'SVHN').
        root (str, optional): Root directory for dataset storage. Defaults to './data'.
        train (bool, optional): Whether to load the training set (True) or test set (False). Defaults to True.
        download (bool, optional): Whether to download the dataset if not found. Defaults to True.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 64.
        shuffle (bool, optional): Whether to shuffle the data at the start of each epoch. Defaults to True.

    Returns:
        DataLoader: The DataLoader for the specified dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize
    ])

    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.Resize(32),  # Resize images to 32x32
            transforms.Lambda(lambda x: x.convert("RGB")),  # Convert grayscale images to RGB
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize((0.5,), (0.5,))  # Normalize
        ])
        dataset = MNIST(root=root, train=train, download=download, transform=transform)
    elif dataset_name == 'CIFAR10':
        dataset = CIFAR10(root=root, train=train, download=download, transform=transform)
    elif dataset_name == 'CIFAR100':
        dataset = CIFAR100(root=root, train=train, download=download, transform=transform)
    elif dataset_name == 'SVHN':
        split = 'train' if train else 'test'
        dataset = SVHN(root=root, split=split, download=download, transform=transform)
    elif dataset_name == 'CIFAR10-LT':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = build_cifar10_lt(root=root, imb_factor=0.1, train=True)
        test_dataset = CIFAR10(root=root, train=False, transform=test_transform, download=True)
        train_dataset = ApplyTransform(train_dataset, transform=train_transform)
        if train:
            dataset = train_dataset
        else:
            dataset = test_dataset
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets are 'MNIST', 'CIFAR10', 'CIFAR100', 'SVHN'.")

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader

def get_num_classes(dataset_name: str):
    """
    Return the number of classes for the specified dataset.

    Args:
        dataset_name (str): Name of the dataset ('MNIST', 'CIFAR10', 'CIFAR100', 'SVHN').

    Returns:
        int: Number of classes for the dataset.
    """
    if dataset_name == 'MNIST':
        num_classes = 10  # MNIST dataset has 10 classes
    elif dataset_name == 'CIFAR10':
        num_classes = 10  # CIFAR10 dataset has 10 classes
    elif dataset_name == 'CIFAR100':
        num_classes = 100  # CIFAR100 dataset has 100 classes
    elif dataset_name == 'SVHN':
        num_classes = 10  # SVHN dataset has 10 classes
    elif dataset_name == 'CIFAR10-LT':
        num_classes = 10  # CIFAR10-LT dataset has 10 classes
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets are 'MNIST', 'CIFAR10', 'CIFAR100', 'SVHN'.")

    return num_classes

def get_model(model_name: str, num_classes: int):
    """
    Return the model for the specified model name and number of classes.

    Args:
        model_name (str): Name of the model ('ResNet18', 'ResNet34', 'ResNet50').
        num_classes (int): Number of classes.

    Returns:
        torch.nn.Module: The model.
    """
    if model_name == "ResNet18":
        model = ResNet18(num_classes=num_classes)
    elif model_name == "ResNet34":
        model = ResNet34(num_classes=num_classes)
    elif model_name == "ResNet50":
        model = ResNet50(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Supported models are 'ResNet18', 'ResNet34', 'ResNet50'.")

    return model