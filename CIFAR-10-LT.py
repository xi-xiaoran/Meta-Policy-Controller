import random
import numpy as np
import os
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.datasets.svhn import SVHN
from torchvision.datasets.cifar import CIFAR10, CIFAR100
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from models.Backbone import ResNet18, ResNet34, ResNet50, ResNet101
from Loss.EDL_Loss import EDL_Loss
from ECE import Get_ECE
from MUE import Get_Best_MUE
from Uncertainty_Density import Plot_Uncertainty_Density
from Accuracy_with_rejection import Get_Accuracy_With_Rejection

import warnings
warnings.filterwarnings('ignore')

def seed_everything(seed=2024):
    """
    Set all random seeds for reproducibility.

    Args:
        seed (int, optional): Seed value. Defaults to 2024.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_cifar10_lt(root, imb_factor=0.01, train=True, download=True):
    """
    Generate the CIFAR-10-LT dataset with long-tailed class distribution.

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

# Parameter settings
num_epochs = 50
num_classes = 10
alphas = [0.1]  # More alpha values can be added
data_name = 'CIFAR10-LT'
method = 'EDL'
model_name = 'ResNet18'
seeds = [2025, 14514, 1234]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create result directories
if not os.path.exists(f'results/{method}'):
    os.makedirs(f'results/{method}')
if not os.path.exists(f'save_models'):
    os.makedirs(f'save_models')

# Define data augmentation (more aggressive for long-tailed data)
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

# Generate training set (long-tailed) and test set (balanced)
train_dataset = build_cifar10_lt(root='./data', imb_factor=0.01, train=True)
test_dataset = CIFAR10(root='./data', train=False, transform=test_transform, download=True)

# Apply data augmentation (note: SubsetDataset requires manual application of Transform)
class ApplyTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)

# Wrap the training set to apply Transform
train_dataset = ApplyTransform(train_dataset, transform=train_transform)

# Create DataLoaders
dataloader_train = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

dataloader_test = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

import matplotlib.pyplot as plt

# Visualize the class distribution of the long-tailed dataset
targets = [train_dataset[i][1] for i in range(len(train_dataset))]
class_counts = np.bincount(targets)
print(class_counts)

plt.bar(range(10), class_counts)
plt.title("Class Distribution in CIFAR-10-LT")
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.xticks(range(10), [str(i) for i in range(10)])
plt.savefig('CIFAR10-LT.png')
# plt.show()

for seed in seeds:
    for alpha in alphas:
        seed_everything(seed)
        print(f"\nTraining with alpha={alpha}")
        save_root = f'results/{method}/{data_name}_{model_name}_alpha{alpha}_'

        # Initialize the model
        model = ResNet18(num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        # optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.005)

        criterion = EDL_Loss(num_classes=num_classes, alpha=alpha, device=device)

        # Training parameters
        train_loss_plt = []
        test_loss_plt = []
        train_acc_plt = []
        test_acc_plt = []
        best_test_acc = 0
        best_model_wts = None

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0

            # Training step
            for inputs, labels in dataloader_train:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, pred = torch.max(outputs, 1)
                correct_train += (pred == labels).sum().item()
                total_train += labels.size(0)

            train_loss_plt.append(train_loss / total_train)
            train_acc = correct_train / total_train
            train_acc_plt.append(train_acc)

            # Validation on the test set
            model.eval()
            test_loss = 0.0
            correct_test = 0
            total_test = 0
            with torch.no_grad():
                for inputs, labels in dataloader_test:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    test_loss += loss.item() * inputs.size(0)
                    _, pred = torch.max(outputs, 1)
                    correct_test += (pred == labels).sum().item()
                    total_test += labels.size(0)

            test_loss_plt.append(test_loss / total_test)
            test_acc = correct_test / total_test
            test_acc_plt.append(test_acc)

            # Save the best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model_wts = model.state_dict().copy()

            # Print results for each epoch
            print(f'{epoch}/{num_epochs}')
            print(f'Train Loss: {train_loss / total_train:.4f}, Train Acc: {train_acc * 100:.2f}%')
            print(f'Test Loss: {test_loss / total_test:.4f}, Test Acc: {test_acc * 100:.2f}%')

        # Load the best model for final evaluation
        model.load_state_dict(best_model_wts)
        torch.save(model, f'save_models/EDL_{data_name}_alpha{alpha}.pth')

        # Final evaluation
        model.eval()
        class_true = 0
        class_false = 0
        all_U = []
        all_result = []
        correct_per_class = [0] * num_classes  # New: Correct count per class
        total_per_class = [0] * num_classes  # New: Total samples per class

        with torch.no_grad():
            for inputs, labels in dataloader_test:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                outputs = outputs + 1  # Adjust for EDL implementation
                _, pred = torch.max(outputs, 1)
                U = num_classes / torch.sum(outputs, dim=1)
                result = (pred == labels)
                # New: Count correct and total samples per class
                for c in range(num_classes):
                    mask = (labels == c)
                    correct_per_class[c] += (pred[mask] == labels[mask]).sum().item()
                    total_per_class[c] += mask.sum().item()

                class_true += result.sum().item()
                class_false += (~result).sum().item()
                all_U.append(U.cpu().numpy())
                all_result.append(result.cpu().numpy())
        # New: Calculate accuracy per class
        class_accuracies = [correct_per_class[c] / total_per_class[c] if total_per_class[c] > 0 else 0
                            for c in range(num_classes)]
        # Calculate metrics
        final_test_acc = class_true / (class_true + class_false)
        ece_train = Get_ECE(model, dataloader_train, device, plot=False, figure_path=None)
        ece_test = Get_ECE(model, dataloader_test, device, plot=False, figure_path=None)
        min_mue_value, min_mue_threshold = Get_Best_MUE(all_U, all_result, plot=False, figure_path=None)
        accuracy, _ = Get_Accuracy_With_Rejection(model, dataloader_test, device, num_classes, min_mue_threshold)

        # Save results
        with open('recording.txt', 'a') as f:
            f.write(f'\nMethod: {method}, Dataset: {data_name}, Alpha: {alpha}\n')
            f.write(f'Final Test Accuracy: {final_test_acc:.4f}\n')
            # New: Write class-wise accuracy
            f.write("\n--- Class-wise Accuracy ---\n")
            for c in range(num_classes):
                f.write(f'Class {c}: {class_accuracies[c]:.4f} ({correct_per_class[c]}/{total_per_class[c]})\n')
            f.write(f'Train ECE: {ece_train:.4f}\n')
            f.write(f'Test ECE: {ece_test:.4f}\n')
            f.write(f'MUE: {min_mue_value:.4f}\n')
            f.write(f'Accuracy with Rejection: {accuracy:.4f}\n')
            f.write('=' * 50 + '\n\n')

print("Training completed!")