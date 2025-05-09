import random
import numpy as np
import os
from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.svhn import SVHN
from torchvision.datasets.cifar import CIFAR10, CIFAR100
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
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

# Parameter settings
num_epochs = 50
num_classes = 10
alphas = [0.1]  # More alpha values can be added
data_name = 'MNIST'
method = 'RED'
model_name = 'ResNet18'
seeds = [2025, 14514, 1234]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create result directories
if not os.path.exists(f'results/{method}'):
    os.makedirs(f'results/{method}')
if not os.path.exists(f'save_models'):
    os.makedirs(f'save_models')

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize(32),  # Resize images to 32x32
    transforms.Lambda(lambda x: x.convert("RGB")),  # Convert grayscale images to RGB
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize
])

# Load datasets
data_train = MNIST(root="data/", train=True, download=True, transform=transform)
data_test = MNIST(root="data/", train=False, download=True, transform=transform)

# Create data loaders
dataloader_train = DataLoader(data_train, batch_size=1000, shuffle=True, num_workers=0)
dataloader_test = DataLoader(data_test, batch_size=1000, num_workers=0)

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

        with torch.no_grad():
            for inputs, labels in dataloader_test:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                outputs = outputs + 1  # Adjust for EDL implementation
                _, pred = torch.max(outputs, 1)
                U = num_classes / torch.sum(outputs, dim=1)
                result = (pred == labels)

                class_true += result.sum().item()
                class_false += (~result).sum().item()
                all_U.append(U.cpu().numpy())
                all_result.append(result.cpu().numpy())

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
            f.write(f'Train ECE: {ece_train:.4f}\n')
            f.write(f'Test ECE: {ece_test:.4f}\n')
            f.write(f'MUE: {min_mue_value:.4f}\n')
            f.write(f'Accuracy with Rejection: {accuracy:.4f}\n')
            f.write('=' * 50 + '\n\n')

print("Training completed!")