import random
import numpy as np
import os
from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch.nn.functional as F

from models.ResNet18 import ResNet18
from ECE import Get_ECE
from MUE import Get_Best_MUE
from scipy.optimize import minimize, minimize_scalar

import warnings
warnings.filterwarnings('ignore')

def add_noise(outputs, mean, stddev, device):
    """
    Add Gaussian noise to the model outputs.

    Args:
        outputs (torch.Tensor): Model outputs.
        mean (float): Mean of the noise.
        stddev (float): Standard deviation of the noise.
        device (torch.device): Device to use.

    Returns:
        torch.Tensor: Noisy outputs.
    """
    noise = np.random.normal(mean, stddev, outputs.shape)
    noise = torch.from_numpy(noise).to(device)
    outputs = outputs.to(device)
    outputs = outputs + noise
    outputs = outputs.type(torch.float32)
    return outputs

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

num_epochs = 200
num_classes = 10
random_list = [2025, 1234, 14514]
data_name = 'CIFAR10'
method = 'Original'
model_name = 'ResNet18'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(f'results/{method}'):
    os.makedirs(f'results/{method}')

for seed in random_list:
    seed_everything(seed)

    save_root = f'results/{method}/{data_name}_{model_name}_{str(seed)}_'
    data_train = CIFAR10("data/",
                         download=True,
                         train=True,
                         transform=transforms.Compose([transforms.ToTensor()]))

    data_val = CIFAR10("data/",
                       train=False,
                       download=True,
                       transform=transforms.Compose([transforms.ToTensor()]))

    dataloader_train = DataLoader(data_train, batch_size=1000, shuffle=True, num_workers=0)
    dataloader_val = DataLoader(data_val, batch_size=1000, num_workers=0)
    dataloader_test = DataLoader(data_val, batch_size=1000, num_workers=0)

    model = ResNet18(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.005)
    criterion = nn.CrossEntropyLoss()
    save_path = f'save_models/EDL_{data_name}.pth'

    train_loss_plt = []
    val_loss_plt = []
    train_acc_plt = []
    val_acc_plt = []
    best_loss = float('inf')
    best_model_wts = None

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1} training starts')
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for i, (inputs, labels) in enumerate(dataloader_train):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            # Calculate training accuracy
            _, pred = torch.max(outputs, dim=1)
            correct_train += (pred == labels).sum().item()
            total_train += labels.size(0)

        train_loss_plt.append(train_loss / total_train)  # Average loss
        train_acc = correct_train / total_train  # Training accuracy
        train_acc_plt.append(train_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader_val):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                _, pred = torch.max(outputs, dim=1)
                correct_val += (pred == labels).sum().item()
                total_val += labels.size(0)

        val_loss_plt.append(val_loss / total_val)  # Average loss
        val_acc = correct_val / total_val  # Validation accuracy
        val_acc_plt.append(val_acc)

        # Record the best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = model.state_dict()

        # Save model every epoch
        model.load_state_dict(best_model_wts)
        torch.save(model, save_path)

        # Print results for each epoch
        print(f'train_loss: {train_loss / total_train:.4f}, train_acc: {train_acc * 100:.2f}%')
        print(f'val_loss: {val_loss / total_val:.4f}, val_acc: {val_acc * 100:.2f}%')

    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), train_loss_plt, label='Train Loss')
    plt.plot(range(num_epochs), val_loss_plt, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), train_acc_plt, label='Train Accuracy')
    plt.plot(range(num_epochs), val_acc_plt, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.legend()
    plt.savefig(save_root + 'Train.png')
    plt.close()

    # Model evaluation
    class_true = 0.0
    class_false = 0.0
    all_U = []
    all_result = []
    for i, (inputs, labels) in enumerate(dataloader_test):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            outputs = outputs + 1
            _, pred = torch.max(outputs, dim=1)
            U = num_classes / torch.sum(outputs, dim=1)
            result = torch.eq(pred, labels)

            # Accumulate correct and incorrect predictions
            class_true += result.sum().item()
            class_false += (~result).sum().item()

            # Append results to lists
            all_U.append(U.cpu().numpy())
            all_result.append(result.cpu().numpy())
    print(f'{data_name} test set accuracy: {class_true / (class_false + class_true) * 100}%')

    class_true = float(class_true)
    class_false = float(class_false)

    # ECE
    ece_train = Get_ECE(model, dataloader_train, device, plot=True,
                        figure_path=f'results/{method}/{data_name} {model_name} Train ECE.png', num_bins=5)
    ece_test = Get_ECE(model, dataloader_test, device, plot=True,
                       figure_path=f'results/{method}/{data_name} {model_name} Test ECE.png', num_bins=5)
    print(f'Train ECE: {ece_train:.4f}')
    print(f'Test ECE: {ece_test:.4f}')

    with open('recording.txt', 'a') as file:
        file.write(f'method:{method}   data_name:{data_name}   seed:{seed}\n')
        file.write(f'Test Acc:{class_true / (class_false + class_true) * 100}%\n')
        file.write(f'Train ECE: {ece_train}\n')
        file.write(f'Test ECE: {ece_test}\n')
        file.write(f'-----------------------------------------------\n\n')
    file.close()