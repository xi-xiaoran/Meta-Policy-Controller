import random
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from ECE import Get_ECE
from MUE import Get_Best_MUE
from Uncertainty_Density import Plot_Uncertainty_Density
from Accuracy_with_rejection import Get_Accuracy_With_Rejection
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
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
num_classes = 10
random_sees = 2025
data_name = 'CIFAR10'
method = 'DS-RL-EDL'
model_name = 'ResNet18'
seed_everything(random_sees)
save_root = f'results/{method}/{data_name}_{model_name}_'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create result directories
if not os.path.exists(f'results/{method}'):
    os.makedirs(f'results/{method}')

# Load the test dataset
data_test = CIFAR10("data/",
                    train=False,
                    download=True,
                    transform=transforms.Compose([transforms.ToTensor()]))

dataloader_test = DataLoader(data_test, batch_size=1000, num_workers=0)

# Load the entire model
model_path = f'save_models/RL_EDL_{data_name}.pth'
model = torch.load(model_path)  # Load the entire model directly
model.to(device)
model.eval()

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

# MUE
min_mue_value, min_mue_threshold = Get_Best_MUE(all_U=all_U, all_result=all_result, plot=True, figure_path=f'results/{method}/{data_name} {model_name} MUE.png')
print(f"Minimum MUE value: {min_mue_value}, corresponding threshold: {min_mue_threshold}")

# ECE
ece_test = Get_ECE(model, dataloader_test, device, plot=True, figure_path=f'results/{method}/{data_name} {model_name} Test ECE.png', num_bins=5)
print(f'Test ECE: {ece_test:.4f}')

# Plot Uncertainty Density
Plot_Uncertainty_Density(model, dataloader_test, device, num_classes=num_classes, figure_path=f'results/{method}/{data_name} {model_name} Density.png')

# Get Accuracy With Rejection
accuracy, rejection_rate = Get_Accuracy_With_Rejection(model, dataloader_test, device, num_classes=num_classes, threshold=min_mue_threshold)
print(f'Accuracy after rejection: {accuracy * 100:.2f}%')
print(f'Rejection rate: {rejection_rate * 100:.2f}%')