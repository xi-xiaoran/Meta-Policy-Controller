import argparse
import os
import tqdm
import random
import torchvision.transforms as transforms
import numpy as np
import time
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
from Loss.DS_EDL_Loss import DS_EDL_Loss
from ECE import Get_ECE
from tool_functions import get_dataloader, get_model, get_num_classes
from Train import Train_Backbone, Train_Policy_model
from Test import Test_policy_model
from models.DS_PolicyNetwork import PolicyNetwork
from Test import OOD_Test
from ECE import Get_ECE
from MUE import Get_Best_MUE
from torchvision.datasets.svhn import SVHN
from torchvision.datasets.cifar import CIFAR10, CIFAR100
from torchvision.datasets.mnist import MNIST
from Uncertainty_Density import Plot_Uncertainty_Density
from Accuracy_with_rejection import Get_Accuracy_With_Rejection

import warnings
warnings.filterwarnings('ignore')

torch.autograd.set_detect_anomaly(True)

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

if __name__ == '__main__':
    """
    Experimental settings for comparison.
    Dataset: MNIST    optimizer: Adam   lr: 1e-4   batchsize: 1000   model: ResNet18   KL-lambda: 0.1   epochs: 50
    Dataset: SVHN     optimizer: Adam   lr: 1e-4   batchsize: 1000   model: ResNet18   KL-lambda: 0.1   epochs: 100
    Dataset: CIFAR10  optimizer: Adam   lr: 1e-4   batchsize: 1000   model: ResNet18   KL-lambda: 0.1   epochs: 200
    """
    # 1. Experimental variable settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str,
        default="CIFAR10-LT",
        choices=["SVHN", "MNIST", "CIFAR10", "CIFAR100", "CIFAR10-LT"],
        help="Dataset to train models.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4,
        help="Learning rate for Policy/Backbone models.",
    )
    parser.add_argument(
        "--Backbone", type=str,
        default="ResNet18",
        choices=["ResNet18", "ResNet34", 'ResNet50'],
        help="Backbone model.",
    )
    parser.add_argument(
        "--dataset_root", type=str, default="data",
        help="Path for downloading datasets.",
    )
    parser.add_argument(
        "--Backbone_root", type=str, default="Backbone",
        help="Path for saving/loading Backbone.",
    )
    parser.add_argument(
        "--Policy_root", type=str, default="Policy_models",
        help="Path for saving/loading Policy models.",
    )
    parser.add_argument(
        "--Backbone_epochs", type=int, default=0,
        help="Pre-training epochs of the backbone network.",
    )
    parser.add_argument(
        "--relate_reward", type=int, default=2,
        help="Use related reward turns.",
    )
    parser.add_argument(
        "--Policy_epochs", type=int, default=100,
        help="Training epochs of the policy network.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1000,
        help="Training batch size.",
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=1000,
        help="Testing batch size.",
    )
    parser.add_argument(
        "--state_dim", type=int, default=6,
        help="Dimension of the state.",
    )
    parser.add_argument(
        "--txt_path", type=str, default="recording.txt",
        help="Path for recording results.",
    )
    parser.add_argument(
        "--seed_list", type=list, default=[2025, 14514, 1234],
        help="List of random seeds.",
    )
    parser.add_argument(
        "--OOD_Test", type=bool, default=False,
        help="Whether to use OOD dataset for testing.",
    )
    parser.add_argument(
        "--OOD_dataset", type=str, default='SVHN',
        help="OOD dataset.",
    )
    parser.add_argument(
        "--method", type=str, default="RL-EDL 3",
        help="Record important information about this experiment, such as special variable settings.",
    )

    config = parser.parse_args()

    # 2. Create corresponding folders and handle paths
    """
    dataset_root: Store datasets
    dataset_root/dataset: Store specific datasets
    Backbone_root: Store trained Backbone models, model names are Dataset+Backbone model name, e.g., MNIST_ResNet18.pth
    Policy_root: Store trained Policy models, model names are Dataset, e.g., MNIST.pth
    """
    dataset_path = os.path.join(config.dataset_root, config.dataset)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, exist_ok=True)
    if not os.path.exists(config.dataset_root):
        os.makedirs(config.dataset_root, exist_ok=True)
    if not os.path.exists(config.Backbone_root):
        os.makedirs(config.Backbone_root, exist_ok=True)
    if not os.path.exists(config.Policy_root):
        os.makedirs(config.Policy_root, exist_ok=True)
    Backbone_name = config.dataset + config.Backbone + '.pth'
    Backbone_path = os.path.join(config.Backbone_root, Backbone_name)
    Policy_name = config.dataset + '.pth'
    Policy_path = os.path.join(config.Policy_root, Policy_name)

    for seed in config.seed_list:
        # 3. Preparations before training
        seed_everything(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        # Get the full training dataset
        train_dataloader = get_dataloader(config.dataset,
                                          root=dataset_path,
                                          train=True,
                                          batch_size=config.train_batch_size)
        # Get the test dataset
        test_dataloader = get_dataloader(config.dataset,
                                         root=dataset_path,
                                         train=False,
                                         batch_size=config.test_batch_size)

        num_classes = get_num_classes(config.dataset)
        backbone_model = get_model(config.Backbone, num_classes=num_classes).to(device)
        criterion = DS_EDL_Loss(num_classes=num_classes, device=device)
        Policy_model = PolicyNetwork(config.state_dim, num_classes).to(device)

        # Train the policy model using the same backbone
        print(f"\n--- Training Policy with Backbone ---")
        optimizer_Backbone = optim.Adam(backbone_model.parameters(), lr=config.learning_rate, weight_decay=0.005)
        optimizer_Policy = optim.Adam(Policy_model.parameters(), lr=config.learning_rate, weight_decay=0.005)

        # Train the policy model
        training_metrics, Backbone_model = Train_Policy_model(
            Backbone_model=backbone_model,
            Policy_model=Policy_model,  # Use the same Policy_model instance
            Policy_path=config.Policy_root,
            dataset=config.dataset,
            epochs=config.Policy_epochs,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            device=device,
            optimizer_Backbone=optimizer_Backbone,
            optimizer_Policy=optimizer_Policy,
            criterion=criterion,
            num_classes=num_classes,
            plot_interval=5,
            relate_reward=config.relate_reward,
            save_path=f"result/EDL/training_curves_backbone"
        )
        with open(config.txt_path, 'a') as file:
            file.write(f'method:{config.method}   data_name:{config.dataset}   seed:{seed}\n')
            file.write(f'Accuracy: {training_metrics["Accuracy"] * 100:.2f}%\n')
            file.write(f'ECE: {training_metrics["ECE"]:.4f}\n')
            file.write(f'MUE: {training_metrics["MUE"]:.4f}\n')
            file.write(f'RACC: {training_metrics["RACC"]:.4f}\n')
            file.write('-----------------------------------------------\n\n')

        if config.OOD_Test:
            print(f"\n--- OOD dataset Testing Backbone ---")
            OOD_dataloader = get_dataloader(config.OOD_dataset,
                                            root=dataset_path,
                                            train=False,
                                            batch_size=config.test_batch_size)
            OOD_Test(Backbone_model=Backbone_model,
                     test_dataloader=test_dataloader,
                     train_dataloader=train_dataloader,
                     device=device,
                     num_classes=num_classes,
                     OOD_dataloader=OOD_dataloader,
                     recording_path=config.txt_path)