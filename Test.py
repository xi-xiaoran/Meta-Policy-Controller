import os
import matplotlib.pyplot as plt
import torch.optim as optim
from Accuracy_with_rejection import Get_Accuracy_With_Rejection
import torch
from collections import deque
from models.DS_PolicyNetwork import get_state_features
from MUE import Get_Best_MUE
from ECE import Get_ECE
import numpy as np
from collections import defaultdict
from torch.nn.utils import clip_grad_norm_
from Loss.DS_EDL_Loss import DS_EDL_Loss

def Test_policy_model(Policy_model, Backbone_model, train_dataloader, test_dataloader,
                      device, num_classes, config, save_root="test_results/"):
    """
    Test the policy network-guided new Backbone model.

    Args:
        Policy_model (torch.nn.Module): The policy network model.
        Backbone_model (torch.nn.Module): The backbone model.
        train_dataloader (DataLoader): Training dataset loader.
        test_dataloader (DataLoader): Testing dataset loader.
        device (torch.device): Device to use (e.g., 'cuda' or 'cpu').
        num_classes (int): Number of classes.
        config (argparse.Namespace): Configuration for the test.
        save_root (str, optional): Directory to save results. Defaults to "test_results/".
    """
    # Create the save directory
    os.makedirs(save_root, exist_ok=True)

    # Initialize KL history (same as during training)
    kl_history = deque(maxlen=100)  # Same length as in training code

    # Reuse the existing Backbone architecture
    test_model = type(Backbone_model)(num_classes=num_classes).to(device)
    optimizer = optim.Adam(test_model.parameters(), lr=config.learning_rate)

    # Training record container (including KL history)
    history = defaultdict(list)

    # Main training loop
    for epoch in range(config.Test_epochs):
        # ==================== Training Phase ====================
        test_model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = test_model(inputs)

            # --- Policy parameters generation (same as during training) ---
            with torch.no_grad():
                state = get_state_features(  # Use the same function as in training
                    outputs.detach(),
                    labels,
                    num_classes=num_classes,
                    epoch=epoch,
                    num_epochs=config.Test_epochs,
                    loss=history.get('train_loss', [0])[-1],
                    val_acc=history.get('val_acc', [0])[-1],
                    hist_kl=np.mean(kl_history) if kl_history else 0
                ).to(device)
                alpha_prior, lambda_kl = Policy_model(state)

            # --- Loss calculation and backpropagation ---
            loss = DS_EDL_Loss(num_classes=num_classes, device=device)(
                outputs, labels, alpha_prior, lambda_kl
            )

            # Record KL history (same as training logic)
            current_kl = lambda_kl.mean().item()
            kl_history.append(current_kl)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(test_model.parameters(), 2.0)
            optimizer.step()

            # --- Record metrics ---
            epoch_loss += loss.item() * inputs.size(0)
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        history['train_loss'].append(epoch_loss / total)
        history['train_acc'].append(correct / total)

        # ==================== Validation Phase ====================
        test_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = test_model(inputs)
                state = get_state_features(  # Use the same parameters as in validation
                    outputs.detach(),
                    labels,
                    num_classes=num_classes,
                    epoch=epoch,
                    num_epochs=config.Test_epochs,
                    loss=history.get('train_loss', [0])[-1],
                    val_acc=history.get('val_acc', [0])[-1],
                    hist_kl=np.mean(kl_history) if kl_history else 0
                ).to(device)
                alpha_prior, lambda_kl = Policy_model(state)

                loss = DS_EDL_Loss(num_classes=num_classes, device=device)(
                    outputs, labels, alpha_prior, lambda_kl
                )
                val_loss += loss.item() * inputs.size(0)
                _, pred = torch.max(outputs, 1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)

        history['val_loss'].append(val_loss / val_total)
        history['val_acc'].append(val_correct / val_total)

        # Print progress (add KL display)
        print(f"Test Phase Epoch {epoch + 1}/{config.Test_epochs}")
        print(f"Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {history['val_loss'][-1]:.4f}")
        print(f"Train Acc: {history['train_acc'][-1] * 100:.2f}% | Val Acc: {history['val_acc'][-1] * 100:.2f}%")
        print(f"Current KL: {current_kl:.4f} | Avg KL: {np.mean(kl_history):.4f}")  # New KL display
        print("-" * 60)

    # ==================== Final Evaluation ====================
    # Calculate metrics with the same data collection method as during training
    final_metrics = calculate_final_metrics(test_model, test_dataloader, device, num_classes, save_root)

    return final_metrics, history


def calculate_final_metrics(model, dataloader, device, num_classes, save_path):
    """
    Calculate final metrics (consistent with training evaluation logic).

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): Device to use (e.g., 'cuda' or 'cpu').
        num_classes (int): Number of classes.
        save_path (str): Directory to save results.

    Returns:
        dict: Dictionary containing final metrics.
    """
    model.eval()
    all_U, all_result = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Keep the same calculation method as during training
            alpha = outputs + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            U = num_classes / S.cpu().numpy()
            all_U.append(U)

            _, pred = torch.max(outputs, 1)
            # Ensure that each addition is a one-dimensional array
            all_result.append((pred == labels).cpu().numpy().reshape(-1, 1))

    return {
        'MUE': Get_Best_MUE(all_U, all_result, figure_path=os.path.join(save_path, 'MUE.png'))[0],
        'ECE': Get_ECE(model, dataloader, device, figure_path=os.path.join(save_path, 'ECE.png')),
        'Rejection_Acc': Get_Accuracy_With_Rejection(model, dataloader, device, num_classes)[0]
    }


def OOD_Test(
    Backbone_model,
    test_dataloader,
    train_dataloader,
    device,
    num_classes,
    OOD_dataloader,
    recording_path
):
    """
    Perform OOD testing.

    Args:
        Backbone_model (torch.nn.Module): The backbone model.
        test_dataloader (DataLoader): Test dataset loader.
        train_dataloader (DataLoader): Training dataset loader.
        device (torch.device): Device to use (e.g., 'cuda' or 'cpu').
        num_classes (int): Number of classes.
        OOD_dataloader (DataLoader): OOD dataset loader.
        recording_path (str): Path to record results.
    """
    # Add a function for Top-K% accuracy evaluation
    def Get_TopK_Acc(U_all, result_all, topk_list=[10, 20, 30, 50, 80, 100]):
        """
        Calculate Top-K% accuracy based on confidence.

        Args:
            U_all (list): List of uncertainties.
            result_all (list): List of results (correct/incorrect).
            topk_list (list, optional): List of top-K percentages. Defaults to [10, 20, 30, 50, 80, 100].

        Returns:
            dict: Dictionary of top-K% accuracies.
        """
        U_all = np.concatenate(U_all)
        result_all = np.concatenate(result_all)
        confidence = 1 - U_all
        sorted_indices = np.argsort(-confidence)
        total = len(confidence)
        topk_accuracies = {}

        for k in topk_list:
            k_count = int(total * (k / 100))
            selected = sorted_indices[:k_count]
            acc = np.sum(result_all[selected]) / k_count
            topk_accuracies[k] = acc
        return topk_accuracies

    # Final evaluation (within CIFAR10)
    model = Backbone_model
    model.eval()
    class_true = 0
    class_false = 0
    all_U = []
    all_result = []

    dataloader_test = test_dataloader
    dataloader_train = train_dataloader

    with torch.no_grad():
        for inputs, labels in dataloader_test:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs + 1
            _, pred = torch.max(outputs, 1)
            U = num_classes / torch.sum(outputs, dim=1)
            result = (pred == labels)

            class_true += result.sum().item()
            class_false += (~result).sum().item()
            all_U.append(U.cpu().numpy())
            all_result.append(result.cpu().numpy())

    # MUE threshold + rejection accuracy
    final_test_acc = class_true / (class_true + class_false)
    ece_train = Get_ECE(model, dataloader_train, device, plot=False, figure_path=None)
    ece_test = Get_ECE(model, dataloader_test, device, plot=False, figure_path=None)
    min_mue_value, min_mue_threshold = Get_Best_MUE(all_U, all_result, plot=False, figure_path=None)
    acc_with_rejection, reject_rate = Get_Accuracy_With_Rejection(model, dataloader_test, device, num_classes,
                                                                  min_mue_threshold)

    # Top-K% accuracy
    topk_accuracies = Get_TopK_Acc(all_U, all_result)

    # ============ Add SVHN OOD testing ===============

    ood_uncertainties = []
    with torch.no_grad():
        for inputs, _ in OOD_dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs + 1
            U = num_classes / torch.sum(outputs, dim=1)
            ood_uncertainties.append(U.cpu().numpy())

    ood_uncertainties = np.concatenate(ood_uncertainties)
    avg_ood_uncertainty = np.mean(ood_uncertainties)

    # Use min_mue_threshold to calculate OOD rejection rate (reject if uncertainty >= threshold)
    ood_reject_count = np.sum(ood_uncertainties >= min_mue_threshold)
    ood_total = len(ood_uncertainties)
    ood_rejection_rate = ood_reject_count / ood_total
    ood_accept_rate = 1 - ood_rejection_rate

    # Save all results
    with open(recording_path, 'a') as f:
        f.write(f'\nMethod: R-EDL, Dataset: CIFAR10, Alpha: 0.1\n')
        f.write(f'Final Test Accuracy: {final_test_acc:.4f}\n')
        f.write(f'Train ECE: {ece_train:.4f}\n')
        f.write(f'Test ECE: {ece_test:.4f}\n')
        f.write(f'MUE: {min_mue_value:.4f}, Threshold: {min_mue_threshold:.4f}\n')
        f.write(f'Accuracy with Rejection: {acc_with_rejection:.4f}, Rejection Rate: {reject_rate:.4f}\n')
        for k, v in topk_accuracies.items():
            f.write(f'Top-{k}% Accuracy: {v:.4f}\n')
        f.write(f'OOD Dataset (SVHN) Avg Uncertainty: {avg_ood_uncertainty:.4f}\n')
        f.write(f'OOD Rejection Rate (U >= threshold): {ood_rejection_rate:.4f}\n')
        f.write('=' * 60 + '\n\n')