import torch
import os
import time
from tqdm import tqdm
from collections import deque
from models.DS_PolicyNetwork import get_state_features
from MUE import Get_Best_MUE
from ECE import Get_ECE
from Accuracy_with_rejection import Get_Accuracy_With_Rejection
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from torch.nn.utils import clip_grad_norm_

def compute_ece(confidences, accuracies, num_bins=10):
    """
    Compute the Expected Calibration Error (ECE).

    Args:
        confidences (np.ndarray): Model confidences.
        accuracies (np.ndarray): Model accuracies.
        num_bins (int, optional): Number of bins. Defaults to 10.

    Returns:
        float: ECE value.
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total_samples = len(confidences)
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = np.logical_and(confidences >= bin_lower, confidences < bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count == 0:
            bin_acc = 0.0
            avg_confidence = 0.0
        else:
            bin_acc = np.mean(accuracies[in_bin])
            avg_confidence = np.mean(confidences[in_bin])

        ece += (bin_count / total_samples) * np.abs(bin_acc - avg_confidence)
    return ece


# Modify Get_Best_MUE to handle single batch data
def Get_Best_MUE(all_U, all_result, plot=True, figure_path='results/MUE.png'):
    """
    Calculate the best MUE value and corresponding threshold.

    Args:
        all_U (np.ndarray): Uncertainty values.
        all_result (np.ndarray): Results (correct/incorrect).
        plot (bool, optional): Whether to plot the MUE curve. Defaults to True.
        figure_path (str, optional): Path to save the MUE plot. Defaults to 'results/MUE.png'.

    Returns:
        tuple: (best_mue_value, best_threshold)
    """
    if isinstance(all_U, list) and isinstance(all_result, list):
        all_U = np.concatenate(all_U, axis=0)
        all_result = np.concatenate(all_result, axis=0)

    def calculate_mue(threshold, U, result):
        D_c = U[result == 1]
        D_i = U[result == 0]
        term_c = 0.5 * (np.sum(D_c > threshold) / len(D_c)) if len(D_c) > 0 else 0
        term_i = 0.5 * (np.sum(D_i <= threshold) / len(D_i)) if len(D_i) > 0 else 0
        return term_c + term_i

    thresholds = np.linspace(0, 1, 100)
    mues = [calculate_mue(t, all_U, all_result) for t in thresholds]
    min_idx = np.argmin(mues)
    return mues[min_idx], thresholds[min_idx]


def plot_alpha_prior_evolution(alpha_records, save_path="alpha_prior_dynamics.png"):
    """
    Plot the evolution of Dirichlet prior alpha_prior values over epochs.

    Args:
        alpha_records (list): List of (epoch_index, alpha_prior_tensor).
        save_path (str, optional): Path to save the plot. Defaults to "alpha_prior_dynamics.png".
    """
    if len(alpha_records) == 0:
        print("No alpha_prior records to plot.")
        return

    alpha_records.sort(key=lambda x: x[0])
    epochs = [rec[0] for rec in alpha_records]
    alphas = [rec[1].detach().cpu().numpy().flatten() for rec in alpha_records]
    alphas = np.stack(alphas)

    plt.figure(figsize=(10, 6))
    num_classes = alphas.shape[1]
    for k in range(num_classes):
        plt.plot(epochs, alphas[:, k], label=f'Class {k}')

    plt.xlabel('Epoch')
    plt.ylabel('Alpha Prior Value')
    plt.title('Evolution of Dirichlet Prior $\\alpha_0$ per Class')
    plt.legend(ncol=4, bbox_to_anchor=(0.5, -0.15), loc='upper center', frameon=False)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def Train_Backbone(
        model,
        epochs,
        train_dataloader,
        test_dataloader,
        device,
        optimizer,
        criterion,
        Backbone_path,
        KL_lamda,
        num_classes
):
    """
    Train the backbone model.

    Args:
        model (torch.nn.Module): Backbone model.
        epochs (int): Number of training epochs.
        train_dataloader (DataLoader): Training data loader.
        test_dataloader (DataLoader): Testing data loader.
        device (torch.device): Device to use.
        optimizer (torch.optim.Optimizer): Optimizer for the backbone model.
        criterion (callable): Loss function.
        Backbone_path (str): Path to save the backbone model.
        KL_lamda (float): KL divergence lambda value.
        num_classes (int): Number of classes.

    Returns:
        torch.nn.Module: Trained backbone model.
    """
    alpha_prior = torch.ones([1, num_classes], device=device)
    train_losses = []
    train_accuracy = []
    val_losses = []
    val_accuracy = []
    best_loss = float('inf')
    best_model_wts = None

    for _ in tqdm(range(epochs), desc="Epochs", unit="epoch"):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels, alpha_prior, KL_lamda)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            _, pred = torch.max(outputs, dim=1)
            correct_train += (pred == labels).sum().item()
            total_train += labels.size(0)

        train_losses.append(train_loss / total_train)
        train_acc = correct_train / total_train
        train_accuracy.append(train_acc)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels, alpha_prior, KL_lamda)
                val_loss += loss.item() * inputs.size(0)

                _, pred = torch.max(outputs, dim=1)
                correct_val += (pred == labels).sum().item()
                total_val += labels.size(0)

        val_losses.append(val_loss / total_val)
        val_acc = correct_val / total_val
        val_accuracy.append(val_acc)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = model.state_dict()

        if best_model_wts is not None and isinstance(best_model_wts, dict):
            model.load_state_dict(best_model_wts)
        else:
            print("Invalid best_model_wts. Cannot load the model weights.")
        torch.save(model, Backbone_path)

    return model


def compute_classwise_accuracy(model, dataloader, num_classes, device, save_path="classwise_accuracy.txt"):
    """
    Compute class-wise accuracy and save to a text file.

    Args:
        model (torch.nn.Module): Model to evaluate.
        dataloader (DataLoader): Data loader for evaluation.
        num_classes (int): Number of classes.
        device (torch.device): Device to use.
        save_path (str, optional): Path to save the results. Defaults to "classwise_accuracy.txt".
    """
    model.eval()
    correct = [0 for _ in range(num_classes)]
    total = [0 for _ in range(num_classes)]

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for label, pred in zip(labels, preds):
                total[label.item()] += 1
                if label == pred:
                    correct[label.item()] += 1

    acc_per_class = [correct[i] / total[i] if total[i] != 0 else 0.0 for i in range(num_classes)]
    with open(save_path, 'a') as f:
        f.write("Class-wise Accuracy:\n")
        for i, acc in enumerate(acc_per_class):
            f.write(f"Class {i}: {acc * 100:.2f}% ({correct[i]}/{total[i]})\n")

    return acc_per_class


def Train_Policy_model(
        Backbone_model,
        Policy_model,
        Policy_path,
        dataset,
        epochs,
        train_dataloader,
        test_dataloader,
        device,
        optimizer_Backbone,
        optimizer_Policy,
        criterion,
        num_classes,
        plot_interval=5,
        relate_reward=1,
        save_path="training_curves.png",
        text_path=None
):
    """
    Train the policy model.

    Args:
        Backbone_model (torch.nn.Module): Backbone model.
        Policy_model (torch.nn.Module): Policy model.
        Policy_path (str): Path to save the policy model.
        dataset (str): Dataset name.
        epochs (int): Number of training epochs.
        train_dataloader (DataLoader): Training data loader.
        test_dataloader (DataLoader): Testing data loader.
        device (torch.device): Device to use.
        optimizer_Backbone (torch.optim.Optimizer): Optimizer for the backbone model.
        optimizer_Policy (torch.optim.Optimizer): Optimizer for the policy model.
        criterion (callable): Loss function.
        num_classes (int): Number of classes.
        plot_interval (int, optional): Interval for plotting training curves. Defaults to 5.
        relate_reward (int, optional): Number of related reward calculations. Defaults to 1.
        save_path (str, optional): Path to save the training curves. Defaults to "training_curves.png".
        text_path (str, optional): Path to save the training results. Defaults to None.

    Returns:
        dict: Best metrics (Accuracy, ECE, MUE, RACC).
    """
    alpha_records = []
    history = defaultdict(list)
    best_metrics = {'Accuracy': 0.0, 'ECE': float('inf'), 'MUE': float('inf'), 'RACC': float('inf')}
    kl_history = deque(maxlen=100)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    for epoch in range(epochs):
        epoch_metrics = {
            'train_loss': 0.0,
            'train_acc': 0.0,
            'rewards': [],
            'kl_coeffs': []
        }

        if epoch == 0:
            alpha_batch_list = []

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            Backbone_model.train()
            Policy_model.eval()

            old_state = get_state_features(
                Backbone_model(inputs).detach(),
                labels,
                num_classes,
                epoch=epoch,
                num_epochs=epochs,
                loss=history.get('train_loss', [0])[-1],
                val_acc=history.get('val_acc', [0])[-1],
                hist_kl=np.mean(kl_history) if kl_history else 0
            ).to(device)
            old_alpha, old_kl = Policy_model(old_state)
            alpha_batch_list.append(old_alpha.detach().cpu())

            for i in range(relate_reward):
                old_outputs = Backbone_model(inputs)
                alpha_old = old_outputs + 1
                S_old = torch.sum(alpha_old, dim=1, keepdim=True)
                probs_old = alpha_old / S_old
                confidence_old, preds_old = torch.max(probs_old, dim=1)
                correct_old = (preds_old == labels).float().cpu().numpy()
                confidences_old = confidence_old.detach().cpu().numpy()

                old_ece = compute_ece(confidences_old, correct_old)
                old_loss = criterion(old_outputs, labels, old_alpha, old_kl)
                old_acc = (old_outputs.argmax(1) == labels).float().mean().item()
                S_old = S_old.detach().cpu().numpy()
                U_old = num_classes / S_old
                result_old = (preds_old.cpu() == labels.cpu()).numpy().astype(int)
                old_mue, _ = Get_Best_MUE([U_old], [result_old], plot=False)

                optimizer_Backbone.zero_grad()
                old_loss.backward(retain_graph=True)
                clip_grad_norm_(Backbone_model.parameters(), 2.0)
                optimizer_Backbone.step()

            new_outputs = Backbone_model(inputs)
            alpha_new = new_outputs + 1
            S_new = torch.sum(alpha_new, dim=1, keepdim=True)
            probs_new = alpha_new / S_new
            confidence_new, preds_new = torch.max(probs_new, dim=1)
            correct_new = (preds_new == labels).float().cpu().numpy()
            confidences_new = confidence_new.detach().cpu().numpy()

            new_ece = compute_ece(confidences_new, correct_new)
            new_acc = (new_outputs.argmax(1) == labels).float().mean().item()
            result_new = (preds_new.cpu() == labels.cpu()).numpy().astype(int)
            S_new = S_new.detach().cpu().numpy()
            U_new = num_classes / S_new
            new_mue, _ = Get_Best_MUE([U_new], [result_new], plot=False)

            delta_acc = new_acc - old_acc
            delta_ece = old_ece - new_ece
            delta_mue = old_mue - new_mue

            kl_mean = old_kl.mean().item()
            reward = delta_ece + delta_acc + delta_mue - 0.01 * kl_mean

            Backbone_model.eval()
            Policy_model.train()

            policy_loss = -torch.mean(torch.log(old_kl + 1e-8) + torch.log(old_alpha)) * reward + 0.005 * (kl_mean ** 2)
            optimizer_Policy.zero_grad()
            policy_loss.backward()
            clip_grad_norm_(Policy_model.parameters(), 1.0)
            optimizer_Policy.step()

            epoch_metrics['train_loss'] += old_loss.item() * inputs.size(0)
            epoch_metrics['train_acc'] += (new_outputs.argmax(1) == labels).sum().item()
            epoch_metrics['rewards'].append(reward)
            epoch_metrics['kl_coeffs'].append(old_kl.mean().item())
            kl_history.append(old_kl.mean().item())

            if len(alpha_batch_list) > 0:
                alpha_epoch = torch.stack(alpha_batch_list, dim=0)
            alpha_mean = alpha_epoch.mean(dim=0)
            alpha_records.append((epoch + 1, alpha_mean))
            alpha_batch_list.clear()

        total = len(train_dataloader.dataset)
        epoch_metrics['train_loss'] /= total
        epoch_metrics['train_acc'] /= total

        ACC, ECE, MUE, RACC, val_loss = evaluate(
            Backbone_model, Policy_model, test_dataloader, criterion, device, num_classes
        )

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(epoch_metrics['train_loss'])
        history['train_acc'].append(epoch_metrics['train_acc'])
        history['val_loss'].append(val_loss)
        history['val_acc'].append(ACC)
        history['kl_coeff'].append(np.mean(epoch_metrics['kl_coeffs']))
        history['reward'].append(np.mean(epoch_metrics['rewards']))

        if ACC > best_metrics['Accuracy']:
            best_metrics.update({
                'Accuracy': ACC,
                'ECE': ECE,
                'MUE': MUE,
                'RACC': RACC,
                'epoch': epoch + 1
            })
            torch.save({
                'backbone': Backbone_model.state_dict(),
                'policy': Policy_model.state_dict(),
            }, os.path.join(Policy_path, f'{dataset}_best_model.pth'))

        if (epoch + 1) % plot_interval == 0 or epoch == epochs - 1:
            plot_training_curves(history, save_path)

        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Train Loss: {epoch_metrics["train_loss"]:.4f} | Val Loss: {val_loss:.4f}')
        print(f'Train Acc: {epoch_metrics["train_acc"] * 100:.2f}% | Val Acc: {ACC * 100:.2f}%')
        print(
            f'Avg Reward: {np.mean(epoch_metrics["rewards"]):.4f} | KL Coeff: {np.mean(epoch_metrics["kl_coeffs"]):.4f}')
        print('-' * 60)

    compute_classwise_accuracy(
        model=Backbone_model,
        dataloader=test_dataloader,
        num_classes=num_classes,
        device=device,
        save_path="classwise_accuracy.txt"
    )
    plot_alpha_prior_evolution(alpha_records, save_path="alpha_prior_dynamics.png")

    return best_metrics, Backbone_model


def evaluate(model, policy, dataloader, criterion, device, num_classes):
    """
    Evaluate the model and policy.

    Args:
        model (torch.nn.Module): Backbone model.
        policy (torch.nn.Module): Policy model.
        dataloader (DataLoader): Data loader for evaluation.
        criterion (callable): Loss function.
        device (torch.device): Device to use.
        num_classes (int): Number of classes.

    Returns:
        tuple: (ACC, ECE, MUE, RACC, val_loss)
    """
    model.eval()
    total = 0
    val_loss = 0.0
    class_true = 0
    class_false = 0
    all_U = []
    all_result = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            state = get_state_features(
                outputs.detach(),
                labels,
                num_classes=num_classes,
            ).to(device)
            alpha_prior, lambda_kl = policy(state)

            loss = criterion(
                outputs, labels, alpha_prior, lambda_kl
            )
            val_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)

            outputs = outputs + 1
            _, pred = torch.max(outputs, 1)
            U = num_classes / torch.sum(outputs, dim=1)
            result = (pred == labels)

            class_true += result.sum().item()
            class_false += (~result).sum().item()
            all_U.append(U.cpu().numpy())
            all_result.append(result.cpu().numpy())

        val_loss /= total
        ACC = class_true / (class_true + class_false)
        ECE = Get_ECE(model, dataloader, device, plot=False, figure_path=None)
        MUE, min_mue_threshold = Get_Best_MUE(all_U, all_result, plot=False)
        RACC, reject_rate = Get_Accuracy_With_Rejection(model, dataloader, device, num_classes, min_mue_threshold)

    model.train()
    return ACC, ECE, MUE, RACC, val_loss


def plot_training_curves(history, save_path):
    """
    Plot training curves.

    Args:
        history (dict): Training history.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(18, 12))

    plt.subplot(2, 2, 1)
    plt.plot(history['epoch'], history['kl_coeff'], 'b-o')
    plt.title('KL Coefficient Dynamics')
    plt.xlabel('Epoch'), plt.ylabel('KL Coefficient')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(history['epoch'], history['reward'], 'r-s')
    plt.title('Policy Network Reward')
    plt.xlabel('Epoch'), plt.ylabel('Average Reward')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(history['epoch'], history['train_acc'], 'g-', label='Train')
    plt.plot(history['epoch'], history['val_acc'], 'g--', label='Validation')
    plt.title('Accuracy Progress')
    plt.xlabel('Epoch'), plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend(), plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(history['epoch'], history['train_loss'], 'm-', label='Train')
    plt.plot(history['epoch'], history['val_loss'], 'm--', label='Validation')
    plt.title('Loss Dynamics')
    plt.xlabel('Epoch'), plt.ylabel('Loss')
    plt.legend(), plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()