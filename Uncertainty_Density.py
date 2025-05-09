import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def Plot_Uncertainty_Density(model, dataloader, device, num_classes, figure_path='results/Density.png'):
    """
    Plot the uncertainty density for correct and incorrect predictions.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): Device to run the model on (e.g., 'cuda' or 'cpu').
        num_classes (int): Number of classes in the dataset.
        figure_path (str, optional): Path to save the density plot. Defaults to 'results/Density.png'.
    """
    model.eval()
    correct_uncertainties = np.array([])  # Uncertainties for correct predictions
    incorrect_uncertainties = np.array([])  # Uncertainties for incorrect predictions

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            evidence = outputs  # Assuming the model outputs the evidence

            # Calculate alpha = evidence + 1 for each class
            alpha = evidence + 1
            S = torch.sum(alpha, dim=1, keepdim=True)  # Sum of alpha values for each sample

            # Calculate uncertainty U = C / S
            U = num_classes / S

            # Predicted class
            _, pred = torch.max(outputs, dim=1)

            # Separate correct and incorrect predictions
            correct_mask = (pred == labels).cpu().numpy()
            incorrect_mask = (pred != labels).cpu().numpy()
            U = U.cpu().numpy()
            correct = U[correct_mask].ravel()
            incorrect = U[incorrect_mask].ravel()

            # Concatenate uncertainties for correct and incorrect predictions
            correct_uncertainties = np.concatenate((correct_uncertainties, correct))
            incorrect_uncertainties = np.concatenate((incorrect_uncertainties, incorrect))

    # Plot the uncertainty density
    plt.figure(figsize=(8, 6))
    sns.kdeplot(correct_uncertainties, label="Correct Predictions", color="blue", shade=True, cut=0)
    sns.kdeplot(incorrect_uncertainties, label="Incorrect Predictions", color="red", shade=True, cut=0)
    plt.xlim(0, 1)  # Limit the range to 0-1
    plt.legend()
    plt.xlabel('Uncertainty')
    plt.ylabel('Density')
    plt.title('Uncertainty Density for Correct and Incorrect Predictions')
    plt.savefig(figure_path)
    plt.close()