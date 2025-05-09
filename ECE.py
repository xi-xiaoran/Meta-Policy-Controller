import numpy as np
import torch
import matplotlib.pyplot as plt


# Expected Calibration Error (ECE)
def Get_ECE(model, dataloader, device, plot=False, figure_path='results/ECE.png', num_bins=5):
    """
    Calculate the Expected Calibration Error (ECE) of a model.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): Device to run the model on (e.g., 'cuda' or 'cpu').
        plot (bool, optional): Whether to plot the calibration curve. Defaults to False.
        figure_path (str, optional): Path to save the calibration plot. Defaults to 'results/ECE.png'.
        num_bins (int, optional): Number of bins for calibration. Defaults to 5.

    Returns:
        float: The Expected Calibration Error (ECE) value.
    """
    model.eval()
    correct = torch.zeros(num_bins).to(device)  # Correct predictions for each bin
    total = torch.zeros(num_bins).to(device)  # Total samples for each bin
    bin_boundaries = np.linspace(0, 1, num_bins + 1)  # Confidence bin boundaries
    confidences = []  # Store confidence values
    accuracies = []  # Store accuracy values

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            alpha = outputs + 1  # Evidence to alpha conversion
            S = torch.sum(alpha, dim=1, keepdim=True)  # Sum of alpha values
            probs = alpha / S  # Normalized probabilities

            # Get the maximum probability (confidence) for each sample
            confidence, predictions = torch.max(probs, dim=1)

            # Determine if each prediction is correct
            correct_preds = (predictions == labels).float()

            # Assign samples to bins based on confidence
            for i in range(num_bins):
                bin_min = bin_boundaries[i]
                bin_max = bin_boundaries[i + 1]
                bin_mask = (confidence >= bin_min) & (confidence < bin_max)

                # Calculate the number of correct predictions and total samples in each bin
                correct[i] += correct_preds[bin_mask].sum()
                total[i] += bin_mask.sum()

            confidences.extend(confidence.cpu().numpy())
            accuracies.extend(correct_preds.cpu().numpy())

    # Calculate the average accuracy and confidence for each bin
    bin_accuracies = torch.zeros(num_bins).to(device)
    for i in range(num_bins):
        if total[i] > 0:
            bin_accuracies[i] = correct[i] / total[i]
        else:
            bin_accuracies[i] = 0.0  # Set accuracy to 0 if no samples in the bin
    bin_accuracies = bin_accuracies.cpu().numpy()
    total = total.cpu().numpy()

    # Calculate ECE
    if total.sum() > 0:
        ece = np.sum((total / total.sum()) * np.abs(bin_accuracies - bin_boundaries[:-1]))
    else:
        ece = 0.0  # Set ECE to 0 if no samples in any bin

    if plot:
        # Plot the calibration curve
        plt.figure(figsize=(8, 6))
        plt.bar(bin_boundaries[:-1], bin_accuracies, width=0.1, align='edge', alpha=0.7, color='blue')
        plt.xlabel('Confidence Interval')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Confidence Interval')
        plt.xticks(np.linspace(0, 1, num_bins + 1))
        plt.savefig(figure_path)
        plt.close()

    return ece.item()