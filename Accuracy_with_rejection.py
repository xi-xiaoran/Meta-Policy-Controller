import torch

# Calculate accuracy with rejection
def Get_Accuracy_With_Rejection(model, dataloader, device, num_classes, threshold=0.5):
    """
    Calculate the accuracy of a model with rejection based on uncertainty.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): Device to run the model on (e.g., 'cuda' or 'cpu').
        num_classes (int): Number of classes in the dataset.
        threshold (float, optional): Uncertainty threshold for rejection. Defaults to 0.5.

    Returns:
        tuple: (accuracy, rejection_rate)
            - accuracy (float): Accuracy of the model after rejecting uncertain samples.
            - rejection_rate (float): Proportion of samples rejected due to high uncertainty.
    """
    model.eval()
    correct = 0  # Number of correctly classified samples
    total = 0  # Total number of samples
    rejected = 0  # Number of rejected samples
    total_rejected = 0  # Total number of samples not rejected

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

            # Calculate accuracy
            correct_mask = (pred == labels)  # Mask for correctly classified samples
            reject_mask = (U > threshold)  # Mask for samples with uncertainty greater than the threshold
            reject_mask = reject_mask.squeeze(1)

            # Count the number of rejected samples
            rejected += reject_mask.sum().item()

            # Count the number of correctly classified samples that are not rejected
            correct += ((correct_mask) & (~reject_mask)).sum().item()

            # Count the number of samples not rejected
            total_rejected += (~reject_mask).sum().item()
            total += inputs.size(0)

    # Calculate final accuracy and rejection rate
    accuracy = correct / total_rejected if total_rejected > 0 else 0
    rejection_rate = rejected / total if total > 0 else 0

    # Uncomment the following lines to print the results
    # print(f'Accuracy after rejection: {accuracy * 100:.2f}%')
    # print(f'Rejection rate: {rejection_rate * 100:.2f}%')

    return accuracy, rejection_rate