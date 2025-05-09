import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Beta

def get_state_features(e, target, num_classes, epoch=None, num_epochs=None, loss=None, val_acc=None, hist_kl=None):
    """
    Construct state features for reinforcement learning (dimension expansion).

    Args:
        e (torch.Tensor): Evidence tensor.
        target (torch.Tensor): Target labels.
        num_classes (int): Number of classes.
        epoch (int, optional): Current epoch. Defaults to None.
        num_epochs (int, optional): Total number of epochs. Defaults to None.
        loss (float, optional): Current training loss. Defaults to None.
        val_acc (float, optional): Recent validation accuracy. Defaults to None.
        hist_kl (float, optional): Historical average KL divergence. Defaults to None.

    Returns:
        torch.Tensor: State features tensor.
    """
    batch_size = e.size(0)
    alpha = e + 1  # Add 1 to ensure alpha is positive
    S = torch.sum(alpha, dim=1, keepdim=True)  # Sum of alpha values
    P = alpha / S  # Normalized alpha values

    # Basic features
    acc = (P.argmax(dim=1) == target).float().mean().item()  # Accuracy
    entropy = -torch.sum(P * torch.log(P + 1e-8), dim=1).mean().item()  # Entropy
    mean_evidence = e.mean().item()  # Mean evidence

    # Features for incorrect samples
    wrong_mask = (P.argmax(dim=1) != target)  # Mask for incorrect predictions
    u_wrong = (num_classes / S.squeeze())[wrong_mask].mean().item() if wrong_mask.any() else 1.0  # Uncertainty for incorrect samples
    conf_wrong = P.max(dim=1).values[wrong_mask].mean().item() if wrong_mask.any() else 0.0  # Confidence for incorrect samples

    # Additional feature dimensions
    epoch_feat = epoch / num_epochs if epoch else 0.0  # Training progress
    loss_feat = loss if loss else 0.0  # Current training loss
    val_acc_feat = val_acc if val_acc else 0.0  # Recent validation accuracy
    hist_kl_feat = hist_kl if hist_kl else 0.0  # Historical average KL divergence

    return torch.tensor([
        acc, mean_evidence,
        epoch_feat, loss_feat, val_acc_feat, hist_kl_feat
    ], dtype=torch.float32).unsqueeze(0)

class PolicyNetwork(nn.Module):
    """
    Policy network with two output branches: alpha_prior and lambda_kl.
    """
    def __init__(self, state_dim, num_classes):
        """
        Initialize the policy network.

        Args:
            state_dim (int): Dimension of the state features.
            num_classes (int): Number of classes.
        """
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, 128),  # Expand network capacity
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # Prior parameter branch (use Softplus to ensure positive values)
        self.alpha_head = nn.Sequential(
            nn.Linear(64, num_classes),  # Generate independent parameters for each non-GT class
            nn.Sigmoid()
        )
        self.lambda_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, state):
        """
        Forward pass of the policy network.

        Args:
            state (torch.Tensor): State features tensor.

        Returns:
            tuple: alpha_prior and lambda_kl tensors.
        """
        shared = self.shared_net(state)
        alpha_prior = self.alpha_head(shared) + 1.0  # Ensure alpha >= 1
        lambda_kl = self.lambda_head(shared) * 10.0 + 1e-8  # Ensure lambda is in [0, 10]
        return alpha_prior, lambda_kl.squeeze()