import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

def kl_divergence(alpha, num_classes, device=None):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl

class EDL_Loss(nn.Module):
    def __init__(self, num_classes=10, alpha=1, device=None):
        super(EDL_Loss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.device = device

    def forward(self, outputs, targets):
        one_hot = F.one_hot(targets, num_classes=self.num_classes)
        evidence = outputs
        alpha = evidence + 1
        S = torch.sum(alpha,dim=1,keepdim=True)
        u = self.num_classes / S
        A = torch.sum(one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

        # annealing_coef = torch.min(
        #     torch.tensor(1, dtype=torch.float32),
        #     torch.tensor(self.epoch_num / 10, dtype=torch.float32),
        # )
        annealing_coef = self.alpha

        kl_alpha = (alpha - 1) * (1 - one_hot) + 1
        kl_div = annealing_coef * kl_divergence(kl_alpha, self.num_classes, device=self.device)
        P_pred = torch.log(torch.mean(one_hot * evidence))
        RED = - torch.mean(u) * P_pred
        # loss = torch.mean(A + kl_div) + 0.1 * RED
        loss = torch.mean(A + kl_div)
        return loss