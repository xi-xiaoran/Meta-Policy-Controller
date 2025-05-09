import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

def kl_divergence(alpha, num_classes, device=None, prior=None):

    # ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    ones = prior
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

class DS_EDL_Loss(nn.Module):
    def __init__(self, num_classes=10, device=None):
        super(DS_EDL_Loss, self).__init__()
        self.num_classes = num_classes
        self.device = device

    def forward(self, outputs, targets, alpha_prior, lambda_kl):
        batch_size = outputs.size(0)
        one_hot = F.one_hot(targets, num_classes=self.num_classes)
        evidence = outputs
        alpha = evidence + 1
        S = torch.sum(alpha,dim=1,keepdim=True)
        A = torch.sum(one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

        # 动态构造先验分布（核心修改）
        # prior = torch.ones(batch_size, self.num_classes, device=self.device).to(self.device)
        prior = alpha_prior.to(self.device)
        # for i in range(batch_size):
        #     non_gt_mask = torch.arange(self.num_classes).to(self.device) != targets[i]
        #     prior[i, non_gt_mask] = alpha_prior[0].to(self.device)  # 为每个样本设置非GT先验

        # annealing_coef = torch.min(
        #     torch.tensor(1, dtype=torch.float32),
        #     torch.tensor(self.epoch_num / 10, dtype=torch.float32),
        # )
        annealing_coef = lambda_kl

        kl_alpha = (alpha - 1) * (1 - one_hot) + 1
        kl_div = annealing_coef * kl_divergence(kl_alpha, self.num_classes, device=self.device, prior=prior)
        loss = torch.mean(A + kl_div)
        # loss = torch.mean(A)
        return loss