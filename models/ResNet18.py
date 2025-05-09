import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.resnet18 = resnet18(pretrained=False, num_classes=num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.resnet18(x)
        return x
