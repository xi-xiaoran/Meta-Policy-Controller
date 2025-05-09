import torch
import torch.nn as nn
from torchvision.models import resnet18,resnet34,resnet50,resnet101
import torch.nn.functional as F


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.resnet18 = resnet18(pretrained=False, num_classes=num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.resnet18(x)
        x = self.relu(x)
        return x

class ResNet34(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet34, self).__init__()
        self.resnet34 = resnet34(pretrained=False, num_classes=num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.resnet34(x)
        x = self.relu(x)
        return x

class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        self.resnet50 = resnet50(pretrained=False, num_classes=num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.resnet50(x)
        x = self.relu(x)
        return x

class ResNet101(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet101, self).__init__()
        self.resnet101 = resnet101(pretrained=False, num_classes=num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.resnet101(x)
        x = self.relu(x)
        return x