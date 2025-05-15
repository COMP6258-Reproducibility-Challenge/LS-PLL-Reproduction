'''
Author: Jedidiah-Zhang yanzhe_zhang@protonmail.com
Date: 2025-05-07 00:05:12
LastEditors: Jedidiah-Zhang yanzhe_zhang@protonmail.com
LastEditTime: 2025-05-10 17:16:39
FilePath: /LS-PLL-Reproduction/codes/ResNet18.py
Description: ResNet18 model implementation.
'''

import torch.nn as nn
from torchvision.models import resnet18

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        # Load the ResNet18 model directly from torchvision without pre-trained weights
        self.resnet = resnet18(weights=None)
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        # Modify the classifier to match the number of classes
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.get_features(x)
        return self.fc(x)
    
    def get_features(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)
