'''
Author: Jedidiah-Zhang yanzhe_zhang@protonmail.com
Date: 2025-05-07 00:03:25
LastEditors: Jedidiah-Zhang yanzhe_zhang@protonmail.com
LastEditTime: 2025-05-15 17:44:36
FilePath: /LS-PLL-Reproduction/codes/LeNet5.py
Description: LeNet5 model implementation.
'''

import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
        )
        self.fc = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.get_features(x)
        return self.fc(x)
    
    def get_features(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x