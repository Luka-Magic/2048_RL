import torch
from torch import nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        c, h, w = input_size  # (16, 4, 4)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=2, stride=1, padding=0),
        )

        self.values = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        v = self.values(x)
        return v