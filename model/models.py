import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import transforms, utils

import numpy as np
from PIL import Image
import os
import math
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReduceBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2, kernel_size=3):
        super(ReduceBlock, self).__init__()
        self.padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=scale, padding=self.padding)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=self.padding)
        self.act2 = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        return x

class YOLOModel(nn.Module):
    def __init__(self, num_classes=2):
        super(YOLOModel, self).__init__()
        output_size = 5 + num_classes
        self.num_classes = num_classes

        self.features = nn.Sequential(
            ReduceBlock(3, 24, scale=2, kernel_size=3),
            ReduceBlock(24, 48, scale=2, kernel_size=3),
            ReduceBlock(48, 96, scale=2, kernel_size=3),
            ReduceBlock(96, 128, scale=2, kernel_size=3),
            # ReduceBlock(128, 140, scale=2, kernel_size=3),
            # ReduceBlock(128, 192, scale=2, kernel_size=3),
        )

        self.prediction = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, output_size, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.prediction(x)
        return x
