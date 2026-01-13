import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import os

import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath
from torchinfo import summary

torch.manual_seed(5)
minEpoch = 200
maxEpoch = 100
maxWait = 5
#原始batchsize是128
batchSize = 64
lr = 3e-4
saveModel = True
kernelSize = 3
channel = 64
layers = 20

# net
class DnCNN(nn.Module):
    def __init__(self):
        super().__init__()
        padding = (kernelSize - 1) // 2
        self.input = nn.Sequential(
            nn.Conv1d(1, channel, kernelSize, padding=padding, bias=False),
            nn.ReLU(inplace=True))
        middle = []
        for _ in range(layers - 2):
            middle.append(nn.Conv1d(channel, channel, kernelSize, padding=padding, bias=False))
            middle.append(nn.BatchNorm1d(channel))
            middle.append(nn.ReLU(inplace=True))
        self.middle = nn.Sequential(*middle)
        self.out = nn.Conv1d(channel, 1, kernelSize, padding=padding, bias=False)

    def forward(self, x):
        return x - self.out(self.middle(self.input(x)))


if __name__ == '__main__':
    model = DnCNN()
    model.eval()
    print('------------------- training-time model -------------')
    print(model)



    summary(model, input_size=(1,1,10000))


