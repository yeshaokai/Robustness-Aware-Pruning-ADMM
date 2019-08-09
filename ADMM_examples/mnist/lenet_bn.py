from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
from numpy import linalg as LA
import sys

class LeNet_BN(nn.Module):
    def __init__(self,rho = 0.001):
        super(LeNet_BN, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv1_bn = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.conv2_bn = nn.BatchNorm2d(50)        
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc1_bn = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        #x = F.relu(self.conv1(x))
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)        
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = self.conv2_bn(x)        
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
