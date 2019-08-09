'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class LeNet_adv(nn.Module):
    def __init__(self,w = 1):
        super(LeNet_adv, self).__init__()
        self.w = int(w)        
        self.conv1 = nn.Conv2d(3, 6*self.w, 5)
        self.conv2 = nn.Conv2d(6*self.w, 16*self.w, 5)
        self.fc1   = nn.Linear(16*5*5*self.w, 120*self.w)
        self.fc2   = nn.Linear(120*self.w, 84*self.w)
        self.fc3   = nn.Linear(84*self.w, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class LeNet(nn.Module):
    def __init__(self,w = 1):
        super(LeNet, self).__init__()
        self.w = int(w)        
        self.conv1 = nn.Conv2d(3, 6*self.w, 5)
        self.conv2 = nn.Conv2d(6*self.w, 16*self.w, 5)
        self.fc1   = nn.Linear(16*5*5*self.w, 120*self.w)
        self.fc2   = nn.Linear(120*self.w, 84*self.w)
        self.fc3   = nn.Linear(84*self.w, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
