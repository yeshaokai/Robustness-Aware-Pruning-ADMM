import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['AlexNet_BN', 'alexnet_bn']



class AlexNet_BN(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet_BN, self).__init__()
        '''
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.conv1_bn = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5,groups=2,padding=2)
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(384)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(384)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc1_bn = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc2_bn = nn.BatchNorm1d(4096)
        self.fc3 = nn.Linear(4096, num_classes)
        '''
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            #self.conv1,
            nn.BatchNorm2d(96),
            #self.conv1_bn,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5,groups=2,padding=2),
            #self.conv2,
            nn.BatchNorm2d(256),
            #self.conv2_bn,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            #self.conv3,
            nn.BatchNorm2d(384),
            #self.conv3_bn,
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            #self.conv4,
            nn.BatchNorm2d(384),
            #self.conv4_bn,
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            #self.conv5,
            nn.BatchNorm2d(256),
            #self.conv5_bn,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            #self.fc1,
            nn.BatchNorm1d(4096),
            #self.fc1_bn,
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            #self.fc2,
            nn.BatchNorm1d(4096),
            #self.fc2_bn,
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            #self.fc3
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    """
    No pretrained model for bn alexnet

    """
    model = AlexNet_BN(**kwargs)

    return model
