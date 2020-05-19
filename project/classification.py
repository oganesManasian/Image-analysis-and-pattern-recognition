import torch
from torch import nn
import torch.functional as F


class Conv_Net(nn.Module):

    def __init__(self, nb_hidden=50, nb_classes=15):
        super(Conv_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.fc1 = nn.Linear(1152, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, nb_classes)

    # Creating the forward pass
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class BaseClassifier:
    def predict(self, image):
        pass


class CNNClassifier(BaseClassifier):
    def __init__(self):
        # Build model
        self.model = Conv_Net()
        # Load weights

        pass

    def predict(self, image):
        pass


class FourierClasssifier(BaseClassifier):
    def __init__(self):
        pass

    def predict(self, image):
        pass
