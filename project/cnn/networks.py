import torch
from torch import nn


class ConvNetSmall(nn.Module):

    def __init__(self, nb_classes, nb_hidden=50):
        super(ConvNetSmall, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.fc1 = nn.Linear(800, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, nb_classes)

        self.dropout = torch.nn.Dropout2d(p=0.25)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        self.softmax = torch.nn.Softmax(dim=1)

    # Creating the forward pass
    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def predict(self, input):
        output = self.softmax(self.forward(input))
        confidence, predicted = torch.max(output, dim=1)

        return predicted, confidence


class ConvNet(nn.Module):

    def __init__(self, nb_classes, nb_hidden=100):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3)

        self.fc1 = nn.Linear(2048, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, nb_classes)

        self.dropout = torch.nn.Dropout2d(p=0.25)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        self.softmax = torch.nn.Softmax(dim=1)

    # Creating the forward pass
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))

        x = self.relu(self.conv3(x))
        x = self.pool(self.relu(self.conv4(x)))

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def predict(self, input):
        output = self.softmax(self.forward(input))
        confidence, predicted = torch.max(output, dim=1)

        return predicted, confidence