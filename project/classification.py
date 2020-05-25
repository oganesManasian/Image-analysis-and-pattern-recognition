import torch
from torch import nn
import torch.functional as F


# from keras.regularizers import L1L2
# from keras.layers import Conv2D
# Creating a Net class object, which consists of 2 convolutional layers, max-pool layers and fully-connected layers
class Conv_Net(nn.Module):

    def __init__(self, nb_hidden=50):
        super(Conv_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)  # the first convolutional layer, which processes the input image
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)  # the second convolutional layer, which gets the max-pooled set
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)  # the second convolutional layer, which gets the max-pooled set

        self.fc1 = nn.Linear(576, nb_hidden)  # the first fully-connected layer, which gets flattened max-pooled set
        self.fc2 = nn.Linear(nb_hidden, 15)  # the second fully-connected layer that outputs the result

    # Creating the forward pass
    def forward(self, x):

        # The first two layers
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))

        # The second two layers
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))

        x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=2))

        # Flattening the data set for fully-connected layer
        x = x.view(x.size(0), -1)

        # The first fully-connected layer
        x = F.relu(self.fc1(x))

        x = nn.Dropout(p=0.3)(x)

        # The second full-connected layer
        x = self.fc2(x)

        return x


class BaseClassifier:
    def predict(self, image):
        pass


class CNNClassifier(BaseClassifier):
    def __init__(self, data_type):
        self.data_type = data_type

        # Build model
        self.model = Conv_Net()

        # Load weights
        if self.data_type == "digits":
            self.model.load_state_dict(torch.load("digit_model"))
        elif self.data_type == "operators":
            self.model.load_state_dict(torch.load("operator_model"))
        else:
            raise ValueError

        # In case of operator return operator as string
        if self.data_type == "operators":
            self.pred2oper = {0: "/", 1: "=", 2: "-", 3: "*", 4: "+"}

    def predict(self, image):
        class_ = self.model(image).argmax()

        if self.data_type == "digits":
            return class_
        elif self.data_type == "operators":
            mapping = {0: "/", 1: "=", 2: "-", 3: "*", 4: "+"}
            return mapping[class_]


class FourierClasssifier(BaseClassifier):
    def __init__(self):
        pass

    def predict(self, image):
        pass
