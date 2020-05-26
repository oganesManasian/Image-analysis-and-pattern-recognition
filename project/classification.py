import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import PIL
import numpy as np
import collections

binary_image = False

operators_mean, operators_std = 0.2031257599592209, 0.279224157333374
minst_mean, minst_std = 0.09182075411081314, 0.195708230137825

def inverse_color(img):
    return PIL.Image.eval(img, lambda val: 255 - val)


def remove_background(img):
    return PIL.Image.eval(img, lambda val: 0 if val < (256 / 2) else val)


def to_binary(img):
    return PIL.Image.eval(img, lambda val: 255 if val < (256 / 2) else 0)

# from keras.regularizers import L1L2
# from keras.layers import Conv2D
# Creating a Net class object, which consists of 2 convolutional layers, max-pool layers and fully-connected layers
class Conv_Net(nn.Module):

    def __init__(self, nb_hidden=100, nb_conv3=128, nb_out=10):
        super(Conv_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)  # the first convolutional layer, which processes the input image
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)  # the second convolutional layer, which gets the max-pooled set
        self.conv3 = nn.Conv2d(32, nb_conv3, kernel_size=3)  # the second convolutional layer, which gets the max-pooled set

        self.fc1 = nn.Linear(nb_conv3, nb_hidden)  # the first fully-connected layer, which gets flattened max-pooled set
        self.fc2 = nn.Linear(nb_hidden, nb_out)  # the second fully-connected layer that outputs the result

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

        # Load weights
        if self.data_type == "digits":
            # Build model
            self.model = Conv_Net(nb_hidden=100, nb_conv3=128, nb_out=10)
            self.model.load_state_dict(torch.load("digit_model_binary"))
            self.model.eval()
        elif self.data_type == "operators":
            self.model = Conv_Net(nb_hidden=25, nb_conv3=64, nb_out=5)
            self.model.load_state_dict(torch.load("operator_model"))
            self.model.eval()
        else:
            pass
            # raise ValueError

    def get_transforms(self, binary=False):
        side = 28 if self.data_type == "digits" else 25
        mean, std = (minst_mean, minst_std) if self.data_type == "digits" else (operators_mean, operators_std)

        all_transforms = [
            transforms.Grayscale(num_output_channels=1),
            transforms.Lambda(remove_background),
            transforms.CenterCrop(side),
            # transforms.Resize((side, side)),
        ]

        if binary:
            all_transforms.append(transforms.Lambda(to_binary))

        all_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize((mean, ), (std, ))
        ])

        return all_transforms


    def predict(self, image):

        pil_img = PIL.Image.fromarray((image * 255).astype(np.uint8))
        # pil_img.show()
        binary = self.data_type == "digits"
        tensor = transforms.Compose(self.get_transforms(binary))(pil_img).unsqueeze(dim=0)

        res = []
        for i in range(500):
            res.append(self.model(tensor).argmax().item())
        counter = collections.Counter(res)
        class_ = max(counter, key=lambda x: counter[x])

        if self.data_type == "digits":
            return str(class_)
        elif self.data_type == "operators":
            mapping = {0: "/", 1: "=", 2: "-", 3: "*", 4: "+"}
            return mapping[class_]


class FourierClasssifier(BaseClassifier):
    def __init__(self):
        pass

    def predict(self, image):
        pass
