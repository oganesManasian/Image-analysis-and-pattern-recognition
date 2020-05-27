import os
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from cnn.dataset_creation import inverse_color, to_binary

if os.getcwd().split('\\')[-1] == "project":
    WEIGHTS_PATH = "cnn/weights"
elif os.getcwd().split('\\')[-1] == "cnn":
    WEIGHTS_PATH = "weights"

WHITE = (255, 255, 255)


class Conv_Net(nn.Module):

    def __init__(self, nb_classes, nb_hidden=50):
        super(Conv_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(1600, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, nb_classes)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        self.softmax = torch.nn.Softmax(dim=1)

    # Creating the forward pass
    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def predict(self, input):
        output = self.softmax(self.forward(input))
        confidence, predicted = torch.max(output, dim=1)

        return predicted, confidence


class BaseClassifier:
    def predict(self, image):
        pass


class CNNClassifier(BaseClassifier):
    def __init__(self, data_type):
        self.data_type = data_type
        if data_type == "digits":
            nb_classes = 10
            weights_path = f"{WEIGHTS_PATH}/model_digits.pth"
            self.preprocess_image = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Lambda(inverse_color),
                transforms.Lambda(to_binary),
                transforms.ToTensor(),
                # transforms.Normalize((MEAN_DIGITS,), (STD_DIGITS,)),
            ])
        elif data_type == "operators":
            nb_classes = 5
            weights_path = f"{WEIGHTS_PATH}/model_operators.pth"
            self.preprocess_image = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Lambda(inverse_color),
                transforms.Lambda(to_binary),
                transforms.ToTensor(),
                # transforms.Normalize((MEAN_OPERATORS,), (STD_OPERATORS,)),
            ])
            self.prediction2label = {0: "/", 1: "=", 2: "-", 3: "*", 4: "+"}
        else:
            raise NotImplementedError

        # Build model
        self.model = Conv_Net(nb_classes=nb_classes)
        # Load weights
        self.model.load_state_dict(torch.load(weights_path))

    def predict(self, image, return_raw_label=False):
        """

        :param image: Image to classify as numpy array
        :return: predicted label
        """
        self.model.eval()

        # image = rgb2gray(image)
        # thresh = threshold_otsu(image)
        # binary = (image > thresh).astype(float)

        rotation_step = 10
        angles = [0 + i * rotation_step for i in range(360 // rotation_step)]

        predictions = []
        for angle in angles:
            image_rotated = Image.fromarray((image * 255).astype(np.uint8)) \
                .rotate(angle, fillcolor=WHITE)
            # plt.imshow(image_rotated)
            # plt.show()

            image_to_classify = self.preprocess_image(image_rotated)

            # plt.imshow(image_to_classify.reshape(28, 28), cmap="gray")
            # plt.show()

            pred, conf = self.model.predict(image_to_classify.unsqueeze(0))
            predictions.append([pred.item(), conf.item(), angle])
            # print(f"Angle {angle}, pred: {pred.item()}, conf {conf.item()}")

        predictions = np.array(predictions)
        highest_conf_ind = np.argmax(predictions[:, 1])
        print(f"Prediction: {predictions[highest_conf_ind, 0]}, "
              f"Highest conf: {predictions[highest_conf_ind, 1]}, "
              f"at angle: {predictions[highest_conf_ind, 2]}")

        if return_raw_label:  # For testing cnn performance
            if self.data_type == "digits":
                return predictions[highest_conf_ind, 0]
            else:
                return predictions[highest_conf_ind, 0]
        else:
            if self.data_type == "digits":
                return str(int(predictions[highest_conf_ind, 0]))
            else:
                return self.prediction2label[predictions[highest_conf_ind, 0]]


class FourierClasssifier(BaseClassifier):
    def __init__(self):
        raise NotImplementedError

    def predict(self, image):
        pass
