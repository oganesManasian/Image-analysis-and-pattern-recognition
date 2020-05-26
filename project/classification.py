import torch
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

from cnn.dataset_creation import MEAN_DIGITS, STD_DIGITS

CNN_PATH = "cnn"


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
        # predicted = torch.argmax(output, dim=1)
        confidence, predicted = torch.max(output, dim=1)

        return predicted, confidence


class BaseClassifier:
    def predict(self, image):
        pass


class CNNClassifier(BaseClassifier):
    def __init__(self):
        # Build model
        self.model = Conv_Net(nb_classes=10)
        # Load weights
        self.model.load_state_dict(torch.load(f"{CNN_PATH}/cnn_model.pth"))
        # self.model.eval()

    def predict(self, image):
        """

        :param image: Image to classify as numpy array
        :return: predicted label
        """
        image = rgb2gray(image)
        thresh = threshold_otsu(image)
        binary = (image > thresh).astype(float)

        rotation_step = 10
        angles = [0 + i * rotation_step for i in range(360 // rotation_step)]

        predictions = []
        for angle in angles:
            image_rotated = Image.fromarray((binary * 255).astype(np.uint8)) \
                .rotate(angle, fillcolor=(255))
            # plt.imshow(image_rotated)
            # plt.show()

            preprocess = transforms.Compose([
                # transforms.Grayscale(num_output_channels=1),
                # transforms.Lambda(inverse_color),
                # transforms.Lambda(to_binary),
                transforms.ToTensor(),
                # transforms.Normalize((MEAN_DIGITS,), (STD_DIGITS,)),
            ])
            image_to_classify = preprocess(image_rotated)

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

        # Calculates sum of probabilities of each digit among different rotations
        # prediction_scores = defaultdict(list)
        # for i in range(predictions.shape[0]):
        #     prediction_scores[predictions[i, 0]].append(predictions[i, 1])
        #
        # prediction_scores_sum = np.array([[key, sum(prediction_scores[key])]
        #                                   for key in prediction_scores.keys()])
        # # print(sorted(prediction_scores_sum, key=lambda x: x[1], reverse=True))
        # prediction, confidence = sorted(prediction_scores_sum, key=lambda x: x[1], reverse=True)[0]
        # print(f"Prediction (sum): {prediction}, Confidence (sum): {confidence}")

        return predictions[highest_conf_ind, 0]


class FourierClasssifier(BaseClassifier):
    def __init__(self):
        pass

    def predict(self, image):
        pass
