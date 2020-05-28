import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from cnn.networks import ConvNetSmall, ConvNet
from cnn.dataset_creation import inverse_color, to_binary

if os.getcwd().split('\\')[-1] == "project":
    WEIGHTS_PATH = "cnn/weights180"
elif os.getcwd().split('\\')[-1] == "cnn":
    WEIGHTS_PATH = "weights180"

WHITE = (255, 255, 255)


class BaseClassifier:
    def predict(self, image):
        pass


class CNNClassifier(BaseClassifier):
    def __init__(self, data_type, method='manual'):
        """
        Initializes model with pretrained weights
        :param data_type: Type of data to work on. Either 'digits' or 'operators'
        """
        self.data_type = data_type
        if data_type == "digits":
            nb_classes = 10
            weights_path = f"{WEIGHTS_PATH}/model_digits.pth"
            self.model = ConvNet(nb_classes=nb_classes)
        elif data_type == "operators":
            nb_classes = 5
            self.model = ConvNetSmall(nb_classes=nb_classes)
            weights_path = f"{WEIGHTS_PATH}/model_operators.pth"
            self.prediction2label = {0: "/", 1: "=", 2: "-", 3: "*", 4: "+"}
        else:
            raise NotImplementedError

        # Load weights
        self.model.load_state_dict(torch.load(weights_path))

        self.preprocess_image = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Lambda(inverse_color),
            transforms.Lambda(lambda img: to_binary(img, method)),
            # TODO use instead of resizing in box2image function
            # Issue: Fills with opposite colors
            # transforms.Resize((28, 28)),
            transforms.ToTensor(),
            # transforms.Normalize((MEAN,), (STD,)),
        ])

    def predict(self, image, return_raw_label=False, print_info=False):
        """
        Predicts class with use of 36 rotations of input image
        :param print_info: If True prints
        :param return_raw_label:
        :param image: Image to classify as numpy array
        :return: predicted class as string
        """
        image = Image.fromarray((image * 255).astype(np.uint8))
        background_color = image.getpixel((0, 0))

        with torch.no_grad():
            self.model.eval()

            rotation_step = 10
            angles = [0 + i * rotation_step for i in range(360 // rotation_step)]

            predictions = []
            for angle in angles:
                image_rotated = image.rotate(angle, fillcolor=background_color)

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
        if print_info:
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
