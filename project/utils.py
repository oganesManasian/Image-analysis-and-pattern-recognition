from skimage.transform import resize
import numpy as np

import cv2


def box2image(image, box, image_size=(30, 30)):
    """
    Extracts objects from image bounded by box
    :param image:
    :param box:
    :param image_size: return image size
    :return: object bounded by box on original image
    """
    return image[box[1]:box[3], box[0]:box[2]]
    # return resize(image[box[1]:box[3], box[0]:box[2]], image_size)


def postprocess_predicted_sequence(seq):
    return seq
