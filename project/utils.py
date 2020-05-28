import os
import shutil

from skimage.io import imsave
from skimage.transform import resize


def box2image(image, box, image_size=(28, 28)):
    """
    Extracts objects from image bounded by box
    :param image:
    :param box:
    :param image_size: return image size
    :return: object bounded by box on original image
    """
    # return image[box[1]:box[3], box[0]:box[2]]
    return resize(image[box[1]:box[3], box[0]:box[2]], image_size)


def postprocess_predicted_sequence(seq):
    """
    Fixes some mistakes made by model using domain knowledge
    :param seq: predicts expression
    :return: fixed expression
    """
    # 9s to 6s
    for i in range(len(seq)):
        pred, frame_id = seq[i]
        if pred == "9":
            seq[i] = "6", frame_id

    # Last operator have to be =
    if seq[-1][0] != '=':
        print("Warning: equal sign!")
        seq[-1] = ('=', seq[-1][1])

    # Substitute all equal signs at wrong position by '+'
    for i in range(len(seq) - 1):
        if seq[i][0] == "=":
            print("Warning: equal sign")
            seq[i] = ("+", seq[i][1])

    # Before equal sign there must be a digit
    if not seq[-2][0].isdigit():
        seq[-2] = ('1', seq[-2][1])
    return seq


def create_video_dataset(initial_image, object_boxes, video_dataset_path="cnn/video_dataset"):
    """
    Creates dataset of digits and operators from the video
    :param initial_image:
    :param object_boxes:
    :param video_dataset_path: path where to save dataset
    :return:
    """
    if os.path.isdir(video_dataset_path):
        shutil.rmtree(video_dataset_path)
    os.mkdir(video_dataset_path)

    digits_path = "digits"
    os.mkdir(os.path.join(video_dataset_path, digits_path))
    digit_labels = list(map(str, range(10)))

    operators_path = "operators"
    os.mkdir(os.path.join(video_dataset_path, operators_path))
    operator_labels = ['plus', 'minus', 'multiply', 'divide', 'equal']

    for label in digit_labels + operator_labels:
        os.mkdir(os.path.join(video_dataset_path,
                              digits_path if label in digit_labels else operators_path,
                              label))

    digits_ind = [29, 28, 20, 4, 6, 17]
    digit_targets = list(map(str, [3, 2, 7, 2, 3, 7]))
    operators_ind = [25, 35, 12, 14]
    operator_targets = ['divide', 'plus', 'multiply', 'equal']

    boxes_to_save = [object_boxes[i] for i in digits_ind + operators_ind]
    objects_to_save = [box2image(initial_image, box) for box in boxes_to_save]

    for i, (image, label) in enumerate(zip(objects_to_save, digit_targets + operator_targets)):
        # image = rgb2gray(image)
        # thresh = threshold_otsu(image)
        # binary = (image > thresh).astype(float)
        imsave(os.path.join(video_dataset_path,
                            digits_path if label in digit_labels else operators_path,
                            label,
                            f"objects{i}.png"),
               image)
