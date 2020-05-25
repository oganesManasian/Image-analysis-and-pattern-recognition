from skimage.transform import resize
import numpy as np
from PIL import ImageDraw, ImageFont, Image
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

def write_on_frame(frame, text):
    img = Image.fromarray(frame)

    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype("montserrat/Montserrat-Bold.otf", 20)
    draw.text((5, img.height - 25), text, font=font, fill=(255,255,255))

    return np.array(img)

def create_video(frames, filename):
    height, width, layers = frames[0].shape

    out = cv2.VideoWriter('{}.avi'.format(filename), cv2.VideoWriter_fourcc(*'DIVX'), 5, (width, height))

    for i in range(len(frames)):
        out.write(frames[i])
    out.release()
