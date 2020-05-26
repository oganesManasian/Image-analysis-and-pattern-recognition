import cv2
from PIL import ImageDraw, ImageFont, Image
import numpy as np


def read_video(filename):
    frames = []

    cap = cv2.VideoCapture(filename)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Reached end of video
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    return frames


def annotate_frames(frames, seq, expression_result):
    reversed_seq = list(reversed(seq))
    complete_expression = ""
    new_frames = []
    for i in range(len(frames)):
        # Adding char to expression if already detected
        if len(reversed_seq) > 0 and reversed_seq[-1][1] == i:
            complete_expression += reversed_seq.pop()[0]

        # Adding the final result after the final equal sign
        elif len(reversed_seq) == 0 and len(complete_expression) == len(seq):
            complete_expression += str(expression_result)

        # Writing on image
        new_frame = write_on_frame(frames[i], complete_expression)
        new_frames.append(new_frame)

    return new_frames


def write_on_frame(frame, text):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype("fonts/Montserrat-Bold.otf", 30)
    fill_color = (0, 0, 0)  # Making font black
    draw.text((img.width // 4, int(img.height * 0.85)), text,
              font=font, fill=fill_color)

    return np.array(img)


def frames2video(frames, filename):
    height, width, layers = frames[0].shape

    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'DIVX'), 5, (width, height))

    for i in range(len(frames)):
        out.write(frames[i])
    out.release()
