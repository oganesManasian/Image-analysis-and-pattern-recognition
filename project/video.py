import cv2
from PIL import ImageDraw, ImageFont, Image
import numpy as np
import os
from robot_tracking import get_robot_locations

def read_video(filepath):
    frames = []

    assert os.path.isfile(filepath) and "Can't find input filename"
    cap = cv2.VideoCapture(filepath)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Reached end of video
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    assert frames and "Empty frames list"
    return frames


def annotate_frames(frames, seq, expression_result, trajectory):
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
        new_frame = write_on_frame(frames[i], complete_expression, trajectory[:i])
        new_frames.append(new_frame)

    return new_frames


def write_on_frame(frame, text, line):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype("fonts/Montserrat-Bold.otf", 30)
    fill_color = (0, 0, 0)  # Making font black
    draw.text((img.width // 4, int(img.height * 0.85)), text,
              font=font, fill=fill_color)

    red = (255, 0, 0)
    draw.line(xy=line, width=1, fill=red)

    return np.array(img)


def frames2video(frames, filepath):
    height, width, layers = frames[0].shape

    out = cv2.VideoWriter(filepath,
                          cv2.VideoWriter_fourcc(*'DIVX'),
                          2,
                          (width, height))

    for i in range(len(frames)):
        out.write(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
    out.release()
