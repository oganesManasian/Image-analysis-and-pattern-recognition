import cv2


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
