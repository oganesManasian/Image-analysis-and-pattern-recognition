import cv2
import numpy as np

from skimage.filters import gaussian
from skimage.morphology import binary_erosion, binary_dilation, binary_opening, binary_closing, disk, square
from skimage.feature import canny
from skimage.color import rgb2gray
from skimage.transform import resize


def get_robot_locations(frames, method="frame_differencing", return_centers=True):
    """
    Tracks robot positions at each frame
    :param method:
    :param frames:
    :param return_centers: If True alg returns robot's centers, else returns bounding boxes
    :return:
    """
    if method == "frame_differencing" and return_centers:
        return frame_differencing(frames)
    elif method == "red_channel_tracking" and return_centers:
        return red_channel_tracking(frames)
    else:
        raise NotImplementedError


def red_channel_tracking(frames, red_threshold = 100, green_threshold = 100, blue_threshold = 100):
    frames_red = [(frame[:, :, 0] > red_threshold) &
                  (frame[:, :, 1] < green_threshold) &
                  (frame[:, :, 2] < blue_threshold) for frame in frames]
    robot_locations = [get_center(frame, threshold=0) for frame in frames_red]

    return robot_locations


def frame_differencing(frames, blur_sigma=1, change_threshold=0.5):
    # Get differences in frames
    frame_changes = []
    for i in range(1, len(frames)):
        frame1 = gaussian(frames[i - 1], sigma=blur_sigma, multichannel=True)
        frame2 = gaussian(frames[i], sigma=blur_sigma, multichannel=True)
        changes = np.linalg.norm(frame1 - frame2, axis=2)
        frame_changes.append(changes)

    # Calculate robot's center
    robot_locations = [get_center(changes, threshold=change_threshold) for changes in frame_changes]

    # Process case when there are no changes in frames
    robot_locations = postprocess_locations(robot_locations)

    return robot_locations


def get_center(image, threshold):
    """
    Thresholds image and then returns center of nonzeros
    :param image:
    :param threshold:
    :return:
    """
    Y, X = np.where(image > threshold)
    if len(X) == 0:
        return None
    return np.mean(X), np.mean(Y)


def postprocess_locations(robot_locations):
    if robot_locations[0] is None:
        # Fill initial position
        initial_position = None
        for loc in robot_locations:
            if loc:
                initial_position = loc
                break

        assert (initial_position is not None)
        robot_locations[0] = initial_position

    # Fill remaining positions
    last_position = robot_locations[0]
    for i in range(1, len(robot_locations)):
        if robot_locations[i] is None:
            robot_locations[i] = last_position
        else:
            last_position = robot_locations[i]
    return robot_locations
