import cv2
import numpy as np

from skimage.filters import gaussian
from skimage.morphology import binary_erosion, binary_dilation, binary_opening, binary_closing, disk, square
from skimage.feature import canny
from skimage.color import rgb2gray



def get_robot_locations(frames, method="auto", return_centers=True):
    """
    Tracks robot positions at each frame
    :param method:
    :param frames:
    :param return_centers: If True alg returns robot's centers, else returns bounding boxes
    :return:
    """
    if method == "frame_differencing" and return_centers:
        robot_locations = frame_differencing(frames)
    elif method == "red_channel_tracking" and return_centers:
        robot_locations = red_channel_tracking(frames)
    elif method == "auto" and return_centers:
        # Check both methods and return best result in terms of smoothness of trajectory steps
        method_results = [frame_differencing(frames), red_channel_tracking(frames)]
        method_stds = [get_location_steps_std(robot_locations) for robot_locations in method_results]
        print(f"Frame differencing std: {method_stds[0]}, Red channel tracking std: {method_stds[1]}.",
              f"Decided to use {'Frame differencing' if np.argmin(method_stds) == 0 else 'Red channel tracking'}.")
        robot_locations = method_results[np.argmin(method_stds)]
    else:
        raise NotImplementedError

    return robot_locations


def red_channel_tracking(frames, red_threshold=100, green_threshold=100, blue_threshold=100):
    """

    :param frames:
    :param red_threshold:
    :param green_threshold:
    :param blue_threshold:
    :return:
    """
    frames_red = [(frame[:, :, 0] > red_threshold) &
                  (frame[:, :, 1] < green_threshold) &
                  (frame[:, :, 2] < blue_threshold) for frame in frames]
    robot_locations = [get_center(frame, threshold=0) for frame in frames_red]

    return robot_locations


def frame_differencing(frames, blur_sigma=1, change_threshold=0.5):
    """

    :param frames:
    :param blur_sigma:
    :param change_threshold:
    :return:
    """
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
    """
    Fills None values of location with values from the closest frame
    :param robot_locations:
    :return:
    """
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


def dist_l2(point1, point2):
    return np.linalg.norm([point1[0] - point2[0], point1[1] - point2[1]])


def get_location_steps_std(robot_locations):
    trajectory_steps = [dist_l2(robot_locations[i], robot_locations[i - 1]) for i in range(1, len(robot_locations))]
    # steps = 0 in first and last frames when robot is not moving
    steps_std = np.std([steps for steps in trajectory_steps if steps > 0])
    return steps_std
