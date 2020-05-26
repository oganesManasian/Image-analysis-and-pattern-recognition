import numpy as np

from skimage.morphology import binary_closing, disk
from skimage.filters import gaussian

from region_growing import region_growing


def get_robot_locations(frames, method="auto", return_centers=True, return_boxes=True):
    """
        Get robot location at each frame. The first possible method is
        'frame_differencing' which subtracts adjacent frames in order to
        detect robot movements. The second possible method is 'red_channel_tracking'
        which uses the information that robot arrow is red. 'auto' method uses both
        and then chooses the smoothest one. You can return centers of
        robot locations, bounding boxes of robot location or both.

        Parameters
        ----------
        frames : list of 2D arrays
            The list containing all video frames
        method : str
            Could be either 'frame_differencing', 'red_channel_tracking' or 'auto'
        return_centers : bool
            True if return centers of robot locations, False if not
        return_boxes : bool
            True if return bounding boxes of robot locations, False if not

        Returns
        -------
        list or (list, list)
            Either list of center coords, list of boxes coords or both.
    """

    if return_centers:
        if method == "frame_differencing":
            robot_locations = frame_differencing(frames)
        elif method == "red_channel_tracking":
            robot_locations = red_channel_tracking(frames)
        elif method == "auto":
            # Check both methods and return best result in terms of smoothness of trajectory steps
            method_results = [frame_differencing(frames), red_channel_tracking(frames)]
            method_stds = [get_location_steps_std(robot_locations) for robot_locations in method_results]
            print(f"Frame differencing std: {method_stds[0]}, Red channel tracking std: {method_stds[1]}.",
                  f"Decided to use {'Frame differencing' if np.argmin(method_stds) == 0 else 'Red channel tracking'}.")
            robot_locations = method_results[np.argmin(method_stds)]
        else:
            raise NotImplementedError

        if not return_boxes:
            return robot_locations

    if return_boxes:
        robot_boxes = get_bounding_boxes(frames)

        if not return_centers:
            return robot_boxes

    return robot_locations, robot_boxes


def red_channel_tracking(frames):
    """
        Get centers of the red objects (arrow) over all frames.

        Parameters
        ----------
        frames : list of 2D arrays
            The list containing all video frames

        Returns
        -------
        robot_locations : list
            The coordinates of robot center in all frames.
    """

    frames_red = get_red_objects(frames)
    robot_locations = [get_center(frame, threshold=0) for frame in frames_red]

    return robot_locations


def frame_differencing(frames, blur_sigma=1, change_threshold=0.5):
    """
        Get robot center locations in all frames by subtracting adjacent frames
        and detecting the robot movement. To erase the possible noise we apply
        gaussian filter with 'blur_sigma' parameter.

        Parameters
        ----------
        frames : list of 2D arrays
            The list containing all video frames
        blur_sigma : float
            The sigma in the gaussian filter
        change_threshold : float
            The threshold for calculating center of robot.

        Returns
        -------
        list
            The coordinates of robot center in all frames.
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


def get_red_objects(frames, red_threshold=100, green_threshold=100, blue_threshold=100):
    """
        Get objects that are red over all frames. The red color is defined by thresholding
        RGB channels.

        Parameters
        ----------
        frames : list of 2D arrays
            The list containing all video frames
        red_threshold : float
            The threshold for red channel (by default > 128)
        green_threshold : float
            The threshold for green channel (by default < 128)
        blue_threshold : float
            The threshold for blue channel (by default < 128)

        Returns
        -------
        list
            The list of masks that identify red objects.
    """

    return [(frame[:, :, 0] > red_threshold) &
            (frame[:, :, 1] < green_threshold) &
            (frame[:, :, 2] < blue_threshold) for frame in frames]


def get_bounding_boxes(frames):
    """
        Get bounding boxes coordianates of the arrow.

        Parameters
        ----------
        frames : list of 2D arrays
            The list containing all video frames

        Returns
        -------
        arrow_boxes : tuple of 4 numbers
            [xmin, ymin, xmax, ymax] coordinates of the arrow box.
    """

    frames_red = get_red_objects(frames)
    frames_red_cleaned = [binary_closing(frame_red, selem=disk(2)) for frame_red in frames_red]
    frames_red_regions = [region_growing(frame_red) for frame_red in frames_red_cleaned]
    arrows = [np.array(max(red_regions, key=len)) for red_regions in frames_red_regions]

    image_height, image_width = frames[0].shape[1], frames[1].shape[0]  # also take into account image boundaries
    arrow_boxes = [[max(min(arrow[:, 1]), 0),
                    max(min(arrow[:, 0]), 0),
                    min(max(arrow[:, 1]), image_height),
                    min(max(arrow[:, 0]), image_width)] for arrow in arrows]

    return arrow_boxes


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
