import numpy as np


def detect_intersections(trajectory_raw, boxes, interpolate_trajectory=True, interpolation_param=10):
    """

    :param trajectory:
    :param boxes:
    :param interpolate_trajectory: Augment trajectory by points on line between trajectory points
    :return: list of (visited box, index of frame at which box was visited) tuples
    """

    # Augment trajectory by points on line between trajectory points
    if interpolate_trajectory:
        trajectory = []
        for i in range(1, len(trajectory_raw)):
            trajectory.extend(get_points_on_segment(trajectory_raw[i - 1], trajectory_raw[i], N=interpolation_param))
    else:
        trajectory = trajectory_raw

    last_passed_box_ind = None
    visited_boxes = []

    # Make algorithm to not count robot's box as passed one
    for box_ind, box in enumerate(boxes):
        if is_point_in_box(trajectory[0], box):
            last_passed_box_ind = box_ind

    # Find intersections
    for point_ind, point in enumerate(trajectory):
        for box_ind, box in enumerate(boxes):
            if is_point_in_box(point, box) and box_ind != last_passed_box_ind:
                frame_ind = point_ind // interpolation_param
                visited_boxes.append([box, frame_ind])
                last_passed_box_ind = box_ind

    return visited_boxes


def is_point_in_box(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def get_points_on_segment(point1, point2, N=10):
    points = [((1 - lambda_) * point1[0] + lambda_ * point2[0],
               (1 - lambda_) * point1[1] + lambda_ * point2[1]) for lambda_ in np.linspace(0, 1, N)]
    return points
