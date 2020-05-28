import numpy as np


def detect_intersections(trajectory_raw, boxes, interpolate_trajectory=True, interpolation_param=10, tol_percent=0.1):
    """

    :param trajectory_raw:
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
            if is_point_in_box(point, box, tol_percent=tol_percent) and box_ind != last_passed_box_ind:
                frame_ind = point_ind // interpolation_param
                visited_boxes.append([box, frame_ind])
                last_passed_box_ind = box_ind

    return visited_boxes


def detect_intersections_reverse(robot_boxes, object_centers):
    """
        Detect the intersections of robot positions and symbol positions by boundaring
        object center coordinates with coordinates of robot boxes.
        
        Parameters
        ----------
        robot_boxes : list of 4 float numberts
            The robot boxes coordinates at each frame
        object_centers : list
            The coordinates of object centers
            
        Returns
        -------
        list of 2 elements
            The list containing the coordinates of object center and frame index
    """
    visited_object_centers = []
    last_passed_center_i = None

    for box_i, robot_box in enumerate(robot_boxes):
        for object_i, object_center in enumerate(object_centers):
            if is_point_in_box(object_center, robot_box) and object_i != last_passed_center_i:
                visited_object_centers.append([object_center, box_i])
                last_passed_center_i = object_i

    return visited_object_centers


def is_point_in_box(point, box, tol_percent=0):
    tol = (box[2] - box[0]) * tol_percent
    return box[0] - tol <= point[0] <= box[2] + tol \
           and box[1] - tol <= point[1] <= box[3] + tol


def get_points_on_segment(point1, point2, N=10):
    points = [((1 - lambda_) * point1[0] + lambda_ * point2[0],
               (1 - lambda_) * point1[1] + lambda_ * point2[1]) for lambda_ in np.linspace(0, 1, N)]
    return points
