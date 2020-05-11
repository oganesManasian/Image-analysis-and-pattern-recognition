import numpy as np


def detect_intersections(trajectory, boxes, interpolate_trajectory=True):
    if interpolate_trajectory:
        new_trajectory = []
        for i in range(1, len(trajectory)):
            new_trajectory.extend(get_points_on_segment(trajectory[i - 1], trajectory[i]))
        trajectory = new_trajectory

    last_passed_box_ind = None
    visited_boxes = []

    for point in trajectory:
        for box_ind, box in enumerate(boxes):
            if is_point_in_box(point, box) and box_ind != last_passed_box_ind:
                visited_boxes.append(box)
                last_passed_box_ind = box_ind

    return visited_boxes


def is_point_in_box(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def get_points_on_segment(point1, point2, N=10):
    points = [((1 - lambda_) * point1[0] + lambda_ * point2[0],
               (1 - lambda_) * point1[1] + lambda_ * point2[1]) for lambda_ in np.linspace(0, 1, N)]
    return points
