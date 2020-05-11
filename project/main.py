from video import read_video
from robot_tracking import get_robot_locations
from object_detection import extract_objects
from intersections import detect_intersections
from draw import draw

VIDEO_FILENAME = "robot_parcours_1.avi"

ROBOT_TRACKING_METHOD = "red_channel_tracking"  # "frame_differencing"


def main():
    # Reading video
    frames = read_video(VIDEO_FILENAME)
    print(f"Read video with {len(frames)} frames")

    # Extracting trajectory
    trajectory = get_robot_locations(frames, method=ROBOT_TRACKING_METHOD)
    draw(frames[-1], trajectory=trajectory, title=f"Extracted robot's trajectory using {ROBOT_TRACKING_METHOD}")

    # Extracting objects
    image = frames[0]
    boxes = extract_objects(image)
    draw(image, boxes=boxes, title="Detected objects")

    # Find passed objects
    passed_boxes = detect_intersections(trajectory, boxes)
    draw(image, trajectory=trajectory, boxes=passed_boxes, title="Passed objects")

    # Passed objects classification
    # TODO


main()
