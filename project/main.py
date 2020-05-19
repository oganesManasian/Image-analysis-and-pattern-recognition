from video import read_video
from robot_tracking import get_robot_locations
from object_detection import extract_objects
from intersections import detect_intersections
from draw import draw
from classification import CNNClassifier, FourierClasssifier
from utils import box2image

VIDEO_FILENAME = "robot_parcours_1.avi"


# ROBOT_TRACKING_METHOD = "red_channel_tracking"  # "frame_differencing"


def main():
    # Reading video
    frames = read_video(VIDEO_FILENAME)
    print(f"Read video with {len(frames)} frames")

    # Extracting trajectory
    trajectory = get_robot_locations(frames, method="auto")
    draw(frames[-1], trajectory=trajectory, title=f"Extracted robot's trajectory")

    # Extracting objects
    initial_image = frames[0]
    boxes = extract_objects(initial_image)
    draw(initial_image, boxes=boxes, title="Detected objects")

    # Find passed objects
    passed_boxes = detect_intersections(trajectory, boxes)
    draw(initial_image, trajectory=trajectory, boxes=passed_boxes, title="Passed objects")
    passed_objects = [box2image(initial_image, box) for box in passed_boxes]

    # Passed objects classification
    classifier = CNNClassifier()
    # Preprocess image (make it gray if you need)
    # TODO
    predictions = [classifier.predict(image) for image in passed_objects]
    print(predictions)

    # Postprocess predictions
    # Check that 1) digit is always followed by operation 2) last object is 'equal' sign
    # TODO

    # Calculate expression
    # TODO


main()
