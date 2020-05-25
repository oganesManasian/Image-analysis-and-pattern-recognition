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
    print(len(trajectory))
    passed_boxes = detect_intersections(trajectory, boxes)
    draw(initial_image, trajectory=trajectory, boxes=passed_boxes, title="Passed objects")
    passed_objects = [box2image(initial_image, box) for box in passed_boxes]

    # Passed objects classification
    digits = passed_objects[::2]
    classifier_digit = CNNClassifier(data_type="digits")
    predictions_digit = [classifier_digit.predict(image) for image in digits]

    operators = passed_objects[1::2]
    classifier_operator = CNNClassifier(data_type="operators")
    predictions_operator = [classifier_operator.predict(image) for image in operators]

    predicted_seq = [None] * (len(predictions_digit) + len(predictions_operator))
    predicted_seq[::2] = predictions_digit
    predicted_seq[1::2] = predictions_operator
    print("Predicted sequence:", predicted_seq)

    # Postprocess predicted_seq TODO
    # 1) digit is always followed by operation
    # 2) check that last object is 'equal' sign
    seq = predicted_seq
    # seq = "2+2="

    # Calculate expression
    expression = "".join(seq)
    expression_result = eval(expression[:-1])  # Eval needs expression without = sign
    print(f"Expression evaluation\n{expression}{expression_result}")


if __name__ == "__main__":
    main()
