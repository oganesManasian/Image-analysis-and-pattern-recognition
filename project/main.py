from video import read_video
from robot_tracking import get_robot_locations
from object_detection import extract_objects
from intersections import detect_intersections
from draw import draw
from classification import CNNClassifier, FourierClasssifier
from utils import box2image, postprocess_predicted_sequence, write_on_frame, create_video

VIDEO_FILENAME = "robot_parcours_1.avi"


# ROBOT_TRACKING_METHOD = "red_channel_tracking"  # "frame_differencing"


def main():
    # Reading video
    frames = read_video(VIDEO_FILENAME)
    print(f"Read video with {len(frames)} frames")

    # Extracting trajectory
    robot_trajectory, robot_boxes = get_robot_locations(frames, method="auto")
    draw(frames[-1], trajectory=robot_trajectory, title=f"Extracted robot's trajectory")

    # Extracting objects
    initial_image = frames[0]
    object_centers, object_boxes = extract_objects(initial_image)
    draw(initial_image, boxes=object_boxes, title="Detected objects")

    # Find passed objects
    passed_boxes = detect_intersections(robot_trajectory, object_boxes)
    draw(initial_image, trajectory=robot_trajectory, boxes=[box for (box, _) in passed_boxes], title="Passed objects")
    passed_objects = [(box2image(initial_image, box), frame_ind) for (box, frame_ind) in passed_boxes]
    print(f"Passed through {len(passed_objects)} objects")

    # Passed objects classification
    digits = passed_objects[::2]
    classifier_digit = CNNClassifier(data_type="digits")
    predictions_digit = [(classifier_digit.predict(image), frame_ind) for (image, frame_ind) in digits]
    print("Digit predictions", predictions_digit)

    operators = passed_objects[1::2]
    classifier_operator = CNNClassifier(data_type="operators")
    predictions_operator = [(classifier_operator.predict(image), frame_ind) for (image, frame_ind) in operators]
    print("Operator predictions", predictions_operator)

    predicted_seq = [None] * (len(predictions_digit) + len(predictions_operator))
    predicted_seq[::2] = predictions_digit
    predicted_seq[1::2] = predictions_operator
    print("Predicted sequence:", predicted_seq)

    # Postprocess predicted_seq
    seq = postprocess_predicted_sequence(predicted_seq)  # TODO

    # Calculate expression
    expression = "".join([character for (character, _) in seq])
    expression_result = eval(expression[:-1])  # Eval needs expression without = sign
    print(f"Expression evaluation\n{expression}{expression_result}")

    # Create output video
    reversed_seq = list(reversed(seq))
    complete_expression = ""
    new_frames = []
    for i in range(len(frames)):
        # Adding char to expression if already detected
        if len(reversed_seq) > 0 and reversed_seq[-1][1] == i:
            complete_expression += reversed_seq.pop()[0]

        # Adding the final result after the final equal sign
        elif len(reversed_seq) == 0 and len(complete_expression) == len(seq):
            complete_expression += str(expression_result)

        # Writing on image
        new_frame = write_on_frame(frames[i], complete_expression)
        new_frames.append(new_frame)

    # Combining frames into a video
    create_video(new_frames, "output")


if __name__ == "__main__":
    main()
