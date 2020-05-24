import os
import shutil
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.io import imsave

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

    create_video_dataset = True
    if create_video_dataset:
        video_dataset_path = "video_dataset"
        if os.path.isdir(video_dataset_path):
            shutil.rmtree(video_dataset_path)
        os.mkdir(video_dataset_path)

        # Save digits
        digits_path = "digits"
        os.mkdir(os.path.join(video_dataset_path, digits_path))
        for i in range(10):
            os.mkdir(os.path.join(video_dataset_path, digits_path, str(i)))

        digits_ind = [29, 28, 20, 4, 6, 17]
        labels = [3, 2, 7, 2, 3, 7]
        boxes_to_save = [boxes[i] for i in digits_ind]
        objects_to_save = [box2image(initial_image, box) for box in boxes_to_save]

        for i, (image, label) in enumerate(zip(objects_to_save, labels)):
            image = rgb2gray(image)
            thresh = threshold_otsu(image)
            binary = (image > thresh).astype(float)
            imsave(os.path.join(video_dataset_path, digits_path, str(label), f"objects{i}.png"), binary)
        print("Video dataset created.")

        # Save operators
        # TODO

    # Find passed objects
    passed_boxes = detect_intersections(trajectory, boxes)
    draw(initial_image, trajectory=trajectory, boxes=passed_boxes, title="Passed objects")
    passed_objects = [box2image(initial_image, box) for box in passed_boxes]

    # Passed objects classification
    classifier = CNNClassifier()
    digits_to_classify = passed_objects[::2]
    digit_predictions = [classifier.predict(image) for image in digits_to_classify]
    print(digit_predictions)

    # Postprocess predictions
    # Check that 1) digit is always followed by operation 2) last object is 'equal' sign
    # TODO

    # Calculate expression
    # TODO


main()
