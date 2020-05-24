import numpy as np
from skimage.morphology import binary_closing, disk
from skimage.feature import canny
from skimage.color import rgb2gray

from region_growing import region_growing


def extract_objects(image, objects_shape=(28, 28)):
    # Extract edges
    grayscale = rgb2gray(image) * 255
    edges = canny(grayscale / 255.)  # We will use popular Canny filter for edge detection

    # Fill object regions
    square_size = [2, 5, 10]
    mask_similarity_threshold = 0.999

    shapes_mask = binary_closing(edges, selem=disk(square_size[0]))
    for size in square_size[1:]:
        new_shapes_mask = binary_closing(edges, selem=disk(size))
        if np.sum(new_shapes_mask == shapes_mask) / shapes_mask.size > mask_similarity_threshold:
            print(f"Chose size {size}")
            break
        shapes_mask = new_shapes_mask

    # Get object regions
    regions = region_growing(shapes_mask)
    regions = [np.array(region) for region in regions]

    # Get bounding boxes of objects
    pad = 7
    boxes = [(max(min(region[:, 1]) - pad, 0),
              max(min(region[:, 0]) - pad, 0),
              min(max(region[:, 1]) + pad, image.shape[1]),
              min(max(region[:, 0]) + pad, image.shape[0]))
             for region in regions]

    # Delete false object detections
    boxes_reduced = []
    threshold_ratio = 5

    for box in boxes:
        width = box[2] - box[0]
        height = box[3] - box[1]
        if width / height < threshold_ratio and height / width < threshold_ratio:
            boxes_reduced.append(box)

    return boxes_reduced
