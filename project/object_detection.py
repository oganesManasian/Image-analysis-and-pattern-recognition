import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation, binary_closing, disk, square
from skimage.feature import canny
from skimage.color import rgb2gray

from region_growing import region_growing


def extract_objects(image, method='canny', fill_by='binary_closing', return_boxes=True, return_centers=True):
    """
        Get extracted objects (operators and digits) from the image. 
        You can use different methods to do that. 'canny' method which uses 
        Canny filter for edge detection. 'manual_threshold' applies manual threshold
        of 128 to grayscaled image, while 'otsu' applies Otsu threshold (by default).
        
        Also during shapes extraction from mask you can use either binary_closing 
        (by default) or binary_dilation.
        
        You can extract both object bounding boxes or object centers.
        
        Parameters
        ----------
        image : 2D array of RGB image
            The frame image
        method : str
            Could be either 'canny', 'manual_threshold' or 'otsu'
        fill_by : 'binary_closing' or 'binary_dilation'
            
        Returns
        -------
        list or (list, list)
            The list of object centers or the object boxes or both.
    """

    # Get grayscaled image
    grayscale = rgb2gray(image) * 255

    if method == 'canny':
        object_regions = canny(grayscale / 255.)  # We will use popular Canny filter for edge detection
    elif method == 'otsu':
        thresh = threshold_otsu(grayscale)
        object_regions = grayscale < thresh
    elif method == 'manual_threshold':
        thresh = 128
        object_regions = grayscale < thresh
    else:
        raise NotImplementedError

    if fill_by == 'binary_closing':
        # Fill object regions
        # square_size = [2, 5, 7, 10, 12]
        #mask_similarity_threshold = 0.999
        
        square_size = 10
        
        shapes_mask = binary_closing(object_regions, selem=disk(square_size))

        """
        shapes_mask = binary_closing(object_regions, selem=disk(square_size[0]))
        
        for size in square_size[1:]:
            new_shapes_mask = binary_closing(object_regions, selem=disk(size))
            if np.sum(new_shapes_mask == shapes_mask) / shapes_mask.size > mask_similarity_threshold:
                print(f"Chosen size {size}")
                break
            shapes_mask = new_shapes_mask
        """

    # This method produces bigger boxes
    elif fill_by == 'binary_dilation':
        # Manual threshold for dilation using square element
        square_size = 7
        shapes_mask = binary_dilation(object_regions, selem=square(square_size))
    else:
        raise NotImplementedError

    # Get object regions
    regions = region_growing(shapes_mask)
    regions = [np.array(region) for region in regions]

    # Get bounding boxes of objects
    boxes = [(min(region[:, 1]),
              min(region[:, 0]),
              max(region[:, 1]),
              max(region[:, 0]))
             for region in regions]

    # Delete false object detections
    boxes_reduced = []
    regions_reduced = []
    threshold_ratio = 5
    min_size_threshold = 5
    max_size_threshold = 40

    for region, box in zip(regions, boxes):
        width = box[2] - box[0] + 1
        height = box[3] - box[1] + 1
        if (width / height < threshold_ratio) \
                and (height / width < threshold_ratio) \
                and (max_size_threshold > height > min_size_threshold) \
                and (max_size_threshold > width > min_size_threshold) and len(region) > 70:
            boxes_reduced.append(box)
            regions_reduced.append(region)
     
    # Adding padding
    if fill_by == 'binary_closing':
        # Adding pads on the edges
        pad = 8
    elif fill_by == 'binary_dilation':
        # Dilation already adds padding
        pad = 0
    else:
        raise NotImplementedError
        
    boxes_reduced = [(max(box[0] - pad, 0),
                      max(box[1] - pad, 0),
                      min(box[2] + pad, image.shape[1]),
                      min(box[3] + pad, image.shape[0])) for box in boxes_reduced]

    region_centers = [(np.mean(region[:, 1]), np.mean(region[:, 0])) for region in regions_reduced]

    if return_boxes and return_centers:
        return region_centers, boxes_reduced

    if return_centers:
        return region_centers

    if return_boxes:
        return boxes_reduced
