import numpy as np


def get_neighbours(pixel):
    """
        Return 4 neighbours of the input pixel.

        Parameters
        ----------
        pixel : list of length 2
            The coordinates of the pixel

        Returns
        -------
        list of 4 tuples of length 2
            The 4 pixel coordinates of the neighbours.
    """

    return [(pixel[0] + 1, pixel[1]),
            (pixel[0] - 1, pixel[1]),
            (pixel[0], pixel[1] + 1),
            (pixel[0], pixel[1] - 1)]


def grow(pixel_to_start, mask, visited):
    """
        Return region starting from pixel_to_start.
        The region should correspond to mask that contain isolated regions of interest.

        Parameters
        ----------
        pixel_to_start : list of length 2
            The coordinates of the starting pixel.
        mask : array of shape (image_height, image_width)
            The logical matrix which mark the regions of interest.
        visited : list of lists of length 2
            The list that contains all visited pixels.

        Returns
        -------
        list of lists of length 2
            The list of pixels of the grown region.
    """

    stack = [pixel_to_start]
    region = [pixel_to_start]
    visited[pixel_to_start] = 1

    while len(stack):

        # Pick a pixel
        cur_pixel = stack.pop()

        pixels_to_check = get_neighbours(cur_pixel)

        # Delete pixels which are out of image
        pixels_to_check = [pixel for pixel in pixels_to_check
                           if 0 <= pixel[0] < mask.shape[0] and 0 <= pixel[1] < mask.shape[1]]

        # Check if neighbours are not visited and correspond to mask
        pixels_from_region = [pixel for pixel in pixels_to_check if mask[pixel] and not visited[pixel]]

        # Update containers
        stack.extend(pixels_from_region)
        for pixel in pixels_from_region:
            visited[pixel] = 1
        region.extend(pixels_from_region)

    return region


def region_growing(mask):
    """
        Region growing method that finds the connected objects
        in the image indicated by mask and labels them.

        Parameters
        ----------
        mask : array of shape (image_height, image_width)
            The logical matrix which mark the regions of interest.

        Returns
        -------
        list of lists of lists of length 2
            The list that contains all grown regions.
    """

    regions = []
    visited = np.zeros(mask.shape)

    while True:  # Each iteration is aimed to process one region
        # Search pixels not attached for any processed region
        pixels_to_grow = mask * (1 - visited)

        # If all pixels from the mask are labeled, algorithm identified all objects
        if len(pixels_to_grow[pixels_to_grow > 0]) == 0:
            break

        X, Y = np.where(pixels_to_grow > 0)
        pixel_to_start = (X[0], Y[0])  # Pixel from unknown region

        region = grow(pixel_to_start, mask, visited)
        regions.append(region)

    return regions
