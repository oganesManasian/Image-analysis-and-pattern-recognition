from skimage.transform import rotate
from skimage import exposure
from numpy.random import uniform
from video import frames2video

def rotate_and_pert_brightness(frames, save_path):
    """ Randomly changes brightness and rotation of the frames. 
        It outputs the adjusted frames and save the video in the video_files.
    """
    
    angle = uniform(-180, 180)
    gamma = uniform(0.5, 1.5)
    
    frames_adjusted = [(rotate(brightness(frame, gamma=gamma), angle=angle) * 255).astype('uint8') for frame in frames]
    
    frames2video(frames_adjusted, save_path)
    
    return frames_adjusted
    
def brightness(x, gamma=1, gain=1, is_random=False):
    """Change the brightness of a single image, randomly or non-randomly.

    Parameters
    -----------
    x : numpy array
        An image with dimension of [row, col, channel] (default).
    gamma : float, small than 1 means brighter.
        Non negative real number. Default value is 1.

        - If is_random is True, gamma in a range of (1-gamma, 1+gamma).
    gain : float
        The constant multiplier. Default value is 1.
    is_random : boolean, default False
        - If True, randomly change brightness.

    References
    -----------
    - `skimage.exposure.adjust_gamma <http://scikit-image.org/docs/dev/api/skimage.exposure.html>`_
    - `chinese blog <http://www.cnblogs.com/denny402/p/5124402.html>`_
    """
    if is_random:
        gamma = uniform(1-gamma, 1+gamma)
    x = exposure.adjust_gamma(x, gamma, gain)
    return x 