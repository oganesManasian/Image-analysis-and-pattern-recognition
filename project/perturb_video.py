import os

from skimage.transform import rotate
from skimage import exposure
from numpy.random import uniform
from video import frames2video, read_video


def rotate_and_pert_brightness(frames):
    """ Randomly changes brightness and rotation of the frames.
        It outputs the adjusted frames and save the video in the video_files.
    """

    angle = uniform(-60, 60)
    gamma = uniform(0.3, 1.7)

    frames_adjusted = [(rotate(brightness(frame, gamma=gamma), angle=angle) * 255).astype('uint8') for frame in frames]

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
        gamma = uniform(1 - gamma, 1 + gamma)
    x = exposure.adjust_gamma(x, gamma, gain)
    return x


def generate_new_video():
    src_video_path = "videos/robot_parcours_1.avi"
    existing_videos = os.listdir("videos")
    max_ind = 1
    for file in existing_videos:
        last2characters = file.split(".")[0][-2:]
        if last2characters.isdigit():
            video_ind = int(last2characters)
        else:
            last_character = file.split(".")[0][-1]
            if last_character.isdigit():
                video_ind = int(last_character)
            else:
                continue

        max_ind = max(max_ind, video_ind)
    new_video_path = f"videos/robot_parcours_{max_ind + 1}.avi"

    frames = read_video(src_video_path)
    frames_adjusted = rotate_and_pert_brightness(frames)
    frames2video(frames_adjusted, new_video_path)
    return new_video_path


if __name__ == "__main__":
    generate_new_video()
