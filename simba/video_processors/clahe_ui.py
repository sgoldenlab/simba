import os
from typing import Tuple, Union

import cv2
import numpy as np

from simba.utils.checks import check_if_valid_img, check_instance
from simba.utils.read_write import get_video_meta_data, read_frm_of_video
from simba.utils.warnings import InValidUserInputWarning


def interactive_clahe_ui(data: Union[str, os.PathLike, np.ndarray]) -> Tuple[float, int]:
    """
    Create a user interface using OpenCV to explore and set appropriate CLAHE settings tile size and clip limit.

    .. image:: _static/img/interactive_clahe_ui.gif
       :width: 500
       :align: center

    :param Union[str, os.PathLike, np.ndarray] data: Path to a video file or a NumPy array representing an image.
    :return Tuple[float, int]: Tuple containing the chosen clip limit and tile size.

    :example:
    >>> img = cv2.imread('/Users/simon/Downloads/PXL_20240429_222923838.jpg',)
    >>> interactive_clahe_ui(data=img)
    """

    def _get_trackbar_values(v):
        clip_limit = cv2.getTrackbarPos('Clip Limit', 'Interactive CLAHE') / 10.0
        tile_size = cv2.getTrackbarPos('Tile Size', 'Interactive CLAHE')
        if tile_size % 2 == 0: tile_size += 1
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        img_clahe = clahe.apply(original_img)
        cv2.imshow('Interactive CLAHE', img_clahe)

    check_instance(source=interactive_clahe_ui.__name__, instance=data, accepted_types=(np.ndarray, str))
    if isinstance(data, str):
        _ = get_video_meta_data(video_path=data)
        original_img = read_frm_of_video(video_path=data, frame_index=0, greyscale=True)
    else:
        check_if_valid_img(data=data, source=interactive_clahe_ui.__name__)
        if len(data.shape) > 2: data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        original_img = data

    img = np.copy(original_img)
    cv2.namedWindow('Interactive CLAHE', cv2.WINDOW_NORMAL)
    cv2.imshow('Interactive CLAHE', img)
    cv2.createTrackbar('Clip Limit', 'Interactive CLAHE', 10, 300, _get_trackbar_values)
    cv2.createTrackbar('Tile Size', 'Interactive CLAHE',  8, 64,  _get_trackbar_values)

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            clip_limit = cv2.getTrackbarPos('Clip Limit', 'Interactive CLAHE') / 10.0
            tile_size = cv2.getTrackbarPos('Tile Size', 'Interactive CLAHE')
            if tile_size % 2 == 0: tile_size += 1
            cv2.destroyAllWindows()
            return clip_limit, tile_size




# # Function to update CLAHE
# def update_clahe(x):
#     global img, clahe
#     clip_limit = cv2.getTrackbarPos('Clip Limit', 'CLAHE') / 10.0  # Scale the trackbar value
#     tile_size = cv2.getTrackbarPos('Tile Size', 'CLAHE')
#     if tile_size % 2 == 0:
#         tile_size += 1  # Ensure tile size is odd
#     clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
#     img_clahe = clahe.apply(img)
#     cv2.imshow('CLAHE', img_clahe)
#
# # Load an image
# img = cv2.imread('/Users/simon/Downloads/PXL_20240429_222923838.jpg', cv2.IMREAD_GRAYSCALE)
#
# # Create a window
# cv2.namedWindow('CLAHE', cv2.WINDOW_NORMAL)
#
# # Initialize the clip limit trackbar
# cv2.createTrackbar('Clip Limit', 'CLAHE', 10, 300, update_clahe)
#
# # Initialize the tile size trackbar
# cv2.createTrackbar('Tile Size', 'CLAHE', 8, 64, update_clahe)
#
# # Apply CLAHE with initial parameters
# clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
# img_clahe = clahe.apply(img)
# cv2.imshow('Original', img)
# cv2.imshow('CLAHE', img_clahe)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
