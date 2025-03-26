import os
from typing import Tuple, Union

import cv2
import numpy as np

from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import check_instance
from simba.utils.enums import TextOptions
from simba.utils.errors import InvalidInputError
from simba.utils.read_write import get_video_meta_data, read_frm_of_video

WIN_NAME = 'INTERACTIVE CLAHE - HIT ESC TO RUN'
CLIP_LIMIT = 'CLIP LIMIT'
TILE_SIZE = 'TILE SIZE'
SELECT_VIDEO_FRAME = 'SHOW FRAME'

def interactive_clahe_ui(data: Union[str, os.PathLike]) -> Tuple[float, int]:
    """
    Create a user interface using OpenCV to explore and set appropriate CLAHE settings tile size and clip limit.

    .. image:: _static/img/interactive_clahe_ui.gif
       :width: 500
       :align: center

    :param Union[str, os.PathLike, np.ndarray] data: Path to a video file.
    :return Tuple[float, int]: Tuple containing the chosen clip limit and tile size.

    :example:
    >>> video = cv2.imread(r"D:\EPM\sample_2\video_1.mp4")
    >>> interactive_clahe_ui(data=video)
    """
    global original_img, font_size, x_spacer, y_spacer, txt

    def _get_trackbar_values(v):
        global original_img, font_size, x_spacer, y_spacer, txt
        clip_limit = cv2.getTrackbarPos(CLIP_LIMIT, WIN_NAME) / 10.0
        tile_size = cv2.getTrackbarPos(TILE_SIZE, WIN_NAME)
        if tile_size % 2 == 0: tile_size += 1
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        img_clahe = clahe.apply(original_img)
        cv2.putText(img_clahe, txt,(TextOptions.BORDER_BUFFER_X.value, TextOptions.BORDER_BUFFER_Y.value + y_spacer), TextOptions.FONT.value, font_size, (255, 255, 255), 3)
        cv2.imshow(WIN_NAME, img_clahe)
        cv2.waitKey(100)

    def _change_img(v):
        global original_img, font_size, x_spacer, y_spacer, txt
        new_frm_id = cv2.getTrackbarPos(SELECT_VIDEO_FRAME, WIN_NAME)
        original_img = read_frm_of_video(video_path=data, frame_index=new_frm_id, greyscale=True)
        clip_limit = cv2.getTrackbarPos(CLIP_LIMIT, WIN_NAME) / 10.0
        tile_size = cv2.getTrackbarPos(TILE_SIZE, WIN_NAME)
        if tile_size % 2 == 0: tile_size += 1
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        img_clahe = clahe.apply(original_img)
        cv2.putText(img_clahe, txt,(TextOptions.BORDER_BUFFER_X.value, TextOptions.BORDER_BUFFER_Y.value + y_spacer), TextOptions.FONT.value, font_size, (255, 255, 255), 3)
        cv2.imshow(WIN_NAME, img_clahe)
        cv2.waitKey(100)

    check_instance(source=interactive_clahe_ui.__name__, instance=data, accepted_types=(np.ndarray, str))
    if isinstance(data, str):
        video_meta_data = get_video_meta_data(video_path=data)
        original_img = read_frm_of_video(video_path=data, frame_index=0, greyscale=True)
    else:
        raise InvalidInputError(msg=f'data has to be a path to a video file, but got {type(data)}', source=interactive_clahe_ui.__name__)
    txt = 'Hit ESC to run with chosen settings'
    font_size, x_spacer, y_spacer = PlottingMixin().get_optimal_font_scales(text=txt, accepted_px_width=int(video_meta_data["width"] / 2), accepted_px_height=int(video_meta_data["height"] / 15), text_thickness=3)

    img = np.copy(original_img)
    cv2.putText(img, txt, (TextOptions.BORDER_BUFFER_X.value, TextOptions.BORDER_BUFFER_Y.value + y_spacer), TextOptions.FONT.value, font_size, (255, 255, 255), 3)
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, video_meta_data['width'], video_meta_data['height'])
    cv2.imshow(WIN_NAME, img)
    cv2.createTrackbar(CLIP_LIMIT, WIN_NAME, 10, 300, _get_trackbar_values)
    cv2.createTrackbar(TILE_SIZE, WIN_NAME,  8, 64,  _get_trackbar_values)
    cv2.createTrackbar(SELECT_VIDEO_FRAME, WIN_NAME, 0, video_meta_data['frame_count'], _change_img)

    while True:
        if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            break
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            clip_limit = cv2.getTrackbarPos(CLIP_LIMIT, WIN_NAME) / 10.0
            tile_size = cv2.getTrackbarPos(TILE_SIZE, WIN_NAME)
            if tile_size % 2 == 0: tile_size += 1
            cv2.destroyAllWindows()
            return clip_limit, tile_size


#interactive_clahe_ui(data=r"D:\EPM\sample_2\video_1.mp4")

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
