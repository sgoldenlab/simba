from typing import Union, Tuple
import os
import numpy as np
import cv2

from simba.utils.read_write import get_video_meta_data, read_frm_of_video
from simba.utils.warnings import InValidUserInputWarning

def brightness_contrast_ui(video_path: Union[str, os.PathLike]) -> Tuple[float, float]:
    """
    Create a user interface using OpenCV to explore and change the brightness and contrast of a video.

    .. note::
       Adapted from `geeksforgeeks <https://www.geeksforgeeks.org/changing-the-contrast-and-brightness-of-an-image-using-python-opencv/>`_.

    .. image:: _static/img/brightness_contrast_ui.png
       :width: 700
       :align: center

    :param Union[str, os.PathLike] video_path: Path to the video file.
    :return Tuple: The scaled brightness and scaled contrast values on scale -1 to +1 suitable for FFmpeg conversion

    :example:
    >>> brightness_contrast_ui(video_path='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/frames/output/ROI_features/2022-06-20_NOB_DOT_4.mp4')
    """
    def _get_trackbar_values(v):
        brightness = cv2.getTrackbarPos('Brightness', 'Contrast / Brightness')
        contrast = cv2.getTrackbarPos('Contrast', 'Contrast / Brightness')
        brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
        contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
        if brightness != 0:
            if brightness > 0:
                shadow, max = brightness, 255
            else:
                shadow, max = 0, 255 + brightness
            cal = cv2.addWeighted(original_img, (max - shadow) / 255, original_img, 0, shadow)
        else:
            cal = original_img
        if contrast != 0:
            Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
            Gamma = 127 * (1 - Alpha)
            cal = cv2.addWeighted(cal, Alpha, cal, 0, Gamma)
        img = np.copy(cal)
        cv2.imshow('Contrast / Brightness', img)

    _ = get_video_meta_data(video_path=video_path)
    original_img = read_frm_of_video(video_path=video_path, frame_index=0)
    img = np.copy(original_img)
    cv2.namedWindow('Contrast / Brightness', cv2.WINDOW_NORMAL)
    cv2.imshow('Contrast / Brightness', img)
    cv2.createTrackbar('Brightness', 'Contrast / Brightness', 255, 2 * 255, _get_trackbar_values)
    cv2.createTrackbar('Contrast', 'Contrast / Brightness',  127, 2 * 127,  _get_trackbar_values)
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            brightness = cv2.getTrackbarPos('Brightness', 'Contrast / Brightness')
            contrast = cv2.getTrackbarPos('Contrast', 'Contrast / Brightness')
            scaled_brightness = ((brightness - 0) / (510 - 0)) * (1 - -1) + -1
            scaled_contrast= ((contrast - 0) / (254 - 0)) * (2 - 0) + 0
            if scaled_contrast == 0.0 and scaled_brightness == 0.0:
                InValidUserInputWarning(msg=f'Both the selected brightness and contrast are the same as in the input video. Select different values.')
            else:
                cv2.destroyAllWindows()
                return scaled_brightness, scaled_contrast