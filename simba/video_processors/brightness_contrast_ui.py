import os
from typing import Tuple, Union

import cv2
import numpy as np

from simba.utils.checks import check_if_valid_img, check_instance
from simba.utils.read_write import get_video_meta_data, read_frm_of_video
from simba.utils.warnings import InValidUserInputWarning


class BrightnessContrastUI:
    """
     Create a user interface using OpenCV to explore and change the brightness and contrast of a video.

     .. note::
        Adapted from `geeksforgeeks <https://www.geeksforgeeks.org/changing-the-contrast-and-brightness-of-an-image-using-python-opencv/>`_.

     .. image:: _static/img/brightness_contrast_ui.gif
        :width: 700
        :align: center

     :param Union[str, os.PathLike] video_path: Path to the video file or an image in numpy array format.
     :return Tuple: The scaled brightness and scaled contrast values on scale -1 to +1 and 0-2 respectively, suitable for FFmpeg conversion

     :example I:
     >>> brightness_contrast_ui(video_path='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/frames/output/ROI_features/2022-06-20_NOB_DOT_4.mp4')

     :example II:
     >>> img = cv2.imread('/Users/simon/Downloads/PXL_20240429_222923838.jpg')
     >>> brightness_contrast_ui(data=img)

     """

    def __init__(self, data: Union[str, os.PathLike, np.ndarray]):
        self.WINDOW_NAME = "CONTRAST / BRIGHTNESS: HIT ESC TO CONTINUE"
        self.BRIGHTNESS = 'BRIGHTNESS'
        self.CONTRAST = 'CONTRAST'
        self.last_values = {self.BRIGHTNESS: 255, self.CONTRAST: 127}
        self._trackbars_created = False

        check_instance(source=self.__class__.__name__, instance=data, accepted_types=(np.ndarray, str))
        if isinstance(data, str):
            _ = get_video_meta_data(video_path=data)
            self.original_img = read_frm_of_video(video_path=data, frame_index=0)
        else:
            check_if_valid_img(data=data, source=self.__class__.__name__)
            self.original_img = data

    def _on_trackbar_change(self, _):
        if not self._trackbars_created:
            return

        if cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return

        try:
            brightness_raw = cv2.getTrackbarPos(self.BRIGHTNESS, self.WINDOW_NAME)
            contrast_raw = cv2.getTrackbarPos(self.CONTRAST, self.WINDOW_NAME)
        except cv2.error:
            return

        if (brightness_raw == self.last_values[self.BRIGHTNESS] and
                contrast_raw == self.last_values[self.CONTRAST]):
            return

        self.last_values[self.BRIGHTNESS] = brightness_raw
        self.last_values[self.CONTRAST] = contrast_raw

        # Convert raw trackbar positions to ffmpeg-like brightness and contrast
        brightness = (brightness_raw - 255) / 255.0  # Scale to [-1, 1]
        contrast = contrast_raw / 127.0  # Scale to [0, 2]

        # Apply eq filter-like brightness/contrast: new_img = (img - 128)*contrast + 128 + brightness*255
        img_float = self.original_img.astype(np.float32)
        buf = (img_float - 128) * contrast + 128 + brightness * 255
        buf = np.clip(buf, 0, 255).astype(np.uint8)

        cv2.imshow(self.WINDOW_NAME, buf)

    def _create_window_and_trackbars(self):
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(self.WINDOW_NAME, self.original_img)
        cv2.createTrackbar(self.BRIGHTNESS, self.WINDOW_NAME, self.last_values[self.BRIGHTNESS], 510, self._on_trackbar_change)
        cv2.createTrackbar(self.CONTRAST, self.WINDOW_NAME, self.last_values[self.CONTRAST], 254, self._on_trackbar_change)
        self._trackbars_created = True

    def run(self) -> Tuple[float, float]:
        self._trackbars_created = False
        self._create_window_and_trackbars()

        while True:
            if cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                raise InValidUserInputWarning("Window was closed before selecting values.", source=self.__class__.__name__)

            k = cv2.waitKey(10) & 0xFF
            if k == 27:
                try:
                    brightness_raw = cv2.getTrackbarPos(self.BRIGHTNESS, self.WINDOW_NAME)
                    contrast_raw = cv2.getTrackbarPos(self.CONTRAST, self.WINDOW_NAME)
                except cv2.error:
                    continue

                cv2.destroyAllWindows()

                scaled_brightness = (brightness_raw - 255) / 255.0 # [-1, 1]
                scaled_contrast = contrast_raw / 127.0              # [0, 2]

                if scaled_brightness == 0.0 and scaled_contrast == 1.0:
                    InValidUserInputWarning("Both the selected brightness and contrast are default. Select different values.", source=self.__class__.__name__)
                else:
                    return scaled_brightness, scaled_contrast

#ui = BrightnessContrastUI(data=r"/mnt/c/Users/sroni/OneDrive/Desktop/light_dark_box_eq_20250603150339.mp4")
#ui.run()
# ui = BrightnessContrastUI(data=r"D:\brightness_contrast\F1_2.mp4")
# ui.run()