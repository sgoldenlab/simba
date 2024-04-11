import os
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_valid_img, check_if_valid_rgb_tuple,
                                check_int, check_str)
from simba.utils.enums import Options
from simba.utils.errors import InvalidFileTypeError
from simba.utils.read_write import get_fn_ext, read_frm_of_video
from simba.utils.warnings import CropWarning


class ROISelectorCircle(object):
    """
    Class for selecting a circular region of interest (ROI) within an image or video frame.
    The selected region variables are stored in self: circle_center, circle_radius.

    :param Union[str, os.PathLike] path: Path to the image or video file. Can also be an image represented as a numpy array.
    :param int thickness: Thickness of the circle border when visualizing the ROI.
    :param Tuple[int, int, int] clr: BGR color tuple for visualizing the ROI. Default: deep pink.
    :param Optional[str] title: Title of the drawing window. If None, then `Draw ROI - Press ESC when drawn`.

    :raises InvalidFileTypeError: If the file type is not supported.
    :raises CropWarning: If the selected ROI extends beyond the image boundaries or if the ROI radius equals zero.

    .. image:: _static/img/circle_crop_2.gif
       :width: 700
       :align: center

    .. note::
       `Circle crop tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#circle-crop>`__.

    :example:
    >>> circle_selector = ROISelectorCircle(path='/Users/simon/Desktop/amber.png')
    >>> circle_selector.run()
    >>> circle_selector = ROISelectorCircle(path='/Users/simon/Desktop/Box2_IF19_7_20211109T173625_4_851_873_1_cropped.mp4')
    >>> circle_selector.run()
    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        thickness: int = 10,
        clr: Tuple[int, int, int] = (147, 20, 255),
        title: Optional[str] = None,
    ) -> None:

        check_file_exist_and_readable(file_path=path)
        check_if_valid_rgb_tuple(data=clr)
        check_int(name="Thickness", value=thickness, min_value=1, raise_error=True)
        if title is not None:
            check_str(name="ROI title window", value=title)

        if isinstance(path, np.ndarray):
            check_if_valid_img(path, source=self.__class__.__name__, raise_error=True)
            self.image = path
        else:
            check_file_exist_and_readable(file_path=path)
            _, filename, ext = get_fn_ext(filepath=path)
            if ext in Options.ALL_VIDEO_FORMAT_OPTIONS.value:
                self.image = read_frm_of_video(video_path=path)
            elif ext in Options.ALL_IMAGE_FORMAT_OPTIONS.value:
                self.image = cv2.imread(path)
            else:
                raise InvalidFileTypeError(
                    msg=f"Cannot crop a {ext} file.", source=self.__class__.__name__
                )
            if title is None:
                title = f"Draw ROI video {filename} - Press ESC when drawn"

        self.img_cpy = self.image.copy()
        self.w, self.h = self.image.shape[1], self.image.shape[0]
        if title is None:
            self.title = "Draw ROI - Press ESC when drawn"
        else:
            self.title = title

        self.drawing, self.clr, self.thickness = False, clr, thickness
        self.circle_center, self.circle_radius = (-1, -1), 1

    def draw_circle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.circle_center = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.circle_radius = max(
                    abs(x - self.circle_center[0]), abs(y - self.circle_center[1])
                )
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.circle_radius = max(
                abs(x - self.circle_center[0]), abs(y - self.circle_center[1])
            )

    def run(self):
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.title, self.draw_circle)
        while True:
            img_copy = self.image.copy()
            if self.drawing:
                cv2.circle(
                    img_copy,
                    self.circle_center,
                    self.circle_radius,
                    self.clr,
                    self.thickness,
                )
            cv2.imshow(self.title, img_copy)
            key = cv2.waitKey(1)
            if key in [27, ord("q"), ord("Q"), ord(" ")]:
                if self.run_checks():
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                    break

    def run_checks(self):
        if (
            ((self.circle_center[0] - self.circle_radius) < 0)
            or ((self.circle_center[0] + self.circle_radius) > self.w)
            or ((self.circle_center[1] - self.circle_radius) < 0)
            or ((self.circle_center[1] + self.circle_radius) > self.h)
        ):
            CropWarning(
                msg="CROP WARNING: Selected ROI radius extends beyond the image. Please try again.",
                source=self.__class__.__name__,
            )
            return False
        if self.circle_radius == 0:
            CropWarning(
                msg="CROP WARNING: Selected ROI radius equals 0. Please try again.",
                source=self.__class__.__name__,
            )
            return False
        else:
            return True
