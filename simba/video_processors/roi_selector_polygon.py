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


class ROISelectorPolygon(object):
    """
    Class for selecting a polygonal region of interest (ROI) within an image or video frame.
    The selected region vertices are stored in self: polygon_vertices.

    .. image:: _static/img/roi_selector_polygon.gif
       :width: 700
       :align: center

    :param Union[str, os.PathLike] path: Path to the image or video file. Can also be an image represented as a numpy array.
    :param int thickness: Thickness of the polygon border when visualizing the ROI.
    :param Tuple[int, int, int] clr: BGR color tuple for visualizing the ROI. Default: deep pink.
    :param Optional[str] title: Title of the drawing window. If None, then `Draw ROI - Press ESC when drawn`.

    :raises InvalidFileTypeError: If the file type is not supported.
    :raises CropWarning: If the selected ROI extends beyond the image boundaries or if the number of vertices is less than 3.

    :example:
    >>> polygon_selector = ROISelectorPolygon(path='/Users/simon/Desktop/amber.png')
    >>> polygon_selector.run()
    >>> polygon_selector = ROISelectorPolygon(path='/Users/simon/Desktop/Box2_IF19_7_20211109T173625_4_851_873_1_cropped.mp4')
    >>> polygon_selector.run()
    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        thickness: int = 2,
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
        self.polygon_vertices = []

    def draw_polygon(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.polygon_vertices.append((x, y))
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                temp_img = self.image.copy()
                for i in range(len(self.polygon_vertices) - 1):
                    cv2.line(
                        temp_img,
                        self.polygon_vertices[i],
                        self.polygon_vertices[i + 1],
                        self.clr,
                        self.thickness,
                    )
                cv2.imshow(self.title, temp_img)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            temp_img = self.image.copy()
            for i in range(len(self.polygon_vertices) - 1):
                cv2.line(
                    temp_img,
                    self.polygon_vertices[i],
                    self.polygon_vertices[i + 1],
                    self.clr,
                    self.thickness,
                )
            cv2.imshow(self.title, temp_img)

    def run(self):
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.title, self.draw_polygon)
        while True:
            img_copy = self.image.copy()
            for i in range(len(self.polygon_vertices) - 1):
                cv2.line(
                    img_copy,
                    self.polygon_vertices[i],
                    self.polygon_vertices[i + 1],
                    self.clr,
                    self.thickness,
                )
            if self.drawing and len(self.polygon_vertices) > 0:
                cv2.line(
                    img_copy,
                    self.polygon_vertices[-1],
                    (self.polygon_vertices[-1][0], self.polygon_vertices[-1][1]),
                    self.clr,
                    self.thickness,
                )
            cv2.imshow(self.title, img_copy)
            key = cv2.waitKey(1)
            if key in [27, ord("q"), ord("Q"), ord(" ")]:
                if self.run_checks():
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                    self.polygon_vertices = np.array(self.polygon_vertices)
                    break

    def run_checks(self):
        if len(list(set(self.polygon_vertices))) < 3:
            CropWarning(
                msg="CROP WARNING: At least 3 unique vertices are needed to form a polygon. Please try again.",
                source=self.__class__.__name__,
            )
            return False
        else:
            return True
