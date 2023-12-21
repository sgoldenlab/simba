import os
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_valid_rgb_tuple, check_int, check_str)
from simba.utils.enums import Options
from simba.utils.errors import InvalidFileTypeError
from simba.utils.read_write import get_fn_ext, get_video_meta_data
from simba.utils.warnings import CropWarning


class ROISelector:
    """
    A class for selecting and reflecting Regions of Interest (ROI) in an image.
    The selected region variables:  top_left, bottom_right, width, height are stored in self.

    :param Union[str, os.PathLike] path: Path to the image or video file. Can also be an image represented as a numpy array.
    :param int thickness: Thickness of the rectangle border for visualizing the ROI.
    :param Tuple[int, int, int] clr: BGR color tuple for visualizing the ROI. Default: deep pink.
    :param Optional[str] title: Title of the drawing window. If None, then `Draw ROI - Press ESC when drawn`.

    :example:
    >>> img_selector = ROISelector(path='/Users/simon/Desktop/compute_overlap.png', clr=(0, 255, 0), thickness=2)
    >>> img_selector.run()
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
        self.roi_start = None
        self.roi_end = None
        self.selecting_roi = False
        self.clr = clr
        self.thickness = int(thickness)

        if isinstance(path, np.ndarray):
            self.image = path
        else:
            _, filename, ext = get_fn_ext(filepath=path)
            if ext in Options.ALL_VIDEO_FORMAT_OPTIONS.value:
                _ = get_video_meta_data(video_path=path)
                cap = cv2.VideoCapture(path)
                cap.set(1, 0)
                _, self.image = cap.read()

            elif ext in Options.ALL_IMAGE_FORMAT_OPTIONS.value:
                self.image = cv2.imread(path).astype(np.uint8)

            else:
                raise InvalidFileTypeError(
                    msg=f"Cannot crop a {ext} file.", source=self.__class__.__name__
                )

            title = f"Draw ROI video {filename} - Press ESC when drawn"

        self.img_cpy = self.image.copy()

        if title is None:
            self.title = "Draw ROI - Press ESC when drawn"
        else:
            self.title = title

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_start = (x, y)
            self.roi_end = (x, y)
            self.selecting_roi = True

        elif event == cv2.EVENT_LBUTTONUP:
            self.roi_end = (x, y)
            self.selecting_roi = False

        elif event == cv2.EVENT_MOUSEMOVE and self.selecting_roi:
            self.roi_end = (x, y)
            self.img_cpy = self.image.copy()
            cv2.rectangle(
                self.img_cpy, self.roi_start, self.roi_end, self.clr, self.thickness
            )

    def run(self):
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.title, self.mouse_callback)

        while True:
            if self.selecting_roi or self.img_cpy is None:
                self.img_cpy = self.image.copy()
                if self.roi_start is not None and self.roi_end is not None:
                    cv2.rectangle(
                        self.img_cpy,
                        self.roi_start,
                        self.roi_end,
                        self.clr,
                        self.thickness,
                    )

            cv2.imshow(self.title, self.img_cpy)
            key = cv2.waitKey(1) & 0xFF

            if (
                key == ord("c")
                and self.roi_start is not None
                and self.roi_end is not None
            ):
                selected_roi = self.image[
                    self.roi_start[1] : self.roi_end[1],
                    self.roi_start[0] : self.roi_end[0],
                ].copy()
                reflected_roi = cv2.flip(selected_roi, 1)
                self.image[
                    self.roi_start[1] : self.roi_end[1],
                    self.roi_start[0] : self.roi_end[0],
                ] = reflected_roi
                self.roi_start = None
                self.roi_end = None

            elif key == 27:
                if self.run_checks():
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                    break

    def run_checks(self):
        self.top_left = min(self.roi_start[0], self.roi_end[0]), min(
            self.roi_start[1], self.roi_end[1]
        )
        self.bottom_right = max(self.roi_start[0], self.roi_end[0]), max(
            self.roi_start[1], self.roi_end[1]
        )
        self.width = self.bottom_right[0] - self.top_left[0]
        self.height = self.bottom_right[1] - self.top_left[1]

        if (self.width == 0 and self.height == 0) or (
            self.width + self.height + self.top_left[0] + self.top_left[1] == 0
        ):
            CropWarning(
                msg="CROP WARNING: Cropping height and width are both 0. Please try again.",
                source=self.__class__.__name__,
            )
            return False

        else:
            return True


# img_selector = ROISelector(path='/Users/simon/Desktop/envs/troubleshooting/ARES_data/Termite Test 1/Termite Test 1.mp4')
# img_selector.run()

# img_selector = ROISelector(path='/Users/simon/Desktop/compute_overlap.png', clr=(0, 255, 0), thickness=2)
# img_selector.run()

#
# # Global variables to store the ROI coordinates
# roi_start = None
# roi_end = None
# selecting_roi = False
# image_copy = None
#
# def mouse_callback(event, x, y, flags, param):
#     global roi_start, roi_end, selecting_roi, image_copy
#
#     if event == cv2.EVENT_LBUTTONDOWN:
#         roi_start = (x, y)
#         roi_end = (x, y)
#         selecting_roi = True
#
#     elif event == cv2.EVENT_LBUTTONUP:
#         roi_end = (x, y)
#         selecting_roi = False
#
#     elif event == cv2.EVENT_MOUSEMOVE and selecting_roi:
#         roi_end = (x, y)
#         image_copy = image.copy()
#         cv2.rectangle(image_copy, roi_start, roi_end, (0, 255, 0), 2)
#
# # Load your image
# image_path = '/Users/simon/Desktop/compute_overlap.png'
# image = cv2.imread(image_path)
# image_copy = image.copy()
#
# # Create a window and set the mouse callback
# cv2.namedWindow('Select ROI')
# cv2.setMouseCallback('Select ROI', mouse_callback)
#
# while True:
#     if roi_start is not None and roi_end is not None:
#         # Draw the selected ROI on the image copy
#         image_copy = image.copy()
#         cv2.rectangle(image_copy, roi_start, roi_end, (0, 255, 0), 2)
#
#     cv2.imshow('Select ROI', image_copy)
#
#     key = cv2.waitKey(1) & 0xFF
#
#     if key == ord('c') and roi_start is not None and roi_end is not None:
#         # Copy the selected ROI
#         selected_roi = image[roi_start[1]:roi_end[1], roi_start[0]:roi_end[0]].copy()
#
#         # Reflect the ROI horizontally
#         reflected_roi = cv2.flip(selected_roi, 1)
#
#         # Replace the original ROI with the reflected one
#         image[roi_start[1]:roi_end[1], roi_start[0]:roi_end[0]] = reflected_roi
#
#         # Reset ROI coordinates
#         roi_start = None
#         roi_end = None
#
#     elif key == 27:  # Press Esc to exit
#         break
#
# cv2.destroyAllWindows()
