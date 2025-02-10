import os
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon

from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_valid_img, check_if_valid_rgb_tuple,
                                check_int, check_str)
from simba.utils.enums import Options
from simba.utils.errors import InvalidFileTypeError
from simba.utils.read_write import get_fn_ext, read_frm_of_video
from simba.utils.warnings import ROIWarning


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

    def __init__(self,
                 path: Union[str, os.PathLike, np.ndarray],
                 thickness: int = 2,
                 vertice_size: int = 3,
                 clr: Tuple[int, int, int] = (147, 20, 255),
                 title: Optional[str] = None,
                 destroy: bool = True) -> None:

        check_if_valid_rgb_tuple(data=clr)
        check_int(name="Thickness", value=thickness, min_value=1, raise_error=True)
        check_int(name="vertice_size", value=vertice_size, min_value=1, raise_error=True)
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

        self.drawing, self.clr, self.thickness, self.destroy, self.vertice_size = False, clr, thickness, destroy, vertice_size
        self.polygon_vertices, self.terminate = [], False

    def draw_polygon(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.polygon_vertices.append((x, y))
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                temp_img = self.image.copy()
                for i in range(len(self.polygon_vertices) - 1):
                    cv2.line(temp_img, self.polygon_vertices[i], self.polygon_vertices[i + 1], self.clr, self.thickness)
                cv2.imshow(self.title, temp_img)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            temp_img = self.image.copy()
            for i in range(len(self.polygon_vertices) - 1):
                cv2.line(temp_img, self.polygon_vertices[i], self.polygon_vertices[i + 1], self.clr, self.thickness)
            cv2.imshow(self.title, temp_img)

    def run(self):
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.title, self.draw_polygon)
        while not self.terminate:
            img_copy = self.image.copy()
            for i in range(len(self.polygon_vertices)):
                cv2.circle(img_copy, center=self.polygon_vertices[i], radius=self.vertice_size, color=self.clr, thickness=-1)
            for i in range(len(self.polygon_vertices) - 1):
                cv2.line(img_copy, self.polygon_vertices[i], self.polygon_vertices[i + 1], self.clr, self.thickness)
            if self.drawing and len(self.polygon_vertices) > 0:
                cv2.line(img_copy, self.polygon_vertices[-1], (self.polygon_vertices[-1][0], self.polygon_vertices[-1][1]), self.clr, self.thickness,)
            cv2.imshow(self.title, img_copy)
            key = cv2.waitKey(1)
            if key in [27, ord("q"), ord("Q"), ord(" ")]:
                if self.run_checks():
                    if self.destroy:
                        cv2.destroyAllWindows()
                    cv2.waitKey(1)
                    self.terminate = True
                    break

    def run_checks(self):
        if len(list(set(self.polygon_vertices))) < 3:
            ROIWarning(msg="ROI WARNING: At least 3 unique vertices are needed to form a polygon. Please try again.", source=self.__class__.__name__)
            return False
        else:
            self.polygon = Polygon(np.array(self.polygon_vertices)).simplify(tolerance=20, preserve_topology=True)
            self.polygon_vertices = np.array(self.polygon.exterior.coords)
            self.polygon_area = self.polygon.area
            self.polygon_arr = np.array(self.polygon.exterior.coords).astype(np.int32)[1:]
            self.max_vertice_distance = np.max(cdist(self.polygon_vertices, self.polygon_vertices).astype(np.int32))
            self.polygon_centroid = np.array(self.polygon.centroid).astype(int)
            self.tags = {f'Tag_{cnt}': tuple(y) for cnt, y in enumerate(self.polygon_arr)}
            self.tags['Center_tag'] = tuple(self.polygon_centroid)
            return True
