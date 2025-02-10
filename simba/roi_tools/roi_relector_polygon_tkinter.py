import os
from typing import Optional, Tuple, Union
from tkinter import *
import cv2
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from PIL import Image, ImageTk

from simba.utils.checks import (check_if_valid_rgb_tuple, check_int, check_instance)
from simba.roi_tools.roi_utils import get_image_from_label
from simba.utils.warnings import ROIWarning

DRAW_FRAME_NAME = "DEFINE SHAPE"

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
                 img_window: Toplevel,
                 thickness: int = 10,
                 vertice_size: int = 2,
                 clr: Tuple[int, int, int] = (147, 20, 255)) -> None:



        check_instance(source=self.__class__.__name__, instance=img_window, accepted_types=(Toplevel,))
        check_int(name=f'{self.__class__.__name__} thickness', value=thickness, min_value=1)
        check_int(name=f'{self.__class__.__name__} vertice_size', value=vertice_size, min_value=1)
        check_if_valid_rgb_tuple(data=clr)
        self.thickness, self.clr, self.img_window = thickness, clr, img_window
        self.drawing, self.clr, self.thickness, self.vertice_size = False, clr, thickness, vertice_size
        self.polygon_vertices = []

        self.img_lbl = img_window.nametowidget("img_lbl")
        self.img = get_image_from_label(self.img_lbl)
        self.img_cpy = self.img.copy()
        self.w, self.h = self.img.shape[1], self.img.shape[0]

        self.img_window.bind("<ButtonPress-1>", self.on_left_click)
        self.img_window.bind("<B1-Motion>")

    def on_left_click(self, event):
        self.polygon_vertices.append( (event.x, event.y))
        self.img_cpy = cv2.circle(self.img_cpy, center=self.polygon_vertices[-1], radius=self.vertice_size, color=self.clr, thickness=-1, lineType=-1)
        if len(self.polygon_vertices) > 1:
            for i in range(len(self.polygon_vertices) - 1):
                self.img_cpy = cv2.line(self.img_cpy, self.polygon_vertices[i], self.polygon_vertices[i + 1], self.clr, self.thickness)
        self.update_img(self.img_cpy)

    def update_img(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        tk_image = ImageTk.PhotoImage(pil_image)
        self.img_lbl.configure(image=tk_image)
        self.img_lbl.image = tk_image

    def get_attributes(self):
        self.img_window.unbind("<ButtonPress-1>")
        self.img_window.unbind("<B1-Motion>")

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

# img = cv2.imread(r"C:\Users\sroni\OneDrive\Desktop\Screenshot 2024-11-15 123805.png")
# root = Toplevel()
# root.title(DRAW_FRAME_NAME)
# img_lbl = Label(root, name='img_lbl')
# img_lbl.pack()
#
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# pil_image = Image.fromarray(img_rgb)
# tk_image = ImageTk.PhotoImage(pil_image)
# img_lbl.configure(image=tk_image)
# img_lbl.image = tk_image
#
# _ = ROISelectorPolygon(img_window=root)
# root.mainloop()
#
#     #
    #
    # def get_attributes(self):
    #     if len(list(set(self.polygon_vertices))) < 3:
    #         ROIWarning(msg="ROI WARNING: At least 3 unique vertices are needed to form a polygon. Please try again.", source=self.__class__.__name__)
    #         return False
    #     else:
    #         self.polygon = Polygon(np.array(self.polygon_vertices)).simplify(tolerance=20, preserve_topology=True)
    #         self.polygon_vertices = np.array(self.polygon.exterior.coords)
    #         self.polygon_area = self.polygon.area
    #         self.polygon_arr = np.array(self.polygon.exterior.coords).astype(np.int32)[1:]
    #         self.max_vertice_distance = np.max(cdist(self.polygon_vertices, self.polygon_vertices).astype(np.int32))
    #         self.polygon_centroid = np.array(self.polygon.centroid).astype(int)
    #         self.tags = {f'Tag_{cnt}': tuple(y) for cnt, y in enumerate(self.polygon_arr)}
    #         self.tags['Center_tag'] = tuple(self.polygon_centroid)
    #         return True
