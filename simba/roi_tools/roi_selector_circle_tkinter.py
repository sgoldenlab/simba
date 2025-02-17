from tkinter import *
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from simba.roi_tools.roi_utils import get_image_from_label
from simba.utils.checks import (check_if_valid_rgb_tuple, check_instance,
                                check_int)
from simba.utils.warnings import ROIWarning

DRAW_FRAME_NAME = "DEFINE SHAPE"


class ROISelectorCircle(object):

    """
    A class to allow users to interactively select a Region of Interest (ROI) circle from an image displayed in a Tkinter window.

    .. video:: tutorials_rst/img/roi/draw_circle.webm
       :width: 900
       :loop:

    .. seealso::
       For OpenCV based method, see :func:`simba.video_processors.roi_selector_circle.ROISelectorCircle`

    :param Toplevel img_window: The Tkinter Toplevel window containing the image.
    :param int thickness: The thickness of the ROI selection circle.
    :param Tuple[int, int, int] clr: The color of the ROI circle rectangle in RGB format.

    :example:
    """
    def __init__(self,
                 img_window: Toplevel,
                 thickness: int = 10,
                 clr: Tuple[int, int, int] = (147, 20, 255)):

        check_instance(source=self.__class__.__name__, instance=img_window, accepted_types=(Toplevel,))
        check_int(name=f'{self.__class__.__name__} thickness', value=thickness, min_value=1)
        check_if_valid_rgb_tuple(data=clr)
        self.thickness, self.clr, self.img_window = thickness, clr, img_window

        self.img_lbl = img_window.nametowidget("img_lbl")
        self.img = get_image_from_label(self.img_lbl)
        self.img_cpy = self.img.copy()
        self.w, self.h = self.img.shape[1], self.img.shape[0]
        self.drawing, self.clr, self.thickness = False, clr, thickness
        self.circle_center, self.circle_radius = (-1, -1), 1

        self.img_window.bind("<Button-1>", self.mouse_start)
        self.img_window.bind("<B1-Motion>", self.mouse_move)
        self.img_window.bind("<ButtonRelease-1>", self.mouse_release)

    def mouse_start(self, event):
        self.drawing = True
        self.circle_center = (event.x, event.y)
        self.circle_radius = 0

    def mouse_move(self, event):
        if self.drawing:
            self.circle_radius = int(np.sqrt((event.x - self.circle_center[0]) ** 2 + (event.y - self.circle_center[1]) ** 2))
            self.update_image()

    def mouse_release(self, event):
        self.drawing = False
        self.update_image()

    def update_image(self):
        self.img_cpy = self.img.copy()
        self.img_cpy = cv2.circle(self.img_cpy, self.circle_center, self.circle_radius, self.clr, self.thickness)
        self.img_cpy = cv2.cvtColor(self.img_cpy, cv2.COLOR_RGB2BGR)
        self.tk_img = ImageTk.PhotoImage(image=Image.fromarray(self.img_cpy))
        self.img_lbl.config(image=self.tk_img)

    def get_attributes(self):
        self.left_border_tag = (int(self.circle_center[0] - self.circle_radius), self.circle_center[1])
        self.right_border_tag = (int(self.circle_center[0] + self.circle_radius), self.circle_center[1])
        self.top_border_tag = (self.circle_center[0], int(self.circle_center[1] - self.circle_radius))
        self.bottom_border_tag = (self.circle_center[0], int(self.circle_center[1] + self.circle_radius))

        self.img_window.unbind("<ButtonPress-1>")
        self.img_window.unbind("<B1-Motion>")
        self.img_window.unbind("<ButtonRelease-1>")

        if (((self.circle_center[0] - self.circle_radius) < 0) or ((self.circle_center[0] + self.circle_radius) > self.w) or ((self.circle_center[1] - self.circle_radius) < 0) or ((self.circle_center[1] + self.circle_radius) > self.h)):
            ROIWarning(msg="ROI WARNING: The drawn circular ROI radius extends beyond the image. Please try again.", source=self.__class__.__name__)
            return False
        elif self.circle_radius == 0:
            ROIWarning(msg="ROI WARNING: The drawn circular ROI radius equals 0. Please try again.", source=self.__class__.__name__)
            return False
        else:
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
# _ = ROISelectorCircle(img_window=root)
# root.mainloop()




# circle_selector = ROISelectorCircle(path=r"C:\troubleshooting\mitra\test\503_MA109_Gi_CNO_0521.mp4")
# circle_selector.run()