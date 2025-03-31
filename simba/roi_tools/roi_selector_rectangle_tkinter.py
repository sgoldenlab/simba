from tkinter import *
from typing import Tuple

import cv2
from PIL import Image, ImageTk

from simba.roi_tools.roi_utils import get_image_from_label
from simba.utils.checks import (check_if_valid_rgb_tuple, check_instance,
                                check_int)
from simba.utils.warnings import ROIWarning

DRAW_FRAME_NAME = "DEFINE SHAPE"

class ROISelector:
    """
    A class to allow users to interactively select a Region of Interest (ROI) rectangle from an image displayed in a Tkinter window.

    .. video:: tutorials_rst/img/roi/draw_rectangle.webm
       :width: 900
       :loop:

    .. seealso::
       For OpenCV based method, see :func:`simba.video_processors.roi_selector.ROISelector`

    :param img_window (Toplevel): The Tkinter Toplevel window containing the image.
    :param thickness (int): The thickness of the ROI selection rectangle.
    :param clr (Tuple[int, int, int]): The color of the ROI selection rectangle in RGB format.

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

        self.roi_start, self.roi_end = None, None
        self.selecting_roi, self.complete = False, False

        self.img_window.bind("<ButtonPress-1>", self.on_mouse_down)
        self.img_window.bind("<B1-Motion>", self.on_mouse_drag)
        self.img_window.bind("<ButtonRelease-1>", self.on_mouse_up)

    def on_mouse_down(self, event):
        self.roi_start = (event.x, event.y)
        self.roi_end = (event.x, event.y)
        self.selecting_roi = True
#
    def on_mouse_drag(self, event):
        if self.selecting_roi:
            self.roi_end = (event.x, event.y)
            self.img_cpy = self.img.copy()
            cv2.rectangle(self.img_cpy, self.roi_start, self.roi_end, self.clr, self.thickness)
            self.update_image(self.img_cpy)

    def on_mouse_up(self, event):
        self.roi_end = (event.x, event.y)
        self.selecting_roi = False
        self.complete = True
    #
    def update_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        tk_image = ImageTk.PhotoImage(pil_image)
        self.img_lbl.configure(image=tk_image)
        self.img_lbl.image = tk_image

    def get_attributes(self):
        self.top_left = min(self.roi_start[0], self.roi_end[0]), min(self.roi_start[1], self.roi_end[1])
        self.bottom_right = max(self.roi_start[0], self.roi_end[0]), max(self.roi_start[1], self.roi_end[1])
        if self.top_left[0] < 0:
            self.top_left = (0, self.top_left[1])
        if self.top_left[1] < 0:
            self.top_left = (self.top_left[0], 0)
        if self.bottom_right[0] < 0:
            self.bottom_right = (0, self.bottom_right[1])
        if self.bottom_right[1] < 0:
            self.bottom_right = (self.bottom_right[0], 0)
        if self.bottom_right[0] > self.w:
            self.bottom_right = (self.w, self.bottom_right[1])
        if self.bottom_right[1] > self.h:
            self.bottom_right = (self.bottom_right[0], self.h)
        self.width = self.bottom_right[0] - self.top_left[0]
        self.height = self.bottom_right[1] - self.top_left[1]
        self.center = (int(self.top_left[0] + (self.width / 2)), int(self.top_left[1] + (self.height / 2)))

        self.bottom_right_tag = (int(self.top_left[0] + self.width), int(self.top_left[1] + self.height))
        self.top_right_tag = (int(self.top_left[0] + self.width), int(self.top_left[1]))
        self.bottom_left_tag = (int(self.top_left[0]), int(self.top_left[1] + self.height))
        self.top_tag = (int(self.top_left[0] + self.width / 2), int(self.top_left[1]))
        self.right_tag = (int(self.top_left[0] + self.width), int(self.top_left[1] + self.height / 2))
        self.left_tag = (int(self.top_left[0]), int(self.top_left[1] + self.height / 2))
        self.bottom_tag = (int(self.top_left[0] + self.width / 2), int(self.top_left[1] + self.height))

        self.img_window.unbind("<ButtonPress-1>")
        self.img_window.unbind("<B1-Motion>")
        self.img_window.unbind("<ButtonRelease-1>")

        if (self.width == 0 and self.height == 0) or (self.width + self.height + self.top_left[0] + self.top_left[1] == 0):
            ROIWarning(msg="ROI WARNING: ROI height and width are both 0. Please try again.", source=self.__class__.__name__)
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
# _ = ROISelector(img_window=root)
# root.mainloop()