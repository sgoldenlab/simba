__author__ = "Simon Nilsson; sronilsson@gmail.com"

import math
from tkinter import *
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from simba.mixins.plotting_mixin import PlottingMixin
from simba.roi_tools.roi_utils import get_image_from_label
from simba.ui.tkinter_functions import SimBALabel
from simba.utils.checks import (check_float, check_if_valid_rgb_tuple,
                                check_instance, check_int)
from simba.utils.enums import TextOptions, TkBinds

DRAW_FRAME_NAME = "DEFINE SHAPE"

class ROIRuler(object):

    """
    Interactive Tkinter-based ruler tool for measuring distances on ROI images.

    .. video:: _static/img/ROIRuler.webm
       :width: 800
       :autoplay:
       :loop:

    :param Toplevel img_window: Tkinter Toplevel window containing an image label named 'img_lbl'.
    :param Optional[int] thickness: Thickness of the main line in pixels. If None, automatically calculated based on image size using optimal circle size ratio. Default None.
    :param Optional[int] second_thickness: Thickness of the outline line in pixels. If None, set to 2x the main line thickness. Default None.
    :param Optional[Tuple[int, int, int]] clr: RGB color tuple (R, G, B) for the main line. If None, uses default text color from TextOptions. Default None.
    :param Optional[Tuple[int, int, int]] second_clr: RGB color tuple (R, G, B) for the outline line. If None, no outline is drawn. Default None.
    :param int tolerance: Maximum distance in pixels from a line endpoint to detect a click for moving it. Also minimum line length required to register the line. Default 10.
    :param Optional[float] px_per_mm: Pixels per millimeter conversion factor. If provided, calculates line length in millimeters. Must be > 0. Default None.
    :param Optional[SimBALabel] info_label: Optional Tkinter label to display line length information. If provided, automatically updates with "RULER LENGTH: X mm, Y pixels" when line is drawn. Default None.
    """

    def __init__(self,
                 img_window: Toplevel,
                 thickness: Optional[int] = None,
                 second_thickness: Optional[int] = None,
                 clr: Tuple[int, int, int] = None,
                 second_clr: Tuple[int, int, int] = None,
                 tolerance: int = 10,
                 px_per_mm: Optional[float] = None,
                 info_label: Optional[SimBALabel] = None) -> None:

        check_instance(source=self.__class__.__name__, instance=img_window, accepted_types=(Toplevel,))
        if thickness is not None: check_int(name=f'{self.__class__.__name__} thickness', value=thickness, min_value=1)
        if second_thickness is not None: check_int(name=f'{self.__class__.__name__} second_thickness', value=second_thickness, min_value=1)
        if px_per_mm is not None: check_float(name=f'{self.__class__.__name__} px_per_mm', value=px_per_mm, min_value=10e-16)
        check_int(name=f'{self.__class__.__name__} tolerance', value=tolerance, min_value=1)
        #if info_label is not None: check_instance(source=f'{self.__class__.__name__} info_label', instance=info_label, accepted_types=type(SimBALabel), raise_error=True, warning=False)
        if clr is not None: check_if_valid_rgb_tuple(data=clr, raise_error=True, source=f'{self.__class__.__name__} clr')
        else: clr = (0, 255, 255)
        if second_clr is not None: check_if_valid_rgb_tuple(data=second_clr, raise_error=True, source=f'{self.__class__.__name__} second_clr')
        self.thickness, self.clr, self.img_window = thickness, clr, img_window
        self.drawing, self.clr, self.thickness, self.second_thickness, self.tolerance = False, clr, thickness, second_thickness, tolerance
        self.img_lbl = img_window.nametowidget("img_lbl")
        self.img, self.info_lbl = get_image_from_label(self.img_lbl), info_label
        self.click_locs, self.px_per_mm = {'start': None, 'end': None}, px_per_mm
        self.move_tag, self.second_clr = None, second_clr
        self.auto_size = PlottingMixin().get_optimal_circle_size(frame_size=tuple(self.img.shape[0:2]), circle_frame_ratio=200)
        if thickness is None: self.thickness = self.auto_size
        if second_thickness is None: self.second_thickness = int(self.thickness * 2.0)
        self.img_cpy, self.got_attributes = self.img.copy(), False
        self.w, self.h, self.drawing = self.img.shape[1], self.img.shape[0], False
        self._bind_mouse()

    def _bind_mouse(self):
        self.img_window.bind(TkBinds.B1_PRESS.value, self._mouse_press)
        self.img_window.bind(TkBinds.B1_MOTION.value, self._mouse_move)
        self.img_window.bind(TkBinds.B1_RELEASE.value, self._mouse_release)

    def unbind_mouse(self):
        self.img_window.unbind(TkBinds.B1_PRESS.value)
        self.img_window.unbind(TkBinds.B1_MOTION.value)
        self.img_window.unbind(TkBinds.B1_RELEASE.value)

    def _find_proximal_tag(self, click_coordinate: Tuple[int, int]):
        proximal_loc, proximal_name = None, None
        for loc_name, loc_cord in self.click_locs.items():
            if loc_cord is not None:
                distance = math.sqrt((loc_cord[0] - click_coordinate[0]) ** 2 + (loc_cord[1] - click_coordinate[1]) ** 2)
                if distance <= self.tolerance:
                    proximal_loc, proximal_name = loc_cord, loc_name
        return proximal_loc, proximal_name

    def _mouse_press(self, event):
        if not self.drawing:
            self.click_locs['start'], self.drawing = (event.x, event.y), True
            self._update_image(self.img_cpy)
        else:
            self.proximal_loc, self.proximal_name = self._find_proximal_tag(click_coordinate=(event.x, event.y))
            if self.proximal_loc is not None:
                self.move_tag = self.proximal_name


    def _mouse_move(self, event):
        if self.move_tag is not None:
            self.click_locs[self.move_tag] = (event.x, event.y)
            if self.click_locs['start'] is not None and self.click_locs['end'] is not None:
                self.img_cpy = self.img.copy()
                if self.second_clr is not None:
                    self.img_cpy = cv2.arrowedLine(self.img_cpy, tuple(self.click_locs['start']), tuple(self.click_locs['end']), self.second_clr, self.second_thickness, tipLength=0.2)
                    self.img_cpy = cv2.arrowedLine(self.img_cpy, tuple(self.click_locs['end']), tuple(self.click_locs['start']), self.second_clr, self.second_thickness, tipLength=0.2)
                self.img_cpy = cv2.arrowedLine(self.img_cpy, tuple(self.click_locs['start']), tuple(self.click_locs['end']), self.clr, self.thickness, tipLength=0.2)
                self.img_cpy = cv2.arrowedLine(self.img_cpy, tuple(self.click_locs['end']), tuple(self.click_locs['start']), self.clr, self.thickness, tipLength=0.2)

                self._update_image(self.img_cpy)
        elif self.drawing and self.click_locs['start'] is not None:
            self.click_locs['end'] = (event.x, event.y)
            if self.click_locs['end'] is not None:
                distance = np.linalg.norm(np.array(self.click_locs['start']) - np.array(self.click_locs['end']))
                if distance >= self.tolerance:
                    self.img_cpy = self.img.copy()
                    if self.second_clr is not None:
                        self.img_cpy = cv2.arrowedLine(self.img_cpy, tuple(self.click_locs['start']), tuple(self.click_locs['end']), self.second_clr, self.second_thickness, tipLength=0.2)
                        self.img_cpy = cv2.arrowedLine(self.img_cpy, tuple(self.click_locs['end']), tuple(self.click_locs['start']), self.second_clr, self.second_thickness, tipLength=0.2)
                    self.img_cpy = cv2.arrowedLine(self.img_cpy, tuple(self.click_locs['start']), tuple(self.click_locs['end']), self.clr, self.thickness, tipLength=0.2)
                    self.img_cpy = cv2.arrowedLine(self.img_cpy, tuple(self.click_locs['end']), tuple(self.click_locs['start']), self.clr, self.thickness, tipLength=0.2)

                    self._update_image(self.img_cpy)

    def _update_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        tk_image = ImageTk.PhotoImage(pil_image)
        self.img_lbl.configure(image=tk_image)
        self.img_lbl.image = tk_image


    def _mouse_release(self, event):
        if self.move_tag is not None:
            self.click_locs[self.move_tag] = (event.x, event.y)
            self.move_tag = None
        else:
            self.click_locs['end'] = (event.x, event.y)
        distance = np.linalg.norm(np.array(self.click_locs['start']) - np.array(self.click_locs['end']))
        if distance >= self.tolerance:
            self.img_cpy = self.img.copy()
            if self.second_clr is not None:
                self.img_cpy = cv2.arrowedLine(self.img_cpy, tuple(self.click_locs['start']), tuple(self.click_locs['end']), self.second_clr, self.second_thickness, tipLength=0.2)
                self.img_cpy = cv2.arrowedLine(self.img_cpy, tuple(self.click_locs['end']), tuple(self.click_locs['start']), self.second_clr, self.second_thickness, tipLength=0.2)
            self.img_cpy = cv2.arrowedLine(self.img_cpy, tuple(self.click_locs['start']), tuple(self.click_locs['end']), self.clr, self.thickness, tipLength=0.2)
            self.img_cpy = cv2.arrowedLine(self.img_cpy, tuple(self.click_locs['end']), tuple(self.click_locs['start']), self.clr, self.thickness, tipLength=0.2)
            self._update_image(self.img_cpy)
            self._get_attributes()

    def _get_attributes(self):
        self.start_pos, self.end_pos = self.click_locs['start'], self.click_locs['end']
        self.length_px = round(np.linalg.norm(np.array(self.click_locs['start']) - np.array(self.click_locs['end'])), 4)
        if self.px_per_mm is not None: self.length_mm = round((self.length_px / self.px_per_mm), 4)
        else: self.length_mm = None
        self.got_attributes = True
        if self.info_lbl is not None:
            self.info_lbl.configure(text=f'RULER LENGTH: {self.length_mm} mm, {self.length_px} pixels (Convertion factor: {self.px_per_mm})', fg='blue')
            self.info_lbl.update_idletasks()




# img = cv2.imread(r"C:\Users\sroni\OneDrive\Desktop\webp_20251218114745\BlobTrackingUI.webp")
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
# _ = ROIRuler(img_window=root, second_clr=(0, 0, 0), px_per_mm=1.5)
# root.mainloop()

