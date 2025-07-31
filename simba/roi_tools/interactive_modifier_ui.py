import math
from tkinter import *
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from shapely.geometry import Polygon

from simba.mixins.plotting_mixin import PlottingMixin
from simba.roi_tools.roi_utils import (get_circle_df_headers,
                                       get_polygon_df_headers,
                                       get_rectangle_df_headers,
                                       insert_gridlines_on_roi_img)
from simba.utils.checks import check_valid_polygon
from simba.utils.enums import ROI_SETTINGS, Keys

DRAW_FRAME_NAME = "DEFINE SHAPE"

BR_TAG = 'Bottom right tag'
B_TAG = 'Bottom tag'
T_TAG = 'Top tag'
C_TAG = 'Center tag'
BL_TAG = 'Bottom left tag'
TR_TAG = 'Top right tag'
TL_TAG = 'Top left tag'
R_TAG = 'Right tag'
L_TAG = 'Left tag'
BR_X = "Bottom_right_X"
BR_Y = "Bottom_right_Y"
TL_X = 'topLeftX'
TL_Y = "topLeftY"
CENTER_X, CENTER_Y = "Center_X", "Center_Y"
HEIGHT, WIDTH = 'height', 'width'
BORDER_TAG = 'Border tag'
CIRCLE_C_X = 'centerX'
CIRCLE_C_Y = 'centerY'
RADIUS = 'radius'
VERTICES = 'vertices'
OVERLAY_GRID_COLOR = 'OVERLAY_GRID_COLOR'

def _plot_roi(roi_dict: dict, 
              img: np.ndarray,
              show_tags: bool = False,
              omitted_roi: Optional[str] = None):
    
    rectangles_df, circles_df, polygon_df = pd.DataFrame(columns=get_rectangle_df_headers()), pd.DataFrame(columns=get_circle_df_headers()), pd.DataFrame(columns=get_polygon_df_headers())
    for roi_name, roi_data in roi_dict.items():
        if (roi_data['Shape_type'].lower() == ROI_SETTINGS.RECTANGLE.value) and (roi_name != omitted_roi):
            rectangles_df = pd.concat([rectangles_df, pd.DataFrame([roi_data])], ignore_index=True)
        elif roi_data['Shape_type'].lower() == ROI_SETTINGS.CIRCLE.value and (roi_name != omitted_roi):
            circles_df = pd.concat([circles_df, pd.DataFrame([roi_data])], ignore_index=True)
        elif roi_data['Shape_type'].lower() == ROI_SETTINGS.POLYGON.value and (roi_name != omitted_roi):
            polygon_df = pd.concat([polygon_df, pd.DataFrame([roi_data])], ignore_index=True)
    roi_dict = {Keys.ROI_RECTANGLES.value: rectangles_df, Keys.ROI_CIRCLES.value: circles_df, Keys.ROI_POLYGONS.value: polygon_df}
    img = PlottingMixin.roi_dict_onto_img(img=img, roi_dict=roi_dict, circle_size=None, show_tags=show_tags)


    return img


class InteractiveROIModifier():

    def __init__(self,
                 window_name: DRAW_FRAME_NAME,
                 roi_dict: dict,
                 img: np.ndarray,
                 orginal_img: np.ndarray,
                 settings: Optional[dict] = None,
                 tkinter_window: bool = True,
                 hex_grid: Optional[List[Polygon]] = None,
                 rectangle_grid: Optional[List[Polygon]] = None):

        self.window_name = window_name
        self.roi_dict = roi_dict
        self.img = img
        self.img_h, self.img_w = (img.shape[0], img.shape[1])
        self.circle_size = PlottingMixin().get_optimal_circle_size(frame_size=(self.img_w, self.img_h), circle_frame_ratio=100)

        self.original = orginal_img
        self.dragging = False
        self.temp_img = None
        self.tkinter_window = tkinter_window
        self.terminate = False
        self.rectangle_grid, self.hex_grid = rectangle_grid, hex_grid
        if settings is None:
            self.settings = {item.name: item.value for item in ROI_SETTINGS}
        else:
            self.settings = settings
        if self.tkinter_window:
            self.root = Tk()
            self.root.title(DRAW_FRAME_NAME)
            self.root.geometry(f"{self.img_w}x{self.img_h}")
            self.img_tk_frame = Frame(self.root, width=self.img_w, height=self.img_h, bg="gray")
            self.img_tk_frame.pack(pady=20)
            self.root.mainloop()


    def get_gridline_copy(self):
        gridline_img = self.original.copy()
        if self.rectangle_grid is not None:
            gridline_img = insert_gridlines_on_roi_img(img=gridline_img, grid=self.rectangle_grid, color=self.settings[OVERLAY_GRID_COLOR], thickness=max(1, int(self.circle_size/5)))
        if self.hex_grid is not None:
            gridline_img = insert_gridlines_on_roi_img(img=gridline_img, grid=self.rectangle_grid, color=self.settings[OVERLAY_GRID_COLOR], thickness=max(1, int(self.circle_size / 5)))
        return gridline_img


    def find_closest_tag(self, roi_dict: dict, click_coordinate: Tuple[int, int]):
        clicked_roi, clicked_tag = None, None
        for roi_name, roi_data in roi_dict.items():
            ear_tag_size = roi_data['Ear_tag_size']
            for roi_tag_name, roi_tag_coordinate in roi_data['Tags'].items():
                distance = math.sqrt((roi_tag_coordinate[0] - click_coordinate[0]) ** 2 + (roi_tag_coordinate[1] - click_coordinate[1]) ** 2)
                if distance <= ear_tag_size:
                    clicked_roi, clicked_tag = roi_data, roi_tag_name
        return clicked_roi, clicked_tag

    def _select_rectangle_bottom_right(self):
        self.clicked_roi[BR_X], self.clicked_roi[BR_Y] = self.x, self.y
        self.clicked_roi[HEIGHT], self.clicked_roi[WIDTH] = self.h, self.w
        self.clicked_roi[CENTER_X], self.clicked_roi[CENTER_Y] = int(self.tl_x + self.w / 2), int(self.tl_y + self.h / 2)
        self.clicked_roi['Tags'][TR_TAG] = (self.x, self.y - self.h)
        self.clicked_roi['Tags'][BL_TAG] = (self.x - self.w, self.y)
        self.clicked_roi['Tags'][BR_TAG] = (self.x, self.y)
        self.clicked_roi['Tags'][R_TAG] = (self.x, int(self.tl_y + (self.h / 2)))
        self.clicked_roi['Tags'][L_TAG] = (self.tl_x, int(self.tl_y + (self.h / 2)))
        self.clicked_roi['Tags'][B_TAG] = (int(self.tl_x + self.w / 2), self.y)
        self.clicked_roi['Tags'][T_TAG] = (int(self.tl_x + self.w / 2), self.y - self.h)
        self.clicked_roi['Tags'][C_TAG] = (self.clicked_roi[CENTER_X], self.clicked_roi[CENTER_Y])
        cv2.rectangle(self.temp_img, self.clicked_roi['Tags'][TL_TAG], (self.x, self.y), self.clicked_roi['Color BGR'],self.clicked_roi['Thickness'])
        cv2.line(self.temp_img, (self.x, self.y), self.clicked_roi['Tags'][BL_TAG], self.settings['ROI_SELECT_CLR'], self.clicked_roi['Thickness'], lineType=self.settings['LINE_TYPE'])
        cv2.line(self.temp_img, (self.x, self.y), self.clicked_roi['Tags'][TR_TAG], self.settings['ROI_SELECT_CLR'], self.clicked_roi['Thickness'], lineType=self.settings['LINE_TYPE'])

    def _select_rectangle_right(self):
        self.clicked_roi[WIDTH] = self.w
        self.clicked_roi[CENTER_X], self.clicked_roi[CENTER_Y] = int(self.tl_x + self.w / 2), int(self.tl_y + self.h / 2)
        self.clicked_roi['Tags'][TR_TAG] = (self.x, self.tl_y)
        self.clicked_roi['Tags'][BR_TAG] = (self.x, self.tl_y + self.h)
        self.clicked_roi['Tags'][R_TAG] = (self.tl_x + self.w, self.tl_y + int(self.h / 2))
        self.clicked_roi['Tags'][T_TAG] = (self.tl_x + int(self.w / 2), self.tl_y)
        self.clicked_roi['Tags'][B_TAG] = (self.tl_x + int(self.w / 2), self.tl_y + self.h)
        self.clicked_roi['Tags'][C_TAG] = (self.tl_x + int(self.w / 2), self.tl_y + int(self.h / 2))
        self.clicked_roi[BR_X] = self.x
        self.clicked_roi[BR_Y] = self.tl_y + self.h
        cv2.rectangle(self.temp_img, self.clicked_roi['Tags'][TL_TAG], self.clicked_roi['Tags'][BR_TAG], self.clicked_roi['Color BGR'], self.clicked_roi['Thickness'])
        cv2.line(self.temp_img, self.clicked_roi['Tags'][BR_TAG], self.clicked_roi['Tags'][TR_TAG], self.settings['ROI_SELECT_CLR'], self.clicked_roi['Thickness'], lineType=self.settings['LINE_TYPE'])

    def _select_rectangle_bottom(self):
        self.clicked_roi['Tags'][BR_TAG] = (self.tl_x + self.w, self.tl_y + self.h)
        self.clicked_roi['Tags'][BL_TAG] = (self.tl_x, self.tl_y + self.h)
        self.clicked_roi['Tags'][L_TAG] = (self.tl_x, self.tl_y + int(self.h / 2))
        self.clicked_roi['Tags'][R_TAG] = (self.tl_x + self.w, self.tl_y + int(self.h / 2))
        self.clicked_roi['Tags'][B_TAG] = (self.tl_x + int(self.w / 2), self.tl_y + self.h)
        self.clicked_roi['Tags'][C_TAG] = (self.tl_x + int(self.w / 2), self.tl_y + int(self.h / 2))
        self.clicked_roi[BR_X] = self.tl_x + self.w
        self.clicked_roi[BR_Y] = self.tl_y + self.h
        self.clicked_roi[HEIGHT] = self.h
        self.clicked_roi[CENTER_X], self.clicked_roi[CENTER_Y] = self.tl_x + int(self.w / 2), self.tl_y + int(self.h / 2)
        cv2.rectangle(self.temp_img, self.clicked_roi['Tags'][TL_TAG], self.clicked_roi['Tags'][BR_TAG], self.clicked_roi['Color BGR'], self.clicked_roi['Thickness'])
        cv2.line(self.temp_img, self.clicked_roi['Tags'][BL_TAG], self.clicked_roi['Tags'][BR_TAG], self.settings['ROI_SELECT_CLR'], self.clicked_roi['Thickness'], lineType=self.settings['LINE_TYPE'])


    def _select_rectangle_bottom_left(self):
        self.clicked_roi['Tags'][TL_TAG] = (self.x, self.tr_y)
        self.clicked_roi['Tags'][L_TAG] = (self.x, self.tr_y + int(self.h / 2))
        self.clicked_roi['Tags'][BL_TAG] = (self.x, self.y)
        self.clicked_roi['Tags'][B_TAG] = (self.x + int(self.w / 2), self.y)
        self.clicked_roi['Tags'][BR_TAG] = (self.x + self.w, self.tr_y + self.h)
        self.clicked_roi['Tags'][R_TAG] = (self.x + self.w, self.tr_y + int(self.h / 2))
        self.clicked_roi['Tags'][T_TAG] = (self.x + int(self.w / 2), self.tr_y)
        self.clicked_roi['Tags'][C_TAG] = (self.x + int(self.w / 2), self.tr_y + int(self.h / 2))
        self.clicked_roi[TL_X] = self.x
        self.clicked_roi[BR_X] = self.x + self.w
        self.clicked_roi[BR_Y] = self.tr_y + self.h
        self.clicked_roi[HEIGHT] = self.h
        self.clicked_roi[WIDTH] = self.w
        self.clicked_roi[CENTER_X], self.clicked_roi[CENTER_Y] = self.x + int(self.w / 2), self.tr_y + int(self.h / 2)
        cv2.rectangle(self.temp_img, self.clicked_roi['Tags'][TL_TAG], self.clicked_roi['Tags'][BR_TAG], self.clicked_roi['Color BGR'], self.clicked_roi['Thickness'])
        cv2.line(self.temp_img, self.clicked_roi['Tags'][BL_TAG], self.clicked_roi['Tags'][TL_TAG], self.settings['ROI_SELECT_CLR'], self.clicked_roi['Thickness'], lineType=self.settings['LINE_TYPE'])
        cv2.line(self.temp_img, self.clicked_roi['Tags'][BL_TAG], self.clicked_roi['Tags'][BR_TAG], self.settings['ROI_SELECT_CLR'], self.clicked_roi['Thickness'], lineType=self.settings['LINE_TYPE'])

    def _select_rectangle_left_tag(self):
        self.clicked_roi[WIDTH] = self.w
        self.clicked_roi[CENTER_X], self.clicked_roi[CENTER_Y] = int(self.tr_x - (self.w / 2)), int(self.tr_y + self.h / 2)
        self.clicked_roi['Tags'][BL_TAG] = (self.tr_x - self.w, self.tr_y + self.h)
        self.clicked_roi['Tags'][TL_TAG] = (self.tr_x - self.w, self.tr_y)
        self.clicked_roi['Tags'][L_TAG] = (self.x, int(self.tr_y + self.h / 2))
        self.clicked_roi['Tags'][B_TAG] = (self.tr_x - int(self.w / 2), self.tr_y + self.h)
        self.clicked_roi['Tags'][T_TAG] = (self.tr_x - int(self.w / 2), self.tr_y)
        self.clicked_roi['Tags'][C_TAG] = (int(self.tr_x - (self.w / 2)), int(self.tr_y + self.h / 2))
        self.clicked_roi[TL_X] = self.tr_x - self.w
        cv2.rectangle(self.temp_img, self.clicked_roi['Tags'][TL_TAG], self.clicked_roi['Tags'][BR_TAG], self.clicked_roi['Color BGR'], self.clicked_roi['Thickness'])
        cv2.line(self.temp_img, self.clicked_roi['Tags'][BL_TAG], self.clicked_roi['Tags'][TL_TAG], self.settings['ROI_SELECT_CLR'], self.clicked_roi['Thickness'], lineType=self.settings['LINE_TYPE'])

    def _select_rectangle_top_tag(self):
        self.clicked_roi['Tags'][TR_TAG] = (self.br_x, self.br_y - self.h)
        self.clicked_roi['Tags'][TL_TAG] = (self.br_x - self.w, self.br_y - self.h)
        self.clicked_roi['Tags'][T_TAG] = (self.br_x - int(self.w / 2), self.br_y - self.h)
        self.clicked_roi['Tags'][L_TAG] = (self.br_x - self.w, self.br_y - int(self.h / 2))
        self.clicked_roi['Tags'][R_TAG] = (self.br_x, self.br_y - int(self.h / 2))
        self.clicked_roi['Tags'][C_TAG] = (self.br_x - int(self.w / 2), self.br_y - int(self.h / 2))
        self.clicked_roi[TL_X] = self.br_x - self.w
        self.clicked_roi[TL_Y] = self.br_y - self.h
        self.clicked_roi[HEIGHT] = self.h
        self.clicked_roi[CENTER_X], self.clicked_roi[CENTER_Y] = self.br_x - int(self.w / 2), self.br_y - int(self.h / 2)
        cv2.rectangle(self.temp_img, self.clicked_roi['Tags'][TL_TAG], self.clicked_roi['Tags'][BR_TAG], self.clicked_roi['Color BGR'], self.clicked_roi['Thickness'])
        cv2.line(self.temp_img, self.clicked_roi['Tags'][TL_TAG], self.clicked_roi['Tags'][TR_TAG], self.settings['ROI_SELECT_CLR'], self.clicked_roi['Thickness'], lineType=self.settings['LINE_TYPE'])


    def _select_rectangle_top_left_tag(self):
        self.clicked_roi['Tags'][TR_TAG] = (self.br_x, self.br_y - self.h)
        self.clicked_roi['Tags'][T_TAG] = (self.br_x - int(self.w / 2), self.br_y - self.h)
        self.clicked_roi['Tags'][L_TAG] = (self.br_x - self.w, self.br_y - int(self.h / 2))
        self.clicked_roi['Tags'][R_TAG] = (self.br_x, self.br_y - int(self.h / 2))
        self.clicked_roi['Tags'][BL_TAG] = (self.x, self.br_y)
        self.clicked_roi['Tags'][B_TAG] = (self.x + int(self.w/2), self.br_y)
        self.clicked_roi['Tags'][TL_TAG] = (self.x, self.y)
        self.clicked_roi['Tags'][C_TAG] = (self.x + int(self.w / 2), self.y + int(self.h / 2))
        self.clicked_roi[TL_X] = self.x
        self.clicked_roi[TL_Y] = self.y
        self.clicked_roi[HEIGHT] = self.h
        self.clicked_roi[WIDTH] = self.w
        self.clicked_roi[CENTER_X], self.clicked_roi[CENTER_Y] = self.x + int(self.w / 2), self.y + int(self.h / 2)
        cv2.rectangle(self.temp_img, self.clicked_roi['Tags'][TL_TAG], self.clicked_roi['Tags'][BR_TAG], self.clicked_roi['Color BGR'], self.clicked_roi['Thickness'])
        cv2.line(self.temp_img, self.clicked_roi['Tags'][TL_TAG], self.clicked_roi['Tags'][TR_TAG], self.settings['ROI_SELECT_CLR'], self.clicked_roi['Thickness'], lineType=self.settings['LINE_TYPE'])
        cv2.line(self.temp_img, self.clicked_roi['Tags'][TL_TAG], self.clicked_roi['Tags'][BL_TAG], self.settings['ROI_SELECT_CLR'], self.clicked_roi['Thickness'], lineType=self.settings['LINE_TYPE'])

    def _select_rectangle_top_right_tag(self):
        self.clicked_roi['Tags'][TL_TAG] = (self.bl_x, self.y)
        self.clicked_roi['Tags'][L_TAG] = (self.bl_x, self.bl_y - int(self.h / 2))
        self.clicked_roi['Tags'][T_TAG] = (self.bl_x + int(self.w / 2), self.y)
        self.clicked_roi['Tags'][TR_TAG] = (self.x, self.y)
        self.clicked_roi['Tags'][R_TAG] = (self.x, self.bl_y - int(self.h / 2))
        self.clicked_roi['Tags'][BR_TAG] = (self.x, self.y + self.h)
        self.clicked_roi['Tags'][B_TAG] = (self.x - int(self.w / 2), self.bl_y)
        self.clicked_roi['Tags'][C_TAG] = (self.bl_x + int(self.w / 2), self.bl_y - int(self.h / 2))
        self.clicked_roi[TL_X] = self.bl_x
        self.clicked_roi[TL_Y] = self.y
        self.clicked_roi[BR_X] = self.x
        self.clicked_roi[BR_Y] = self.y + self.h
        self.clicked_roi[HEIGHT] = self.h
        self.clicked_roi[WIDTH] = self.w
        self.clicked_roi[CENTER_X], self.clicked_roi[CENTER_Y] = self.bl_x + int(self.w / 2), self.bl_y - int(self.h / 2)
        cv2.rectangle(self.temp_img, self.clicked_roi['Tags'][TL_TAG], self.clicked_roi['Tags'][BR_TAG], self.clicked_roi['Color BGR'], self.clicked_roi['Thickness'])
        cv2.line(self.temp_img, self.clicked_roi['Tags'][TL_TAG], self.clicked_roi['Tags'][TR_TAG], self.settings['ROI_SELECT_CLR'], self.clicked_roi['Thickness'], lineType=self.settings['LINE_TYPE'])
        cv2.line(self.temp_img, self.clicked_roi['Tags'][TR_TAG], self.clicked_roi['Tags'][BR_TAG], self.settings['ROI_SELECT_CLR'], self.clicked_roi['Thickness'], lineType=self.settings['LINE_TYPE'])

    def _select_rectangle_center_tag(self):
        self.clicked_roi['Tags'][TL_TAG] = (self.tl_x, self.tl_y)
        self.clicked_roi['Tags'][L_TAG] = (self.tl_x, self.tl_y + int(self.h / 2))
        self.clicked_roi['Tags'][BL_TAG] = (self.tl_x, self.tl_y + self.h)
        self.clicked_roi['Tags'][B_TAG] = (self.tl_x + int(self.w / 2), self.tl_y + self.h)
        self.clicked_roi['Tags'][BR_TAG] = (self.br_x, self.br_y)
        self.clicked_roi['Tags'][R_TAG] = (self.tl_x + self.w, self.tl_y + int(self.h / 2))
        self.clicked_roi['Tags'][TR_TAG] = (self.tl_x + self.w, self.tl_y)
        self.clicked_roi['Tags'][T_TAG] = (self.tl_x + int(self.w / 2), self.tl_y)
        self.clicked_roi['Tags'][C_TAG] = (self.x, self.y)
        self.clicked_roi[TL_X] = self.tl_x
        self.clicked_roi[TL_Y] = self.tl_y
        self.clicked_roi[BR_X] = self.br_x
        self.clicked_roi[BR_Y] = self.br_y
        self.clicked_roi[HEIGHT] = self.h
        self.clicked_roi[WIDTH] = self.w
        self.clicked_roi[CENTER_X], self.clicked_roi[CENTER_Y] = self.x, self.y
        cv2.rectangle(self.temp_img, self.clicked_roi['Tags'][TL_TAG], self.clicked_roi['Tags'][BR_TAG], self.settings['ROI_SELECT_CLR'], self.clicked_roi['Thickness'], lineType=self.settings['LINE_TYPE'])

    def _select_circle_border(self):
        self.clicked_roi[RADIUS] = self.radius
        self.clicked_roi["Tags"][BORDER_TAG] = self.l_edge
        cv2.circle(self.temp_img, center=self.clicked_roi["Tags"][C_TAG], radius=self.radius, color=self.settings['ROI_SELECT_CLR'], thickness=self.clicked_roi['Thickness'], lineType=self.settings['LINE_TYPE'])

    def _select_circle_center(self):
        self.clicked_roi[CIRCLE_C_X], self.clicked_roi[CIRCLE_C_Y] = self.x, self.y
        self.clicked_roi["Tags"][C_TAG] = (self.x, self.y)
        self.clicked_roi["Tags"][BORDER_TAG] = self.l_edge
        cv2.circle(self.temp_img, center=self.clicked_roi["Tags"][C_TAG], radius=self.radius, color=self.settings['ROI_SELECT_CLR'], thickness=self.clicked_roi['Thickness'], lineType=self.settings['LINE_TYPE'])

    def _select_polygon_center(self):
        self.new_vertices = np.array([np.array(v) for k, v in self.new_tags.items() if k != 'Center_tag']).astype(np.int32)
        self.clicked_roi['Tags'] = self.new_tags
        self.clicked_roi[VERTICES] = self.new_vertices
        self.clicked_roi[CENTER_X], self.clicked_roi[CENTER_Y] = self.clicked_roi['Tags']["Center_tag"]
        self.clicked_roi['center'] = self.clicked_roi['Tags']['Center_tag']
        cv2.polylines(self.temp_img, [self.clicked_roi[VERTICES]], True, color=self.settings['ROI_SELECT_CLR'], thickness=self.clicked_roi['Thickness'],lineType=self.settings['LINE_TYPE'])

    def _select_polygon_vertice(self):
        self.clicked_roi['Tags'][self.clicked_tag] = self.new_poly_tag_loc
        self.clicked_roi[VERTICES][self.clicked_tag_id] = np.array([self.new_poly_tag_loc])
        self.clicked_roi['center'] = self.polygon_centroid
        self.clicked_roi['area'] = self.polygon_area
        self.clicked_roi['Tags']['Center_tag'] = tuple(self.polygon_centroid)
        cv2.polylines(self.temp_img, [self.clicked_roi[VERTICES]], True, color=self.clicked_roi['Color BGR'], thickness=self.clicked_roi['Thickness'])
        cv2.line(self.temp_img, self.clicked_roi['Tags'][self.clicked_tag], self.clicked_roi['Tags'][self.n_tag_1_id], self.settings['ROI_SELECT_CLR'], self.clicked_roi['Thickness'], lineType=self.settings['LINE_TYPE'])
        cv2.line(self.temp_img, self.clicked_roi['Tags'][self.clicked_tag], self.clicked_roi['Tags'][self.n_tag_2_id], self.settings['ROI_SELECT_CLR'], self.clicked_roi['Thickness'], lineType=self.settings['LINE_TYPE'])


    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_roi, self.clicked_tag = self.find_closest_tag(roi_dict=self.roi_dict, click_coordinate=(x, y))
            if self.clicked_roi is not None:
                self.dragging = True

        elif event == cv2.EVENT_LBUTTONUP:
            if self.clicked_roi is not None:
                self.roi_dict[self.clicked_roi['Name']] = self.clicked_roi
                draw_img = self.get_gridline_copy() if self.rectangle_grid is not None or self.hex_grid is not None else self.original.copy()
                self.temp_img = _plot_roi(roi_dict=self.roi_dict, img=draw_img, show_tags=True)
            self.dragging = False

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.x, self.y = x, y
            draw_img = self.get_gridline_copy() if self.rectangle_grid is not None or self.hex_grid is not None else self.original.copy()
            self.temp_img = _plot_roi(roi_dict=self.roi_dict, omitted_roi=self.clicked_roi['Name'], img=draw_img)
            if self.clicked_roi['Shape_type'].lower() == ROI_SETTINGS.RECTANGLE.value:
                self.tl_x, self.tl_y = self.clicked_roi['Tags'][TL_TAG]
                self.tr_x, self.tr_y = self.clicked_roi['Tags'][TR_TAG]
                self.br_x, self.br_y = self.clicked_roi['Tags'][BR_TAG]
                self.bl_x, self.bl_y = self.clicked_roi['Tags'][BL_TAG]
                if self.clicked_tag == BR_TAG:
                    self.w, self.h = x - self.tl_x, y - self.tl_y
                    if (self.w > 1 < self.h) and (0 <= self.x <= self.img_w) and (0 <= self.y <= self.img_h):
                        self._select_rectangle_bottom_right()
                elif self.clicked_tag == R_TAG:
                    self.w, self.h = x - self.tl_x, self.clicked_roi[HEIGHT]
                    if (self.w > 1 < self.h) and (0 <= self.x <= self.img_w) and (0 <= self.y <= self.img_h):
                        self._select_rectangle_right()
                elif self.clicked_tag == B_TAG:
                    self.w, self.h = self.clicked_roi[WIDTH], y - self.tl_y
                    if (self.w > 1 < self.h) and (0 <= self.x <= self.img_w) and (0 <= self.y <= self.img_h):
                        self._select_rectangle_bottom()
                elif self.clicked_tag == BL_TAG:
                    self.w, self.h = self.tr_x - x, y - self.tr_y
                    if (self.w > 1 < self.h) and (0 <= self.x <= self.img_w) and (0 <= self.y <= self.img_h):
                        self._select_rectangle_bottom_left()
                elif self.clicked_tag == L_TAG:
                    self.w, self.h = self.tr_x - x, self.clicked_roi[HEIGHT]
                    if (self.w > 1 < self.h) and (0 <= self.x <= self.img_w) and (0 <= self.y <= self.img_h):
                        self._select_rectangle_left_tag()
                elif self.clicked_tag == T_TAG:
                    self.w, self.h = self.clicked_roi[WIDTH], self.br_y - self.y
                    if (self.w > 1 < self.h) and (0 <= self.x <= self.img_w) and (0 <= self.y <= self.img_h):
                        self._select_rectangle_top_tag()
                elif self.clicked_tag == TL_TAG:
                    self.w, self.h = self.br_x - self.x, self.br_y - self.y
                    if (self.w > 1 < self.h) and (0 <= self.x <= self.img_w) and (0 <= self.y <= self.img_h):
                        self._select_rectangle_top_left_tag()
                elif self.clicked_tag == TR_TAG:
                    self.w, self.h = self.x - self.bl_x, self.bl_y - self.y
                    if (self.w > 1 < self.h) and (0 <= self.x <= self.img_w) and (0 <= self.y <= self.img_h):
                        self._select_rectangle_top_right_tag()
                elif self.clicked_tag == C_TAG:
                    self.w, self.h = self.clicked_roi[WIDTH], self.clicked_roi[HEIGHT]
                    self.tl_x, self.tl_y = self.x - int(self.w/2), self.y - int(self.h/2)
                    self.tr_x, self.tr_y = self.x + int(self.w/2), self.y - int(self.h/2)
                    self.br_x, self.br_y = self.x + int(self.w/2), self.y + int(self.h/2)
                    if (0 <= self.tl_x) and (0 <= self.tl_y) and (self.tr_x <= self.img_w) and (self.br_y <= self.img_h):
                        self._select_rectangle_center_tag()

            elif self.clicked_roi['Shape_type'].lower() == ROI_SETTINGS.CIRCLE.value:
                if self.clicked_tag == BORDER_TAG:
                    c_x, c_y = self.clicked_roi["Tags"][C_TAG]
                    self.radius = self.clicked_roi["Tags"][C_TAG][0] - self.x
                    self.l_edge, t_edge =  (c_x - self.radius, c_y), (c_x, c_y - self.radius)
                    r_edge, b_edge = (c_x +self. radius, c_y), (c_x, c_y + self.radius)
                    if (0 <= self.l_edge[0]) and (0 <= t_edge[1]) and (r_edge[0] <= self.img_w) and (b_edge[1] <= self.img_h) and (self.radius >= 1):
                        self._select_circle_border()
                if self.clicked_tag == C_TAG:
                    self.radius = self.clicked_roi[RADIUS]
                    self.l_edge, t_edge = (self.x - self.radius, self.y), (self.x, self.y - self.radius)
                    r_edge, b_edge = (self.x + self.radius, self.y), (self.x, self.y + self.radius)
                    if (0 <= self.l_edge[0]) and (0 <= t_edge[1]) and (r_edge[0] <= self.img_w) and (b_edge[1] <= self.img_h):
                        self._select_circle_center()

            elif self.clicked_roi['Shape_type'].lower() == ROI_SETTINGS.POLYGON.value:
                if self.clicked_tag in ['Center_tag', 'Center tag']:
                    x_diff, y_diff = self.x - self.clicked_roi[CENTER_X], self.y - self.clicked_roi[CENTER_Y]
                    self.new_tags, outside_img_flag = {}, False
                    for tag_name, tag in self.clicked_roi['Tags'].items():
                        t  = (tag[0] + x_diff, tag[1] + y_diff)
                        self.new_tags[tag_name] = t
                        # if (0 > t[0]) or (t[0] > self.img_w) or (0 > t[1]) or (t[1] > self.img_h):
                        #     outside_img_flag = True
                    if not outside_img_flag:
                        self._select_polygon_center()
                else:
                    print()
                    self.clicked_tag_id = int(self.clicked_tag[-1])
                    click_tag = self.clicked_roi['Tags'][self.clicked_tag]
                    max_tag_id = len(self.clicked_roi["Tags"].keys()) - 2
                    self.n_tag_1_id = f'Tag_{0 if self.clicked_tag_id == max_tag_id else max_tag_id if self.clicked_tag_id == 0 else self.clicked_tag_id - 1}'
                    self.n_tag_2_id = f'Tag_{self.clicked_tag_id - 1 if self.clicked_tag_id == max_tag_id else self.clicked_tag_id + 1}'
                    x_diff, y_diff = self.x - click_tag[0], self.y - click_tag[1]
                    self.new_poly_tag_loc = (click_tag[0] + x_diff, click_tag[1] + y_diff)
                    self.new_vertices = np.copy(self.clicked_roi[VERTICES]).astype(np.int32)
                    self.new_vertices[self.clicked_tag_id] = np.array([self.new_poly_tag_loc])
                    polygon = Polygon(self.new_vertices)
                    self.polygon_area = polygon.area
                    valid_polygon = check_valid_polygon(polygon=polygon, raise_error=False)
                    self.polygon_centroid = np.array(polygon.centroid).astype(int)
                    if valid_polygon:
                        self._select_polygon_vertice()

            else:
                pass

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        while not self.terminate:
            if not self.dragging and self.temp_img is None:
                self.temp_img = np.copy(self.img)
            if not self.tkinter_window:
                cv2.imshow(self.window_name, self.temp_img)
                cv2.waitKey(1)
            else:
                pil_image = Image.fromarray(self.temp_img)
                tk_image = ImageTk.PhotoImage(pil_image)
                label = Label(self.img_tk_frame, image=tk_image)
                label.image = tk_image
                label.pack()

        if self.terminate:
            cv2.waitKey(1)
            cv2.imshow(self.window_name, self.temp_img)






#
#
# rectangle_example =  {'Video':             'My_video',
#                       'Shape_type':        ROI_SETTINGS.RECTANGLE.value,
#                       'Name':              'My_rectangle_shape',
#                       'Color name':        'red',
#                       'Color BGR':         (0, 0, 255),
#                       'Thickness':         7,
#                       'Center_X':          175,
#                       'Center_Y':          175,
#                       'topLeftX':          100,
#                       'topLeftY':          100,
#                       'Bottom_right_X':    250,
#                       'Bottom_right_Y':    250,
#                       'width':             150,
#                       'height':            150,
#                       "Tags":             {"Center tag": (175, 175),
#                                            "Top left tag": (100, 100),
#                                            "Bottom right tag": (250, 250),
#                                            "Top right tag": (250, 100),
#                                            "Bottom left tag": (100, 250),
#                                            "Top tag": (175, 100),
#                                            "Right tag": (250, 175),
#                                            "Left tag": (100, 175),
#                                            "Bottom tag": (175, 250)},
#                       'Ear_tag_size':      15}
#
#
# circle_example =      {'Video':            'My_video',
#                        'Shape_type':        ROI_SETTINGS.CIRCLE.value,
#                        'Name':              'My_circle_shape',
#                        'Color name':        'blue',
#                        'Color BGR':          (255, 0, 255),
#                        'Thickness':          7,
#                        'centerX':           400,
#                        'centerY':           400,
#                        'radius':            50,
#                        "Tags":             {"Center tag": (400, 400),
#                                             "Border tag": (350, 400)},
#                        'Ear_tag_size':      15}
#
#
#
#
# polygon  = Polygon(np.array([[300, 450], [100, 500], [400, 300], [600, 300], [650, 400]]))
# polygon_vertices  = np.array(polygon.convex_hull.exterior.coords)[1:]
# polygon_center = np.array(polygon.centroid).astype(np.int32)
# polygon_tags = {}
# for cnt, poly in enumerate(polygon_vertices):
#     polygon_tags[f'Tag_{cnt}'] = tuple(poly.astype(np.int32))
# polygon_tags['Center tag'] = tuple(polygon_center)
#
# polygon_example =      {'Video':                    'My_video',
#                        'Shape_type':                ROI_SETTINGS.POLYGON.value,
#                        'Name':                      'My_polygon_shape',
#                        'Color name':                'yellow',
#                        'Color BGR':                 (0, 255, 255),
#                        'Thickness':                 7,
#                        'Center_X':                  polygon_center[0],
#                        'Center_Y':                  polygon_center[1],
#                        'vertices':                  polygon_vertices,
#                        'center':                    (polygon_center[0], polygon_center[1]),
#                        'area':                      None,
#                        'max_vertice_distance':      None,
#                        "area_cm":                   None,
#                        'Tags':                      polygon_tags,
#                        'Ear_tag_size':              15}
#
#
#
#
#
#
#
#
# roi_dict = {'My_rectangle_shape': rectangle_example, 'My_circle_shape': circle_example, 'My_polygon_shape': polygon_example}
#
# frm = read_frm_of_video(video_path=r"C:\troubleshooting\mitra\project_folder\videos\501_MA142_Gi_CNO_0514.mp4")
# original_img = np.copy(frm)
# for shape_name, shape in roi_dict.items():
#     if shape['Shape_type'] == ROI_SETTINGS.RECTANGLE.value:
#         cv2.rectangle(frm, shape['Tags']['Top left tag'], shape['Tags']['Bottom right tag'], shape['Color BGR'], shape['Thickness'])
#         for tag_name, tag in shape['Tags'].items():
#             cv2.circle(frm, tag, shape['Ear_tag_size'], shape['Color BGR'], -1)
#     if shape['Shape_type'] == ROI_SETTINGS.CIRCLE.value:
#         cv2.circle(frm, center=shape['Tags']['Center tag'], radius=shape['radius'], color=shape['Color BGR'], thickness=shape['Thickness'])
#         for tag_name, tag in shape['Tags'].items():
#             cv2.circle(frm, tag, shape['Ear_tag_size'], shape['Color BGR'], -1)
#     if shape['Shape_type'] == ROI_SETTINGS.POLYGON.value:
#         cv2.polylines(frm, [shape["vertices"].astype(int)], True, color=shape['Color BGR'], thickness=shape['Thickness'])
#         for tag_name, tag in shape['Tags'].items():
#             cv2.circle(frm, tag, shape['Ear_tag_size'], shape['Color BGR'], -1)
#
#
# cv2.namedWindow(DRAW_FRAME_NAME, cv2.WINDOW_NORMAL)
# cv2.imshow(DRAW_FRAME_NAME, frm)
#
#
# inter = InteractiveROIModifier(window_name=DRAW_FRAME_NAME, roi_dict=roi_dict, img=frm, orginal_img=original_img)
# inter.run()
# print('s')









