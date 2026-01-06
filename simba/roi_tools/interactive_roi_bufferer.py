import math
from copy import deepcopy
from tkinter import Event, Toplevel
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from simba.mixins.geometry_mixin import GeometryMixin
from simba.mixins.plotting_mixin import PlottingMixin
from simba.roi_tools.roi_utils import (create_circle_entry,
                                       create_rectangle_entry,
                                       get_circle_df_headers,
                                       get_image_from_label,
                                       get_polygon_df_headers,
                                       get_rectangle_df_headers)
from simba.utils.checks import check_instance
from simba.utils.enums import ROI_SETTINGS, Keys, TkBinds

TAGS, SHAPE_TYPE = 'Tags', 'Shape_type'


def _plot_roi(roi_dict: dict,
              img: np.ndarray):

    rectangles_df, circles_df, polygon_df = pd.DataFrame(columns=get_rectangle_df_headers()), pd.DataFrame(columns=get_circle_df_headers()), pd.DataFrame(columns=get_polygon_df_headers())
    for roi_name, roi_data in roi_dict.items():
        if (roi_data['Shape_type'].lower() == ROI_SETTINGS.RECTANGLE.value):
            rectangles_df = pd.concat([rectangles_df, pd.DataFrame([roi_data])], ignore_index=True)
        elif roi_data['Shape_type'].lower() == ROI_SETTINGS.CIRCLE.value:
            circles_df = pd.concat([circles_df, pd.DataFrame([roi_data])], ignore_index=True)
        elif roi_data['Shape_type'].lower() == ROI_SETTINGS.POLYGON.value:
            polygon_df = pd.concat([polygon_df, pd.DataFrame([roi_data])], ignore_index=True)
    roi_dict = {Keys.ROI_RECTANGLES.value: rectangles_df, Keys.ROI_CIRCLES.value: circles_df, Keys.ROI_POLYGONS.value: polygon_df}
    img = PlottingMixin.roi_dict_onto_img(img=img, roi_dict=roi_dict, circle_size=None, show_tags=False, show_center=True, omitted_centers=list(polygon_df['Name'].unique()))
    return img

class InteractiveROIBufferer():
    """
    Interactive Tkinter-based tool for buffering (expanding or shrinking) ROI shapes by specified metric millimeter by clicking on their tags.

    :param Toplevel img_window: Tkinter Toplevel window containing an image label named 'img_lbl' displaying the ROI image.
    :param np.ndarray original_img: Original image as a numpy array in BGR format. Used as the base for redrawing ROIs.
    :param dict roi_dict: Dictionary containing ROI definitions. Keys are ROI names (str), values are dictionaries with ROI properties including 'Shape_type', 'Tags', 'Color BGR', 'Thickness', 'Ear_tag_size', etc. Expected shape types: 'Rectangle', 'Circle', or 'Polygon'.
    :param int buffer_mm: Buffer distance in millimeters. Positive values expand the shape, negative values shrink it.
    :param float px_per_mm: Pixels per millimeter conversion factor. Used to convert buffer_mm to pixels. Must be > 0.
    :param Optional[dict] settings: Optional dictionary of ROI settings. If None, uses default ROI_SETTINGS values. Default None.
    :param Optional[List[Polygon]] hex_grid: Optional list of Shapely Polygon objects representing a hexagon grid overlay. Default None.
    :param Optional[List[Polygon]] rectangle_grid: Optional list of Shapely Polygon objects representing a rectangle grid overlay. Default None.
    """

    def __init__(self,
                 img_window: Toplevel,
                 original_img: np.ndarray,
                 roi_dict: dict,
                 buffer_mm: int,
                 px_per_mm: float,
                 settings: Optional[dict] = None,
                 hex_grid: Optional[List[Polygon]] = None,
                 rectangle_grid: Optional[List[Polygon]] = None):

        check_instance(source=self.__class__.__name__, instance=img_window, accepted_types=(Toplevel,))
        if settings is None: settings = {item.name: item.value for item in ROI_SETTINGS}
        self.hex_grid, self.rectangle_grid = hex_grid, rectangle_grid
        self.img_lbl = img_window.nametowidget("img_lbl")
        self.img = get_image_from_label(self.img_lbl)
        self.original_img, self.roi_dict = deepcopy(original_img), deepcopy(roi_dict)
        _plot_roi(roi_dict=self.roi_dict, img=self.original_img.copy())
        self.img_w, self.img_h = self.img.shape[1], self.img.shape[0]
        self.img_window, self.settings, self.buffer_mm, self.px_per_mm = img_window, settings, buffer_mm, px_per_mm
        self.bind_mouse()

    def _find_closest_tag(self, roi_dict: dict, click_coordinate: Tuple[int, int]):
        clicked_roi, clicked_tag = None, None
        for roi_name, roi_data in roi_dict.items():
            ear_tag_size = roi_data['Ear_tag_size']
            for roi_tag_name, roi_tag_coordinate in roi_data[TAGS].items():
                distance = math.sqrt((roi_tag_coordinate[0] - click_coordinate[0]) ** 2 + (roi_tag_coordinate[1] - click_coordinate[1]) ** 2)
                if distance <= ear_tag_size:
                    clicked_roi, clicked_tag = roi_data, roi_tag_name
        return clicked_roi, clicked_tag

    def bind_mouse(self):
        self.img_window.bind(TkBinds.B1_PRESS.value, self.left_mouse_down)

    def unbind_mouse(self):
        self.img_window.unbind(TkBinds.B1_PRESS.value)

    def __update_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        tk_image = ImageTk.PhotoImage(pil_image)
        self.img_lbl.configure(image=tk_image)
        self.img_lbl.image = tk_image

    def left_mouse_down(self, event: Event):
        self.click_loc = (event.x, event.y)
        self.clicked_roi, self.clicked_tag = self._find_closest_tag(roi_dict=self.roi_dict, click_coordinate=self.click_loc)
        if self.clicked_roi is not None and self.clicked_roi[SHAPE_TYPE] != ROI_SETTINGS.POLYGON.value:
            original_tags = self.clicked_roi[TAGS]
            buffer_px = int(self.buffer_mm / self.px_per_mm)
            if self.clicked_roi[SHAPE_TYPE] == ROI_SETTINGS.CIRCLE.value:
                center, radius = original_tags['Center tag'], self.clicked_roi['radius']
                roi_geometry = Point(center).buffer(distance=radius)
                new_geometry = GeometryMixin().buffer_shape(shape=roi_geometry, size_mm=self.buffer_mm, pixels_per_mm=self.px_per_mm, resolution=1)

                if isinstance(new_geometry, MultiPolygon):
                    new_geometry = unary_union(new_geometry)
                    if isinstance(new_geometry, MultiPolygon):
                        new_geometry = max(new_geometry.geoms, key=lambda p: p.area)

                minx, miny, maxx, maxy = new_geometry.bounds
                center_x, center_y = (minx + maxx) / 2, (miny + maxy) / 2
                new_center = (int(center_x), int(center_y))
                new_radius = int((maxx - minx) / 2)

                self.circle_center, self.circle_radius = new_center, new_radius
                self.width, self.height, self.center = new_radius * 2, new_radius * 2, new_center
                self.top_left = (new_center[0] - new_radius, new_center[1] - new_radius)
                self.bottom_right = (new_center[0] + new_radius, new_center[1] + new_radius)
                self.left_border_tag = (new_center[0] - new_radius, new_center[1])
                
                new_roi = create_circle_entry(circle_selector=self, video_name=self.clicked_roi['Video'], shape_name=self.clicked_roi['Name'], clr_name=self.clicked_roi['Color name'], clr_bgr=self.clicked_roi['Color BGR'], thickness=self.clicked_roi['Thickness'], ear_tag_size=int(self.clicked_roi['Ear_tag_size']), px_conversion_factor=self.px_per_mm)
                
            else:
                center_tag_names = ['Center tag', 'Center_tag']
                corner_tags = {k: v for k, v in original_tags.items() if k not in center_tag_names}
                if len(corner_tags) > 0:
                    tag_coords = np.array(list(corner_tags.values()))
                    centroid = np.mean(tag_coords, axis=0)
                    
                    buffered_tags = {}
                    for tag_name, tag_coord in corner_tags.items():
                        tag_coord_np = np.array(tag_coord)
                        vec_to_tag = tag_coord_np - centroid
                        vec_length = np.linalg.norm(vec_to_tag)
                        
                        if vec_length > 1e-10:
                            vec_normalized = vec_to_tag / vec_length
                            new_coord = tag_coord_np + vec_normalized * buffer_px
                            new_coord[0] = max(0, min(new_coord[0], self.img_w - 1))
                            new_coord[1] = max(0, min(new_coord[1], self.img_h - 1))
                            buffered_tags[tag_name] = tuple(new_coord.astype(int))
                        else:
                            buffered_tags[tag_name] = tag_coord

                    buffered_coords = np.array(list(buffered_tags.values()))
                    new_center = tuple(np.mean(buffered_coords, axis=0).astype(int))
                    if 'Center tag' in original_tags: buffered_tags['Center tag'] = new_center
                    elif 'Center_tag' in original_tags: buffered_tags['Center_tag'] = new_center
                else:
                    buffered_tags = original_tags.copy()

                self.top_left = buffered_tags.get('Top left tag', buffered_tags.get('Tag_0', (0, 0)))
                self.bottom_right = buffered_tags.get('Bottom right tag', buffered_tags.get('Tag_2', (0, 0)))
                self.top_right_tag = buffered_tags.get('Top right tag', buffered_tags.get('Tag_1', (0, 0)))
                self.bottom_left_tag = buffered_tags.get('Bottom left tag', buffered_tags.get('Tag_3', (0, 0)))
                self.center = buffered_tags.get('Center tag', buffered_tags.get('Center_tag', (0, 0)))
                self.left_tag = buffered_tags.get('Left tag', (self.top_left[0], (self.top_left[1] + self.bottom_right[1]) // 2))
                self.right_tag = buffered_tags.get('Right tag', (self.bottom_right[0], (self.top_left[1] + self.bottom_right[1]) // 2))
                self.top_tag = buffered_tags.get('Top tag', ((self.top_left[0] + self.bottom_right[0]) // 2, self.top_left[1]))
                self.bottom_tag = buffered_tags.get('Bottom tag', ((self.top_left[0] + self.bottom_right[0]) // 2, self.bottom_right[1]))
                self.width = abs(self.bottom_right[0] - self.top_left[0])
                self.height = abs(self.bottom_right[1] - self.top_left[1])
                self.circle_radius = int(self.width / 2)
                self.left_border_tag = self.left_tag
                
                new_roi = create_rectangle_entry(rectangle_selector=self, video_name=self.clicked_roi['Video'], shape_name=self.clicked_roi['Name'], clr_name=self.clicked_roi['Color name'], clr_bgr=self.clicked_roi['Color BGR'], thickness=self.clicked_roi['Thickness'], ear_tag_size=int(self.clicked_roi['Ear_tag_size']), px_conversion_factor=self.px_per_mm)
            
            self.roi_dict[self.clicked_roi['Name']] = new_roi
            self.temp_img = _plot_roi(roi_dict=self.roi_dict, img=self.original_img.copy())
            self.__update_image(img=self.temp_img)










