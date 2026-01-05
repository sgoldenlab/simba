from typing import Optional, List, Tuple
from tkinter import Toplevel, Event
import numpy as np
import cv2
from PIL import Image, ImageTk
import pandas as pd
from copy import deepcopy
from scipy.spatial.distance import cdist
import math
from shapely.geometry import Polygon, Point
from simba.utils.checks import check_instance
from simba.utils.enums import ROI_SETTINGS, TkBinds, Keys
from simba.roi_tools.roi_utils import get_image_from_label, create_rectangle_entry, create_circle_entry, create_polygon_entry, get_circle_df_headers, get_rectangle_df_headers, get_polygon_df_headers
from simba.mixins.geometry_mixin import GeometryMixin
from simba.mixins.plotting_mixin import PlottingMixin


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
    img = PlottingMixin.roi_dict_onto_img(img=img, roi_dict=roi_dict, circle_size=None, show_tags=False, show_center=True)
    return img

class InteractiveROIBufferer():

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
        if self.clicked_roi is not None:
            if self.clicked_roi[SHAPE_TYPE] != ROI_SETTINGS.CIRCLE.value:
                roi_data_tags = list(self.clicked_roi[TAGS].values())
                roi_geometry = Polygon(np.array(roi_data_tags).reshape(-1, 2).astype(np.int32))
            else:
                center, radius = self.clicked_roi[TAGS]['Center tag'], self.clicked_roi['radius']
                roi_geometry = Point(center).buffer(distance=radius)
            if self.clicked_roi[SHAPE_TYPE] != ROI_SETTINGS.POLYGON.value:
                new_geometry = GeometryMixin().buffer_shape(shape=roi_geometry, size_mm=self.buffer_mm, pixels_per_mm=self.px_per_mm, resolution=1)
            else:
                new_geometry = GeometryMixin().parallel_offset_polygon(polygon=roi_geometry, size_mm=self.buffer_mm, pixels_per_mm=self.px_per_mm)
            minx, miny, maxx, maxy = new_geometry.bounds
            center_x, center_y = (minx + maxx) / 2, (miny + maxy) / 2
            self.width, self.height, self.center = maxx - minx, maxy - miny, (int(center_x), int(center_y))
            self.top_left, self.bottom_right = (int(minx), int(miny)), (int(maxx), int(maxy))
            self.top_right_tag, self.bottom_left_tag = (int(maxx), int(miny)), (int(minx), int(maxy))
            self.left_tag, self.right_tag = (int(minx), int(center_y)), (int(maxx), int(center_y))
            self.bottom_tag, self.top_tag = (int(center_x), int(maxy)), (int(center_x), int(miny))
            self.circle_radius, self.left_border_tag = int(self.width/2), self.left_tag
            self.polygon_vertices, self.circle_center = np.array(new_geometry.exterior.coords), (int(center_x), int(center_y))
            self.polygon_centroid, self.polygon_area = self.circle_center, new_geometry.area
            self.max_vertice_distance = np.max(cdist(self.polygon_vertices, self.polygon_vertices).astype(np.int32))
            self.polygon_arr = self.polygon_vertices.astype(np.int32)[1:]
            self.tags = {f'Tag_{cnt}': tuple(y) for cnt, y in enumerate(self.polygon_arr)}
            if self.clicked_roi[SHAPE_TYPE] == ROI_SETTINGS.RECTANGLE.value:
                new_roi = create_rectangle_entry(rectangle_selector=self, video_name=self.clicked_roi['Video'], shape_name=self.clicked_roi['Name'], clr_name=self.clicked_roi['Color name'], clr_bgr=self.clicked_roi['Color BGR'], thickness=self.clicked_roi['Thickness'], ear_tag_size=int(self.clicked_roi['Ear_tag_size']), px_conversion_factor=self.px_per_mm)
            elif self.clicked_roi[SHAPE_TYPE] == ROI_SETTINGS.CIRCLE.value:
                new_roi = create_circle_entry(circle_selector=self, video_name=self.clicked_roi['Video'], shape_name=self.clicked_roi['Name'], clr_name=self.clicked_roi['Color name'], clr_bgr=self.clicked_roi['Color BGR'], thickness=self.clicked_roi['Thickness'], ear_tag_size=int(self.clicked_roi['Ear_tag_size']), px_conversion_factor=self.px_per_mm)
            else:
                new_roi = create_polygon_entry(polygon_selector=self, video_name=self.clicked_roi['Video'], shape_name=self.clicked_roi['Name'], clr_name=self.clicked_roi['Color name'], clr_bgr=self.clicked_roi['Color BGR'], thickness=self.clicked_roi['Thickness'], ear_tag_size=int(self.clicked_roi['Ear_tag_size']), px_conversion_factor=self.px_per_mm)
            self.roi_dict[self.clicked_roi['Name']] = new_roi
            self.temp_img = _plot_roi(roi_dict=self.roi_dict, img=self.original_img.copy())
            self.__update_image(img=self.temp_img)










