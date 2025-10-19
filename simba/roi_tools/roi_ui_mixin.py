import ctypes
import math
import os
import platform
from copy import copy, deepcopy
from tkinter import *
from typing import List, Optional, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon

from simba.mixins.config_reader import ConfigReader
from simba.mixins.geometry_mixin import GeometryMixin
from simba.mixins.plotting_mixin import PlottingMixin
from simba.roi_tools.interactive_roi_modifier_tkinter import \
    InteractiveROIModifier
from simba.roi_tools.roi_selector_circle_tkinter import ROISelectorCircle
from simba.roi_tools.roi_selector_polygon_tkinter import ROISelectorPolygon
from simba.roi_tools.roi_selector_rectangle_tkinter import ROISelector
from simba.roi_tools.roi_utils import (change_roi_dict_video_name,
                                       create_circle_entry,
                                       create_duplicated_circle_entry,
                                       create_duplicated_polygon_entry,
                                       create_duplicated_rectangle_entry,
                                       create_polygon_entry,
                                       create_rectangle_entry,
                                       get_circle_df_headers,
                                       get_ear_tags_for_rectangle,
                                       get_half_circle_vertices,
                                       get_polygon_df_headers,
                                       get_pose_for_roi_ui,
                                       get_rectangle_df_headers, get_roi_data,
                                       get_roi_data_for_video_name,
                                       get_roi_df_from_dict,
                                       get_triangle_vertices,
                                       get_vertices_hexagon,
                                       insert_gridlines_on_roi_img)
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        Entry_Box, SimbaButton, SimBADropDown,
                                        SimBALabel, get_menu_icons)
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_int, check_str, check_valid_array)
from simba.utils.enums import OS, ROI_SETTINGS, Formats, Keys
from simba.utils.errors import (FrameRangeError, InvalidInputError,
                                NoROIDataError)
from simba.utils.lookups import (create_color_palettes, get_color_dict,
                                 get_img_resize_info, get_monitor_info)
from simba.utils.printing import stdout_information, stdout_success
from simba.utils.read_write import (get_fn_ext, get_video_meta_data,
                                    read_frm_of_video)
from simba.utils.warnings import DuplicateNamesWarning

WINDOW_SIZE_OPTIONS = [round(x * 0.05, 2) for x in range(21)]

MAX_DRAW_UI_DISPLAY_RATIO = (0.5, 0.75) #(0.5, 0.75)  # W, H - THE INTERFACE IMAGE DISPLAY WILL BE DOWN-SCALED, PRESERVING THE ASPECT RATIO, UNTIL IT MEETS OR EXCEEDS OF THESE CRITERA. E.G (0.5, 0.75) MEANS IMAGES WILL COVER NO MORE THAN HALF THE DISPLAY WIDTH AND 3/4 OF THE DISPLAY HEIGHT.
MIN_DRAW_UI_DISPLAY_RATIO = (0.2, 0.2) #0.30, 0.60 W, H - THE INTERFACE IMAGE DISPLAY WILL BE UP-SCALED, PRESERVING THE ASPECT RATIO, UNTIL IT MEETS AND EXCEEDS BOTH CRITERIA. E.G (0.25, 0.25) MEANS IMAGES WILL COVER NO MORE THAN A QUARTER OF THE USERS DISPLAY HEIGHT AND NO MORE THAN A QUARTER OF THE USERS DISPLAY WIDTH.

DRAW_FRAME_NAME = "DEFINE SHAPE"
SHAPE_TYPE = 'Shape_type'
CIRCLE = 'circle'
POLYGON = 'polygon'
RECTANGLE = 'rectangle'
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
TAGS = 'Tags'
SHOW_TRACKING = 'SHOW_TRACKING'
ROI_TRACKING_STYLE = 'ROI_TRACKING_STYLE'
KEYPOINTS = 'keypoints'
BBOX = 'bbox'
KEYPOINTS_BBOX = 'keypoints & bbox'
SHOW_GRID_OVERLAY = 'SHOW_GRID_OVERLAY'
OVERLAY_GRID_COLOR = 'OVERLAY_GRID_COLOR'
SHOW_HEXAGON_OVERLAY = 'SHOW_HEXAGON_OVERLAY'
POLYGON_TOLERANCE = 'POLYGON_TOLERANCE'

PLATFORM = platform.system()


class ROI_mixin(ConfigReader):

    def __init__(self,
                 video_path: Union[str, os.PathLike],
                 img_idx: int,
                 main_frm: Toplevel,
                 config_path: Optional[Union[str, os.PathLike]] = None,
                 pose_data: Optional[Union[np.ndarray, str, os.PathLike]] = None,
                 animal_cnt: Optional[int] = None,
                 roi_coordinates_path: Optional[Union[str, os.PathLike]] = None):

        self.video_meta = get_video_meta_data(video_path=video_path)
        if config_path is not None:
            ConfigReader.__init__(self, config_path=config_path, read_video_info=False, create_logger=False)
            check_file_exist_and_readable(file_path=config_path)
            _, self.px_per_mm, _ = self.read_video_info(video_name=self.video_meta['video_name'], video_info_df_path=self.video_info_path)
            self.expected_pose_path = os.path.join(self.outlier_corrected_dir, f'{get_fn_ext(video_path)[1]}.{self.file_type}')
            self.pose_path = None if not os.path.isfile(self.expected_pose_path) else self.expected_pose_path

        else:
            self.px_per_mm, self.pose_path, self.expected_pose_path, self.animal_cnt = 1, None, None, animal_cnt
            self.animal_bp_dict = None
            self.min_draw_display_ratio_h, self.min_draw_display_ratio_w = MIN_DRAW_UI_DISPLAY_RATIO[1], MIN_DRAW_UI_DISPLAY_RATIO[0]
            self.max_draw_display_ratio_h, self.max_draw_display_ratio_w = MAX_DRAW_UI_DISPLAY_RATIO[1], MAX_DRAW_UI_DISPLAY_RATIO[0]


        self.monitor_info, (self.display_w, self.display_h) = get_monitor_info()
        stdout_information(msg=f'AVAILABLE MONITOR(S): \n {self.monitor_info}')
        self.display_img_width, self.display_img_height, self.downscale_factor, self.upscale_factor = get_img_resize_info(img_size=(self.video_meta['width'], self.video_meta['height']), display_resolution=(self.display_w, self.display_h), max_height_ratio=self.max_draw_display_ratio_h, max_width_ratio=self.max_draw_display_ratio_w, min_height_ratio=self.min_draw_display_ratio_h, min_width_ratio=self.min_draw_display_ratio_w)

        if roi_coordinates_path is not None:
            self.roi_coordinates_path = roi_coordinates_path
        if pose_data is not None:
            if isinstance(pose_data, (str, os.PathLike)):
                pose_data = get_pose_for_roi_ui(pose_path=pose_data, video_path=video_path)
            check_valid_array(data=pose_data, source=f'{self.__class__.__name__} pose_data', accepted_ndims=(3,), accepted_axis_0_shape=[self.video_meta['frame_count']], accepted_dtypes=Formats.INTEGER_DTYPES.value)
        self.circle_size = PlottingMixin().get_optimal_circle_size(frame_size=(self.display_img_width, self.display_img_height), circle_frame_ratio=100)
        self.clrs = create_color_palettes(no_animals=self.animal_cnt, map_size=int(pose_data.shape[1]/2) + 10) if pose_data is not None else None
        self.img_center = (int(self.display_img_width / 2), int(self.display_img_height / 2))
        self.video_path, self.pose_data = video_path, pose_data
        self.pose_data_cpy = deepcopy(self.pose_data)
        self.img_idx = img_idx
        self.main_frm = main_frm
        self.selected_shape_type, self.grid, self.hexagon_grid = None, None, None
        self.color_option_dict = get_color_dict()
        self.menu_icons = get_menu_icons()

        if PLATFORM == OS.WINDOWS.value:
            self.draw_frm_handle = ctypes.windll.user32.FindWindowW(None, DRAW_FRAME_NAME)
            ctypes.windll.user32.SetWindowPos(self.draw_frm_handle, -1, 0, 0, 0, 0, 3)
        self.img_window = Toplevel()
        self.img_window.geometry(f"{self.display_img_width}x{self.display_img_height}")  # Set the window size
        self.img_window.resizable(False, False)
        self.img_window.title(DRAW_FRAME_NAME)
        self.img_window.iconphoto(False, self.menu_icons['paint']["img"])
        self.img_lbl = Label(self.img_window, name='img_lbl')
        self.img_lbl.pack()
        self.img_window.protocol("WM_DELETE_WINDOW", self.close_img)
        self.settings = {item.name: item.value for item in ROI_SETTINGS}
        self.settings[POLYGON_TOLERANCE] = 2

        self.rectangles_df, self.circles_df, self.polygon_df, self.roi_dict, self.roi_names, self.other_roi_dict, self.other_video_names_w_rois = get_roi_data(roi_path=self.roi_coordinates_path, video_name=self.video_meta['video_name'])

        if self.downscale_factor != 1.0:
            self.roi_dict = self.scale_roi_dict(roi_dict=self.roi_dict, scale_factor=self.downscale_factor)
            self.scaled_other_roi_dict = self.scale_roi_dict(roi_dict=self.other_roi_dict, scale_factor=self.downscale_factor, nesting=True)
            self.rectangles_df, self.circles_df, self.polygon_df = get_roi_df_from_dict(roi_dict=self.roi_dict)
            if self.pose_data is not None:
                self.pose_data = self.pose_data * self.downscale_factor
                self.pose_data_cpy = deepcopy(self.pose_data)
        else:
            self.scaled_other_roi_dict = deepcopy(self.other_roi_dict)


        self.overlay_rois_on_image(show_ear_tags=False, show_roi_info=False)

    def scale_roi_dict(self, roi_dict: dict, scale_factor: float, nesting: bool = False) -> dict:
        new_roi_dict = deepcopy(roi_dict)
        if not nesting:
            for roi_name, roi_data in roi_dict.items():
                if roi_data[SHAPE_TYPE] == ROI_SETTINGS.RECTANGLE.value:
                    new_roi_dict[roi_name][TL_X] = round(roi_data[TL_X] * scale_factor)
                    new_roi_dict[roi_name][TL_Y] = round(roi_data[TL_Y] * scale_factor)
                    new_roi_dict[roi_name][BR_X] = round(roi_data[BR_X] * scale_factor)
                    new_roi_dict[roi_name][BR_Y] = round(roi_data[BR_Y] * scale_factor)
                    new_roi_dict[roi_name][CENTER_X] = round(roi_data[CENTER_X] * scale_factor)
                    new_roi_dict[roi_name][CENTER_Y] = round(roi_data[CENTER_Y] * scale_factor)
                    new_roi_dict[roi_name][WIDTH] = round(new_roi_dict[roi_name][BR_X] - new_roi_dict[roi_name][TL_X])
                    new_roi_dict[roi_name][HEIGHT] = round(new_roi_dict[roi_name][BR_Y] - new_roi_dict[roi_name][TL_Y])
                    for tag in roi_data[TAGS].keys():
                        new_roi_dict[roi_name][TAGS][tag] = (round(roi_data[TAGS][tag][0] * scale_factor), round(roi_data[TAGS][tag][1] * scale_factor))
                elif roi_data[SHAPE_TYPE] == ROI_SETTINGS.CIRCLE.value:
                    new_roi_dict[roi_name][CIRCLE_C_X] = round(roi_data[CIRCLE_C_X] * scale_factor)
                    new_roi_dict[roi_name][CIRCLE_C_Y] = round(roi_data[CIRCLE_C_Y] * scale_factor)
                    new_roi_dict[roi_name][RADIUS] = round(roi_data[RADIUS] * scale_factor)
                    for tag in roi_data[TAGS].keys():
                        new_roi_dict[roi_name][TAGS][tag] = (round(roi_data[TAGS][tag][0] * scale_factor), round(roi_data[TAGS][tag][1] * scale_factor))
                elif roi_data[SHAPE_TYPE] == ROI_SETTINGS.POLYGON.value:
                    new_roi_dict[roi_name][CENTER_X] = round(roi_data[CENTER_X] * scale_factor)
                    new_roi_dict[roi_name][CENTER_Y] = round(roi_data[CENTER_Y] * scale_factor)
                    new_roi_dict[roi_name][CENTER] = (new_roi_dict[roi_name][CENTER_X], new_roi_dict[roi_name][CENTER_Y])
                    for tag in roi_data[TAGS].keys():
                        new_roi_dict[roi_name][TAGS][tag] = (round(roi_data[TAGS][tag][0] * scale_factor), round(roi_data[TAGS][tag][1] * scale_factor))
                    new_vertices = np.full_like(a=roi_data[VERTICES], fill_value=0, dtype=np.int32)
                    for vertice_idx in range(roi_data[VERTICES].shape[0]):
                        #new_vertices[vertice_idx][0], new_vertices[vertice_idx][1] = roi_data[VERTICES][vertice_idx][0] * scale_factor, roi_data[VERTICES][vertice_idx][1] * scale_factor
                        new_vertices[vertice_idx][0] = round(roi_data[VERTICES][vertice_idx][0] * scale_factor)
                        new_vertices[vertice_idx][1] = round(roi_data[VERTICES][vertice_idx][1] * scale_factor)
                    new_roi_dict[roi_name][VERTICES] = new_vertices
        else:
            for video_name, video_data in roi_dict.items():
                for roi_name, roi_data in video_data.items():
                    if roi_data[SHAPE_TYPE] == ROI_SETTINGS.RECTANGLE.value:
                        new_roi_dict[video_name][roi_name][TL_X] = round(roi_data[TL_X] * scale_factor)
                        new_roi_dict[video_name][roi_name][TL_Y] = round(roi_data[TL_Y] * scale_factor)
                        new_roi_dict[video_name][roi_name][BR_X] = round(roi_data[BR_X] * scale_factor)
                        new_roi_dict[video_name][roi_name][BR_Y] = round(roi_data[BR_Y] * scale_factor)
                        new_roi_dict[video_name][roi_name][CENTER_X] = round(roi_data[CENTER_X] * scale_factor)
                        new_roi_dict[video_name][roi_name][CENTER_Y] = round(roi_data[CENTER_Y] * scale_factor)
                        new_roi_dict[video_name][roi_name][WIDTH] = round(new_roi_dict[video_name][roi_name][BR_X] - new_roi_dict[video_name][roi_name][TL_X])
                        new_roi_dict[video_name][roi_name][HEIGHT] = round(new_roi_dict[video_name][roi_name][BR_Y] - new_roi_dict[video_name][roi_name][TL_Y])
                        for tag in roi_data[TAGS].keys():
                            new_roi_dict[video_name][roi_name][TAGS][tag] = (round(roi_data[TAGS][tag][0] * scale_factor), round(roi_data[TAGS][tag][1] * scale_factor))
                    elif roi_data[SHAPE_TYPE] == ROI_SETTINGS.CIRCLE.value:
                        new_roi_dict[video_name][roi_name][CIRCLE_C_X] = round(roi_data[CIRCLE_C_X] * scale_factor)
                        new_roi_dict[video_name][roi_name][CIRCLE_C_Y] = round(roi_data[CIRCLE_C_Y] * scale_factor)
                        new_roi_dict[video_name][roi_name][RADIUS] = round(roi_data[RADIUS] * scale_factor)
                        for tag in roi_data[TAGS].keys():
                            new_roi_dict[video_name][roi_name][TAGS][tag] = (round(roi_data[TAGS][tag][0] * scale_factor), round(roi_data[TAGS][tag][1] * scale_factor))
                    elif roi_data[SHAPE_TYPE] == ROI_SETTINGS.POLYGON.value:
                        new_roi_dict[video_name][roi_name][CENTER_X] = round(roi_data[CENTER_X] * scale_factor)
                        new_roi_dict[video_name][roi_name][CENTER_Y] = round(roi_data[CENTER_Y] * scale_factor)
                        new_roi_dict[video_name][roi_name][CENTER] = (new_roi_dict[video_name][roi_name][CENTER_X], new_roi_dict[video_name][roi_name][CENTER_Y])
                        for tag in roi_data[TAGS].keys():
                            new_roi_dict[video_name][roi_name][TAGS][tag] = (round(roi_data[TAGS][tag][0] * scale_factor), round(roi_data[TAGS][tag][1] * scale_factor))
                        new_vertices = np.full_like(a=roi_data[VERTICES], fill_value=0, dtype=np.int32)
                        for vertice_idx in range(roi_data[VERTICES].shape[0]):
                            new_vertices[vertice_idx][0], new_vertices[vertice_idx][1] = roi_data[VERTICES][vertice_idx][0] * scale_factor, roi_data[VERTICES][vertice_idx][1] * scale_factor
                        new_roi_dict[video_name][roi_name][VERTICES] = new_vertices




        return deepcopy(new_roi_dict)


    def read_img(self, frame_idx: int):
        return read_frm_of_video(video_path=self.video_path, frame_index=frame_idx, size=(self.display_img_width, self.display_img_height))

    def set_img(self, frame_idx: int):
        self.img = read_frm_of_video(video_path=self.video_path, frame_index=frame_idx, size=(self.display_img_width, self.display_img_height))

    def get_frm_pose(self, frame_idx: int):
        if (self.pose_data is not None) and (0 <= frame_idx < self.pose_data.shape[0]):
            return self.pose_data[frame_idx]
        else:
            return None

    def insert_pose(self, img: np.ndarray, pose_img_data: np.ndarray):
        if pose_img_data is None:
            return img
        if self.settings[ROI_TRACKING_STYLE].lower() in [KEYPOINTS, KEYPOINTS_BBOX]:
            for animal_cnt, (animal_name, animal_bp_data) in enumerate(self.animal_bp_dict.items()):
                bp_cnt = len(animal_bp_data['X_bps'])
            for cnt, i in enumerate(range(0, pose_img_data.shape[0], bp_cnt)):
                animal_kp_data = pose_img_data[i:i+bp_cnt]
                for pos_idx in range(animal_kp_data.shape[0]):
                    img = cv2.circle(img, (int(animal_kp_data[pos_idx][0]), int(animal_kp_data[pos_idx][1])), self.circle_size, self.clrs[cnt][pos_idx], -1, lineType=-1)
        if self.settings[ROI_TRACKING_STYLE].lower() in [BBOX, KEYPOINTS_BBOX]:
            try:
                for animal_cnt, (animal_name, animal_bp_data) in enumerate(self.animal_bp_dict.items()):
                    bp_cnt = len(animal_bp_data['X_bps'])
                for cnt, i in enumerate(range(0, pose_img_data.shape[0], bp_cnt)):
                    animal_kp_data = pose_img_data[i:i + bp_cnt]
                    bbox = GeometryMixin().keypoints_to_axis_aligned_bounding_box(keypoints=animal_kp_data.reshape(-1, len(animal_kp_data), 2).astype(np.int32))
                    img = cv2.polylines(img, [bbox], True, self.clrs[cnt][0], thickness=self.circle_size, lineType=-1)
            except Exception as e:
                msg = f'Cannot show bounding box tracking style: {e.args}.'
                self.set_status_bar_panel(text=msg, fg='red')
        return img


    # def insert_grid(self, img: np.ndarray, grid: List[Polygon]) -> np.ndarray:
    #     if grid is None or len(grid) == 0:
    #         return img
    #     else:
    #         try:
    #             for polygon in grid:
    #                 cords = np.array(polygon.exterior.coords).astype(np.int32)
    #                 img = cv2.polylines(img=img, pts=[cords], isClosed=True, color=self.settings[OVERLAY_GRID_COLOR], thickness=max(1, int(self.circle_size/5)), lineType=8)
    #             return img
    #         except Exception as e:
    #             msg = f'Cannot draw gridlines: {e.args}'
    #             self.set_status_bar_panel(text=msg, fg='red')
    #             raise InvalidInputError(msg=msg, source=f'{self.__class__.__name__} draw')

    def overlay_rois_on_image(self,
                              show_ear_tags: bool = False,
                              show_roi_info: bool = False):

        self.set_img(frame_idx=self.img_idx)
        self.img = PlottingMixin.rectangles_onto_image(img=self.img, rectangles=self.rectangles_df, show_tags=show_ear_tags, circle_size=None, print_metrics=show_roi_info, line_type=self.settings['LINE_TYPE'])
        self.img = PlottingMixin.circles_onto_image(img=self.img, circles=self.circles_df, show_tags=show_ear_tags, circle_size=None, print_metrics=show_roi_info, line_type=self.settings['LINE_TYPE'])
        self.img = PlottingMixin.polygons_onto_image(img=self.img, polygons=self.polygon_df, show_tags=show_ear_tags, circle_size=None, print_metrics=show_roi_info, line_type=self.settings['LINE_TYPE'])
        if self.pose_data is not None:
            self.img = self.insert_pose(img=self.img, pose_img_data=self.get_frm_pose(frame_idx=self.img_idx))
        if self.grid is not None:
            self.img = insert_gridlines_on_roi_img(img=self.img, grid=self.grid, color=self.settings[OVERLAY_GRID_COLOR], thickness=max(1, int(self.circle_size/5)))
        if self.hexagon_grid is not None:
            self.img = insert_gridlines_on_roi_img(img=self.img, grid=self.hexagon_grid, color=self.settings[OVERLAY_GRID_COLOR], thickness=max(1, int(self.circle_size/5)))
        self.draw_img()

    def draw_img(self):
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.pil_image = Image.fromarray(img_rgb)
        self.tk_image = ImageTk.PhotoImage(self.pil_image)
        self.img_lbl.configure(image=self.tk_image)
        self.img_lbl.image = self.tk_image


    def get_video_info_panel(self,
                             parent_frame: Union[Frame, Canvas, LabelFrame, Toplevel],
                             row_idx: int):

        self.change_attr_panel = CreateLabelFrameWithIcon(parent=parent_frame, header="VIDEO AND FRAME INFORMATION", font=Formats.FONT_HEADER.value, padx=5, pady=5, icon_name='info', relief='solid')
        self.video_name_lbl = SimBALabel(parent=self.change_attr_panel, txt=f'VIDEO NAME: {self.video_meta["video_name"]}')
        self.video_size_lbl = SimBALabel(parent=self.change_attr_panel, txt=f'SIZE (PX): {self.video_meta["resolution_str"]}')
        self.video_fps_lbl = SimBALabel(parent=self.change_attr_panel, txt=f'FPS: {self.video_meta["fps"]}')
        self.video_frame_id_lbl = SimBALabel(parent=self.change_attr_panel, txt=f'DISPLAY FRAME #: {self.img_idx}')
        self.video_frame_time_lbl = SimBALabel(parent=self.change_attr_panel, txt=f'DISPLAY FRAME (S): {(round((self.img_idx / self.video_meta["fps"]), 2))}')
        self.display_size = SimBALabel(parent=self.change_attr_panel, txt=f'PRIMARY MONITOR (WxH): {self.display_w, self.display_h}')
        self.change_attr_panel.grid(row=row_idx, sticky=W)
        self.video_name_lbl.grid(row=0, column=0, sticky=NW)
        self.video_fps_lbl.grid(row=0, column=1, sticky=NW)
        self.video_size_lbl.grid(row=0, column=2, sticky=NW)
        self.video_frame_id_lbl.grid(row=0, column=3, sticky=NW)
        self.video_frame_time_lbl.grid(row=0, column=4, sticky=NW)
        self.display_size.grid(row=1, column=0, sticky=NW)

    def get_file_menu(self,
                      root: Toplevel):

        menu = Menu(root)
        file_menu = Menu(menu)
        menu.add_cascade(label="File (ROI)", menu=file_menu)
        file_menu.add_command(label="Preferences...", compound="left", image=self.menu_icons["settings"]["img"], command=lambda: self.preferences_pop_up())
        file_menu.add_command(label="Draw ROIs of pre-defined sizes...", compound="left", image=self.menu_icons["size_black"]["img"], command=lambda: self.fixed_roi_pop_up())
        root.config(menu=menu)

    def get_select_img_panel(self,
                             parent_frame: Union[Frame, Canvas, LabelFrame, Toplevel],
                             row_idx: int):

        self.select_img_panel = CreateLabelFrameWithIcon(parent=parent_frame, header="CHANGE IMAGE", font=Formats.FONT_HEADER.value, padx=5, pady=5, icon_name='frames', relief='solid')
        self.forward_1s_btn = SimbaButton(parent=self.select_img_panel, txt="+1s", img='plus_green_2', font=Formats.FONT_REGULAR.value, txt_clr='darkgreen', cmd=self.change_img, cmd_kwargs={'stride': int(self.video_meta['fps'])})
        self.backwards_1s_btn = SimbaButton(parent=self.select_img_panel, txt="-1s", img='minus_blue_2', font=Formats.FONT_REGULAR.value, txt_clr='darkblue', cmd=self.change_img, cmd_kwargs={'stride': -int(self.video_meta['fps'])})
        self.custom_seconds_entry = Entry_Box(parent=self.select_img_panel, fileDescription='CUSTOM SECONDS:', labelwidth=18, validation='numeric', entry_box_width=4, value=5)
        self.custom_fwd_btn = SimbaButton(parent=self.select_img_panel, txt="FORWARD", img='fastforward_green_2', font=Formats.FONT_REGULAR.value, txt_clr='darkgreen', cmd=self.change_img, cmd_kwargs={'stride': 'custom_forward'})
        self.custom_backwards_btn = SimbaButton(parent=self.select_img_panel, txt="REVERSE", img='rewind_blue_2', font=Formats.FONT_REGULAR.value, txt_clr='darkblue', cmd=self.change_img,cmd_kwargs={'stride': 'custom_backward'})
        self.first_frm_btn = SimbaButton(parent=self.select_img_panel, txt="FIRST FRAME", img='first_frame_blue', font=Formats.FONT_REGULAR.value, txt_clr='darkblue', cmd=self.change_img, cmd_kwargs={'stride': 'first'})
        self.last_frm_btn = SimbaButton(parent=self.select_img_panel, txt="LAST FRAME", img='last_frame_blue', font=Formats.FONT_REGULAR.value, txt_clr='darkblue', cmd=self.change_img, cmd_kwargs={'stride': 'last'})
        self.select_img_panel.grid(row=row_idx, sticky=NW)
        self.forward_1s_btn.grid(row=0, column=0, sticky=NW, pady=10)
        self.backwards_1s_btn.grid(row=0, column=1, sticky=NW, pady=10, padx=10)
        self.custom_seconds_entry.grid(row=0, column=2, sticky=NW, pady=10)
        self.custom_fwd_btn.grid(row=0, column=3, sticky=NW, pady=10)
        self.custom_backwards_btn.grid(row=0, column=4, sticky=NW, pady=10)
        self.first_frm_btn.grid(row=0, column=5, sticky=NW, pady=10)
        self.last_frm_btn.grid(row=0, column=6, sticky=NW, pady=10)

    def get_select_shape_type_panel(self,
                                    parent_frame: Union[Frame, Canvas, LabelFrame, Toplevel],
                                    row_idx: int):

        self.select_shape_type_panel = CreateLabelFrameWithIcon(parent=parent_frame, header="SET NEW SHAPE", font=Formats.FONT_HEADER.value, padx=5, pady=5, icon_name='shapes_large', relief='solid')
        self.rectangle_button = SimbaButton(parent=self.select_shape_type_panel, txt='RECTANGLE', txt_clr='black', font=Formats.FONT_REGULAR.value, img='rectangle_1_large', cmd=self.set_selected_shape_type, cmd_kwargs={'shape_type': lambda: RECTANGLE})
        self.circle_button = SimbaButton(parent=self.select_shape_type_panel, txt='CIRCLE', txt_clr='black', font=Formats.FONT_REGULAR.value, img='circle_large', cmd=self.set_selected_shape_type, cmd_kwargs={'shape_type': lambda: CIRCLE})
        self.polygon_button = SimbaButton(parent=self.select_shape_type_panel, txt='POLYGON', txt_clr='black', font=Formats.FONT_REGULAR.value, img='polygon_large', cmd=self.set_selected_shape_type, cmd_kwargs={'shape_type': lambda: POLYGON})
        self.select_shape_type_panel.grid(row=row_idx, sticky=NW)
        self.rectangle_button.grid(row=0, sticky=NW, pady=10, padx=(0, 10))
        self.circle_button.grid(row=0, column=1, sticky=NW, pady=10, padx=(0, 10))
        self.polygon_button.grid(row=0, column=2, sticky=NW, pady=10, padx=(0, 10))

    def get_shape_attr_panel(self,
                             parent_frame: Union[Frame, Canvas, LabelFrame, Toplevel],
                             row_idx: int):

        self.shape_attr_panel = CreateLabelFrameWithIcon(parent=parent_frame, header="SHAPE ATTRIBUTES", font=Formats.FONT_HEADER.value, padx=5, pady=5, icon_name='attributes_large', relief='solid')
        self.thickness_dropdown = SimBADropDown(parent=self.shape_attr_panel, dropdown_options=ROI_SETTINGS.SHAPE_THICKNESS_OPTIONS.value, label="SHAPE THICKNESS: ", label_width=17, value=5, dropdown_width=5)
        self.color_dropdown = SimBADropDown(parent=self.shape_attr_panel, dropdown_options=list(self.color_option_dict.keys()), label="SHAPE COLOR: ", label_width=17, value='Red', dropdown_width=20)
        self.ear_tag_size_dropdown = SimBADropDown(parent=self.shape_attr_panel, dropdown_options=ROI_SETTINGS.EAR_TAG_SIZE_OPTIONS.value, label="EAR TAG SIZE: ", label_width=17, value=15, dropdown_width=5)


        self.shape_attr_panel.grid(row=row_idx, sticky=W, pady=10)
        self.thickness_dropdown.grid(row=0, column=1, sticky=W, pady=10, padx=(0, 10))
        self.ear_tag_size_dropdown.grid(row=0, column=2, sticky=W, pady=10, padx=(0, 10))
        self.color_dropdown.grid(row=0, column=3, sticky=W, pady=10, padx=(0, 10))

    def get_shape_name_panel(self,
                             parent_frame: Union[Frame, Canvas, LabelFrame, Toplevel],
                             row_idx: int):

        self.shape_name_panel = CreateLabelFrameWithIcon(parent=parent_frame, header="SHAPE NAME", font=Formats.FONT_HEADER.value, padx=5, pady=5, icon_name='label_large', relief='solid')
        self.shape_name_eb = Entry_Box(parent=self.shape_name_panel, fileDescription="SHAPE NAME: ", labelwidth=15, entry_box_width=55)
        self.shape_name_panel.grid(row=row_idx, sticky=W, pady=10)
        self.shape_name_eb.grid(row=0, column=0, sticky=W, pady=10)

    def get_save_roi_panel(self,
                           parent_frame: Union[Frame, Canvas, LabelFrame, Toplevel],
                           row_idx: int):

        self.save_roi_panel = CreateLabelFrameWithIcon(parent=parent_frame, header="SAVE ROI DRAWINGS", font=Formats.FONT_HEADER.value, padx=5, pady=5, icon_name='save_large', relief='solid')
        self.save_data_btn = SimbaButton(parent=self.save_roi_panel, txt="SAVE VIDEO ROI DATA", img='save_large', txt_clr='black', cmd=self.save_video_rois)
        self.save_roi_panel.grid(row=row_idx, sticky=W, pady=10)
        self.save_data_btn.grid(row=0, column=0, sticky=W, pady=10, padx=(0, 10))


    def get_interact_panel(self,
                           parent_frame: Union[Frame, Canvas, LabelFrame, Toplevel],
                           row_idx: int,
                           top_level: Toplevel):

        self.interact_panel = CreateLabelFrameWithIcon(parent=parent_frame, header="SHAPE INTERACTION", font=Formats.FONT_HEADER.value, padx=5, pady=5, icon_name='interaction_large', relief='solid')
        self.move_shape_btn = SimbaButton(parent=self.interact_panel, txt="MOVE SHAPE", img='move_large', txt_clr='black', cmd=self.move_shapes, cmd_kwargs={'parent_frame': lambda : top_level})
        self.shape_info_btn = SimbaButton(parent=self.interact_panel, txt="SHOW SHAPE INFO", img='info_large', txt_clr='black', enabled=True, cmd=self.show_shape_info)
        self.interact_panel.grid(row=row_idx, sticky=W, pady=10)
        self.move_shape_btn.grid(row=0, column=0, sticky=W, pady=10, padx=(0, 10))
        self.shape_info_btn.grid(row=0, column=1, sticky=W, pady=10, padx=(0, 10))

    def get_status_bar_panel(self,
                             parent_frame: Union[Frame, Canvas, LabelFrame, Toplevel],
                             row_idx: int):

        self.status_bar = SimBALabel(parent=parent_frame, txt='', txt_clr='black', bg_clr=None, font=Formats.FONT_REGULAR.value, relief='sunken')
        self.status_bar.grid(row=row_idx, column=0, sticky='we')
        parent_frame.grid_rowconfigure(row_idx, weight=0)

    def get_draw_panel(self,
                       parent_frame: Union[Frame, Canvas, LabelFrame, Toplevel],
                       row_idx: int,
                       top_level: Union[Frame, Canvas, LabelFrame, Toplevel]):

        self.draw_panel = CreateLabelFrameWithIcon(parent=parent_frame, header="DRAW", font=Formats.FONT_HEADER.value, padx=5, pady=5, icon_name='palette_large', relief='solid')
        self.draw_btn = SimbaButton(parent=self.draw_panel, txt='DRAW', img='brush_large', txt_clr='black', cmd=self.draw, cmd_kwargs={'parent_frame': top_level})
        self.delete_all_btn = SimbaButton(parent=self.draw_panel, txt='DELETE ALL', img='delete_large_red', txt_clr='black', cmd=self.delete_all)

        self.roi_dropdown = SimBADropDown(parent=self.draw_panel, dropdown_options=self.roi_names, label="ROI: ", label_width=5, value=self.roi_names[0], dropdown_width=max(5, max(len(s) for s in self.roi_names)))
        if self.roi_names == ['']: self.roi_dropdown.disable()



        #self.roi_dropdown = DropDownMenu(self.draw_panel, "ROI: ", self.roi_names, 5)
        #self.roi_dropdown.setChoices(self.roi_names[0])
        self.delete_selected_btn = SimbaButton(parent=self.draw_panel, txt='DELETE SELECTED ROI', img='delete_large_orange', txt_clr='black', cmd=self.delete_named_shape, cmd_kwargs={'name': lambda: self.roi_dropdown.getChoices()})
        self.duplicate_selected_btn = SimbaButton(parent=self.draw_panel, txt='DUPLICATE SELECTED ROI', img='duplicate_large',  txt_clr='black', cmd=self.duplicate_selected)
        self.chg_attr_btn = SimbaButton(parent=self.draw_panel, txt='CHANGE ROI', img='edit_roi_large', txt_clr='black', cmd=self.change_attr_pop_up)
        self.draw_panel.grid(row=row_idx, sticky=W)
        self.draw_btn.grid(row=0, column=1, sticky=W, pady=2, padx=(0, 10))
        self.delete_all_btn.grid(row=0, column=2, sticky=W, pady=2, padx=(0, 10))
        self.roi_dropdown.grid(row=0, column=3, sticky=W, pady=2, padx=(0, 10))
        self.delete_selected_btn.grid(row=0, column=4, sticky=W, pady=2, padx=(0, 10))
        self.duplicate_selected_btn.grid(row=0, column=5, sticky=W, pady=2, padx=(0, 10))
        self.chg_attr_btn.grid(row=0, column=6, sticky=W, pady=2, padx=(0, 10))


    def get_shapes_from_other_video_panel(self,
                                          parent_frame: Union[Frame, Canvas, LabelFrame, Toplevel],
                                          row_idx: int):

        self.shapes_from_other_video_panel = CreateLabelFrameWithIcon(parent=parent_frame, header="APPLY SHAPES FROM DIFFERENT VIDEO", font=Formats.FONT_HEADER.value, padx=5, pady=5, icon_name='duplicate_2_large', relief='solid')
        dropdown_width = max(len(s) for s in self.other_video_names_w_rois)
        self.other_videos_dropdown = SimBADropDown(parent=self.shapes_from_other_video_panel, dropdown_options=self.other_video_names_w_rois, label="FROM VIDEO: ", label_width=15, dropdown_width=dropdown_width, value=self.other_video_names_w_rois[0])
        self.apply_other_video_btn = SimbaButton(parent=self.shapes_from_other_video_panel, txt="APPLY", img='tick_large', txt_clr='black', enabled=True, cmd=self.apply_different_video, cmd_kwargs={'video_name': lambda: self.other_videos_dropdown.getChoices()})
        if self.other_video_names_w_rois == ['']:
            self.other_videos_dropdown.disable()
            self.apply_other_video_btn.config(state=DISABLED)
        self.shapes_from_other_video_panel.grid(row=row_idx, sticky=W, pady=10)
        self.other_videos_dropdown.grid(row=0, column=0, sticky=W, pady=10, padx=(0, 10))
        self.apply_other_video_btn.grid(row=0, column=1, sticky=W, pady=10, padx=(0, 10))

    def draw(self, parent_frame):
        def on_click(event):
            self.click_event.set(True)
            self.got_attributes = self.selector.get_attributes()
            self.root.unbind("<Button-1>"); self.root.unbind("<Escape>"); self.img_window.unbind("<Escape>");
            self.set_btn_clrs()

        self.shape_info_btn.configure(text="SHOW SHAPE INFO")
        shape_name = self.shape_name_eb.entry_get.strip()
        self.click_event, self.got_attributes = BooleanVar(value=False), False
        self.root = parent_frame

        if not check_str(name=f'shape name', value=shape_name, allow_blank=False, raise_error=False)[0]:
            msg = f"Invalid shape name: {shape_name}. Type a shape name before drawing."
            self.set_status_bar_panel(text=msg, fg='red')
            raise InvalidInputError(msg=msg, source=f'{self.__class__.__name__} draw')
        elif shape_name in self.roi_names:
            msg = f'Cannot draw ROI named {shape_name}. An ROI named {shape_name} already exist in for the video.'
            self.set_status_bar_panel(text=msg, fg='red')
            raise InvalidInputError(msg=msg, source=f'{self.__class__.__name__} draw')
        else:
            msg = f'Draw {self.selected_shape_type} ROI {shape_name}.'
            self.set_status_bar_panel(text=msg, fg='blue')
        self.set_btn_clrs(btn=self.draw_btn)
        if PLATFORM == OS.WINDOWS.value:
            ctypes.windll.user32.SetWindowPos(self.draw_frm_handle, -1, 0, 0, 0, 0, 3)
        if self.selected_shape_type == RECTANGLE:
            self.selector = ROISelector(img_window=self.img_window, thickness=int(self.thickness_dropdown.getChoices()), clr=self.color_option_dict[self.color_dropdown.getChoices()])
        elif self.selected_shape_type == CIRCLE:
            self.selector = ROISelectorCircle(img_window=self.img_window, thickness=int(self.thickness_dropdown.getChoices()), clr=self.color_option_dict[self.color_dropdown.getChoices()])
        elif self.selected_shape_type == POLYGON:
            self.selector = ROISelectorPolygon(img_window=self.img_window, thickness=int(self.thickness_dropdown.getChoices()), clr=self.color_option_dict[self.color_dropdown.getChoices()], vertice_size=int(self.ear_tag_size_dropdown.getChoices()), tolerance=int(self.settings[POLYGON_TOLERANCE]))
        self.root.bind("<Button-1>", on_click); self.root.bind("<Escape>", on_click); self.img_window.bind("<Escape>", on_click)
        self.root.wait_variable(self.click_event)
        if self.got_attributes:
            if self.selected_shape_type == RECTANGLE:
                shape_entry = create_rectangle_entry(rectangle_selector=self.selector, video_name=self.video_meta['video_name'], shape_name=shape_name, clr_name=self.color_dropdown.getChoices(), clr_bgr=self.color_option_dict[self.color_dropdown.getChoices()], thickness=int(self.thickness_dropdown.getChoices()), ear_tag_size=int(self.ear_tag_size_dropdown.getChoices()), px_conversion_factor=self.px_per_mm)
            elif self.selected_shape_type == CIRCLE:
                shape_entry = create_circle_entry(circle_selector=self.selector, video_name=self.video_meta['video_name'], shape_name=shape_name, clr_name=self.color_dropdown.getChoices(), clr_bgr=self.color_option_dict[self.color_dropdown.getChoices()], thickness=int(self.thickness_dropdown.getChoices()), ear_tag_size=int(self.ear_tag_size_dropdown.getChoices()), px_conversion_factor=self.px_per_mm)
            else:
                shape_entry = create_polygon_entry(polygon_selector=self.selector, video_name=self.video_meta['video_name'], shape_name=shape_name, clr_name=self.color_dropdown.getChoices(), clr_bgr=self.color_option_dict[self.color_dropdown.getChoices()], thickness=int(self.thickness_dropdown.getChoices()), ear_tag_size=int(self.ear_tag_size_dropdown.getChoices()), px_conversion_factor=self.px_per_mm)
            self.add_roi(shape_entry=shape_entry)
            self.update_dropdown_menu(dropdown=self.roi_dropdown, new_options=self.roi_names, set_index=0)
            self.img_window = self.selector.img_window
            self.set_status_bar_panel(text=F'ROI ADDED (NAME: {shape_entry["Name"]}, TYPE {self.selected_shape_type.upper()})', fg='blue')

        del self.selector; del self.root
        self.overlay_rois_on_image()

    def set_btn_clrs(self, btn: Optional[SimbaButton] = None):
        if btn is not None:
            btn.configure(fg=ROI_SETTINGS.SELECT_COLOR.value)
        for other_btns in [self.delete_all_btn, self.draw_btn, self.chg_attr_btn, self.delete_selected_btn, self.duplicate_selected_btn, self.move_shape_btn, self.shape_info_btn, self.save_data_btn, self.apply_other_video_btn]:
            if btn != other_btns:
                other_btns.configure(fg=ROI_SETTINGS.UNSELECT_COLOR.value)


    def move_shapes(self, parent_frame):
        if len(list(self.roi_dict.keys())) == 0:
            msg = f'Cannot move ROIs: No ROIs have been drawn on video {self.video_meta["video_name"]}.'
            self.set_status_bar_panel(text=msg, fg="red")
            raise NoROIDataError(msg, source=self.__class__.__name__)

        def on_click(event):
            self.click_event.set(True)
            self.root.unbind("<Button-1>"); self.root.unbind("<Escape>"); self.img_window.unbind("<Escape>");
            self.interactive_modifier.unbind_keys()
            self.set_btn_clrs()
            self.set_status_bar_panel(text="ROI MOVE MODE EXITED", fg="blue")

        self.set_btn_clrs(btn=self.move_shape_btn)

        self.overlay_rois_on_image(show_ear_tags=True, show_roi_info=False)
        self.set_status_bar_panel(text="IN ROI MOVE MODE. MODIFY ROI'S BY DRAGGING EAR TAGS. CLICK ESC OCH CLICK SETTINGS WINDOW TO EXIT MOVE MODE", fg="darkred")
        self.click_event, self.got_attributes = BooleanVar(value=False), False
        self.root = parent_frame
        self.interactive_modifier = InteractiveROIModifier(img_window=self.img_window, original_img=self.read_img(frame_idx=self.img_idx), roi_dict=deepcopy(self.roi_dict), settings=self.settings, rectangle_grid=self.grid, hex_grid=self.hexagon_grid)
        self.root.bind("<Button-1>", on_click); self.root.bind("<Escape>", on_click); self.img_window.bind("<Escape>", on_click)
        self.root.wait_variable(self.click_event)

        self.img_window = self.interactive_modifier.img_window
        self.roi_dict = self.interactive_modifier.roi_dict
        self.rectangles_df, self.circles_df, self.polygon_df = get_roi_df_from_dict(roi_dict=self.roi_dict)
        del self.interactive_modifier; del self.root
        self.overlay_rois_on_image(show_ear_tags=False, show_roi_info=False)


    def change_img(self, stride: Union[int, str]):
        custom_s, new_frm_idx = self.custom_seconds_entry.entry_get.strip(), None
        if isinstance(stride, int):
            new_frm_idx = self.img_idx + stride
        elif stride == 'custom_forward':
            check_int(name='CUSTOM SECONDS', value=custom_s, min_value=1)
            custom_s = int(custom_s)
            new_frm_idx = int(self.img_idx + (custom_s * self.video_meta['fps']))
        elif stride == 'custom_backward':
            check_int(name='CUSTOM SECONDS', value=custom_s, min_value=1)
            custom_s = int(custom_s)
            new_frm_idx = int(self.img_idx - (custom_s * self.video_meta['fps']))
        elif stride == 'first':
            new_frm_idx = 0
        elif stride == 'last':
            new_frm_idx = self.video_meta['frame_count']-1
        if (0 > new_frm_idx) or (new_frm_idx > self.video_meta['frame_count']-1):
            msg = f'Cannot change frame. The new frame index {new_frm_idx} is outside the video {self.video_meta["video_name"]} frame range (video has {self.video_meta["frame_count"]} frames).'
            self.set_status_bar_panel(text=msg, fg='red')
            raise FrameRangeError(msg=msg, source=self.__class__.__name__)
        else:
            self.set_img(frame_idx=new_frm_idx)
            self.img_idx = copy(new_frm_idx)
            self.overlay_rois_on_image(show_roi_info=False, show_ear_tags=False)
            frm_time = round((self.img_idx / self.video_meta["fps"]), 2)
            self.video_frame_id_lbl.configure(text=f'DISPLAY FRAME #: {self.img_idx}')
            self.video_frame_time_lbl.configure(text=f'DISPLAY FRAME (S): {frm_time}')
            self.set_status_bar_panel(text=f'Set frame to frame number {new_frm_idx} ({frm_time}s)', fg='blue')

    def set_selected_shape_type(self, shape_type: Optional[str] = None):
        if shape_type == None:
            self.rectangle_button.configure(fg=ROI_SETTINGS.UNSELECT_COLOR.value)
            self.circle_button.configure(fg=ROI_SETTINGS.UNSELECT_COLOR.value)
            self.polygon_button.configure(fg=ROI_SETTINGS.UNSELECT_COLOR.value)
        elif shape_type != self.selected_shape_type:
            if shape_type == RECTANGLE:
                self.rectangle_button.configure(fg=ROI_SETTINGS.SELECT_COLOR.value)
                self.circle_button.configure(fg=ROI_SETTINGS.UNSELECT_COLOR.value)
                self.polygon_button.configure(fg=ROI_SETTINGS.UNSELECT_COLOR.value)
            elif shape_type == CIRCLE:
                self.rectangle_button.configure(fg=ROI_SETTINGS.UNSELECT_COLOR.value)
                self.circle_button.configure(fg=ROI_SETTINGS.SELECT_COLOR.value)
                self.polygon_button.configure(fg=ROI_SETTINGS.UNSELECT_COLOR.value)
            elif shape_type == POLYGON:
                self.rectangle_button.configure(fg=ROI_SETTINGS.UNSELECT_COLOR.value)
                self.circle_button.configure(fg=ROI_SETTINGS.UNSELECT_COLOR.value)
                self.polygon_button.configure(fg=ROI_SETTINGS.SELECT_COLOR.value)
        self.selected_shape_type = copy(shape_type)
        self.set_btn_clrs()


    def show_shape_info(self):
        self.set_btn_clrs(btn=self.shape_info_btn)
        if self.shape_info_btn.cget("text") == "SHOW SHAPE INFO":
            self.shape_info_btn.configure(text="HIDE SHAPE INFO")
            self.overlay_rois_on_image(show_ear_tags=False, show_roi_info=True)
        else:
            self.shape_info_btn.configure(text="SHOW SHAPE INFO")
            self.overlay_rois_on_image(show_ear_tags=False, show_roi_info=False)


    def delete_all(self):
        self.set_btn_clrs(btn=self.delete_all_btn)
        self.reset_img_shape_memory()
        self.set_img(frame_idx=self.img_idx)
        self.draw_img()
        self.set_status_bar_panel(text='Deleted all ROIs', fg='blue')
        self.overlay_rois_on_image()

    def reset_img_shape_memory(self):
        self.roi_names, self.roi_dict = [''], {}
        self.rectangles_df, self.circles_df, self.polygon_df = pd.DataFrame(columns=get_rectangle_df_headers()), pd.DataFrame(columns=get_circle_df_headers()), pd.DataFrame(columns=get_polygon_df_headers())
        self.update_dropdown_menu(dropdown=self.roi_dropdown, new_options=self.roi_names, set_index=0)

    def update_dropdown_menu(self,
                             dropdown: SimBADropDown,
                             new_options: list,
                             set_index: Optional[int] = 0,
                             set_str: Optional[str] = None):


        dropdown.change_options(values=new_options, set_index=set_index, set_str=set_str)
        # dropdown.dropdown['values'] = new_options
        # dropdown.dropdown_var.set('')
        # if isinstance(set_index, int) and (0 <= set_index <= len(new_options) - 1):
        #     dropdown.dropdown.set(new_options[set_index])
        # elif (set_str is not None) and (set_str in new_options):
        #     dropdown.dropdown.set(set_str)
        # else:
        #     dropdown.dropdown.set(new_options[0])
        # dropdown.set_width(width=max(5, max(len(s) for s in new_options)))
        # if dropdown.dropdown['values'] == ('',):
        #     dropdown.disable()
        # else:
        #     dropdown.enable()

    def delete_named_shape(self, name: str):
        self.set_btn_clrs(btn=self.delete_selected_btn)
        if not check_str(name='', value=name, raise_error=False)[0]:
            msg = 'No ROI selected. First select an ROI in drop-down to delete it'
            self.set_status_bar_panel(text=msg, fg='red')
            raise NoROIDataError(msg=msg, source=self.__class__.__name__)
        self.rectangles_df = self.rectangles_df[self.rectangles_df['Name'] != name].reset_index(drop=True)
        self.polygon_df = self.polygon_df[self.polygon_df['Name'] != name].reset_index(drop=True)
        self.circles_df = self.circles_df[self.circles_df['Name'] != name].reset_index(drop=True)
        self.roi_names = [x for x in self.roi_names if x != name]
        if len(self.roi_names) == 0: self.roi_names = ['']
        self.roi_dict = {k: v for k, v in self.roi_dict.items() if k != name}
        self.update_dropdown_menu(dropdown=self.roi_dropdown, new_options=self.roi_names, set_index=0)
        self.set_img(frame_idx=self.img_idx)
        self.overlay_rois_on_image()
        self.set_status_bar_panel(text=f"DELETED ROI: {name}", fg='blue')


    def duplicate_selected(self):
        self.set_btn_clrs(btn=self.duplicate_selected_btn)
        selected_roi_name, duplicated_shape_entry = self.roi_dropdown.getChoices(), None
        if not check_str(name='', value=selected_roi_name, raise_error=False)[0]:
            msg = 'First select an ROI in drop-down to duplicate it'
            self.set_status_bar_panel(text=msg, fg='red')
            raise NoROIDataError(msg=msg, source=self.__class__.__name__)
        if selected_roi_name not in list(self.roi_dict.keys()):
            msg = f'{selected_roi_name} is not a valid ROI'
            self.set_status_bar_panel(text=msg, fg='red')
            raise NoROIDataError(msg=msg, source=self.__class__.__name__)
        shape_to_duplicate = copy(self.roi_dict[selected_roi_name])
        if selected_roi_name in list(self.rectangles_df['Name'].unique()):
            duplicated_shape_entry = create_duplicated_rectangle_entry(shape_entry=shape_to_duplicate, jump_size=self.settings['DUPLICATION_JUMP_SIZE'])
        elif selected_roi_name in list(self.circles_df['Name'].unique()):
            duplicated_shape_entry = create_duplicated_circle_entry(shape_entry=shape_to_duplicate, jump_size=self.settings['DUPLICATION_JUMP_SIZE'])
        elif selected_roi_name in list(self.polygon_df['Name'].unique()):
            duplicated_shape_entry = create_duplicated_polygon_entry(shape_entry=shape_to_duplicate, jump_size=self.settings['DUPLICATION_JUMP_SIZE'])
        self.add_roi(shape_entry=duplicated_shape_entry)
        self.overlay_rois_on_image()
        self.update_dropdown_menu(dropdown=self.roi_dropdown, new_options=self.roi_names, set_index=-1)
        self.set_status_bar_panel(text=f'DUPLICATED ROI {selected_roi_name}', fg='blue')


    def add_roi(self, shape_entry: dict):
        if shape_entry['Name'] in self.roi_names:
            error_txt = f'Cannot add ROI named {shape_entry["Name"]} to video {self.video_meta["video_name"]}. An ROI named {shape_entry["Name"]} already exist'
            #self.status_bar['txt'] = error_txt
            DuplicateNamesWarning(msg=error_txt, source=self.__class__.__name__)
        else:
            if shape_entry['Shape_type'].lower() == ROI_SETTINGS.RECTANGLE.value:
                self.rectangles_df = pd.concat([self.rectangles_df, pd.DataFrame([shape_entry])], ignore_index=True)
            elif shape_entry['Shape_type'].lower() == ROI_SETTINGS.CIRCLE.value:
                self.circles_df = pd.concat([self.circles_df, pd.DataFrame([shape_entry])], ignore_index=True)
            elif shape_entry['Shape_type'].lower() == ROI_SETTINGS.POLYGON.value:
                self.polygon_df = pd.concat([self.polygon_df, pd.DataFrame([shape_entry])], ignore_index=True)
            self.roi_dict[shape_entry['Name']] = shape_entry
            if self.roi_names[0] == '':
                self.roi_names = [shape_entry['Name']]
            else:
                self.roi_names.append(shape_entry["Name"])


    def change_attr_pop_up(self):
        if len(list(self.roi_dict.keys())) == 0:
            msg = 'Cannot change attributes of ROI. Create an ROI and select it in the drop-down to change its attributes'
            self.set_status_bar_panel(text=msg, fg="red")
            raise NoROIDataError(msg=msg,  source=self.__class__.__name__)
        if hasattr(self, 'change_attr_frm'):
            self.change_attr_frm.destroy()
        self.set_btn_clrs(btn=self.chg_attr_btn)
        selected_roi_name = self.roi_dropdown.getChoices()
        self.change_attr_frm = Toplevel()
        self.change_attr_frm.minsize(400, 300)
        self.change_attr_frm.wm_title("CHANGE ROI SHAPE ATTRIBUTES")
        self.change_attr_frm.iconphoto(False, self.menu_icons['edit_roi_large']["img"])

        self.change_attr_panel = LabelFrame(self.change_attr_frm, text="SHAPE ATTRIBUTES", font=Formats.FONT_HEADER.value, padx=5, pady=5)
        self.change_attr_input_dropdown = SimBADropDown(parent=self.change_attr_panel, dropdown_options=self.roi_names, label="CHANGE SHAPE: ", label_width=25, command=lambda x: self._set_shape_attributes_from_selection(x), value=selected_roi_name)
        self.new_shape_name_eb = Entry_Box(parent=self.change_attr_panel, fileDescription="NEW SHAPE NAME: ", labelwidth=25, entry_box_width=40, value=selected_roi_name)

        self.new_thickness_dropdown = SimBADropDown(parent=self.change_attr_panel, dropdown_options=ROI_SETTINGS.SHAPE_THICKNESS_OPTIONS.value, label="NEW SHAPE THICKNESS: ", label_width=25, value=self.roi_dict[selected_roi_name]['Thickness'])
        self.new_color_dropdown = SimBADropDown(parent=self.change_attr_panel, dropdown_options=list(self.color_option_dict.keys()), label="NEW SHAPE COLOR: ", label_width=25, value=self.roi_dict[selected_roi_name]['Color name'])
        self.new_ear_tag_size_dropdown = SimBADropDown(parent=self.change_attr_panel, dropdown_options=ROI_SETTINGS.EAR_TAG_SIZE_OPTIONS.value, label="NEW EAR TAG SIZE: ", label_width=25, value=self.roi_dict[selected_roi_name]['Ear_tag_size'])
        self.change_attr_save_btn = SimbaButton(parent=self.change_attr_panel, txt='SAVE ATTRIBUTES', txt_clr='black', img='save_large', cmd=self.save_attr_changes)
        self.change_attr_panel.grid(row=0, sticky=NW)
        self.change_attr_input_dropdown.grid(row=0, column=0, sticky=NW)
        self.new_shape_name_eb.grid(row=1, column=0, sticky=NW)
        self.new_thickness_dropdown.grid(row=2, column=0, sticky=NW)
        self.new_color_dropdown.grid(row=3, column=0, sticky=NW)
        self.new_ear_tag_size_dropdown.grid(row=4, column=0, sticky=NW)
        self.change_attr_save_btn.grid(row=5, column=0, sticky=NW)

    def _set_shape_attributes_from_selection(self, selected_shape_name: str):
        self.new_shape_name_eb.entry_set(val=selected_shape_name)
        self.new_thickness_dropdown.setChoices(self.roi_dict[selected_shape_name]['Thickness'])
        self.new_color_dropdown.setChoices(self.roi_dict[selected_shape_name]['Color name'])
        self.new_ear_tag_size_dropdown.setChoices(self.roi_dict[selected_shape_name]['Ear_tag_size'])


    def save_attr_changes(self):
        name = self.change_attr_input_dropdown.getChoices()
        new_name = self.new_shape_name_eb.entry_get.strip()
        shape_entry = copy(self.roi_dict[name])
        if not check_str(name='', value=new_name, raise_error=False)[0]:
            msg = f'New ROI name for {name} is invalid: {new_name}'
            self.set_status_bar_panel(text=msg, fg="red")
            raise NoROIDataError(msg=msg, source=self.__class__.__name__)
        elif (new_name in self.roi_names) and (new_name != name):
            msg = f'Cannot change ROI name from {name} to {new_name}: an ROI named {new_name} already exist.'
            self.set_status_bar_panel(text=msg, fg="red")
            raise NoROIDataError(msg=msg, source=self.__class__.__name__)

        shape_entry['Name'] = new_name
        shape_entry['Thickness'] = int(self.new_thickness_dropdown.getChoices())
        shape_entry['Color name'] = self.new_color_dropdown.getChoices()
        shape_entry['Color BGR'] = self.color_option_dict[shape_entry['Color name']]
        shape_entry['Ear_tag_size'] = int(self.new_ear_tag_size_dropdown.getChoices())
        self.delete_named_shape(name=name)
        self.add_roi(shape_entry=shape_entry)
        self.overlay_rois_on_image()
        self.update_dropdown_menu(dropdown=self.change_attr_input_dropdown, new_options=self.roi_names, set_index=None, set_str=new_name)
        self.update_dropdown_menu(dropdown=self.roi_dropdown, new_options=self.roi_names, set_index=None, set_str=new_name)
        if hasattr(self, 'change_attr_frm'):
            self.change_attr_frm.destroy()
            self.set_btn_clrs()
        self.set_status_bar_panel(text=f'Changed attributes for shape {name}', fg='blue')


    def apply_different_video(self, video_name: str):
        if video_name == '':
            error_txt = f'No other video in the SimBA project has ROIs. Draw ROIs on other videos in the SimBA project to transfer ROIs between videos'
            self.status_bar.configure(text=error_txt, fg="red")
            raise InvalidInputError(msg=error_txt, source=self.__class__.__name__)
        video_roi_dict = change_roi_dict_video_name(roi_dict=self.scaled_other_roi_dict[video_name], video_name=self.video_meta['video_name'])
        if len(video_roi_dict.keys()) > 0:
            self.reset_img_shape_memory()
            self.roi_names = list(video_roi_dict.keys())
            self.roi_dict = video_roi_dict
            self.rectangles_df, self.circles_df, self.polygon_df = get_roi_df_from_dict(roi_dict=self.roi_dict)
            self.update_dropdown_menu(dropdown=self.roi_dropdown, new_options=self.roi_names, set_index=0)
            self.overlay_rois_on_image(show_roi_info=False, show_ear_tags=False)
            self.set_status_bar_panel(text=f'COPIED {len(self.roi_names)} ROI(s) FROM VIDEO {video_name}', fg='blue')
        else:
            self.set_status_bar_panel(text=f'NO ROIs FOUND FOR VIDEO {video_name}', fg='darkred')
        self.set_btn_clrs(btn=self.apply_other_video_btn)

    def save_video_rois(self):
        self.set_btn_clrs(btn=self.save_data_btn)
        if self.upscale_factor != 1.0:
            save_rois = self.scale_roi_dict(roi_dict=self.roi_dict, scale_factor=self.upscale_factor, nesting=False)
        else:
            save_rois = deepcopy(self.roi_dict)
        video_rectangles_df, video_circles_df, video_polygon_df = get_roi_df_from_dict(roi_dict=save_rois)
        other_rectangles_df, other_circles_df, other_polygons_df = get_roi_df_from_dict(roi_dict=self.other_roi_dict, video_name_nesting=True)
        out_rectangles = pd.concat([video_rectangles_df, other_rectangles_df], axis=0).reset_index(drop=True)
        out_circles = pd.concat([video_circles_df, other_circles_df], axis=0).reset_index(drop=True)
        out_polygons = pd.concat([video_polygon_df, other_polygons_df], axis=0).reset_index(drop=True)
        store = pd.HDFStore(self.roi_coordinates_path, mode="w")
        store[Keys.ROI_RECTANGLES.value] = out_rectangles
        store[Keys.ROI_CIRCLES.value] = out_circles
        store[Keys.ROI_POLYGONS.value] = out_polygons
        store.close()
        msg = f"ROI definitions saved for video: {self.video_meta['video_name']} ({len(list(self.roi_dict.keys()))} ROI(s))"
        self.set_status_bar_panel(text=msg, fg='blue')
        stdout_success(msg=f"ROI definitions saved for video: {self.video_meta['video_name']}", source=self.__class__.__name__)

    def set_status_bar_panel(self, text: str, fg: str = 'blue'):
        self.status_bar.configure(text=text, fg=fg)
        self.status_bar.update_idletasks()


    def preferences_pop_up(self):
        if hasattr(self, 'preferences_frm'):
            self.preferences_frm.destroy()
        self.preferences_frm = Toplevel()
        self.preferences_frm.minsize(400, 300)
        self.preferences_frm.wm_title("PREFERENCES")


        pref_frm_panel = CreateLabelFrameWithIcon(parent=self.preferences_frm, header="PREFERENCES", icon_name='settings', padx=5, pady=5)
        self.line_type_dropdown = SimBADropDown(parent=pref_frm_panel, dropdown_options=ROI_SETTINGS.LINE_TYPE_OPTIONS.value, label="LINE TYPE: ", label_width=35, dropdown_width=35, value=self.settings['LINE_TYPE'])
        self.roi_select_clr_dropdown = SimBADropDown(parent=pref_frm_panel, dropdown_options=list(self.color_option_dict.keys()), label="ROI SELECT COLOR: ", label_width=35, dropdown_width=35, value=next(key for key, val in self.color_option_dict.items() if val == self.settings['ROI_SELECT_CLR']))
        self.duplication_jump_dropdown = SimBADropDown(parent=pref_frm_panel, dropdown_options=list(range(1, 100, 5)), label="DUPLICATION JUMP SIZE: ", label_width=35, dropdown_width=35, value=self.settings['DUPLICATION_JUMP_SIZE'])
        self.show_tracking_dropdown = SimBADropDown(parent=pref_frm_panel, dropdown_options=['FALSE', 'KEYPOINTS', 'BBOX', 'KEYPOINTS & BBOX'], label="SHOW TRACKING DATA: ", label_width=35, dropdown_width=35, value=self.settings[ROI_TRACKING_STYLE].upper())
        self.overlay_color_dropdown = SimBADropDown(parent=pref_frm_panel, dropdown_options=list(self.color_option_dict.keys()), label="OVERLAY GRID COLOR: ", label_width=35, dropdown_width=35, value=next(key for key, val in self.color_option_dict.items() if val == self.settings[OVERLAY_GRID_COLOR]))
        self.show_grid_overlay_dropdown = SimBADropDown(parent=pref_frm_panel, dropdown_options=['FALSE', '10MM', '20MM', '40MM', '80MM', '160MM'], label="SHOW GRID OVERLAY: ", label_width=35, dropdown_width=35, value=self.settings[SHOW_GRID_OVERLAY])
        self.show_hexagon_overlay_dropdown = SimBADropDown(parent=pref_frm_panel, dropdown_options=['FALSE', '10MM', '20MM', '40MM', '80MM', '160MM'], label="SHOW HEXAGON OVERLAY: ", label_width=35, dropdown_width=35, value=self.settings[SHOW_GRID_OVERLAY])
        self.polygon_tolerance_dropdown = SimBADropDown(parent=pref_frm_panel, dropdown_options=list(range(2, 22, 2)), label="POLYGON TOLERANCE: ", label_width=35, dropdown_width=35, value=self.settings[POLYGON_TOLERANCE], tooltip_txt='Higher values will simplify polygons. \n Smaller values will retain more polygon details')


        #self.max_width_ratio_dropdown = SimBADropDown(parent=pref_frm_panel, dropdown_options=WINDOW_SIZE_OPTIONS, label="MAX DRAW DISPLAY RATIO WIDTH: ", label_width=35, dropdown_width=35, value=MAX_DRAW_UI_DISPLAY_RATIO[0])
        #self.max_height_ratio_dropdown = SimBADropDown(parent=pref_frm_panel, dropdown_options=WINDOW_SIZE_OPTIONS, label="HEIGHT: ", label_width=10, dropdown_width=10, value=MAX_DRAW_UI_DISPLAY_RATIO[1])
        #self.min_width_ratio_dropdown = SimBADropDown(parent=pref_frm_panel, dropdown_options=WINDOW_SIZE_OPTIONS, label="MIN DRAW DISPLAY RATIO WIDTH: ", label_width=35, dropdown_width=35, value=MIN_DRAW_UI_DISPLAY_RATIO[0])
       # self.min_height_ratio_dropdown = SimBADropDown(parent=pref_frm_panel, dropdown_options=WINDOW_SIZE_OPTIONS, label="HEIGHT: ", label_width=10, dropdown_width=10, value=MIN_DRAW_UI_DISPLAY_RATIO[1])
        pref_save_btn = SimbaButton(parent=pref_frm_panel, txt="SAVE", img='save_large', font=Formats.FONT_REGULAR.value, cmd=self.set_settings)
        pref_frm_panel.grid(row=0, column=0, sticky=NW)
        self.line_type_dropdown.grid(row=0, column=0, sticky=NW, pady=5)
        self.roi_select_clr_dropdown.grid(row=1, column=0, sticky=NW, pady=5)
        self.duplication_jump_dropdown.grid(row=2, column=0, sticky=NW, pady=5)
        self.show_tracking_dropdown.grid(row=3, column=0, sticky=NW, pady=5)
        self.overlay_color_dropdown.grid(row=4, column=0, sticky=NW, pady=5)
        self.show_grid_overlay_dropdown.grid(row=5, column=0, sticky=NW, pady=5)
        self.show_hexagon_overlay_dropdown.grid(row=6, column=0, sticky=NW, pady=5)
        self.polygon_tolerance_dropdown.grid(row=7, column=0, sticky=NW, pady=5)
        #self.max_width_ratio_dropdown.grid(row=5, column=0, sticky=NW, pady=5)
        #self.max_height_ratio_dropdown.grid(row=5, column=1, sticky=NW, pady=5)
        #self.min_width_ratio_dropdown.grid(row=6, column=0, sticky=NW, pady=5)
        #self.min_height_ratio_dropdown.grid(row=6, column=1, sticky=NW, pady=5)
        pref_save_btn.grid(row=8, column=0, sticky=NW, pady=5)

    def set_settings(self):
        self.settings['LINE_TYPE'] = int(self.line_type_dropdown.get_value())
        self.settings['ROI_SELECT_CLR'] = self.color_option_dict[self.roi_select_clr_dropdown.get_value()]
        self.settings['DUPLICATION_JUMP_SIZE'] = int(self.duplication_jump_dropdown.get_value())
        self.settings[ROI_TRACKING_STYLE] = self.show_tracking_dropdown.get_value()
        self.settings[SHOW_GRID_OVERLAY] = self.show_grid_overlay_dropdown.get_value()
        self.settings[SHOW_HEXAGON_OVERLAY] = self.show_hexagon_overlay_dropdown.get_value()
        self.settings[OVERLAY_GRID_COLOR] = self.color_option_dict[self.overlay_color_dropdown.get_value()]
        if self.settings[ROI_TRACKING_STYLE] == 'FALSE':
            self.pose_data = None
        else:
            if self.expected_pose_path is None and self.pose_data_cpy is None:
                error_txt = f'Cannot show tracking data on ROI image. Initialize the interface by passing the config_path OR pose_path.'
                self.status_bar.configure(text=error_txt, fg="red")
                raise InvalidInputError(msg=error_txt, source=self.__class__.__name__)
            elif self.expected_pose_path is not None and self.pose_data_cpy is None:
                self.pose_data = get_pose_for_roi_ui(pose_path=self.expected_pose_path, video_path=self.video_path)
                if self.downscale_factor != 1.0 and self.pose_data is not None: self.pose_data = self.pose_data * self.downscale_factor
                self.pose_data_cpy = deepcopy(self.pose_data)
                self.clrs = create_color_palettes(no_animals=self.animal_cnt, map_size=int(self.pose_data.shape[1]/2) + 10)
            self.pose_data = self.pose_data_cpy
            self.settings[SHOW_TRACKING] = True
        if self.settings[SHOW_GRID_OVERLAY] != 'FALSE':
            bucket_grid_size_mm = int(self.settings[SHOW_GRID_OVERLAY][:-2])
            self.grid = list(GeometryMixin().bucket_img_into_grid_square(img_size=(self.display_img_width, self.display_img_height), bucket_grid_size_mm=bucket_grid_size_mm, px_per_mm=self.px_per_mm, add_correction=True, verbose=False)[0].values())
        else:
            self.grid = None
        if self.settings[SHOW_HEXAGON_OVERLAY] != 'FALSE':
            bucket_grid_size_mm = int(self.settings[SHOW_HEXAGON_OVERLAY][:-2])
            self.hexagon_grid = list(GeometryMixin().bucket_img_into_grid_hexagon(img_size=(self.display_img_width, self.display_img_height), bucket_size_mm=bucket_grid_size_mm, px_per_mm=self.px_per_mm, verbose=False)[0].values())
        else:
            self.hexagon_grid = None




        # max_width = float(self.max_width_ratio_dropdown.get_value())
        # max_height = float(self.max_height_ratio_dropdown.get_value())
        # min_width = float(self.min_width_ratio_dropdown.get_value())
        # min_height = float(self.min_height_ratio_dropdown.get_value())
        #
        # if (max_width, max_height) != MAX_DRAW_UI_DISPLAY_RATIO or (min_width, min_height) != MIN_DRAW_UI_DISPLAY_RATIO:
        #     self.set_screen_display(max_width=max_width, max_height=max_height, min_width=min_width, min_height=min_height)

        self.overlay_rois_on_image()


    def fixed_roi_pop_up(self):
        if hasattr(self, 'fixed_roi_frm'):
            self.fixed_roi_frm.destroy()

        self.fixed_roi_frm = Toplevel()
        self.fixed_roi_frm.minsize(400, 300)
        self.fixed_roi_frm.wm_title("FIXED ROI PREFERENCES")
        self.fixed_roi_frm.iconphoto(False, self.menu_icons['size_black']["img"])

        settings = LabelFrame(self.fixed_roi_frm, text="SETTINGS", font=Formats.FONT_HEADER.value, padx=5, pady=5)

        self.fixed_roi_name_eb = Entry_Box(parent=settings, fileDescription='ROI NAME: ', labelwidth=15, entry_box_width=25)
        self.fixed_roi_clr_drpdwn = SimBADropDown(parent=settings, dropdown_options=list(self.color_option_dict.keys()), label='COLOR:', label_width=15, value='Red')
        self.fixed_roi_thickness_drpdwn = SimBADropDown(parent=settings, dropdown_options=self.settings['SHAPE_THICKNESS_OPTIONS'], label='THICKNESS:', label_width=15, value=10)
        self.fixed_roi_eartag_size_drpdwn = SimBADropDown(parent=settings, dropdown_options=self.settings['EAR_TAG_SIZE_OPTIONS'], label='EAR TAG SIZE:', label_width=15, value=15)
        settings.grid(row=0, column=0, sticky=NW)
        self.fixed_roi_name_eb.grid(row=0, column=0, sticky=NW)
        self.fixed_roi_clr_drpdwn.grid(row=1, column=0, sticky=NW)
        self.fixed_roi_thickness_drpdwn.grid(row=1, column=1, sticky=NW)
        self.fixed_roi_eartag_size_drpdwn.grid(row=1, column=2, sticky=NW)

        rectangle_frm = LabelFrame(self.fixed_roi_frm, text="ADD RECTANGLE", pady=10, font=Formats.FONT_HEADER.value, fg="black")
        self.rectangle_width_eb = Entry_Box(rectangle_frm, '', 0, None, validation='numeric', entry_box_width='11')
        self.rectangle_width_eb.entry_set('WIDTH (MM)')
        self.rectangle_height_eb = Entry_Box(rectangle_frm, '', 0, None, validation='numeric', entry_box_width='11')
        self.rectangle_height_eb.entry_set('HEIGHT (MM)')
        add_rect_btn = SimbaButton(parent=rectangle_frm, txt='ADD RECTANGLE', img='rectangle_small', txt_clr='black', cmd= lambda: self.fixed_roi_rectangle())
        rectangle_frm.grid(row=1, column=0, sticky=NW)
        self.rectangle_width_eb.grid(row=0, column=0, sticky=NW)
        self.rectangle_height_eb.grid(row=0, column=1, sticky=NW)
        add_rect_btn.grid(row=0, column=2, sticky=NW)

        circle_frm = LabelFrame(self.fixed_roi_frm, text="ADD CIRCLE", pady=10, font=Formats.FONT_HEADER.value, fg="black")
        self.fixed_roi_circle_radius_eb = Entry_Box(circle_frm, '', 0, None, validation='numeric', entry_box_width='11')
        self.fixed_roi_circle_radius_eb.entry_set('RADIUS (MM)')
        add_circle_btn = SimbaButton(parent=circle_frm, txt='ADD CIRCLE', img='circle_small', txt_clr='black', cmd= lambda: self.fixed_roi_circle())
        circle_frm.grid(row=2, column=0, sticky=NW)
        self.fixed_roi_circle_radius_eb.grid(row=0, column=0, sticky=NW)
        add_circle_btn.grid(row=0, column=1, sticky=NW)

        self.hexagon_frm = LabelFrame(self.fixed_roi_frm, text="ADD HEXAGON", pady=10, font=Formats.FONT_HEADER.value, fg="black")
        self.hexagon_radius_eb = Entry_Box(self.hexagon_frm, '', 0, None, validation='numeric', entry_box_width='11')
        self.hexagon_radius_eb.entry_set('RADIUS (MM)')
        add_hex_btn = SimbaButton(parent=self.hexagon_frm, txt='ADD HEXAGON', img='hexagon_small', txt_clr='black', cmd= lambda: self.fixed_roi_hexagon())
        self.hexagon_frm.grid(row=3, column=0, sticky=NW)
        self.hexagon_radius_eb.grid(row=0, column=0, sticky=NW)
        add_hex_btn.grid(row=0, column=1, sticky=NW)

        self.half_circle_frm = LabelFrame(self.fixed_roi_frm, text="ADD HALF CIRCLE", pady=10, font=Formats.FONT_HEADER.value, fg="black")
        self.half_circle_radius_eb = Entry_Box(self.half_circle_frm, '', 0, None, validation='numeric', entry_box_width='11')
        self.half_circle_radius_eb.entry_set('RADIUS (MM)')
        self.half_circle_direction_drpdwn = DropDownMenu(self.half_circle_frm, 'DIRECTION:', ['NORTH', 'SOUTH', 'WEST', 'EAST', 'NORTH-EAST', 'NORTH-WEST', 'SOUTH-EAST', 'SOUTH-WEST'], 10)
        self.half_circle_direction_drpdwn.setChoices('NORTH')
        add_half_circle_btn = SimbaButton(parent=self.half_circle_frm, txt='ADD HALF CIRCLE', img='half_circle_small', txt_clr='black', cmd= lambda: self.fixed_roi_half_circle())
        self.half_circle_frm.grid(row=4, column=0, sticky=NW)
        self.half_circle_radius_eb.grid(row=0, column=0, sticky=NW)
        self.half_circle_direction_drpdwn.grid(row=0, column=1, sticky=NW)
        add_half_circle_btn.grid(row=0, column=2, sticky=NW)


        self.triangle_frm = LabelFrame(self.fixed_roi_frm, text="ADD EQUILATERAL TRIANGLE", pady=10, font=Formats.FONT_HEADER.value, fg="black")
        self.triangle_side_length_eb = Entry_Box(self.triangle_frm, '', 0, None, validation='numeric', entry_box_width='15')
        self.triangle_side_length_eb.entry_set('SIDE LENGTH (MM)')
        self.triangle_direction_drpdwn = DropDownMenu(self.triangle_frm, 'DIRECTION DEGREES:', list(range(1, 361)), 18)
        self.triangle_direction_drpdwn.setChoices('90')
        add_triangle_btn = SimbaButton(parent=self.triangle_frm, txt='ADD TRIANGLE', img='triangle_small', txt_clr='black', cmd= lambda: self.fixed_roi_triangle())

        self.triangle_frm.grid(row=5, column=0, sticky=NW)
        self.triangle_side_length_eb.grid(row=0, column=0, sticky=NW)
        self.triangle_direction_drpdwn.grid(row=0, column=1, sticky=NW)
        add_triangle_btn.grid(row=0, column=2, sticky=NW)

        self.fixed_roi_status_bar = SimBALabel(parent=self.fixed_roi_frm, txt='', txt_clr='black', bg_clr=None, font=Formats.FONT_REGULAR.value, relief='sunken')
        self.fixed_roi_status_bar.grid(row=6, column=0, sticky='we')

    def _fixed_roi_checks(self):
        self.fixed_roi_name = self.fixed_roi_name_eb.entry_get.strip()
        valid, error_msg = check_str(name='ROI NAME', value=self.fixed_roi_name, invalid_options=['NAME'], allow_blank=False, raise_error=False)
        if not valid: self.fixed_roi_status_bar['text'] = error_msg; raise InvalidInputError(msg=error_msg, source=self.__class__.__name__)
        valid, error_msg = check_int(name='THICKNESS', value=self.fixed_roi_thickness_drpdwn.getChoices(), min_value=1, raise_error=False)
        if not valid: self.fixed_roi_status_bar['text'] = error_msg; raise InvalidInputError(msg=error_msg, source=self.__class__.__name__)
        valid, error_msg = check_int(name='EAR TAG SIZE', value=self.fixed_roi_eartag_size_drpdwn.getChoices(), min_value=1, raise_error=False)
        if not valid: self.fixed_roi_status_bar['text'] = error_msg; raise InvalidInputError(msg=error_msg, source=self.__class__.__name__)
        valid, error_msg = check_str(name='COLOR', value=self.fixed_roi_clr_drpdwn.getChoices(), raise_error=False)
        if not valid: self.fixed_roi_status_bar['text'] = error_msg; raise InvalidInputError(msg=error_msg, source=self.__class__.__name__)
        if self.fixed_roi_name in self.roi_names:
            error_msg = f'An ROI named {self.fixed_roi_name} already exist for video {self.video_meta["video_name"]}. PLease choose a different name'
            self.fixed_roi_status_bar['text'] = error_msg
            raise InvalidInputError(error_msg, source=self.__class__.__name__)
        self.clr_name = self.fixed_roi_clr_drpdwn.getChoices()
        self.clr_bgr = self.color_option_dict[self.clr_name]
        self.thickness = int(self.fixed_roi_thickness_drpdwn.getChoices())
        self.ear_tag_size = int(self.fixed_roi_eartag_size_drpdwn.getChoices())
        self.shape_cnt = len(list(self.roi_dict.keys()))
        self.shape_center = (round(self.img_center[0] + (self.settings['DUPLICATION_JUMP_SIZE'] * self.shape_cnt)), round(self.img_center[1] + (self.settings['DUPLICATION_JUMP_SIZE'] * self.shape_cnt)))
    def _fixed_roi_draw(self):
        self.rectangles_df, self.circles_df, self.polygon_df = get_roi_df_from_dict(roi_dict=self.roi_dict)
        self.roi_names = list(self.roi_dict.keys())
        self.update_dropdown_menu(dropdown=self.roi_dropdown, new_options=self.roi_names, set_index=0)
        self.overlay_rois_on_image(show_ear_tags=False, show_roi_info=False)
        self.fixed_roi_status_bar['text'] = self.txt
        stdout_success(msg=self.txt)
        self.set_btn_clrs()


    def fixed_roi_rectangle(self):
        self._fixed_roi_checks()
        valid, error_msg = check_int(name='WIDTH', value=self.rectangle_width_eb.entry_get, min_value=1)
        if not valid: self.fixed_roi_status_bar['text'] = error_msg; raise InvalidInputError(msg=error_msg, source=self.__class__.__name__)
        valid, error_msg = check_int(name='HEIGHT', value=self.rectangle_height_eb.entry_get, min_value=1)
        if not valid: self.fixed_roi_status_bar['text'] = error_msg; raise InvalidInputError(msg=error_msg, source=self.__class__.__name__)
        mm_width, mm_height = round(self.rectangle_width_eb.entry_get), round(self.rectangle_height_eb.entry_get)
        width, height = round(self.rectangle_width_eb.entry_get * (float(self.px_per_mm) * self.downscale_factor)), round(self.rectangle_height_eb.entry_get * (float(self.px_per_mm) * self.downscale_factor))
        tags = get_ear_tags_for_rectangle(center=self.shape_center, width=width, height=height)
        self.roi_dict[self.fixed_roi_name] =  {'Video':                self.video_meta['video_name'],
                                               'Shape_type':           ROI_SETTINGS.RECTANGLE.value,
                                               'Name':                 self.fixed_roi_name,
                                               'Color name':           self.clr_name,
                                               'Color BGR':            self.clr_bgr,
                                               'Thickness':            self.thickness,
                                               'Center_X':             self.shape_center[0],
                                               'Center_Y':             self.shape_center[0],
                                               'topLeftX':             tags['Top left tag'][0],
                                               'topLeftY':             tags['Top left tag'][1],
                                               'Bottom_right_X':       tags['Bottom right tag'][0],
                                               'Bottom_right_Y':       tags['Bottom right tag'][1],
                                               'width':                width,
                                               'height':               height,
                                               'width_cm':             round(mm_width / 10),
                                               'height_cm':            round(mm_height / 10),
                                               'area_cm':              round(int(mm_width / 10) * int(mm_height / 10), 2),
                                               "Tags":                 tags,
                                               'Ear_tag_size': self.ear_tag_size}
        self.txt = f'New rectangle {self.fixed_roi_name} (MM h: {mm_height}, w: {mm_width}; PIXELS h {height}, w: {width}) inserted using pixel per millimeter {self.px_per_mm} conversion factor.)'
        self._fixed_roi_draw()



    def fixed_roi_circle(self):
        self._fixed_roi_checks()
        valid, error_msg = check_int(name='RADIUS', value=self.fixed_roi_circle_radius_eb.entry_get, min_value=1)
        if not valid: self.fixed_roi_status_bar['text'] = error_msg; raise InvalidInputError(msg=error_msg, source=self.__class__.__name__)
        mm_radius = round(float(self.fixed_roi_circle_radius_eb.entry_get))
        radius = round(float(self.fixed_roi_circle_radius_eb.entry_get) * (float(self.px_per_mm) * self.downscale_factor))
        self.roi_dict[self.fixed_roi_name] =  {'Video':                self.video_meta['video_name'],
                                               'Shape_type':           ROI_SETTINGS.CIRCLE.value,
                                               'Name':                 self.fixed_roi_name,
                                               'Color name':           self.clr_name,
                                               'Color BGR':            self.clr_bgr,
                                               'Thickness':            self.thickness,
                                               'centerX':              self.shape_center[0],
                                               'centerY':              self.shape_center[1],
                                               'radius':               radius,
                                               'radius_cm':            round(mm_radius / 10, 2),
                                               'area_cm':              round(math.pi * ((mm_radius / 10) / 10) ** 2, 2),
                                               "Tags":                  {"Center tag": self.shape_center,
                                                                         "Border tag": (self.shape_center[0] - radius, self.shape_center[1])},
                                               'Ear_tag_size': self.ear_tag_size}
        self.txt = f'New circle {self.fixed_roi_name} (MM radius: {mm_radius}, PIXELS radius: {radius}) inserted using pixel per millimeter {self.px_per_mm} conversion factor.)'
        self._fixed_roi_draw()


    def fixed_roi_hexagon(self):
        self._fixed_roi_checks()
        valid, error_msg = check_int(name='RADIUS', value=self.hexagon_radius_eb.entry_get, min_value=1)
        if not valid: self.fixed_roi_status_bar['text'] = error_msg; raise InvalidInputError(msg=error_msg, source=self.__class__.__name__)
        radius = round(round(float(self.hexagon_radius_eb.entry_get)) * (float(self.px_per_mm) * self.downscale_factor))
        vertices, vertices_dict = get_vertices_hexagon(center=self.shape_center, radius=radius)
        area = Polygon(vertices).simplify(tolerance=20, preserve_topology=True).area
        self.roi_dict[self.fixed_roi_name] =  {'Video':                 self.video_meta['video_name'],
                                               'Shape_type':            ROI_SETTINGS.POLYGON.value,
                                               'Name':                  self.fixed_roi_name,
                                               'Color name':            self.clr_name,
                                               'Color BGR':             self.clr_bgr,
                                               'Thickness':             self.thickness,
                                               'Center_X':              self.shape_center[0],
                                               'Center_Y':              self.shape_center[1],
                                               'vertices':              vertices,
                                               'center':                self.shape_center,
                                               'area':                  area,
                                               'max_vertice_distance':  np.max(cdist(vertices, vertices).astype(np.int32)),
                                               "area_cm":               round(area * self.px_per_mm, 2),
                                               'Tags':                  vertices_dict,
                                               'Ear_tag_size':          self.ear_tag_size}
        self.txt = f'New HEXAGON {self.fixed_roi_name} (MM radius: {self.hexagon_radius_eb.entry_get}, PIXELS radius: {radius}) inserted using pixel per millimeter {self.px_per_mm} conversion factor.)'
        self._fixed_roi_draw()


    def fixed_roi_half_circle(self):
        self._fixed_roi_checks()
        valid, error_msg = check_int(name='RADIUS', value=self.half_circle_radius_eb.entry_get, min_value=1)
        if not valid: self.fixed_roi_status_bar['text'] = error_msg; raise InvalidInputError(msg=error_msg, source=self.__class__.__name__)
        radius = round(round(self.half_circle_radius_eb.entry_get) * (float(self.px_per_mm) * self.downscale_factor))
        direction = self.half_circle_direction_drpdwn.getChoices()
        vertices, vertices_dict = get_half_circle_vertices(center=self.shape_center, radius=radius, direction=direction)
        area = Polygon(vertices).simplify(tolerance=20, preserve_topology=True).area
        self.roi_dict[self.fixed_roi_name] = {'Video': self.video_meta['video_name'],
                                              'Shape_type': ROI_SETTINGS.POLYGON.value,
                                              'Name': self.fixed_roi_name,
                                              'Color name': self.clr_name,
                                              'Color BGR': self.clr_bgr,
                                              'Thickness': self.thickness,
                                              'Center_X': self.shape_center[0],
                                              'Center_Y': self.shape_center[1],
                                              'vertices': vertices,
                                              'center': self.shape_center,
                                              'area': area,
                                              'max_vertice_distance': np.max(cdist(vertices, vertices).astype(np.int32)),
                                              "area_cm": round(area * self.px_per_mm, 2),
                                              'Tags': vertices_dict,
                                              'Ear_tag_size': self.ear_tag_size}
        self.txt = f'New HALF CIRCLE {self.fixed_roi_name} (MM radius: {self.half_circle_radius_eb.entry_get}, PIXELS radius: {radius}) inserted using pixel per millimeter {self.px_per_mm} conversion factor.)'
        self._fixed_roi_draw()

    def fixed_roi_triangle(self):
        self._fixed_roi_checks()
        valid, error_msg = check_int(name='TRIANGLE SIDE LENGTH', value=self.triangle_side_length_eb.entry_get, min_value=1)
        if not valid: self.fixed_roi_status_bar['text'] = error_msg; raise InvalidInputError(msg=error_msg, source=self.__class__.__name__)
        side_length = round(round(self.triangle_side_length_eb.entry_get) * (float(self.px_per_mm) * self.downscale_factor))
        direction = int(self.triangle_direction_drpdwn.getChoices())
        vertices, vertices_dict = get_triangle_vertices(center=self.shape_center, side_length=side_length, direction=direction)
        area = Polygon(vertices).simplify(tolerance=20, preserve_topology=True).area
        self.roi_dict[self.fixed_roi_name] = {'Video': self.video_meta['video_name'],
                                              'Shape_type': ROI_SETTINGS.POLYGON.value,
                                              'Name': self.fixed_roi_name,
                                              'Color name': self.clr_name,
                                              'Color BGR': self.clr_bgr,
                                              'Thickness': self.thickness,
                                              'Center_X': self.shape_center[0],
                                              'Center_Y': self.shape_center[1],
                                              'vertices': vertices,
                                              'center': self.shape_center,
                                              'area': area,
                                              'max_vertice_distance': np.max(cdist(vertices, vertices).astype(np.int32)),
                                              "area_cm": round(area * self.px_per_mm, 2),
                                              'Tags': vertices_dict,
                                              'Ear_tag_size': self.ear_tag_size}
        self.txt = f'New EQUILATERAL TRIANGLE {self.fixed_roi_frm} (MM radius: {self.triangle_side_length_eb.entry_get}, PIXELS radius: {side_length}) inserted using pixel per millimeter {self.px_per_mm} conversion factor.)'
        self._fixed_roi_draw()


    def close_img(self):
        try:
            self.img_window.destroy()
            self.preferences_frm.destroy()
        except:
            pass
        try:
            self.main_frm.destroy()
            self.main_frm.quit()
            self.preferences_frm.quit()
        except:
            pass

    def set_screen_display(self,
                           max_width: float,
                           max_height: float,
                           min_width: float,
                           min_height: float):

        for i in [max_width, max_height, min_width, min_height]:
            check_float(name=f'set_screen_display', value=i, min_value=0.0, max_value=1.0)
        MAX_DRAW_UI_DISPLAY_RATIO = (max_width, max_height)
        MIN_DRAW_UI_DISPLAY_RATIO = (min_width, min_height)

        self.display_img_width, self.display_img_height, self.downscale_factor, self.upscale_factor = get_img_resize_info(img_size=(self.video_meta['width'], self.video_meta['height']), display_resolution=(self.display_w, self.display_h), max_height_ratio=MAX_DRAW_UI_DISPLAY_RATIO[1], max_width_ratio=MAX_DRAW_UI_DISPLAY_RATIO[0], min_height_ratio=MIN_DRAW_UI_DISPLAY_RATIO[1], min_width_ratio=MIN_DRAW_UI_DISPLAY_RATIO[0])
        self.circle_size = PlottingMixin().get_optimal_circle_size(frame_size=(self.display_img_width, self.display_img_height), circle_frame_ratio=100)
        self.img_center = (int(self.display_img_width / 2), int(self.display_img_height / 2))
        self.img_window.update_idletasks()
        self.img_window.geometry(f"{self.display_img_width}x{self.display_img_height}")
        if self.downscale_factor != 1.0:
            self.roi_dict = self.scale_roi_dict(roi_dict=self.roi_dict, scale_factor=self.downscale_factor)
            self.other_roi_dict = self.scale_roi_dict(roi_dict=self.other_roi_dict, scale_factor=self.downscale_factor, nesting=True)
            self.rectangles_df, self.circles_df, self.polygon_df = get_roi_df_from_dict(roi_dict=self.roi_dict)
            if self.pose_data is not None:
                self.pose_data = self.pose_data * self.downscale_factor
                self.pose_data_cpy = deepcopy(self.pose_data)

            #self.overlay_rois_on_image(show_ear_tags=False, show_roi_info=False)









