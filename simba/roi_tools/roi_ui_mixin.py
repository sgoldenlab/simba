import os
import time
from typing import Union, Optional
from tkinter import *
import cv2
import numpy as np
from copy import copy, deepcopy
import ctypes
import pandas as pd
import math
from shapely.geometry import Polygon
from scipy.spatial.distance import cdist

from simba.utils.checks import check_str, check_int
from simba.utils.read_write import get_video_meta_data, read_frm_of_video
from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.errors import FrameRangeError, NoROIDataError, InvalidInputError
from simba.utils.enums import ROI_SETTINGS, Formats, Keys
from simba.utils.lookups import get_color_dict
from simba.utils.warnings import DuplicateNamesWarning
from simba.ui.tkinter_functions import SimbaButton, Entry_Box, SimBALabel, DropDownMenu, get_menu_icons
from simba.video_processors.roi_selector import ROISelector
from simba.utils.printing import stdout_success
from simba.video_processors.roi_selector_circle import ROISelectorCircle
from simba.video_processors.roi_selector_polygon import ROISelectorPolygon
from simba.roi_tools.roi_utils import (get_video_roi_data_from_dict,
                                         get_roi_data,
                                         set_roi_metric_sizes,
                                         get_roi_df_from_dict,
                                         create_polygon_entry,
                                         create_rectangle_entry,
                                         get_half_circle_vertices,
                                         get_triangle_vertices,
                                         create_circle_entry,
                                         get_rectangle_df_headers,
                                         get_circle_df_headers,
                                         get_polygon_df_headers,
                                         create_duplicated_rectangle_entry,
                                         create_duplicated_circle_entry,
                                         create_duplicated_polygon_entry,
                                         get_ear_tags_for_rectangle,
                                         get_vertices_hexagon)
from simba.sandbox.roi.interactive_modifier_ui import InteractiveROIModifier

DRAW_FRAME_NAME = "DEFINE SHAPE"
CIRCLE = 'circle'
POLYGON = 'polygon'
RECTANGLE = 'rectangle'


class ROI_mixin(ConfigReader):

    def __init__(self,
                 video_path: Union[str, os.PathLike],
                 config_path: Union[str, os.PathLike],
                 img_idx: int):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=True, create_logger=False)
        self.video_meta = get_video_meta_data(video_path=video_path)
        self.img_center = (int(self.video_meta['height']  / 2), int(self.video_meta['height'] / 2))
        _, self.px_per_mm, _ = self.read_video_info(video_name=self.video_meta['video_name'], video_info_df_path=self.video_info_path)
        self.video_path = video_path
        self.img_idx = img_idx
        self.selected_shape_type = None
        self.color_option_dict = get_color_dict()
        self.menu_icons = get_menu_icons()
        cv2.namedWindow(DRAW_FRAME_NAME, cv2.WINDOW_NORMAL)
        self.draw_frm_handle = ctypes.windll.user32.FindWindowW(None, DRAW_FRAME_NAME)
        ctypes.windll.user32.SetWindowPos(self.draw_frm_handle, -1, 0, 0, 0, 0, 3)
        self.settings = {item.name: item.value for item in ROI_SETTINGS}
        self.rectangles_df, self.circles_df, self.polygon_df, self.roi_dict, self.roi_names, self.other_roi_dict, self.other_video_names_w_rois = get_roi_data(roi_path=self.roi_coordinates_path, video_name=self.video_meta['video_name'])
        self.overlay_rois_on_image(show_ear_tags=False, show_roi_info=False)

    def get_file_menu(self,
                      root: Toplevel):

        menu = Menu(root)
        file_menu = Menu(menu)
        menu.add_cascade(label="File (ROI)", menu=file_menu)
        file_menu.add_command(label="Preferences...", compound="left", image=self.menu_icons["settings"]["img"], command=lambda: self.preferences_pop_up())
        file_menu.add_command(label="Draw ROIs of pre-defined sizes...", compound="left", image=self.menu_icons["size_black"]["img"], command=lambda: self.fixed_roi_pop_up())
        root.config(menu=menu)


    def preferences_pop_up(self):
        if hasattr(self, 'preferences_frm'):
            self.preferences_frm.destroy()
        self.preferences_frm = Toplevel()
        self.preferences_frm.minsize(400, 300)
        self.preferences_frm.wm_title("PREFERENCES")
        pref_frm_panel = LabelFrame(self.preferences_frm, text="PREFERENCES", font=Formats.FONT_HEADER.value, padx=5, pady=5)
        self.line_type_dropdown = DropDownMenu(self.preferences_frm, "LINE TYPE: ", ROI_SETTINGS.LINE_TYPE_OPTIONS.value, 25)
        self.line_type_dropdown.setChoices(self.settings['LINE_TYPE'])
        self.roi_select_clr_dropdown = DropDownMenu(self.preferences_frm, "ROI SELECT COLOR: ", list(self.color_option_dict.keys()), 25)
        set_clr = next(key for key, val in self.color_option_dict.items() if val == self.settings['ROI_SELECT_CLR'])
        self.roi_select_clr_dropdown.setChoices(set_clr)
        self.duplication_jump_dropdown = DropDownMenu(self.preferences_frm, "DUPLICATION JUMP SIZE: ", list(range(1, 100, 5)), 25)
        self.duplication_jump_dropdown.setChoices(self.settings['DUPLICATION_JUMP_SIZE'])
        pref_save_btn = SimbaButton(parent=self.preferences_frm, txt="SAVE", img='save_large', font=Formats.FONT_REGULAR.value, cmd=self.set_settings)
        pref_frm_panel.grid(row=0, column=0, sticky=NW)
        self.line_type_dropdown.grid(row=0, column=0, sticky=NW, pady=5)
        self.roi_select_clr_dropdown.grid(row=1, column=0, sticky=NW, pady=5)
        self.duplication_jump_dropdown.grid(row=2, column=0, sticky=NW, pady=5)
        pref_save_btn.grid(row=3, column=0, sticky=NW, pady=5)

    def set_settings(self):
        self.settings['LINE_TYPE'] = int(self.line_type_dropdown.getChoices())
        self.settings['ROI_SELECT_CLR'] = self.color_option_dict[self.roi_select_clr_dropdown.getChoices()]
        self.settings['DUPLICATION_JUMP_SIZE'] = int(self.duplication_jump_dropdown.getChoices())
        self.overlay_rois_on_image()

    def get_video_info_panel(self,
                             parent_frame: Union[Frame, Canvas, LabelFrame, Toplevel],
                             row_idx: int):

        self.change_attr_panel = LabelFrame(parent_frame, text="VIDEO AND FRAME INFORMATION", font=Formats.FONT_HEADER.value, padx=5, pady=5)
        self.video_name_lbl = SimBALabel(parent=self.change_attr_panel, txt=f'VIDEO NAME: {self.video_meta["video_name"]}')
        self.video_fps_lbl = SimBALabel(parent=self.change_attr_panel, txt=f'FPS: {self.video_meta["fps"]}')
        self.video_frame_id_lbl = SimBALabel(parent=self.change_attr_panel, txt=f'DISPLAY FRAME #: {self.img_idx}')
        self.video_frame_time_lbl = SimBALabel(parent=self.change_attr_panel, txt=f'DISPLAY FRAME (S): {(round((self.img_idx / self.video_meta["fps"]), 2))}')
        self.change_attr_panel.grid(row=row_idx, sticky=W)
        self.video_name_lbl.grid(row=0, column=1)
        self.video_fps_lbl.grid(row=0, column=2)
        self.video_frame_id_lbl.grid(row=0, column=3)
        self.video_frame_time_lbl.grid(row=0, column=4)

    def get_select_img_panel(self,
                             parent_frame: Union[Frame, Canvas, LabelFrame, Toplevel],
                             row_idx: int):

        self.select_img_panel = LabelFrame(parent_frame, text="CHANGE IMAGE", font=Formats.FONT_HEADER.value, padx=5, pady=5)
        self.forward_1s_btn = SimbaButton(parent=self.select_img_panel, txt="+1s", img='plus_green_2', font=Formats.FONT_REGULAR.value, txt_clr='darkgreen', cmd=self.change_img, cmd_kwargs={'stride': int(self.video_meta['fps'])})
        self.backwards_1s_btn = SimbaButton(parent=self.select_img_panel, txt="-1s", img='minus_blue_2', font=Formats.FONT_REGULAR.value, txt_clr='darkblue', cmd=self.change_img, cmd_kwargs={'stride': -int(self.video_meta['fps'])})
        self.custom_seconds_entry = Entry_Box(parent=self.select_img_panel, fileDescription='CUSTOM SECONDS:', labelwidth=18, validation='numeric', entry_box_width=4)
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

        self.select_shape_type_panel = LabelFrame(parent_frame, text="SET NEW SHAPE", font=Formats.FONT_HEADER.value, padx=5, pady=5, bd=5)
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

        self.shape_attr_panel = LabelFrame(parent_frame, text="SHAPE ATTRIBUTES", font=Formats.FONT_HEADER.value, padx=5, pady=5)
        self.thickness_dropdown = DropDownMenu(self.shape_attr_panel, "SHAPE THICKNESS: ", ROI_SETTINGS.SHAPE_THICKNESS_OPTIONS.value, 17)
        self.thickness_dropdown.setChoices(5)
        self.color_dropdown = DropDownMenu(self.shape_attr_panel, "SHAPE COLOR: ", list(self.color_option_dict.keys()), 17)
        self.color_dropdown.setChoices('Red')
        self.ear_tag_size_dropdown = DropDownMenu(self.shape_attr_panel, "EAR TAG SIZE: ", ROI_SETTINGS.EAR_TAG_SIZE_OPTIONS.value,17)
        self.ear_tag_size_dropdown.setChoices(15)
        self.shape_attr_panel.grid(row=row_idx, sticky=W, pady=10)
        self.thickness_dropdown.grid(row=0, column=1, sticky=W, pady=10, padx=(0, 10))
        self.ear_tag_size_dropdown.grid(row=0, column=2, sticky=W, pady=10, padx=(0, 10))
        self.color_dropdown.grid(row=0, column=3, sticky=W, pady=10, padx=(0, 10))


    def get_shape_name_panel(self,
                             parent_frame: Union[Frame, Canvas, LabelFrame, Toplevel],
                             row_idx: int):

        self.shape_name_panel = LabelFrame(parent_frame, text="SHAPE NAME", font=Formats.FONT_HEADER.value, padx=5, pady=5)
        self.shape_name_eb = Entry_Box(parent=self.shape_name_panel, fileDescription="SHAPE NAME: ", labelwidth=15, entry_box_width=55)
        self.shape_name_panel.grid(row=row_idx, sticky=W, pady=10)
        self.shape_name_eb.grid(row=0, column=0, sticky=W, pady=10)


    def get_interact_panel(self,
                           parent_frame: Union[Frame, Canvas, LabelFrame, Toplevel],
                           row_idx: int):

        self.interact_panel = LabelFrame(parent_frame, text="SHAPE INTERACTION", font=Formats.FONT_HEADER.value, padx=5, pady=5)
        self.move_shape_btn = SimbaButton(parent=self.interact_panel, txt="MOVE SHAPE", img='move_large', txt_clr='black', cmd=self.move_shapes)
        self.shape_info_btn = SimbaButton(parent=self.interact_panel, txt="SHOW SHAPE INFO", img='info_large', txt_clr='black', enabled=True, cmd=self.show_shape_info)
        self.interact_panel.grid(row=row_idx, sticky=W, pady=10)
        self.move_shape_btn.grid(row=0, column=0, sticky=W, pady=10, padx=(0, 10))
        self.shape_info_btn.grid(row=0, column=1, sticky=W, pady=10, padx=(0, 10))


    def get_save_roi_panel(self,
                           parent_frame: Union[Frame, Canvas, LabelFrame, Toplevel],
                           row_idx: int):

        self.save_roi_panel = LabelFrame(parent_frame, text="SAVE VIDEO ROI DATA", font=Formats.FONT_HEADER.value, padx=5, pady=5)
        self.save_data_btn = SimbaButton(parent=self.save_roi_panel, txt="SAVE VIDEO ROI DATA", img='save_large', txt_clr='black', cmd=self.save_video_rois)
        self.save_roi_panel.grid(row=row_idx, sticky=W, pady=10)
        self.save_data_btn.grid(row=0, column=0, sticky=W, pady=10, padx=(0, 10))


    def save_video_rois(self):
        other_rectangles_df, other_circles_df, other_polygons_df = get_roi_df_from_dict(roi_dict=self.other_roi_dict)
        out_rectangles = pd.concat([self.rectangles_df, other_rectangles_df], axis=0).reset_index(drop=True)
        out_circles = pd.concat([self.circles_df, other_circles_df], axis=0).reset_index(drop=True)
        out_polygons = pd.concat([self.polygon_df, other_polygons_df], axis=0).reset_index(drop=True)
        store = pd.HDFStore(self.roi_coordinates_path, mode="w")
        store[Keys.ROI_RECTANGLES.value] = out_rectangles
        store[Keys.ROI_CIRCLES.value] = out_circles
        store[Keys.ROI_POLYGONS.value] = out_polygons
        store.close()
        msg = f"ROI definitions saved for video: {self.video_meta['video_name']} ({len(list(self.roi_dict.keys()))} ROI(s))"
        self.set_status_bar_panel(text=msg, fg='blue')
        stdout_success(msg=f"ROI definitions saved for video: {self.video_meta['video_name']}", source=self.__class__.__name__)


    def get_shapes_from_other_video_panel(self,
                                          parent_frame: Union[Frame, Canvas, LabelFrame, Toplevel],
                                          row_idx: int):

        self.shapes_from_other_video_panel = LabelFrame(parent_frame, text="APPLY SHAPES FROM DIFFERENT VIDEO", font=Formats.FONT_HEADER.value, padx=5, pady=5)
        self.other_videos_dropdown = DropDownMenu(self.shapes_from_other_video_panel, "FROM VIDEO: ", self.other_video_names_w_rois, 15)
        self.other_videos_dropdown.setChoices(self.other_video_names_w_rois[0])
        self.apply_other_video_btn = SimbaButton(parent=self.shapes_from_other_video_panel, txt="APPLY", img='tick_large', txt_clr='black', enabled=True, cmd=self.apply_different_video, cmd_kwargs={'video_name': lambda: self.other_videos_dropdown.getChoices()})
        self.shapes_from_other_video_panel.grid(row=row_idx, sticky=W, pady=10)
        self.other_videos_dropdown.grid(row=0, column=0, sticky=W, pady=10, padx=(0, 10))
        self.apply_other_video_btn.grid(row=0, column=1, sticky=W, pady=10, padx=(0, 10))

    def apply_different_video(self, video_name: str):
        if video_name == '':
            error_txt = f'No other video in the SimBA project has ROIs. Draw ROIs on other videos in the SimBA project to transfer ROIs between videos'
            self.status_bar.configure(text=error_txt, fg="red")
            raise InvalidInputError(msg=error_txt, source=self.__class__.__name__)
        video_roi_dict = get_video_roi_data_from_dict(roi_dict=self.other_roi_dict, video_name=video_name)
        if len(video_roi_dict.keys()) > 0:
            self.reset_img_shape_memory()
            self.roi_names = list(video_roi_dict.keys())
            self.roi_dict = copy(video_roi_dict)
            self.rectangles_df, self.circles_df, self.polygon_df = get_roi_df_from_dict(roi_dict=self.roi_dict)
            self.update_dropdown_menu(dropdown=self.roi_dropdown, new_options=self.roi_names, set_index=0)
            self.overlay_rois_on_image(show_roi_info=False, show_ear_tags=False)

    def move_shapes(self):
        if len(list(self.roi_dict.keys())) == 0:
            msg = f'Cannot move ROIs: No ROIs have been drawn on video {self.video_meta["video_name"]}.'
            self.set_status_bar_panel(text=msg, fg="red")
            raise NoROIDataError(msg, source=self.__class__.__name__)
        self.overlay_rois_on_image(show_ear_tags=True, show_roi_info=False)
        self.set_status_bar_panel(text='IN ROI MOVE MODE. SELECT "DEFINE SHAPE" WINDOW AND CLICK ESCAPE TO EXIT MOVE MODE', fg="darkred")
        interactive_modifier = InteractiveROIModifier(window_name=DRAW_FRAME_NAME, roi_dict=self.roi_dict, img=self.img, orginal_img=self.read_img(frame_idx=self.img_idx), settings=self.settings)
        interactive_modifier.run()
        self.set_status_bar_panel(text='Exited ROI move mode', fg="black")
        self.roi_dict = deepcopy(interactive_modifier.roi_dict)
        del interactive_modifier
        self.roi_dict = set_roi_metric_sizes(roi_dict=self.roi_dict, px_conversion_factor=self.px_per_mm)
        self.rectangles_df, self.circles_df, self.polygon_df = get_roi_df_from_dict(roi_dict=self.roi_dict)
        self.overlay_rois_on_image(show_ear_tags=False, show_roi_info=False)

    def get_draw_panel(self,
                       parent_frame: Union[Frame, Canvas, LabelFrame, Toplevel],
                       row_idx: int):

        self.draw_panel = LabelFrame(parent_frame, text="DRAW", font=Formats.FONT_HEADER.value, padx=5, pady=5)
        self.draw_btn = SimbaButton(parent=self.draw_panel, txt='DRAW', img='brush_large', txt_clr='black', cmd=self.draw)
        self.delete_all_btn = SimbaButton(parent=self.draw_panel, txt='DELETE ALL', img='delete_large_red', txt_clr='black', cmd=self.delete_all)
        self.roi_dropdown = DropDownMenu(self.draw_panel, "ROI: ", self.roi_names, 5)
        self.roi_dropdown.setChoices(self.roi_names[0])
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



    def draw(self):
        self.shape_info_btn.configure(text="SHOW SHAPE INFO")
        shape_name = self.shape_name_eb.entry_get.strip()
        #self.win_pos_x, self.win_pos_y, self.win_pos_w, self.win_pos_h = cv2.getWindowImageRect(DRAW_FRAME_NAME)
        if self.selected_shape_type is None:
            msg = "No shape type selected. Select shape type before drawing"
            self.set_status_bar_panel(text=msg, fg='red')
            raise NoROIDataError(msg=msg, source=f'{self.__class__.__name__} draw')
        elif not check_str(name=f'shape name', value=shape_name, allow_blank=False, raise_error=False)[0]:
            msg = f"Invalid shape name: {shape_name}. Type a shape name before drawing."
            self.set_status_bar_panel(text=msg, fg='red')
            raise InvalidInputError(msg=msg, source=f'{self.__class__.__name__} draw')
        elif shape_name in self.roi_names:
            msg = f'Cannot draw ROI named {shape_name}. An ROI named {shape_name} already exist in for the video.'
            self.set_status_bar_panel(text=msg, fg='red')
            raise InvalidInputError(msg=msg, source=f'{self.__class__.__name__} draw')
        ctypes.windll.user32.SetWindowPos(self.draw_frm_handle, -1, 0, 0, 0, 0, 3)
        if self.selected_shape_type == RECTANGLE:
            rectangle_selector = ROISelector(path=self.img, thickness=int(self.thickness_dropdown.getChoices()), clr=self.color_option_dict[self.color_dropdown.getChoices()], title=DRAW_FRAME_NAME, destroy=True)
            rectangle_selector.run()
            shape_entry = create_rectangle_entry(rectangle_selector=rectangle_selector, video_name=self.video_meta['video_name'], shape_name=shape_name, clr_name=self.color_dropdown.getChoices(), clr_bgr=self.color_option_dict[self.color_dropdown.getChoices()], thickness=int(self.thickness_dropdown.getChoices()), ear_tag_size=int(self.ear_tag_size_dropdown.getChoices()), px_conversion_factor=self.px_per_mm)
            del rectangle_selector
        elif self.selected_shape_type == CIRCLE:
            circle_selector = ROISelectorCircle(path=self.img, thickness=int(self.thickness_dropdown.getChoices()), clr=self.color_option_dict[self.color_dropdown.getChoices()], title=DRAW_FRAME_NAME, destroy=True)
            circle_selector.run()
            shape_entry = create_circle_entry(circle_selector=circle_selector, video_name=self.video_meta['video_name'], shape_name=shape_name, clr_name=self.color_dropdown.getChoices(), clr_bgr=self.color_option_dict[self.color_dropdown.getChoices()], thickness=int(self.thickness_dropdown.getChoices()), ear_tag_size=int(self.ear_tag_size_dropdown.getChoices()), px_conversion_factor=self.px_per_mm)
            del circle_selector
        else:
            polygon_selector = ROISelectorPolygon(path=self.img, thickness=int(self.thickness_dropdown.getChoices()), vertice_size=int(self.ear_tag_size_dropdown.getChoices()), clr=self.color_option_dict[self.color_dropdown.getChoices()], title=DRAW_FRAME_NAME, destroy=True)
            polygon_selector.run()
            shape_entry = create_polygon_entry(polygon_selector=polygon_selector, video_name=self.video_meta['video_name'], shape_name=shape_name, clr_name=self.color_dropdown.getChoices(), clr_bgr=self.color_option_dict[self.color_dropdown.getChoices()], thickness=int(self.thickness_dropdown.getChoices()), ear_tag_size=int(self.ear_tag_size_dropdown.getChoices()), px_conversion_factor=self.px_per_mm)
            del polygon_selector
        self.add_roi(shape_entry=shape_entry)
        self.overlay_rois_on_image()
        self.update_dropdown_menu(dropdown=self.roi_dropdown, new_options=self.roi_names, set_index=0)
        self.set_status_bar_panel(text=f'Added ROI {shape_entry["Name"]}.', fg='blue')

    # def place_cv2_window_at_location(self, x, y, w, h):
    #     cv2.destroyAllWindows()
    #     cv2.namedWindow(DRAW_FRAME_NAME, cv2.WINDOW_NORMAL)
    #     cv2.moveWindow(DRAW_FRAME_NAME, x, y)
    #     cv2.resizeWindow(DRAW_FRAME_NAME, w, h)
    #     self.img = cv2.resize(self.img, (w, h))
    #     self.draw_cv2_window()

    def get_status_bar_panel(self,
                             parent_frame: Union[Frame, Canvas, LabelFrame, Toplevel],
                             row_idx: int):

        self.status_bar = SimBALabel(parent=parent_frame, txt='', txt_clr='black', bg_clr=None, font=Formats.FONT_REGULAR.value, relief='sunken')
        self.status_bar.grid(row=row_idx, column=0, sticky='we')
        parent_frame.grid_rowconfigure(row_idx, weight=0)

    def set_status_bar_panel(self, text: str, fg: str):
        self.status_bar.configure(text=text, fg=fg)
        self.status_bar.update_idletasks()

    def change_attr_pop_up(self):
        if len(list(self.roi_dict.keys())) == 0:
            msg = 'Cannot change attributes of ROI. Create an ROI and select it in the drop-down to change its attributes'
            self.set_status_bar_panel(text=msg, fg="red")
            raise NoROIDataError(msg=msg,  source=self.__class__.__name__)
        if hasattr(self, 'change_attr_frm'):
            self.change_attr_frm.destroy()
        selected_roi_name = self.roi_dropdown.getChoices()
        self.change_attr_frm = Toplevel()
        self.change_attr_frm.minsize(400, 300)
        self.change_attr_frm.wm_title("CHANGE ROI SHAPE ATTRIBUTES")
        self.change_attr_panel = LabelFrame(self.change_attr_frm, text="SHAPE ATTRIBUTES", font=Formats.FONT_HEADER.value, padx=5, pady=5)
        self.change_attr_input_dropdown = DropDownMenu(self.change_attr_panel, "CHANGE SHAPE: ", self.roi_names, 25, com= lambda x: self._set_shape_attributes_from_selection(x))
        self.change_attr_input_dropdown.setChoices(selected_roi_name)
        self.new_shape_name_eb = Entry_Box(parent=self.change_attr_panel, fileDescription="NEW SHAPE NAME: ", labelwidth=25, entry_box_width=40)
        self.new_shape_name_eb.entry_set(val=selected_roi_name)
        self.new_thickness_dropdown = DropDownMenu(self.change_attr_panel, "NEW SHAPE THICKNESS: ", ROI_SETTINGS.SHAPE_THICKNESS_OPTIONS.value, 25)
        self.new_thickness_dropdown.setChoices(self.roi_dict[selected_roi_name]['Thickness'])
        self.new_color_dropdown = DropDownMenu(self.change_attr_panel, "NEW SHAPE COLOR: ", list(self.color_option_dict.keys()), 25)
        self.new_color_dropdown.setChoices(self.roi_dict[selected_roi_name]['Color name'])
        self.new_ear_tag_size_dropdown = DropDownMenu(self.change_attr_panel, "NEW EAR TAG SIZE: ", ROI_SETTINGS.EAR_TAG_SIZE_OPTIONS.value, 25)
        self.new_ear_tag_size_dropdown.setChoices(self.roi_dict[selected_roi_name]['Ear_tag_size'])
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
        if (new_name in self.roi_names) and (new_name != name):
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
        self.set_status_bar_panel(text=f'Changed attributes for shape {name}', fg='blue')

    def overlay_rois_on_image(self,
                              show_ear_tags: bool = False,
                              show_roi_info: bool = False):

        self.set_img(img=self.read_img(frame_idx=self.img_idx))
        self.img = PlottingMixin.rectangles_onto_image(img=self.img, rectangles=self.rectangles_df, show_tags=show_ear_tags, circle_size=None, print_metrics=show_roi_info, line_type=self.settings['LINE_TYPE'])
        self.img = PlottingMixin.circles_onto_image(img=self.img, circles=self.circles_df, show_tags=show_ear_tags, circle_size=None, print_metrics=show_roi_info, line_type=self.settings['LINE_TYPE'])
        self.img = PlottingMixin.polygons_onto_image(img=self.img, polygons=self.polygon_df, show_tags=show_ear_tags, circle_size=None, print_metrics=show_roi_info, line_type=self.settings['LINE_TYPE'])
        self.draw_cv2_window()

    def draw_cv2_window(self):
        cv2.destroyAllWindows()
        cv2.namedWindow(DRAW_FRAME_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(DRAW_FRAME_NAME, self.img)

    def show_shape_info(self):
        if self.shape_info_btn.cget("text") == "SHOW SHAPE INFO":
            self.shape_info_btn.configure(text="HIDE SHAPE INFO")
            self.overlay_rois_on_image(show_ear_tags=False, show_roi_info=True)
        else:
            self.shape_info_btn.configure(text="SHOW SHAPE INFO")
            self.overlay_rois_on_image(show_ear_tags=False, show_roi_info=False)

    def update_dropdown_menu(self, dropdown: DropDownMenu, new_options: list, set_index: Optional[int] = 0, set_str: Optional[str] = None):
        dropdown.popupMenu['menu'].delete(0, 'end')
        for option in new_options:
            dropdown.popupMenu['menu'].add_command(label=option, command=lambda value=option: dropdown.dropdownvar.set(value))
        if isinstance(set_index, int) and (set_index < (len(new_options) - 1)):
            dropdown.setChoices(new_options[set_index])
        elif (set_str is not None) and (set_str in new_options):
            dropdown.setChoices(set_str)
        else:
            dropdown.setChoices(new_options[0])

    def delete_all(self):
        self.reset_img_shape_memory()
        self.set_img(img=self.read_img(frame_idx=self.img_idx))
        self.set_status_bar_panel(text='Deleted all ROIs', fg='blue')

    def delete_named_shape(self, name: str):
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
        self.set_img(img=self.read_img(frame_idx=self.img_idx))
        self.overlay_rois_on_image()

    def reset_img_shape_memory(self):
        self.roi_names, self.roi_dict = [''], {}
        self.rectangles_df, self.circles_df, self.polygon_df = pd.DataFrame(columns=get_rectangle_df_headers()), pd.DataFrame(columns=get_circle_df_headers()), pd.DataFrame(columns=get_polygon_df_headers())
        self.update_dropdown_menu(dropdown=self.roi_dropdown, new_options=self.roi_names, set_index=0)

    def duplicate_selected(self):
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
        self.set_status_bar_panel(text=f'Duplicated shape {selected_roi_name}', fg='blue')


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
            new_img = self.read_img(frame_idx=new_frm_idx)
            self.set_img(img=new_img)
            self.img_idx = copy(new_frm_idx)
            self.overlay_rois_on_image(show_roi_info=False, show_ear_tags=False)
            frm_time = round((self.img_idx / self.video_meta["fps"]), 2)
            self.video_frame_id_lbl.configure(text=f'DISPLAY FRAME #: {self.img_idx}')
            self.video_frame_time_lbl.configure(text=f'DISPLAY FRAME (S): {frm_time}')
            self.set_status_bar_panel(text=f'Set frame to frame number {new_frm_idx} ({frm_time}s)', fg='blue')

    def set_img(self, img: np.ndarray):
        self.img = img
        self.draw_cv2_window()

    def read_img(self, frame_idx: int):
        return read_frm_of_video(video_path=self.video_path, frame_index=frame_idx)

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

    def add_roi(self, shape_entry: dict):
        if shape_entry['Name'] in self.roi_names:
            error_txt = f'Cannot add ROI named {shape_entry["Name"]} to video {self.video_meta["video_name"]}. An ROI named {shape_entry["Name"]} already exist'
            self.status_bar['txt'] = error_txt
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


    def fixed_roi_pop_up(self):
        if hasattr(self, 'fixed_roi_frm'):
            self.fixed_roi_frm.destroy()

        self.fixed_roi_frm = Toplevel()
        self.fixed_roi_frm.minsize(400, 300)
        self.fixed_roi_frm.wm_title("FIXED ROI PREFERENCES")
        settings = LabelFrame(self.fixed_roi_frm, text="SETTINGS", font=Formats.FONT_HEADER.value, padx=5, pady=5)

        self.fixed_roi_name_eb = Entry_Box(parent=settings, fileDescription='ROI NAME: ', labelwidth=15, entry_box_width=25)
        self.fixed_roi_clr_drpdwn = DropDownMenu(settings, 'ROI COLOR: ', list(self.color_option_dict.keys()), 15)
        self.fixed_roi_clr_drpdwn.setChoices('Red')
        self.fixed_roi_thickness_drpdwn = DropDownMenu(settings, 'ROI THICKNESS:', self.settings['SHAPE_THICKNESS_OPTIONS'], 15)
        self.fixed_roi_thickness_drpdwn.setChoices(10)
        self.fixed_roi_eartag_size_drpdwn = DropDownMenu(settings, 'ROI EAR TAG SIZE: ', self.settings['EAR_TAG_SIZE_OPTIONS'], 15)
        self.fixed_roi_eartag_size_drpdwn.setChoices(15)

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
        self.shape_center = (self.img_center[0] + (self.settings['DUPLICATION_JUMP_SIZE'] * self.shape_cnt), int(self.img_center[1] + (self.settings['DUPLICATION_JUMP_SIZE'] * self.shape_cnt)))

    def _fixed_roi_draw(self):
        self.rectangles_df, self.circles_df, self.polygon_df = get_roi_df_from_dict(roi_dict=self.roi_dict)
        self.roi_names = list(self.roi_dict.keys())
        self.update_dropdown_menu(dropdown=self.roi_dropdown, new_options=self.roi_names, set_index=0)
        self.overlay_rois_on_image(show_ear_tags=False, show_roi_info=False)
        self.fixed_roi_status_bar['text'] = self.txt
        stdout_success(msg=self.txt)


    def fixed_roi_rectangle(self):
        self._fixed_roi_checks()
        valid, error_msg = check_int(name='WIDTH', value=self.rectangle_width_eb.entry_get, min_value=1)
        if not valid: self.fixed_roi_status_bar['text'] = error_msg; raise InvalidInputError(msg=error_msg, source=self.__class__.__name__)
        valid, error_msg = check_int(name='HEIGHT', value=self.rectangle_height_eb.entry_get, min_value=1)
        if not valid: self.fixed_roi_status_bar['text'] = error_msg; raise InvalidInputError(msg=error_msg, source=self.__class__.__name__)
        mm_width, mm_height = int(self.rectangle_width_eb.entry_get), int(self.rectangle_height_eb.entry_get)
        width, height = int(int(self.rectangle_width_eb.entry_get) * float(self.px_per_mm)), int(int(self.rectangle_height_eb.entry_get) * float(self.px_per_mm))
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
                                               'width_cm':             int(mm_width / 10),
                                               'height_cm':            int(mm_height / 10),
                                               'area_cm':              round(int(mm_width / 10) * int(mm_height / 10), 2),
                                               "Tags":                 tags,
                                               'Ear_tag_size': self.ear_tag_size}
        self.txt = f'New rectangle {self.fixed_roi_name} (MM h: {mm_height}, w: {mm_width}; PIXELS h {height}, w: {width}) inserted using pixel per millimeter {self.px_per_mm} conversion factor.)'
        self._fixed_roi_draw()


    def fixed_roi_circle(self):
        self._fixed_roi_checks()
        valid, error_msg = check_int(name='RADIUS', value=self.fixed_roi_circle_radius_eb.entry_get, min_value=1)
        if not valid: self.fixed_roi_status_bar['text'] = error_msg; raise InvalidInputError(msg=error_msg, source=self.__class__.__name__)
        mm_radius = int(self.fixed_roi_circle_radius_eb.entry_get)
        radius = int(int(self.fixed_roi_circle_radius_eb.entry_get) * float(self.px_per_mm))
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
                                               'area_cm':              round(math.pi * (round(round(mm_radius / 10, 2) / 10, 2) ** 2), 2),
                                               "Tags":                  {"Center tag": self.shape_center,
                                                                         "Border tag": (self.shape_center[0] - radius, self.shape_center[1])},
                                               'Ear_tag_size': self.ear_tag_size}
        self.txt = f'New circle {self.fixed_roi_name} (MM radius: {mm_radius}, PIXELS radius: {radius}) inserted using pixel per millimeter {self.px_per_mm} conversion factor.)'
        self._fixed_roi_draw()




    def fixed_roi_hexagon(self):
        self._fixed_roi_checks()
        valid, error_msg = check_int(name='RADIUS', value=self.hexagon_radius_eb.entry_get, min_value=1)
        if not valid: self.fixed_roi_status_bar['text'] = error_msg; raise InvalidInputError(msg=error_msg, source=self.__class__.__name__)
        radius = int(int(self.hexagon_radius_eb.entry_get) * float(self.px_per_mm))
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
        radius = int(int(self.half_circle_radius_eb.entry_get) * float(self.px_per_mm))
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
        side_length = int(int(self.triangle_side_length_eb.entry_get) * float(self.px_per_mm))
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
        cv2.destroyAllWindows()

