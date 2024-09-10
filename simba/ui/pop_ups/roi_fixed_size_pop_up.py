__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"




from tkinter import NW, Label, LabelFrame
from typing import Dict, Optional, Tuple

import numpy as np

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.roi_tools.ROI_image import ROI_image_class
from simba.roi_tools.ROI_size_calculations import (get_ear_tags_for_rectangle,
                                                   get_half_circle_vertices,
                                                   get_triangle_vertices,
                                                   get_vertices_hexagon)
from simba.ui.tkinter_functions import DropDownMenu, Entry_Box, SimbaButton
from simba.utils.checks import check_int, check_str, check_valid_tuple
from simba.utils.enums import Formats
from simba.utils.errors import InvalidInputError
from simba.utils.lookups import get_color_dict
from simba.utils.printing import stdout_success

THICKNESS_OPTIONS = list(range(1, 26, 1))
EAR_TAG_SIZE_OPTIONS = list(range(1, 26, 1))
THICKNESS_OPTIONS.insert(0, 'THICKNESS')
EAR_TAG_SIZE_OPTIONS.insert(0, 'EAR TAG SIZE')


class DrawFixedROIPopUp(PopUpMixin):

    """
    GUI for drawing specifying
    """
    def __init__(self,
                 roi_image: ROI_image_class):

        PopUpMixin.__init__(self, title="DRAW ROI OF FIXED SIZE")
        self.clrs_dict = get_color_dict()
        self.clrs = list(self.clrs_dict.keys())
        self.shape_cnt = 0
        self.roi_image = roi_image
        self.roi_define = roi_image.roi_define
        self.jump_size = roi_image.roi_define.duplicate_jump_size
        self.px_per_mm = roi_image.roi_define.curr_px_mm
        self.w, self.h = self.roi_image.frame_height, self.roi_image.frame_width
        self.img_center = (int(self.h/2), int(self.w/2))

        self.settings_frm = LabelFrame(self.main_frm, text="SETTINGS", pady=10, font=Formats.FONT_HEADER.value, fg="black")
        self.name_eb = Entry_Box(self.settings_frm, 'NAME', 10)
        self.clr_drpdwn = DropDownMenu(self.settings_frm, 'COLOR:', self.clrs, 10)
        self.clr_drpdwn.setChoices('Red')
        self.thickness_drpdwn = DropDownMenu(self.settings_frm, 'THICKNESS:', THICKNESS_OPTIONS, 10)
        self.thickness_drpdwn.setChoices(10)
        self.eartag_size_drpdwn = DropDownMenu(self.settings_frm, 'EAR TAG SIZE', EAR_TAG_SIZE_OPTIONS, 10)
        self.eartag_size_drpdwn.setChoices(5)

        self.settings_frm.grid(row=0, column=0, sticky=NW)
        self.name_eb.grid(row=0, column=0, sticky=NW)
        self.clr_drpdwn.grid(row=0, column=1, sticky=NW)
        self.thickness_drpdwn.grid(row=0, column=2, sticky=NW)
        self.eartag_size_drpdwn.grid(row=0, column=3, sticky=NW)

        self.rectangle_frm = LabelFrame(self.main_frm, text="ADD RECTANGLE", pady=10, font=Formats.FONT_HEADER.value, fg="black")
        self.rectangle_width_eb = Entry_Box(self.rectangle_frm, '', 0, None, validation='numeric', entry_box_width='11')
        self.rectangle_width_eb.entry_set('WIDTH (MM)')
        self.rectangle_height_eb = Entry_Box(self.rectangle_frm, '', 0, None, validation='numeric', entry_box_width='11')
        self.rectangle_height_eb.entry_set('HEIGHT (MM)')
        add_rect_btn = SimbaButton(parent=self.rectangle_frm, txt='ADD RECTANGLE', img='square_black', cmd=self.add_rect, txt_clr='blue')
        self.rectangle_frm.grid(row=1, column=0, sticky=NW)
        self.rectangle_width_eb.grid(row=0, column=0, sticky=NW)
        self.rectangle_height_eb.grid(row=0, column=1, sticky=NW)
        add_rect_btn.grid(row=0, column=2, sticky=NW)

        self.circle_frm = LabelFrame(self.main_frm, text="ADD CIRCLE", pady=10, font=Formats.FONT_HEADER.value, fg="black")
        self.circle_radius_eb = Entry_Box(self.circle_frm, '', 0, None, validation='numeric', entry_box_width='11')
        self.circle_radius_eb.entry_set('RADIUS (MM)')
        add_circle_btn = SimbaButton(parent=self.circle_frm, txt='ADD CIRCLE', img='circle_2', cmd=self.add_circle, txt_clr='blue')
        self.circle_frm.grid(row=2, column=0, sticky=NW)
        self.circle_radius_eb.grid(row=0, column=0, sticky=NW)
        add_circle_btn.grid(row=0, column=1, sticky=NW)

        self.hexagon_frm = LabelFrame(self.main_frm, text="ADD HEXAGON", pady=10, font=Formats.FONT_HEADER.value, fg="black")
        self.hexagon_radius_eb = Entry_Box(self.hexagon_frm, '', 0, None, validation='numeric', entry_box_width='11')
        self.hexagon_radius_eb.entry_set('RADIUS (MM)')
        add_hex_btn = SimbaButton(parent=self.hexagon_frm, txt='ADD HEXAGON', img='hexagon', cmd=self.add_hex, txt_clr='blue')

        self.hexagon_frm.grid(row=3, column=0, sticky=NW)
        self.hexagon_radius_eb.grid(row=0, column=0, sticky=NW)
        add_hex_btn.grid(row=0, column=1, sticky=NW)

        self.half_circle_frm = LabelFrame(self.main_frm, text="ADD HALF CIRCLE", pady=10, font=Formats.FONT_HEADER.value, fg="black")
        self.half_circle_radius_eb = Entry_Box(self.half_circle_frm, '', 0, None, validation='numeric', entry_box_width='11')
        self.half_circle_radius_eb.entry_set('RADIUS (MM)')
        self.half_circle_direction_drpdwn = DropDownMenu(self.half_circle_frm, 'DIRECTION:', ['NORTH', 'SOUTH', 'WEST', 'EAST' 'NORTH-EAST', 'NORTH-WEST', 'SOUTH-EAST', 'SOUTH-WEST'], 10)
        self.half_circle_direction_drpdwn.setChoices('NORTH')
        add_half_circle_btn = SimbaButton(parent=self.half_circle_frm, txt='ADD HALF CIRCLE', img='half_circle', cmd=self.add_half_circle, txt_clr='blue')

        self.half_circle_frm.grid(row=4, column=0, sticky=NW)
        self.half_circle_radius_eb.grid(row=0, column=0, sticky=NW)
        self.half_circle_direction_drpdwn.grid(row=0, column=1, sticky=NW)
        add_half_circle_btn.grid(row=0, column=2, sticky=NW)

        self.triangle_frm = LabelFrame(self.main_frm, text="ADD EQUILATERAL TRIANGLE", pady=10, font=Formats.FONT_HEADER.value, fg="black")
        self.triangle_side_length_eb = Entry_Box(self.triangle_frm, '', 0, None, validation='numeric', entry_box_width='15')
        self.triangle_side_length_eb.entry_set('SIDE LENGTH (MM)')
        self.triangle_direction_drpdwn = DropDownMenu(self.triangle_frm, 'DIRECTION DEGREES:', list(range(1, 361)), 18)
        self.triangle_direction_drpdwn.setChoices('90')
        add_triangle_btn = SimbaButton(parent=self.triangle_frm, txt='ADD TRIANGLE', img='triangle_black', cmd=self.add_triangle, txt_clr='blue')

        self.triangle_frm.grid(row=5, column=0, sticky=NW)
        self.triangle_side_length_eb.grid(row=0, column=0, sticky=NW)
        self.triangle_direction_drpdwn.grid(row=0, column=1, sticky=NW)
        add_triangle_btn.grid(row=0, column=2, sticky=NW)


        self.info_txt = Label(self.main_frm, text='', font=Formats.FONT_REGULAR.value)
        self.info_txt.grid(row=6, column=0, sticky=NW)
        self.main_frm.mainloop()

    def _checks(self):
        name = self.name_eb.entry_get.strip()
        valid, error_msg = check_str(name='ROI NAME', value=name, invalid_options=['NAME'], allow_blank=False, raise_error=False)
        if not valid: self.info_txt['text'] = error_msg; raise InvalidInputError(msg=error_msg, source=self.__class__.__name__)
        valid, error_msg = check_int(name='THICKNESS', value=self.thickness_drpdwn.getChoices(), min_value=1, raise_error=False)
        if not valid: self.info_txt['text'] = error_msg; raise InvalidInputError(msg=error_msg, source=self.__class__.__name__)
        valid, error_msg = check_int(name='EAR TAG SIZE', value=self.eartag_size_drpdwn.getChoices(), min_value=1, raise_error=False)
        if not valid: self.info_txt['text'] = error_msg; raise InvalidInputError(msg=error_msg, source=self.__class__.__name__)
        valid, error_msg = check_str(name='COLOR', value=self.clr_drpdwn.getChoices(), options=self.clrs, raise_error=False)
        if not valid: self.info_txt['text'] = error_msg; raise InvalidInputError(msg=error_msg, source=self.__class__.__name__)
        names_of_existing_rois = [x['Name'] for x in self.roi_image.out_rectangles] + [x['Name'] for x in self.roi_image.out_circles] + [x['Name'] for x in self.roi_image.out_rectangles]
        if name in names_of_existing_rois:
            error_msg = f'An ROI named {name} already exist for video {self.roi_define.file_name}. PLease choose a different name'
            self.info_txt['text'] = error_msg
            raise InvalidInputError(error_msg, source=self.__class__.__name__)

        self.clr_name = self.clr_drpdwn.getChoices()
        self.thickness = int(self.thickness_drpdwn.getChoices())
        self.ear_tag_size = int(self.eartag_size_drpdwn.getChoices())
        self.name = self.name_eb.entry_get.strip()

    def add_rect(self):
        self._checks()
        valid, error_msg = check_int(name='WIDTH', value=self.rectangle_width_eb.entry_get, min_value=1)
        if not valid: self.info_txt['text'] = error_msg; raise InvalidInputError(msg=error_msg, source=self.__class__.__name__)
        valid, error_msg = check_int(name='HEIGHT', value=self.rectangle_height_eb.entry_get, min_value=1)
        if not valid: self.info_txt['text'] = error_msg; raise InvalidInputError(msg=error_msg, source=self.__class__.__name__)
        mm_width, mm_height = int(self.rectangle_width_eb.entry_get), int(self.rectangle_height_eb.entry_get)
        width, height = int(int(self.rectangle_width_eb.entry_get) * float(self.px_per_mm)), int(int(self.rectangle_height_eb.entry_get) * float(self.px_per_mm))
        shape_center = (int(self.img_center[0]) + (self.jump_size*self.shape_cnt), int(self.img_center[1] + (self.jump_size*self.shape_cnt)))
        tags = get_ear_tags_for_rectangle(center=shape_center, width=width, height=height)

        results = {"Video": self.roi_define.file_name,
                   "Shape_type": 'Rectangle',
                   "Name": self.name,
                   "Color name": self.clr_name,
                   "Color BGR": self.clrs_dict[self.clr_name],
                   "Thickness": self.thickness,
                   "Center_X": shape_center[1],
                   "Center_Y": shape_center[0],
                   "topLeftX": tags['top_left_x'],
                   "topLeftY": tags['top_left_y'],
                   "Bottom_right_X": tags['bottom_right_x'],
                   "Bottom_right_Y": tags['bottom_right_y'],
                   'width': width,
                   'height': height,
                   "Tags": {"Center tag": (shape_center[1], shape_center[0]),
                            "Top left tag": (tags['top_left_x'], tags['top_left_y']),
                            "Bottom right tag": (tags['bottom_right_x'], tags['bottom_right_y']),
                            "Top right tag": tags['top_right_tag'],
                            "Bottom left tag": tags['bottom_left_tag'],
                            "Top tag": tags['top_tag'],
                            "Right tag": tags['right_tag'],
                            "Left tag": tags['left_tag'],
                            "Bottom tag": tags['bottom_tag']},
                   "Ear_tag_size": self.ear_tag_size}

        self.roi_image.out_rectangles.append(results)
        self.roi_define.get_all_ROI_names()
        self.roi_define.update_delete_ROI_menu()
        self.roi_image.insert_all_ROIs_into_image()
        txt = f'New rectangle {self.name} (MM h: {mm_height}, w: {mm_width}; PIXELS h {height}, w: {width}) inserted using pixel per millimeter {self.px_per_mm} conversion factor.)'
        self.info_txt['text'] = txt
        stdout_success(msg=txt)
        self.shape_cnt += 1

    def add_circle(self):
        self._checks()
        valid, error_msg = check_int(name='RADIUS', value=self.circle_radius_eb.entry_get, min_value=1)
        if not valid: self.info_txt['text'] = error_msg; raise InvalidInputError(msg=error_msg, source=self.__class__.__name__)
        mm_radius = int(self.circle_radius_eb.entry_get)
        radius = int(int(self.circle_radius_eb.entry_get) * float(self.px_per_mm))
        shape_center = (int(self.img_center[0]) + (self.jump_size*self.shape_cnt), int(self.img_center[1] + (self.jump_size*self.shape_cnt)))
        results = {'Video': self.roi_define.file_name,
                   'Shape_type': "Circle",
                   'Name': self.name,
                   'Color name': self.clr_name,
                   "Color BGR": self.clrs_dict[self.clr_name],
                   "Thickness": self.thickness,
                   "centerX": shape_center[0],
                   "centerY": shape_center[1],
                   "radius": radius,
                   "Tags": {
                       "Center tag": (shape_center[0], shape_center[1]),
                       "Border tag": (shape_center[0], int(shape_center[1]-radius))},
                   "Ear_tag_size": self.ear_tag_size,
                   }

        self.roi_image.out_circles.append(results)
        self.roi_define.get_all_ROI_names()
        self.roi_define.update_delete_ROI_menu()
        self.roi_image.insert_all_ROIs_into_image()
        txt = f'New circle {self.name} (MM radius: {mm_radius}, PIXELS radius: {radius}) inserted using pixel per millimeter {self.px_per_mm} conversion factor.)'
        self.info_txt['text'] = txt
        stdout_success(msg=txt)
        self.shape_cnt += 1

    def add_hex(self):
        self._checks()
        valid, error_msg = check_int(name='RADIUS', value=self.hexagon_radius_eb.entry_get, min_value=1)
        if not valid: self.info_txt['text'] = error_msg; raise InvalidInputError(msg=error_msg, source=self.__class__.__name__)
        mm_radius = int(self.hexagon_radius_eb.entry_get)
        radius = int(int(self.hexagon_radius_eb.entry_get) * float(self.px_per_mm))
        shape_center = (int(self.img_center[0]) + (self.jump_size*self.shape_cnt), int(self.img_center[1] + (self.jump_size*self.shape_cnt)))
        vertices, vertices_dict = get_vertices_hexagon(center=shape_center, radius=radius)
        results = {"Video": self.roi_define.file_name,
                   "Shape_type": "Polygon",
                   "Name": self.name,
                   "Color name": self.clr_name,
                   "Color BGR": self.clrs_dict[self.clr_name],
                   "Thickness": self.thickness,
                    "Center_X": shape_center[0],
                    "Center_Y": shape_center[1],
                    "vertices": vertices,
                    "Tags": vertices_dict,
                    "Ear_tag_size": self.ear_tag_size}

        self.roi_image.out_polygon.append(results)
        self.roi_define.get_all_ROI_names()
        self.roi_define.update_delete_ROI_menu()
        self.roi_image.insert_all_ROIs_into_image()
        txt = f'New HEXAGON {self.name} (MM radius: {mm_radius}, PIXELS radius: {radius}) inserted using pixel per millimeter {self.px_per_mm} conversion factor.)'
        self.info_txt['text'] = txt
        stdout_success(msg=txt)
        self.shape_cnt += 1

    def add_half_circle(self):
        self._checks()
        valid, error_msg = check_int(name='RADIUS', value=self.half_circle_radius_eb.entry_get, min_value=1)
        if not valid: self.info_txt['text'] = error_msg; raise InvalidInputError(msg=error_msg, source=self.__class__.__name__)
        mm_radius = int(self.half_circle_radius_eb.entry_get)
        radius = int(int(self.half_circle_radius_eb.entry_get) * float(self.px_per_mm))
        shape_center = (int(self.img_center[0]) + (self.jump_size*self.shape_cnt), int(self.img_center[1] + (self.jump_size*self.shape_cnt)))
        direction = self.half_circle_direction_drpdwn.getChoices()
        vertices, vertices_dict = get_half_circle_vertices(center=shape_center, radius=radius, direction=direction)

        results = {"Video": self.roi_define.file_name,
                   "Shape_type": "Polygon",
                   "Name": self.name,
                   "Color name": self.clr_name,
                   "Color BGR": self.clrs_dict[self.clr_name],
                   "Thickness": self.thickness,
                    "Center_X": shape_center[0],
                    "Center_Y": shape_center[1],
                    "vertices": vertices,
                    "Tags": vertices_dict,
                    "Ear_tag_size": self.ear_tag_size}

        self.roi_image.out_polygon.append(results)
        self.roi_define.get_all_ROI_names()
        self.roi_define.update_delete_ROI_menu()
        self.roi_image.insert_all_ROIs_into_image()
        txt = f'New HALF CIRCLE {self.name} (MM radius: {mm_radius}, PIXELS radius: {radius}) inserted using pixel per millimeter {self.px_per_mm} conversion factor.)'
        self.info_txt['text'] = txt
        stdout_success(msg=txt)
        self.shape_cnt += 1


    def add_triangle(self):
        self._checks()
        valid, error_msg = check_int(name='TRIANGLE SIDE LENGTH', value=self.triangle_side_length_eb.entry_get, min_value=1)
        if not valid: self.info_txt['text'] = error_msg; raise InvalidInputError(msg=error_msg, source=self.__class__.__name__)
        side_length_mm = int(self.triangle_side_length_eb.entry_get)
        side_length = int(int(self.triangle_side_length_eb.entry_get) * float(self.px_per_mm))
        shape_center = (int(self.img_center[0]) + (self.jump_size*self.shape_cnt), int(self.img_center[1] + (self.jump_size*self.shape_cnt)))
        direction = int(self.triangle_direction_drpdwn.getChoices())
        vertices, vertices_dict = get_triangle_vertices(center=shape_center, side_length=side_length, direction=direction)

        results = {"Video": self.roi_define.file_name,
                   "Shape_type": "Polygon",
                   "Name": self.name,
                   "Color name": self.clr_name,
                   "Color BGR": self.clrs_dict[self.clr_name],
                   "Thickness": self.thickness,
                    "Center_X": shape_center[0],
                    "Center_Y": shape_center[1],
                    "vertices": vertices,
                    "Tags": vertices_dict,
                    "Ear_tag_size": self.ear_tag_size}

        self.roi_image.out_polygon.append(results)
        self.roi_define.get_all_ROI_names()
        self.roi_define.update_delete_ROI_menu()
        self.roi_image.insert_all_ROIs_into_image()
        txt = f'New EQUILATERAL TRIANGLE {self.name} (MM radius: {side_length_mm}, PIXELS radius: {side_length}) inserted using pixel per millimeter {self.px_per_mm} conversion factor.)'
        self.info_txt['text'] = txt
        stdout_success(msg=txt)
        self.shape_cnt += 1

