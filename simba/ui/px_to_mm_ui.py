__author__ = "Simon Nilsson"

import os
from typing import Optional, Union

import cv2
import numpy as np

from simba.mixins.image_mixin import ImageMixin
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_instance, check_int)
from simba.utils.enums import TextOptions
from simba.utils.errors import FrameRangeError
from simba.utils.read_write import (get_fn_ext, get_video_meta_data,
                                    read_frm_of_video)

PIXEL_SENSITIVITY = 20
DRAW_COLOR = (144, 0, 255)

class GetPixelsPerMillimeterInterface():
    """
    Graphical interface to compute how many pixels represents a metric millimeter.

    .. video:: _static/img/GetPixelsPerMillimeterInterface.webm
       :width: 800
       :autoplay:
       :loop:

    :param Union[str, os.PathLike] video_path: Path to a video file on disk.
    :param float known_metric_mm: Known millimeter distance to get the pixels conversion factor for.
    :returns float: The number of pixels per metric millimeter.

    :example:
    >>> runner = GetPixelsPerMillimeterInterface(video_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/videos/Together_1.avi', known_metric_mm=140)
    >>> runner.run()
    >>> runner = GetPixelsPerMillimeterInterface(video_path='/Users/simon/Desktop/blank_video.mp4', known_metric_mm=140)
    >>> runner.run()
    """

    def __init__(self,
                 video_path: Union[str, os.PathLike],
                 known_metric_mm: float):

        check_file_exist_and_readable(file_path=video_path)
        self.video_meta_data = get_video_meta_data(video_path=video_path)
        check_float(name='distance', value=known_metric_mm, min_value=1)
        known_metric_mm = float(known_metric_mm)
        self.video_path, self.known_metric_mm = video_path, known_metric_mm
        self.frame = ImageMixin.find_first_non_uniform_clr_frm(video_path=video_path, start_idx=0, end_idx=self.video_meta_data['fps'])
        self.video_dir, self.video_name, _ = get_fn_ext(filepath=self.video_path)
        self.font_scale, self.spacing_x, self.spacing_y = PlottingMixin().get_optimal_font_scales(text='"Select coordinates: double left mouse click at two locations. Press ESC when done"', accepted_px_width=int(self.video_meta_data['width']), accepted_px_height=int(self.video_meta_data['height'] * 0.3))
        self.circle_scale = PlottingMixin().get_optimal_circle_size(frame_size=(int(self.video_meta_data['width']), int(self.video_meta_data['height'])), circle_frame_ratio=70)
        self.original_frm = self.frame.copy()
        self.overlay_frm = self.frame.copy()
        self.ix, self.iy = -1, -1
        self.cord_results = []
        self.cord_status = False
        self.move_status = False
        self.insert_status = False
        self.change_loop = False
        self.coord_change = []
        self.new_cord_lst = []


    def find_first_non_uniform_clr_frm(self,
                                       video_path: Union[str, os.PathLike, cv2.VideoCapture],
                                       start_idx: Optional[int] = None,
                                       end_idx: Optional[int] = None):

        check_instance(source='find_first_non_uniform_clr_frm', instance=video_path, accepted_types=(str, cv2.VideoCapture))
        video_meta_data = get_video_meta_data(video_path=video_path)
        if isinstance(start_idx, int) and isinstance(end_idx, int):
            if start_idx >= end_idx:
                raise FrameRangeError(msg=f'Start frame ({start_idx}) has to be before the end frame ({end_idx})', source='find_first_non_uniform_clr_frm')

        if isinstance(start_idx, int):
            check_int(name='start_idx', value=start_idx, min_value=0)
            if start_idx > (video_meta_data['frame_count'] - 1):
                start_idx = int(video_meta_data['frame_count'] - 1)
        elif start_idx is None:
            start_idx = 0
        else:
            raise FrameRangeError(msg=f'Start frame idx {(start_idx)} has to be None or an integer', source='find_first_non_uniform_clr_frm')
        if isinstance(end_idx, int):
            check_int(name='end_idx', value=start_idx, min_value=0)
            if end_idx > (video_meta_data['frame_count']):
                end_idx = int(video_meta_data['frame_count'])
        elif end_idx is None:
            end_idx = int(video_meta_data['fps'])
        else:
            raise FrameRangeError(msg=f'End frame idx {(end_idx)} has to be None or an integer', source='find_first_non_uniform_clr_frm')

        for frame_index in range(int(start_idx), int(end_idx)):
            self.frame = read_frm_of_video(video_path=video_path, frame_index=frame_index)
            if self.frame.ndim == 2:
                clr_cnt = np.unique(self.frame).shape[0]
            else:
                clr_cnt = np.unique(self.frame.reshape(-1, self.frame.shape[-1]), axis=0).shape[0]
            if clr_cnt > 1:
                return self.frame





    def _draw_circle(self, event, x, y, flags, param):
        if (event == cv2.EVENT_LBUTTONDBLCLK) and (len(self.cord_results) < 4):
            cv2.circle(self.overlay_frm, (x, y), self.circle_scale, DRAW_COLOR, -1)
            self.cord_results.append(x)
            self.cord_results.append(y)
            if len(self.cord_results) == 4:
                self.cord_status = True
                cv2.line(self.overlay_frm, (self.cord_results[0], self.cord_results[1]), (self.cord_results[2], self.cord_results[3]), DRAW_COLOR, 6)

    def _select_cord_to_change(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            if np.sqrt((x - self.cord_results[0]) ** 2 + (y - self.cord_results[1]) ** 2) <= PIXEL_SENSITIVITY:
                self.coord_change = [1, self.cord_results[0], self.cord_results[1]]
                self.move_status = True
            elif np.sqrt((x - self.cord_results[2]) ** 2 + (y - self.cord_results[3]) ** 2) <= PIXEL_SENSITIVITY:
                self.coord_change = [2, self.cord_results[2], self.cord_results[3]]
                self.move_status = True

    def _select_new_dot_location(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.new_cord_lst.append(x)
            self.new_cord_lst.append(y)
            self.insert_status = True

    def run(self):
        cv2.namedWindow("Select coordinates: double left mouse click at two locations. Press ESC when done", cv2.WINDOW_NORMAL)
        while 1:
            if self.cord_status == False and (self.move_status == False) and (self.insert_status == False):
                cv2.setMouseCallback("Select coordinates: double left mouse click at two locations. Press ESC when done", self._draw_circle)
                cv2.imshow("Select coordinates: double left mouse click at two locations. Press ESC when done", self.overlay_frm)
                k = cv2.waitKey(20) & 0xFF
                if k == 27:
                    break
            if (self.cord_status == True) and (self.move_status == False) and (self.insert_status == False):
                if self.change_loop == True:
                    self.overlay_frm = self.original_frm.copy()
                    cv2.circle(self.overlay_frm, (self.cord_results[0], self.cord_results[1]), self.circle_scale, DRAW_COLOR, -1)
                    cv2.circle(self.overlay_frm, (self.cord_results[2], self.cord_results[3]), self.circle_scale, DRAW_COLOR, -1)
                    cv2.line(self.overlay_frm, (self.cord_results[0], self.cord_results[1]), (self.cord_results[2], self.cord_results[3]), DRAW_COLOR, int(self.circle_scale / 3))
                cv2.putText(self.overlay_frm, "Click on circle to move", (TextOptions.BORDER_BUFFER_X.value, 50), cv2.FONT_HERSHEY_TRIPLEX, self.font_scale, DRAW_COLOR, 2)
                cv2.putText(self.overlay_frm, "Press ESC to save and exit", (TextOptions.BORDER_BUFFER_X.value, 50 + self.spacing_y), cv2.FONT_HERSHEY_TRIPLEX, self.font_scale, DRAW_COLOR, 2)
                cv2.imshow("Select coordinates: double left mouse click at two locations. Press ESC when done", self.overlay_frm)
                cv2.setMouseCallback("Select coordinates: double left mouse click at two locations. Press ESC when done", self._select_cord_to_change)

            if (self.move_status == True) and (self.insert_status == False):
                if self.change_loop == True:
                    self.frame = self.original_frm.copy()
                    self.change_loop = False
                if self.coord_change[0] == 1:
                    cv2.circle(self.frame, (self.cord_results[2], self.cord_results[3]), self.circle_scale, DRAW_COLOR, -1)
                if self.coord_change[0] == 2:
                    cv2.circle(self.frame, (self.cord_results[0], self.cord_results[1]), self.circle_scale, DRAW_COLOR, -1)
                cv2.imshow("Select coordinates: double left mouse click at two locations. Press ESC when done", self.frame)
                cv2.putText(self.frame, "Click on new circle location", (TextOptions.BORDER_BUFFER_X.value, 50), cv2.FONT_HERSHEY_TRIPLEX, self.font_scale, DRAW_COLOR, 2)
                cv2.setMouseCallback("Select coordinates: double left mouse click at two locations. Press ESC when done", self._select_new_dot_location)

            if self.insert_status == True:
                if self.coord_change[0] == 1:
                    cv2.circle(self.frame, (self.cord_results[2], self.cord_results[3]), self.circle_scale, DRAW_COLOR, -1)
                    cv2.circle(self.frame, (self.new_cord_lst[-2], self.new_cord_lst[-1]), self.circle_scale, DRAW_COLOR, -1)
                    cv2.line(self.frame, (self.cord_results[2], self.cord_results[3]), (self.new_cord_lst[-2], self.new_cord_lst[-1]), DRAW_COLOR, int(self.circle_scale / 3))
                    self.cord_results = [self.new_cord_lst[-2], self.new_cord_lst[-1], self.cord_results[2], self.cord_results[3]]
                    self.cord_status = True
                    self.move_status = False
                    self.insert_status = False
                    self.change_loop = True
                if self.coord_change[0] == 2:
                    cv2.circle(self.frame, (self.cord_results[0], self.cord_results[1]), self.circle_scale, DRAW_COLOR, -1)
                    cv2.circle(self.frame, (self.new_cord_lst[-2], self.new_cord_lst[-1]), self.circle_scale, DRAW_COLOR, -1)
                    cv2.line(self.frame, (self.cord_results[0], self.cord_results[1]), (self.new_cord_lst[-2], self.new_cord_lst[-1]), DRAW_COLOR, int(self.circle_scale / 3))
                    self.cord_results = [self.cord_results[0], self.cord_results[1], self.new_cord_lst[-2], self.new_cord_lst[-1]]
                    self.cord_status = True
                    self.move_status = False
                    self.insert_status = False
                    self.change_loop = True
                cv2.imshow("Select coordinates: double left mouse click at two locations. Press ESC when done", self.frame)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break

        self.cord_status = False
        self.move_status = False
        self.insert_status = False
        self.change_loop = False
        cv2.destroyAllWindows()
        euclidean_px_dist = np.sqrt((self.cord_results[0] - self.cord_results[2]) ** 2 + (self.cord_results[1] - self.cord_results[3]) ** 2)
        self.ppm = euclidean_px_dist / self.known_metric_mm