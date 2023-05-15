__author__ = "Simon Nilsson"

import cv2
import os
from copy import deepcopy
import numpy as np
from typing import Union
from simba.utils.read_write import get_fn_ext, get_video_meta_data

class CalculatePixelDistanceTool(object):
    def __init__(self,
                 video_path: Union[str, os.PathLike],
                 known_mm_distance: float) -> None:

        self.video_path = video_path
        self.known_mm_distance = known_mm_distance
        self.video_dir, self.video_name, self.video_ext = get_fn_ext(video_path)
        if not os.access(video_path, os.R_OK):
            print('{} is not readable.'.format(video_path))
            raise FileNotFoundError

        self.video_meta_data = get_video_meta_data(self.video_path)
        self.cap = cv2.VideoCapture(video_path)
        self.cap.set(1, 0)
        self.sensitivity = 20
        _, self.frame = self.cap.read()
        self.original_img = deepcopy(self.frame)
        max_res = max(self.video_meta_data['width'], self.video_meta_data['height'])
        self.add_spacer = 2
        space_scaler, radius_scaler, resolution_scaler, font_scaler = 80, 20, 1500, 1.5
        self.circle_scale = int(radius_scaler / (resolution_scaler / max_res))
        self.font_scale = float(font_scaler / (resolution_scaler / max_res))
        self.spacing_scale = int(font_scaler / (resolution_scaler / max_res))
        self.coordinate_lst = []
        self.choose_first_coordinate()
        self.choose_second_coordinate()
        self.manipulate_choices()

    def create_first_instructions_window(self):
        self.side_img = np.ones((int(self.video_meta_data['height'] / 2), self.video_meta_data['width'], 3))
        cv2.putText(self.side_img, 'Double left mouse click on the first coordinate location.', (10, self.spacing_scale + 50), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (10, 255, 10), 3)
        self.first_frame = np.uint8(np.concatenate((self.frame, self.side_img), axis=0))

    def create_second_instructions_window(self):
        self.side_img = np.ones((int(self.video_meta_data['height'] / 2), self.video_meta_data['width'], 3))
        cv2.putText(self.side_img, 'Double left mouse click on the second coordinate location.', (10, self.spacing_scale + 50), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (10, 255, 10), 3)
        self.second_frame = np.uint8(np.concatenate((self.frame, self.side_img), axis=0))

    def create_third_instructions_window(self):
        self.side_img = np.ones((int(self.video_meta_data['height'] / 2), self.video_meta_data['width'], 3))
        cv2.putText(self.side_img, 'Are you happy with the displayed choice?.', (10, self.spacing_scale + 50), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (10, 255, 10), 3)
        cv2.putText(self.side_img, 'Press ESC to proceed with the displayed choice.', (10, int(self.spacing_scale + 50 * (self.add_spacer * 2))), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), 2)
        cv2.putText(self.side_img, 'Double left mouse click on circle to move circle.', (10, int(self.spacing_scale + 50 * (self.add_spacer * 3))), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), 2)
        cv2.putText(self.side_img, 'Double left mouse click on circle to place circle to new location.', (10, int(self.spacing_scale + 50 * (self.add_spacer * 4))), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), 2)
        cv2.putText(self.side_img, 'Current pixels per millimeter: {}.'.format(str(self.ppm)), (10, int(self.spacing_scale + 50 * (self.add_spacer * 5))), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), 2)
        self.third_frame = np.uint8(np.concatenate((self.frame, self.side_img), axis=0))

    def calc_px_per_mm(self):
        euclid_px_dist = np.sqrt((self.coordinate_lst[0][0] - self.coordinate_lst[1][0]) ** 2 + (self.coordinate_lst[0][1] - self.coordinate_lst[0][1]) ** 2)
        self.ppm = round(euclid_px_dist / self.known_mm_distance, 5)

    def draw_all(self):
        for c in self.coordinate_lst:
            cv2.circle(self.third_frame, c, self.circle_scale, (10, 255, 10), -1, lineType=cv2.LINE_AA)
        cv2.line(self.third_frame, self.coordinate_lst[0], self.coordinate_lst[1], (10, 255, 10), int(self.circle_scale/5))

    def choose_first_coordinate(self):
        cv2.namedWindow('SELECT_COORDINATES: double left mouse click at two locations. Press ESC when done', cv2.WINDOW_NORMAL)
        def get_coordinate(event, x, y, flags, param):
            if (event == cv2.EVENT_LBUTTONDBLCLK):
                self.coordinate_lst.append((x,y))

        self.create_first_instructions_window()
        while True:
            cv2.setMouseCallback('SELECT_COORDINATES: double left mouse click at two locations. Press ESC when done', get_coordinate)
            cv2.imshow('SELECT_COORDINATES: double left mouse click at two locations. Press ESC when done', self.first_frame)
            k = cv2.waitKey(20) & 0xFF
            if (k == 27) or (len(self.coordinate_lst) == 1):
                cv2.destroyAllWindows()
                break

    def choose_second_coordinate(self):
        self.frame = deepcopy(self.original_img)
        cv2.namedWindow('SELECT_COORDINATES: double left mouse click at two locations. Press ESC when done',cv2.WINDOW_NORMAL)
        self.create_second_instructions_window()
        cv2.circle(self.second_frame, self.coordinate_lst[0], self.circle_scale, (10, 255, 10), -1, lineType=cv2.LINE_AA)

        def get_coordinate(event, x, y, flags, param):
            if (event == cv2.EVENT_LBUTTONDBLCLK):
                self.coordinate_lst.append((x,y))

        while True:
            cv2.setMouseCallback('SELECT_COORDINATES: double left mouse click at two locations. Press ESC when done', get_coordinate)
            cv2.imshow('SELECT_COORDINATES: double left mouse click at two locations. Press ESC when done', self.second_frame)
            k = cv2.waitKey(20) & 0xFF
            if (k == 27) or (len(self.coordinate_lst) == 2):
                cv2.destroyAllWindows()
                break

    def check_if_near_a_circle(self, clicked_cords):
        self.change_cord = None
        for cord_cnt, cord in enumerate(self.coordinate_lst):
            if (clicked_cords[0] >= (cord[0] - self.sensitivity)) and (clicked_cords[0] <= (cord[0] + self.sensitivity)) and (clicked_cords[1] >= (cord[1] - self.sensitivity)) and (clicked_cords[1] <= (cord[1] + self.sensitivity)):
                self.change_cord = cord_cnt

    def manipulate_choices(self):
        self.frame = deepcopy(self.original_img)
        cv2.namedWindow('SELECT_COORDINATES: double left mouse click at two locations. Press ESC when done',cv2.WINDOW_NORMAL)
        self.calc_px_per_mm()
        self.create_third_instructions_window()
        self.draw_all()
        self.clicked_cord = (-np.inf, -np.inf)
        cv2.imshow('SELECT_COORDINATES: double left mouse click at two locations. Press ESC when done', self.third_frame)
        k = cv2.waitKey(20) & 0xFF

        def get_coordinate(event, x, y, flags, param):
            if (event == cv2.EVENT_LBUTTONDBLCLK):
                self.clicked_cord = (x, y)

        def remove_single_coordinate():
            self.coordinate_lst.pop(self.change_cord)
            self.frame = deepcopy(self.original_img)
            cv2.circle(self.frame, self.coordinate_lst[0], self.circle_scale, (10, 255, 10), -1, lineType=cv2.LINE_AA)
            cv2.namedWindow('SELECT_COORDINATES: double left mouse click at two locations. Press ESC when done', cv2.WINDOW_NORMAL)
            cv2.imshow('SELECT_COORDINATES: double left mouse click at two locations. Press ESC when done', self.third_frame)



        while True:
            cv2.setMouseCallback('SELECT_COORDINATES: double left mouse click at two locations. Press ESC when done', get_coordinate)
            cv2.imshow('SELECT_COORDINATES: double left mouse click at two locations. Press ESC when done', self.third_frame)
            k = cv2.waitKey(20) & 0xFF
            self.check_if_near_a_circle(self.clicked_cord)
            if (k == 27) and self.change_cord is not None:
                print(self.clicked_cord, self.change_cord)
                cv2.destroyAllWindows()
                remove_single_coordinate()
                self.clicked_cord = None