import itertools
import os
import re
import threading
import time
from configparser import ConfigParser
from copy import deepcopy
from threading import Thread

import cv2
import numpy as np
import pandas as pd

from simba.roi_tools.ROI_move_shape import move_edge, move_edge_align
from simba.roi_tools.ROI_zoom import zoom_in
from simba.utils.data import add_missing_ROI_cols
from simba.utils.enums import ConfigKey, Keys, Paths
from simba.utils.read_write import get_fn_ext


class ROI_image_class:
    def __del__(self):
        self.stop_display_image = True
        self.display_image_thread.join()

    def __init__(
        self,
        config_path,
        colors_dict,
        duplicate_jump_size,
        line_type,
        click_sens,
        text_size,
        text_thickness,
    ):
        config = ConfigParser()
        configFile = str(config_path)
        config.read(configFile)
        self.project_path = config.get(
            ConfigKey.GENERAL_SETTINGS.value, ConfigKey.PROJECT_PATH.value
        )
        (
            self.duplicate_jump_size,
            self.line_type,
            self.click_sens,
            self.text_size,
            self.text_thickness,
        ) = (duplicate_jump_size, line_type, click_sens, text_size, text_thickness)
        self.colors = colors_dict
        self.select_color = (128, 128, 128)
        self.zoomed_img_ratio = 1
        self.no_shapes = 0
        self.current_zoom = 100
        self.out_rectangles = []
        self.out_circles = []
        self.out_polygon = []
        self.keyWaitTime = 100
        self.keyWaitChar = -1
        self.cap = None
        self.circle_info = None
        self.window_name = ""
        self.working_frame = None
        self.orig_frame = None
        self.stop_display_image = False
        self.draw_rectangle_roi = False
        self.user_rectangle_roi = []
        self.lock = threading.Lock()
        self.display_image_imshow_callback = self.get_x_y_callback
        self.display_image_thread = Thread(target=self.display_image)
        self.display_image_thread.start()

    def reset(self):
        self.stop_display_image = True

    def update_working_frame(self, cap, start_running, new_video_name):
        with self.lock:
            self.cap = deepcopy(self.cap)
            _, new_frame = cap.read()
            self.working_frame = deepcopy(new_frame)
            self.orig_frame = deepcopy(new_frame)
            if new_video_name:
                self.curr_vid_name = new_video_name
                self.window_name = "Define shape for " + new_video_name
                self.select_color = (128, 128, 128)
                self.zoomed_img_ratio = 1
                self.no_shapes = 0
                self.current_zoom = 100
                self.out_rectangles = []
                self.out_circles = []
                self.out_polygon = []
                self.check_if_ROIs_exist()
                self.keyWaitTime = 100
                self.keyWaitChar = -1
        self.stop_display_image = not start_running
        self.display_image_imshow_callback = self.get_x_y_callback

    def check_if_ROIs_exist(self):
        roi_measurement_path = os.path.join(
            self.project_path, "logs", Paths.ROI_DEFINITIONS.value
        )

        if os.path.isfile(roi_measurement_path):
            rectangles_found = pd.read_hdf(
                roi_measurement_path, key=Keys.ROI_RECTANGLES.value
            )
            circles_found = pd.read_hdf(
                roi_measurement_path, key=Keys.ROI_CIRCLES.value
            )
            polygons_found = pd.read_hdf(
                roi_measurement_path, key=Keys.ROI_POLYGONS.value
            )

            if len(rectangles_found) > 0:
                rectangles_found = rectangles_found[
                    rectangles_found["Video"] == self.curr_vid_name
                ]
                rectangles_found = add_missing_ROI_cols(rectangles_found)
                self.out_rectangles = rectangles_found.to_dict(orient="records")

            if len(circles_found) > 0:
                circles_found = circles_found[
                    circles_found["Video"] == self.curr_vid_name
                ]
                circles_found = add_missing_ROI_cols(circles_found)
                self.out_circles = circles_found.to_dict(orient="records")

            if len(polygons_found) > 0:
                polygons_found = polygons_found[
                    polygons_found["Video"] == self.curr_vid_name
                ]
                polygons_found = add_missing_ROI_cols(polygons_found)
                self.out_polygon = polygons_found.to_dict(orient="records")

            self.insert_all_ROIs_into_image()

    def update_frame_no(self, new_frame_no):
        self.cap.set(1, new_frame_no)
        _, self.working_frame = self.cap.read()
        self.insert_all_ROIs_into_image(change_frame_no=True)

    def draw_rectangle(self, rectangle_info):
        self.draw_rectangle_roi = True
        while len(self.user_rectangle_roi) == 0:
            time.sleep(1)
        ROI = self.user_rectangle_roi
        top_left_x, top_left_y = ROI[0], ROI[1]
        width = abs(ROI[0] - (ROI[2] + ROI[0]))
        height = abs(ROI[2] - (ROI[3] + ROI[2]))
        bottom_right_x, bottom_right_y = top_left_x + width, top_left_y + height

        center_tag_loc = (int(top_left_x + width / 2), int(top_left_y + height / 2))
        br_tag = (int(top_left_x + width), int(top_left_y + height))
        tr_tag = (int(top_left_x + width), int(top_left_y))
        bl_tag = (int(top_left_x), int(top_left_y + height))
        top_tag = (int(top_left_x + width / 2), int(top_left_y))
        right_tag = (int(top_left_x + width), int(top_left_y + height / 2))
        left_tag = (int(top_left_x), int(top_left_y + height / 2))
        bottom_tag = (int(top_left_x + width / 2), int(top_left_y + height))
        self.out_rectangles.append(
            {
                "Video": rectangle_info["Video_name"],
                "Shape_type": "Rectangle",
                "Name": rectangle_info["Name"],
                "Color name": rectangle_info["Shape_color_name"],
                "Color BGR": rectangle_info["Shape_color_BGR"],
                "Thickness": rectangle_info["Shape_thickness"],
                "topLeftX": top_left_x,
                "topLeftY": top_left_y,
                "Bottom_right_X": bottom_right_x,
                "Bottom_right_Y": bottom_right_y,
                "width": width,
                "height": height,
                "Tags": {
                    "Center tag": center_tag_loc,
                    "Top left tag": (top_left_x, top_left_y),
                    "Bottom right tag": br_tag,
                    "Top right tag": tr_tag,
                    "Bottom left tag": bl_tag,
                    "Top tag": top_tag,
                    "Right tag": right_tag,
                    "Left tag": left_tag,
                    "Bottom tag": bottom_tag,
                },
                "Ear_tag_size": int(rectangle_info["Shape_ear_tag_size"]),
            }
        )

        self.insert_all_ROIs_into_image()
        self.user_rectangle_roi = []

    def draw_circle_callback(self, event, x, y, flags, param):
        if event == 1:
            if not self.center_status:
                self.center_X, self.center_Y = int(x), int(y)
                cv2.circle(
                    self.working_frame,
                    (self.center_X, self.center_Y),
                    int(self.circle_info["Shape_ear_tag_size"]),
                    self.circle_info["Shape_color_BGR"],
                    -1,
                )
                self.center_status = True
            else:
                self.border_x, self.border_y = int(x), int(y)
                self.radius = int(
                    (
                        np.sqrt(
                            (self.center_X - self.border_x) ** 2
                            + (self.center_Y - self.border_y) ** 2
                        )
                    )
                )
                border_tag = (int(self.center_X - self.radius), self.center_Y)
                self.out_circles.append(
                    {
                        "Video": self.circle_info["Video_name"],
                        "Shape_type": "Circle",
                        "Name": self.circle_info["Name"],
                        "Color name": self.circle_info["Shape_color_name"],
                        "Color BGR": self.circle_info["Shape_color_BGR"],
                        "Thickness": self.circle_info["Shape_thickness"],
                        "centerX": self.center_X,
                        "centerY": self.center_Y,
                        "radius": self.radius,
                        "Tags": {
                            "Center tag": (self.center_X, self.center_Y),
                            "Border tag": border_tag,
                        },
                        "Ear_tag_size": int(self.circle_info["Shape_ear_tag_size"]),
                    }
                )
                self.insert_all_ROIs_into_image()
                self.not_done = False

    def draw_circle(self, circle_info):
        self.circle_info = circle_info
        self.center_status = False
        self.not_done = True
        self.keyWaitTime = 800
        self.display_image_imshow_callback = self.draw_circle_callback
        while True and not self.stop_display_image:
            if self.keyWaitChar == 27:
                break
            time.sleep(0.1)

    def polygon_x_y_callback(self, event, x, y, flags, param):
        if event == 1:
            if isinstance(self.polygon_pts, np.ndarray):
                print("please press draw again")
                return
            self.click_loc = (int(x), int(y))
            cv2.circle(
                self.working_frame,
                (self.click_loc[0], self.click_loc[1]),
                self.draw_info["Shape_thickness"],
                self.draw_info["Shape_color_BGR"],
                -1,
            )
            self.polygon_pts.append([int(x), int(y)])

    def draw_polygon(self):
        self.polygon_pts = []
        self.display_image_imshow_callback = self.polygon_x_y_callback
        while True and not self.stop_display_image:
            if self.keyWaitChar == 27:
                break
            time.sleep(0.1)
        self.polygon_pts = list(
            k for k, _ in itertools.groupby(self.polygon_pts)
        )  # REMOVES DUPLICATES

        self.polygon_pts = np.array(self.polygon_pts).astype("int32")
        self.poly_center = self.polygon_pts.mean(axis=0)
        self.polygon_pts_dict = {}

        for v, p in enumerate(self.polygon_pts):
            self.polygon_pts_dict["Tag_" + str(v)] = (p[0], p[1])
        self.polygon_pts_dict["Center_tag"] = (
            int(self.poly_center[0]),
            int(self.poly_center[1]),
        )

        self.out_polygon.append(
            {
                "Video": self.draw_info["Video_name"],
                "Shape_type": "Polygon",
                "Name": self.draw_info["Name"],
                "Color name": self.draw_info["Shape_color_name"],
                "Color BGR": self.draw_info["Shape_color_BGR"],
                "Thickness": self.draw_info["Shape_thickness"],
                "Center_X": int(self.poly_center[0]),
                "Center_Y": int(self.poly_center[1]),
                "vertices": self.polygon_pts,
                "Tags": self.polygon_pts_dict,
                "Ear_tag_size": int(self.draw_info["Shape_ear_tag_size"]),
            }
        )

        self.insert_all_ROIs_into_image()

    def initiate_draw(self, draw_dict):
        self.keyWaitChar = -1
        self.draw_info = draw_dict
        if self.draw_info["Shape_type"] is "rectangle":
            self.draw_rectangle(self.draw_info)
        if self.draw_info["Shape_type"] is "circle":
            self.draw_circle(self.draw_info)
        if self.draw_info["Shape_type"] is "polygon":
            self.draw_polygon()

        self.all_shape_names = []
        for r in self.out_rectangles:
            self.all_shape_names.append("Rectangle: " + str(r["Name"]))
        for c in self.out_circles:
            self.all_shape_names.append("Circle: " + str(c["Name"]))
        for p in self.out_polygon:
            self.all_shape_names.append("Polygon: " + str(p["Name"]))
        if (len(self.all_shape_names) > 1) and ("None" in self.all_shape_names):
            self.all_shape_names.remove("None")

        return self.all_shape_names

    def get_x_y_callback(self, event, x, y, flags, param):
        if event == 1:
            self.click_loc = (int(x), int(y))
            self.not_done = False

    def interact_functions(self, interact_method, zoom_val=None):
        self.not_done = True

        def recolor_roi_tags():
            self.not_done = True
            if self.closest_roi["Shape_type"] == "Rectangle":
                if self.closest_tag == "Center tag":
                    cv2.rectangle(
                        self.working_frame,
                        (self.closest_roi["topLeftX"], self.closest_roi["topLeftY"]),
                        (
                            self.closest_roi["topLeftX"] + self.closest_roi["width"],
                            self.closest_roi["topLeftY"] + self.closest_roi["height"],
                        ),
                        self.select_color,
                        self.closest_roi["Thickness"],
                    )
                if (
                    (self.closest_tag == "Top tag")
                    or (self.closest_tag == "Top left tag")
                    or (self.closest_tag == "Top right tag")
                ):
                    cv2.line(
                        self.working_frame,
                        (self.closest_roi["topLeftX"], self.closest_roi["topLeftY"]),
                        (
                            self.closest_roi["topLeftX"] + self.closest_roi["width"],
                            self.closest_roi["topLeftY"],
                        ),
                        self.select_color,
                        self.closest_roi["Thickness"],
                    )
                if (
                    (self.closest_tag == "Bottom tag")
                    or (self.closest_tag == "Bottom left tag")
                    or (self.closest_tag == "Bottom right tag")
                ):
                    cv2.line(
                        self.working_frame,
                        self.closest_roi["Tags"]["Bottom left tag"],
                        self.closest_roi["Tags"]["Bottom right tag"],
                        self.select_color,
                        self.closest_roi["Thickness"],
                    )
                if (
                    (self.closest_tag == "Left tag")
                    or (self.closest_tag == "Bottom left tag")
                    or (self.closest_tag == "Top left tag")
                ):
                    cv2.line(
                        self.working_frame,
                        self.closest_roi["Tags"]["Top left tag"],
                        self.closest_roi["Tags"]["Bottom left tag"],
                        self.select_color,
                        self.closest_roi["Thickness"],
                    )
                if (
                    (self.closest_tag == "Right tag")
                    or (self.closest_tag == "Top right tag")
                    or (self.closest_tag == "Bottom right tag")
                ):
                    cv2.line(
                        self.working_frame,
                        self.closest_roi["Tags"]["Top right tag"],
                        self.closest_roi["Tags"]["Bottom right tag"],
                        self.select_color,
                        self.closest_roi["Thickness"],
                    )

            if self.closest_roi["Shape_type"] == "Circle":
                cv2.circle(
                    self.working_frame,
                    (self.closest_roi["centerX"], self.closest_roi["centerY"]),
                    self.closest_roi["radius"],
                    self.select_color,
                    int(self.closest_roi["Thickness"]),
                )

            if self.closest_roi["Shape_type"] == "Polygon":
                if self.closest_tag == "Center_tag":
                    pts = self.closest_roi["vertices"].reshape((-1, 1, 2))
                    cv2.polylines(
                        self.working_frame,
                        [pts],
                        True,
                        self.select_color,
                        int(self.closest_roi["Thickness"]),
                    )
                else:
                    picked_tag_no = int(re.sub("[^0-9]", "", self.closest_tag))
                    border_tag_1, border_tag_2 = "Tag_" + str(
                        picked_tag_no - 1
                    ), "Tag_" + str(picked_tag_no + 1)
                    if border_tag_1 not in self.closest_roi["Tags"]:
                        border_tag_1 = list(self.closest_roi["Tags"].keys())[-1]
                        if border_tag_1 == "Center_tag":
                            border_tag_1 = list(self.closest_roi["Tags"].keys())[-2]
                    if border_tag_2 not in self.closest_roi["Tags"]:
                        border_tag_2 = list(self.closest_roi["Tags"].keys())[0]
                        if border_tag_2 == "Center_tag":
                            border_tag_2 = list(self.closest_roi["Tags"].keys())[1]
                    cv2.line(
                        self.working_frame,
                        self.closest_roi["Tags"][self.closest_tag],
                        self.closest_roi["Tags"][border_tag_1],
                        self.select_color,
                        self.closest_roi["Thickness"],
                    )
                    cv2.line(
                        self.working_frame,
                        self.closest_roi["Tags"][self.closest_tag],
                        self.closest_roi["Tags"][border_tag_2],
                        self.select_color,
                        self.closest_roi["Thickness"],
                    )

            cv2.circle(
                self.working_frame,
                (
                    self.closest_roi["Tags"][self.closest_tag][0],
                    self.closest_roi["Tags"][self.closest_tag][1],
                ),
                self.closest_roi["Ear_tag_size"],
                self.select_color,
                self.closest_roi["Thickness"],
            )

        def find_closest_ROI_tag():
            self.closest_roi, self.closest_tag, self.closest_dist = {}, {}, np.inf
            merged_out = self.out_rectangles + self.out_circles + self.out_polygon
            for s in merged_out:
                for t in s["Tags"]:
                    dist = int(
                        (
                            np.sqrt(
                                (self.click_loc[0] - s["Tags"][t][0]) ** 2
                                + (self.click_loc[1] - s["Tags"][t][1]) ** 2
                            )
                        )
                    )
                    if ((not self.closest_roi) and (dist < self.click_sens)) or (
                        dist < self.closest_dist
                    ):
                        self.closest_roi, self.closest_tag, self.closest_dist = (
                            s,
                            t,
                            dist,
                        )
            if self.closest_dist is not np.inf:
                recolor_roi_tags()

        def check_if_click_is_tag():
            self.click_roi, self.click_tag, self.click_dist = {}, {}, -np.inf
            merged_out = self.out_rectangles + self.out_circles + self.out_polygon
            merged_out[:] = [
                d for d in merged_out if d.get("Name") != self.closest_roi["Name"]
            ]
            for shape in merged_out:
                for tag in shape["Tags"]:
                    distance = int(
                        (
                            np.sqrt(
                                (self.click_loc[0] - shape["Tags"][tag][0]) ** 2
                                + (self.click_loc[1] - shape["Tags"][tag][1]) ** 2
                            )
                        )
                    )
                    if ((not self.click_roi) and (distance < self.click_sens)) or (
                        distance < self.click_dist
                    ):
                        self.click_roi, self.click_tag, self.closest_dist = (
                            shape,
                            tag,
                            distance,
                        )

        if interact_method == "zoom_home":
            self.zoomed_img = self.orig_frame
            self.current_zoom = 100

        if interact_method == "move_shape":
            self.insert_all_ROIs_into_image(ROI_ear_tags=True)
            # initiate_x_y_callback()  # get x y loc ROI tag
            find_closest_ROI_tag()  # find closest ROI tag to x y loc
            # initiate_x_y_callback()  # get x y loc for new x y loc
            check_if_click_is_tag()
            if self.click_roi:
                move_edge_align(
                    self.closest_roi, self.closest_tag, self.click_roi, self.click_tag
                )
            else:
                move_edge(self.closest_roi, self.closest_tag, self.click_loc)
            self.insert_all_ROIs_into_image()  # re-insert ROIs

        if (interact_method == "zoom_in") or (interact_method == "zoom_out"):
            zoom_in(self, zoom_val)

        if interact_method == None:
            self.insert_all_ROIs_into_image(ROI_ear_tags=False)

    def remove_ROI(self, roi_to_delete):
        if roi_to_delete.startswith("Rectangle"):
            rectangle_name = roi_to_delete.split("Rectangle: ")[1]
            self.out_rectangles[:] = [
                d for d in self.out_rectangles if d.get("Name") != rectangle_name
            ]
        if roi_to_delete.startswith("Circle"):
            circle_name = roi_to_delete.split("Circle: ")[1]
            self.out_circles[:] = [
                d for d in self.out_circles if d.get("Name") != circle_name
            ]
        if roi_to_delete.startswith("Polygon"):
            polygon_name = roi_to_delete.split("Polygon: ")[1]
            self.out_polygon[:] = [
                d for d in self.out_polygon if d.get("Name") != polygon_name
            ]
        self.insert_all_ROIs_into_image()

    def insert_all_ROIs_into_image(
        self,
        ROI_ear_tags=False,
        change_frame_no=False,
        show_zoomed_img=False,
        show_size_info=False,
    ):
        self.no_shapes = 0
        if (change_frame_no is False) and (show_zoomed_img is False):
            self.working_frame = deepcopy(self.orig_frame)

        for e in self.out_rectangles:
            self.no_shapes += 1
            if int(e["Thickness"]) == 1:
                cv2.rectangle(
                    self.working_frame,
                    (e["topLeftX"], e["topLeftY"]),
                    (e["topLeftX"] + e["width"], e["topLeftY"] + e["height"]),
                    self.colors[e["Color name"]],
                    int(e["Thickness"]),
                    lineType=4,
                )
            else:
                cv2.rectangle(
                    self.working_frame,
                    (e["topLeftX"], e["topLeftY"]),
                    (e["topLeftX"] + e["width"], e["topLeftY"] + e["height"]),
                    self.colors[e["Color name"]],
                    int(e["Thickness"]),
                    lineType=self.line_type,
                )
            if ROI_ear_tags is True:
                for t in e["Tags"]:
                    cv2.circle(
                        self.working_frame,
                        e["Tags"][t],
                        e["Ear_tag_size"],
                        self.colors[e["Color name"]],
                        -1,
                    )
            if show_size_info is True:
                area_cm = self.rectangle_size_dict["Rectangles"][e["Name"]]["area_cm"]
                width_cm = self.rectangle_size_dict["Rectangles"][e["Name"]]["width_cm"]
                height_cm = self.rectangle_size_dict["Rectangles"][e["Name"]][
                    "height_cm"
                ]
                cv2.putText(
                    self.working_frame,
                    str(height_cm),
                    (
                        int(e["Tags"]["Left tag"][0] + e["Thickness"]),
                        e["Tags"]["Left tag"][1],
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.text_size / 10,
                    self.colors[e["Color name"]],
                    self.text_thickness,
                    lineType=self.line_type,
                )
                cv2.putText(
                    self.working_frame,
                    str(width_cm),
                    (
                        e["Tags"]["Bottom tag"][0],
                        int(e["Tags"]["Bottom tag"][1] - e["Thickness"]),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.text_size / 10,
                    self.colors[e["Color name"]],
                    self.text_thickness,
                    lineType=self.line_type,
                )
                cv2.putText(
                    self.working_frame,
                    str(area_cm),
                    (e["Tags"]["Center tag"][0], e["Tags"]["Center tag"][1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.text_size / 10,
                    self.colors[e["Color name"]],
                    self.text_thickness,
                    lineType=self.line_type,
                )

        for c in self.out_circles:
            self.no_shapes += 1
            cv2.circle(
                self.working_frame,
                (c["centerX"], c["centerY"]),
                c["radius"],
                c["Color BGR"],
                int(c["Thickness"]),
                lineType=self.line_type,
            )
            if ROI_ear_tags is True:
                for t in c["Tags"]:
                    cv2.circle(
                        self.working_frame,
                        c["Tags"][t],
                        c["Ear_tag_size"],
                        self.colors[c["Color name"]],
                        -1,
                    )
            if show_size_info is True:
                area_cm = self.circle_size_dict["Circles"][c["Name"]]["area_cm"]
                cv2.putText(
                    self.working_frame,
                    str(area_cm),
                    (c["Tags"]["Center tag"][0], c["Tags"]["Center tag"][1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.text_size / 10,
                    self.colors[c["Color name"]],
                    self.text_thickness,
                    lineType=self.line_type,
                )

        for pg in self.out_polygon:
            self.no_shapes += 1
            pts = np.array(pg["vertices"]).reshape((-1, 1, 2))
            cv2.polylines(
                self.working_frame,
                [pts],
                True,
                pg["Color BGR"],
                int(pg["Thickness"]),
                lineType=self.line_type,
            )
            if ROI_ear_tags is True:
                for p in pg["Tags"]:
                    cv2.circle(
                        self.working_frame,
                        pg["Tags"][p],
                        pg["Ear_tag_size"],
                        self.colors[pg["Color name"]],
                        -1,
                    )
            if show_size_info is True:
                area_cm = self.polygon_size_dict["Polygons"][pg["Name"]]["area_cm"]
                cv2.putText(
                    self.working_frame,
                    str(area_cm),
                    (pg["Center_X"], pg["Center_Y"]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.text_size / 10,
                    self.colors[pg["Color name"]],
                    self.text_thickness,
                    lineType=self.line_type,
                )

    def destroy_windows(self):
        cv2.destroyWindow(self.window_name)

    def display_image(self):
        self.stop_display_image = False
        while True:
            if self.stop_display_image:
                cv2.destroyAllWindows()
            else:
                if self.working_frame is None:
                    time.sleep(1)
                    continue
                with self.lock:
                    if self.draw_rectangle_roi:
                        self.user_rectangle_roi = cv2.selectROI(
                            self.window_name, self.working_frame
                        )
                    else:
                        cv2.imshow(self.window_name, self.working_frame)
                cv2.setMouseCallback(
                    self.window_name, self.display_image_imshow_callback
                )
                c = cv2.waitKey(self.keyWaitTime)
                if c != -1:
                    self.keyWaitChar = c
