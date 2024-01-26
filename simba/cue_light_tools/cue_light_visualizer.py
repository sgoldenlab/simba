__author__ = "Simon Nilsson"

import itertools
import os
from typing import List, Union

import cv2
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import check_file_exist_and_readable
from simba.utils.errors import NoROIDataError, NoSpecifiedOutputError
from simba.utils.printing import stdout_success
from simba.utils.read_write import get_fn_ext, get_video_meta_data, read_df


class CueLightVisualizer(ConfigReader):
    """

    Visualize SimBA computed cue-light ON and OFF states and the aggregate statistics of ON and OFF
    states.

    :parameter str config_path: path to SimBA project config file in Configparser format.
    :parameter List[str] cue_light_names: Names of cue lights, as defined in the SimBA ROI interface.
    :parameter str video_path: Path to video which user wants to create visualizations of cue light states and aggregate statistics for.
    :parameter bool frame_setting: If True, creates individual frames in png format.
    :parameter bool video_setting: If True, creates compressed videos in mp4 format.

    .. notes:
       `Cue light tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/cue_light_tutorial.md>`__.

    Examples
    ----------
    >>> cue_light_visualizer = CueLightVisualizer(config_path='SimBAConfig', cue_light_names=['Cue_light'], video_path='VideoPath', video_setting=True, frame_setting=False)
    >>> cue_light_visualizer.run()
    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        cue_light_names: List[str],
        video_path: str,
        frame_setting: bool,
        video_setting: bool,
    ):
        ConfigReader.__init__(self, config_path=config_path)

        if (not frame_setting) and (not video_setting):
            raise NoSpecifiedOutputError(
                msg="SIMBA ERROR: Please choose to select either videos, frames, or both frames and videos."
            )
        self.video_setting, self.frame_setting = video_setting, frame_setting
        self.in_dir = os.path.join(self.project_path, "csv", "cue_lights")
        self.cue_light_names, self.video_path = cue_light_names, video_path
        _, self.video_name, _ = get_fn_ext(video_path)
        self.video_meta_data = get_video_meta_data(self.video_path)
        self.logs_path, self.video_dir = os.path.join(
            self.project_path, "logs"
        ), os.path.join(self.project_path, "videos")
        self.data_file_path = os.path.join(
            self.in_dir, self.video_name + "." + self.file_type
        )
        check_file_exist_and_readable(self.data_file_path)
        self.data_df = read_df(self.data_file_path, self.file_type)
        self.output_folder = os.path.join(
            self.project_path, "frames", "output", "cue_lights"
        )
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.video_settings, pix_per_mm, self.fps = self.read_video_info(
            video_name=self.video_name
        )
        self.space_scale, radius_scale, res_scale, font_scale = 25, 10, 1500, 0.8
        max_dim = max(self.video_meta_data["width"], self.video_meta_data["height"])
        self.draw_scale, self.font_size = int(
            radius_scale / (res_scale / max_dim)
        ), float(font_scale / (res_scale / max_dim))
        self.spacing_scaler = int(self.space_scale / (res_scale / max_dim))
        self.font = cv2.FONT_HERSHEY_TRIPLEX
        self.__read_roi_dfs()

    def __update_video_meta_data(self):
        new_cap = cv2.VideoCapture(self.video_path)
        new_cap.set(1, 1)
        _, img = self.cap.read()
        bordered_img = cv2.copyMakeBorder(
            img,
            0,
            0,
            0,
            int(self.video_meta_data["width"]),
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )
        self.border_img_h, self.border_img_w = (
            bordered_img.shape[0],
            bordered_img.shape[1],
        )
        new_cap.release()

    def __read_roi_dfs(self):
        if not os.path.isfile(
            os.path.join(self.logs_path, "measures", "ROI_definitions.h5")
        ):
            raise NoROIDataError(
                msg="No ROI definitions were found in your SimBA project. Please draw some ROIs before analyzing your ROI data"
            )
        else:
            self.roi_h5_path = os.path.join(
                self.logs_path, "measures", "ROI_definitions.h5"
            )
            self.rectangles_df = pd.read_hdf(self.roi_h5_path, key="rectangles")
            self.circles_df = pd.read_hdf(self.roi_h5_path, key="circleDf")
            self.polygon_df = pd.read_hdf(self.roi_h5_path, key="polygons")
            self.shape_names = list(
                itertools.chain(
                    self.rectangles_df["Name"].unique(),
                    self.circles_df["Name"].unique(),
                    self.polygon_df["Name"].unique(),
                )
            )
            self.video_recs = self.rectangles_df.loc[
                (self.rectangles_df["Video"] == self.video_name)
                & (self.rectangles_df["Name"].isin(self.cue_light_names))
            ]
            self.video_circs = self.circles_df.loc[
                (self.circles_df["Video"] == self.video_name)
                & (self.circles_df["Name"].isin(self.cue_light_names))
            ]
            self.video_polys = self.polygon_df.loc[
                (self.polygon_df["Video"] == self.video_name)
                & (self.polygon_df["Name"].isin(self.cue_light_names))
            ]
            self.shape_names = list(
                itertools.chain(
                    self.rectangles_df["Name"].unique(),
                    self.circles_df["Name"].unique(),
                    self.polygon_df["Name"].unique(),
                )
            )

    def __calc_text_locs(self):
        add_spacer = 2
        self.loc_dict = {}
        for light_cnt, light_name in enumerate(self.cue_light_names):
            self.loc_dict[light_name] = {}
            self.loc_dict[light_name]["status_text"] = "{} {}".format(
                light_name, "status:"
            )
            self.loc_dict[light_name]["onset_cnt_text"] = "{} {}".format(
                light_name, "onset counts:"
            )
            self.loc_dict[light_name]["seconds_on_text"] = "{} {}".format(
                light_name, "time ON (s):"
            )
            self.loc_dict[light_name]["seconds_off_text"] = "{} {}".format(
                light_name, "time OFF (s):"
            )
            self.loc_dict[light_name]["status_text_loc"] = (
                (self.video_meta_data["width"] + 5),
                (
                    self.video_meta_data["height"]
                    - (self.video_meta_data["height"] + 10)
                    + self.spacing_scaler * add_spacer
                ),
            )
            self.loc_dict[light_name]["status_data_loc"] = (
                int(self.border_img_w - (self.border_img_w / 8)),
                (
                    self.video_meta_data["height"]
                    - (self.video_meta_data["height"] + 10)
                    + self.spacing_scaler * add_spacer
                ),
            )
            add_spacer += 1
            self.loc_dict[light_name]["onset_cnt_text_loc"] = (
                (self.video_meta_data["width"] + 5),
                (
                    self.video_meta_data["height"]
                    - (self.video_meta_data["height"] + 10)
                    + self.spacing_scaler * add_spacer
                ),
            )
            self.loc_dict[light_name]["onset_cnt_data_loc"] = (
                int(self.border_img_w - (self.border_img_w / 8)),
                (
                    self.video_meta_data["height"]
                    - (self.video_meta_data["height"] + 10)
                    + self.spacing_scaler * add_spacer
                ),
            )
            add_spacer += 1
            self.loc_dict[light_name]["seconds_on_text_loc"] = (
                (self.video_meta_data["width"] + 5),
                (
                    self.video_meta_data["height"]
                    - (self.video_meta_data["height"] + 10)
                    + self.spacing_scaler * add_spacer
                ),
            )
            self.loc_dict[light_name]["seconds_on_data_loc"] = (
                int(self.border_img_w - (self.border_img_w / 8)),
                (
                    self.video_meta_data["height"]
                    - (self.video_meta_data["height"] + 10)
                    + self.spacing_scaler * add_spacer
                ),
            )
            add_spacer += 1
            self.loc_dict[light_name]["seconds_off_text_loc"] = (
                (self.video_meta_data["width"] + 5),
                (
                    self.video_meta_data["height"]
                    - (self.video_meta_data["height"] + 10)
                    + self.spacing_scaler * add_spacer
                ),
            )
            self.loc_dict[light_name]["seconds_off_data_loc"] = (
                int(self.border_img_w - (self.border_img_w / 8)),
                (
                    self.video_meta_data["height"]
                    - (self.video_meta_data["height"] + 10)
                    + self.spacing_scaler * add_spacer
                ),
            )
            add_spacer += 1

    def __create_text_dict(self):
        self.light_dict = {}
        for light_cnt, light_name in enumerate(self.cue_light_names):
            self.light_dict[light_name] = {}
            self.light_dict[light_name]["status"] = False
            self.light_dict[light_name]["onsets"] = 0
            self.light_dict[light_name]["time_on"] = 0
            self.light_dict[light_name]["time_off"] = 0
            self.light_dict[light_name]["prior_frame_status"] = 0
            self.light_dict[light_name]["color"] = (0, 0, 0)

    def __draw_shapes_and_text(self, shape_data):
        shape_name = shape_data["Name"]
        cv2.putText(
            self.border_img,
            self.loc_dict[shape_name]["status_text"],
            self.loc_dict[shape_name]["status_text_loc"],
            self.font,
            self.font_size,
            shape_data["Color BGR"],
            1,
        )
        cv2.putText(
            self.border_img,
            self.loc_dict[shape_name]["onset_cnt_text"],
            self.loc_dict[shape_name]["onset_cnt_text_loc"],
            self.font,
            self.font_size,
            shape_data["Color BGR"],
            1,
        )
        cv2.putText(
            self.border_img,
            self.loc_dict[shape_name]["seconds_on_text"],
            self.loc_dict[shape_name]["seconds_on_text_loc"],
            self.font,
            self.font_size,
            shape_data["Color BGR"],
            1,
        )
        cv2.putText(
            self.border_img,
            self.loc_dict[shape_name]["seconds_off_text"],
            self.loc_dict[shape_name]["seconds_off_text_loc"],
            self.font,
            self.font_size,
            shape_data["Color BGR"],
            1,
        )
        if shape_data["Shape_type"] == "Rectangle":
            cv2.rectangle(
                self.border_img,
                (shape_data["topLeftX"], shape_data["topLeftY"]),
                (shape_data["Bottom_right_X"], shape_data["Bottom_right_Y"]),
                shape_data["Color BGR"],
                shape_data["Thickness"],
            )
        if shape_data["Shape_type"] == "Circle":
            cv2.circle(
                self.border_img,
                (shape_data["centerX"], shape_data["centerY"]),
                shape_data["radius"],
                shape_data["Color BGR"],
                shape_data["Thickness"],
            )
        if shape_data["Shape_type"] == "Polygon":
            cv2.polylines(
                self.border_img,
                shape_data["vertices"],
                True,
                shape_data["Color BGR"],
                thickness=shape_data["Thickness"],
            )

    def __insert_texts_and_shapes(self):
        for light_cnt, light_name in enumerate(self.cue_light_names):
            for i, r in self.video_recs.iterrows():
                if light_name == r["Name"]:
                    self.__draw_shapes_and_text(shape_data=r)
            for i, r in self.video_circs.iterrows():
                if light_name == r["Name"]:
                    self.__draw_shapes_and_text(shape_data=r)
            for i, r in self.video_polys.iterrows():
                if light_name == r["Name"]:
                    self.__draw_shapes_and_text(shape_data=r)

    def __insert_body_parts(self):
        for animal_name, animal_data in self.animal_bp_dict.items():
            for cnt, (x_bp, y_bp) in enumerate(
                zip(animal_data["X_bps"], animal_data["Y_bps"])
            ):
                cord = tuple(
                    self.data_df.loc[self.frame_cnt, [x_bp, y_bp]].astype(int).values
                )
                cv2.circle(
                    self.border_img,
                    cord,
                    0,
                    animal_data["colors"][cnt],
                    self.draw_scale,
                )

    def run(self):
        """
        Method to create cue light visualizations. Results are stored in the ``project_folder/frames/output/cue_lights``
        directory of the SimBA project.
        """
        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_cnt = 0
        self.__update_video_meta_data()
        if self.video_setting:
            self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.save_video_path = os.path.join(
                self.output_folder, self.video_name + ".mp4"
            )
            self.writer = cv2.VideoWriter(
                self.save_video_path,
                self.fourcc,
                self.fps,
                (self.border_img_w, self.border_img_h),
            )
        if self.frame_setting:
            self.save_frame_folder_dir = os.path.join(
                self.output_folder, self.video_name
            )
            if not os.path.exists(self.save_frame_folder_dir):
                os.makedirs(self.save_frame_folder_dir)
        self.__calc_text_locs()
        self.__create_text_dict()
        while self.cap.isOpened():
            try:
                _, img = self.cap.read()
                self.border_img = cv2.copyMakeBorder(
                    img,
                    0,
                    0,
                    0,
                    int(self.video_meta_data["width"]),
                    borderType=cv2.BORDER_CONSTANT,
                    value=[0, 0, 0],
                )
                self.border_img_h, self.border_img_w = (
                    self.border_img.shape[0],
                    self.border_img.shape[1],
                )
                self.__insert_texts_and_shapes()
                self.__insert_body_parts()
                for light_name in self.cue_light_names:
                    if (self.light_dict[light_name]["prior_frame_status"] == 0) & (
                        self.data_df.loc[self.frame_cnt, light_name] == 1
                    ):
                        self.light_dict[light_name]["onsets"] += 1
                    if self.data_df.loc[self.frame_cnt, light_name] == 1:
                        self.light_dict[light_name]["color"] = (0, 255, 255)
                        self.light_dict[light_name]["status"] = "ON"
                        self.light_dict[light_name]["time_on"] += 1 / self.fps
                    else:
                        self.light_dict[light_name]["color"] = (90, 10, 10)
                        self.light_dict[light_name]["status"] = "OFF"
                        self.light_dict[light_name]["time_off"] += 1 / self.fps
                    self.light_dict[light_name]["prior_frame_status"] = (
                        self.data_df.loc[self.frame_cnt, light_name]
                    )
                    cv2.putText(
                        self.border_img,
                        str(self.light_dict[light_name]["status"]),
                        self.loc_dict[light_name]["status_data_loc"],
                        self.font,
                        self.font_size,
                        self.light_dict[light_name]["color"],
                        1,
                    )
                    cv2.putText(
                        self.border_img,
                        str(self.light_dict[light_name]["onsets"]),
                        self.loc_dict[light_name]["onset_cnt_data_loc"],
                        self.font,
                        self.font_size,
                        (255, 255, 255),
                        1,
                    )
                    cv2.putText(
                        self.border_img,
                        str(round(self.light_dict[light_name]["time_on"], 2)),
                        self.loc_dict[light_name]["seconds_on_data_loc"],
                        self.font,
                        self.font_size,
                        (255, 255, 255),
                        1,
                    )
                    cv2.putText(
                        self.border_img,
                        str(round(self.light_dict[light_name]["time_off"], 2)),
                        self.loc_dict[light_name]["seconds_off_data_loc"],
                        self.font,
                        self.font_size,
                        (255, 255, 255),
                        1,
                    )
                if self.video_setting:
                    self.writer.write(self.border_img)
                if self.frame_setting:
                    frame_save_path = os.path.join(
                        self.save_frame_folder_dir, str(self.frame_cnt) + ".png"
                    )
                    cv2.imwrite(frame_save_path, self.border_img)
                print(
                    "Cue light frame: {} / {}. Video: {} ".format(
                        str(self.frame_cnt + 1), str(len(self.data_df)), self.video_name
                    )
                )
                self.frame_cnt += 1

            except Exception as e:
                if self.video_setting:
                    self.writer.release()
                print(e.args)
                print(
                    "NOTE: index error / keyerror. Some frames of the video may be missing. Make sure you are running latest version of SimBA with pip install simba-uw-tf-dev"
                )
                break

        if self.video_setting:
            self.writer.release()
        stdout_success(
            msg=f"Cue light visualization for video {self.video_name} saved..."
        )


# test = CueLightVisualizer(config_path='/Users/simon/Desktop/troubleshooting/light_analyzer/project_folder/project_config.ini',
#                           cue_light_names=['Cue_light'],
#                           video_path='/Users/simon/Desktop/troubleshooting/light_analyzer/project_folder/videos/20220422_ALMEAG02_B0.avi',
#                           video_setting=True,
#                           frame_setting=False)
# test.visualize_cue_light_data()
