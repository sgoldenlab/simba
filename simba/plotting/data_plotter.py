__author__ = "Simon Nilsson"

import os
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from simba.data_processors.movement_calculator import MovementCalculator
from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_if_keys_exist_in_dict, check_valid_lst)
from simba.utils.enums import Formats
from simba.utils.errors import NoSpecifiedOutputError
from simba.utils.lookups import get_color_dict
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext

BG_COLOR = "bg_color"
HEADER_COLOR = "header_color"
FONT_THICKNESS = "font_thickness"
SIZE = "size"
DATA_ACCURACY = "data_accuracy"
STYLE_KEYS = [BG_COLOR, HEADER_COLOR, FONT_THICKNESS, SIZE, DATA_ACCURACY]


class DataPlotter(ConfigReader):
    """
    Tabular data visualization of animal movement and distances in the current frame and their aggregate
    statistics.

    :parameter str config_path: path to SimBA project config file in Configparser format

    .. note::
        `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#visualizing-data-tables`>_.

    .. image:: _static/img/data_plot.png
       :width: 300
       :align: center

    :examples:
    >>> _ = DataPlotter(config_path='MyConfigPath').run()
    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        style_attr: Dict[str, Any],
        body_part_attr: List[List[str]],
        data_paths: List[str],
        video_setting: Optional[bool] = True,
        frame_setting: Optional[bool] = True,
    ):

        if (not video_setting) and (not frame_setting):
            raise NoSpecifiedOutputError(
                msg="SIMBA ERROR: Please choose to create video and/or frames data plots. SimBA found that you ticked neither video and/or frames"
            )
        check_valid_lst(
            data=data_paths, source=self.__class__.__name__, valid_dtypes=(str,)
        )
        check_valid_lst(
            data=body_part_attr, source=self.__class__.__name__, valid_dtypes=(list,)
        )
        for i in body_part_attr:
            check_valid_lst(
                data=i, source=self.__class__.__name__, valid_dtypes=(str,), exact_len=2
            )
        check_if_keys_exist_in_dict(data=style_attr, key=STYLE_KEYS)
        ConfigReader.__init__(self, config_path=config_path)
        self.video_setting, self.frame_setting = video_setting, frame_setting
        self.files_found, self.style_attr, self.body_part_attr = (
            data_paths,
            style_attr,
            body_part_attr,
        )
        if not os.path.exists(self.data_table_path):
            os.makedirs(self.data_table_path)
        self.process_movement()
        print(f"Processing {len(self.files_found)} video(s)...")

    def __compute_spacings(self):
        """
        Private helper to compute appropriate spacing between printed text.
        """
        self.loc_dict = {}
        self.loc_dict["Animal"] = (50, 20)
        self.loc_dict["total_movement_header"] = (250, 20)
        self.loc_dict["current_velocity_header"] = (475, 20)
        self.loc_dict["animals"] = {}
        y_cord, x_cord = 75, 15
        for animal_cnt, animal_name in enumerate(self.video_data.columns):
            self.loc_dict["animals"][animal_name] = {}
            self.loc_dict["animals"][animal_name]["index_loc"] = (50, y_cord)
            self.loc_dict["animals"][animal_name]["total_movement_loc"] = (250, y_cord)
            self.loc_dict["animals"][animal_name]["current_velocity_loc"] = (
                475,
                y_cord,
            )
            y_cord += 50

    def process_movement(self):
        movement_processor = MovementCalculator(
            config_path=self.config_path,
            file_paths=self.files_found,
            threshold=0.00,
            body_parts=[x[0] for x in self.body_part_attr],
        )
        movement_processor.run()
        self.movement = movement_processor.movement_dfs

    def run(self):
        def multiprocess_img_creation(
            video_data_slice: list,
            location_dict: dict,
            animal_ids: list,
            video_data: pd.DataFrame,
            style_attr: dict,
            body_part_attr: dict,
        ):
            color_dict = get_color_dict()
            img = np.zeros((style_attr["size"][1], style_attr["size"][0], 3))
            img[:] = color_dict[style_attr["bg_color"]]
            cv2.putText(
                img,
                "Animal",
                location_dict["Animal"],
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                color_dict[style_attr["header_color"]],
                style_attr["font_thickness"],
            )
            cv2.putText(
                img,
                "Total movement (cm)",
                location_dict["total_movement_header"],
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                color_dict[style_attr["header_color"]],
                style_attr["font_thickness"],
            )
            cv2.putText(
                img,
                "Velocity (cm/s)",
                location_dict["current_velocity_header"],
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                color_dict[style_attr["header_color"]],
                style_attr["font_thickness"],
            )
            for animal_cnt, animal_name in enumerate(animal_ids):
                clr = color_dict[body_part_attr[animal_cnt][1]]
                total_movement = str(
                    round(
                        video_data[animal_name]
                        .iloc[0 : video_data_slice.index.max()]
                        .sum()
                        / 10,
                        style_attr["data_accuracy"],
                    )
                )
                current_velocity = str(
                    round(
                        video_data_slice[animal_name].sum() / 10,
                        style_attr["data_accuracy"],
                    )
                )
                cv2.putText(
                    img,
                    animal_name,
                    location_dict["animals"][animal_name]["index_loc"],
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    clr,
                    1,
                )
                cv2.putText(
                    img,
                    total_movement,
                    location_dict["animals"][animal_name]["total_movement_loc"],
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    clr,
                    1,
                )
                cv2.putText(
                    img,
                    current_velocity,
                    location_dict["animals"][animal_name]["current_velocity_loc"],
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    clr,
                    1,
                )
            return img

        check_all_file_names_are_represented_in_video_log(
            video_info_df=self.video_info_df, data_paths=self.files_found
        )
        for file_cnt, file_path in enumerate(self.files_found):
            video_timer = SimbaTimer(start=True)
            _, video_name, _ = get_fn_ext(file_path)
            self.video_data = pd.DataFrame(self.movement[video_name])
            self.__compute_spacings()
            _, _, self.fps = self.read_video_info(video_name=video_name)
            if self.video_setting:
                self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
                self.video_save_path = os.path.join(
                    self.data_table_path, f"{video_name}.mp4"
                )
                self.writer = cv2.VideoWriter(
                    self.video_save_path, self.fourcc, self.fps, self.style_attr["size"]
                )
            if self.frame_setting:
                self.frame_save_path = os.path.join(self.data_table_path, video_name)
                if not os.path.exists(self.frame_save_path):
                    os.makedirs(self.frame_save_path)
            video_data_lst = np.array_split(
                pd.DataFrame(self.video_data), int(len(self.video_data) / self.fps)
            )
            self.imgs = Parallel(
                n_jobs=self.cpu_to_use, verbose=1, backend="threading"
            )(
                delayed(multiprocess_img_creation)(
                    x,
                    self.loc_dict,
                    self.video_data.columns,
                    self.video_data,
                    self.style_attr,
                    self.body_part_attr,
                )
                for x in video_data_lst
            )
            frm_cnt = 0
            for img_cnt, img in enumerate(self.imgs):
                for frame_cnt in range(int(self.fps)):
                    if self.video_setting:
                        self.writer.write(np.uint8(img))
                    if self.frame_setting:
                        frm_save_name = os.path.join(
                            self.frame_save_path, "{}.png".format(str(frm_cnt))
                        )
                        cv2.imwrite(frm_save_name, np.uint8(img))
                    frm_cnt += 1
                    print(
                        "Frame: {} / {}. Video: {} ({}/{})".format(
                            str(frm_cnt),
                            str(len(self.video_data)),
                            video_name,
                            str(file_cnt + 1),
                            len(self.files_found),
                        )
                    )

            print("Data tables created for video {}...".format(video_name))
            if self.video_setting:
                self.writer.release()
                video_timer.stop_timer()
                print(
                    "Video {} complete (elapsed time {}s)...".format(
                        video_name, video_timer.elapsed_time_str
                    )
                )

        self.timer.stop_timer()
        stdout_success(
            msg=f"All data table videos created inside {self.data_table_path}",
            elapsed_time=self.timer.elapsed_time_str,
        )


# style_attr = {'bg_color': 'White', 'header_color': 'Black', 'font_thickness': 1, 'size': (640, 480), 'data_accuracy': 2}
# body_part_attr = [['Ear_left_1', 'Green'], ['Ear_right_2', 'Red']]
# data_paths = ['/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/csv/machine_results/Trial    10.csv']
#
#
# test = DataPlotter(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/project_config.ini',
#                    style_attr=style_attr,
#                    body_part_attr=body_part_attr,
#                    data_paths=data_paths,
#                    video_setting=True,
#                    frame_setting=False)
# test.run()

# style_attr = {'bg_color': 'White', 'header_color': 'Black', 'font_thickness': 1, 'size': (640, 480), 'data_accuracy': 2}
# body_part_attr = [['Ear_left_1', 'Grey'], ['Ear_right_2', 'Red']]
# data_paths = ['/Users/simon/Desktop/envs/simba_dev/tests/test_data/two_C57_madlc/project_folder/csv/outlier_corrected_movement_location/Together_1.csv']
#
#
# test = DataPlotter(config_path='/Users/simon/Desktop/envs/simba_dev/tests/test_data/two_C57_madlc/project_folder/project_config.ini',
#                    style_attr=style_attr,
#                    body_part_attr=body_part_attr,
#                    data_paths=data_paths,
#                    video_setting=True,
#                    frame_setting=False)
# test.create_data_plots()
