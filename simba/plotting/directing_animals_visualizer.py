__author__ = "Simon Nilsson"

import os
from typing import Any, Dict, Union

import cv2
import numpy as np

from simba.data_processors.directing_other_animals_calculator import \
    DirectingOtherAnimalsAnalyzer
from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_keys_exist_in_dict,
                                check_if_valid_rgb_tuple, check_valid_array,
                                check_valid_lst,
                                check_video_and_data_frm_count_align)
from simba.utils.data import create_color_palettes
from simba.utils.enums import Formats, TextOptions
from simba.utils.errors import AnimalNumberError, NoFilesFoundError
from simba.utils.printing import stdout_success
from simba.utils.read_write import get_fn_ext, get_video_meta_data, read_df
from simba.utils.warnings import NoDataFoundWarning

DIRECTION_THICKNESS = "direction_thickness"
DIRECTIONALITY_COLOR = "directionality_color"
CIRCLE_SIZE = "circle_size"
HIGHLIGHT_ENDPOINTS = "highlight_endpoints"
SHOW_POSE = "show_pose"
ANIMAL_NAMES = "animal_names"

STYLE_ATTR = [
    DIRECTION_THICKNESS,
    DIRECTIONALITY_COLOR,
    CIRCLE_SIZE,
    HIGHLIGHT_ENDPOINTS,
    SHOW_POSE,
    ANIMAL_NAMES,
]


class DirectingOtherAnimalsVisualizer(ConfigReader, PlottingMixin):
    """
    Visualize when animals are directing towards body-parts of other animals.

    .. important::
       Requires the pose-estimation data for the left ear, right ears and nose of individual animals.
       For better runtime, use :meth:`simba.plotting.Directing_animals_visualizer.DirectingOtherAnimalsVisualizerMultiprocess`.

    .. note::
       `Example of expected output <https://www.youtube.com/watch?v=d6pAatreb1E&list=PLi5Vwf0hhy1R6NDQJ3U28MOUJPfl2YWYl&index=22`_.

        `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#visualizing-data-tables`>_.

    .. image:: _static/img/directing_other_animals.png
       :width: 500
       :align: center

    .. image:: _static/img/DirectingOtherAnimalsVisualizer.png
       :width: 500
       :align: center

    :parameter Union[str, os.PathLike] config_path: path to SimBA project config file in Configparser format
    :parameter Union[str, os.PathLike] video_path: Path to video for to visualize directionality.
    :parameter Dict[str, Any] style_attr: Video style attributes (colors and sizes etc.)

    :example:
    >>> style_attr = {'show_pose': True, 'animal_names': True, 'circle_size': 3, 'directionality_color': [(255, 0, 0), (0, 0, 255)], 'direction_thickness': 10, 'highlight_endpoints': True}
    >>> test = DirectingOtherAnimalsVisualizer(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini', video_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/videos/Together_1.avi', style_attr=style_attr)
    >>> test.run()

    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        video_path: Union[str, os.PathLike],
        style_attr: Dict[str, Any],
    ):

        check_file_exist_and_readable(file_path=video_path)
        check_file_exist_and_readable(file_path=config_path)
        check_if_keys_exist_in_dict(
            data=style_attr,
            key=STYLE_ATTR,
            name=f"{self.__class__.__name__} style_attr",
        )
        ConfigReader.__init__(self, config_path=config_path)
        PlottingMixin.__init__(self)
        if self.animal_cnt < 2:
            raise AnimalNumberError(
                "Cannot analyze directionality between animals in a project with less than two animals.",
                source=self.__class__.__name__,
            )
        self.animal_names = [k for k in self.animal_bp_dict.keys()]
        _, self.video_name, _ = get_fn_ext(video_path)
        self.data_path = os.path.join(
            self.outlier_corrected_dir, f"{self.video_name}.{self.file_type}"
        )
        if not os.path.isfile(self.data_path):
            raise NoFilesFoundError(
                msg=f"SIMBA ERROR: Could not find the file at path {self.data_path}. Make sure the data file exist to create directionality visualizations",
                source=self.__class__.__name__,
            )
        self.direction_analyzer = DirectingOtherAnimalsAnalyzer(
            config_path=config_path,
            bool_tables=False,
            summary_tables=False,
            aggregate_statistics_tables=False,
            data_paths=self.data_path,
        )
        self.direction_analyzer.run()
        self.direction_analyzer.transpose_results()
        self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        self.style_attr = style_attr
        self.direction_colors = {}
        if isinstance(self.style_attr[DIRECTIONALITY_COLOR], list):
            check_valid_lst(
                data=self.style_attr[DIRECTIONALITY_COLOR],
                source=f"{self.__class__.__name__} colors",
                valid_dtypes=(tuple,),
                min_len=self.animal_cnt,
            )
            for i in range(len(self.animal_names)):
                check_if_valid_rgb_tuple(data=self.style_attr[DIRECTIONALITY_COLOR][i])
                self.direction_colors[self.animal_names[i]] = self.style_attr[
                    DIRECTIONALITY_COLOR
                ][i]
        if isinstance(self.style_attr[DIRECTIONALITY_COLOR], tuple):
            check_if_valid_rgb_tuple(self.style_attr[DIRECTIONALITY_COLOR])
            for i in range(len(self.animal_names)):
                self.direction_colors[self.animal_names[i]] = self.style_attr[
                    DIRECTIONALITY_COLOR
                ]
        else:
            self.random_colors = create_color_palettes(1, int(self.animal_cnt**2))[0]
            self.random_colors = [
                [int(item) for item in sublist] for sublist in self.random_colors
            ]
            for cnt in range(len(self.animal_names)):
                self.direction_colors[self.animal_names[cnt]] = self.random_colors[cnt]
        self.data_dict = self.direction_analyzer.directionality_df_dict
        if not os.path.exists(self.directing_animals_video_output_path):
            os.makedirs(self.directing_animals_video_output_path)
        self.data_df = read_df(self.data_path, file_type=self.file_type)
        self.video_save_path = os.path.join(
            self.directing_animals_video_output_path, f"{self.video_name}.mp4"
        )
        self.cap = cv2.VideoCapture(video_path)
        self.video_meta_data = get_video_meta_data(video_path)
        check_video_and_data_frm_count_align(
            video=video_path, data=self.data_path, name=video_path, raise_error=False
        )
        print(f"Processing video {self.video_name}...")

    def run(self):
        video_data = self.data_dict[self.video_name]
        self.writer = cv2.VideoWriter(
            self.video_save_path,
            self.fourcc,
            self.video_meta_data["fps"],
            (self.video_meta_data["width"], self.video_meta_data["height"]),
        )
        if len(video_data) < 1:
            NoDataFoundWarning(
                msg=f"SimBA skipping video {self.video_name}: No animals are directing each other in the video."
            )
        else:
            frm_cnt = 0
            while self.cap.isOpened():
                ret, img = self.cap.read()
                if ret:
                    bp_data = self.data_df.iloc[frm_cnt]
                    if self.style_attr[SHOW_POSE]:
                        for animal_cnt, (animal_name, animal_bps) in enumerate(
                            self.animal_bp_dict.items()
                        ):
                            for bp_cnt, bp in enumerate(
                                zip(animal_bps["X_bps"], animal_bps["Y_bps"])
                            ):
                                x_bp, y_bp = bp_data[bp[0]], bp_data[bp[1]]
                                cv2.circle(
                                    img,
                                    (int(x_bp), int(y_bp)),
                                    self.style_attr[CIRCLE_SIZE],
                                    self.animal_bp_dict[animal_name]["colors"][bp_cnt],
                                    -1,
                                )
                    if self.style_attr[ANIMAL_NAMES]:
                        for animal_name, bp_v in self.animal_bp_dict.items():
                            headers = [bp_v["X_bps"][-1], bp_v["Y_bps"][-1]]
                            bp_cords = self.data_df.loc[frm_cnt, headers].values.astype(
                                np.int64
                            )
                            cv2.putText(
                                img,
                                animal_name,
                                (bp_cords[0], bp_cords[1]),
                                TextOptions.FONT.value,
                                2,
                                self.animal_bp_dict[animal_name]["colors"][0],
                                1,
                            )

                    if frm_cnt in list(video_data["Frame_#"].unique()):
                        img_data = video_data[video_data["Frame_#"] == frm_cnt]
                        for animal_name in img_data["Animal_1"].unique():
                            animal_img_data = img_data[
                                img_data["Animal_1"] == animal_name
                            ].reset_index(drop=True)
                            img = PlottingMixin.draw_lines_on_img(
                                img=img,
                                start_positions=animal_img_data[
                                    ["Eye_x", "Eye_y"]
                                ].values.astype(np.int64),
                                end_positions=animal_img_data[
                                    ["Animal_2_bodypart_x", "Animal_2_bodypart_y"]
                                ].values.astype(np.int64),
                                color=tuple(self.direction_colors[animal_name]),
                                highlight_endpoint=self.style_attr[HIGHLIGHT_ENDPOINTS],
                                thickness=self.style_attr[DIRECTION_THICKNESS],
                                circle_size=self.style_attr[CIRCLE_SIZE],
                            )
                    frm_cnt += 1
                    self.writer.write(np.uint8(img))
                    print(
                        f"Frame: {frm_cnt} / {self.video_meta_data['frame_count']}. Video: {self.video_name}"
                    )
                else:
                    break
            self.writer.release()
            self.timer.stop_timer()
            stdout_success(
                msg=f"Directionality video {self.video_name} saved in {self.directing_animals_video_output_path} directory",
                elapsed_time=self.timer.elapsed_time_str,
            )


# style_attr = {SHOW_POSE: True,
#               ANIMAL_NAMES: True,
#               CIRCLE_SIZE: 10,
#               DIRECTIONALITY_COLOR: (0, 255, 0),
#               DIRECTION_THICKNESS: 10,
#               HIGHLIGHT_ENDPOINTS: True}
#
# test = DirectingOtherAnimalsVisualizer(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                        video_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/videos/Together_1.avi',
#                                        style_attr=style_attr)
# #
# test.run()


# style_attr = {'Show_pose': True, 'Pose_circle_size': 3, "Direction_color": 'Random', 'Direction_thickness': 4, 'Highlight_endpoints': True, 'Polyfill': True}
# test = DirectingOtherAnimalsVisualizer(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                        data_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/outlier_corrected_movement_location/Together_1.csv',
#                                        style_attr=style_attr)
#
# test.run()
