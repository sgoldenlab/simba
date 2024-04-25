__author__ = "Simon Nilsson"

import functools
import multiprocessing
import os
import platform
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from simba.data_processors.directing_other_animals_calculator import \
    DirectingOtherAnimalsAnalyzer
from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_keys_exist_in_dict,
                                check_if_valid_rgb_tuple, check_int,
                                check_valid_lst,
                                check_video_and_data_frm_count_align)
from simba.utils.data import create_color_palettes
from simba.utils.enums import Formats, TextOptions
from simba.utils.errors import AnimalNumberError, NoFilesFoundError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    find_core_cnt, get_fn_ext,
                                    get_video_meta_data, read_df)
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


def _directing_animals_mp(
    frm_range: Tuple[int, np.ndarray],
    directionality_data: pd.DataFrame,
    pose_data: pd.DataFrame,
    style_attr: dict,
    animal_bp_dict: dict,
    save_temp_dir: str,
    video_path: str,
    video_meta_data: dict,
    colors: list,
):
    batch = frm_range[0]
    fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
    start_frm, current_frm, end_frm = frm_range[1][0], frm_range[1][0], frm_range[1][-1]
    save_path = os.path.join(save_temp_dir, f"{batch}.mp4")
    writer = cv2.VideoWriter(
        save_path,
        fourcc,
        video_meta_data["fps"],
        (video_meta_data["width"], video_meta_data["height"]),
    )
    cap = cv2.VideoCapture(video_path)
    cap.set(1, start_frm)
    while current_frm <= end_frm:
        ret, img = cap.read()
        if ret:
            frm_data = pose_data.iloc[current_frm]
            if style_attr[SHOW_POSE]:
                for animal_cnt, (animal_name, animal_bps) in enumerate(
                    animal_bp_dict.items()
                ):
                    for bp_cnt, bp in enumerate(
                        zip(animal_bps["X_bps"], animal_bps["Y_bps"])
                    ):
                        x_bp, y_bp = frm_data[bp[0]], frm_data[bp[1]]
                        cv2.circle(
                            img,
                            (int(x_bp), int(y_bp)),
                            style_attr[CIRCLE_SIZE],
                            animal_bp_dict[animal_name]["colors"][bp_cnt],
                            -1,
                        )
            if style_attr[ANIMAL_NAMES]:
                for animal_name, bp_data in animal_bp_dict.items():
                    headers = [bp_data["X_bps"][-1], bp_data["Y_bps"][-1]]
                    bp_cords = pose_data.loc[current_frm, headers].values.astype(
                        np.int64
                    )
                    cv2.putText(
                        img,
                        animal_name,
                        (bp_cords[0], bp_cords[1]),
                        TextOptions.FONT.value,
                        2,
                        animal_bp_dict[animal_name]["colors"][0],
                        1,
                    )

            if current_frm in list(directionality_data["Frame_#"].unique()):
                img_data = directionality_data[
                    directionality_data["Frame_#"] == current_frm
                ]
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
                        color=tuple(colors[animal_name]),
                        highlight_endpoint=style_attr[HIGHLIGHT_ENDPOINTS],
                        thickness=style_attr[DIRECTION_THICKNESS],
                        circle_size=style_attr[CIRCLE_SIZE],
                    )
            current_frm += 1
            writer.write(np.uint8(img))
            print(
                f"Frame: {current_frm} / {video_meta_data['frame_count']}. Core batch: {batch}"
            )

        else:
            break

    writer.release()
    return batch


class DirectingOtherAnimalsVisualizerMultiprocess(ConfigReader, PlottingMixin):
    """
    Class for visualizing when animals are directing towards body-parts of other animals using multiprocessing.

    .. important::
       Requires the pose-estimation data for the left ear, right ears and nose of individual animals.

    .. note::
        Example of expected output https://www.youtube.com/watch?v=d6pAatreb1E&list=PLi5Vwf0hhy1R6NDQJ3U28MOUJPfl2YWYl&index=22

        `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#visualizing-data-tables`>_.

    .. image:: _static/img/directing_other_animals.png
       :width: 450
       :align: center

    :parameter Union[str, os.PathLike] config_path: path to SimBA project config file in Configparser format
    :parameter Union[str, os.PathLike] video_path: Path to video for to visualize directionality.
    :parameter dict style_attr: Video style attribitions (colors and sizes etc.)
    :parameter Optional[int] core_cnt: How many cores to use to create the video. Deafult -1 which is all the cores.

    :examples:
    >>> style_attr = {'show_pose': True, 'animal_names': False, 'circle_size': 3, 'directionality_color': [(255, 0, 0), (0, 0, 255)], 'direction_thickness': 10, 'highlight_endpoints': True}
    >>> test = DirectingOtherAnimalsVisualizerMultiprocess(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini', video_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/videos/Together_1.avi', style_attr=style_attr, core_cnt=-1)
    >>> test.run()
    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        video_path: Union[str, os.PathLike],
        style_attr: Dict[str, Any],
        core_cnt: Optional[int] = -1,
    ):

        if platform.system() == "Darwin":
            multiprocessing.set_start_method("spawn", force=True)
        check_file_exist_and_readable(file_path=video_path)
        check_file_exist_and_readable(file_path=config_path)
        check_if_keys_exist_in_dict(
            data=style_attr,
            key=STYLE_ATTR,
            name=f"{self.__class__.__name__} style_attr",
        )
        check_int(
            name=f"{self.__class__.__name__} core_cnt",
            value=core_cnt,
            min_value=-1,
            max_value=find_core_cnt()[0],
        )
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]
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
        self.video_save_path = os.path.join(
            self.directing_animals_video_output_path, f"{self.video_name}.mp4"
        )
        if not os.path.exists(self.directing_animals_video_output_path):
            os.makedirs(self.directing_animals_video_output_path)
        self.save_path = os.path.join(
            self.directing_animals_video_output_path, self.video_name + ".mp4"
        )
        self.save_temp_path = os.path.join(
            self.directing_animals_video_output_path, "temp"
        )
        if os.path.exists(self.save_temp_path):
            self.remove_a_folder(folder_dir=self.save_temp_path)
        os.makedirs(self.save_temp_path)
        self.core_cnt, self.video_path = core_cnt, video_path
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
            frm_data = np.array_split(
                list(range(0, self.video_meta_data["frame_count"] + 1)), self.core_cnt
            )
            frm_ranges = []
            for i in range(len(frm_data)):
                frm_ranges.append((i, frm_data[i]))
            print(
                f"Creating directing images, multiprocessing (chunksize: {self.multiprocess_chunksize}, cores: {self.core_cnt})..."
            )
            with multiprocessing.Pool(
                self.core_cnt, maxtasksperchild=self.maxtasksperchild
            ) as pool:
                constants = functools.partial(
                    _directing_animals_mp,
                    directionality_data=video_data,
                    pose_data=self.data_df,
                    video_meta_data=self.video_meta_data,
                    style_attr=self.style_attr,
                    save_temp_dir=self.save_temp_path,
                    video_path=self.video_path,
                    animal_bp_dict=self.animal_bp_dict,
                    colors=self.direction_colors,
                )
                for cnt, result in enumerate(
                    pool.imap(
                        constants, frm_ranges, chunksize=self.multiprocess_chunksize
                    )
                ):
                    print(f"Core batch {result+1} complete...")
            print(f"Joining {self.video_name} multi-processed video...")
            concatenate_videos_in_folder(
                in_folder=self.save_temp_path,
                save_path=self.save_path,
                video_format="mp4",
                remove_splits=True,
            )
            self.timer.stop_timer()
            pool.terminate()
            pool.join()
            stdout_success(
                msg=f"Video {self.video_name} complete. Video saved in {self.directing_animals_video_output_path} directory",
                elapsed_time=self.timer.elapsed_time_str,
            )


# style_attr = {SHOW_POSE: True,
#               ANIMAL_NAMES: False,
#               CIRCLE_SIZE: 3,
#               DIRECTIONALITY_COLOR: [(255, 0, 0), (0, 0, 255)],
#               DIRECTION_THICKNESS: 10,
#               HIGHLIGHT_ENDPOINTS: True}
# test = DirectingOtherAnimalsVisualizerMultiprocess(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                                    video_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/videos/Together_1.avi',
#                                                    style_attr=style_attr,
#                                                    core_cnt=-1)
#
# test.run()


# style_attr = {'Show_pose': True,
#               'Pose_circle_size': 3,
#               "Direction_color": 'Random',
#               'Direction_thickness': 4,
#               'Highlight_endpoints': True,
#               'Polyfill': True}
# test = DirectingOtherAnimalsVisualizerMultiprocess(config_path='/Users/simon/Desktop/envs/troubleshooting/sleap_5_animals/project_folder/project_config.ini',
#                                                    data_path='/Users/simon/Desktop/envs/troubleshooting/sleap_5_animals/project_folder/csv/outlier_corrected_movement_location/Testing_Video_3.csv',
#                                                    style_attr=style_attr,
#                                                    core_cnt=5)
#
# test.run()


# style_attr = {'Show_pose': True, 'Pose_circle_size': 3, "Direction_color": 'Random', 'Direction_thickness': 4, 'Highlight_endpoints': True, 'Polyfill': True}
# test = DirectingOtherAnimalsVisualizerMultiprocess(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                        data_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/outlier_corrected_movement_location/Together_1.csv',
#                                        style_attr=style_attr,
#                                                    core_cnt=5)
#
# test.run()
