__author__ = "Simon Nilsson"

import functools
import multiprocessing
import os
import platform
from collections import deque
from copy import deepcopy
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from numba import jit, prange

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import check_if_filepath_list_is_empty
from simba.utils.data import find_frame_numbers_from_time_stamp
from simba.utils.errors import FrameRangeError, NoSpecifiedOutputError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder, get_fn_ext,
                                    read_df, remove_a_folder)


class PathPlotterMulticore(ConfigReader, PlottingMixin):
    """
    Class for creating "path plots" videos and/or images detailing the movement paths of
    individual animals in SimBA. Uses multiprocessing.

    :param str config_path: Path to SimBA project config file in Configparser format
    :param bool frame_setting: If True, individual frames will be created.
    :param bool video_setting: If True, compressed videos will be created.
    :param bool last_frame: If True, png of the last frame will be created.
    :param List[str] files_found: Data paths to create path plots for (e.g., ['project_folder/csv/machine_results/MyVideo.csv'])
    :param dict animal_attr: Animal body-parts to use when creating paths and their colors.
    :param Optional[dict] input_style_attr: Plot sttributes (line thickness, color, etc..). If None, then autocomputed. Max lines will be set to 2s.
    :param Optional[dict] input_clf_attr: Dict reprenting classified behavior locations, their color and size. If None, then no classified behavior locations are shown.
    :param Optional[dict] slicing: If Dict, start time and end time of video slice to create path plot from. E.g., {'start_time': '00:00:01', 'end_time': '00:00:03'}. If None, creates path plot for entire video.
    :param int cores: Number of cores to use.

    .. note::
       `Visualization tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-11-visualizations>`__.

    .. important::
       See :meth:`simba.utils.lookups.get_color_dict` for keys for named color options.

    .. image:: _static/img/path_plot.png
       :width: 300
       :align: center

    :example:
    >>> input_style_attr = {'width': 'As input', 'height': 'As input', 'line width': 5, 'font size': 5, 'font thickness': 2, 'circle size': 5, 'bg color': 'White', 'max lines': 100}
    >>> animal_attr = {0: ['Ear_right_1', 'Red']}
    >>> input_clf_attr = {0: ['Attack', 'Black', 'Size: 30'], 1: ['Sniffing', 'Red', 'Size: 30']}
    >>> path_plotter = PathPlotterMulticore(config_path=r'MyConfigPath', frame_setting=False, video_setting=True, style_attr=style_attr, animal_attr=animal_attr, files_found=['project_folder/csv/machine_results/MyVideo.csv'], cores=5, clf_attr=clf_attr)
    >>> path_plotter.run()
    """

    def __init__(
        self,
        config_path: str,
        frame_setting: bool,
        video_setting: bool,
        last_frame: bool,
        files_found: List[str],
        cores: int,
        input_style_attr: Optional[Dict] = None,
        animal_attr: Dict[int, List[str]] = None,
        input_clf_attr: Optional[Dict[int, List[str]]] = None,
        slicing: Optional[Dict] = None,
    ):
        if platform.system() == "Darwin":
            multiprocessing.set_start_method("spawn", force=True)

        if (not frame_setting) and (not video_setting) and (not last_frame):
            raise NoSpecifiedOutputError(
                msg="SIMBA ERROR: Please choice to create path frames and/or video path plots"
            )
        ConfigReader.__init__(self, config_path=config_path)
        PlottingMixin.__init__(self)
        (
            self.video_setting,
            self.frame_setting,
            self.input_style_attr,
            self.files_found,
            self.animal_attr,
            self.input_clf_attr,
            self.last_frame,
            self.cores,
        ) = (
            video_setting,
            frame_setting,
            input_style_attr,
            files_found,
            animal_attr,
            input_clf_attr,
            last_frame,
            cores,
        )
        self.no_animals_path_plot, self.clf_attr, self.slicing = (
            len(animal_attr.keys()),
            None,
            slicing,
        )
        if not os.path.exists(self.path_plot_dir):
            os.makedirs(self.path_plot_dir)
        check_if_filepath_list_is_empty(
            filepaths=self.files_found,
            error_msg="SIMBA ERROR: Zero files found in the project_folder/csv/machine_results directory. To plot paths without performing machine classifications, use path plotter functions in [ROI] tab.",
        )
        print(f"Processing {str(len(self.files_found))} videos...")

    def run(self):
        """
        Method to create path plot videos and/or frames.Results are store in the
        'project_folder/frames/path_plots' directory of the SimBA project.
        """

        for file_cnt, file_path in enumerate(self.files_found):
            video_timer = SimbaTimer(start=True)
            _, self.video_name, _ = get_fn_ext(file_path)
            self.video_info, _, self.fps = self.read_video_info(
                video_name=self.video_name
            )
            self.data_df = read_df(file_path, self.file_type)
            if self.slicing:
                frm_numbers = find_frame_numbers_from_time_stamp(
                    start_time=self.slicing["start_time"],
                    end_time=self.slicing["end_time"],
                    fps=self.fps,
                )
                if len(set(frm_numbers) - set(self.data_df.index)) > 0:
                    raise FrameRangeError(
                        msg=f'The chosen time-period ({self.slicing["start_time"]} - {self.slicing["end_time"]}) does not exist in {self.video_name}.'
                    )
                else:
                    self.data_df = self.data_df.loc[
                        frm_numbers[0] : frm_numbers[-1]
                    ].reset_index(drop=True)
            self.__get_styles()
            self.temp_folder = os.path.join(self.path_plot_dir, self.video_name, "temp")
            self.save_frame_folder_dir = os.path.join(
                self.path_plot_dir, self.video_name
            )
            if self.frame_setting:
                if os.path.exists(self.save_frame_folder_dir):
                    remove_a_folder(self.save_frame_folder_dir)
                if not os.path.exists(self.save_frame_folder_dir):
                    os.makedirs(self.save_frame_folder_dir)
            if self.video_setting:
                self.video_folder = os.path.join(self.path_plot_dir, self.video_name)
                if os.path.exists(self.temp_folder):
                    remove_a_folder(self.temp_folder)
                    remove_a_folder(self.video_folder)
                os.makedirs(self.temp_folder)
                self.save_video_path = os.path.join(
                    self.path_plot_dir, self.video_name + ".mp4"
                )

            if self.input_clf_attr:
                clf_names, self.clf_attr = [], {}
                self.clf_attr["attr"] = deepcopy(self.input_clf_attr)
                for v in self.input_clf_attr.values():
                    clf_names.append(v[0])
                self.clf_attr["data"] = self.data_df[clf_names]
                self.clf_attr["positions"] = self.data_df[
                    [self.animal_attr[0][0] + "_x", self.animal_attr[0][0] + "_y"]
                ]

            if self.last_frame:
                self.__get_deque_lookups()
                _ = self.make_path_plot(
                    data_df=self.data_df,
                    video_info=self.video_info,
                    style_attr=self.style_attr,
                    deque_dict=self.deque_dict,
                    clf_attr=self.clf_attr,
                    save_path=os.path.join(
                        self.path_plot_dir, self.video_name + "_final_frame.png"
                    ),
                )

            if self.video_setting or self.frame_setting:
                data_arr = np.array(list(self.data_df.index)).reshape(-1, 1)
                for animal_cnt, animal_data in self.animal_attr.items():
                    bp_x_name = "{}_{}".format(animal_data[0], "x")
                    bp_y_name = "{}_{}".format(animal_data[0], "y")
                    data_arr = np.hstack(
                        (
                            data_arr,
                            self.data_df[[bp_x_name, bp_y_name]].astype(int).values,
                        )
                    )
                    if animal_cnt == 0 and self.clf_attr:
                        self.clf_attr["positions"] = deepcopy(data_arr[:, 1:3])
                data_arr = self.__split_array_into_max_lines(
                    data=data_arr, max_lines=self.style_attr["max lines"]
                )
                data_arr = np.array_split(data_arr, self.cores)
                data = []
                for cnt, i in enumerate(data_arr):
                    data.append(self.__insert_group_idx_column(data=i, group=cnt))
                frm_per_core = data[0].shape[0]

                print(
                    "Creating path plots, multiprocessing (chunksize: {}, cores: {})...".format(
                        str(self.multiprocess_chunksize), str(self.cores)
                    )
                )
                with multiprocessing.Pool(
                    self.cores, maxtasksperchild=self.maxtasksperchild
                ) as pool:
                    constants = functools.partial(
                        self.path_plot_mp,
                        video_setting=self.video_setting,
                        video_name=self.video_name,
                        frame_setting=self.frame_setting,
                        video_save_dir=self.temp_folder,
                        frame_folder_dir=self.save_frame_folder_dir,
                        style_attr=self.style_attr,
                        fps=self.fps,
                        animal_attr=self.animal_attr,
                        video_info=self.video_info,
                        clf_attr=self.clf_attr,
                    )
                    for cnt, result in enumerate(
                        pool.imap(
                            constants, data, chunksize=self.multiprocess_chunksize
                        )
                    ):
                        print(
                            "Image {}/{}, Video {}/{}...".format(
                                str(int(frm_per_core * (result + 1))),
                                str(len(self.data_df)),
                                str(file_cnt + 1),
                                str(len(self.files_found)),
                            )
                        )

                    pool.terminate()
                    pool.join()

                if self.video_setting:
                    print("Joining {} multiprocessed video...".format(self.video_name))
                    concatenate_videos_in_folder(
                        in_folder=self.temp_folder, save_path=self.save_video_path
                    )

                video_timer.stop_timer()
                print(
                    "Path plot video {} complete (elapsed time: {}s) ...".format(
                        self.video_name, video_timer.elapsed_time_str
                    )
                )

        self.timer.stop_timer()
        stdout_success(
            msg=f"Path plot visualizations for {str(len(self.files_found))} videos created in project_folder/frames/output/path_plots directory",
            elapsed_time=self.timer.elapsed_time_str,
        )

    @staticmethod
    @jit(nopython=True)
    def __split_array_into_max_lines(data: np.array, max_lines: int):
        results = np.full((data.shape[0], max_lines, data.shape[1]), np.nan, data.dtype)
        for i in prange(data.shape[0]):
            start = int(i - max_lines)
            if start < 0:
                start = 0
            frm_data = data[start:i, :]

            missing_cnt = max_lines - frm_data.shape[0]

            if missing_cnt > 0:
                frm_data = np.vstack(
                    (
                        np.full((missing_cnt, frm_data.shape[1]), -1.0, frm_data.dtype),
                        frm_data,
                    )
                )

            results[i] = frm_data
        return results

    @staticmethod
    @jit(nopython=True)
    def __insert_group_idx_column(data: np.array, group: int):
        results = np.full((data.shape[0], data.shape[1], data.shape[2] + 1), np.nan)
        group_col = np.full((data.shape[1], 1), group)
        for frm_idx in prange(data.shape[0]):
            results[frm_idx] = np.hstack((group_col, data[frm_idx]))
        return results

    def __get_styles(self):
        self.style_attr = {}
        if self.input_style_attr is not None:
            self.style_attr["bg color"] = self.color_dict[
                self.input_style_attr["bg color"]
            ]
            if self.input_style_attr["max lines"] == "entire video":
                self.style_attr["max lines"] = len(self.data_df)
            else:
                self.style_attr["max lines"] = int(
                    int(self.input_style_attr["max lines"])
                    * (int(self.video_info["fps"].values[0]) / 1000)
                )
            self.style_attr["font thickness"] = self.input_style_attr["font thickness"]
            self.style_attr["line width"] = self.input_style_attr["line width"]
            self.style_attr["font size"] = self.input_style_attr["font size"]
            self.style_attr["circle size"] = self.input_style_attr["circle size"]
            if self.input_style_attr["width"] == "As input":
                self.style_attr["width"], self.style_attr["height"] = int(
                    self.video_info["Resolution_width"].values[0]
                ), int(self.video_info["Resolution_height"].values[0])
            else:
                pass
        else:
            space_scaler, radius_scaler, res_scaler, font_scaler = 25, 10, 1500, 0.8
            self.style_attr["width"] = int(
                self.video_info["Resolution_width"].values[0]
            )
            self.style_attr["height"] = int(
                self.video_info["Resolution_height"].values[0]
            )
            max_res = max(self.style_attr["width"], self.style_attr["height"])
            self.style_attr["circle size"] = int(radius_scaler / (res_scaler / max_res))
            self.style_attr["font size"] = font_scaler / (res_scaler / max_res)
            self.style_attr["bg color"] = self.color_dict["White"]
            self.style_attr["max lines"] = len(self.data_df)
            self.style_attr["font thickness"] = 2
            self.style_attr["line width"] = 2

        self.style_attr["animal names"] = []
        self.style_attr["animal clrs"] = []
        for animal_cnt, animal_data in self.animal_attr.items():
            self.style_attr["animal names"].append(
                self.find_animal_name_from_body_part_name(
                    bp_name=animal_data[0], bp_dict=self.animal_bp_dict
                )
            )

        for animal_cnt, animal_data in self.animal_attr.items():
            self.style_attr["animal clrs"].append(
                self.color_dict[self.animal_attr[animal_cnt][1]]
            )

    def __get_deque_lookups(self):
        self.deque_dict = {}
        for animal_cnt, animal in enumerate(self.style_attr["animal names"]):
            self.deque_dict[animal] = {}
            self.deque_dict[animal]["deque"] = deque(
                maxlen=self.style_attr["max lines"]
            )
            self.deque_dict[animal]["bp"] = self.animal_attr[animal_cnt][0]
            self.deque_dict[animal]["clr"] = self.color_dict[
                self.animal_attr[animal_cnt][1]
            ]


# style_attr = {'width': 'As input',
#               'height': 'As input',
#               'line width': 5, 'font size': 5, 'font thickness': 2, 'circle size': 5, 'bg color': 'White', 'max lines': 'entire video'}
# animal_attr = {0: ['Ear_right_1', 'Red'], 1: ['Ear_right_2', 'Green']}
# clf_attr = {0: ['Attack', 'Black', 'Size: 30']}
# # #
# # # clf_attr = None
# # # style_attr = None
# #
# path_plotter = PathPlotterMulticore(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                     frame_setting=False,
#                                     video_setting=True,
#                                     last_frame=True,
#                                     input_clf_attr=clf_attr,
#                                     input_style_attr=style_attr,
#                                     animal_attr=animal_attr,
#                                     files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'],
#                                     cores=5,
#                                     slicing = {'start_time': '00:00:01', 'end_time': '00:00:03'})
# path_plotter.run()


# style_attr = None
# #style_attr = {'width': 'As input', 'height': 'As input', 'line width': 5, 'font size': 5, 'font thickness': 2, 'circle size': 5, 'bg color': 'White', 'max lines': 10000}
# animal_attr = {0: ['mouth', 'Red']}
# clf_attr = {0: ['Freezing', 'Black', 'Size: 10'], 1: ['Normal Swimming', 'Red', 'Size: 10']}
# #
# # clf_attr = None
#
#
# path_plotter = PathPlotterMulticore(config_path='/Users/simon/Desktop/envs/troubleshooting/naresh/project_folder/project_config.ini',
#                                     frame_setting=False,
#                                     video_setting=True,
#                                     last_frame=True,
#                                     input_clf_attr=clf_attr,
#                                     input_style_attr=style_attr,
#                                     animal_attr=animal_attr,
#                                     files_found=['/Users/simon/Desktop/envs/troubleshooting/naresh/project_folder/csv/machine_results/SF8.csv'],
#                                     cores=5)
# path_plotter.run()
