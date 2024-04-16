__author__ = "Simon Nilsson"

import functools
import multiprocessing
import os
import platform
from collections import deque
from copy import deepcopy
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import pandas as pd
from numba import jit, prange

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_filepath_list_is_empty,
                                check_if_valid_rgb_str, check_int,
                                check_that_column_exist)
from simba.utils.data import find_frame_numbers_from_time_stamp
from simba.utils.enums import Formats, TagNames
from simba.utils.errors import FrameRangeError, NoSpecifiedOutputError
from simba.utils.lookups import get_color_dict
from simba.utils.printing import SimbaTimer, log_event, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    find_core_cnt, find_video_of_file,
                                    get_fn_ext, get_video_meta_data, read_df,
                                    read_frm_of_video, remove_a_folder)


def path_plot_mp(
    batch_id: np.ndarray,
    data: np.array,
    video_setting: bool,
    frame_setting: bool,
    video_save_dir: str,
    video_name: str,
    frame_folder_dir: str,
    style_attr: dict,
    print_animal_names: bool,
    animal_attr: dict,
    fps: int,
    video_info: pd.DataFrame,
    clf_attr: dict,
    input_style_attr: dict,
    video_path: Optional[Union[str, os.PathLike]] = None,
):

    batch_id = batch_id[0]
    batch_data = data[np.argwhere(data[:, 0] == batch_id)].reshape(-1, data.shape[1])
    frm_cnts = batch_data[:, 1]
    color_dict = get_color_dict()
    if video_setting:
        fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        video_save_path = os.path.join(video_save_dir, f"{batch_id}.mp4")
        video_writer = cv2.VideoWriter(
            video_save_path, fourcc, fps, (style_attr["width"], style_attr["height"])
        )

    if input_style_attr is not None:
        if (isinstance(input_style_attr["bg color"], dict)) and (
            input_style_attr["bg color"]["type"]
        ) == "moving":
            check_file_exist_and_readable(file_path=video_path)
            video_cap = cv2.VideoCapture(video_path)

    bg_clr = style_attr["bg color"]
    for frame_id in frm_cnts:
        frm_data = data[np.argwhere(data[:, 1] <= frame_id)].reshape(-1, data.shape[1])[
            :, 2:
        ]
        frm_data = np.split(frm_data, len(list(animal_attr.keys())), axis=1)
        if (isinstance(style_attr["bg color"], dict)) and (
            style_attr["bg color"]["type"]
        ) == "moving":
            bg_clr = read_frm_of_video(
                video_path=video_cap,
                opacity=style_attr["bg color"]["opacity"],
                frame_index=frame_id,
            )
        img = np.zeros(
            (
                int(video_info["Resolution_height"].values[0]),
                int(video_info["Resolution_width"].values[0]),
                3,
            )
        )
        img[:] = bg_clr
        for animal_cnt, pose_data in enumerate(frm_data):
            animal_clr = style_attr["animal clrs"][animal_cnt]
            for j in range(pose_data.shape[0]):
                cv2.line(
                    img,
                    (pose_data[j][0], pose_data[j][1]),
                    (pose_data[j][2], pose_data[j][3]),
                    animal_clr,
                    int(style_attr["line width"]),
                )
            cv2.circle(
                img,
                (pose_data[-1][2], pose_data[-1][3]),
                0,
                animal_clr,
                style_attr["circle size"],
            )
            if print_animal_names:
                cv2.putText(
                    img,
                    style_attr["animal names"][animal_cnt],
                    (pose_data[-1][2], pose_data[-1][3]),
                    cv2.FONT_HERSHEY_COMPLEX,
                    style_attr["font size"],
                    animal_clr,
                    style_attr["font thickness"],
                )
        if clf_attr:
            for clf_cnt, clf_name in enumerate(clf_attr["data"].columns):
                clf_size = int(clf_attr["attr"][clf_cnt][-1].split(": ")[-1])
                clf_clr = color_dict[clf_attr["attr"][clf_cnt][1]]
                clf_sliced = clf_attr["data"][clf_name].loc[0:frame_id]
                clf_sliced_idx = list(clf_sliced[clf_sliced == 1].index)
                locations = clf_attr["positions"][clf_sliced_idx, :]
                for i in range(locations.shape[0]):
                    cv2.circle(
                        img, (locations[i][0], locations[i][1]), 0, clf_clr, clf_size
                    )
        img = cv2.resize(img, (style_attr["width"], style_attr["height"]))
        if video_setting:
            video_writer.write(np.uint8(img))
        if frame_setting:
            frm_name = os.path.join(frame_folder_dir, f"{frame_id}.png")
            cv2.imwrite(frm_name, np.uint8(img))
        print(
            f"Path frame created: {frame_id}, Video: {video_name}, Processing core: {batch_id}"
        )
    if video_setting:
        video_writer.release()
    return batch_id


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

    .. image:: _static/img/path_plot.png
       :width: 300
       :align: center

    .. image:: _static/img/path_plot_mp.gif
       :width: 500
       :align: center

    :example:
    >>> input_style_attr = {'width': 'As input', 'height': 'As input', 'line width': 5, 'font size': 5, 'font thickness': 2, 'circle size': 5, 'bg color': 'White', 'max lines': 100}
    >>> animal_attr = {0: ['Ear_right_1', 'Red']}
    >>> input_clf_attr = {0: ['Attack', 'Black', 'Size: 30'], 1: ['Sniffing', 'Red', 'Size: 30']}
    >>> path_plotter = PathPlotterMulticore(config_path=r'MyConfigPath', frame_setting=False, video_setting=True, style_attr=style_attr, animal_attr=animal_attr, files_found=['project_folder/csv/machine_results/MyVideo.csv'], cores=5, clf_attr=clf_attr, print_animal_names=True)
    >>> path_plotter.run()
    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        files_found: List[str],
        frame_setting: Optional[bool] = False,
        video_setting: Optional[bool] = False,
        last_frame: Optional[bool] = True,
        cores: Optional[int] = -1,
        print_animal_names: Optional[bool] = False,
        input_style_attr: Optional[Dict] = None,
        animal_attr: Dict[int, List[str]] = None,
        input_clf_attr: Optional[Dict[int, List[str]]] = None,
        slicing: Optional[Dict] = None,
    ):

        if platform.system() == "Darwin":
            multiprocessing.set_start_method("spawn", force=True)
        if (not frame_setting) and (not video_setting) and (not last_frame):
            raise NoSpecifiedOutputError(
                msg="SIMBA ERROR: Please choice to create path frames and/or video path plots",
                source=self.__class__.__name__,
            )
        check_if_filepath_list_is_empty(
            filepaths=files_found,
            error_msg="SIMBA ERROR: Zero files found in the project_folder/csv/machine_results directory. To plot paths without performing machine classifications, use path plotter functions in [ROI] tab.",
        )
        check_int(
            name=f"{self.__class__.__name__} core_cnt",
            value=cores,
            min_value=-1,
            max_value=find_core_cnt()[0],
        )
        if cores == -1:
            cores = find_core_cnt()[0]

        ConfigReader.__init__(self, config_path=config_path)
        PlottingMixin.__init__(self)
        log_event(
            logger_name=str(__class__.__name__),
            log_type=TagNames.CLASS_INIT.value,
            msg=self.create_log_msg_from_init_args(locals=locals()),
        )
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
        self.print_animal_names = print_animal_names
        self.no_animals_path_plot, self.clf_attr, self.slicing = (
            len(animal_attr.keys()),
            None,
            slicing,
        )
        if not os.path.exists(self.path_plot_dir):
            os.makedirs(self.path_plot_dir)
        print(f"Processing {len(self.files_found)} videos...")

    def __get_styles(self):
        self.style_attr = {}
        if self.input_style_attr is not None:
            if not type(self.input_style_attr["bg color"]) == dict:
                self.style_attr["bg color"] = self.color_dict[
                    self.input_style_attr["bg color"]
                ]
            else:
                self.style_attr["bg color"] = self.input_style_attr["bg color"]
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
            self.style_attr["print_animal_names"] = self.print_animal_names
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
            self.style_attr["print_animal_names"] = self.print_animal_names
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
            if type(self.animal_attr[animal_cnt][1]) == tuple:
                check_if_valid_rgb_str(
                    str(self.animal_attr[animal_cnt][1]), return_cleaned_rgb_tuple=False
                )
                self.style_attr["animal clrs"] = self.animal_attr[animal_cnt][1]
            else:
                self.style_attr["animal clrs"].append(
                    self.color_dict[self.animal_attr[animal_cnt][1]]
                )

    @staticmethod
    @jit(nopython=True)
    def __insert_group_idx_column(data: np.array, group: int):
        group_col = np.full((data.shape[0], 1), group)
        return np.hstack((group_col, data))

    @staticmethod
    # @jit(nopython=True)
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
                ).astype(float)
                mask = frm_data == -1.0
                frm_data[mask] = np.nan
                frm_data = pd.DataFrame(frm_data)
                for col in frm_data.columns:
                    frm_data = frm_data.fillna(method="ffill").fillna(method="bfill")
            results[i] = frm_data.values
        return results

    def __get_deque_lookups(self):
        self.deque_dict = {}
        for animal_cnt, animal in enumerate(self.style_attr["animal names"]):
            self.deque_dict[animal] = {}
            self.deque_dict[animal]["deque"] = deque(
                maxlen=self.style_attr["max lines"]
            )
            self.deque_dict[animal]["bp"] = self.animal_attr[animal_cnt][0]
            if type(self.animal_attr[animal_cnt][1]) == tuple:
                check_if_valid_rgb_str(
                    str(self.animal_attr[animal_cnt][1]), return_cleaned_rgb_tuple=False
                )
                self.deque_dict[animal]["clr"] = self.animal_attr[animal_cnt][1]
            else:
                self.deque_dict[animal]["clr"] = self.color_dict[
                    self.animal_attr[animal_cnt][1]
                ]

    def run(self):
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
                        msg=f'The chosen time-period ({self.slicing["start_time"]} - {self.slicing["end_time"]}) does not exist in {self.video_name}.',
                        source=self.__class__.__name__,
                    )
                else:
                    self.data_df = self.data_df.loc[frm_numbers[0] : frm_numbers[-1]]
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
                check_that_column_exist(
                    df=self.data_df, column_name=clf_names, file_name=self.video_name
                )
                self.clf_attr["data"] = self.data_df[clf_names]
                self.clf_attr["positions"] = self.data_df[
                    [self.animal_attr[0][0] + "_x", self.animal_attr[0][0] + "_y"]
                ]

            if self.last_frame:
                self.__get_deque_lookups()
                if isinstance(self.style_attr["bg color"], dict):
                    self.video_path = find_video_of_file(
                        video_dir=self.video_dir,
                        filename=self.video_name,
                        raise_error=True,
                    )
                    if "frame_index" in self.style_attr["bg color"].keys():
                        check_int(
                            name="Static frame index",
                            value=self.style_attr["bg color"]["frame_index"],
                            min_value=0,
                        )
                        frame_index = self.style_attr["bg color"]["frame_index"]
                    else:
                        video_meta_data = get_video_meta_data(
                            video_path=self.video_path
                        )
                        frame_index = video_meta_data["frame_count"] - 1
                    self.style_attr["bg color"] = read_frm_of_video(
                        video_path=self.video_path,
                        opacity=self.style_attr["bg color"]["opacity"],
                        frame_index=frame_index,
                    )
                self.make_path_plot(
                    data_df=self.data_df,
                    video_info=self.video_info,
                    style_attr=self.style_attr,
                    print_animal_names=self.print_animal_names,
                    deque_dict=self.deque_dict,
                    clf_attr=self.clf_attr,
                    save_path=os.path.join(
                        self.path_plot_dir, self.video_name + "_final_frame.png"
                    ),
                )

            if self.video_setting or self.frame_setting:
                data_arr = np.array(list(self.data_df.index)).reshape(-1, 1)
                for animal_cnt, animal_data in self.animal_attr.items():
                    bp_x_name = f"{animal_data[0]}_x"
                    bp_y_name = f"{animal_data[0]}_y"
                    check_that_column_exist(
                        df=self.data_df,
                        column_name=[bp_x_name, bp_y_name],
                        file_name=self.video_name,
                    )
                    animal_df = self.data_df[[bp_x_name, bp_y_name]]
                    animal_vals = FeatureExtractionMixin.create_shifted_df(
                        df=animal_df
                    ).values.astype(int)
                    data_arr = np.hstack((data_arr, animal_vals))
                if self.clf_attr:
                    self.clf_attr["positions"] = deepcopy(data_arr[:, 1:3])
                frm_range = [[x] for x in range(self.cores)]
                data_arr = np.array_split(data_arr, self.cores)
                data_arr = [
                    self.__insert_group_idx_column(data=i, group=cnt)
                    for cnt, i in enumerate(data_arr)
                ]
                data_arr = np.concatenate(data_arr, axis=0)
                self.video_path = None
                if isinstance(self.input_style_attr["bg color"], dict):
                    self.video_path = find_video_of_file(
                        video_dir=self.video_dir,
                        filename=self.video_name,
                        raise_error=True,
                    )
                    if self.input_style_attr["bg color"]["type"] == "static":
                        self.style_attr["bg color"] = read_frm_of_video(
                            video_path=self.video_path,
                            opacity=self.style_attr["bg color"]["opacity"],
                            frame_index=self.style_attr["bg color"]["frame_index"],
                        )
                    else:
                        self.style_attr["bg color"] = self.input_style_attr["bg color"]
                print(
                    f"Creating path plots, multiprocessing (chunksize: {self.multiprocess_chunksize}, cores: {self.cores})..."
                )

                with multiprocessing.Pool(
                    self.cores, maxtasksperchild=self.maxtasksperchild
                ) as pool:
                    constants = functools.partial(
                        path_plot_mp,
                        data=data_arr,
                        video_setting=self.video_setting,
                        video_name=self.video_name,
                        frame_setting=self.frame_setting,
                        video_save_dir=self.temp_folder,
                        frame_folder_dir=self.save_frame_folder_dir,
                        style_attr=self.style_attr,
                        print_animal_names=self.print_animal_names,
                        fps=self.fps,
                        animal_attr=self.animal_attr,
                        video_info=self.video_info,
                        clf_attr=self.clf_attr,
                        input_style_attr=self.input_style_attr,
                        video_path=self.video_path,
                    )
                    for cnt, result in enumerate(
                        pool.imap(
                            constants, frm_range, chunksize=self.multiprocess_chunksize
                        )
                    ):
                        print(f"Path batch {result+1}/{self.cores} complete...")
                    pool.terminate()
                    pool.join()

                if self.video_setting:
                    print(f"Joining {self.video_name} multiprocessed video...")
                    concatenate_videos_in_folder(
                        in_folder=self.temp_folder, save_path=self.save_video_path
                    )

                video_timer.stop_timer()
                print(
                    f"Path plot video {self.video_name} complete (elapsed time: {video_timer.elapsed_time_str}s) ..."
                )

        self.timer.stop_timer()
        stdout_success(
            msg=f"Path plot visualizations for {len(self.files_found)} videos created in {self.path_plot_dir} directory",
            elapsed_time=self.timer.elapsed_time_str,
            source=self.__class__.__name__,
        )


# style_attr = {'width': 'As input', 'height': 'As input', 'line width': 5, 'font size': 2, 'font thickness': 2, 'circle size': 5, 'bg color': {'type': 'moving', 'opacity': 100, 'frame_index': 100}, 'max lines': 'entire video'}
# animal_attr = {0: ['Ear_right_1', 'Red'], 1: ['Ear_right_2', 'Green']}
# clf_attr = {0: ['Nose to Nose', 'Black', 'Size: 30']}
# # #
# clf_attr = None
# style_attr = None

# path_plotter = PathPlotterMulticore(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/project_config.ini',
#                                     frame_setting=False,
#                                     video_setting=True,
#                                     last_frame=True,
#                                     input_clf_attr=clf_attr,
#                                     input_style_attr=style_attr,
#                                     animal_attr=animal_attr,
#                                     files_found=['/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/csv/machine_results/Trial    10.csv'],
#                                     cores=-1,
#                                     slicing = None, # {'start_time': '00:00:00', 'end_time': '00:00:05'}, # , #None,
#                                     print_animal_names=False)
# path_plotter.run()


# style_attr = {'width': 'As input',
#               'height': 'As input',
#               'line width': 5, 'font size': 5, 'font thickness': 2, 'circle size': 5, 'bg color': {'type': 'static', 'opacity': 100, 'frame_index': 100}, 'max lines': 'entire video'}
# animal_attr = {0: ['Ear_right_1', 'Red'], 1: ['Ear_right_2', 'Green']}
# # clf_attr = {0: ['Attack', 'Black', 'Size: 30']}
# # # #
# clf_attr = None
# style_attr = None
#
# path_plotter = PathPlotterMulticore(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                     frame_setting=False,
#                                     video_setting=True,
#                                     last_frame=True,
#                                     input_clf_attr=clf_attr,
#                                     input_style_attr=style_attr,
#                                     animal_attr=animal_attr,
#                                     files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'],
#                                     cores=5,
#                                     slicing = None)
# path_plotter.run()

#


# style_attr = None
# style_attr = {'width': 'As input',
#               'height': 'As input',
#               'line width': 5, 'font size': 5, 'font thickness': 2, 'circle size': 5, 'bg color': {'opacity': 100}, 'max lines': 10000}
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
