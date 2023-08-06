__author__ = "Simon Nilsson"

import os
from collections import deque
from copy import deepcopy
from typing import Dict, List, Optional

import cv2
import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import check_if_filepath_list_is_empty
from simba.utils.data import find_frame_numbers_from_time_stamp
from simba.utils.enums import Formats
from simba.utils.errors import FrameRangeError, NoSpecifiedOutputError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, read_df


class PathPlotterSingleCore(ConfigReader, PlottingMixin):
    """
    Create "path plots" videos and/or images detailing the movement paths of
    individual animals in SimBA.

    .. note::
        For improved run-time, see :meth:`simba.path_plotter_mp.PathPlotterMulticore` for multiprocess class.
       `Visualization tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-11-visualizations>`__.

    .. image:: _static/img/path_plot.png
       :width: 300
       :align: center

    :param str config_path: Path to SimBA project config file in Configparser format
    :param bool frame_setting: If True, individual frames will be created.
    :param bool video_setting: If True, compressed videos will be created.
    :param List[str] files_found: Data paths to create from which to create plots
    :param dict animal_attr: Animal body-parts and colors
    :param dict style_attr: Plot sttributes (line thickness, color, etc..)
    :param Optional[dict] slicing: If Dict, start time and end time of video slice to create path plot from. E.g., {'start_time': '00:00:01', 'end_time': '00:00:03'}. If None, creates path plot for entire video.



    Examples
    ----------
    >>> style_attr = {'width': 'As input', 'height': 'As input', 'line width': 5, 'font size': 5, 'font thickness': 2, 'circle size': 5, 'bg color': 'White', 'max lines': 100}
    >>> animal_attr = {0: ['Ear_right_1', 'Red']}
    >>> path_plotter = PathPlotterSingleCore(config_path=r'MyConfigPath', frame_setting=False, video_setting=True, style_attr=style_attr, animal_attr=animal_attr, files_found=['project_folder/csv/machine_results/MyVideo.csv']).run()
    """

    def __init__(
        self,
        config_path: str,
        frame_setting: bool,
        video_setting: bool,
        last_frame: bool,
        files_found: List[str],
        input_style_attr: dict,
        animal_attr: dict,
        input_clf_attr: dict,
        slicing: Optional[Dict] = None,
    ):
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
        ) = (
            video_setting,
            frame_setting,
            input_style_attr,
            files_found,
            animal_attr,
            input_clf_attr,
            last_frame,
        )
        if (not frame_setting) and (not video_setting) and (not last_frame):
            raise NoSpecifiedOutputError(
                msg="SIMBA ERROR: Please choice to create path frames and/or video path plots"
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
            error_msg="Zero files found in the project_folder/csv/machine_results directory. To plot paths without performing machine classifications, use path plotter functions in [ROI] tab.",
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
            self.data_df = read_df(file_path, self.file_type).reset_index(drop=True)
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
            self.__get_deque_lookups()
            if self.video_setting:
                self.video_save_path = os.path.join(
                    self.path_plot_dir, self.video_name + ".mp4"
                )
                self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
                self.writer = cv2.VideoWriter(
                    self.video_save_path,
                    self.fourcc,
                    self.fps,
                    (self.style_attr["width"], self.style_attr["height"]),
                )

            if self.frame_setting:
                self.save_video_folder = os.path.join(
                    self.path_plot_dir, self.video_name
                )
                if not os.path.exists(self.save_video_folder):
                    os.makedirs(self.save_video_folder)

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
                for frm_cnt in range(len(self.data_df)):
                    img = np.zeros(
                        (
                            int(self.video_info["Resolution_height"].values[0]),
                            int(self.video_info["Resolution_width"].values[0]),
                            3,
                        )
                    )
                    img[:] = self.style_attr["bg color"]
                    for animal_cnt, (animal_name, animal_data) in enumerate(
                        self.deque_dict.items()
                    ):
                        bp_x = int(
                            self.data_df.loc[
                                frm_cnt, "{}_{}".format(animal_data["bp"], "x")
                            ]
                        )
                        bp_y = int(
                            self.data_df.loc[
                                frm_cnt, "{}_{}".format(animal_data["bp"], "y")
                            ]
                        )
                        self.deque_dict[animal_name]["deque"].appendleft((bp_x, bp_y))
                    for animal_name, animal_data in self.deque_dict.items():
                        cv2.circle(
                            img,
                            (self.deque_dict[animal_name]["deque"][0]),
                            0,
                            self.deque_dict[animal_name]["clr"],
                            self.style_attr["circle size"],
                        )
                        cv2.putText(
                            img,
                            animal_name,
                            (self.deque_dict[animal_name]["deque"][0]),
                            Formats.FONT.value,
                            self.style_attr["font size"],
                            self.deque_dict[animal_name]["clr"],
                            self.style_attr["font thickness"],
                        )

                    for animal_name, animal_data in self.deque_dict.items():
                        for i in range(len(self.deque_dict[animal_name]["deque"]) - 1):
                            line_clr = self.deque_dict[animal_name]["clr"]
                            position_1 = self.deque_dict[animal_name]["deque"][i]
                            position_2 = self.deque_dict[animal_name]["deque"][i + 1]
                            cv2.line(
                                img,
                                position_1,
                                position_2,
                                line_clr,
                                self.style_attr["line width"],
                            )

                    if self.input_clf_attr:
                        animal_1_name = list(self.deque_dict.keys())[0]
                        animal_bp_x, animal_bp_y = (
                            self.deque_dict[animal_1_name]["bp"] + "_x",
                            self.deque_dict[animal_1_name]["bp"] + "_y",
                        )
                        for clf_cnt, clf_name in enumerate(
                            self.clf_attr["data"].columns
                        ):
                            clf_size = int(
                                self.clf_attr["attr"][clf_cnt][-1].split(": ")[-1]
                            )
                            clf_clr = self.color_dict[self.clf_attr["attr"][clf_cnt][1]]
                            sliced_df = self.clf_attr["data"].loc[0:frm_cnt]
                            sliced_df_idx = list(
                                sliced_df[
                                    sliced_df[self.clf_attr["attr"][clf_cnt][0]] == 1
                                ].index
                            )
                            locations = (
                                self.data_df.loc[
                                    sliced_df_idx, [animal_bp_x, animal_bp_y]
                                ]
                                .astype(int)
                                .values
                            )
                            for i in range(locations.shape[0]):
                                cv2.circle(
                                    img,
                                    (locations[i][0], locations[i][1]),
                                    0,
                                    clf_clr,
                                    clf_size,
                                )

                    img = cv2.resize(
                        img, (self.style_attr["width"], self.style_attr["height"])
                    )
                    if self.video_setting:
                        self.writer.write(np.uint8(img))
                    if self.frame_setting:
                        frm_name = os.path.join(
                            self.save_video_folder, str(frm_cnt) + ".png"
                        )
                        cv2.imwrite(frm_name, np.uint8(img))
                    print(
                        "Frame: {} / {}. Video: {} ({}/{})".format(
                            str(frm_cnt + 1),
                            str(len(self.data_df)),
                            self.video_name,
                            str(file_cnt + 1),
                            len(self.files_found),
                        )
                    )

                if self.video_setting:
                    self.writer.release()
                video_timer.stop_timer()
                print(
                    "Path visualization for video {} saved (elapsed time {}s)...".format(
                        self.video_name, video_timer.elapsed_time_str
                    )
                )

        self.timer.stop_timer()
        stdout_success(
            msg=f"Path visualizations for {str(len(self.files_found))} videos saved in project_folder/frames/output/path_plots directory",
            elapsed_time=self.timer.elapsed_time_str,
        )

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
            self.style_attr["font size"] = int(font_scaler / (res_scaler / max_res))
            self.style_attr["bg color"] = self.color_dict["White"]
            self.style_attr["max lines"] = len(self.data_df)
            self.style_attr["font thickness"] = 2
            self.style_attr["line width"] = 2

    def __get_deque_lookups(self):
        self.deque_dict = {}
        for animal_cnt, animal_data in self.animal_attr.items():
            animal_name = self.find_animal_name_from_body_part_name(
                bp_name=animal_data[0], bp_dict=self.animal_bp_dict
            )
            self.deque_dict[animal_name] = {}
            self.deque_dict[animal_name]["deque"] = deque(
                maxlen=self.style_attr["max lines"]
            )
            self.deque_dict[animal_name]["bp"] = self.animal_attr[animal_cnt][0]
            self.deque_dict[animal_name]["clr"] = self.color_dict[
                self.animal_attr[animal_cnt][1]
            ]


# style_attr = {'width': 'As input',
#               'height': 'As input',
#               'line width': 5,
#               'font size': 5,
#               'font thickness': 2,
#               'circle size': 5,
#               'bg color': 'White',
#               'max lines': 'entire video'}
#
# animal_attr = {0: ['Ear_right_1', 'Red'], 1: ['Ear_right_2', 'Green']}
# clf_attr = {0: ['Attack', 'Black', 'Size: 30'], 1: ['Sniffing', 'Black', 'Size: 30']}

# test = PathPlotterSingleCore(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                             frame_setting=False,
#                             video_setting=True,
#                              last_frame=True,
#                             input_style_attr=style_attr,
#                    animal_attr=animal_attr,
#                              input_clf_attr=clf_attr,
#                     files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv',
#                                  '/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'])
# test.run()

# test = PathPlotterSingleCore(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                              frame_setting=False,
#                              video_setting=True,
#                              last_frame=True,
#                              input_style_attr=style_attr,
#                              animal_attr=animal_attr,
#                              input_clf_attr=clf_attr,
#                              slicing = {'start_time': '00:00:01', 'end_time': '00:00:03'},
#                              files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'])
# test.run()
