__author__ = "Simon Nilsson"

import os
import shutil
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_if_keys_exist_in_dict,
    check_if_string_value_is_valid_video_timestamp, check_if_valid_rgb_str,
    check_instance, check_int, check_that_column_exist,
    check_that_hhmmss_start_is_before_end, check_valid_lst)
from simba.utils.data import find_frame_numbers_from_time_stamp
from simba.utils.enums import Formats, TagNames
from simba.utils.errors import FrameRangeError, NoSpecifiedOutputError
from simba.utils.printing import SimbaTimer, log_event, stdout_success
from simba.utils.read_write import (find_video_of_file, get_fn_ext,
                                    get_video_meta_data, read_df,
                                    read_frm_of_video)

STYLE_WIDTH = "width"
STYLE_HEIGHT = "height"
STYLE_LINE_WIDTH = "line width"
STYLE_FONT_SIZE = "font size"
STYLE_FONT_THICKNESS = "font thickness"
STYLE_CIRCLE_SIZE = "circle size"
STYLE_MAX_LINES = "max lines"

STYLE_KEYS = [
    STYLE_WIDTH,
    STYLE_HEIGHT,
    STYLE_LINE_WIDTH,
    STYLE_FONT_SIZE,
    STYLE_FONT_THICKNESS,
    STYLE_CIRCLE_SIZE,
    STYLE_MAX_LINES,
]


class PathPlotterSingleCore(ConfigReader, PlottingMixin):
    """
    Create "path plots" videos and/or images detailing the movement paths of individual animals in SimBA.

    .. note::
        For improved run-time, see :meth:`simba.path_plotter_mp.PathPlotterMulticore` for multiprocess class.

       `Visualization tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-11-visualizations>`__.

       `Notebook example <https://simba-uw-tf-dev.readthedocs.io/en/latest/nb/path_plots.html>`__.

    .. image:: _static/img/path_plot.png
       :width: 300
       :align: center

    .. image:: _static/img/path_plot_mp.gif
       :width: 500
       :align: center

    .. image:: _static/img/path_plot_1.png
       :width: 500
       :align: center



    :param str config_path: Path to SimBA project config file in Configparser format
    :param bool frame_setting: If True, individual frames will be created.
    :param bool video_setting: If True, compressed videos will be created.
    :param List[str] files_found: Data paths to create from which to create plots
    :param dict animal_attr: Animal body-parts and colors
    :param dict style_attr: Plot sttributes (line thickness, color, etc..)
    :param Optional[dict] slicing: If Dict, start time and end time of video slice to create path plot from. E.g., {'start_time': '00:00:01', 'end_time': '00:00:03'}. If None, creates path plot for entire video.

    .. note::
       If style_attr['bg color'] is a dictionary, e.g., {'opacity': 100%}, then SimBA will use the video as background with set opacity.

    :examples:
    >>> style_attr = {'width': 'As input', 'height': 'As input', 'line width': 5, 'font size': 5, 'font thickness': 2, 'circle size': 5, 'bg color': 'White', 'max lines': 100, 'animal_names': True}
    >>> animal_attr = {0: ['Ear_right_1', 'Red']}
    >>> path_plotter = PathPlotterSingleCore(config_path=r'MyConfigPath', frame_setting=False, video_setting=True, style_attr=style_attr, animal_attr=animal_attr, files_found=['project_folder/csv/machine_results/MyVideo.csv'], print_animal_names=True).run()
    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        files_found: List[Union[str, os.PathLike]],
        animal_attr: dict,
        input_style_attr: Optional[Union[Dict[str, Any], None]] = None,
        clf_attr: Optional[Dict[int, List[str]]] = None,
        frame_setting: Union[bool] = False,
        video_setting: Union[bool] = False,
        last_frame: Union[bool] = False,
        print_animal_names: Optional[bool] = True,
        slicing: Optional[Dict] = None,
    ):

        log_event(
            logger_name=str(__class__.__name__),
            log_type=TagNames.CLASS_INIT.value,
            msg=self.create_log_msg_from_init_args(locals=locals()),
        )
        if (not frame_setting) and (not video_setting) and (not last_frame):
            raise NoSpecifiedOutputError(
                msg="SIMBA ERROR: Please choice to create path frames and/or video path plots",
                source=self.__class__.__name__,
            )
        check_valid_lst(
            data=files_found,
            source=self.__class__.__name__,
            valid_dtypes=(str,),
            min_len=1,
        )
        if input_style_attr is not None:
            check_if_keys_exist_in_dict(
                data=input_style_attr, key=STYLE_KEYS, name="input_style_attr"
            )
        ConfigReader.__init__(self, config_path=config_path)
        PlottingMixin.__init__(self)
        (
            self.video_setting,
            self.frame_setting,
            self.input_style_attr,
            self.files_found,
            self.animal_attr,
            self.clf_attr,
            self.last_frame,
        ) = (
            video_setting,
            frame_setting,
            input_style_attr,
            files_found,
            animal_attr,
            clf_attr,
            last_frame,
        )
        self.print_animal_names, self.slicing = print_animal_names, slicing
        if not os.path.exists(self.path_plot_dir):
            os.makedirs(self.path_plot_dir)
        print(f"Processing {len(self.files_found)} videos...")

    def __get_styles(self):
        self.style_attr = {}
        if self.input_style_attr is not None:
            self.style_attr["bg color"] = self.input_style_attr["bg color"]
            if self.input_style_attr["max lines"] == "entire video":
                self.style_attr["max lines"] = len(self.data_df)
            else:
                self.style_attr["max lines"] = max(
                    1,
                    int(
                        int(self.input_style_attr["max lines"] / 1000)
                        * (int(self.video_info["fps"].values[0]))
                    ),
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
            self.style_attr["font size"] = min(
                0.5, (font_scaler / (res_scaler / max_res))
            )
            self.style_attr["bg color"] = self.color_dict["White"]
            self.style_attr["print_animal_names"] = self.print_animal_names
            self.style_attr["max lines"] = len(self.data_df)
            self.style_attr["font thickness"] = 2
            self.style_attr["line width"] = 2

    def run(self):
        check_all_file_names_are_represented_in_video_log(
            video_info_df=self.video_info_df, data_paths=self.files_found
        )
        for file_cnt, file_path in enumerate(self.files_found):
            video_timer = SimbaTimer(start=True)
            _, self.video_name, _ = get_fn_ext(file_path)
            self.video_info, _, self.fps = self.read_video_info(
                video_name=self.video_name
            )
            self.data_df = read_df(file_path, self.file_type)
            line_data, colors, animal_names = [], [], []
            for k, v in self.animal_attr.items():
                check_if_keys_exist_in_dict(
                    data=v, key=["bp", "color"], name=f"animal attr {k}"
                )
                line_data.append(
                    self.data_df[[f'{v["bp"]}_x', f'{v["bp"]}_y']].values.astype(
                        np.int64
                    )
                )
                colors.append(v["color"])
                if self.print_animal_names:
                    animal_names.append(
                        self.find_animal_name_from_body_part_name(
                            bp_name=v["bp"], bp_dict=self.animal_bp_dict
                        )
                    )
            if not self.print_animal_names:
                animal_names = None

            if self.slicing:
                check_if_keys_exist_in_dict(
                    data=self.slicing, key=["start_time", "end_time"]
                )
                check_if_string_value_is_valid_video_timestamp(
                    value=self.slicing["start_time"], name="slice start time"
                )
                check_if_string_value_is_valid_video_timestamp(
                    value=self.slicing["end_time"], name="slice end time"
                )
                check_that_hhmmss_start_is_before_end(
                    start_time=self.slicing["start_time"],
                    end_time=self.slicing["end_time"],
                    name="slice times",
                )
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
                for i in range(len(line_data)):
                    line_data[i] = line_data[i][frm_numbers, :]
            self.__get_styles()

            if self.video_setting:
                self.video_save_path = os.path.join(
                    self.path_plot_dir, f"{self.video_name}.mp4"
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
                if os.path.exists(self.save_video_folder):
                    shutil.rmtree(path=self.save_video_folder)
                os.makedirs(self.save_video_folder)

            if self.clf_attr is not None:
                self.clf_attr_appended = {}
                check_instance(
                    source=self.__class__.__name__,
                    instance=self.clf_attr,
                    accepted_types=(dict,),
                )
                for k, v in self.clf_attr.items():
                    check_if_keys_exist_in_dict(
                        data=v, key=["color", "size"], name=f"clf_attr {k}"
                    )
                    check_that_column_exist(
                        df=self.data_df, column_name=k, file_name=file_path
                    )
                    self.clf_attr_appended[k] = self.clf_attr[k]
                    self.clf_attr_appended[k]["clfs"] = self.data_df[k].values.astype(
                        np.int8
                    )
                    self.clf_attr_appended[k]["positions"] = self.data_df[
                        [
                            self.animal_attr[0]["bp"] + "_x",
                            self.animal_attr[0]["bp"] + "_y",
                        ]
                    ].values.astype(np.int64)
                self.clf_attr = deepcopy(self.clf_attr_appended)
                del self.clf_attr_appended

            self.video_path = None
            if type(self.style_attr["bg color"]) == dict:
                check_if_keys_exist_in_dict(
                    data=self.style_attr["bg color"],
                    key=["type", "opacity", "frame_index"],
                )
                self.video_path = find_video_of_file(
                    video_dir=self.video_dir, filename=self.video_name, raise_error=True
                )
                video_meta_data = get_video_meta_data(video_path=self.video_path)
                if "frame_index" in self.style_attr["bg color"].keys():
                    check_int(
                        name="Static frame index",
                        value=self.style_attr["bg color"]["frame_index"],
                        min_value=0,
                    )
                    frame_index = self.style_attr["bg color"]["frame_index"]
                else:
                    frame_index = video_meta_data["frame_count"] - 1
                self.style_attr["bg color"] = read_frm_of_video(
                    video_path=self.video_path,
                    opacity=self.style_attr["bg color"]["opacity"],
                    frame_index=frame_index,
                )

            if self.last_frame:
                PlottingMixin.make_path_plot(
                    data=line_data,
                    colors=colors,
                    width=self.style_attr["width"],
                    height=self.style_attr["height"],
                    max_lines=self.style_attr["max lines"],
                    bg_clr=self.style_attr["bg color"],
                    circle_size=self.style_attr["circle size"],
                    font_size=self.style_attr["font size"],
                    font_thickness=self.style_attr["font thickness"],
                    line_width=self.style_attr["line width"],
                    animal_names=animal_names,
                    clf_attr=self.clf_attr,
                    save_path=os.path.join(
                        self.path_plot_dir, f"{self.video_name}_final_frame.png"
                    ),
                )

            bg = self.style_attr["bg color"]
            if self.video_setting or self.frame_setting:
                if self.input_style_attr is not None:
                    self.capture = cv2.VideoCapture(self.video_path)
                    if (type(self.input_style_attr["bg color"]) == dict) and (
                        self.input_style_attr["bg color"]["type"]
                    ) == "static":
                        bg = read_frm_of_video(
                            video_path=self.capture,
                            opacity=self.input_style_attr["bg color"]["opacity"],
                            frame_index=self.input_style_attr["bg color"][
                                "frame_index"
                            ],
                        )

                for frm_cnt in range(1, line_data[0].shape[0]):
                    plot_arrs = [x[:frm_cnt, :] for x in line_data]
                    if self.input_style_attr is not None:
                        if (type(self.input_style_attr["bg color"]) == dict) and (
                            self.input_style_attr["bg color"]["type"] == "moving"
                        ):
                            bg = read_frm_of_video(
                                video_path=self.capture,
                                opacity=self.input_style_attr["bg color"]["opacity"],
                                frame_index=frm_cnt,
                            )

                    self.clf_attr_cpy = deepcopy(self.clf_attr)
                    if self.clf_attr is not None:
                        for k, v in self.clf_attr.items():
                            self.clf_attr_cpy[k]["clfs"][frm_cnt + 1 :] = 0

                    img = PlottingMixin.make_path_plot(
                        data=plot_arrs,
                        colors=colors,
                        width=self.style_attr["width"],
                        height=self.style_attr["height"],
                        max_lines=self.style_attr["max lines"],
                        bg_clr=bg,
                        circle_size=self.style_attr["circle size"],
                        font_size=self.style_attr["font size"],
                        font_thickness=self.style_attr["font thickness"],
                        line_width=self.style_attr["line width"],
                        animal_names=animal_names,
                        clf_attr=self.clf_attr_cpy,
                        save_path=None,
                    )

                    if self.video_setting:
                        self.writer.write(np.uint8(img))
                    if self.frame_setting:
                        frm_name = os.path.join(
                            self.save_video_folder, str() + f"{frm_cnt}.png"
                        )
                        cv2.imwrite(frm_name, np.uint8(img))
                    print(
                        f"Path frame: {frm_cnt + 1} / {line_data[0].shape[0]}) created. Video: {self.video_name} ({str(file_cnt + 1)}/{len(self.files_found)})"
                    )
                if self.video_setting:
                    self.writer.release()
                video_timer.stop_timer()
                print(
                    f"Path visualization for video {self.video_name} saved (elapsed time {video_timer.elapsed_time_str}s)..."
                )

        self.timer.stop_timer()
        stdout_success(
            msg=f"Path visualizations for {len(self.files_found)} video(s) saved in {self.path_plot_dir} directory",
            elapsed_time=self.timer.elapsed_time_str,
            source=self.__class__.__name__,
        )


#
# style_attr = {'width': 'As input',
#               'height': 'As input',
#               'line width': 2,
#               'font size': 0.9,
#               'font thickness': 2,
#               'circle size': 5,
#               'bg color': {'type': 'moving', 'opacity': 50, 'frame_index': 200}, #{'type': 'static', 'opacity': 100, 'frame_index': 200}
#               'max lines': 'entire video'}
# #
# animal_attr = {0: {'bp': 'Ear_right_1', 'color': (255, 0, 0)}, 1: {'bp': 'Ear_right_2', 'color': (0, 0, 255)}}  #['Ear_right_1', 'Red'], 1: ['Ear_right_2', 'Green']}
# clf_attr = {'Nose to Nose': {'color': (155, 1, 10), 'size': 30}, 'Nose to Tailbase': {'color': (155, 90, 10), 'size': 30}}
# style_attr = None
# test = PathPlotterSingleCore(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/project_config.ini',
#                              frame_setting=True,
#                              video_setting=True,
#                              last_frame=True,
#                              slicing=None,#{'start_time': '00:00:00', 'end_time': '00:00:05'}, #{'start_time': '00:00:00', 'end_time': '00:00:50'}, #{'start_time': '00:00:00', 'end_time': '00:00:01'},, #{'start_time': '00:00:00', 'end_time': '00:00:01'}, #{'start_time': '00:00:00', 'end_time': '00:00:01'},
#                              input_style_attr=style_attr,
#                              animal_attr=animal_attr,
#                              clf_attr=clf_attr,
#                              print_animal_names=True,
#                              files_found=['/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/csv/machine_results/Trial    10.csv'])
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


# style_attr = {'width': 'As input',
#               'height': 'As input',
#               'line width': 5,
#               'font size': 5,
#               'font thickness': 2,
#               'circle size': 5,
#               'bg color': {'type': 'static', 'opacity': 70},
#               'max lines': 'entire video'}
# # #
# animal_attr = {0: ['LM_Ear_right_1', 'Red'], 1: ['UM_Ear_right_2', 'Green']}
# #clf_attr = {0: ['Attack', 'Black', 'Size: 30'], 1: ['Sniffing', 'Black', 'Size: 30']}
#
# test = PathPlotterSingleCore(config_path='/Users/simon/Desktop/envs/troubleshooting/dorian_2/project_folder/project_config.ini',
#                              frame_setting=False,
#                              video_setting=False,
#                              last_frame=True,
#                              input_style_attr=style_attr,
#                              animal_attr=animal_attr,
#                              input_clf_attr=None,
#                              files_found=['/Users/simon/Desktop/envs/troubleshooting/dorian_2/project_folder/csv/machine_results/HybCD1-B2-D6-Urine.csv'])
# test.run()
