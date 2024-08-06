__author__ = "Simon Nilsson"


import glob
import os
from copy import deepcopy
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.mixins.train_model_mixin import TrainModelMixin
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_keys_exist_in_dict, check_int,
                                check_valid_boolean,
                                check_video_and_data_frm_count_align)
from simba.utils.data import create_color_palette
from simba.utils.enums import ConfigKey, Dtypes, Formats, TagNames, TextOptions
from simba.utils.errors import NoSpecifiedOutputError
from simba.utils.printing import log_event, stdout_success
from simba.utils.read_write import (get_fn_ext, get_video_meta_data,
                                    read_config_entry, read_df)

CIRCLE_SCALE = 'circle_scale'
FONT_SIZE = 'font_size'
SPACE_SCALE = 'spacing_scale'
TEXT_THICKNESS = 'text_thickness'
TEXT_SETTING_KEYS = ['circle_scale', 'font_size', 'spacing_scale', 'text_thickness']
CENTER_BP_TXT = ['centroid', 'center']


class PlotSklearnResultsSingleCore(ConfigReader, TrainModelMixin, PlottingMixin):
    """
    Plot classification results overlays on videos. Results are stored in the
    `project_folder/frames/output/sklearn_results` directory of the SimBA project.

    .. note::
       For improved run-time, see :meth:`simba.plotting.plot_clf_results_mp.PlotSklearnResultsMultiProcess` for multiprocess class.
       Scikit visualization documentation <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-10-sklearn-visualization__.

    .. image:: _static/img/sklearn_visualization.gif
       :width: 600
       :align: center

    :param Union[str, os.PathLike] config_path: path to SimBA project config file in Configparser format
    :param Optional[bool] video_setting: If True, SimBA will create compressed videos. Default True.
    :param Optional[bool] frame_setting: If True, SimBA will create individual frames. Default True.
    :param Optional[str] video_file_path: Path to video file to create classification visualizations for. If None, then all the videos in the csv/machine_results will be used. Default None.
    :param Optional[Union[Dict[str, float], bool]] text_settings: Dictionary holding the circle size, font size, spacing size, and text thickness of the printed text. If None, then these are autocomputed.
    :param Optional[bool] rotate: If True, the output video will be rotated 90 degrees from the input. Default False.
    :param Optional[str] palette: The name of the palette used for the pose-estimation key-points. Default ``Set1``.
    :param Optional[bool] print_timers: If True, the output video will have the cumulative time of the classified behaviours overlaid. Default True.

    :example:
    >>> text_settings = {'circle_scale': 5, 'font_size': 5, 'spacing_scale': 2, 'text_thickness': 10}
    >>> test = PlotSklearnResultsSingleCore(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
    >>>                                       video_setting=True,
    >>>                                       frame_setting=False,
    >>>                                       video_file_path='Together_1.avi',
    >>>                                       print_timers=True,
    >>>                                       text_settings=text_settings,
    >>>                                       rotate=False)
    >>> test.run()

    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 video_setting: Optional[bool] = True,
                 frame_setting: Optional[bool] = False,
                 text_settings: Optional[Union[Dict[str, float], bool]] = False,
                 video_file_path: Optional[List] = None,
                 palette: Optional[str] = 'Set1',
                 rotate: Optional[bool] = False,
                 print_timers: Optional[bool] = True):

        ConfigReader.__init__(self, config_path=config_path)
        TrainModelMixin.__init__(self)
        PlottingMixin.__init__(self)
        log_event(logger_name=str(__class__.__name__), log_type=TagNames.CLASS_INIT.value, msg=self.create_log_msg_from_init_args(locals=locals()))
        if not isinstance(text_settings, (bool)) and text_settings != None:
            check_if_keys_exist_in_dict(data=text_settings, key=TEXT_SETTING_KEYS)
        for i in [video_setting, frame_setting, rotate, print_timers]:
            check_valid_boolean(value=i, source=self.__class__.__name__, raise_error=True)
        if (not video_setting) and (not frame_setting):
            raise NoSpecifiedOutputError(msg="Please choose to create a video and/or frames. SimBA found that you ticked neither video and/or frames", source=self.__class__.__name__)
        self.video_file_path, self.print_timers, self.text_settings = (video_file_path, print_timers, text_settings)
        self.video_setting, self.frame_setting = video_setting, frame_setting
        if video_file_path is not None:
            check_file_exist_and_readable(os.path.join(self.video_dir, video_file_path))
        if not os.path.exists(self.sklearn_plot_dir):
            os.makedirs(self.sklearn_plot_dir)
        self.pose_threshold = read_config_entry(self.config, ConfigKey.THRESHOLD_SETTINGS.value, ConfigKey.SKLEARN_BP_PROB_THRESH.value, Dtypes.FLOAT.value, 0.00)
        self.clr_lst = create_color_palette(pallete_name=palette, increments=self.clf_cnt)
        self.files_found = glob.glob(self.machine_results_dir + "/*." + self.file_type)
        self.model_dict = self.get_model_info(self.config, self.clf_cnt)
        self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        self.rotate = rotate
        self.a = np.deg2rad(90)
        print(f"Processing {str(len(self.files_found))} videos...")

    def __get_print_settings(self):
        if (self.text_settings is False) or (self.text_settings is None):
            img_width, img_height = self.video_meta_data["width"], self.video_meta_data["height"]
            longest_str = max(['TIMERS:', 'ENSEMBLE PREDICTION:'] + self.clf_names, key=len)
            self.font_size, _, self.spacing_scale = self.get_optimal_font_scales(text=longest_str, accepted_px_width=int(img_width / 3), accepted_px_height=int(img_width / 10), text_thickness=2)
            self.circle_scale = self.get_optimal_circle_size(frame_size=(img_width, img_height), circle_frame_ratio=100)
            self.text_thickness = TextOptions.TEXT_THICKNESS.value
        else:
            check_float(name="TEXT SIZE", value=self.text_settings[FONT_SIZE])
            check_int(name="SPACE SIZE", value=self.text_settings[SPACE_SCALE])
            check_int(name="TEXT THICKNESS", value=self.text_settings[TEXT_THICKNESS])
            check_int(name="CIRCLE SIZE", value=self.text_settings[CIRCLE_SCALE])
            self.font_size = float(self.text_settings[FONT_SIZE])
            self.spacing_scale = int(self.text_settings[SPACE_SCALE])
            self.text_thickness = int(self.text_settings[TEXT_THICKNESS])
            self.circle_scale = int(self.text_settings[CIRCLE_SCALE])

    def create_visualizations(self):
        _, self.video_name, _ = get_fn_ext(self.file_path)
        self.data_df = read_df(self.file_path, self.file_type).reset_index(drop=True).fillna(0)
        self.video_settings, _, self.fps = self.read_video_info(video_name=self.video_name)
        self.video_path = self.find_video_of_file(self.video_dir, self.video_name)
        self.cap = cv2.VideoCapture(self.video_path)
        self.save_path = os.path.join(self.sklearn_plot_dir, self.video_name + ".mp4")
        self.video_meta_data = get_video_meta_data(self.video_path)
        height, width = deepcopy(self.video_meta_data["height"]), deepcopy(self.video_meta_data["width"])
        if self.frame_setting:
            self.video_frame_dir = os.path.join(self.sklearn_plot_dir, self.video_name)
            if not os.path.exists(self.video_frame_dir):
                os.makedirs(self.video_frame_dir)
        if self.rotate:
            self.video_meta_data["height"], self.video_meta_data["width"] = (width, height)
        self.writer = cv2.VideoWriter(self.save_path, self.fourcc, self.fps, (self.video_meta_data["width"], self.video_meta_data["height"]))
        self.__get_print_settings()
        self.video_model_dict = deepcopy(self.model_dict)
        for model in self.video_model_dict:
            self.video_model_dict[model]["time"] = 0
        check_video_and_data_frm_count_align(video=self.cap, data=self.data_df, name=self.video_name, raise_error=False)

        row_n = 0
        while self.cap.isOpened():
            ret, self.frame = self.cap.read()
            try:
                if ret:
                    self.id_flag_cords = {}
                    for animal_name, animal_data in self.animal_bp_dict.items():
                        animal_clr = animal_data["colors"]
                        ID_flag = False
                        for bp_no in range(len(animal_data["X_bps"])):
                            bp_clr = animal_clr[bp_no]
                            x_bp, y_bp = (animal_data["X_bps"][bp_no], animal_data["Y_bps"][bp_no])
                            p_bp = x_bp[:-2] + "_p"
                            bp_cords = self.data_df.loc[row_n, [x_bp, y_bp, p_bp]]
                            if bp_cords[p_bp] > self.pose_threshold:
                                cv2.circle(self.frame, (int(bp_cords[x_bp]), int(bp_cords[y_bp])), 0, bp_clr, self.circle_scale)
                                if ("centroid" in x_bp.lower()) or ("center" in x_bp.lower()):
                                    self.id_flag_cords[animal_name] = (int(bp_cords[x_bp]), int(bp_cords[y_bp]))
                                    ID_flag = True
                        if not ID_flag:
                            self.id_flag_cords[animal_name] = (int(bp_cords[x_bp]), int(bp_cords[y_bp]))

                    for animal_name, animal_cords in self.id_flag_cords.items():
                        cv2.putText(self.frame, animal_name, animal_cords, self.font, self.font_size, self.animal_bp_dict[animal_name]["colors"][0], self.text_thickness)
                    if self.rotate:
                        self.frame = np.array(Image.fromarray(self.frame).rotate(90, Image.BICUBIC, expand=True))
                    if self.print_timers:
                        self.frame = PlottingMixin().put_text(img=self.frame, text="TIMERS:", pos=(TextOptions.BORDER_BUFFER_Y.value, ((self.video_meta_data["height"] - self.video_meta_data["height"]) + self.spacing_scale)), font_size=self.font_size, font_thickness=self.text_thickness, font=self.font)
                    self.add_spacer = 2
                    for model_no, model_info in self.video_model_dict.items():
                        frame_results = self.data_df.loc[row_n, model_info["model_name"]]
                        self.video_model_dict[model_no]["frame_results"] = frame_results
                        self.video_model_dict[model_no]["time"] += (frame_results / self.fps)
                        if self.print_timers:
                            self.frame = PlottingMixin().put_text(img=self.frame, text=model_info["model_name"] + " " + str(round(self.video_model_dict[model_no]["time"], 2)), pos=(TextOptions.BORDER_BUFFER_Y.value, ((self.video_meta_data["height"] - self.video_meta_data["height"]) + self.spacing_scale * self.add_spacer)), font_size=self.font_size, font_thickness=self.text_thickness, font=self.font)
                            self.add_spacer += 1
                    self.frame = PlottingMixin().put_text(img=self.frame, text="ENSEMBLE PREDICTION:", pos=(TextOptions.BORDER_BUFFER_Y.value, ((self.video_meta_data["height"] - self.video_meta_data["height"]) + self.spacing_scale * self.add_spacer)), font_size=self.font_size, font_thickness=self.text_thickness, font=self.font)
                    self.add_spacer += 1
                    for model_cnt, model_info in self.video_model_dict.items():
                        if self.video_model_dict[model_cnt]["frame_results"] == 1:
                            self.frame = PlottingMixin().put_text(img=self.frame, text=model_info["model_name"], pos=(TextOptions.BORDER_BUFFER_Y.value, (self.video_meta_data["height"] - self.video_meta_data["height"]) + self.spacing_scale * self.add_spacer), font_size=self.font_size, font_thickness=self.text_thickness, font=self.font, text_color=TextOptions.COLOR.value)
                            self.add_spacer += 1
                    if self.video_setting:
                        self.writer.write(self.frame)
                    if self.frame_setting:
                        frame_save_name = os.path.join(self.video_frame_dir, str(row_n) + ".png")
                        cv2.imwrite(frame_save_name, self.frame)
                    print(f'Frame: {row_n} / {self.video_meta_data["frame_count"]}. Video: {self.video_name} ({self.file_cnt + 1}/{len(self.files_found)})')
                    row_n += 1
                else:
                    print(f"Video {self.video_name} saved...")
                    self.cap.release()
                    self.writer.release()
            except KeyError as e:
                print(e.args, e)
                print("SIMBA INDEX WARNING: Some frames appears to be missing in the dataframe and could not be created")
                print(f"Video {self.video_name} saved...")
                self.cap.release()
                self.writer.release()

    def run(self):
        if self.video_file_path is None:
            for file_cnt, file_path in enumerate(self.files_found):
                self.file_cnt, self.file_path = file_cnt, file_path
                self.create_visualizations()
        else:
            self.file_cnt, file_path = 0, self.video_file_path
            _, file_name, _ = get_fn_ext(file_path)
            self.file_path = os.path.join(self.machine_results_dir, file_name + f".{self.file_type}")
            self.files_found = [self.file_path]
            check_file_exist_and_readable(self.file_path)
            self.create_visualizations()

        self.timer.stop_timer()
        stdout_success(msg=f"All visualizations created in {self.sklearn_plot_dir} directory", elapsed_time=self.timer.elapsed_time_str, source=self.__class__.__name__)


# text_settings = {'circle_scale': 5, 'font_size': 5, 'spacing_scale': 2, 'text_thickness': 10}
# test = PlotSklearnResultsSingleCore(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                       video_setting=True,
#                                       frame_setting=False,
#                                       video_file_path='Together_1.avi',
#                                       print_timers=True,
#                                       text_settings=None,
#                                       rotate=False)
# test.run()
