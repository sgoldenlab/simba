__author__ = "Simon Nilsson"

import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.mixins.train_model_mixin import TrainModelMixin
from simba.utils.checks import (check_float, check_if_valid_rgb_tuple,
                                check_str, check_that_column_exist,
                                check_valid_boolean,
                                check_video_and_data_frm_count_align)
from simba.utils.data import create_color_palette
from simba.utils.enums import (ConfigKey, Dtypes, Formats, Options, TagNames,
                               TextOptions)
from simba.utils.errors import (InvalidInputError, NoDataError,
                                NoSpecifiedOutputError)
from simba.utils.printing import log_event, stdout_success
from simba.utils.read_write import (find_all_videos_in_project, get_fn_ext,
                                    get_video_meta_data, read_config_entry,
                                    read_df)
from simba.utils.warnings import FrameRangeWarning

CIRCLE_SCALE = 'circle_scale'
FONT_SIZE = 'font_size'
SPACE_SCALE = 'spacing_scale'
TEXT_THICKNESS = 'text_thickness'
TEXT_SETTING_KEYS = ['circle_scale', 'font_size', 'spacing_scale', 'text_thickness']
CENTER_BP_TXT = ['centroid', 'center']

FOURCC = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)

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
                 video_setting: bool = True,
                 frame_setting: bool = False,
                 video_paths: Optional[Union[List[Union[str, os.PathLike]], Union[str, os.PathLike]]] = None,
                 rotate: bool = False,
                 animal_names: bool = False,
                 show_pose: bool = True,
                 font_size: Optional[Union[int, float]] = None,
                 space_size: Optional[Union[int, float]] = None,
                 text_opacity: Optional[Union[int, float]] = None,
                 text_thickness: Optional[Union[int, float]] = None,
                 circle_size: Optional[Union[int, float]] = None,
                 pose_palette: Optional[str] = 'Set1',
                 print_timers: bool = True,
                 text_clr: Tuple[int, int,int] = (255, 255, 255),
                 text_bg_clr: Tuple[int, int,int] = (0, 0, 0)):

        ConfigReader.__init__(self, config_path=config_path)
        TrainModelMixin.__init__(self)
        PlottingMixin.__init__(self)
        log_event(logger_name=str(__class__.__name__), log_type=TagNames.CLASS_INIT.value, msg=self.create_log_msg_from_init_args(locals=locals()))
        for i in [video_setting, frame_setting, rotate, print_timers, animal_names, show_pose]:
            check_valid_boolean(value=i, source=self.__class__.__name__, raise_error=True)
        if (not video_setting) and (not frame_setting):
            raise NoSpecifiedOutputError(msg="Please choose to create a video and/or frames. SimBA found that you ticked neither video and/or frames", source=self.__class__.__name__)
        if font_size is not None: check_float(name=f'{self.__class__.__name__} font_size', value=font_size, min_value=0.1)
        if space_size is not None: check_float(name=f'{self.__class__.__name__} space_size', value=space_size, min_value=0.1)
        if text_thickness is not None: check_float(name=f'{self.__class__.__name__} text_thickness', value=text_thickness, min_value=0.1)
        if circle_size is not None: check_float(name=f'{self.__class__.__name__} text_thickness', value=circle_size, min_value=0.1)
        if circle_size is not None: check_float(name=f'{self.__class__.__name__} text_thickness', value=circle_size, min_value=0.1)
        if text_opacity is not None: check_float(name=f'{self.__class__.__name__} text_opacity', value=text_opacity, min_value=0.1)
        check_if_valid_rgb_tuple(data=text_bg_clr, source=f'{self.__class__.__name__} text_bg_clr')
        check_if_valid_rgb_tuple(data=text_clr, source=f'{self.__class__.__name__} text_clr')
        self.text_color, self.text_bg_color = text_clr, text_bg_clr
        self.video_paths, self.print_timers = video_paths, print_timers
        if self.video_paths is None:
            self.video_paths = find_all_videos_in_project(videos_dir=self.video_dir)
            if len(self.video_paths) == 0:
                raise NoDataError(msg=f'Cannot create classification videos. No videos exist in {self.video_dir} directory', source=self.__class__.__name__)
        self.video_setting, self.frame_setting, self.rotate, self.text_opacity = video_setting, frame_setting, rotate, text_opacity
        self.circle_size, self.font_size, self.animal_names = circle_size, font_size, animal_names
        self.text_thickness, self.space_size, self.show_pose = text_thickness, space_size, show_pose
        self.pose_threshold = read_config_entry(self.config, ConfigKey.THRESHOLD_SETTINGS.value, ConfigKey.SKLEARN_BP_PROB_THRESH.value, Dtypes.FLOAT.value, 0.00)
        if not os.path.exists(self.sklearn_plot_dir):
            os.makedirs(self.sklearn_plot_dir)
        pose_palettes = Options.PALETTE_OPTIONS_CATEGORICAL.value + Options.PALETTE_OPTIONS.value
        check_str(name=f'{self.__class__.__name__} pose_palette', value=pose_palette, options=pose_palettes)
        self.clr_lst = create_color_palette(pallete_name=pose_palette, increments=len(self.body_parts_lst)+1)
        if isinstance(self.video_paths, str): self.video_paths = [video_paths]
        elif isinstance(self.video_paths, list): self.video_paths = video_paths
        else:
            raise InvalidInputError(msg=f'video_paths has to be a path of a list of paths. Got {type(video_paths)}', source=self.__class__.__name__)
        for video_path in self.video_paths:
            video_name = get_fn_ext(filepath=video_path)[1]
            data_path = os.path.join(self.machine_results_dir, f'{video_name}.{self.file_type}')
            if not os.path.isfile(data_path): raise NoDataError(msg=f'Cannot create classification videos for {video_name}. Expected classification data at location {data_path} but file does not exist', source=self.__class__.__name__)


    def __get_print_settings(self):
        self.video_text_thickness = TextOptions.TEXT_THICKNESS.value if self.text_thickness is None else int(max(self.text_thickness, 1))
        longest_str = str(max(['TIMERS:', 'ENSEMBLE PREDICTION:'] + self.clf_names, key=len))
        optimal_font_size, _, optimal_spacing_scale = self.get_optimal_font_scales(text=longest_str, accepted_px_width=int(self.video_meta_data["width"] / 3), accepted_px_height=int(self.video_meta_data["height"] / 10), text_thickness=self.video_text_thickness)
        optimal_circle_size = self.get_optimal_circle_size(frame_size=(self.video_meta_data["width"], self.video_meta_data["height"]), circle_frame_ratio=100)
        self.video_circle_size = optimal_circle_size if self.circle_size is None else int(self.circle_size)
        self.video_font_size = optimal_font_size if self.font_size is None else self.font_size
        self.video_space_size = optimal_spacing_scale if self.space_size is None else int(max(self.space_size, 1))
        self.video_text_opacity = 0.8 if self.text_opacity is None else float(self.text_opacity)

    def run(self):
        for video_cnt, video_path in enumerate(self.video_paths):
            _, self.video_name, _ = get_fn_ext(video_path)
            self.data_path = os.path.join(self.machine_results_dir, f'{self.video_name}.{self.file_type}')
            self.data_df = read_df(self.data_path, self.file_type).reset_index(drop=True).fillna(0)
            if self.show_pose: check_that_column_exist(df=self.data_df, column_name=self.bp_col_names, file_name=self.data_path)
            self.video_meta_data = get_video_meta_data(video_path=video_path)
            height, width = deepcopy(self.video_meta_data["height"]), deepcopy(self.video_meta_data["width"])
            self.save_path = os.path.join(self.sklearn_plot_dir, f"{self.video_name}.mp4")
            if self.frame_setting:
                self.video_frame_dir = os.path.join(self.sklearn_plot_dir, self.video_name)
                if not os.path.exists(self.video_frame_dir): os.makedirs(self.video_frame_dir)
            if self.rotate:
                self.video_meta_data["height"], self.video_meta_data["width"] = (width, height)
            self.clf_timers = {k: 0 for k in self.clf_names}
            check_video_and_data_frm_count_align(video=video_path, data=self.data_df, name=self.video_name, raise_error=False)
            check_that_column_exist(df=self.data_df, column_name=list(self.clf_timers.keys()), file_name=self.data_path)
            self.writer = cv2.VideoWriter(self.save_path, FOURCC, self.video_meta_data['fps'], (self.video_meta_data["width"], self.video_meta_data["height"]))
            self.__get_print_settings()
            self.cap = cv2.VideoCapture(video_path)

            frm_idx = 0
            while self.cap.isOpened():
                ret, self.frame = self.cap.read()
                if ret:
                    clr_cnt = 0
                    for animal_name, animal_data in self.animal_bp_dict.items():
                        if self.show_pose:
                            for bp_num in range(len(animal_data["X_bps"])):
                                x_bp, y_bp, p_bp = (animal_data["X_bps"][bp_num], animal_data["Y_bps"][bp_num], animal_data["P_bps"][bp_num])
                                bp_cords = self.data_df.loc[frm_idx, [x_bp, y_bp, p_bp]]
                                if bp_cords[p_bp] >= self.pose_threshold:
                                    self.frame = cv2.circle(self.frame, (int(bp_cords[x_bp]), int(bp_cords[y_bp])), self.video_circle_size, self.clr_lst[clr_cnt], -1)
                                clr_cnt+=1
                        if self.animal_names:
                            x_bp, y_bp, p_bp = (animal_data["X_bps"][0], animal_data["Y_bps"][0], animal_data["P_bps"][0])
                            bp_cords = self.data_df.loc[frm_idx, [x_bp, y_bp, p_bp]]
                            cv2.putText(self.frame, animal_name, (int(bp_cords[x_bp]), int(bp_cords[y_bp])), self.font, self.video_font_size, self.clr_lst[0], self.video_text_thickness)
                    if self.rotate:
                        self.frame = np.array(Image.fromarray(self.frame).rotate(90, Image.BICUBIC, expand=True))
                    if self.print_timers:
                        self.frame = PlottingMixin().put_text(img=self.frame, text="TIMERS:", pos=(TextOptions.BORDER_BUFFER_Y.value, ((self.video_meta_data["height"] - self.video_meta_data["height"]) + self.video_space_size)), font_size=self.video_font_size, font_thickness=self.video_text_thickness, font=self.font, text_bg_alpha=self.video_text_opacity, text_color_bg=self.text_bg_color, text_color=self.text_color)
                    self.add_spacer = 2
                    for clf_name, clf_time in self.clf_timers.items():
                        frame_results = self.data_df.loc[frm_idx, clf_name]
                        self.clf_timers[clf_name] += (frame_results / self.video_meta_data['fps'])
                        if self.print_timers:
                            self.frame = PlottingMixin().put_text(img=self.frame, text=f"{clf_name} {round(self.clf_timers[clf_name], 2)}", pos=(TextOptions.BORDER_BUFFER_Y.value, ((self.video_meta_data["height"] - self.video_meta_data["height"]) + self.video_space_size * self.add_spacer)), font_size=self.video_font_size, font_thickness=self.video_text_thickness, font=self.font, text_bg_alpha=self.video_text_opacity, text_color_bg=self.text_bg_color, text_color=self.text_color)
                            self.add_spacer += 1
                    self.frame = PlottingMixin().put_text(img=self.frame, text="ENSEMBLE PREDICTION:", pos=(TextOptions.BORDER_BUFFER_Y.value, ((self.video_meta_data["height"] - self.video_meta_data["height"]) + self.video_space_size * self.add_spacer)), font_size=self.video_font_size, font_thickness=self.video_text_thickness, font=self.font, text_bg_alpha=self.video_text_opacity, text_color_bg=self.text_bg_color, text_color=self.text_color)
                    self.add_spacer += 1
                    for clf_name, clf_time in self.clf_timers.items():
                        if self.data_df.loc[frm_idx, clf_name] == 1:
                            self.frame = PlottingMixin().put_text(img=self.frame, text=clf_name, pos=(TextOptions.BORDER_BUFFER_Y.value, (self.video_meta_data["height"] - self.video_meta_data["height"]) + self.video_space_size * self.add_spacer), font_size=self.video_font_size, font_thickness=self.video_text_thickness, font=self.font, text_color=TextOptions.COLOR.value, text_bg_alpha=self.video_text_opacity)
                            self.add_spacer += 1
                    if self.video_setting:
                        self.writer.write(self.frame.astype(np.uint8))
                    if self.frame_setting:
                        frame_save_name = os.path.join(self.video_frame_dir, f"{frm_idx}.png")
                        cv2.imwrite(frame_save_name, self.frame)
                    frm_idx += 1
                    print(f'Frame: {frm_idx} / {self.video_meta_data["frame_count"]}. Video: {self.video_name} ({video_cnt + 1}/{len(self.video_paths)})')
                else:
                    FrameRangeWarning(msg=f'Could not read frame {frm_idx} in video {video_path}. Stoping video creation.')
                    break
            print(f"Video {self.video_name} saved at {self.save_path}...")
            self.cap.release()
            self.writer.release()

        self.timer.stop_timer()
        stdout_success(msg=f"{len(self.video_paths)} visualization(s) created in {self.sklearn_plot_dir} directory", elapsed_time=self.timer.elapsed_time_str, source=self.__class__.__name__)

# test = PlotSklearnResultsSingleCore(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini",
#                                     video_setting=True,
#                                     frame_setting=False,
#                                     video_paths=r"C:\troubleshooting\RAT_NOR\project_folder\videos\03152021_NOB_IOT_8.mp4",
#                                     print_timers=True,
#                                     rotate=True,
#                                     animal_names=True)
# test.run()
