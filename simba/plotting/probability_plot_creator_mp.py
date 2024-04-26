__author__ = "Simon Nilsson"

import functools
import multiprocessing
import os
import platform
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_if_keys_exist_in_dict, check_int, check_str, check_that_column_exist,
    check_valid_lst)
from simba.utils.enums import Formats
from simba.utils.errors import NoSpecifiedOutputError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    find_core_cnt, get_fn_ext, read_df)

STYLE_WIDTH = 'width'
STYLE_HEIGHT = 'height'
STYLE_FONT_SIZE = 'font size'
STYLE_LINE_WIDTH = 'line width'
STYLE_YMAX = 'y_max'
STYLE_COLOR = 'color'

STYLE_ATTR = [STYLE_WIDTH, STYLE_HEIGHT, STYLE_FONT_SIZE, STYLE_LINE_WIDTH, STYLE_COLOR, STYLE_YMAX]

def _probability_plot_mp(frm_range: Tuple[int, np.ndarray],
                         clf_data: np.ndarray,
                         clf_name: str,
                         video_setting: bool,
                         frame_setting: bool,
                         video_dir: str,
                         frame_dir: str,
                         fps: int,
                         style_attr: dict,
                         video_name: str):



    group, data = frm_range[0], frm_range[1]
    start_frm, end_frm, current_frm = data[0], data[-1], data[0]

    if video_setting:
        fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        video_save_path = os.path.join(video_dir, f"{group}.mp4")
        video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, (style_attr[STYLE_WIDTH], style_attr[STYLE_HEIGHT]))

    while current_frm < end_frm:
        current_lst = [np.array(clf_data[0 : current_frm + 1])]
        current_frm += 1
        img = PlottingMixin.make_line_plot_plotly(data=current_lst,
                                                  colors=[style_attr[STYLE_COLOR]],
                                                  width=style_attr[STYLE_WIDTH],
                                                  height=style_attr[STYLE_HEIGHT],
                                                  line_width=style_attr[STYLE_LINE_WIDTH],
                                                  font_size=style_attr[STYLE_FONT_SIZE],
                                                  y_lbl=f"{clf_name} probability",
                                                  title=clf_name,
                                                  y_max=style_attr[STYLE_YMAX],
                                                  x_lbl='frame count')

        if video_setting:
            video_writer.write(img[:, :, :3])
        if frame_setting:
            frame_save_name = os.path.join(frame_dir, f"{current_frm}.png")
            cv2.imwrite(frame_save_name, img)
        current_frm += 1
        print(f"Probability frame created: {current_frm + 1}, Video: {video_name}, Processing core: {group}")
    return group


class TresholdPlotCreatorMultiprocess(ConfigReader, PlottingMixin):
    """
    Class for line chart visualizations displaying the classification probabilities of a single classifier.
    Uses multiprocessing.

    :param str config_path: path to SimBA project config file in Configparser format
    :param str clf_name: Name of the classifier to create visualizations for
    :param bool frame_setting: When True, SimBA creates indidvidual frames in png format
    :param bool video_setting: When True, SimBA creates compressed video in mp4 format
    :param bool last_image: When True, creates image .png representing last frame of the video.
    :param dict style_attr: User-defined style attributes of the visualization (line size, color etc).
    :param List[str] files_found: Files to create threshold plots for.
    :param int cores: Number of cores to use.

    .. note::
       `Visualization tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-11-visualizations>`__.

    .. image:: _static/img/prob_plot.png
       :width: 300
       :align: center

    :example:
    >>> plot_creator = TresholdPlotCreatorMultiprocess(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini', frame_setting=True, video_setting=True, clf_name='Attack', style_attr={'width': 640, 'height': 480, 'font size': 10, 'line width': 6, 'color': 'magneta', 'circle size': 20}, cores=5)
    >>> plot_creator.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 files_found: List[Union[str, os.PathLike]],
                 style_attr: Dict[str, Any],
                 clf_name: str,
                 frame_setting: Optional[bool] = False,
                 video_setting: Optional[bool] = False,
                 last_frame: Optional[bool] = True,
                 cores: Optional[int] = -1):

        if platform.system() == "Darwin":
            multiprocessing.set_start_method("spawn", force=True)
        if (not video_setting) and (not frame_setting) and (not last_frame):
            raise NoSpecifiedOutputError(msg="SIMBA ERROR: Please choose to create video and/or frames data plots. SimBA found that you ticked neither video and/or frames")
        check_valid_lst(data=files_found, source=self.__class__.__name__, valid_dtypes=(str,), min_len=1)
        check_int(name=f"{self.__class__.__name__} core_cnt", value=cores, min_value=-1, max_value=find_core_cnt()[0])
        if cores == -1: cores = find_core_cnt()[0]
        check_if_keys_exist_in_dict(data=style_attr, key=STYLE_ATTR, name=f'{self.__class__.__name__} style_attr')
        ConfigReader.__init__(self, config_path=config_path)
        PlottingMixin.__init__(self)
        check_str(name=f"{self.__class__.__name__} clf_name", value=clf_name, options=(self.clf_names))
        self.frame_setting, self.video_setting, self.cores, self.style_attr, self.last_frame = (frame_setting, video_setting, cores, style_attr, last_frame)
        self.clf_name, self.files_found = clf_name, files_found
        self.probability_col = f"Probability_{self.clf_name}"
        self.out_width, self.out_height = (self.style_attr["width"], self.style_attr["height"],)
        self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        if not os.path.exists(self.probability_plot_dir):
            os.makedirs(self.probability_plot_dir)
        print(f"Processing {len(self.files_found)} video(s)...")

    def run(self):
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.files_found)
        for file_cnt, file_path in enumerate(self.files_found):
            video_timer = SimbaTimer(start=True)
            _, self.video_name, _ = get_fn_ext(file_path)
            video_info, self.px_per_mm, self.fps = self.read_video_info(video_name=self.video_name)
            data_df = read_df(file_path, self.file_type)
            check_that_column_exist(df=data_df, column_name=[self.clf_name, self.probability_col], file_name=file_path)
            self.save_frame_folder_dir = os.path.join(self.probability_plot_dir, self.video_name + f"_{self.clf_name}")
            self.video_folder = os.path.join(self.probability_plot_dir, self.video_name + f"_{self.clf_name}")
            self.temp_folder = os.path.join(self.probability_plot_dir, f"{self.video_name}_{self.clf_name}", "temp")
            if self.frame_setting:
                if os.path.exists(self.save_frame_folder_dir):
                    shutil.rmtree(self.save_frame_folder_dir)
                os.makedirs(self.save_frame_folder_dir)
            if self.video_setting:
                if os.path.exists(self.temp_folder):
                    shutil.rmtree(self.temp_folder)
                    shutil.rmtree(self.video_folder)
                os.makedirs(self.temp_folder)
                self.save_video_path = os.path.join(self.probability_plot_dir, f"{self.video_name}_{self.clf_name}.mp4")

            clf_data = data_df[self.probability_col].values
            if self.style_attr[STYLE_YMAX] == 'auto': self.style_attr[STYLE_YMAX] = np.max(clf_data)
            if self.last_frame:
                final_frm_save_path = os.path.join(self.probability_plot_dir, f'{self.video_name}_{self.clf_name}_final_frm_{self.datetime}.png')
                _ = PlottingMixin.make_line_plot_plotly(data=[clf_data],
                                                        colors=[self.style_attr[STYLE_COLOR]],
                                                        width=self.style_attr[STYLE_WIDTH],
                                                        height=self.style_attr[STYLE_HEIGHT],
                                                        line_width=self.style_attr[STYLE_LINE_WIDTH],
                                                        font_size=self.style_attr[STYLE_FONT_SIZE],
                                                        y_lbl=f"{self.clf_name} probability",
                                                        y_max= self.style_attr[STYLE_YMAX],
                                                        x_lbl='frame count',
                                                        title=self.clf_name,
                                                        save_path=final_frm_save_path)

            if self.video_setting or self.frame_setting:
                frm_nums = np.arange(0, len(data_df)+1)
                data_split = np.array_split(frm_nums, self.cores)
                frm_range = []
                for cnt, i in enumerate(data_split): frm_range.append((cnt, i))
                print(f"Creating probability images, multiprocessing (chunksize: {self.multiprocess_chunksize}, cores: {self.cores})...")
                with multiprocessing.Pool(self.cores, maxtasksperchild=self.maxtasksperchild) as pool:
                    constants = functools.partial(_probability_plot_mp,
                                                  clf_name=self.clf_name,
                                                  clf_data=clf_data,
                                                  video_setting=self.video_setting,
                                                  frame_setting=self.frame_setting,
                                                  fps=self.fps,
                                                  video_dir=self.temp_folder,
                                                  frame_dir=self.save_frame_folder_dir,
                                                  style_attr=self.style_attr,
                                                  video_name=self.video_name)
                    for cnt, result in enumerate(pool.imap(constants, frm_range, chunksize=self.multiprocess_chunksize)):
                        print(f"Core batch {result} complete...")

                pool.join()
                pool.terminate()
                if self.video_setting:
                    print(f"Joining {self.video_name} multiprocessed video...")
                    concatenate_videos_in_folder(in_folder=self.temp_folder, save_path=self.save_video_path)

                video_timer.stop_timer()
                print(f"Probability video {self.video_name} complete (elapsed time: {video_timer.elapsed_time_str}s) ...")

        self.timer.stop_timer()
        stdout_success(msg=f"Probability visualizations for {str(len(self.files_found))} videos created in {self.probability_plot_dir} directory", elapsed_time=self.timer.elapsed_time_str,)


# test = TresholdPlotCreatorMultiprocess(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/project_config.ini',
#                                         frame_setting=False,
#                                         video_setting=True,
#                                         last_frame=True,
#                                         clf_name='Nose to Nose',
#                                         cores=-1,
#                                         files_found=['/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/csv/machine_results/Trial    10.csv'],
#                                         style_attr={'width': 640, 'height': 480, 'font size': 10, 'line width': 6, 'color': 'Red', 'circle size': 20, 'y_max': 'auto'})
# #test = TresholdPlotCreatorMultiprocess(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini', frame_setting=False, video_setting=True, clf_name='Attack')
# test.run()


# test = TresholdPlotCreatorMultiprocess(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                         frame_setting=False,
#                                         video_setting=True,
#                                         last_frame=True,
#                                         clf_name='Attack',
#                                         cores=5,
#                                         files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'],
#                                         style_attr={'width': 640, 'height': 480, 'font size': 10, 'line width': 3, 'color': 'blue', 'circle size': 20, 'y_max': 'auto'})
# test.create_plots()
