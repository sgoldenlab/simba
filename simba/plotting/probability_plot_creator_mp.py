__author__ = "Simon Nilsson; sronilsson@gmail.com"

import functools
import multiprocessing
import os
import platform
import shutil
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_file_exist_and_readable, check_float, check_instance, check_int,
    check_str, check_that_column_exist, check_valid_boolean, check_valid_tuple)
from simba.utils.data import terminate_cpu_pool
from simba.utils.enums import Formats
from simba.utils.errors import NoSpecifiedOutputError
from simba.utils.lookups import get_color_dict
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    find_core_cnt, get_fn_ext, read_df)

STYLE_WIDTH = 'width'
STYLE_HEIGHT = 'height'
STYLE_FONT_SIZE = 'font size'
STYLE_LINE_WIDTH = 'line width'
STYLE_YMAX = 'y_max'
STYLE_COLOR = 'color'
AUTO = 'AUTO'
STYLE_OPACITY = 'opacity'

VALID_COLORS = list(get_color_dict().keys())
FOURCC = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)

STYLE_ATTR = [STYLE_WIDTH, STYLE_HEIGHT, STYLE_FONT_SIZE, STYLE_LINE_WIDTH, STYLE_COLOR, STYLE_YMAX, STYLE_OPACITY]

def _probability_plot_mp(frm_range: Tuple[int, np.ndarray],
                         clf_data: np.ndarray,
                         clf_name: str,
                         video_setting: bool,
                         frame_setting: bool,
                         video_dir: str,
                         frame_dir: str,
                         fps: int,
                         video_name: str,
                         y_max: Union[int, float],
                         size: tuple,
                         line_width: int,
                         font_size: int,
                         opacity: float,
                         color: str,
                         show_thresholds: bool):



    group, data = frm_range[0], frm_range[1]
    start_frm, end_frm, current_frm = data[0], data[-1], data[0]

    if video_setting:
        fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        video_save_path = os.path.join(video_dir, f"{group}.mp4")
        video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, size)

    while current_frm < end_frm:
        current_lst = [np.array(clf_data[0 : current_frm + 1])]
        current_frm += 1
        img = PlottingMixin.make_line_plot(data=current_lst,
                                           colors=[color],
                                           width=size[0],
                                           height=size[1],
                                           line_width=line_width,
                                           font_size=font_size,
                                           line_opacity=opacity,
                                           y_lbl=f"{clf_name} probability",
                                           title=f'{video_name} - {clf_name}',
                                           y_max=y_max,
                                           x_lbl='frame count',
                                           show_thresholds=show_thresholds)

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
                 data_path: Union[List[Union[str, os.PathLike]], str, os.PathLike],
                 clf_name: str,
                 frame_setting: Optional[bool] = False,
                 video_setting: Optional[bool] = False,
                 last_frame: Optional[bool] = True,
                 size: Tuple[int, int] = (640, 480),
                 font_size: int = 10,
                 line_width: int = 2,
                 y_max: Optional[int] = None,
                 line_color: str = 'Red',
                 line_opacity: float = 0.8,
                 cores: Optional[int] = -1,
                 show_thresholds: bool = True):

        if platform.system() == "Darwin":
            multiprocessing.set_start_method("spawn", force=True)
        if (not video_setting) and (not frame_setting) and (not last_frame):
            raise NoSpecifiedOutputError(msg="SIMBA ERROR: Please choose to create video and/or frames data plots. SimBA found that you ticked neither video and/or frames")
        check_int(name=f"{self.__class__.__name__} core_cnt", value=cores, min_value=-1, max_value=find_core_cnt()[0], unaccepted_vals=[0])
        if cores == -1: cores = find_core_cnt()[0]
        check_valid_tuple(x=size, source=f'{self.__class__.__name__} size', accepted_lengths=(2,), valid_dtypes=Formats.INTEGER_DTYPES.value, min_integer=100)
        check_int(name=f'{self.__class__.__name__} font_size', value=font_size, min_value=1, raise_error=True)
        check_int(name=f'{self.__class__.__name__} line_width', value=line_width, min_value=1, raise_error=True)
        check_valid_boolean(value=show_thresholds, source=f'{self.__class__.__name__} show_thresholds')
        if y_max is not None:
            check_float(name=f'{self.__class__.__name__} y_max', value=y_max, min_value=0.00001, raise_error=True)
        check_str(name=f'{self.__class__.__name__} color', value=line_color, options=VALID_COLORS)
        check_float(name=f'{self.__class__.__name__} line_opacity', value=line_opacity, min_value=0.001, max_value=1.0, raise_error=True)
        ConfigReader.__init__(self, config_path=config_path)
        PlottingMixin.__init__(self)
        check_str(name=f"{self.__class__.__name__} clf_name", value=clf_name, options=(self.clf_names))
        self.frame_setting, self.video_setting, self.last_image = frame_setting, video_setting, last_frame
        self.line_opacity, self.line_clr, self.line_width = line_opacity, line_color, line_width
        self.font_size, self.img_size, self.y_max = font_size, size, y_max
        check_instance(source=f'{self.__class__.__name__} data_path' , instance=data_path, accepted_types=(str, list,), raise_error=True)
        if isinstance(data_path, str):
            data_path = [data_path]
        for path in data_path:
            check_file_exist_and_readable(file_path=path, raise_error=True)
        check_str(name=f"{self.__class__.__name__} clf_name", value=clf_name, options=(self.clf_names))
        self.show_thresholds = show_thresholds
        self.frame_setting, self.video_setting, self.cores, self.last_frame = (frame_setting, video_setting, cores, last_frame)
        self.clf_name, self.data_paths = clf_name, data_path
        self.probability_col, self.img_size = f"Probability_{self.clf_name}", size
        if not os.path.exists(self.probability_plot_dir): os.makedirs(self.probability_plot_dir)
        print(f"Processing {len(self.data_paths)} video(s)...")

    def run(self):
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        for file_cnt, file_path in enumerate(self.data_paths):
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
            y_max = deepcopy(self.y_max) if self.y_max is not None else float(np.max(clf_data))

            if self.last_frame:
                final_frm_save_path = os.path.join(self.probability_plot_dir, f'{self.video_name}_{self.clf_name}_final_frm_{self.datetime}.png')
                _ = PlottingMixin.make_line_plot(data=[clf_data],
                                                       colors=[self.line_clr],
                                                       width=self.img_size[0],
                                                       height=self.img_size[1],
                                                       line_width=self.line_width,
                                                       font_size=self.font_size,
                                                       y_lbl=f"{self.clf_name} probability",
                                                       y_max=y_max,
                                                       x_lbl='frame count',
                                                       title=f'{self.video_name} - {self.clf_name}',
                                                       save_path=final_frm_save_path,
                                                       line_opacity=self.line_opacity,
                                                       show_thresholds=self.show_thresholds)

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
                                                  video_name=self.video_name,
                                                  y_max=y_max,
                                                  size=self.img_size,
                                                  line_width=self.line_width,
                                                  font_size=self.font_size,
                                                  opacity=self.line_opacity,
                                                  color=self.line_clr,
                                                  show_thresholds=self.show_thresholds)

                    for cnt, result in enumerate(pool.imap(constants, frm_range, chunksize=self.multiprocess_chunksize)):
                        print(f"Core batch {result} complete...")

                terminate_cpu_pool(pool=pool, force=False)
                if self.video_setting:
                    print(f"Joining {self.video_name} multiprocessed video...")
                    concatenate_videos_in_folder(in_folder=self.temp_folder, save_path=self.save_video_path)

                video_timer.stop_timer()
                print(f"Probability video {self.video_name} complete (elapsed time: {video_timer.elapsed_time_str}s) ...")

        self.timer.stop_timer()
        stdout_success(msg=f"Probability visualizations for {str(len(self.data_paths))} videos created in {self.probability_plot_dir} directory", elapsed_time=self.timer.elapsed_time_str,)


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

# if __name__ == "__main__":
#     test = TresholdPlotCreatorMultiprocess(config_path=r"C:\troubleshooting\sleap_two_animals\project_folder\project_config.ini",
#                                             frame_setting=True,
#                                             video_setting=False,
#                                             last_frame=True,
#                                             clf_name='Attack',
#                                             data_path=[r"C:\troubleshooting\sleap_two_animals\project_folder\csv\machine_results\Together_1.csv"],
#                                             size = (640, 480),
#                                             font_size=10,
#                                             line_width=6,
#                                             line_color='Orange',
#                                             y_max=None,
#                                             line_opacity=0.8,
#                                             cores=4)
#     test.run()
#

