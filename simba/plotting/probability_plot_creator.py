__author__ = "Simon Nilsson"

import os
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
from simba.utils.enums import Formats
from simba.utils.errors import NoSpecifiedOutputError
from simba.utils.lookups import get_color_dict
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, read_df

VALID_COLORS = list(get_color_dict().keys())
FOURCC = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)

class TresholdPlotCreatorSingleProcess(ConfigReader, PlottingMixin):
    """
    Create line chart visualizations displaying the classification probabilities of a single classifier.

    .. note::
       `Documentation <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-11-visualizations>`_.
       For improved run-time, use :meth:`simba.plotting.probability_plot_creator_mp.TresholdPlotCreatorMultiprocess`

    .. image:: _static/img/prob_plot.png
       :width: 300
       :align: center

    :param str config_path: path to SimBA project config file in Configparser format
    :param str clf_name: Name of the classifier to create visualizations for
    :param bool frame_setting: When True, SimBA creates indidvidual frames in png format
    :param bool video_setting: When True, SimBA creates compressed video in mp4 format
    :param bool last_image: When True, creates image .png representing last frame of the video.
    :param dict style_attr: User-defined style attributes of the visualization (line size, color etc).
    :param List[str] files_found: Files to create threshold plots for.

    Examples
    -----
    >>> style_attr = {'width': 640, 'height': 480, 'font size': 10, 'line width': 6, 'color': 'blue', 'circle size': 20}
    >>> clf_name='Attack'
    >>> files_found=['/_test/project_folder/csv/machine_results/Together_1.csv']

    >>> threshold_plot_creator = TresholdPlotCreatorSingleProcess(config_path='/_test/project_folder/project_config.ini', frame_setting=False, video_setting=True, last_frame=True, clf_name=clf_name, files_found=files_found, style_attr=style_attr)
    >>> threshold_plot_creator.run()
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
                 show_thresholds: bool = True):


        if ((not frame_setting) and (not video_setting) and (not last_frame)):
            raise NoSpecifiedOutputError(msg="Please choose to either probability videos, probability frames, and/or last frame.")

        check_file_exist_and_readable(file_path=config_path, raise_error=True)
        check_valid_tuple(x=size, source=f'{self.__class__.__name__} size', accepted_lengths=(2,), valid_dtypes=Formats.INTEGER_DTYPES.value, min_integer=100)
        check_int(name=f'{self.__class__.__name__} font_size', value=font_size, min_value=1, raise_error=True)
        check_int(name=f'{self.__class__.__name__} line_width', value=line_width, min_value=1, raise_error=True)
        if y_max is not None:
            check_float(name=f'{self.__class__.__name__} y_max', value=y_max, min_value=0.00001, raise_error=True)
        check_str(name=f'{self.__class__.__name__} color', value=line_color, options=VALID_COLORS)
        check_float(name=f'{self.__class__.__name__} line_opacity', value=line_opacity, min_value=0.001, max_value=1.0, raise_error=True)
        check_valid_boolean(value=show_thresholds, source=f'{self.__class__.__name__} show_thresholds')
        ConfigReader.__init__(self, config_path=config_path)
        PlottingMixin.__init__(self)
        check_str(name=f"{self.__class__.__name__} clf_name", value=clf_name, options=(self.clf_names))
        self.frame_setting, self.video_setting, self.last_image = frame_setting, video_setting, last_frame
        self.line_opacity, self.line_clr, self.line_width = line_opacity, line_color, line_width
        self.font_size, self.img_size, self.y_max, self.show_thresholds = font_size, size, y_max, show_thresholds
        check_instance(source=f'{self.__class__.__name__} data_path' , instance=data_path, accepted_types=(str, list,), raise_error=True)
        if isinstance(data_path, str):
            data_path = [data_path]
        for path in data_path:
            check_file_exist_and_readable(file_path=path, raise_error=True)
        self.data_paths, self.orginal_clf_name = data_path, clf_name
        self.clf_name = f"Probability_{self.orginal_clf_name}"
        if not os.path.exists(self.probability_plot_dir): os.makedirs(self.probability_plot_dir)
        print(f"Processing probability plots for {len(self.data_paths)} video(s)...")
        self.timer = SimbaTimer(start=True)

    def run(self):
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        for file_cnt, file_path in enumerate(self.data_paths):
            video_timer = SimbaTimer(start=True)
            _, self.video_name, _ = get_fn_ext(file_path)
            video_info, self.px_per_mm, fps = self.read_video_info(video_name=self.video_name)
            data_df = read_df(file_path, self.file_type)
            check_that_column_exist(df=data_df, column_name=self.clf_name, file_name=self.video_name)
            if self.frame_setting:
                self.save_frame_folder_dir = os.path.join(self.probability_plot_dir, f"{self.video_name}_{self.orginal_clf_name}")
                if not os.path.exists(self.save_frame_folder_dir): os.makedirs(self.save_frame_folder_dir)
            if self.video_setting:
                self.save_video_path = os.path.join(self.probability_plot_dir, f"{self.video_name}_{self.orginal_clf_name}.mp4")
                self.writer = cv2.VideoWriter(self.save_video_path, FOURCC, fps, self.img_size)

            clf_data = data_df[self.clf_name].values
            y_max = deepcopy(self.y_max) if self.y_max is not None else float(np.max(clf_data))
            if self.last_image:
                final_frm_save_path = os.path.join(self.probability_plot_dir, f'{self.video_name}_{self.orginal_clf_name}_final_frm_{self.datetime}.png')
                _ = PlottingMixin.make_line_plot(data=[clf_data],
                                                       colors=[self.line_clr],
                                                       width=self.img_size[0],
                                                       height=self.img_size[1],
                                                       line_width=self.line_width,
                                                       font_size=self.font_size,
                                                       y_lbl=f"{self.orginal_clf_name} probability",
                                                       y_max=y_max,
                                                       x_lbl='frame count',
                                                       title=f'{self.video_name} - {self.clf_name}',
                                                       save_path=final_frm_save_path,
                                                       line_opacity=self.line_opacity,
                                                       show_thresholds=self.show_thresholds)

            if self.video_setting or self.frame_setting:
                for i in range(1, clf_data.shape[0]):
                    frm_data = clf_data[0:i]
                    img = PlottingMixin.make_line_plot(data=[frm_data],
                                                       colors=[self.line_clr],
                                                       width=self.img_size[0],
                                                       height=self.img_size[1],
                                                       line_width=self.line_width,
                                                       font_size=self.font_size,
                                                       y_lbl=f"{self.orginal_clf_name} probability",
                                                       y_max = y_max,
                                                       x_lbl = 'frame count',
                                                       title = f'{self.video_name} - {self.clf_name}',
                                                       save_path = None,
                                                       line_opacity = self.line_opacity,
                                                       show_thresholds = self.show_thresholds)
                    if self.frame_setting:
                        frame_save_path = os.path.join(self.save_frame_folder_dir, f"{i}.png")
                        cv2.imwrite(frame_save_path, img)
                    if self.video_setting:
                        self.writer.write(img.astype(np.uint8)[:, :, :3])
                    print(f"Probability frame: {i+1} / {len(data_df)}. Video: {self.video_name} (File {file_cnt + 1}/{len(self.data_paths)})")
                if self.video_setting:
                    self.writer.release()
                video_timer.stop_timer()
                print(f"Probability plot for video {self.video_name} saved (elapsed time: {video_timer.elapsed_time_str}s)...")
        self.timer.stop_timer()
        stdout_success(msg=f"All probability visualizations created in {self.probability_plot_dir} directory", elapsed_time=self.timer.elapsed_time_str)


# test = TresholdPlotCreatorSingleProcess(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                         frame_setting=False,
#                                         video_setting=True,
#                                         last_image=True,
#                                         clf_name='Attack',
#                                         files_found=['/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'],
#                                         style_attr={'width': 640, 'height': 480, 'font size': 10, 'line width': 10, 'color': 'Orange', 'circle size': 20, 'y_max': 'auto'})
# test.run()

# test = TresholdPlotCreatorSingleProcess(config_path=r"C:\troubleshooting\sleap_two_animals\project_folder\project_config.ini",
#                                         frame_setting=False,
#                                         video_setting=True,
#                                         last_frame=True,
#                                         clf_name='Attack',
#                                         data_path=[r"C:\troubleshooting\sleap_two_animals\project_folder\csv\machine_results\Together_1.csv"],
#                                         size = (640, 480),
#                                         font_size=10,
#                                         line_width=6,
#                                         line_color='Orange',
#                                         y_max=None,
#                                         line_opacity=0.8)
# test.run()
#

#
