__author__ = "Simon Nilsson"

import os
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_file_exist_and_readable, check_if_keys_exist_in_dict, check_str,
    check_that_column_exist, check_valid_lst)
from simba.utils.enums import Formats
from simba.utils.errors import NoSpecifiedOutputError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, read_df

STYLE_WIDTH = 'width'
STYLE_HEIGHT = 'height'
STYLE_FONT_SIZE = 'font size'
STYLE_LINE_WIDTH = 'line width'
STYLE_YMAX = 'y_max'
STYLE_COLOR = 'color'

STYLE_ATTR = [STYLE_WIDTH, STYLE_HEIGHT, STYLE_FONT_SIZE, STYLE_LINE_WIDTH, STYLE_COLOR, STYLE_YMAX]

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
                 files_found: List[Union[str, os.PathLike]],
                 clf_name: str,
                 style_attr: Dict[str, Any],
                 frame_setting: Optional[bool] = False,
                 video_setting: Optional[bool] = False,
                 last_image: Optional[bool] = True):

        if ((not frame_setting) and (not video_setting) and (not last_image)):
            raise NoSpecifiedOutputError(msg="Please choose to either probability videos, probability frames, and/or last frame.")

        check_file_exist_and_readable(file_path=config_path)
        check_valid_lst(data=files_found, source=self.__class__.__name__, valid_dtypes=(str,))
        print(style_attr)
        check_if_keys_exist_in_dict(data=style_attr, key=STYLE_ATTR, name=f'{self.__class__.__name__} style_attr')
        ConfigReader.__init__(self, config_path=config_path)
        PlottingMixin.__init__(self)
        check_str(name=f"{self.__class__.__name__} clf_name", value=clf_name, options=(self.clf_names))
        self.frame_setting, self.video_setting, self.style_attr, self.last_image = (frame_setting,video_setting,style_attr,last_image)
        self.files_found = files_found
        self.orginal_clf_name = clf_name
        self.clf_name = f"Probability_{self.orginal_clf_name}"
        self.out_width, self.out_height = (self.style_attr[STYLE_WIDTH], self.style_attr[STYLE_HEIGHT])
        self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        if not os.path.exists(self.probability_plot_dir):
            os.makedirs(self.probability_plot_dir)
        print(f"Processing {len(self.files_found)} video(s)...")
        self.timer = SimbaTimer(start=True)

    def run(self):
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.files_found)
        for file_cnt, file_path in enumerate(self.files_found):
            video_timer = SimbaTimer(start=True)
            _, self.video_name, _ = get_fn_ext(file_path)
            video_info, self.px_per_mm, fps = self.read_video_info(video_name=self.video_name)
            data_df = read_df(file_path, self.file_type)
            check_that_column_exist(df=data_df, column_name=self.clf_name, file_name=self.video_name)
            if self.frame_setting:
                self.save_frame_folder_dir = os.path.join(self.probability_plot_dir, f"{self.video_name}_{self.orginal_clf_name}")
                if not os.path.exists(self.save_frame_folder_dir):
                    os.makedirs(self.save_frame_folder_dir)
            if self.video_setting:
                self.save_video_path = os.path.join(self.probability_plot_dir, f"{self.video_name}_{self.orginal_clf_name}.mp4")
                self.writer = cv2.VideoWriter(self.save_video_path, self.fourcc, fps, (self.out_width, self.out_height))


            clf_data = data_df[self.clf_name].values
            if self.style_attr[STYLE_YMAX] == 'auto': self.style_attr[STYLE_YMAX] = np.max(clf_data)
            if self.last_image:
                final_frm_save_path = os.path.join(self.probability_plot_dir, f'{self.video_name}_{self.orginal_clf_name}_final_frm_{self.datetime}.png')
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
                for i in range(1, clf_data.shape[0]):
                    frm_data = clf_data[0:i]
                    img = PlottingMixin.make_line_plot_plotly(data=[frm_data],
                                                              colors=[self.style_attr[STYLE_COLOR]],
                                                              width=self.style_attr[STYLE_WIDTH],
                                                              height=self.style_attr[STYLE_HEIGHT],
                                                              line_width=self.style_attr[STYLE_LINE_WIDTH],
                                                              font_size=self.style_attr[STYLE_FONT_SIZE],
                                                              y_lbl=f"{self.clf_name} probability",
                                                              y_max=self.style_attr[STYLE_YMAX],
                                                              x_lbl='frame count',
                                                              title=self.clf_name,
                                                              save_path=None)
                    if self.frame_setting:
                        frame_save_path = os.path.join(self.save_frame_folder_dir, f"{i}.png")
                        cv2.imwrite(frame_save_path, img)
                    if self.video_setting:
                        self.writer.write(img.astype(np.uint8)[:, :, :3])
                    print(f"Probability frame: {i+1} / {len(data_df)}. Video: {self.video_name} (File {file_cnt + 1}/{len(self.files_found)})")
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

# test = TresholdPlotCreatorSingleProcess(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                         frame_setting=False,
#                                         video_setting=True,
#                                         last_image=True,
#                                         clf_name='Attack',
#                                         files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'],
#                                         style_attr={'width': 640, 'height': 480, 'font size': 10, 'line width': 6, 'color': 'blue', 'circle size': 20, 'y_max': 'auto'})
# test.create_plots()


#
