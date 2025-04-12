__author__ = "Simon Nilsson"

import os
import shutil
from typing import List, Optional, Union

import cv2
import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_file_exist_and_readable, check_int, check_str, check_valid_dataframe,
    check_valid_lst)
from simba.utils.data import create_color_palette, detect_bouts
from simba.utils.enums import Formats, Options
from simba.utils.errors import NoSpecifiedOutputError
from simba.utils.lookups import get_named_colors
from simba.utils.printing import stdout_success
from simba.utils.read_write import get_fn_ext, read_df

STYLE_WIDTH = 'width'
STYLE_HEIGHT = 'height'
STYLE_FONT_SIZE = 'font size'
STYLE_FONT_ROTATION = 'font size'

STYLE_ATTR = [STYLE_WIDTH, STYLE_HEIGHT, STYLE_FONT_SIZE, STYLE_FONT_ROTATION]

class GanttCreatorSingleProcess(ConfigReader, PlottingMixin):
    """
    Create gantt chart videos and/or images using a single core.

    .. note::
       `GitHub gantt tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#gantt-plot>`__.
       For improved run-time, see :meth:`simba.gantt_creator_mp.GanttCreatorMultiprocess` for multiprocess class.

    .. image:: _static/img/gantt_plot.png
       :width: 300
       :align: center


    :param str config_path: path to SimBA project config file in Configparser format.
    :param bool frame_setting: If True, creates individual frames.
    :param bool last_frm_setting: If True, creates single .png image representing entire video.
    :param bool video_setting: If True, creates videos
    :param dict style_attr: Attributes of gannt chart (size, font size, font rotation etc).
    :param List[str] files_found: File paths representing files with machine predictions e.g., ['project_folder/csv/machine_results/My_results.csv']

    :example:
    >>> style_attr = {'width': 640, 'height': 480, 'font size': 12, 'font rotation': 45}
    >>> gantt_creator = GanttCreatorSingleProcess(config_path='tests/test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini', frame_setting=False, video_setting=True, files_found=['tests/test_data/multi_animal_dlc_two_c57/project_folder/csv/machine_results/Together_1.csv'])
    >>> gantt_creator.run()

    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_paths: List[Union[str, os.PathLike]],
                 width: int = 640,
                 height: int = 480,
                 font_size: int = 8,
                 font_rotation: int = 45,
                 palette: str = 'Set1',
                 frame_setting: Optional[bool] = False,
                 video_setting: Optional[bool] = False,
                 last_frm_setting: Optional[bool] = True):

        if ((frame_setting != True) and (video_setting != True) and (last_frm_setting != True)):
            raise NoSpecifiedOutputError(msg="SIMBA ERROR: Please select gantt videos, frames, and/or last frame.")
        check_file_exist_and_readable(file_path=config_path)
        check_int(value=width, min_value=1, name=f'{self.__class__.__name__} width')
        check_int(value=height, min_value=1, name=f'{self.__class__.__name__} height')
        check_int(value=font_size, min_value=1, name=f'{self.__class__.__name__} font_size')
        check_int(value=font_rotation, min_value=0, max_value=180, name=f'{self.__class__.__name__} font_rotation')
        check_valid_lst(data=data_paths, source=f'{self.__class__.__name__} data_paths', valid_dtypes=(str,), min_len=1)
        palettes = Options.PALETTE_OPTIONS_CATEGORICAL.value + Options.PALETTE_OPTIONS.value
        check_str(name=f'{self.__class__.__name__} palette', value=palette, options=palettes)
        for file_path in data_paths: check_file_exist_and_readable(file_path=file_path)
        self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        ConfigReader.__init__(self, config_path=config_path)
        PlottingMixin.__init__(self)
        self.clr_lst = create_color_palette(pallete_name=palette, increments=len(self.body_parts_lst)+1, as_int=True, as_rgb_ratio=True)
        if not os.path.exists(self.gantt_plot_dir): os.makedirs(self.gantt_plot_dir)
        self.frame_setting, self.video_setting, self.last_frm_setting = frame_setting, video_setting, last_frm_setting
        self.width, self.height, self.font_size, self.font_rotation = width, height, font_size, font_rotation
        self.data_paths = data_paths
        self.colours = get_named_colors()
        self.colour_tuple_x = list(np.arange(3.5, 203.5, 5))

    def run(self):
        print(f"Processing gantt visualizations for {len(self.data_paths)} video(s)...")
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        for file_cnt, file_path in enumerate(self.data_paths):
            _, self.video_name, _ = get_fn_ext(file_path)
            self.data_df = read_df(file_path, self.file_type).reset_index(drop=True)
            check_valid_dataframe(df=self.data_df, source=file_path, required_fields=self.clf_names)
            self.video_info_settings, _, self.fps = self.read_video_info(video_name=self.video_name)
            self.bouts_df = detect_bouts(data_df=self.data_df, target_lst=self.clf_names, fps=self.fps)
            if self.frame_setting:
                self.save_frame_folder_dir = os.path.join(self.gantt_plot_dir, self.video_name)
                if os.path.exists(self.save_frame_folder_dir):
                    shutil.rmtree(self.save_frame_folder_dir)
                os.makedirs(self.save_frame_folder_dir)
            if self.video_setting:
                self.save_video_path = os.path.join(self.gantt_plot_dir, f"{self.video_name}.mp4")
                self.writer = cv2.VideoWriter(self.save_video_path, self.fourcc, self.fps, (self.width, self.height))
            if self.last_frm_setting:
                self.make_gantt_plot(x_length=len(self.data_df),
                                     bouts_df=self.bouts_df,
                                     clf_names=self.clf_names,
                                     fps=self.fps,
                                     width = self.width,
                                     height=self.height,
                                     font_size=self.font_size,
                                     font_rotation=self.font_rotation,
                                     video_name=self.video_name,
                                     save_path=os.path.join(self.gantt_plot_dir, f"{self.video_name }_final_image.png"),
                                     palette=self.clr_lst)

            if self.frame_setting or self.video_setting:
                for image_cnt, k in enumerate(range(len(self.data_df))):
                    frm_bout_df = self.bouts_df.loc[self.bouts_df["End_frame"] <= k]
                    plot = self.make_gantt_plot(x_length=k+1,
                                                bouts_df=frm_bout_df,
                                                clf_names=self.clf_names,
                                                fps=self.fps,
                                                width=self.width,
                                                height=self.height,
                                                font_size=self.font_size,
                                                font_rotation=self.font_rotation,
                                                video_name=self.video_name,
                                                palette=self.clr_lst,
                                                save_path=None)
                    if self.frame_setting:
                        frame_save_path = os.path.join(self.save_frame_folder_dir, f"{k}.png")
                        cv2.imwrite(frame_save_path, plot)
                    if self.video_setting:
                        self.writer.write(plot)
                    print(f"Gantt frame: {image_cnt + 1} / {len(self.data_df)}. Video: {self.video_name} ({file_cnt + 1}/{len(self.data_paths)})")
                if self.video_setting:
                    self.writer.release()
                    print(f"Gantt for video {self.video_name} saved...")
            self.timer.stop_timer()
            stdout_success(msg=f"All gantt visualizations created in {self.gantt_plot_dir} directory", elapsed_time=self.timer.elapsed_time_str)


# test = GanttCreatorSingleProcess(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini",
#                                 frame_setting=False,
#                                 video_setting=False,
#                                 data_paths=[r"C:\troubleshooting\mitra\project_folder\csv\machine_results\592_MA147_Gq_CNO_0515.csv"],
#                                 last_frm_setting=True,
#                                 width=640,
#                                 height= 480,
#                                 font_size=10,
#                                 font_rotation=45,
#                                 palette='Set1')
# test.run()




# style_attr = {'width': 640, 'height': 480, 'font size': 12, 'font rotation': 45}
# test = GanttCreatorSingleProcess(config_path='/Users/simon/Desktop/envs/troubleshooting/sleap_5_animals/project_folder/project_config.ini',
#                                  frame_setting=False,
#                                  video_setting=True,
#                                  last_frm_setting=True,
#                                  style_attr=style_attr,
#                                  files_found=['/Users/simon/Desktop/envs/troubleshooting/sleap_5_animals/project_folder/csv/outlier_corrected_movement_location/Testing_Video_3.csv'])
# test.create_gannt()


# style_attr = {'width': 640, 'height': 480, 'font size': 12, 'font rotation': 45}
# test = GanttCreatorSingleProcess(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                  frame_setting=False,
#                                  video_setting=True,
#                                  last_frm_setting=True,
#                                  style_attr=style_attr,
#                                  files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/targets_inserted/Together_1.csv'])
# test.run()
