__author__ = "Simon Nilsson"

import time
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import functools
import multiprocessing
import os
import platform
from typing import List, Optional, Union

import cv2
import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_file_exist_and_readable, check_int, check_str,
    check_that_column_exist, check_valid_lst)
from simba.utils.data import create_color_palette, detect_bouts
from simba.utils.enums import Formats, Options
from simba.utils.errors import NoSpecifiedOutputError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    create_directory, find_core_cnt,
                                    get_fn_ext, read_df)

HEIGHT = "height"
WIDTH = "width"
FONT_ROTATION = "font rotation"
FONT_SIZE = "font size"
STYLE_KEYS = [HEIGHT, WIDTH, FONT_ROTATION, FONT_SIZE]

def gantt_creator_mp(data: np.array,
                     frame_setting: bool,
                     video_setting: bool,
                     video_save_dir: str,
                     frame_folder_dir: str,
                     bouts_df: pd.DataFrame,
                     clf_names: list,
                     fps: int,
                     video_name: str,
                     width: int,
                     height: int,
                     font_size: int,
                     font_rotation: int,
                     palette: np.ndarray):

    batch_id, frame_rng = data[0], data[1]
    start_frm, end_frm, current_frm = frame_rng[0], frame_rng[-1], frame_rng[0]
    video_writer = None
    if video_setting:
        fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        video_save_path = os.path.join(video_save_dir, f"{batch_id}.mp4")
        video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, (width, height))

    while current_frm <= end_frm:
        bout_rows = bouts_df.loc[bouts_df["End_frame"] <= current_frm]
        plot = PlottingMixin().make_gantt_plot(x_length=current_frm + 1,
                                               bouts_df=bout_rows,
                                               clf_names=clf_names,
                                               fps=fps,
                                               width=width,
                                               height=height,
                                               font_size=font_size,
                                               font_rotation=font_rotation,
                                               video_name=video_name,
                                               save_path=None,
                                               palette=palette)
        current_frm += 1
        if frame_setting:
            frame_save_path = os.path.join(frame_folder_dir, f"{current_frm}.png")
            cv2.imwrite(frame_save_path, plot)
        if video_setting:
            video_writer.write(plot)
        print(f"Gantt frame created: {current_frm + 1}, Video: {video_name}, Processing core: {batch_id + 1}")

    if video_setting:
        video_writer.release()

    return batch_id


class GanttCreatorMultiprocess(ConfigReader, PlottingMixin):
    """
    Multiprocess creation of classifier gantt charts in video and/or image format.
    See meth:`simba.gantt_creator.GanttCreatorSingleProcess` for single-process class.

    ..note::
       `GitHub gantt tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#gantt-plot>`__.

    .. image:: _static/img/gantt_plot.png
       :width: 300
       :align: center

    :param str config_path: path to SimBA project config file in Configparser format.
    :param bool frame_setting: If True, creates individual frames.
    :param bool last_frm_setting: If True, creates single .png image representing entire video.
    :param bool video_setting: If True, creates videos
    :param dict style_attr: Attributes of gannt chart (size, font size, font rotation etc).
    :param List[str] files_found: File paths representing files with machine predictions e.g., ['project_folder/csv/machine_results/My_results.csv']
    :param int cores: Number of cores to use.

    :examples:
    >>> gantt_creator = GanttCreatorMultiprocess(config_path='project_folder/project_config.ini', frame_setting=False, video_setting=True, files_found=['project_folder/csv/machine_results/Together_1.csv'], cores=5, style_attr={'width': 640, 'height': 480, 'font size': 8, 'font rotation': 45}).run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_paths: List[Union[str, os.PathLike]],
                 frame_setting: Optional[bool] = False,
                 video_setting: Optional[bool] = False,
                 last_frm_setting: Optional[bool] = True,
                 width: int = 640,
                 height: int = 480,
                 font_size: int = 8,
                 font_rotation: int = 45,
                 palette: str = 'Set1',
                 core_cnt: Optional[int] = -1):

        check_file_exist_and_readable(file_path=config_path)
        if (not frame_setting) and (not video_setting) and (not last_frm_setting):
            raise NoSpecifiedOutputError(msg="SIMBA ERROR: Please select gantt videos, frames, and/or last frame.", source=self.__class__.__name__)
        check_file_exist_and_readable(file_path=config_path)
        check_valid_lst(data=data_paths, source=f'{self.__class__.__name__} data_paths', valid_dtypes=(str,), min_len=1)
        check_int(value=width, min_value=1, name=f'{self.__class__.__name__} width')
        check_int(value=height, min_value=1, name=f'{self.__class__.__name__} height')
        check_int(value=font_size, min_value=1, name=f'{self.__class__.__name__} font_size')
        check_int(value=font_rotation, min_value=0, max_value=180, name=f'{self.__class__.__name__} font_rotation')
        palettes = Options.PALETTE_OPTIONS_CATEGORICAL.value + Options.PALETTE_OPTIONS.value
        check_str(name=f'{self.__class__.__name__} palette', value=palette, options=palettes)
        for file_path in data_paths: check_file_exist_and_readable(file_path=file_path)
        check_int(name=f"{self.__class__.__name__} core_cnt",value=core_cnt, min_value=-1, unaccepted_vals=[0], max_value=find_core_cnt()[0])
        self.core_cnt = find_core_cnt()[0] if core_cnt == -1 or core_cnt > find_core_cnt()[0] else core_cnt
        self.width, self.height, self.font_size, self.font_rotation = width, height, font_size, font_rotation
        ConfigReader.__init__(self, config_path=config_path, create_logger=False)
        PlottingMixin.__init__(self)
        self.clr_lst = create_color_palette(pallete_name=palette, increments=len(self.body_parts_lst) + 1, as_int=True, as_rgb_ratio=True)
        self.frame_setting, self.video_setting, self.data_paths, self.last_frm_setting = frame_setting, video_setting,data_paths, last_frm_setting
        if not os.path.exists(self.gantt_plot_dir): os.makedirs(self.gantt_plot_dir)
        if platform.system() == "Darwin":
            multiprocessing.set_start_method("spawn", force=True)
        print(f"Processing {len(self.data_paths)} video(s)...")

    def run(self):
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        for file_cnt, file_path in enumerate(self.data_paths):
            video_timer = SimbaTimer(start=True)
            _, self.video_name, _ = get_fn_ext(file_path)
            self.data_df = read_df(file_path, self.file_type)
            check_that_column_exist(df=self.data_df, column_name=self.clf_names, file_name=file_path)
            print(f"Processing video {self.video_name}, Frame count: {len(self.data_df)} (Video {(file_cnt + 1)}/{len(self.data_paths)})...")
            self.video_info_settings, _, self.fps = self.read_video_info(video_name=self.video_name)
            self.bouts_df = detect_bouts(data_df=self.data_df, target_lst=list(self.clf_names), fps=int(self.fps))
            self.temp_folder = os.path.join(self.gantt_plot_dir, self.video_name, "temp")
            self.save_frame_folder_dir = os.path.join(self.gantt_plot_dir, self.video_name)
            if self.frame_setting:
                create_directory(paths=self.save_frame_folder_dir, overwrite=True)
            if self.video_setting:
                create_directory(paths=self.temp_folder)
                self.save_video_path = os.path.join(self.gantt_plot_dir, f"{self.video_name}.mp4")

            if self.last_frm_setting:
                self.make_gantt_plot(x_length=len(self.data_df),
                                     bouts_df=self.bouts_df,
                                     clf_names=self.clf_names,
                                     fps=self.fps,
                                     width=self.width,
                                     height=self.height,
                                     font_size=self.font_size,
                                     font_rotation=self.font_rotation,
                                     video_name=self.video_name,
                                     save_path=os.path.join(self.gantt_plot_dir, f"{self.video_name}_final_image.png"),
                                     palette=self.clr_lst)

            if self.video_setting or self.frame_setting:
                frame_data = np.array_split(list(range(0, len(self.data_df))), self.core_cnt)
                frame_data = [(i, x) for i, x in enumerate(frame_data)]
                print(f"Creating gantt, multiprocessing (chunksize: {(self.multiprocess_chunksize)}, cores: {self.core_cnt})...")
                with multiprocessing.Pool(self.core_cnt, maxtasksperchild=self.maxtasksperchild) as pool:
                    constants = functools.partial(gantt_creator_mp,
                                                  video_setting=self.video_setting,
                                                  frame_setting=self.frame_setting,
                                                  video_save_dir=self.temp_folder,
                                                  frame_folder_dir=self.save_frame_folder_dir,
                                                  bouts_df=self.bouts_df,
                                                  clf_names=self.clf_names,
                                                  fps=self.fps,
                                                  width=self.width,
                                                  height=self.height,
                                                  font_size=self.font_size,
                                                  font_rotation=self.font_rotation,
                                                  video_name=self.video_name,
                                                  palette=self.clr_lst)
                    for cnt, result in enumerate(pool.imap(constants, frame_data, chunksize=self.multiprocess_chunksize)):
                        print(f'Batch {result+1/self.core_cnt} complete...')
                pool.terminate()
                pool.join()
                if self.video_setting:
                    print(f"Joining {self.video_name} multiprocessed video...")
                    concatenate_videos_in_folder(in_folder=self.temp_folder, save_path=self.save_video_path)
                video_timer.stop_timer()
                print(f"Gantt video {self.video_name} complete (elapsed time: {video_timer.elapsed_time_str}s) ...")

        self.timer.stop_timer()
        stdout_success(msg=f"Gantt visualizations for {len(self.data_paths)} videos created in {self.gantt_plot_dir} directory", elapsed_time=self.timer.elapsed_time_str)


# if __name__ == "__main__":
#     test = GanttCreatorMultiprocess(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini",
#                                     frame_setting=False,
#                                     video_setting=False,
#                                     data_paths=[r"C:\troubleshooting\mitra\project_folder\csv\machine_results\592_MA147_Gq_CNO_0515.csv"],
#                                     last_frm_setting=True,
#                                     width=640,
#                                     height= 480,
#                                     font_size=10,
#                                     font_rotation= 45)
#     test.run()




# test = GanttCreatorMultiprocess(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/project_config.ini',
#                                 frame_setting=False,
#                                 video_setting=True,
#                                 data_paths=['/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/csv/machine_results/2022-06-20_NOB_DOT_4.csv'],
#                                 cores=5,
#                                 last_frm_setting=False,
#                                 style_attr={'width': 640, 'height': 480, 'font size': 10, 'font rotation': 65})
# test.run()


# test = GanttCreatorMultiprocess(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                 frame_setting=False,
#                                 video_setting=True,
#                                 data_paths=['/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'],
#                                 last_frm_setting=True,
#                                 style_attr={'width': 640, 'height': 480, 'font size': 10, 'font rotation': 65},
#                                 cores=2)
# test.run()
