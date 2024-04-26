__author__ = "Simon Nilsson"

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import functools
import multiprocessing
import os
import platform
import shutil
from typing import Dict, List, Optional, Union

import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_file_exist_and_readable, check_if_filepath_list_is_empty,
    check_if_keys_exist_in_dict, check_int, check_that_column_exist)
from simba.utils.data import detect_bouts
from simba.utils.enums import Formats
from simba.utils.errors import NoSpecifiedOutputError
from simba.utils.lookups import get_named_colors
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    find_core_cnt, get_fn_ext, read_df)

HEIGHT = "height"
WIDTH = "width"
FONT_ROTATION = "font rotation"
FONT_SIZE = "font size"
STYLE_KEYS = [HEIGHT, WIDTH, FONT_ROTATION, FONT_SIZE]


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

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        style_attr: Dict[str, int],
        files_found: List[Union[str, os.PathLike]],
        frame_setting: Optional[bool] = False,
        video_setting: Optional[bool] = False,
        last_frm_setting: Optional[bool] = True,
        cores: Optional[int] = -1,
    ):

        check_file_exist_and_readable(file_path=config_path)
        if (not frame_setting) and (not video_setting) and (not last_frm_setting):
            raise NoSpecifiedOutputError(
                msg="SIMBA ERROR: Please select gantt videos, frames, and/or last frame.",
                source=self.__class__.__name__,
            )
        check_if_filepath_list_is_empty(
            filepaths=files_found,
            error_msg="SIMBA ERROR: Zero files found in the project_folder/csv/machine_results directory. Create classification results before visualizing gantt charts",
        )
        check_int(
            name=f"{self.__class__.__name__} core_cnt",
            value=cores,
            min_value=-1,
            max_value=find_core_cnt()[0],
        )
        if cores == -1:
            cores = find_core_cnt()[0]

        ConfigReader.__init__(self, config_path=config_path)
        PlottingMixin.__init__(self)

        (
            self.frame_setting,
            self.video_setting,
            self.files_found,
            self.style_attr,
            self.cores,
            self.last_frm_setting,
        ) = (
            frame_setting,
            video_setting,
            files_found,
            style_attr,
            cores,
            last_frm_setting,
        )
        if not os.path.exists(self.gantt_plot_dir):
            os.makedirs(self.gantt_plot_dir)
        check_if_keys_exist_in_dict(
            data=style_attr,
            key=STYLE_KEYS,
            name=f"{self.__class__.__name__} style_attr",
        )
        self.y_rotation, self.y_fontsize = (
            self.style_attr["font rotation"],
            self.style_attr["font size"],
        )
        self.out_width, self.out_height = (
            self.style_attr["width"],
            self.style_attr["height"],
        )
        self.colours = get_named_colors()[:-1]
        self.colour_tuple_x = list(np.arange(3.5, 203.5, 5))
        if platform.system() == "Darwin":
            multiprocessing.set_start_method("spawn", force=True)
        print(f"Processing {len(self.files_found)} video(s)...")

    def run(self):
        check_all_file_names_are_represented_in_video_log(
            video_info_df=self.video_info_df, data_paths=self.files_found
        )
        for file_cnt, file_path in enumerate(self.files_found):
            video_timer = SimbaTimer(start=True)
            _, self.video_name, _ = get_fn_ext(file_path)
            self.data_df = read_df(file_path, self.file_type)
            check_that_column_exist(
                df=self.data_df, column_name=self.clf_names, file_name=file_path
            )
            print(
                f"Processing video {self.video_name}, Frame count: {len(self.data_df)} (Video {(file_cnt + 1)}/{len(self.files_found)})..."
            )
            self.video_info_settings, _, self.fps = self.read_video_info(
                video_name=self.video_name
            )
            self.bouts_df = detect_bouts(
                data_df=self.data_df, target_lst=list(self.clf_names), fps=int(self.fps)
            )
            self.temp_folder = os.path.join(
                self.gantt_plot_dir, self.video_name, "temp"
            )
            self.save_frame_folder_dir = os.path.join(
                self.gantt_plot_dir, self.video_name
            )
            if self.frame_setting:
                if os.path.exists(self.save_frame_folder_dir):
                    shutil.rmtree(self.save_frame_folder_dir)
                os.makedirs(self.save_frame_folder_dir)
            if self.video_setting:
                self.video_folder = os.path.join(self.gantt_plot_dir, self.video_name)
                if os.path.exists(self.temp_folder):
                    shutil.rmtree(self.temp_folder)
                    shutil.rmtree(self.video_folder)
                os.makedirs(self.temp_folder)
                self.save_video_path = os.path.join(
                    self.gantt_plot_dir, f"{self.video_name}.mp4"
                )

            if self.last_frm_setting:
                self.make_gantt_plot(
                    data_df=self.data_df,
                    bouts_df=self.bouts_df,
                    clf_names=self.clf_names,
                    fps=self.fps,
                    style_attr=self.style_attr,
                    video_name=self.video_name,
                    save_path=os.path.join(
                        self.gantt_plot_dir, f"{self.video_name}_final_image.png"
                    ),
                )

            if self.video_setting or self.frame_setting:
                frame_array = np.array_split(
                    list(range(0, len(self.data_df))), self.cores
                )
                frm_per_core = len(frame_array[0])
                for group_cnt, rng in enumerate(frame_array):
                    frame_array[group_cnt] = np.insert(rng, 0, group_cnt)
                print(
                    f"Creating gantt, multiprocessing (chunksize: {(self.multiprocess_chunksize)}, cores: {self.cores})..."
                )
                with multiprocessing.Pool(
                    self.cores, maxtasksperchild=self.maxtasksperchild
                ) as pool:
                    constants = functools.partial(
                        self.gantt_creator_mp,
                        video_setting=self.video_setting,
                        frame_setting=self.frame_setting,
                        video_save_dir=self.temp_folder,
                        frame_folder_dir=self.save_frame_folder_dir,
                        bouts_df=self.bouts_df,
                        clf_names=self.clf_names,
                        fps=self.fps,
                        rotation=self.y_rotation,
                        colors=self.colours,
                        color_tuple=self.colour_tuple_x,
                        font_size=self.y_fontsize,
                        width=self.out_width,
                        height=self.out_height,
                        video_name=self.video_name,
                    )
                    for cnt, result in enumerate(
                        pool.imap(
                            constants,
                            frame_array,
                            chunksize=self.multiprocess_chunksize,
                        )
                    ):
                        print(
                            "Image {}/{}, Video {}/{}...".format(
                                str(int(frm_per_core * (result + 1))),
                                str(len(self.data_df)),
                                str(file_cnt + 1),
                                str(len(self.files_found)),
                            )
                        )
                    pool.terminate()
                    pool.join()

                if self.video_setting:
                    print("Joining {} multiprocessed video...".format(self.video_name))
                    concatenate_videos_in_folder(
                        in_folder=self.temp_folder, save_path=self.save_video_path
                    )
                video_timer.stop_timer()
                print(
                    "Gantt video {} complete (elapsed time: {}s) ...".format(
                        self.video_name, video_timer.elapsed_time_str
                    )
                )

        self.timer.stop_timer()
        stdout_success(
            msg=f"Gantt visualizations for {len(self.files_found)} videos created in project_folder/frames/output/gantt_plots directory",
            elapsed_time=self.timer.elapsed_time_str,
        )


# test = GanttCreatorMultiprocess(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/project_config.ini',
#                                 frame_setting=True,
#                                 video_setting=False,
#                                 files_found=['/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/csv/machine_results/Trial    10.csv'],
#                                 cores=5,
#                                 last_frm_setting=False,
#                                 style_attr={'width': 640, 'height': 480, 'font size': 10, 'font rotation': 65})
# test.run()


# style_attr = {'width': 640, 'height': 480, 'font size': 12, 'font rotation': 45}
# test = GanttCreatorMultiprocess(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals/project_folder/project_config.ini',
#                                  frame_setting=False,
#                                  video_setting=True,
#                                  last_frm_setting=False,
#                                  style_attr=style_attr,
# cores=5,
#                                  files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals/project_folder/csv/machine_results/Together_1.csv'])
# test.create_gannt()
