__author__ = "Simon Nilsson"

import functools
import multiprocessing
import os
import platform
import shutil
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import check_that_column_exist
from simba.utils.enums import Formats
from simba.utils.errors import ColumnNotFoundError, NoSpecifiedOutputError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder, get_fn_ext,
                                    read_df)


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

    Examples
    ----------
    >>> plot_creator = TresholdPlotCreatorMultiprocess(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini', frame_setting=True, video_setting=True, clf_name='Attack', style_attr={'width': 640, 'height': 480, 'font size': 10, 'line width': 6, 'color': 'magneta', 'circle size': 20}, cores=5)
    >>> plot_creator.run()
    """

    def __init__(
        self,
        config_path: str,
        clf_name: str,
        frame_setting: bool,
        video_setting: bool,
        last_frame: bool,
        cores: int,
        style_attr: Dict[str, int],
        files_found: List[str],
    ):
        ConfigReader.__init__(self, config_path=config_path)
        PlottingMixin.__init__(self)
        if platform.system() == "Darwin":
            multiprocessing.set_start_method("spawn", force=True)
        (
            self.frame_setting,
            self.video_setting,
            self.cores,
            self.style_attr,
            self.last_frame,
        ) = (frame_setting, video_setting, cores, style_attr, last_frame)
        if (
            (not self.frame_setting)
            and (not self.video_setting)
            and (not self.last_frame)
        ):
            raise NoSpecifiedOutputError(
                "SIMBA ERROR: Please choose to create either probability videos, frames, and/or last frame.",
                show_window=True,
            )
        self.clf_name, self.files_found = clf_name, files_found
        self.probability_col = "Probability_" + self.clf_name
        self.fontsize = self.style_attr["font size"]
        self.out_width, self.out_height = (
            self.style_attr["width"],
            self.style_attr["height"],
        )
        self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        if not os.path.exists(self.probability_plot_dir):
            os.makedirs(self.probability_plot_dir)
        print(f"Processing {str(len(self.files_found))} video(s)...")

    def run(self):
        for file_cnt, file_path in enumerate(self.files_found):
            video_timer = SimbaTimer(start=True)
            _, self.video_name, _ = get_fn_ext(file_path)
            video_info, self.px_per_mm, self.fps = self.read_video_info(
                video_name=self.video_name
            )
            data_df = read_df(file_path, self.file_type)
            if self.probability_col not in data_df.columns:
                raise ColumnNotFoundError(
                    column_name=self.probability_col, file_name=file_path
                )
            check_that_column_exist(
                df=data_df, column_name=self.clf_name, file_name=self.video_name
            )
            self.save_frame_folder_dir = os.path.join(
                self.probability_plot_dir, self.video_name + "_" + self.clf_name
            )
            self.video_folder = os.path.join(
                self.probability_plot_dir, self.video_name + "_" + self.clf_name
            )
            self.temp_folder = os.path.join(
                self.probability_plot_dir, self.video_name + "_" + self.clf_name, "temp"
            )
            if self.frame_setting:
                if os.path.exists(self.save_frame_folder_dir):
                    shutil.rmtree(self.save_frame_folder_dir)
                if not os.path.exists(self.save_frame_folder_dir):
                    os.makedirs(self.save_frame_folder_dir)
            if self.video_setting:
                if os.path.exists(self.temp_folder):
                    shutil.rmtree(self.temp_folder)
                    shutil.rmtree(self.video_folder)
                os.makedirs(self.temp_folder)
                self.save_video_path = os.path.join(
                    self.probability_plot_dir,
                    "{}_{}.mp4".format(self.video_name, self.clf_name),
                )

            probability_lst = list(data_df[self.probability_col])

            if self.last_frame:
                _ = self.make_probability_plot(
                    data=pd.Series(probability_lst),
                    style_attr=self.style_attr,
                    clf_name=self.clf_name,
                    fps=self.fps,
                    save_path=os.path.join(
                        self.probability_plot_dir,
                        self.video_name
                        + "_{}_{}.png".format(self.clf_name, "final_image"),
                    ),
                )

            if self.video_setting or self.frame_setting:
                if self.style_attr["y_max"] == "auto":
                    highest_p = data_df[self.probability_col].max()
                else:
                    highest_p = float(self.style_attr["y_max"])
                data_split = np.array_split(list(data_df.index), self.cores)
                frm_per_core = len(data_split[0])
                for group_cnt, rng in enumerate(data_split):
                    data_split[group_cnt] = np.insert(rng, 0, group_cnt)

                print(
                    "Creating probability images, multiprocessing (determined chunksize: {}, cores: {})...".format(
                        str(self.multiprocess_chunksize), str(self.cores)
                    )
                )
                with multiprocessing.Pool(
                    self.cores, maxtasksperchild=self.maxtasksperchild
                ) as pool:
                    constants = functools.partial(
                        self.probability_plot_mp,
                        clf_name=self.clf_name,
                        probability_lst=probability_lst,
                        highest_p=highest_p,
                        video_setting=self.video_setting,
                        frame_setting=self.frame_setting,
                        fps=self.fps,
                        video_dir=self.temp_folder,
                        frame_dir=self.save_frame_folder_dir,
                        style_attr=self.style_attr,
                        video_name=self.video_name,
                    )
                    for cnt, result in enumerate(
                        pool.imap(
                            constants, data_split, chunksize=self.multiprocess_chunksize
                        )
                    ):
                        print(
                            "Image {}/{}, Video {}/{}...".format(
                                str(int(frm_per_core * (result + 1))),
                                str(len(data_df)),
                                str(file_cnt + 1),
                                str(len(self.files_found)),
                            )
                        )

                pool.join()
                pool.terminate()
                if self.video_setting:
                    print("Joining {} multiprocessed video...".format(self.video_name))
                    concatenate_videos_in_folder(
                        in_folder=self.temp_folder, save_path=self.save_video_path
                    )

                video_timer.stop_timer()
                print(
                    "Probability video {} complete (elapsed time: {}s) ...".format(
                        self.video_name, video_timer.elapsed_time_str
                    )
                )

        self.timer.stop_timer()
        stdout_success(
            msg=f"Probability visualizations for {str(len(self.files_found))} videos created in project_folder/frames/output/gantt_plots directory",
            elapsed_time=self.timer.elapsed_time_str,
        )


# test = TresholdPlotCreatorMultiprocess(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals/project_folder/project_config.ini',
#                                         frame_setting=False,
#                                         video_setting=True,
#                                         last_frame=False,
#                                         clf_name='Attack',
#                                         cores=5,
#                                         files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals/project_folder/csv/machine_results/Together_1.csv'],
#                                         style_attr={'width': 640, 'height': 480, 'font size': 10, 'line width': 6, 'color': 'blue', 'circle size': 20, 'y_max': 'auto'})
# #test = TresholdPlotCreatorSingleProcess(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini', frame_setting=False, video_setting=True, clf_name='Attack')
# test.create_plots()


# test = TresholdPlotCreatorMultiprocess(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                         frame_setting=False,
#                                         video_setting=True,
#                                         last_frame=True,
#                                         clf_name='Attack',
#                                         cores=5,
#                                         files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'],
#                                         style_attr={'width': 640, 'height': 480, 'font size': 10, 'line width': 3, 'color': 'blue', 'circle size': 20, 'y_max': 'auto'})
# test.create_plots()
