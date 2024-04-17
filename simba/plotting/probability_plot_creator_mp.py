__author__ = "Simon Nilsson"

import functools
import multiprocessing
import os
import platform
import shutil
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log, check_int, check_str,
    check_that_column_exist, check_valid_lst)
from simba.utils.enums import Formats
from simba.utils.errors import ColumnNotFoundError, NoSpecifiedOutputError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    find_core_cnt, get_fn_ext, read_df)


def probability_plot_mp(
    data: list,
    probability_lst: list,
    clf_name: str,
    video_setting: bool,
    frame_setting: bool,
    video_dir: str,
    frame_dir: str,
    highest_p: float,
    fps: int,
    style_attr: dict,
    video_name: str,
):
    group, data = data[0], data[1:]
    start_frm, end_frm, current_frm = data[0], data[-1], data[0]

    if video_setting:
        fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        video_save_path = os.path.join(video_dir, f"{group}.mp4")
        video_writer = cv2.VideoWriter(
            video_save_path, fourcc, fps, (style_attr["width"], style_attr["height"])
        )

    while current_frm < end_frm:
        current_lst = [np.array(probability_lst[0 : current_frm + 1])]
        current_frm += 1
        img = PlottingMixin.make_line_plot_plotly(
            data=current_lst,
            colors=[style_attr["color"]],
            width=style_attr["width"],
            height=style_attr["height"],
            line_width=style_attr["line width"],
            font_size=style_attr["font size"],
            y_lbl=f"{clf_name} probability",
            title=clf_name,
            y_max=highest_p,
        )

        if video_setting:
            video_writer.write(img[:, :, :3])
        if frame_setting:
            frame_save_name = os.path.join(frame_dir, f"{current_frm}.png")
            cv2.imwrite(frame_save_name, img)
        current_frm += 1
        print(
            f"Probability frame created: {current_frm + 1}, Video: {video_name}, Processing core: {group}"
        )
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

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        files_found: List[Union[str, os.PathLike]],
        style_attr: Dict[str, Any],
        clf_name: str,
        frame_setting: Optional[bool] = False,
        video_setting: Optional[bool] = False,
        last_frame: Optional[bool] = True,
        cores: Optional[int] = -1,
    ):

        # if platform.system() == "Darwin":
        #     multiprocessing.set_start_method("spawn", force=True)
        if (not video_setting) and (not frame_setting) and (not frame_setting):
            raise NoSpecifiedOutputError(
                msg="SIMBA ERROR: Please choose to create video and/or frames data plots. SimBA found that you ticked neither video and/or frames"
            )
        check_valid_lst(
            data=files_found, source=self.__class__.__name__, valid_dtypes=(str,)
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
        check_str(
            name=f"{self.__class__.__name__} clf_name",
            value=clf_name,
            options=(self.clf_names),
        )
        (
            self.frame_setting,
            self.video_setting,
            self.cores,
            self.style_attr,
            self.last_frame,
        ) = (frame_setting, video_setting, cores, style_attr, last_frame)
        self.clf_name, self.files_found = clf_name, files_found
        self.probability_col = f"Probability_{self.clf_name}"
        self.fontsize = self.style_attr["font size"]
        self.out_width, self.out_height = (
            self.style_attr["width"],
            self.style_attr["height"],
        )
        self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        if not os.path.exists(self.probability_plot_dir):
            os.makedirs(self.probability_plot_dir)
        print(f"Processing {len(self.files_found)} video(s)...")

    def run(self):
        check_all_file_names_are_represented_in_video_log(
            video_info_df=self.video_info_df, data_paths=self.files_found
        )
        for file_cnt, file_path in enumerate(self.files_found):
            video_timer = SimbaTimer(start=True)
            _, self.video_name, _ = get_fn_ext(file_path)
            video_info, self.px_per_mm, self.fps = self.read_video_info(
                video_name=self.video_name
            )
            data_df = read_df(file_path, self.file_type)
            check_that_column_exist(
                df=data_df,
                column_name=[self.clf_name, self.probability_col],
                file_name=file_path,
            )
            self.save_frame_folder_dir = os.path.join(
                self.probability_plot_dir, self.video_name + f"_{self.clf_name}"
            )
            self.video_folder = os.path.join(
                self.probability_plot_dir, self.video_name + f"_{self.clf_name}"
            )
            self.temp_folder = os.path.join(
                self.probability_plot_dir, f"{self.video_name}_{self.clf_name}", "temp"
            )
            if self.frame_setting:
                if os.path.exists(self.save_frame_folder_dir):
                    shutil.rmtree(self.save_frame_folder_dir)
                os.makedirs(self.save_frame_folder_dir)
            if self.video_setting:
                if os.path.exists(self.temp_folder):
                    shutil.rmtree(self.temp_folder)
                    shutil.rmtree(self.video_folder)
                os.makedirs(self.temp_folder)
                self.save_video_path = os.path.join(
                    self.probability_plot_dir, f"{self.video_name}_{self.clf_name}.mp4"
                )

            probability_lst = list(data_df[self.probability_col])
            # probability_lst = list(np.random.random(size=(len(data_df))))

            if self.last_frame:
                _ = self.make_probability_plot(
                    data=pd.Series(probability_lst),
                    style_attr=self.style_attr,
                    clf_name=self.clf_name,
                    fps=self.fps,
                    save_path=os.path.join(
                        self.probability_plot_dir,
                        f"{self.video_name}_{self.clf_name}_final_image.png",
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
                    f"Creating probability images, multiprocessing (determined chunksize: {self.multiprocess_chunksize}, cores: {self.cores})..."
                )
                with multiprocessing.Pool(
                    self.cores, maxtasksperchild=self.maxtasksperchild
                ) as pool:
                    constants = functools.partial(
                        probability_plot_mp,
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
                            f"Image {int(frm_per_core * (result + 1))}/{len(data_df)}, Video {file_cnt + 1}/{len(self.files_found)}..."
                        )

                pool.join()
                pool.terminate()
                if self.video_setting:
                    print(f"Joining {self.video_name} multiprocessed video...")
                    concatenate_videos_in_folder(
                        in_folder=self.temp_folder, save_path=self.save_video_path
                    )

                video_timer.stop_timer()
                print(
                    f"Probability video {self.video_name} complete (elapsed time: {video_timer.elapsed_time_str}s) ..."
                )

        self.timer.stop_timer()
        stdout_success(
            msg=f"Probability visualizations for {str(len(self.files_found))} videos created in project_folder/frames/output/gantt_plots directory",
            elapsed_time=self.timer.elapsed_time_str,
        )


# test = TresholdPlotCreatorMultiprocess(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/project_config.ini',
#                                         frame_setting=False,
#                                         video_setting=True,
#                                         last_frame=True,
#                                         clf_name='Nose to Nose',
#                                         cores=-1,
#                                         files_found=['/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/csv/machine_results/Trial    10.csv'],
#                                         style_attr={'width': 640, 'height': 480, 'font size': 10, 'line width': 6, 'color': 'Red', 'circle size': 20, 'y_max': 'auto'})
# #test = TresholdPlotCreatorSingleProcess(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini', frame_setting=False, video_setting=True, clf_name='Attack')
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
