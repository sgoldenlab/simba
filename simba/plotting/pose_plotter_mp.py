import functools
import glob
import multiprocessing
import os
import platform
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import check_if_filepath_list_is_empty
from simba.utils.enums import OS, Formats, TagNames
from simba.utils.errors import CountError, InvalidFilepathError
from simba.utils.lookups import get_color_dict
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder, get_fn_ext,
                                    get_video_meta_data, read_df)
from simba.utils.warnings import FrameRangeWarning


def pose_plotter_mp(
    data: pd.DataFrame,
    video_meta_data: dict,
    video_path: str,
    bp_dict: dict,
    circle_size: int,
    video_save_dir: Union[str, os.PathLike],
):
    fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
    group_cnt = int(data.iloc[0]["group"])
    data = data.drop(["group"], axis=1)
    start_frm, current_frm, end_frm = data.index[0], data.index[0], data.index[-1]
    save_path = os.path.join(video_save_dir, f"{group_cnt}.mp4")
    writer = cv2.VideoWriter(
        save_path,
        fourcc,
        video_meta_data["fps"],
        (video_meta_data["width"], video_meta_data["height"]),
    )
    cap = cv2.VideoCapture(video_path)
    cap.set(1, start_frm)

    while current_frm < end_frm:
        ret, img = cap.read()
        if ret:
            for animal_name, animal_data in bp_dict.items():
                for x_name, y_name in zip(animal_data["X_bps"], animal_data["Y_bps"]):
                    bp_tuple = (
                        int(data.at[current_frm, x_name]),
                        int(data.at[current_frm, y_name]),
                    )
                    cv2.circle(img, bp_tuple, circle_size, animal_data["colors"], -1)
            writer.write(img)
            current_frm += 1
            print(f"Multi-processing video frame {current_frm} on core {group_cnt}...")

    cap.release()
    writer.release()


class PosePlotter:
    """
    Create pose-estimation visualizations from data within a SimBA project folder.

    :param str in_directory: Path to SimBA project directory containing pose-estimation data in parquet or CSV format.
    :param str out_directory: Directory to where to save the pose-estimation videos.
    :param int Size of the circles denoting the location of the pose-estimated body-parts.
    :param Optional[dict] clr_attr: Python dict where animals are keys and color attributes values. E.g., {'Animal_1':  (255, 107, 198)}. If None,
                                    random palettes will be used.

    .. image:: _static/img/pose_plotter.png
       :width: 600
       :align: center

    :example:
    >>> test = PosePlotter(in_dir='project_folder/csv/input_csv', out_dir='/project_folder/test_viz', circle_size=10, core_cnt=1, color_settings={'Animal_1':  'Green', 'Animal_2':  'Red'})
    >>> test.run()
    """

    def __init__(
        self,
        in_dir: Union[str, os.PathLike],
        out_dir: Union[str, os.PathLike],
        circle_size: int,
        core_cnt: int,
        color_settings: Optional[Dict[str, str]] = None,
        sample_time: Optional[int] = None,
    ) -> None:
        config_path, self.data, self.color_settings = (
            os.path.join(Path(in_dir).parents[1], "project_config.ini"),
            {},
            color_settings,
        )
        self.circle_size, self.core_cnt, self.out_dir, self.sample_time = (
            circle_size,
            core_cnt,
            out_dir,
            sample_time,
        )
        self.clrs = get_color_dict()
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        if not os.path.isfile(config_path):
            raise InvalidFilepathError(
                msg=f'When visualizing pose-estimation, select an input sub-directory of the project_folder/csv folder. {in_dir} is not a sub-directory to the "project_folder/csv" directory and therefore SimBA cant locate the project_config.ini (expected at {config_path}',
                source=self.__class__.__name__,
            )
        if platform.system() == OS.MAC.value:
            multiprocessing.set_start_method("spawn", force=True)
        self.config = ConfigReader(config_path=config_path, read_video_info=False)
        files_found = glob.glob(in_dir + f"/*.{self.config.file_type}")
        check_if_filepath_list_is_empty(
            filepaths=files_found,
            error_msg=f"0 files found in {in_dir} in {self.config.file_type} file format",
        )

        for file in files_found:
            print(self.config.video_dir)
            self.data[file] = self.config.find_video_of_file(
                video_dir=self.config.video_dir, filename=get_fn_ext(file)[1]
            )
        self.original_bp_dict = deepcopy(self.config.animal_bp_dict)

    def run(self):
        for file_cnt, (pose_path, video_path) in enumerate(self.data.items()):
            video_timer = SimbaTimer(start=True)
            video_name = get_fn_ext(pose_path)[1]
            self.temp_folder = os.path.join(self.out_dir, video_name, "temp")
            if os.path.exists(self.temp_folder):
                self.config.remove_a_folder(self.temp_folder)
            os.makedirs(self.temp_folder)
            save_video_path = os.path.join(self.out_dir, f"{video_name}.mp4")
            pose_df = read_df(
                file_path=pose_path,
                file_type=self.config.file_type,
                check_multiindex=True,
            )
            video_meta_data = get_video_meta_data(video_path=video_path)
            if (self.sample_time is None) and (
                video_meta_data["frame_count"] != len(pose_df)
            ):
                FrameRangeWarning(
                    msg=f'The video {video_name} has pose-estimation data for {len(pose_df)} frames, but the video has {video_meta_data["frame_count"]} frames. Ensure the data and video has an equal number of frames.',
                    source=self.__class__.__name__,
                )
            if type(self.sample_time) is int:
                sample_frm_cnt = int(video_meta_data["fps"] * self.sample_time)
                if sample_frm_cnt > len(pose_df):
                    sample_frm_cnt = len(pose_df)
                pose_df = pose_df.iloc[0:sample_frm_cnt]
            pose_df = self.config.insert_column_headers_for_outlier_correction(
                data_df=pose_df, new_headers=self.config.bp_headers, filepath=pose_path
            )
            pose_df = (
                pose_df.apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .reset_index(drop=True)
            )
            if self.color_settings:
                for cnt, animal in enumerate(self.config.animal_bp_dict.keys()):
                    self.config.animal_bp_dict[animal]["colors"] = self.clrs[
                        self.color_settings[f"Animal_{cnt+1}"]
                    ]
            else:
                for cnt, animal in enumerate(self.config.animal_bp_dict.keys()):
                    self.config.animal_bp_dict[animal]["colors"] = (
                        self.original_bp_dict[animal]["colors"][0]
                    )

            pose_lst, obs_per_split = PlottingMixin().split_and_group_df(
                df=pose_df, splits=self.core_cnt
            )
            print(
                f"Creating pose videos, multiprocessing (chunksize: {self.config.multiprocess_chunksize}, cores: {self.core_cnt})..."
            )
            with multiprocessing.Pool(
                self.core_cnt, maxtasksperchild=self.config.maxtasksperchild
            ) as pool:
                constants = functools.partial(
                    pose_plotter_mp,
                    video_meta_data=video_meta_data,
                    video_path=video_path,
                    bp_dict=self.config.animal_bp_dict,
                    circle_size=self.circle_size,
                    video_save_dir=self.temp_folder,
                )
                for cnt, result in enumerate(
                    pool.imap(
                        constants,
                        pose_lst,
                        chunksize=self.config.multiprocess_chunksize,
                    )
                ):
                    print(
                        f"Image {obs_per_split*(cnt+1)}/{len(pose_df)}, Video {file_cnt+1}/{len(list(self.data.keys()))}..."
                    )
                pool.terminate()
                pool.join()

            print(f"Joining {video_name} multi-processed video...")
            concatenate_videos_in_folder(
                in_folder=self.temp_folder,
                save_path=save_video_path,
                remove_splits=True,
            )
            video_timer.stop_timer()
            stdout_success(
                msg=f"Pose video {video_name} complete",
                elapsed_time=video_timer.elapsed_time_str,
                source=self.__class__.__name__,
            )
        self.config.timer.stop_timer()
        stdout_success(
            f"Pose visualizations for {len(list(self.data.keys()))} video(s) created in {self.out_dir} directory",
            elapsed_time=self.config.timer.elapsed_time_str,
            source=self.__class__.__name__,
        )


# test = PosePlotter(in_dir='/Users/simon/Desktop/envs/troubleshooting/Zelikowsky/project_folder/csv/input_csv',
#                    out_dir='/Users/simon/Desktop/envs/troubleshooting/Zelikowsky/project_folder/test_viz',
#                    circle_size=10,
#                    core_cnt=1,
#                    color_settings={'Animal_1':  'Green', 'Animal_2':  'Red'})
# test.run()


# test = PosePlotter(in_dir='/Users/simon/Desktop/envs/troubleshooting/dam_nest-c-only_ryan/project_folder/csv/outlier_corrected_movement_location',
#                    out_dir='/Users/simon/Desktop/video_tests_',
#                    sample_time=2,
#                    circle_size=10,
#                    core_cnt=1,
#                    color_settings=None) #
# test.run()


# test = PosePlotter(in_dir='/Users/simon/Desktop/envs/troubleshooting/piotr/project_folder/csv/outlier_corrected_movement_location',
#                    out_dir='/Users/simon/Desktop/envs/troubleshooting/piotr/project_folder/frames/output/test',
#                    circle_size=10,
#                    core_cnt=6,
#                    color_settings={'Animal_1':  'Green', 'Animal_2':  'Red'})
# test.run()
