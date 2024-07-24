__author__ = "Simon Nilsson"

import os
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (check_instance, check_int, check_str,
                                check_that_column_exist)
from simba.utils.data import create_color_palette
from simba.utils.enums import Formats, Options
from simba.utils.errors import CountError, InvalidFilepathError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, get_video_meta_data, read_df)
from simba.utils.warnings import FrameRangeWarning


class PosePlotter(object):

    """
    Create pose-estimation visualizations from data within a SimBA project folder.

    :param str data_path: Path to SimBA project directory containing pose-estimation data in parquet or CSV format.
    :param str out_directory: Directory to where to save the pose-estimation videos.
    :param int Size of the circles denoting the location of the pose-estimated body-parts.
    :param Optional[dict] clr_attr: Python dict where animals are keys and color attributes values. E.g., {'Animal_1':  (255, 107, 198)}. If None, random palettes will be used.

    """
    def __init__(self,
                 data_path: Union[str, os.PathLike],
                 out_dir: Optional[Union[str, os.PathLike]] = None,
                 palettes: Optional[Dict[str, str]] = None,
                 circle_size: Optional[int] = None,
                 sample_time: Optional[int] = None) -> None:

        if os.path.isdir(data_path):
            config_path = os.path.join(Path(data_path).parents[1], 'project_config.ini')
        elif os.path.isfile(data_path):
            config_path = os.path.join(Path(data_path).parents[2], 'project_config.ini')
        else:
            raise InvalidFilepathError(msg=f'{data_path} not not a valid file or directory path.', source=self.__class__.__name__)
        if not os.path.isfile(config_path):
            raise InvalidFilepathError(msg=f'When visualizing pose-estimation, select an input sub-directory of the project_folder/csv folder OR file in the project_folder/csv/ANY_FOLDER directory. {data_path} does not meet these requirements and therefore SimBA cant locate the project_config.ini (expected at {config_path}', source=self.__class__.__name__)
        self.config = ConfigReader(config_path=config_path, read_video_info=False)
        if os.path.isdir(data_path):
            self.files_found = find_files_of_filetypes_in_directory(directory=data_path, extensions=[f'.{self.config.file_type}'], raise_error=True)
        else:
            self.files_found = [data_path]
        self.animal_bp_dict = self.config.body_parts_lst
        if circle_size is not None:
            check_int(name='circle_size', value=circle_size, min_value=1)
        self.color_dict = {}
        if palettes is not None:
            check_instance(source=self.__class__.__name__, instance=palettes, accepted_types=(dict,))
            if len(list(palettes.keys())) != self.config.animal_cnt:
                raise CountError(msg=f'The number of color palettes ({(len(list(palettes.keys())))}) is not the same as the number of animals ({(self.config.animal_cnt)}) in the SimBA project at {self.config.project_path}')
            for c, (k, v) in enumerate(palettes.items()):
                check_str(name='palette', value=v, options=Options.PALETTE_OPTIONS_CATEGORICAL.value)
                self.color_dict[c] = create_color_palette(pallete_name=v, increments=len(self.config.body_parts_lst))
        else:
            for cnt, (k, v) in enumerate(self.config.animal_bp_dict.items()):
                self.color_dict[cnt] = self.config.animal_bp_dict[k]["colors"]
        if sample_time is not None:
            check_int(name='sample_time', value=sample_time, min_value=1)
        if out_dir is None:
            out_dir = os.path.join(os.path.dirname(self.files_found[0]), f'pose_videos_{self.config.datetime}')
        self.circle_size,self.out_dir, self.sample_time = (circle_size, out_dir, sample_time)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        self.data = {}
        for file in self.files_found:
            self.data[file] = self.config.find_video_of_file(video_dir=self.config.video_dir, filename=get_fn_ext(file)[1])



    def run(self):
        for file_cnt, (pose_path, video_path) in enumerate(self.data.items()):
            video_timer = SimbaTimer(start=True)
            video_name = get_fn_ext(pose_path)[1]
            save_video_path = os.path.join(self.out_dir, f"{video_name}.mp4")
            pose_df = read_df(file_path=pose_path, file_type=self.config.file_type, check_multiindex=True)
            video_meta_data = get_video_meta_data(video_path=video_path)
            if self.circle_size is None:
                video_circle_size = PlottingMixin().get_optimal_circle_size(frame_size=(int(video_meta_data['width']), int(video_meta_data['height'])), circle_frame_ratio=70)
            else:
                video_circle_size = self.circle_size
            if (self.sample_time is None) and (video_meta_data["frame_count"] != len(pose_df)):
                FrameRangeWarning(msg=f'The video {video_name} has pose-estimation data for {len(pose_df)} frames, but the video has {video_meta_data["frame_count"]} frames. Ensure the data and video has an equal number of frames.', source=self.__class__.__name__)
            elif isinstance(self.sample_time, int):
                sample_frm_cnt = int(video_meta_data["fps"] * self.sample_time)
                if sample_frm_cnt > len(pose_df): sample_frm_cnt = len(pose_df)
                pose_df = pose_df.iloc[0:sample_frm_cnt]
            if 'input_csv' in os.path.dirname(pose_path):
                pose_df = self.config.insert_column_headers_for_outlier_correction(data_df=pose_df, new_headers=self.config.bp_headers, filepath=pose_path)
            pose_df = (pose_df.apply(pd.to_numeric, errors="coerce").fillna(0).reset_index(drop=True))
            fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
            writer = cv2.VideoWriter(save_video_path, fourcc, video_meta_data["fps"], (video_meta_data["width"], video_meta_data["height"]))
            cap = cv2.VideoCapture(video_path)
            frm_cnt = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    for animal_cnt, (animal_name, animal_data) in enumerate(self.config.animal_bp_dict.items()):
                        for bp_cnt, (bp_x, bp_y) in enumerate(zip(animal_data["X_bps"], animal_data["Y_bps"])):
                            check_that_column_exist(df=pose_df, column_name=[bp_x, bp_y], file_name=pose_path)
                            bp_tuple = (int(pose_df.at[frm_cnt, bp_x]), int(pose_df.at[frm_cnt, bp_y]))
                            cv2.circle(frame, bp_tuple, video_circle_size, self.color_dict[animal_cnt][bp_cnt], -1)
                    frm_cnt += 1
                    writer.write(frame)
                    print(f"Video: {file_cnt + 1} / {len(self.files_found)} Frame: {frm_cnt} / {len(pose_df)}")
                else:
                    print(f'Frame {frm_cnt} not found in video, terminating video creation...')
                    break
            video_timer.stop_timer()
            print(f"{save_video_path} complete... (elapsed time: {video_timer.elapsed_time_str}s)", )
            cap.release()
            writer.release()
        self.config.timer.stop_timer()
        stdout_success(msg=f"All pose videos complete. Results located in {self.out_dir} directory", elapsed_time=self.config.timer.elapsed_time_str)


# x = PosePlotter(data_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/csv/input_csv/Together_1.csv')
# x.run()