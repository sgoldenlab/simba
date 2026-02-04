import functools
import multiprocessing
import os
import platform
from pathlib import Path
from typing import Dict, Optional, Union, Tuple

import cv2
import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.geometry_mixin import GeometryMixin
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (check_instance, check_int,
                                check_nvidea_gpu_available, check_str,
                                check_that_column_exist, check_valid_boolean, check_if_valid_rgb_tuple)
from simba.utils.data import (create_color_palette, get_cpu_pool,
                              terminate_cpu_pool)
from simba.utils.enums import OS, Formats, Options
from simba.utils.errors import CountError, InvalidFilepathError
from simba.utils.printing import SimbaTimer, stdout_success, stdout_information
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    find_core_cnt,
                                    find_files_of_filetypes_in_directory,
                                    get_current_time, get_fn_ext,
                                    get_video_meta_data, read_df)
from simba.utils.warnings import FrameRangeWarning
from simba.feature_extractors.perimeter_jit import jitted_centroid


def pose_plotter_mp(data: pd.DataFrame,
                    video_meta_data: dict,
                    video_path: str,
                    bp_dict: dict,
                    colors_dict: dict,
                    circle_size: int,
                    center_of_mass: Optional[dict],
                    center_of_mass_clr: tuple,
                    bbox: bool,
                    video_save_dir: Union[str, os.PathLike],):

    fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
    group_cnt = int(data.iloc[0]["group"])
    data = data.drop(["group"], axis=1)
    start_frm, current_frm, end_frm = data.index[0], data.index[0], data.index[-1]
    save_path = os.path.join(video_save_dir, f"{group_cnt}.mp4")
    writer = cv2.VideoWriter(save_path, fourcc, video_meta_data["fps"], (video_meta_data["width"], video_meta_data["height"]))
    cap = cv2.VideoCapture(video_path)
    cap.set(1, start_frm)

    while current_frm < end_frm:
        ret, img = cap.read()
        if ret:
            for animal_cnt, (animal_name, animal_data) in enumerate(bp_dict.items()):
                animal_bbox = []
                for cnt, (x_name, y_name) in enumerate(zip(animal_data["X_bps"], animal_data["Y_bps"])):
                    check_that_column_exist(df=data, column_name=[x_name, y_name], file_name=video_path)
                    bp_tuple = (int(data.at[current_frm, x_name]), int(data.at[current_frm, y_name]))
                    clr = colors_dict[animal_cnt][cnt]
                    img = cv2.circle(img, bp_tuple, circle_size, clr, -1)
                    animal_bbox.append(list(bp_tuple))
                if bbox and len(animal_bbox) > 4:
                    animal_bbox = GeometryMixin().keypoints_to_axis_aligned_bounding_box(keypoints=np.array(animal_bbox).reshape(-1, len(animal_bbox), 2).astype(np.int32))
                    img = cv2.polylines(img, [animal_bbox], True, colors_dict[animal_cnt][0], thickness=max(1, int(circle_size/1.5)), lineType=-1)
                if center_of_mass is not None:
                    center_point = center_of_mass[animal_name][current_frm]
                    center_point_tuple = (int(center_point[0]), int(center_point[1]))
                    img = cv2.circle(img, center_point_tuple, circle_size+2, center_of_mass_clr, -1)
            writer.write(img)
            current_frm += 1
            stdout_information(msg=f"[{get_current_time()}] Multi-processing video frame {current_frm} on core {group_cnt}...")
        else:
            FrameRangeWarning(msg=f'Frame {current_frm} not found in video {video_path}, terminating video creation...')
            break
    cap.release()
    writer.release()


class PosePlotterMultiProcess():
    """
    Create pose-estimation visualizations from data within a SimBA project folder.

    :param str in_directory: Path to SimBA project directory containing pose-estimation data in parquet or CSV format.
    :param str out_directory: Directory to where to save the pose-estimation videos.
    :param int Size of the circles denoting the location of the pose-estimated body-parts.
    :param Optional[dict] clr_attr: Python dict where animals are keys and color attributes values. E.g., {'Animal_1':  (255, 107, 198)}. If None, random palettes will be used.

    .. image:: _static/img/pose_plotter.png
       :width: 600
       :align: center

    :example:
    >>> test = PosePlotterMultiProcess(in_dir='project_folder/csv/input_csv', out_dir='/project_folder/test_viz', circle_size=10, core_cnt=1, color_settings={'Animal_1':  'Green', 'Animal_2':  'Red'})
    >>> test.run()
    """

    def __init__(self,
                 data_path: Union[str, os.PathLike],
                 out_dir: Optional[Union[str, os.PathLike]] = None,
                 palettes: Optional[Dict[str, str]] = None,
                 circle_size: Optional[int] = None,
                 core_cnt: Optional[int] = -1,
                 gpu: Optional[bool] = False,
                 bbox: Optional[bool] = False,
                 center_of_mass: Optional[Tuple[int, int, int]] = None,
                 sample_time: Optional[int] = None,
                 verbose: bool = True) -> None:

        if os.path.isdir(data_path):
            config_path = os.path.join(Path(data_path).parents[1], 'project_config.ini')
        elif os.path.isfile(data_path):
            config_path = os.path.join(Path(data_path).parents[2], 'project_config.ini')
        else:
            raise InvalidFilepathError(msg=f'{data_path} not not a valid file or directory path.', source=self.__class__.__name__)
        if not os.path.isfile(config_path):
            raise InvalidFilepathError(msg=f'When visualizing pose-estimation, select an input sub-directory of the project_folder/csv folder OR file in the project_folder/csv/ANY_FOLDER directory. {data_path} does not meet these requirements and therefore SimBA cant locate the project_config.ini (expected at {config_path}', source=self.__class__.__name__)
        self.config = ConfigReader(config_path=config_path, read_video_info=False, create_logger=False)
        if os.path.isdir(data_path):
            files_found = find_files_of_filetypes_in_directory(directory=data_path, extensions=[f'.{self.config.file_type}'], raise_error=True)
        else:
            files_found = [data_path]
        self.animal_bp_dict = self.config.body_parts_lst
        if circle_size is not None: check_int(name='circle_size', value=circle_size, min_value=1)
        check_int(name='core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0])
        if core_cnt == -1: core_cnt = find_core_cnt()[0]
        self.color_dict = {}
        if palettes is not None:
            check_instance(source=self.__class__.__name__, instance=palettes, accepted_types=(dict,))
            if len(list(palettes.keys())) != self.config.animal_cnt:
                raise CountError(msg=f'The number of color palettes ({(len(list(palettes.keys())))}) spedificed is not the same as the number of animals ({(self.config.animal_cnt)}) in the SimBA project at {self.config.project_path}')
            for cnt, (k, v) in enumerate(palettes.items()):
                check_str(name='palette', value=v, options=Options.PALETTE_OPTIONS_CATEGORICAL.value)
                self.color_dict[cnt] = create_color_palette(pallete_name=v, increments=len(self.config.body_parts_lst))
        else:
            for cnt, (k, v) in enumerate(self.config.animal_bp_dict.items()):
                self.color_dict[cnt] = self.config.animal_bp_dict[k]["colors"]
        check_valid_boolean(value=bbox, source=f'{self.__class__.__name__} bbox')
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose', raise_error=True)
        if sample_time is not None:
            check_int(name='sample_time', value=sample_time, min_value=1)
        if out_dir is None:
            out_dir = os.path.join(os.path.dirname(files_found[0]), f'pose_videos_{self.config.datetime}')
        self.circle_size, self.core_cnt, self.out_dir, self.sample_time, self.bbox, self.verbose = (circle_size, core_cnt, out_dir, sample_time, bbox, verbose)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        check_valid_boolean(value=gpu, source=f'{self.__class__.__name__} gpu', raise_error=True)
        if center_of_mass is not None:
            check_if_valid_rgb_tuple(data=center_of_mass, raise_error=True, source=f'{self.__class__.__name__} center_of_mass')
        self.data, self.center_of_mass = {}, center_of_mass
        self.gpu = True if gpu and check_nvidea_gpu_available() else False
        for file in files_found:
            self.data[file] = self.config.find_video_of_file(video_dir=self.config.video_dir, filename=get_fn_ext(file)[1])
        if platform.system() == OS.MAC.value:
            multiprocessing.set_start_method("spawn", force=True)

    def _get_center_of_mass(self):
        if self.verbose: stdout_information(msg='Computing animal centroids...')
        center_of_mass_data = {}
        for animal_cnt, (animal_name, animal_data) in enumerate(self.config.animal_bp_dict.items()):
            animal_data_cols = [x for pair in zip(animal_data["X_bps"], animal_data["Y_bps"]) for x in pair]
            animal_df = self.pose_df[animal_data_cols]
            center_of_mass_data[animal_name] = jitted_centroid(points=np.reshape(animal_df.values, (len(animal_df / 2), -1, 2)).astype(np.float32))
        return center_of_mass_data


    def run(self):
        self.pool = get_cpu_pool(core_cnt=self.core_cnt, source=self.__class__.__name__)
        for file_cnt, (pose_path, video_path) in enumerate(self.data.items()):
            video_timer = SimbaTimer(start=True)
            video_name = get_fn_ext(pose_path)[1]
            self.temp_folder = os.path.join(self.out_dir, video_name, "temp")
            if os.path.exists(self.temp_folder): self.config.remove_a_folder(self.temp_folder)
            os.makedirs(self.temp_folder, exist_ok=True)
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
            self.pose_df = (pose_df.apply(pd.to_numeric, errors="coerce").fillna(0).reset_index(drop=True))
            self.centroid_data = self._get_center_of_mass() if self.center_of_mass is not None else None
            pose_lst, obs_per_split = PlottingMixin().split_and_group_df(df=pose_df, splits=self.core_cnt)
            if self.verbose: stdout_information(msg=f"Creating pose videos, multiprocessing (chunksize: {self.config.multiprocess_chunksize}, cores: {self.core_cnt})...")
            constants = functools.partial(pose_plotter_mp,
                                          video_meta_data=video_meta_data,
                                          video_path=video_path,
                                          bp_dict=self.config.animal_bp_dict,
                                          colors_dict=self.color_dict,
                                          circle_size=video_circle_size,
                                          bbox=self.bbox,
                                          center_of_mass=self.centroid_data,
                                          center_of_mass_clr=self.center_of_mass,
                                          video_save_dir=self.temp_folder)
            for cnt, result in enumerate(self.pool.imap(constants, pose_lst, chunksize=self.config.multiprocess_chunksize)):
                if self.verbose: stdout_information(msg=f"Image {min(len(pose_df), obs_per_split*(cnt+1))}/{len(pose_df)}, Video {file_cnt+1}/{len(list(self.data.keys()))}...")
            if self.verbose: stdout_information(msg=f"Joining {video_name} multi-processed video...")
            concatenate_videos_in_folder(in_folder=self.temp_folder, save_path=save_video_path, remove_splits=True, gpu=self.gpu)
            video_timer.stop_timer()
            stdout_success(msg=f"Pose video {video_name} complete and saved at {save_video_path}", elapsed_time=video_timer.elapsed_time_str, source=self.__class__.__name__)
        terminate_cpu_pool(pool=self.pool, force=False, source=self.__class__.__name__)
        self.config.timer.stop_timer()
        stdout_success(f"Pose visualizations for {len(list(self.data.keys()))} video(s) created in {self.out_dir} directory", elapsed_time=self.config.timer.elapsed_time_str, source=self.__class__.__name__)

# if __name__ == "__main__":
#     test = PosePlotterMultiProcess(data_path=r"E:\troubleshooting\mitra_emergence\project_folder\csv\outlier_corrected_movement_location\Box1_180mISOcontrol_Females.csv",
#                                    out_dir=None,
#                                    circle_size=8,
#                                    core_cnt=12,
#                                    palettes=None,
#                                    bbox=True,
#                                    center_of_mass=True)
#     test.run()




# test = PosePlotterMultiProcess(data_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/csv/outlier_corrected_movement_location/Together_1.csv',
#                    out_dir=None,
#                    circle_size=None,
#                    core_cnt=1,
#                    palettes=None)
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
