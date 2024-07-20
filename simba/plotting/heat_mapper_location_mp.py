import functools
import multiprocessing
import os
import platform
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from numba import jit, prange

from simba.mixins.config_reader import ConfigReader
from simba.mixins.geometry_mixin import GeometryMixin
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_file_exist_and_readable, check_float, check_if_keys_exist_in_dict,
    check_int, check_valid_lst)
from simba.utils.enums import Defaults, Formats, TagNames
from simba.utils.errors import NoSpecifiedOutputError
from simba.utils.printing import SimbaTimer, log_event, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    find_core_cnt, get_fn_ext, read_df,
                                    remove_a_folder)

STYLE_PALETTE = 'palette'
STYLE_SHADING = 'shading'
STYLE_BIN_SIZE = 'bin_size'
STYLE_MAX_SCALE = 'max_scale'

STYLE_ATTR = [STYLE_PALETTE, STYLE_SHADING, STYLE_BIN_SIZE, STYLE_MAX_SCALE]

def _heatmap_location(data: np.array,
                      video_setting: bool,
                      frame_setting: bool,
                      video_temp_dir: str,
                      video_name: str,
                      frame_dir: str,
                      fps: int,
                      style_attr: dict,
                      aspect_ratio: float,
                      size: tuple,
                      make_location_heatmap_plot: PlottingMixin.make_location_heatmap_plot):

    group = int(data[0][0][1])
    if video_setting:
        fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        video_save_path = os.path.join(video_temp_dir, f"{group}.mp4")
        video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, size)

    for i in range(data.shape[0]):
        frame_id = int(data[i, 0, 0])
        frm_data = data[i, :, 2:]

        img = PlottingMixin.make_location_heatmap_plot(frm_data=frm_data,
                                         max_scale=style_attr[STYLE_MAX_SCALE],
                                         palette=style_attr[STYLE_PALETTE],
                                         aspect_ratio=aspect_ratio,
                                         shading=style_attr[STYLE_SHADING],
                                         img_size=size,
                                         file_name=None)
        if video_setting:
            video_writer.write(img[:, :, :3])
        if frame_setting:
            file_path = os.path.join(frame_dir, "{}.png".format(frame_id))
            cv2.imwrite(file_path, img)
        print(f"Heatmap location/frame created: {frame_id + 1}, Video: {video_name}, Processing core: {group} ...")

    if video_setting:
        video_writer.release()

    return group


class HeatMapperLocationMultiprocess(ConfigReader, PlottingMixin):
    """
    Create heatmaps representing the location where animals spend time.

    ..note::
       `GitHub visualizations tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-11-visualizations>`__.

    .. image:: _static/img/heatmap_location.gif
       :width: 500
       :align: center

    :param str config_path: path to SimBA project config file in Configparser format
    :param str bodypart: The name of the body-part used to infer the location of the animal.
    :param int bin_size: The rectangular size of each heatmap location in millimeters. For example, `50` will divide the video frames into 5 centimeter rectangular spatial bins.
    :param str palette:  Heatmap pallette. Eg. 'jet', 'magma', 'inferno','plasma', 'viridis', 'gnuplot2'
    :param dict style_attr: Style attributes of heatmap {'palette': 'jet', 'shading': 'gouraud', 'bin_size': 100, 'max_scale': 'auto'}
    :param bool final_img_setting: If True, create a single image representing the last frame of the input video
    :param bool video_setting: If True, then create a video of heatmaps.
    :param bool frame_setting: If True, then create individual heatmap frames.
    :param int core_cnt: The number of CPU cores to use. If -1, then all available cores.

    :example:
    >>> style_attr = {'palette': 'jet', 'shading': 'gouraud', 'bin_size': 100, 'max_scale': 'auto'}
    >>> heatmapper = HeatMapperLocationMultiprocess(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini', style_attr = style_attr, core_cnt=-1, final_img_setting=True, video_setting=True, frame_setting=False, bodypart='Nose_1', files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'])
    >>> heatmapper.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_paths: List[Union[str, os.PathLike]],
                 bodypart: str,
                 style_attr: Dict[str, Any],
                 final_img_setting: Optional[bool] = True,
                 video_setting: Optional[bool] = False,
                 frame_setting: Optional[bool] = False,
                 core_cnt: Optional[int] = -1):

        if (not frame_setting) and (not video_setting) and (not final_img_setting):
            raise NoSpecifiedOutputError(msg="Please choose to select either heatmap videos, frames, and/or final image.", source=self.__class__.__name__)
        check_file_exist_and_readable(file_path=config_path)
        check_valid_lst(data=data_paths, valid_dtypes=(str,), min_len=1)
        check_if_keys_exist_in_dict(data=style_attr, key=STYLE_ATTR, name=f'{self.__class__.__name__} style_attr')
        check_int(name=f'{self.__class__.__name__} core_cnt', value=core_cnt, min_value=-1, max_value=find_core_cnt()[0])
        if core_cnt == -1: core_cnt = find_core_cnt()[0]
        self.core_cnt = core_cnt
        ConfigReader.__init__(self, config_path=config_path, create_logger=False)
        PlottingMixin.__init__(self)
        self.frame_setting, self.video_setting = frame_setting, video_setting
        self.final_img_setting, self.bp = final_img_setting, bodypart
        self.style_attr, self.data_paths = style_attr, data_paths
        if not os.path.exists(self.heatmap_location_dir):
            os.makedirs(self.heatmap_location_dir)
        self.bp_lst = [self.bp + "_x", self.bp + "_y"]
        if platform.system() == "Darwin":
            multiprocessing.set_start_method("spawn", force=True)
        print(f"Processing {len(self.data_paths)} video(s)...")

    @staticmethod
    @jit(nopython=True)
    def __insert_group_idx_column(data: np.array, group: int, last_frm_idx: int):
        results = np.full((data.shape[0], data.shape[1], data.shape[2] + 2), np.nan)
        group_col = np.full((data.shape[1], 1), group)
        for frm_idx in prange(data.shape[0]):
            h_stack = np.hstack((group_col, data[frm_idx]))
            frm_col = np.full((h_stack.shape[0], 1), frm_idx + last_frm_idx)
            results[frm_idx] = np.hstack((frm_col, h_stack))
        return results

    def run(self):
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        for file_cnt, file_path in enumerate(self.data_paths):
            video_timer = SimbaTimer(start=True)
            _, self.video_name, _ = get_fn_ext(file_path)
            self.video_info, self.px_per_mm, self.fps = self.read_video_info(video_name=self.video_name)
            self.width, self.height = int(self.video_info["Resolution_width"].values[0]), int(self.video_info["Resolution_height"].values[0])
            self.save_frame_folder_dir = os.path.join(self.heatmap_location_dir, self.video_name)
            self.video_folder = os.path.join(self.heatmap_location_dir, self.video_name)
            self.temp_folder = os.path.join(self.heatmap_location_dir, self.video_name, "temp")
            if self.frame_setting:
                if os.path.exists(self.save_frame_folder_dir):
                    remove_a_folder(folder_dir=self.save_frame_folder_dir)
                os.makedirs(self.save_frame_folder_dir)
            if self.video_setting:
                if os.path.exists(self.temp_folder):
                    remove_a_folder(folder_dir=self.temp_folder)
                    remove_a_folder(folder_dir=self.video_folder)
                os.makedirs(self.temp_folder)
                self.save_video_path = os.path.join(self.heatmap_location_dir, f"{self.video_name}.mp4")

            self.data_df = read_df(file_path=file_path, file_type=self.file_type, usecols=self.bp_lst)
            squares, aspect_ratio = GeometryMixin().bucket_img_into_grid_square(bucket_grid_size_mm=self.style_attr[STYLE_BIN_SIZE], img_size=(self.width, self.height), px_per_mm=self.px_per_mm)
            cum_sum_squares = GeometryMixin().cumsum_coord_geometries(data=self.data_df.values, fps=self.fps, geometries=squares)
            if self.style_attr[STYLE_MAX_SCALE] == "auto":
                self.style_attr[STYLE_MAX_SCALE]= np.round(np.max(np.max(cum_sum_squares[-1], axis=0)), 3)
                if self.style_attr[STYLE_MAX_SCALE] == 0: self.style_attr[STYLE_MAX_SCALE] = 1
            else:
                check_float(name=f'{self.__class__.__name__} style max scale', value=self.style_attr["max_scale"], min_value=10e-6)
                self.style_attr[STYLE_MAX_SCALE] = self.style_attr[STYLE_MAX_SCALE]

            if self.final_img_setting:
                self.make_location_heatmap_plot(frm_data=cum_sum_squares[-1, :, :],
                                                max_scale=self.style_attr[STYLE_MAX_SCALE],
                                                palette=self.style_attr[STYLE_PALETTE],
                                                aspect_ratio=aspect_ratio,
                                                file_name=os.path.join(self.heatmap_location_dir, f"{self.video_name}_final_frm.png"),
                                                shading=self.style_attr[STYLE_SHADING],
                                                img_size=(self.width, self.height))

            if self.video_setting or self.frame_setting:
                print("Creating heatmap location video frames...")
                frame_arrays = np.array_split(cum_sum_squares, self.core_cnt)
                last_frm_idx = 0
                for frm_group in range(len(frame_arrays)):
                    split_arr = frame_arrays[frm_group]
                    frame_arrays[frm_group] = self.__insert_group_idx_column(data=split_arr, group=frm_group, last_frm_idx=last_frm_idx)
                    last_frm_idx = np.max(frame_arrays[frm_group].reshape((frame_arrays[frm_group].shape[0], -1)))

                with multiprocessing.Pool(self.core_cnt, maxtasksperchild=self.maxtasksperchild) as pool:
                    constants = functools.partial(_heatmap_location,
                                                  video_setting=self.video_setting,
                                                  frame_setting=self.frame_setting,
                                                  style_attr=self.style_attr,
                                                  fps=self.fps,
                                                  video_temp_dir=self.temp_folder,
                                                  frame_dir=self.save_frame_folder_dir,
                                                  aspect_ratio=aspect_ratio,
                                                  size=(self.width, self.height),
                                                  video_name=self.video_name,
                                                  make_location_heatmap_plot=self.make_location_heatmap_plot)
                    for cnt, result in enumerate(pool.imap(constants,frame_arrays,chunksize=self.multiprocess_chunksize)):
                        print(f'Batch {result}/{self.core_cnt} complete... Video: {self.video_name} ({file_cnt+1}/{len(self.data_paths)})')
                    pool.terminate()
                    pool.join()

                if self.video_setting:
                    print(f"Joining {self.video_name} multiprocessed heatmap location video...")
                    concatenate_videos_in_folder(in_folder=self.temp_folder, save_path=self.save_video_path)

                video_timer.stop_timer()
                print(
                    "Heatmap video {} complete (elapsed time: {}s) ...".format(
                        self.video_name, video_timer.elapsed_time_str
                    )
                )

            self.timer.stop_timer()
            stdout_success(msg=f"Heatmap location videos visualizations for {len(self.data_paths)} videos created in {self.heatmap_location_dir} directory", elapsed_time=self.timer.elapsed_time_str, source=self.__class__.__name__)



# test = HeatMapperLocationMultiprocess(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/platea/project_folder/project_config.ini',
#                                       style_attr = {'palette': 'magma', 'shading': 'gouraud', 'bin_size': 80, 'max_scale': 'auto'},
#                                       final_img_setting=True,
#                                       video_setting=True,
#                                       frame_setting=False,
#                                       bodypart='NOSE',
#                                       core_cnt=5,
#                                       data_paths=['/Users/simon/Desktop/envs/simba/troubleshooting/platea/project_folder/csv/outlier_corrected_movement_location/Video_1.csv'])
# test.run()



# test = HeatMapperLocationMultiprocess(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                       style_attr = {'palette': 'jet', 'shading': 'gouraud', 'bin_size': 80, 'max_scale': 'auto'},
#                                       final_img_setting=True,
#                                       video_setting=False,
#                                       frame_setting=False,
#                                       bodypart='Nose_1',
#                                       core_cnt=5,
#                                       files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/outlier_corrected_movement_location/Together_1.csv'])
# test.run()

# test = HeatMapperLocationMultiprocess(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/project_config.ini',
#                                       style_attr = {'palette': 'magma', 'shading': 'gouraud', 'bin_size': 100, 'max_scale': 'auto'},
#                                       final_img_setting=True,
#                                       video_setting=True,
#                                       frame_setting=False,
#                                       bodypart='Nose',
#                                       core_cnt=5,
#                                       data_paths=['/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/csv/machine_results/2022-06-20_NOB_DOT_4.csv'])
# test.run()

# img = np.zeros((img_size[0], img_size[1], 3)).astype(np.uint8)
# for i in polygons:
#     coords = np.array(i.exterior.coords).astype(np.int)
#     print(coords)
#     cv2.polylines(img, [coords], True, (0, 0, 50), 2)
# cv2.imshow('img', img)
# cv2.waitKey()
