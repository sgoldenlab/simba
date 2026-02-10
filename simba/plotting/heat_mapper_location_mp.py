import functools
import multiprocessing
import os
import platform
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from numba import jit, prange

from simba.mixins.config_reader import ConfigReader
from simba.mixins.geometry_mixin import GeometryMixin
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_file_exist_and_readable, check_filepaths_in_iterable_exist,
    check_float, check_if_keys_exist_in_dict,
    check_if_string_value_is_valid_video_timestamp, check_int, check_str,
    check_that_hhmmss_start_is_before_end, check_valid_boolean,
    check_valid_lst)
from simba.utils.data import (find_frame_numbers_from_time_stamp, get_cpu_pool,
                              terminate_cpu_pool)
from simba.utils.enums import OS, Formats
from simba.utils.errors import (FrameRangeError, NoDataError,
                                NoSpecifiedOutputError)
from simba.utils.lookups import get_named_colors
from simba.utils.printing import SimbaTimer, stdout_information, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    find_core_cnt, find_video_of_file,
                                    get_fn_ext, get_video_meta_data, read_df,
                                    read_frm_of_video, remove_a_folder)

STYLE_PALETTE = 'palette'
STYLE_SHADING = 'shading'
STYLE_BIN_SIZE = 'bin_size'
STYLE_MAX_SCALE = 'max_scale'
AUTO = 'auto'
START_TIME = 'start_time'
END_TIME = 'end_time'

STYLE_ATTR = [STYLE_PALETTE, STYLE_SHADING, STYLE_BIN_SIZE, STYLE_MAX_SCALE]

def _heatmap_location(data: np.array,
                      video_setting: bool,
                      frame_setting: bool,
                      video_temp_dir: str,
                      video_name: str,
                      frame_dir: str,
                      fps: int,
                      show_legend: bool,
                      heatmap_opacity: float,
                      kp_data: Union[pd.DataFrame, None],
                      video_path: str,
                      min_seconds: Union[int, None],
                      bg_img: Union[int, None],
                      style_attr: dict,
                      aspect_ratio: float,
                      size: Tuple[int, int]):

    group = int(data[0][0][1])
    video_writer = None
    static_bg, kp_size = None, None
    if video_path is not None:
        if bg_img > -1:
            static_bg = read_frm_of_video(video_path=video_path, frame_index=bg_img, greyscale=False)
        kp_size = PlottingMixin().get_optimal_circle_size(frame_size=size, circle_frame_ratio=50)
    for frm_cnt, i in enumerate(range(data.shape[0])):
        frame_id, frm_data = int(data[i, 0, 0]), data[i, :, 2:]
        if bg_img is not None:
            if static_bg is not None:
                frm_bg_img = deepcopy(static_bg)
            else: frm_bg_img = read_frm_of_video(video_path=video_path, frame_index=frame_id, greyscale=False)
        else:
            frm_bg_img = None
        img = PlottingMixin.make_location_heatmap_plot(frm_data=frm_data,
                                                       max_scale=style_attr[STYLE_MAX_SCALE],
                                                       palette=style_attr[STYLE_PALETTE],
                                                       aspect_ratio=aspect_ratio,
                                                       bg_img=frm_bg_img,
                                                       min_seconds=min_seconds,
                                                       heatmap_opacity=heatmap_opacity,
                                                       shading=style_attr[STYLE_SHADING],
                                                       img_size=size,
                                                       color_legend=show_legend,
                                                       leg_width=None,
                                                       file_name=None)
        if kp_data is not None and kp_size is not None:
            frm_point = kp_data.loc[frame_id].values.astype(np.int32)
            img = cv2.circle(img, (frm_point[0], frm_point[1]), kp_size, (255, 0, 0), -1)

        if video_setting:
            if video_writer is None:
                h, w = img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
                video_save_path = os.path.join(video_temp_dir, f"{group}.mp4")
                out_size = (w, h)
                video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, out_size)
            video_writer.write(img[:, :, :3].astype(np.uint8))
        if frame_setting:
            file_path = os.path.join(frame_dir, f"{frame_id}.png")
            cv2.imwrite(file_path, img)
        stdout_information(msg=f"Heatmap location/frame created: {frame_id + 1} (video: {video_name}, processing core: {group}, core frame: {frm_cnt}/{len(data)})...")
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

    ..  youtube:: O41x96kXUHE
       :width: 640
       :height: 480
       :align: center

    :param str config_path: path to SimBA project config file in Configparser format
    :param bool final_img_setting: If True, then create a single image representing the last frame of the input video
    :param bool video_setting: If True, then create a video of heatmaps.
    :param bool frame_setting: If True, then create individual heatmap frames.
    :param str clf_name: The name of the classified behavior.
    :param str bodypart: The name of the body-part used to infer the location of the classified behavior
    :param Dict style_attr: Dict containing settings for colormap, bin-size, max scale, and smooothing operations. For example: {'palette': 'jet', 'shading': 'gouraud', 'bin_size': 50, 'max_scale': 'auto'}.
    :param int core_cnt: The number of CPU cores to use. If -1, then all available cores.

    :example:
    >>> style_attr = {'palette': 'jet', 'shading': 'gouraud', 'bin_size': 100, 'max_scale': 'auto'}
    >>> heatmapper = HeatMapperLocationMultiprocess(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini', style_attr = style_attr, core_cnt=-1, final_img_setting=True, video_setting=True, frame_setting=False, bodypart='Nose_1', files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'])
    >>> heatmapper.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_paths: Union[List[Union[str, os.PathLike]], str, os.PathLike],
                 bodypart: str,
                 style_attr: Dict[str, Any],
                 bg_img: Optional[int] = None,
                 time_slice: Optional[Dict[str, str]] = None,
                 show_keypoint: bool = False,
                 show_legend: bool = True,
                 heatmap_opacity: Optional[float] = None,
                 min_seconds: Optional[int] = None,
                 line_clr: Optional[str] = None,
                 final_img_setting: Optional[bool] = True,
                 video_setting: Optional[bool] = False,
                 frame_setting: Optional[bool] = False,
                 core_cnt: Optional[int] = -1,
                 verbose: bool = True):

        if (not frame_setting) and (not video_setting) and (not final_img_setting):
            raise NoSpecifiedOutputError(msg="Please choose to select either heatmap videos, frames, and/or final image.", source=self.__class__.__name__)
        check_file_exist_and_readable(file_path=config_path)
        if isinstance(data_paths, list):
            check_valid_lst(data=data_paths, valid_dtypes=(str,), min_len=1)
        elif isinstance(data_paths, str):
            check_file_exist_and_readable(file_path=data_paths)
            data_paths = [data_paths]
        else:
            data_paths = self.outlier_corrected_paths
        check_filepaths_in_iterable_exist(file_paths=data_paths, name=f'{self.__class__.__name__} data_paths')
        check_if_keys_exist_in_dict(data=style_attr, key=STYLE_ATTR, name=f'{self.__class__.__name__} style_attr')
        check_int(name=f'{self.__class__.__name__} core_cnt', value=core_cnt, min_value=-1, max_value=find_core_cnt()[0], unaccepted_vals=[0])
        if line_clr is not None: check_str(name=f'{self.__class__.__name__} line_clr', value=line_clr, options=get_named_colors())
        if heatmap_opacity is not None: check_float(name=f'{self.__class__.__name__} heatmap_opacity', value=heatmap_opacity, min_value=0, max_value=1.0)
        if min_seconds is not None: check_int(name=f'{self.__class__.__name__} min_seconds', value=min_seconds, min_value=1)
        if bg_img is not None: check_int(name=f'{self.__class__.__name__} bg_img', value=bg_img, min_value=-1)
        check_valid_boolean(value=show_keypoint, source=f'{self.__class__.__name__}, show_keypoint', raise_error=True)
        check_valid_boolean(value=show_legend, source=f'{self.__class__.__name__}, show_legend', raise_error=True)

        core_cnt = find_core_cnt()[0] if core_cnt == -1 else core_cnt
        self.core_cnt = core_cnt
        if time_slice is not None:
            check_if_keys_exist_in_dict(data=time_slice, key=[START_TIME, END_TIME], name=f'{self.__class__.__name__} slicing')
            check_if_string_value_is_valid_video_timestamp(value=time_slice[START_TIME], name="Video slicing START TIME")
            check_if_string_value_is_valid_video_timestamp(value=time_slice[END_TIME], name="Video slicing END TIME")
            check_that_hhmmss_start_is_before_end(start_time=time_slice[START_TIME], end_time=time_slice[END_TIME], name="SLICE TIME STAMPS")
        ConfigReader.__init__(self, config_path=config_path, create_logger=False)
        PlottingMixin.__init__(self)
        self.frame_setting, self.video_setting = frame_setting, video_setting
        self.final_img_setting, self.bp = final_img_setting, bodypart
        self.style_attr, self.data_paths = style_attr, data_paths
        if not os.path.exists(self.heatmap_location_dir):
            os.makedirs(self.heatmap_location_dir)
        self.bp_lst = [self.bp + "_x", self.bp + "_y"]
        if platform.system() == OS.MAC.value: multiprocessing.set_start_method(OS.SPAWN.value, force=True)
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose', raise_error=True)
        self.verbose, self.bg_img, self.time_slice, self.show_keypoint, self.line_clr = verbose, bg_img, time_slice, show_keypoint, line_clr
        self.show_legend, self.min_seconds, self.heatmap_opacity = show_legend, min_seconds, heatmap_opacity
        if self.verbose: stdout_information(msg=f"Processing {len(self.data_paths)} video(s)...")

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


    def _get_styles(self):
        video_styles = deepcopy(self.style_attr)
        if self.style_attr[STYLE_MAX_SCALE] == AUTO:
            video_styles[STYLE_MAX_SCALE] = np.round(np.max(np.max(self.cum_sum_squares[-1], axis=0)), 3)
            if video_styles[STYLE_MAX_SCALE] == 0: self.style_attr[STYLE_MAX_SCALE] = 1
        else:
            check_float(name=f'{self.__class__.__name__} style max scale', value=self.style_attr[STYLE_MAX_SCALE], allow_zero=False, allow_negative=False)
        return video_styles

    def run(self):
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        self.pool = get_cpu_pool(core_cnt=self.core_cnt, source=self.__class__.__name__)
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
            if self.time_slice is not None:
                frm_numbers = find_frame_numbers_from_time_stamp(start_time=self.time_slice[START_TIME], end_time=self.time_slice[END_TIME], fps=self.fps)
                if len(set(frm_numbers) - set(self.data_df.index)) > 0:
                    raise FrameRangeError(msg=f'The chosen time-period ({self.time_slice[START_TIME]} - {self.time_slice[END_TIME]}) does not exist in {self.video_name}.', source=self.__class__.__name__)
                self.data_df = self.data_df.loc[frm_numbers]
            squares, aspect_ratio = GeometryMixin().bucket_img_into_grid_square(bucket_grid_size_mm=self.style_attr[STYLE_BIN_SIZE], img_size=(self.width, self.height), px_per_mm=self.px_per_mm, add_correction=True)
            self.cum_sum_squares = GeometryMixin().cumsum_coord_geometries(data=self.data_df.values, fps=self.fps, geometries=squares, core_cnt=self.core_cnt, pool=self.pool)
            video_styles = self._get_styles()
            video_path = find_video_of_file(video_dir=self.video_dir, filename=self.video_name, raise_error=False, warning=False)
            if video_path is None and self.bg_img is not None:
                raise NoDataError(msg=f'Cannot find video file for {self.video_name}. Make sure video is present in project video dir if using video background image for heatmap', source=self.__class__.__name__)
            if self.bg_img is not None:
                video_meta_data = get_video_meta_data(video_path=video_path)
                check_int(name=f'{self.__class__.__name__} bg_img', value=self.bg_img, min_value=-1, max_value=video_meta_data['frame_count'])
                video_bg_index = video_meta_data['frame_count'] - 1 if self.bg_img == -1 or self.bg_img > video_meta_data['frame_count'] else self.bg_img
                video_bg_img = read_frm_of_video(video_path=video_path, frame_index=video_bg_index, greyscale=False)
            else:
                video_bg_img = None
            if self.final_img_setting:
                self.make_location_heatmap_plot(frm_data=self.cum_sum_squares[-1, :, :],
                                                max_scale=video_styles[STYLE_MAX_SCALE],
                                                palette=video_styles[STYLE_PALETTE],
                                                aspect_ratio=aspect_ratio,
                                                bg_img=video_bg_img,
                                                color_legend=self.show_legend,
                                                min_seconds=self.min_seconds,
                                                line_clr=self.line_clr,
                                                heatmap_opacity=self.heatmap_opacity,
                                                file_name=os.path.join(self.heatmap_location_dir, f"{self.video_name}_final_frm.png"),
                                                shading=video_styles[STYLE_SHADING],
                                                img_size=(self.width, self.height))
            if self.video_setting or self.frame_setting:
                stdout_information(msg=f"Creating heatmap location video frames for video {self.video_name} ...")
                frame_arrays = np.array_split(self.cum_sum_squares, self.core_cnt)
                last_frm_idx = 0
                for frm_group in range(len(frame_arrays)):
                    split_arr = frame_arrays[frm_group]
                    frame_arrays[frm_group] = self.__insert_group_idx_column(data=split_arr, group=frm_group, last_frm_idx=last_frm_idx)
                    last_frm_idx = np.max(frame_arrays[frm_group].reshape((frame_arrays[frm_group].shape[0], -1)))
                constants = functools.partial(_heatmap_location,
                                              video_setting=self.video_setting,
                                              frame_setting=self.frame_setting,
                                              style_attr=video_styles,
                                              fps=self.fps,
                                              video_path=video_path,
                                              bg_img=self.bg_img,
                                              video_temp_dir=self.temp_folder,
                                              show_legend=self.show_legend,
                                              min_seconds=self.min_seconds,
                                              heatmap_opacity=self.heatmap_opacity,
                                              frame_dir=self.save_frame_folder_dir,
                                              kp_data=None if not self.show_keypoint else self.data_df,
                                              aspect_ratio=aspect_ratio,
                                              size=(self.width, self.height),
                                              video_name=self.video_name)
                for cnt, result in enumerate(self.pool.imap(constants,frame_arrays,chunksize=self.multiprocess_chunksize)):
                    stdout_information(msg=f'Batch {result}/{self.core_cnt} complete... Video: {self.video_name} ({file_cnt+1}/{len(self.data_paths)})')
                if self.video_setting:
                    stdout_information(msg=f"Joining {self.video_name} multiprocessed heatmap location video...")
                    concatenate_videos_in_folder(in_folder=self.temp_folder, save_path=self.save_video_path)
                video_timer.stop_timer()
                stdout_information(msg=f"Heatmap video {self.video_name} complete...", elapsed_time=video_timer.elapsed_time_str)
        self.timer.stop_timer()
        terminate_cpu_pool(pool=self.pool, force=False, source=self.__class__.__name__)
        stdout_success(msg=f"Heatmap location videos visualizations for {len(self.data_paths)} videos created in {self.heatmap_location_dir} directory", elapsed_time=self.timer.elapsed_time_str, source=self.__class__.__name__)


# if __name__ == "__main__":
#     test = HeatMapperLocationMultiprocess(config_path=r"E:\troubleshooting\mitra_emergence\project_folder\project_config.ini",
#                                           style_attr = {'palette': 'jet', 'shading': 'flat', 'bin_size': 50, 'max_scale': 30},
#                                           final_img_setting=True,
#                                           video_setting=True,
#                                           frame_setting=False,
#                                           show_keypoint=True,
#                                           time_slice={START_TIME: '00:00:00', END_TIME: '00:02:00'},
#                                           min_seconds=10,
#                                           bg_img=-1,
#                                           bodypart='nose',
#                                           core_cnt=15,
#                                           data_paths=r"E:\troubleshooting\mitra_emergence\project_folder\csv\outlier_corrected_movement_location\Box1_180mISOcontrol_Females.csv")
#     test.run()

# test = HeatMapperLocationMultiprocess(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/platea/project_folder/project_config.ini',
#                                       style_attr = {'palette': 'magma', 'shading': 'gouraud', 'bin_size': 80, 'max_scale': 'auto'},
#                                       final_img_setting=True,
#                                       video_setting=False,
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
# if __name__ == "__main__":
#     test = HeatMapperLocationMultiprocess(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/project_config.ini',
#                                           style_attr = {'palette': 'magma', 'shading': 'gouraud', 'bin_size': 100, 'max_scale': 'auto'},
#                                           final_img_setting=True,
#                                           video_setting=True,
#                                           frame_setting=False,
#                                           bodypart='Nose',
#                                           core_cnt=5,
#                                           data_paths=['/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/csv/machine_results/2022-06-20_NOB_DOT_4.csv'])
#     test.run()

# img = np.zeros((img_size[0], img_size[1], 3)).astype(np.uint8)
# for i in polygons:
#     coords = np.array(i.exterior.coords).astype(np.int)
#     print(coords)
#     cv2.polylines(img, [coords], True, (0, 0, 50), 2)
# cv2.imshow('img', img)
# cv2.waitKey()
