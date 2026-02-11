__author__ = "Simon Nilsson; sronilsson@gmail.com"

import functools
import multiprocessing
import os
import platform
from typing import List, Union, Tuple, Optional, Dict
from copy import deepcopy
import cv2
import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.geometry_mixin import GeometryMixin
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log, check_float,
    check_filepaths_in_iterable_exist, check_int, check_str,
    check_valid_boolean, check_valid_dataframe, check_valid_dict, check_if_keys_exist_in_dict, check_if_string_value_is_valid_video_timestamp, check_that_hhmmss_start_is_before_end)
from simba.utils.data import get_cpu_pool, terminate_cpu_pool, find_frame_numbers_from_time_stamp
from simba.utils.enums import Formats
from simba.utils.lookups import get_named_colors
from simba.utils.errors import InvalidInputError, NoSpecifiedOutputError, FrameRangeError, NoDataError
from simba.utils.printing import SimbaTimer, stdout_success, stdout_information
from simba.utils.read_write import (concatenate_videos_in_folder, get_video_meta_data, find_core_cnt, get_fn_ext, read_df, remove_a_folder, seconds_to_timestamp, read_frm_of_video, find_video_of_file)

START_TIME = 'start_time'
END_TIME = 'end_time'
AUTO = 'auto'

def _heatmap_multiprocessor(data: np.array,
                            video_setting: bool,
                            frame_setting: bool,
                            video_temp_dir: str,
                            video_name: str,
                            frame_dir: str,
                            fps: int,
                            verbose: bool,
                            heatmap_opacity: Union[float, None],
                            kp_data: Union[pd.DataFrame, None],
                            style_attr: dict,
                            bg_img: Union[int, None],
                            video_path: Union[str, os.PathLike, None],
                            min_seconds: Optional[int],
                            line_clr: Union[str, None],
                            max_scale: float,
                            clf_name: str,
                            show_legend: bool,
                            aspect_ratio: float,
                            size: Tuple[int, int]):

    batch, frm_ids, data = data[0], data[1], data[2]
    video_writer = None
    static_bg, kp_size = None, None
    if video_path is not None:
        if bg_img > -1:
            static_bg = read_frm_of_video(video_path=video_path, frame_index=bg_img, greyscale=False)
        kp_size = PlottingMixin().get_optimal_circle_size(frame_size=size, circle_frame_ratio=50)
    for frm_idx in range(data.shape[0]):
        frame_id, frm_data = int(frm_ids[frm_idx]), data[frm_idx]
        if bg_img is not None:
            if static_bg is not None:
                frm_bg_img = deepcopy(static_bg)
            else: frm_bg_img = read_frm_of_video(video_path=video_path, frame_index=frame_id, greyscale=False)
        else:
            frm_bg_img = None
        img = PlottingMixin.make_location_heatmap_plot(frm_data=frm_data,
                                                       max_scale=max_scale,
                                                       palette=style_attr["palette"],
                                                       aspect_ratio=aspect_ratio,
                                                       min_seconds=min_seconds,
                                                       bg_img=frm_bg_img,
                                                       line_clr=line_clr,
                                                       heatmap_opacity=heatmap_opacity,
                                                       shading=style_attr["shading"],
                                                       legend_lbl=f'location {clf_name} (seconds)',
                                                       img_size=size,
                                                       color_legend=show_legend)
        if kp_data is not None and kp_size is not None:
            frm_point = kp_data.loc[frame_id].values.astype(np.int32)
            img = cv2.circle(img, (frm_point[0], frm_point[1]), kp_size, (255, 0, 0), -1)
        if video_setting:
            if video_writer is None:
                h, w = img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
                video_save_path = os.path.join(video_temp_dir, f"{batch}.mp4")
                out_size = (w, h)
                video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, out_size)
            video_writer.write(img[:, :, :3].astype(np.uint8))
        if frame_setting:
            file_path = os.path.join(frame_dir, f"{frame_id}.png")
            cv2.imwrite(file_path, img)
        if verbose: stdout_information(msg=f"Heatmap frame created {frame_id + 1}, time-stamp: {seconds_to_timestamp(seconds=frame_id/fps)} (video: {video_name}, processing core: {batch}, core frame: {frm_idx}/{len(data)})...")
    if video_setting:
        video_writer.release()

    return batch


class HeatMapperClfMultiprocess(ConfigReader, PlottingMixin):
    """
    Create heatmaps representing the locations of the classified behavior.

    .. note::
       `GitHub visualizations tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-11-visualizations>`__.

    .. image:: _static/img/heatmap.png
       :width: 500
       :align: center

    .. image:: _static/img/Box1_180mISOcontrol_Females_GROOMING_final_frm.webp
       :width: 600
       :align: center

    :param Union[str, os.PathLike] config_path: Path to SimBA project config file.
    :param str bodypart: Body-part used to locate where the behavior occurs. When the classifier fires, SimBA records this body-part's position.
    :param str clf_name: Name of the classifier/behavior to visualize (e.g. 'Attack', 'Grooming').
    :param List[str] data_paths: Path(s) to classifier results CSV files (from machine_results). Must match videos in project.
    :param Dict[str, Any] style_attr: Dict with keys 'palette', 'shading', 'bin_size', 'max_scale'. E.g. {'palette': 'jet', 'shading': 'gouraud', 'bin_size': 50, 'max_scale': 'auto'}.
    :param bool show_legend: If True, append color bar showing seconds scale. Default True.
    :param bool final_img_setting: If True, create a single cumulative heatmap image. Default True.
    :param Optional[int] bg_img: If set, overlay heatmap on video frame. -1 or None = no background. Non-negative int = frame index for static background.
    :param Optional[float] heatmap_opacity: Opacity of heatmap over background (0–1). Used when bg_img is set. Default None.
    :param bool video_setting: If True, create heatmap video. Default False.
    :param bool verbose: If True, print progress. Default True.
    :param bool show_keypoint: If True, draw body-part position as dot on each frame. Default False.
    :param Optional[int] min_seconds: Hide bins with time below this (seconds). Bins below threshold appear empty. Default None.
    :param bool frame_setting: If True, create individual heatmap frame images. Default False.
    :param Optional[Dict[str, str]] time_slice: If set, restrict analysis to time period. Dict with keys 'start_time' and 'end_time' (HH:MM:SS). Default None.
    :param int core_cnt: Number of CPU cores. -1 = use all available. Default -1.

    :example:
    >>> style_attr = {'palette': 'jet', 'shading': 'gouraud', 'bin_size': 50, 'max_scale': 'auto'}
    >>> heatmapper = HeatMapperClfMultiprocess(config_path='project_config.ini', bodypart='Nose_1', clf_name='Attack', data_paths=['csv/machine_results/Video1.csv'], style_attr=style_attr, final_img_setting=True, video_setting=False, frame_setting=False, core_cnt=-1)
    >>> heatmapper.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 bodypart: str,
                 clf_name: str,
                 data_paths: List[str],
                 style_attr: dict,
                 show_legend: bool = True,
                 final_img_setting: bool = True,
                 bg_img: Optional[int] = None,
                 heatmap_opacity: Optional[float] = None,
                 video_setting: bool = False,
                 verbose: bool = True,
                 line_clr: Optional[str] = None,
                 show_keypoint: bool = False,
                 min_seconds: Optional[int] = None,
                 frame_setting: bool = False,
                 time_slice: Optional[Dict[str, str]] = None,
                 core_cnt: int = -1):

        ConfigReader.__init__(self, config_path=config_path, create_logger=False)
        PlottingMixin.__init__(self)

        if platform.system() == "Darwin":
            multiprocessing.set_start_method("spawn", force=True)
        check_valid_boolean(value=[frame_setting, video_setting, final_img_setting], source=self.__class__.__name__)
        if (not frame_setting) and (not video_setting) and (not final_img_setting):
            raise NoSpecifiedOutputError(msg="Please choose to select either heatmap videos, frames, and/or final image.")
        check_filepaths_in_iterable_exist(file_paths=data_paths, name=f'{self.__class__.__name__} data_paths')
        check_str(name=f'{self.__class__.__name__} clf_name', value=clf_name)
        check_str(name=f'{self.__class__.__name__} bodypart', value=bodypart)
        if line_clr is not None: check_str(name=f'{self.__class__.__name__} line_clr', value=line_clr, options=get_named_colors())
        check_int(name=f'{self.__class__.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0])
        check_valid_boolean(value=show_legend, source=f'{self.__class__.__name__} show_legend', raise_error=True)
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose', raise_error=True)
        if bg_img is not None: check_int(name=f'{self.__class__.__name__} bg_img', value=bg_img, min_value=-1)
        self.frame_setting, self.video_setting, self.show_legend = frame_setting, video_setting, show_legend
        if min_seconds is not None: check_int(name=f'{self.__class__.__name__} min_seconds', value=min_seconds, min_value=1)
        if heatmap_opacity is not None: check_float(name=f'{self.__class__.__name__} heatmap_opacity',  value=heatmap_opacity, min_value=0, max_value=1.0)
        self.final_img_setting, self.bp, self.verbose = final_img_setting, bodypart, verbose
        check_valid_dict(x=style_attr, required_keys=('max_scale', 'bin_size', 'shading', 'palette'))
        check_valid_boolean(value=show_keypoint, source=f'{self.__class__.__name__}, show_keypoint', raise_error=True)
        if time_slice is not None:
            check_if_keys_exist_in_dict(data=time_slice, key=[START_TIME, END_TIME], name=f'{self.__class__.__name__} slicing')
            check_if_string_value_is_valid_video_timestamp(value=time_slice[START_TIME], name="Video slicing START TIME")
            check_if_string_value_is_valid_video_timestamp(value=time_slice[END_TIME], name="Video slicing END TIME")
            check_that_hhmmss_start_is_before_end(start_time=time_slice[START_TIME], end_time=time_slice[END_TIME], name="SLICE TIME STAMPS")


        self.style_attr, self.time_slice, self.bg_img, self.show_keypoint = style_attr, time_slice, bg_img, show_keypoint
        self.bin_size, self.max_scale, self.palette, self.shading, self.line_clr = style_attr["bin_size"], style_attr["max_scale"], style_attr["palette"], style_attr["shading"], line_clr
        self.clf_name, self.data_paths, self.min_seconds, self.heatmap_opacity = clf_name, data_paths, min_seconds, heatmap_opacity
        self.core_cnt = [find_core_cnt()[0] if core_cnt == -1 or core_cnt > find_core_cnt()[0] else core_cnt][0]
        if not os.path.exists(self.heatmap_clf_location_dir):
            os.makedirs(self.heatmap_clf_location_dir)
        self.bp_lst = [f"{self.bp}_x", f"{self.bp}_y"]

    def __calculate_max_scale(self, clf_array: np.array):
        return np.round(np.max(np.max(clf_array[-1], axis=0)), 3)

    def run(self):
        stdout_information(msg=f"Processing {len(self.data_paths)} video(s) for {self.clf_name} heatmaps...")
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        pool = get_cpu_pool(core_cnt=self.core_cnt, source=self.__class__.__name__)
        for file_cnt, file_path in enumerate(self.data_paths):
            video_timer = SimbaTimer(start=True)
            _, self.video_name, _ = get_fn_ext(file_path)
            stdout_information(f'Plotting heatmap classification map for video {self.video_name}...')
            self.video_info, self.px_per_mm, self.fps = self.read_video_info(video_name=self.video_name)
            self.width, self.height = int(self.video_info["Resolution_width"].values[0]), int(self.video_info["Resolution_height"].values[0])
            self.temp_folder = os.path.join(self.heatmap_clf_location_dir, "temp")
            self.frames_save_dir = os.path.join(self.heatmap_clf_location_dir, f"{self.video_name}_{self.clf_name}")
            if self.video_setting:
                if os.path.exists(self.temp_folder):
                    remove_a_folder(folder_dir=self.temp_folder)
                os.makedirs(self.temp_folder)
                self.save_video_path = os.path.join(self.heatmap_clf_location_dir, f"{self.video_name}_{self.clf_name}.mp4")
            if self.frame_setting:
                if os.path.exists(self.frames_save_dir):
                    remove_a_folder(folder_dir=self.frames_save_dir)
                os.makedirs(self.frames_save_dir)
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
            self.data_df = read_df(file_path=file_path, file_type=self.file_type)
            check_valid_dataframe(df=self.data_df, required_fields=[self.clf_name] + self.bp_lst, valid_dtypes=Formats.NUMERIC_DTYPES.value)
            if self.time_slice is not None:
                frm_numbers = find_frame_numbers_from_time_stamp(start_time=self.time_slice[START_TIME], end_time=self.time_slice[END_TIME], fps=self.fps)
                if len(set(frm_numbers) - set(self.data_df.index)) > 0:
                    raise FrameRangeError(msg=f'The chosen time-period ({self.time_slice[START_TIME]} - {self.time_slice[END_TIME]}) does not exist in {self.video_name}.', source=self.__class__.__name__)
                self.data_df = self.data_df.loc[frm_numbers]
            bp_data = self.data_df[self.bp_lst].values.astype(np.int32)
            clf_data = self.data_df[self.clf_name].values.astype(np.int32)
            if len(np.unique(clf_data)) == 1:
                raise InvalidInputError(msg=f'Cannot plot heatmap for behavior {self.clf_name} in video {self.video_name}. The behavior is classified as {np.unique(clf_data)} in every single frame.')
            grid, aspect_ratio = GeometryMixin.bucket_img_into_grid_square(img_size=(self.width, self.height), bucket_grid_size_mm=self.bin_size, px_per_mm=self.px_per_mm, add_correction=False, verbose=False)
            clf_data = GeometryMixin().cumsum_bool_geometries(data=bp_data, geometries=grid, bool_data=clf_data, fps=self.fps, verbose=False, core_cnt=self.core_cnt, pool=pool)
            if self.max_scale == AUTO:
                video_max_scale = max(1, self.__calculate_max_scale(clf_array=clf_data))
            else:
                video_max_scale = deepcopy(self.max_scale)
            if self.final_img_setting:
                print(self.min_seconds)
                file_name = os.path.join(self.heatmap_clf_location_dir, f"{self.video_name}_{self.clf_name}_final_frm.png")
                self.make_location_heatmap_plot(frm_data=clf_data[-1:, :, :][0],
                                                max_scale=video_max_scale,
                                                palette=self.palette,
                                                aspect_ratio=aspect_ratio,
                                                file_name=file_name,
                                                min_seconds=self.min_seconds,
                                                heatmap_opacity=self.heatmap_opacity,
                                                color_legend=self.show_legend,
                                                line_clr=self.line_clr,
                                                bg_img=video_bg_img,
                                                legend_lbl= f'location {self.clf_name} (seconds)',
                                                shading=self.shading,
                                                img_size=(self.width, self.height))
                if self.verbose: stdout_information(f"Final heatmap image saved at {file_name}.")

            if self.video_setting or self.frame_setting:
                frame_arrays = np.array_split(clf_data, self.core_cnt)
                frm_per_core_w_batch = []
                frm_cnt = 0
                for batch_cnt in range(len(frame_arrays)):
                    frm_range = np.arange(frm_cnt, frm_cnt+ frame_arrays[batch_cnt].shape[0])
                    frm_cnt += len(frm_range)
                    frm_per_core_w_batch.append((batch_cnt, frm_range, frame_arrays[batch_cnt]))
                del frame_arrays
                if self.verbose: stdout_information(f"Creating heatmaps, multiprocessing (chunksize: {self.multiprocess_chunksize}, cores: {self.core_cnt})...")
                constants = functools.partial(_heatmap_multiprocessor,
                                              video_setting=self.video_setting,
                                              frame_setting=self.frame_setting,
                                              style_attr=self.style_attr,
                                              fps=self.fps,
                                              video_temp_dir=self.temp_folder,
                                              frame_dir=self.frames_save_dir,
                                              max_scale=video_max_scale,
                                              aspect_ratio=aspect_ratio,
                                              bg_img=self.bg_img,
                                              line_clr=self.line_clr,
                                              heatmap_opacity=self.heatmap_opacity,
                                              min_seconds=self.min_seconds,
                                              kp_data=None if not self.show_keypoint else self.data_df,
                                              clf_name=self.clf_name,
                                              verbose=self.verbose,
                                              show_legend=self.show_legend,
                                              size=(self.width, self.height),
                                              video_name=self.video_name,
                                              video_path=video_path)

                for cnt, batch in enumerate(pool.imap(constants, frm_per_core_w_batch, chunksize=self.multiprocess_chunksize)):
                    if self.verbose: stdout_information(msg=f'Batch core {batch+1}/{self.core_cnt} complete (Video {self.video_name})... ')
                if self.video_setting:
                    if self.verbose: stdout_information(msg=f"Joining {self.video_name} multiprocessed video...")
                    concatenate_videos_in_folder(in_folder=self.temp_folder, save_path=self.save_video_path)
                video_timer.stop_timer()
                if self.verbose: stdout_information(msg=f"Heatmap video {self.video_name} complete, (elapsed time: {video_timer.elapsed_time_str}s) ...")
        terminate_cpu_pool(pool=pool, force=False, source=self.__class__.__name__)
        self.timer.stop_timer()
        if self.verbose: stdout_success(msg=f"Heatmap visualizations for {len(self.data_paths)} video(s) created in {self.heatmap_clf_location_dir} directory", elapsed_time=self.timer.elapsed_time_str)


# if __name__ == "__main__":
#     test = HeatMapperClfMultiprocess(config_path=r"E:\troubleshooting\mitra_emergence\project_folder\project_config.ini",
#                              style_attr = {'palette': 'jet', 'shading': 'flat', 'bin_size': 60, 'max_scale': 60},
#                              final_img_setting=True,
#                              video_setting=True,
#                              frame_setting=False,
#                              min_seconds=1,
#                              bg_img=-1,
#                              bodypart='center',
#                              clf_name='GROOMING',
#                              show_keypoint=True,
#                              time_slice=None, #{START_TIME: '00:00:00', END_TIME: '00:02:00'},
#                              core_cnt=10,
#                              data_paths=[r"E:\troubleshooting\mitra_emergence\project_folder\csv\machine_results\Box1_180mISOcontrol_Females.csv"])
#     test.run()



# if __name__ == "__main__":
#     test = HeatMapperClfMultiprocess(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini",
#                          style_attr = {'palette': 'jet', 'shading': 'gouraud', 'bin_size': 50, 'max_scale': 'auto'},
#                          final_img_setting=True,
#                          video_setting=True,
#                          frame_setting=False,
#                          bodypart='Ear_left',
#                          clf_name='straub_tail',
#                          data_paths=[r"C:\troubleshooting\RAT_NOR\project_folder\csv\test\2022-06-20_NOB_DOT_4.csv"])
#     test.run()

# test = HeatMapperClfMultiprocess(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                      style_attr = {'palette': 'jet', 'shading': 'gouraud', 'bin_size': 100, 'max_scale': 'auto'},
#                      final_img_setting=False,
#                      video_setting=True,
#                      frame_setting=False,
#                      bodypart='Nose_1',
#                      clf_name='Attack',
#                                core_cnt=5,
#                      files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'])
# test.create_heatmaps()
# if __name__ == "__main__":
#     x = HeatMapperClfMultiprocess(config_path=r"E:\troubleshooting\mitra_emergence\project_folder\project_config.ini",
#                               bodypart='nose',
#                               clf_name='GROOMING',
#                               style_attr={'palette': 'jet', 'shading': 'gouraud', 'bin_size': 25, 'max_scale': 'auto'},
#                               final_img_setting=True,
#                               video_setting=False,
#                               frame_setting=False,
#                               core_cnt=12,
#                               data_paths=[r"E:\troubleshooting\mitra_emergence\project_folder\csv\machine_results\Box1_180mISOcontrol_Females.csv"])
#
#     x.run()

    # def __init__(self,
    #              config_path: Union[str, os.PathLike],
    #              bodypart: str,
    #              clf_name: str,
    #              data_paths: List[str],
    #              style_attr: dict,
    #              final_img_setting: bool = True,
    #              video_setting: bool = False,
    #              frame_setting: bool = False,
    #              core_cnt: int = -1):
