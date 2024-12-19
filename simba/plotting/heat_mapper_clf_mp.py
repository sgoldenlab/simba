__author__ = "Simon Nilsson"

import functools
import multiprocessing
import os
import platform
from typing import Union, List

import cv2
import numpy as np
import pandas as pd
from numba import jit, prange

import simba.mixins.plotting_mixin
from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.mixins.geometry_mixin import GeometryMixin
from simba.utils.enums import Formats
from simba.utils.errors import NoSpecifiedOutputError, InvalidInputError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder, get_fn_ext, read_df, remove_a_folder, find_core_cnt)
from simba.utils.checks import check_valid_boolean, check_int, check_str, check_valid_dict, check_filepaths_in_iterable_exist, check_all_file_names_are_represented_in_video_log, check_valid_dataframe


def _heatmap_multiprocessor(data: np.array,
                            video_setting: bool,
                            frame_setting: bool,
                            video_temp_dir: str,
                            video_name: str,
                            frame_dir: str,
                            fps: int,
                            style_attr: dict,
                            max_scale: float,
                            clf_name: str,
                            aspect_ratio: float,
                            size: tuple,
                            make_clf_heatmap_plot: simba.mixins.plotting_mixin.PlottingMixin.make_clf_heatmap_plot):

    batch, frm_ids, data = data[0], data[1], data[2]
    if video_setting:
        fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        video_save_path = os.path.join(video_temp_dir, f"{batch}.mp4")
        video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, size)
    for frm_idx in range(data.shape[0]):
        frame_id, frm_data = int(frm_ids[frm_idx]), data[frm_idx]
        img = make_clf_heatmap_plot(frm_data=frm_data,
                                    max_scale=max_scale,
                                    palette=style_attr["palette"],
                                    aspect_ratio=aspect_ratio,
                                    shading=style_attr["shading"],
                                    clf_name=clf_name,
                                    img_size=size)

        print(f"Heatmap frame created: {frame_id + 1}, Video: {video_name}, Processing core: {batch+1}")
        if video_setting:
            video_writer.write(img)
        if frame_setting:
            file_path = os.path.join(frame_dir, f"{frame_id}.png")
            cv2.imwrite(file_path, img)

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

    :param str config_path: path to SimBA project config file in Configparser format
    :param bool final_img_setting: If True, then create a single image representing the last frame of the input video
    :param bool video_setting: If True, then create a video of heatmaps.
    :param bool frame_setting: If True, then create individual heatmap frames.
    :param int bin_size: The rectangular size of each heatmap location in millimeters. For example, `50` will divide the video into
        5 centimeter rectangular spatial bins.
    :param str palette: Heatmap pallette. Eg. 'jet', 'magma', 'inferno','plasma', 'viridis', 'gnuplot2'
    :param str bodypart: The name of the body-part used to infer the location of the classified behavior
    :param str clf_name: The name of the classified behavior.
    :param int or 'auto' max_scale: The max value in the heatmap in seconds. E.g., with a value of `10`, if the classified behavior has occured
        >= 10 within a rectangular bins, it will be filled with the same color.
    :param int core_cnt: Number of cores to use.

    :example I:
    >>> heat_mapper_clf = HeatMapperClfMultiprocess(config_path='MyConfigPath', final_img_setting=False, video_setting=True, frame_setting=False, bin_size=50, palette='jet', bodypart='Nose_1', clf_name='Attack', max_scale=20)
    >>> heat_mapper_clf.create_heatmaps()


    :example II:
    >>> test = HeatMapperClfMultiprocess(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini",
    >>>                          style_attr = {'palette': 'jet', 'shading': 'gouraud', 'bin_size': 50, 'max_scale': 'auto'},
    >>>                          final_img_setting=True,
    >>>                          video_setting=True,
    >>>                          frame_setting=True,
    >>>                          bodypart='Ear_left',
    >>>                          clf_name='straub_tail',
    >>>                          data_paths=[r"C:\troubleshooting\RAT_NOR\project_folder\csv\test\2022-06-20_NOB_DOT_4.csv"])
    >>> test.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 bodypart: str,
                 clf_name: str,
                 data_paths: List[str],
                 style_attr: dict,
                 final_img_setting: bool = True,
                 video_setting: bool = False,
                 frame_setting: bool = False,
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
        check_int(name=f'{self.__class__.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0])
        self.frame_setting, self.video_setting = frame_setting, video_setting
        self.final_img_setting, self.bp = final_img_setting, bodypart
        check_valid_dict(x=style_attr, required_keys=('max_scale', 'bin_size', 'shading', 'palette'))
        self.style_attr = style_attr
        self.bin_size, self.max_scale, self.palette, self.shading = (style_attr["bin_size"], style_attr["max_scale"], style_attr["palette"], style_attr["shading"])
        self.clf_name, self.data_paths = clf_name, data_paths
        self.core_cnt = [find_core_cnt()[0] if core_cnt == -1 or core_cnt > find_core_cnt()[0] else core_cnt][0]
        if not os.path.exists(self.heatmap_clf_location_dir):
            os.makedirs(self.heatmap_clf_location_dir)
        self.bp_lst = [f"{self.bp}_x", f"{self.bp}_y"]

    def __calculate_max_scale(self, clf_array: np.array):
        return np.round(np.max(np.max(clf_array[-1], axis=0)), 3)

    def run(self):
        print(f"Processing {len(self.data_paths)} video(s)...")
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        for file_cnt, file_path in enumerate(self.data_paths):
            video_timer = SimbaTimer(start=True)
            _, self.video_name, _ = get_fn_ext(file_path)
            print(f'Plotting heatmap classification map for video {self.video_name}...')
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
            self.data_df = read_df(file_path=file_path, file_type=self.file_type)
            check_valid_dataframe(df=self.data_df, required_fields=[self.clf_name] + self.bp_lst, valid_dtypes=Formats.NUMERIC_DTYPES.value)
            bp_data = self.data_df[self.bp_lst].values.astype(np.int32)
            clf_data = self.data_df[self.clf_name].values.astype(np.int32)
            if len(np.unique(clf_data)) == 1:
                raise InvalidInputError(msg=f'Cannot plot heatmap for behavior {self.clf_name} in video {self.video_name}. The behavior is classified as {np.unique(clf_data)} in every single frame.')
            grid, aspect_ratio = GeometryMixin.bucket_img_into_grid_square(img_size=(self.width, self.height), bucket_grid_size_mm=self.bin_size, px_per_mm=self.px_per_mm, add_correction=False, verbose=False)
            clf_data = GeometryMixin().cumsum_bool_geometries(data=bp_data, geometries=grid, bool_data=clf_data, fps=self.fps, verbose=False)
            if self.max_scale == "auto":
                self.max_scale = max(1, self.__calculate_max_scale(clf_array=clf_data))
            if self.final_img_setting:
                file_name = os.path.join(self.heatmap_clf_location_dir, f"{self.video_name}_{self.clf_name}_final_frm.png")
                self.make_location_heatmap_plot(frm_data=clf_data[-1:, :, :][0],
                                                max_scale=self.max_scale,
                                                palette=self.palette,
                                                aspect_ratio=aspect_ratio,
                                                file_name=file_name,
                                                shading=self.shading,
                                                img_size=(self.width, self.height))
                print(f"Final heatmap image saved at {file_name}.")

            if self.video_setting or self.frame_setting:
                frame_arrays = np.array_split(clf_data, self.core_cnt)
                frm_per_core_w_batch = []
                frm_cnt = 0
                for batch_cnt in range(len(frame_arrays)):
                    frm_range = np.arange(frm_cnt, frm_cnt+ frame_arrays[batch_cnt].shape[0])
                    frm_cnt += len(frm_range)
                    frm_per_core_w_batch.append((batch_cnt, frm_range, frame_arrays[batch_cnt]))
                del frame_arrays
                print(f"Creating heatmaps, multiprocessing (chunksize: {self.multiprocess_chunksize}, cores: {self.core_cnt})...")
                with multiprocessing.Pool(self.core_cnt, maxtasksperchild=self.maxtasksperchild) as pool:
                    constants = functools.partial(_heatmap_multiprocessor,
                                                  video_setting=self.video_setting,
                                                  frame_setting=self.frame_setting,
                                                  style_attr=self.style_attr,
                                                  fps=self.fps,
                                                  video_temp_dir=self.temp_folder,
                                                  frame_dir=self.frames_save_dir,
                                                  max_scale=self.max_scale,
                                                  aspect_ratio=aspect_ratio,
                                                  clf_name=self.clf_name,
                                                  size=(self.width, self.height),
                                                  video_name=self.video_name,
                                                  make_clf_heatmap_plot=self.make_clf_heatmap_plot)

                    for cnt, batch in enumerate(pool.imap(constants, frm_per_core_w_batch, chunksize=self.multiprocess_chunksize)):
                        print(f'Batch core {batch+1}/{self.core_cnt} complete (Video {self.video_name})... ')
                    pool.terminate()
                    pool.join()

                if self.video_setting:
                    print(f"Joining {self.video_name} multiprocessed video...")
                    concatenate_videos_in_folder(in_folder=self.temp_folder, save_path=self.save_video_path)

                video_timer.stop_timer()
                print(f"Heatmap video {self.video_name} complete, (elapsed time: {video_timer.elapsed_time_str}s) ...")

        self.timer.stop_timer()
        stdout_success(msg=f"Heatmap visualizations for {len(self.data_paths)} video(s) created in {self.heatmap_clf_location_dir} directory", elapsed_time=self.timer.elapsed_time_str)


# if __name__ == "__main__":
#     test = HeatMapperClfMultiprocess(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini",
#                              style_attr = {'palette': 'jet', 'shading': 'gouraud', 'bin_size': 50, 'max_scale': 'auto'},
#                              final_img_setting=True,
#                              video_setting=True,
#                              frame_setting=True,
#                              bodypart='Ear_left',
#                              clf_name='straub_tail',
#                              data_paths=[r"C:\troubleshooting\RAT_NOR\project_folder\csv\test\2022-06-20_NOB_DOT_4.csv"])
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
