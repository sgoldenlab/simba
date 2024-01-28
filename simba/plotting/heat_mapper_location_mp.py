import functools
import multiprocessing
import os
import platform
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import jit, prange

from simba.mixins.config_reader import ConfigReader
from simba.mixins.geometry_mixin import GeometryMixin
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.enums import Defaults, Formats, TagNames
from simba.utils.errors import NoSpecifiedOutputError
from simba.utils.printing import SimbaTimer, log_event, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    find_core_cnt, get_fn_ext, read_df,
                                    remove_a_folder)


def _heatmap_location(
    data: np.array,
    video_setting: bool,
    frame_setting: bool,
    video_temp_dir: str,
    video_name: str,
    frame_dir: str,
    fps: int,
    style_attr: dict,
    max_scale: float,
    aspect_ratio: float,
    size: tuple,
    make_location_heatmap_plot: PlottingMixin.make_location_heatmap_plot,
):

    group = int(data[0][0][1])
    if video_setting:
        fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        video_save_path = os.path.join(video_temp_dir, "{}.mp4".format(str(group)))
        video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, size)

    for i in range(data.shape[0]):
        frame_id = int(data[i, 0, 0])
        frm_data = data[i, :, 2:]

        img = make_location_heatmap_plot(
            frm_data=frm_data,
            max_scale=float(max_scale),
            palette=style_attr["palette"],
            aspect_ratio=aspect_ratio,
            shading=style_attr["shading"],
            img_size=size,
            file_name=None,
            final_img=False,
        )

        print(
            "Heatmap frame created: {}, Video: {}, Processing core: {}".format(
                str(frame_id + 1), video_name, str(group + 1)
            )
        )

        if video_setting:
            video_writer.write(img)

        if frame_setting:
            file_path = os.path.join(frame_dir, "{}.png".format(frame_id))
            cv2.imwrite(file_path, img)

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

    def __init__(
        self,
        config_path: str,
        final_img_setting: bool,
        video_setting: bool,
        frame_setting: bool,
        bodypart: str,
        files_found: List[str],
        style_attr: dict,
        core_cnt: int,
    ):

        if (not frame_setting) and (not video_setting) and (not final_img_setting):
            raise NoSpecifiedOutputError(
                msg="Please choose to select either heatmap videos, frames, and/or final image.",
                source=self.__class__.__name__,
            )
        ConfigReader.__init__(self, config_path=config_path, create_logger=False)
        PlottingMixin.__init__(self)
        # log_event(logger_name=str(__class__.__name__), log_type=TagNames.CLASS_INIT.value, msg=self.create_log_msg_from_init_args(locals=locals()))
        if platform.system() == "Darwin":
            multiprocessing.set_start_method("spawn", force=True)
        self.frame_setting, self.video_setting = frame_setting, video_setting
        self.final_img_setting, self.bp = final_img_setting, bodypart
        self.style_attr, self.files_found = style_attr, files_found
        self.bin_size, self.max_scale, self.palette, self.shading, self.core_cnt = (
            style_attr["bin_size"],
            style_attr["max_scale"],
            style_attr["palette"],
            style_attr["shading"],
            core_cnt,
        )
        if not os.path.exists(self.heatmap_location_dir):
            os.makedirs(self.heatmap_location_dir)
        self.bp_lst = [self.bp + "_x", self.bp + "_y"]
        print("Processing {} video(s)...".format(str(len(self.files_found))))

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
        for file_cnt, file_path in enumerate(self.files_found):
            video_timer = SimbaTimer(start=True)
            _, self.video_name, _ = get_fn_ext(file_path)
            self.video_info, self.px_per_mm, self.fps = self.read_video_info(
                video_name=self.video_name
            )
            self.width, self.height = int(
                self.video_info["Resolution_width"].values[0]
            ), int(self.video_info["Resolution_height"].values[0])
            self.save_frame_folder_dir = os.path.join(
                self.heatmap_location_dir, self.video_name
            )
            self.video_folder = os.path.join(self.heatmap_location_dir, self.video_name)
            self.temp_folder = os.path.join(
                self.heatmap_location_dir, self.video_name, "temp"
            )
            if self.frame_setting:
                if os.path.exists(self.save_frame_folder_dir):
                    remove_a_folder(folder_dir=self.save_frame_folder_dir)
                if not os.path.exists(self.save_frame_folder_dir):
                    os.makedirs(self.save_frame_folder_dir)
            if self.video_setting:
                if os.path.exists(self.temp_folder):
                    remove_a_folder(folder_dir=self.temp_folder)
                    remove_a_folder(folder_dir=self.video_folder)
                os.makedirs(self.temp_folder)
                self.save_video_path = os.path.join(
                    self.heatmap_location_dir, f"{self.video_name}.mp4"
                )

            self.data_df = read_df(
                file_path=file_path, file_type=self.file_type, usecols=self.bp_lst
            )
            squares, aspect_ratio = GeometryMixin().bucket_img_into_grid_square(
                bucket_size_mm=self.style_attr["bin_size"],
                img_size=(self.width, self.height),
                px_per_mm=self.px_per_mm,
            )
            cum_sum_squares = GeometryMixin().cumsum_geometries(
                data=self.data_df.values, fps=self.fps, geometries=squares
            )
            if self.max_scale == "auto":
                self.max_scale = np.round(
                    np.max(np.max(cum_sum_squares[-1], axis=0)), 3
                )
                if self.max_scale == 0:
                    self.max_scale = 1
            else:
                self.max_scale = self.style_attr["max_scale"]

            if self.final_img_setting:
                self.make_location_heatmap_plot(
                    frm_data=cum_sum_squares[-1, :, :],
                    max_scale=self.max_scale,
                    palette=self.palette,
                    aspect_ratio=aspect_ratio,
                    file_name=os.path.join(
                        self.heatmap_location_dir, self.video_name + "_final_frm.png"
                    ),
                    shading=self.shading,
                    img_size=(self.width, self.height),
                    final_img=True,
                )

            if self.video_setting or self.frame_setting:
                print("Creating video frames...")
                frame_arrays = np.array_split(cum_sum_squares, self.core_cnt)
                last_frm_idx = 0
                for frm_group in range(len(frame_arrays)):
                    split_arr = frame_arrays[frm_group]
                    frame_arrays[frm_group] = self.__insert_group_idx_column(
                        data=split_arr, group=frm_group, last_frm_idx=last_frm_idx
                    )
                    last_frm_idx = np.max(
                        frame_arrays[frm_group].reshape(
                            (frame_arrays[frm_group].shape[0], -1)
                        )
                    )

                frm_per_core = frame_arrays[0].shape[0]
                with multiprocessing.Pool(
                    self.core_cnt, maxtasksperchild=self.maxtasksperchild
                ) as pool:
                    constants = functools.partial(
                        _heatmap_location,
                        video_setting=self.video_setting,
                        frame_setting=self.frame_setting,
                        style_attr=self.style_attr,
                        fps=self.fps,
                        video_temp_dir=self.temp_folder,
                        frame_dir=self.save_frame_folder_dir,
                        max_scale=self.max_scale,
                        aspect_ratio=aspect_ratio,
                        size=(self.width, self.height),
                        video_name=self.video_name,
                        make_location_heatmap_plot=self.make_location_heatmap_plot,
                    )
                    for cnt, result in enumerate(
                        pool.imap(
                            constants,
                            frame_arrays,
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
                    print(
                        "Joining {} multiprocessed heatmap location video...".format(
                            self.video_name
                        )
                    )
                    concatenate_videos_in_folder(
                        in_folder=self.temp_folder, save_path=self.save_video_path
                    )

                video_timer.stop_timer()
                print(
                    "Heatmap video {} complete (elapsed time: {}s) ...".format(
                        self.video_name, video_timer.elapsed_time_str
                    )
                )

            self.timer.stop_timer()
            stdout_success(
                msg=f"Heatmap location videos visualizations for {len(self.files_found)} videos created in project_folder/frames/output/heatmaps_locations directory",
                elapsed_time=self.timer.elapsed_time_str,
                source=self.__class__.__name__,
            )


# test = HeatMapperLocationMultiprocess(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                       style_attr = {'palette': 'jet', 'shading': 'gouraud', 'bin_size': 80, 'max_scale': 'auto'},
#                                       final_img_setting=True,
#                                       video_setting=False,
#                                       frame_setting=False,
#                                       bodypart='Nose_1',
#                                       core_cnt=5,
#                                       files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/outlier_corrected_movement_location/Together_1.csv'])
# test.run()

# test = HeatMapperLocationMultiprocess(config_path='/Users/simon/Desktop/envs/troubleshooting/Rat_NOR/project_folder/project_config.ini',
#                                       style_attr = {'palette': 'jet', 'shading': 'gouraud', 'bin_size': 100, 'max_scale': 'auto'},
#                                       final_img_setting=True,
#                                       video_setting=False,
#                                       frame_setting=False,
#                                       bodypart='Nose',
#                                       core_cnt=5,
#                                       files_found=['/Users/simon/Desktop/envs/troubleshooting/Rat_NOR/project_folder/csv/machine_results/2022-06-26_NOB_DOT_4.csv'])
# test.run()

# img = np.zeros((img_size[0], img_size[1], 3)).astype(np.uint8)
# for i in polygons:
#     coords = np.array(i.exterior.coords).astype(np.int)
#     print(coords)
#     cv2.polylines(img, [coords], True, (0, 0, 50), 2)
# cv2.imshow('img', img)
# cv2.waitKey()
