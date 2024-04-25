__author__ = "Simon Nilsson"
import functools
import multiprocessing
import os
import platform
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from numba import jit

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_file_exist_and_readable, check_instance, check_int, check_valid_lst)
from simba.utils.errors import (CountError, InvalidInputError,
                                NoSpecifiedOutputError)
from simba.utils.lookups import get_color_dict
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    find_core_cnt, get_fn_ext, read_df)


def distance_plotter_mp(
    frm_cnts: np.array,
    distances: np.ndarray,
    colors: List[str],
    video_setting: bool,
    frame_setting: bool,
    video_name: str,
    video_save_dir: str,
    frame_folder_dir: str,
    style_attr: dict,
    fps: int,
):

    group = int(distances[frm_cnts[0], 0])
    video_writer = None
    if video_setting:
        fourcc = cv2.VideoWriter_fourcc(*"DIVX")
        temp_video_save_path = os.path.join(video_save_dir, f"{group}.avi")
        video_writer = cv2.VideoWriter(
            temp_video_save_path,
            fourcc,
            fps,
            (style_attr["width"], style_attr["height"]),
        )

    for frm_cnt in frm_cnts:
        line_data = distances[:frm_cnt, 1:]
        line_data = np.hsplit(line_data, line_data.shape[1])

        img = PlottingMixin.make_line_plot_plotly(
            data=line_data,
            colors=colors,
            width=style_attr["width"],
            height=style_attr["height"],
            line_width=style_attr["line width"],
            font_size=style_attr["font size"],
            title="Animal distances",
            y_lbl="distance (cm)",
            x_lbl="frame count",
            x_lbl_divisor=fps,
            y_max=style_attr["y_max"],
            line_opacity=style_attr["opacity"],
            save_path=None,
        ).astype(np.uint8)
        if video_setting:
            video_writer.write(img[:, :, :3])
        if frame_setting:
            frm_name = os.path.join(frame_folder_dir, f"{frm_cnt}.png")
            cv2.imwrite(frm_name, np.uint8(img))
        print(
            f"Distance frame created: {frm_cnt} (of {distances.shape[0]}), Video: {video_name}, Processing core: {group}"
        )
    if video_setting:
        video_writer.release()
    return group


class DistancePlotterMultiCore(ConfigReader, PlottingMixin):
    """
     Visualize the distances between pose-estimated body-parts (e.g., two animals) through line
     charts. Results are saved as individual line charts, and/or a video of line charts.
     Uses multiprocessing.

     :param str config_path: path to SimBA project config file in Configparser format
     :param bool frame_setting: If True, creates individual frames.
     :param bool video_setting: If True, creates videos.
     :param bool final_img: If True, creates a single .png representing the entire video.
     :param dict style_attr: Video style attributes (font sizes, line opacity etc.)
     :param List[Union[str, os.PathLike]] data_paths: Files to visualize.
     :param dict line_attr: Representing the body-parts to visualize the distance between and their colors.

    .. note::
       `GitHub tutorial/documentation <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-11-visualizations>`__.

    .. image:: _static/img/DistancePlotterMultiCore.png
       :width: 600
       :align: center

    .. image:: _static/img/DistancePlotterMultiCore_1.gif
       :width: 600
       :align: center

    :example:
    >>> style_attr = {'width': 640, 'height': 480, 'line width': 6, 'font size': 8, 'opacity': 0.5}
    >>> line_attr = {0: ['Center_1', 'Center_2', 'Green'], 1: ['Ear_left_2', 'Ear_left_1', 'Red']}
    >>> distance_plotter = DistancePlotterMultiCore(config_path=r'/tests_/project_folder/project_config.ini', frame_setting=False, video_setting=True, final_img=True, style_attr=style_attr, line_attr=line_attr,  files_found=['/test_/project_folder/csv/machine_results/Together_1.csv'], core_cnt=5)
    >>> distance_plotter.run()
    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        data_paths: List[Union[str, os.PathLike]],
        frame_setting: bool,
        video_setting: bool,
        final_img: bool,
        style_attr: Dict[str, int],
        line_attr: Dict[int, list],
        core_cnt: Optional[int] = -1,
    ):

        if (not frame_setting) and (not video_setting) and (not final_img):
            raise NoSpecifiedOutputError(
                msg="Please choice to create frames and/or video distance plots",
                source=self.__class__.__name__,
            )
        check_int(
            name=f"{self.__class__.__name__} core_cnt",
            value=core_cnt,
            min_value=-1,
            max_value=find_core_cnt()[0],
        )
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]
        ConfigReader.__init__(self, config_path=config_path)
        PlottingMixin.__init__(self)
        check_instance(
            source=f"{self.__class__.__name__} line_attr",
            instance=line_attr,
            accepted_types=(list,),
        )
        for cnt, i in enumerate(line_attr):
            check_valid_lst(
                source=f"{self.__class__.__name__} line_attr {cnt}",
                data=i,
                valid_dtypes=(str,),
                exact_len=3,
            )
        check_valid_lst(data=data_paths, valid_dtypes=(str,), min_len=1)
        _ = [check_file_exist_and_readable(i) for i in data_paths]
        (
            self.video_setting,
            self.frame_setting,
            self.data_paths,
            self.style_attr,
            self.line_attr,
            self.final_img,
            self.core_cnt,
        ) = (
            video_setting,
            frame_setting,
            data_paths,
            style_attr,
            line_attr,
            final_img,
            core_cnt,
        )
        if not os.path.exists(self.line_plot_dir):
            os.makedirs(self.line_plot_dir)
        self.color_names = get_color_dict()
        if platform.system() == "Darwin":
            multiprocessing.set_start_method("spawn", force=True)

    @staticmethod
    @jit(nopython=True)
    def __insert_group_idx_column(data: np.array, group: int):
        group_col = np.full((data.shape[0], 1), group)
        return np.hstack((group_col, data))

    def run(self):
        print(f"Processing {len(self.data_paths)} video(s)...")
        check_all_file_names_are_represented_in_video_log(
            video_info_df=self.video_info_df, data_paths=self.data_paths
        )
        for file_cnt, file_path in enumerate(self.data_paths):
            video_timer = SimbaTimer(start=True)
            _, video_name, _ = get_fn_ext(file_path)
            data_df = read_df(file_path, self.file_type)
            try:
                data_df.columns = self.bp_headers
            except ValueError:
                raise CountError(
                    msg=f"SimBA expects {self.bp_headers} columns but found {len(data_df)} columns in {file_path}",
                    source=self.__class__.__name__,
                )
            self.video_info, px_per_mm, fps = self.read_video_info(
                video_name=video_name
            )
            self.save_video_folder = os.path.join(self.line_plot_dir, video_name)
            self.temp_folder = os.path.join(self.line_plot_dir, video_name, "temp")
            self.save_frame_folder_dir = os.path.join(self.line_plot_dir, video_name)
            distances = []
            colors = []
            for cnt, i in enumerate(self.line_attr):
                if i[2] not in list(self.color_names.keys()):
                    raise InvalidInputError(
                        msg=f"{i[2]} is not a valid color. Options: {list(self.color_names.keys())}.",
                        source=self.__class__.__name__,
                    )
                colors.append(i[2])
                bp_1, bp_2 = [f"{i[0]}_x", f"{i[0]}_y"], [f"{i[1]}_x", f"{i[1]}_y"]
                if len(list(set(bp_1) - set(data_df.columns))) > 0:
                    raise InvalidInputError(
                        msg=f"Could not find fields {bp_1} in {file_path}",
                        source=self.__class__.__name__,
                    )
                if len(list(set(bp_2) - set(data_df.columns))) > 0:
                    raise InvalidInputError(
                        msg=f"Could not find fields {bp_2} in {file_path}",
                        source=self.__class__.__name__,
                    )
                distances.append(
                    FeatureExtractionMixin.framewise_euclidean_distance(
                        location_1=data_df[bp_1].values,
                        location_2=data_df[bp_2].values,
                        px_per_mm=px_per_mm,
                        centimeter=True,
                    )
                )
            if self.frame_setting:
                if os.path.exists(self.save_frame_folder_dir):
                    self.remove_a_folder(self.save_frame_folder_dir)
                os.makedirs(self.save_frame_folder_dir)
            if self.video_setting:
                self.video_folder = os.path.join(self.line_plot_dir, video_name)
                if os.path.exists(self.temp_folder):
                    self.remove_a_folder(self.temp_folder)
                os.makedirs(self.temp_folder)
                self.save_video_path = os.path.join(
                    self.line_plot_dir, f"{video_name}.mp4"
                )

            if self.final_img:
                _ = PlottingMixin.make_line_plot(
                    data=distances,
                    colors=colors,
                    width=self.style_attr["width"],
                    height=self.style_attr["height"],
                    line_width=self.style_attr["line width"],
                    font_size=self.style_attr["font size"],
                    title="Animal distances",
                    y_lbl="distance (cm)",
                    x_lbl="time (s)",
                    x_lbl_divisor=fps,
                    y_max=self.style_attr["y_max"],
                    line_opacity=self.style_attr["opacity"],
                    save_path=os.path.join(
                        self.line_plot_dir, f"{video_name}_final_distances.png"
                    ),
                )

            if self.video_setting or self.frame_setting:
                if self.style_attr["y_max"] == -1:
                    self.style_attr["y_max"] = max([np.max(x) for x in distances])
                distances = np.stack(distances, axis=1)
                frm_range = np.arange(0, distances.shape[0])
                frm_range = np.array_split(frm_range, self.core_cnt)

                distances = np.array_split(distances, self.core_cnt)
                distances = [
                    self.__insert_group_idx_column(data=i, group=cnt)
                    for cnt, i in enumerate(distances)
                ]
                distances = np.concatenate(distances, axis=0)
                print(
                    f"Creating distance plots, multiprocessing, follow progress in terminal (chunksize: {self.multiprocess_chunksize}, cores: {self.core_cnt})"
                )
                with multiprocessing.Pool(
                    self.core_cnt, maxtasksperchild=self.maxtasksperchild
                ) as pool:
                    constants = functools.partial(
                        distance_plotter_mp,
                        distances=distances,
                        video_setting=self.video_setting,
                        frame_setting=self.frame_setting,
                        video_name=video_name,
                        video_save_dir=self.temp_folder,
                        frame_folder_dir=self.save_frame_folder_dir,
                        style_attr=self.style_attr,
                        colors=colors,
                        fps=fps,
                    )
                    for cnt, result in enumerate(
                        pool.map(
                            constants, frm_range, chunksize=self.multiprocess_chunksize
                        )
                    ):
                        print(f"Frame batch core {result} complete...")
                        pass
                pool.join()
                pool.terminate()
                if self.video_setting:
                    concatenate_videos_in_folder(
                        in_folder=self.temp_folder,
                        save_path=self.save_video_path,
                        video_format="avi",
                    )
                video_timer.stop_timer()
                stdout_success(
                    msg=f"Distance visualizations created for {video_name} saved at {self.line_plot_dir}",
                    elapsed_time=video_timer.elapsed_time_str,
                )
            self.timer.stop_timer()
            stdout_success(
                msg=f"Distance visualizations complete for {len(self.data_paths)} video(s)",
                elapsed_time=self.timer.elapsed_time_str,
            )


# style_attr = {'width': 640, 'height': 480, 'line width': 6, 'font size': 12, 'y_max': -1, 'opacity': 0.5}
# line_attr = [['Center_1', 'Center_2', 'Green'], ['Ear_left_2', 'Ear_right_2', 'Red']]
# test = DistancePlotterMultiCore(config_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                 frame_setting=True,
#                                 video_setting=True,
#                                 style_attr=style_attr,
#                                 final_img=True,
#                                 data_paths=['/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/csv/outlier_corrected_movement_location/Together_1.csv'],
#                                 line_attr=line_attr,
#                                 core_cnt=-1)
# test.run()


# style_attr = {'width': 640, 'height': 480, 'line width': 6, 'font size': 8, 'y_max': 'auto', 'opacity': 0.9}
# line_attr = {0: ['Center_1', 'Center_2', 'Green'], 1: ['Ear_left_2', 'Ear_left_1', 'Red']}
#
# test = DistancePlotterMultiCore(config_path=r'/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                 frame_setting=False,
#                                 video_setting=True,
#                                 style_attr=style_attr,
#                                 final_img=True,
#                                 files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'],
#                                 line_attr=line_attr,
#                                 core_cnt=3)
# test.create_distance_plot()
# #
# style_attr = {'width': 640, 'height': 480, 'line width': 6, 'font size': 8}
# line_attr = {0: ['Termite_1_Head_1', 'Termite_1_Thorax_1', 'Dark-red']}
#


# style_attr = {'width': 640, 'height': 480, 'line width': 6, 'font size': 8, 'opacity': 0.5, 'y_max': 'auto'}
# line_attr = {0: ['Center_1', 'Center_2', 'Green'], 1: ['Ear_left_2', 'Ear_left_1', 'Red']}
#
# test = DistancePlotterMultiCore(config_path=r'/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                        frame_setting=False,
#                        video_setting=True,
#                        style_attr=style_attr,
#                                  final_img=False,
#                        files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'],
#                        line_attr=line_attr,
#                                 core_cnt=5)
# test.create_distance_plot()

# style_attr = {'width': 640, 'height': 480, 'line width': 6, 'font size': 8}
# line_attr = {0: ['Termite_1_Head_1', 'Termite_1_Thorax_1', 'Dark-red']}

# test = DistancePlotterSingleCore(config_path=r'/Users/simon/Desktop/envs/troubleshooting/Termites_5/project_folder/project_config.ini',
#                                  frame_setting=False,
#                        video_setting=True,
#                        style_attr=style_attr,
#                        files_found=['/Users/simon/Desktop/envs/troubleshooting/Termites_5/project_folder/csv/outlier_corrected_movement_location/termites_1.csv'],
#                        line_attr=line_attr)
# test.create_distance_plot()
