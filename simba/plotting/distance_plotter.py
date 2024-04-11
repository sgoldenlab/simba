__author__ = "Simon Nilsson"

import os
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_file_exist_and_readable, check_instance, check_valid_lst)
from simba.utils.errors import (CountError, InvalidInputError,
                                NoSpecifiedOutputError)
from simba.utils.lookups import get_color_dict
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, read_df


class DistancePlotterSingleCore(ConfigReader):
    """
    Class for visualizing the distance between two pose-estimated body-parts (e.g., two animals) through line
    charts. Results are saved as individual line charts, and/or videos of line charts.

    .. note::
       For better runtime, use :meth:`simba.plotting.distance_plotter_mp.DistancePlotterMultiCore`.
       `GitHub tutorial/documentation <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-11-visualizations>`__.

    .. image:: _static/img/distance_plot.png
       :width: 300
       :align: center

    :parameter str config_path: path to SimBA project config file in Configparser format.
    :parameter bool frame_setting: If True, creates individual frames.
    :parameter bool video_setting: If True, creates videos

    :examples:

    >>> style_attr = {'width': 640, 'height': 480, 'line width': 6, 'font size': 8, 'opacity': 0.5}
    >>> line_attr = {0: ['Center_1', 'Center_2', 'Green'], 1: ['Ear_left_2', 'Ear_left_1', 'Red']}
    >>> distance_plotter = DistancePlotterSingleCore(config_path=r'MyProjectConfig', files_found=['test/two_c57s/project_folder/csv/outlier_corrected_movement_location/Video_1.csv'], frame_setting=False, video_setting=True, final_img=True)
    >>> distance_plotter.run()
    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        data_paths: List[Union[str, os.PathLike]],
        style_attr: Dict[str, int],
        line_attr: List[List[str]],
        frame_setting: Optional[bool] = False,
        video_setting: Optional[bool] = False,
        final_img: Optional[bool] = False,
    ):

        if (not frame_setting) and (not video_setting) and (not final_img):
            raise NoSpecifiedOutputError(
                msg="Please choice to create frames and/or video distance plots",
                source=self.__class__.__name__,
            )
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
        ConfigReader.__init__(self, config_path=config_path)
        (
            self.video_setting,
            self.frame_setting,
            self.data_paths,
            self.style_attr,
            self.line_attr,
            self.final_img,
        ) = (video_setting, frame_setting, data_paths, style_attr, line_attr, final_img)
        self.color_names = get_color_dict()

    def run(self):
        print(f"Processing {len(self.data_paths)} videos...")
        check_all_file_names_are_represented_in_video_log(
            video_info_df=self.video_info_df, data_paths=self.data_paths
        )
        for file_cnt, file_path in enumerate(self.data_paths):
            video_timer = SimbaTimer(start=True)
            data_df = read_df(file_path, self.file_type)
            _, video_name, _ = get_fn_ext(file_path)
            self.video_info, px_per_mm, fps = self.read_video_info(
                video_name=video_name
            )
            self.save_video_folder = os.path.join(self.line_plot_dir, video_name)
            self.save_frame_folder_dir = os.path.join(self.line_plot_dir, video_name)

            try:
                data_df.columns = self.bp_headers
            except ValueError:
                raise CountError(
                    msg=f"SimBA expects {self.bp_headers} columns but found {len(data_df)} columns in {file_path}",
                    source=self.__class__.__name__,
                )
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
                save_video_path = os.path.join(self.line_plot_dir, f"{video_name}.avi")
                fourcc = cv2.VideoWriter_fourcc(*"DIVX")
                video_writer = cv2.VideoWriter(
                    save_video_path,
                    fourcc,
                    fps,
                    (self.style_attr["width"], self.style_attr["height"]),
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
                for frm_cnt in range(distances[0].shape[0]):
                    line_data = [x[:frm_cnt] for x in distances]
                    img = PlottingMixin.make_line_plot_plotly(
                        data=line_data,
                        colors=colors,
                        width=self.style_attr["width"],
                        height=self.style_attr["height"],
                        line_width=self.style_attr["line width"],
                        font_size=self.style_attr["font size"],
                        title="Animal distances",
                        y_lbl="distance (cm)",
                        x_lbl="frame count",
                        x_lbl_divisor=fps,
                        y_max=self.style_attr["y_max"],
                        line_opacity=self.style_attr["opacity"],
                        save_path=None,
                    ).astype(np.uint8)
                    if self.video_setting:
                        video_writer.write(img[:, :, :3])
                    if self.frame_setting:
                        frm_name = os.path.join(
                            self.save_frame_folder_dir, f"{frm_cnt}.png"
                        )
                        cv2.imwrite(frm_name, np.uint8(img))
                    print(f"Distance frame created: {frm_cnt}, Video: {video_name} ...")
                if self.video_setting:
                    video_writer.release()
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
# test = DistancePlotterSingleCore(config_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                  frame_setting=True,
#                                  video_setting=True,
#                                  style_attr=style_attr,
#                                  final_img=True,
#                                  data_paths=['/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/csv/outlier_corrected_movement_location/Together_1.csv'],
#                                  line_attr=line_attr)
# test.run()


#
# style_attr = {'width': 640,
#               'height': 480,
#               'line width': 6,
#               'font size': 8,
#               'y_max': 'auto',
#               'opacity': 0.9}
# line_attr = {0: ['Center_1', 'Center_2', 'Green'], 1: ['Ear_left_2', 'Ear_left_1', 'Red']}
#
# test = DistancePlotterSingleCore(config_path=r'/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                  frame_setting=False,
#                                  video_setting=True,
#                                  style_attr=style_attr,
#                                  final_img=True,
#                                  files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'],
#                                  line_attr=line_attr)
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
