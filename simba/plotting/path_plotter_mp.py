__author__ = "Simon Nilsson"

import functools
import multiprocessing
import os
import platform
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_keys_exist_in_dict,
                                check_if_valid_rgb_str, check_instance,
                                check_int, check_that_column_exist,
                                check_valid_lst)
from simba.utils.data import find_frame_numbers_from_time_stamp
from simba.utils.enums import Formats, TagNames
from simba.utils.errors import FrameRangeError, NoSpecifiedOutputError
from simba.utils.printing import SimbaTimer, log_event, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    find_core_cnt, find_video_of_file,
                                    get_fn_ext, get_video_meta_data, read_df,
                                    read_frm_of_video, remove_a_folder)


def path_plot_mp(
    frm_rng: np.ndarray,
    data: np.array,
    colors: List[Tuple],
    video_setting: bool,
    frame_setting: bool,
    video_save_dir: str,
    video_name: str,
    frame_folder_dir: str,
    style_attr: dict,
    animal_names: Union[None, List[str]],
    fps: int,
    clf_attr: dict,
    input_style_attr: dict,
    video_path: Optional[Union[str, os.PathLike]] = None,
):

    batch_id, frm_rng = frm_rng[0], frm_rng[1]
    if video_setting:
        fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        video_save_path = os.path.join(video_save_dir, f"{batch_id}.mp4")
        video_writer = cv2.VideoWriter(
            video_save_path, fourcc, fps, (style_attr["width"], style_attr["height"])
        )

    if input_style_attr is not None:
        if (isinstance(input_style_attr["bg color"], dict)) and (
            input_style_attr["bg color"]["type"]
        ) == "moving":
            check_file_exist_and_readable(file_path=video_path)
            video_cap = cv2.VideoCapture(video_path)

    bg_clr = style_attr["bg color"]
    for frame_id in frm_rng:
        if (isinstance(style_attr["bg color"], dict)) and (
            style_attr["bg color"]["type"]
        ) == "moving":
            bg_clr = read_frm_of_video(
                video_path=video_cap,
                opacity=style_attr["bg color"]["opacity"],
                frame_index=frame_id,
            )
        plot_arrs = [x[:frame_id, :] for x in data]

        clf_attr_cpy = deepcopy(clf_attr)
        if clf_attr is not None:
            for k, v in clf_attr.items():
                clf_attr_cpy[k]["clfs"][frame_id + 1 :] = 0

        img = PlottingMixin.make_path_plot(
            data=plot_arrs,
            colors=colors,
            width=style_attr["width"],
            height=style_attr["height"],
            max_lines=style_attr["max lines"],
            bg_clr=bg_clr,
            circle_size=style_attr["circle size"],
            font_size=style_attr["font size"],
            font_thickness=style_attr["font thickness"],
            line_width=style_attr["line width"],
            animal_names=animal_names,
            clf_attr=clf_attr_cpy,
            save_path=None,
        )

        if video_setting:
            video_writer.write(np.uint8(img))
        if frame_setting:
            frm_name = os.path.join(frame_folder_dir, f"{frame_id}.png")
            cv2.imwrite(frm_name, np.uint8(img))
        print(
            f"Path frame created: {frame_id}, Video: {video_name}, Processing core: {batch_id}"
        )
    if video_setting:
        video_writer.release()
    return batch_id


class PathPlotterMulticore(ConfigReader, PlottingMixin):
    """
    Class for creating "path plots" videos and/or images detailing the movement paths of
    individual animals in SimBA. Uses multiprocessing.

    :param str config_path: Path to SimBA project config file in Configparser format
    :param bool frame_setting: If True, individual frames will be created.
    :param bool video_setting: If True, compressed videos will be created.
    :param bool last_frame: If True, png of the last frame will be created.
    :param List[str] files_found: Data paths to create path plots for (e.g., ['project_folder/csv/machine_results/MyVideo.csv'])
    :param dict animal_attr: Animal body-parts to use when creating paths and their colors.
    :param Optional[dict] input_style_attr: Plot sttributes (line thickness, color, etc..). If None, then autocomputed. Max lines will be set to 2s.
    :param Optional[dict] input_clf_attr: Dict reprenting classified behavior locations, their color and size. If None, then no classified behavior locations are shown.
    :param Optional[dict] slicing: If Dict, start time and end time of video slice to create path plot from. E.g., {'start_time': '00:00:01', 'end_time': '00:00:03'}. If None, creates path plot for entire video.
    :param int cores: Number of cores to use.

    .. note::
       `Visualization tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-11-visualizations>`__.

    .. image:: _static/img/path_plot.png
       :width: 300
       :align: center

    .. image:: _static/img/path_plot_mp.gif
       :width: 500
       :align: center

    .. image:: _static/img/path_plot_1.png
       :width: 500
       :align: center

    :example:
    >>> input_style_attr = {'width': 'As input', 'height': 'As input', 'line width': 5, 'font size': 5, 'font thickness': 2, 'circle size': 5, 'bg color': 'White', 'max lines': 100}
    >>> animal_attr = {0: ['Ear_right_1', 'Red']}
    >>> input_clf_attr = {0: ['Attack', 'Black', 'Size: 30'], 1: ['Sniffing', 'Red', 'Size: 30']}
    >>> path_plotter = PathPlotterMulticore(config_path=r'MyConfigPath', frame_setting=False, video_setting=True, style_attr=style_attr, animal_attr=animal_attr, files_found=['project_folder/csv/machine_results/MyVideo.csv'], cores=5, clf_attr=clf_attr, print_animal_names=True)
    >>> path_plotter.run()
    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        files_found: List[str],
        frame_setting: Optional[bool] = False,
        video_setting: Optional[bool] = False,
        last_frame: Optional[bool] = True,
        cores: Optional[int] = -1,
        print_animal_names: Optional[bool] = False,
        input_style_attr: Optional[Dict] = None,
        animal_attr: Dict[int, Any] = None,
        clf_attr: Optional[Dict[int, List[str]]] = None,
        slicing: Optional[Dict[str, str]] = None,
    ):

        if platform.system() == "Darwin":
            multiprocessing.set_start_method("spawn", force=True)
        if (not frame_setting) and (not video_setting) and (not last_frame):
            raise NoSpecifiedOutputError(
                msg="SIMBA ERROR: Please choice to create path frames and/or video path plots",
                source=self.__class__.__name__,
            )
        check_valid_lst(
            data=files_found, source=self.__class__.__name__, valid_dtypes=(str,)
        )

        check_int(
            name=f"{self.__class__.__name__} core_cnt",
            value=cores,
            min_value=-1,
            max_value=find_core_cnt()[0],
        )
        if cores == -1:
            cores = find_core_cnt()[0]

        ConfigReader.__init__(self, config_path=config_path)
        PlottingMixin.__init__(self)
        log_event(
            logger_name=str(__class__.__name__),
            log_type=TagNames.CLASS_INIT.value,
            msg=self.create_log_msg_from_init_args(locals=locals()),
        )
        (
            self.video_setting,
            self.frame_setting,
            self.input_style_attr,
            self.files_found,
            self.animal_attr,
            self.input_clf_attr,
            self.last_frame,
            self.cores,
        ) = (
            video_setting,
            frame_setting,
            input_style_attr,
            files_found,
            animal_attr,
            clf_attr,
            last_frame,
            cores,
        )
        self.print_animal_names, self.clf_attr, self.slicing = (
            print_animal_names,
            clf_attr,
            slicing,
        )
        if not os.path.exists(self.path_plot_dir):
            os.makedirs(self.path_plot_dir)
        print(f"Processing {len(self.files_found)} videos...")

    def __get_styles(self):
        self.style_attr = {}
        if self.input_style_attr is not None:
            self.style_attr["bg color"] = self.input_style_attr["bg color"]
            if self.input_style_attr["max lines"] == "entire video":
                self.style_attr["max lines"] = len(self.data_df)
            else:
                self.style_attr["max lines"] = max(
                    1,
                    int(
                        int(self.input_style_attr["max lines"] / 1000)
                        * (int(self.video_info["fps"].values[0]))
                    ),
                )
            self.style_attr["font thickness"] = self.input_style_attr["font thickness"]
            self.style_attr["line width"] = self.input_style_attr["line width"]
            self.style_attr["font size"] = self.input_style_attr["font size"]
            self.style_attr["circle size"] = self.input_style_attr["circle size"]
            self.style_attr["print_animal_names"] = self.print_animal_names
            if self.input_style_attr["width"] == "As input":
                self.style_attr["width"], self.style_attr["height"] = int(
                    self.video_info["Resolution_width"].values[0]
                ), int(self.video_info["Resolution_height"].values[0])
            else:
                pass
        else:
            space_scaler, radius_scaler, res_scaler, font_scaler = 25, 10, 1500, 0.8
            self.style_attr["width"] = int(
                self.video_info["Resolution_width"].values[0]
            )
            self.style_attr["height"] = int(
                self.video_info["Resolution_height"].values[0]
            )
            max_res = max(self.style_attr["width"], self.style_attr["height"])
            self.style_attr["circle size"] = int(radius_scaler / (res_scaler / max_res))
            self.style_attr["font size"] = font_scaler / (res_scaler / max_res)
            self.style_attr["bg color"] = self.color_dict["White"]
            self.style_attr["print_animal_names"] = self.print_animal_names
            self.style_attr["max lines"] = len(self.data_df)
            self.style_attr["font thickness"] = 2
            self.style_attr["line width"] = 2

    def run(self):
        for file_cnt, file_path in enumerate(self.files_found):
            video_timer = SimbaTimer(start=True)
            _, self.video_name, _ = get_fn_ext(file_path)
            self.video_info, _, self.fps = self.read_video_info(
                video_name=self.video_name
            )
            self.data_df = read_df(file_path, self.file_type)
            line_data, colors, animal_names = [], [], []
            for k, v in self.animal_attr.items():
                check_if_keys_exist_in_dict(
                    data=v, key=["bp", "color"], name=f"animal attr {k}"
                )
                line_data.append(
                    self.data_df[[f'{v["bp"]}_x', f'{v["bp"]}_y']].values.astype(
                        np.int64
                    )
                )
                colors.append(v["color"])
                if self.print_animal_names:
                    animal_names.append(
                        self.find_animal_name_from_body_part_name(
                            bp_name=v["bp"], bp_dict=self.animal_bp_dict
                        )
                    )
            if not self.print_animal_names:
                animal_names = None
            if self.slicing:
                check_if_keys_exist_in_dict(
                    data=self.slicing, key=["start_time", "end_time"], name="slicing"
                )
                frm_numbers = find_frame_numbers_from_time_stamp(
                    start_time=self.slicing["start_time"],
                    end_time=self.slicing["end_time"],
                    fps=self.fps,
                )
                if len(set(frm_numbers) - set(self.data_df.index)) > 0:
                    raise FrameRangeError(
                        msg=f'The chosen time-period ({self.slicing["start_time"]} - {self.slicing["end_time"]}) does not exist in {self.video_name}.',
                        source=self.__class__.__name__,
                    )
                for i in range(len(line_data)):
                    line_data[i] = line_data[i][frm_numbers, :]
            self.__get_styles()

            self.temp_folder = os.path.join(self.path_plot_dir, self.video_name, "temp")
            self.save_frame_folder_dir = os.path.join(
                self.path_plot_dir, self.video_name
            )

            if self.frame_setting:
                if os.path.exists(self.save_frame_folder_dir):
                    remove_a_folder(self.save_frame_folder_dir)
                os.makedirs(self.save_frame_folder_dir)

            if self.video_setting:
                self.video_folder = os.path.join(self.path_plot_dir, self.video_name)
                if os.path.exists(self.temp_folder):
                    remove_a_folder(self.temp_folder)
                    remove_a_folder(self.video_folder)
                os.makedirs(self.temp_folder)
                self.save_video_path = os.path.join(
                    self.path_plot_dir, f"{self.video_name}.mp4"
                )

            if self.clf_attr is not None:
                self.clf_attr_appended = {}
                check_instance(
                    source=self.__class__.__name__,
                    instance=self.clf_attr,
                    accepted_types=(dict,),
                )
                for k, v in self.clf_attr.items():
                    check_if_keys_exist_in_dict(
                        data=v, key=["color", "size"], name=f"clf_attr {k}"
                    )
                    check_that_column_exist(
                        df=self.data_df, column_name=k, file_name=file_path
                    )
                    self.clf_attr_appended[k] = self.clf_attr[k]
                    self.clf_attr_appended[k]["clfs"] = self.data_df[k].values.astype(
                        np.int8
                    )
                    self.clf_attr_appended[k]["positions"] = self.data_df[
                        [
                            self.animal_attr[0]["bp"] + "_x",
                            self.animal_attr[0]["bp"] + "_y",
                        ]
                    ].values.astype(np.int64)
                self.clf_attr = deepcopy(self.clf_attr_appended)
                del self.clf_attr_appended

            bg_clr = self.style_attr["bg color"]
            self.video_path = None
            if isinstance(self.style_attr["bg color"], dict):
                self.video_path = find_video_of_file(
                    video_dir=self.video_dir, filename=self.video_name, raise_error=True
                )
                if "frame_index" in self.style_attr["bg color"].keys():
                    check_int(
                        name="Static frame index",
                        value=self.style_attr["bg color"]["frame_index"],
                        min_value=0,
                    )
                    frame_index = self.style_attr["bg color"]["frame_index"]
                else:
                    video_meta_data = get_video_meta_data(video_path=self.video_path)
                    frame_index = video_meta_data["frame_count"] - 1
                bg_clr = read_frm_of_video(
                    video_path=self.video_path,
                    opacity=self.style_attr["bg color"]["opacity"],
                    frame_index=frame_index,
                )

            if self.last_frame:
                PlottingMixin.make_path_plot(
                    data=line_data,
                    colors=colors,
                    width=self.style_attr["width"],
                    height=self.style_attr["height"],
                    max_lines=self.style_attr["max lines"],
                    bg_clr=bg_clr,
                    circle_size=self.style_attr["circle size"],
                    font_size=self.style_attr["font size"],
                    font_thickness=self.style_attr["font thickness"],
                    line_width=self.style_attr["line width"],
                    animal_names=animal_names,
                    clf_attr=self.clf_attr,
                    save_path=os.path.join(
                        self.path_plot_dir, f"{self.video_name}_final_frame.png"
                    ),
                )

            if self.video_setting or self.frame_setting:
                frm_range = np.arange(1, line_data[0].shape[0])
                frm_range = np.array_split(frm_range, self.cores)
                frm_range = [(cnt, x) for cnt, x in enumerate(frm_range)]
                print(
                    f"Creating path plots, multiprocessing (chunksize: {self.multiprocess_chunksize}, cores: {self.cores})..."
                )
                with multiprocessing.Pool(
                    self.cores, maxtasksperchild=self.maxtasksperchild
                ) as pool:
                    constants = functools.partial(
                        path_plot_mp,
                        data=line_data,
                        colors=colors,
                        video_setting=self.video_setting,
                        video_name=self.video_name,
                        frame_setting=self.frame_setting,
                        video_save_dir=self.temp_folder,
                        frame_folder_dir=self.save_frame_folder_dir,
                        style_attr=self.style_attr,
                        animal_names=animal_names,
                        fps=self.fps,
                        clf_attr=self.clf_attr,
                        input_style_attr=self.input_style_attr,
                        video_path=self.video_path,
                    )
                    for cnt, result in enumerate(
                        pool.imap(
                            constants, frm_range, chunksize=self.multiprocess_chunksize
                        )
                    ):
                        print(f"Path batch {result+1}/{self.cores} complete...")
                    pool.terminate()
                    pool.join()

                if self.video_setting:
                    print(f"Joining {self.video_name} multiprocessed video...")
                    concatenate_videos_in_folder(
                        in_folder=self.temp_folder, save_path=self.save_video_path
                    )
                video_timer.stop_timer()
                print(
                    f"Path plot video {self.video_name} complete (elapsed time: {video_timer.elapsed_time_str}s) ..."
                )

        self.timer.stop_timer()
        stdout_success(
            msg=f"Path plot visualizations for {len(self.files_found)} videos created in {self.path_plot_dir} directory",
            elapsed_time=self.timer.elapsed_time_str,
            source=self.__class__.__name__,
        )


# animal_attr = {0: {'bp': 'Ear_right_1', 'color': (255, 0, 0)}, 1: {'bp': 'Ear_right_2', 'color': (0, 0, 255)}}  #['Ear_right_1', 'Red'], 1: ['Ear_right_2', 'Green']}
# style_attr = {'width': 'As input',
#               'height': 'As input',
#               'line width': 2,
#               'font size': 0.9,
#               'font thickness': 2,
#               'circle size': 5,
#               'bg color': {'type': 'moving', 'opacity': 50, 'frame_index': 200}, #{'type': 'static', 'opacity': 100, 'frame_index': 200}
#               'max lines': 'entire video'}
# clf_attr = {'Nose to Nose': {'color': (155, 1, 10), 'size': 30}, 'Nose to Tailbase': {'color': (155, 90, 10), 'size': 30}}
# #clf_attr=None

# path_plotter = PathPlotterMulticore(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/project_config.ini',
#                                     frame_setting=False,
#                                     video_setting=True,
#                                     last_frame=True,
#                                     clf_attr=clf_attr,
#                                     input_style_attr=style_attr,
#                                     animal_attr=animal_attr,
#                                     files_found=['/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/csv/machine_results/Trial    10.csv'],
#                                     cores=-1,
#                                     slicing = {'start_time': '00:00:00', 'end_time': '00:00:05'}, # {'start_time': '00:00:00', 'end_time': '00:00:05'}, # , #None,
#                                     print_animal_names=False)
# path_plotter.run()


# style_attr = {'width': 'As input',
#               'height': 'As input',
#               'line width': 5, 'font size': 5, 'font thickness': 2, 'circle size': 5, 'bg color': {'type': 'static', 'opacity': 100, 'frame_index': 100}, 'max lines': 'entire video'}
# animal_attr = {0: ['Ear_right_1', 'Red'], 1: ['Ear_right_2', 'Green']}
# # clf_attr = {0: ['Attack', 'Black', 'Size: 30']}
# # # #
# clf_attr = None
# style_attr = None
#
# path_plotter = PathPlotterMulticore(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                     frame_setting=False,
#                                     video_setting=True,
#                                     last_frame=True,
#                                     input_clf_attr=clf_attr,
#                                     input_style_attr=style_attr,
#                                     animal_attr=animal_attr,
#                                     files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'],
#                                     cores=5,
#                                     slicing = None)
# path_plotter.run()

#


# style_attr = None
# style_attr = {'width': 'As input',
#               'height': 'As input',
#               'line width': 5, 'font size': 5, 'font thickness': 2, 'circle size': 5, 'bg color': {'opacity': 100}, 'max lines': 10000}
# animal_attr = {0: ['mouth', 'Red']}
# clf_attr = {0: ['Freezing', 'Black', 'Size: 10'], 1: ['Normal Swimming', 'Red', 'Size: 10']}
# #
# # clf_attr = None
#
#
# path_plotter = PathPlotterMulticore(config_path='/Users/simon/Desktop/envs/troubleshooting/naresh/project_folder/project_config.ini',
#                                     frame_setting=False,
#                                     video_setting=True,
#                                     last_frame=True,
#                                     input_clf_attr=clf_attr,
#                                     input_style_attr=style_attr,
#                                     animal_attr=animal_attr,
#                                     files_found=['/Users/simon/Desktop/envs/troubleshooting/naresh/project_folder/csv/machine_results/SF8.csv'],
#                                     cores=5)
# path_plotter.run()
