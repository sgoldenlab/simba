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
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_file_exist_and_readable, check_if_keys_exist_in_dict,
    check_if_string_value_is_valid_video_timestamp, check_if_valid_rgb_tuple,
    check_instance, check_str, check_that_column_exist,
    check_that_hhmmss_start_is_before_end, check_valid_boolean,
    check_valid_dataframe, check_valid_lst,
    check_video_and_data_frm_count_align)
from simba.utils.data import (find_frame_numbers_from_time_stamp,
                              slice_roi_dict_for_video)
from simba.utils.enums import Formats, TagNames
from simba.utils.errors import (FrameRangeError, InvalidVideoFileError,
                                NoSpecifiedOutputError)
from simba.utils.printing import SimbaTimer, log_event, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    create_directory, find_core_cnt,
                                    find_video_of_file, get_fn_ext,
                                    get_video_meta_data, read_df,
                                    read_frm_of_video, remove_a_folder)
from simba.utils.warnings import ROIWarning

STYLE_WIDTH = "width"
STYLE_HEIGHT = "height"
STYLE_LINE_WIDTH = "line width"
STYLE_FONT_SIZE = "font size"
STYLE_FONT_THICKNESS = "font thickness"
STYLE_CIRCLE_SIZE = "circle size"
STYLE_MAX_LINES = "max lines"
STYLE_BG = 'bg'
STYLE_BG_OPACITY = 'bg_opacity'
COLOR = 'color'
SIZE = 'size'
START_TIME = 'start_time'
END_TIME = 'end_time'
BODY_PART = 'body_part'
AUTO = 'AUTO'
ANIMAL_NAME = 'animal_name'

STYLE_KEYS = [
    STYLE_WIDTH,
    STYLE_HEIGHT,
    STYLE_LINE_WIDTH,
    STYLE_FONT_SIZE,
    STYLE_FONT_THICKNESS,
    STYLE_CIRCLE_SIZE,
    STYLE_MAX_LINES,
    STYLE_BG,
    STYLE_BG_OPACITY
]

def path_plot_mp(data: np.ndarray,
                 line_data: np.array,
                 colors: list,
                 video_setting: bool,
                 frame_setting: bool,
                 video_save_dir: str,
                 video_name: str,
                 frame_folder_dir: str,
                 style_attr: dict,
                 roi: Union[dict, None],
                 animal_names: Union[None, List[str]],
                 fps: Union[int, float],
                 clf_attr: dict):

    batch_id, frm_cnt_rng, frm_id_rng = data[0], data[1], data[2]
    if video_setting:
        video_save_path = os.path.join(video_save_dir, f"{batch_id}.mp4")
        FOURCC = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        video_writer = cv2.VideoWriter(video_save_path, FOURCC, int(fps), (style_attr[STYLE_WIDTH], style_attr[STYLE_HEIGHT]))
    for frm_cnt, frm_id in zip(frm_cnt_rng, frm_id_rng):
        if isinstance(style_attr[STYLE_BG], str):
            bg = read_frm_of_video(video_path=style_attr[STYLE_BG], opacity=style_attr[STYLE_BG_OPACITY], frame_index=frm_id)
        else:
            bg = deepcopy(style_attr[STYLE_BG])
        plot_arrs = [x[:frm_cnt, :] for x in line_data]
        clf_attr_cpy = deepcopy(clf_attr)
        if clf_attr is not None:
            for k, v in clf_attr.items(): clf_attr_cpy[k]["clfs"][frm_id + 1 :] = 0

        img = PlottingMixin.make_path_plot(data=plot_arrs,
                                           colors=colors,
                                           width=style_attr[STYLE_WIDTH],
                                           height=style_attr[STYLE_HEIGHT],
                                           max_lines=style_attr[STYLE_MAX_LINES],
                                           bg_clr=bg,
                                           circle_size=style_attr[STYLE_CIRCLE_SIZE],
                                           font_size=style_attr[STYLE_FONT_SIZE],
                                           font_thickness=style_attr[STYLE_FONT_THICKNESS],
                                           line_width=style_attr[STYLE_LINE_WIDTH],
                                           animal_names=animal_names,
                                           clf_attr=clf_attr_cpy,
                                           save_path=None)
        if roi is not None:
            img = PlottingMixin.roi_dict_onto_img(img=img, roi_dict=roi, show_tags=False, show_center=False)

        if video_setting:
            video_writer.write(img)
        if frame_setting:
            frm_name = os.path.join(frame_folder_dir, f"{frm_id}.png")
            cv2.imwrite(frm_name, np.uint8(img))
        print(f"Path frame created: {frm_id}, Video: {video_name}, Processing core: {batch_id}")
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
    :param Optional[bool] roi: If True, also plots the ROIs associated with the video. Default False.
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

    References
    ----------
    .. [1] Battivelli, Dorian, Lucas Boldrini, Mohit Jaiswal, Pradnya Patil, Sofia Torchia, Elizabeth Engelen, Luca Spagnoletti, Sarah Kaspar, and Cornelius T. Gross. “Induction of Territorial Dominance and Subordination Behaviors in Laboratory Mice.” Scientific Reports 14, no. 1 (November 19, 2024): 28655. https://doi.org/10.1038/s41598-024-75545-4.
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_paths: List[Union[str, os.PathLike]],
                 animal_attr: dict,
                 style_attr: Optional[Union[Dict[str, Any], None]] = None,
                 clf_attr: Optional[dict] = None,
                 frame_setting: bool = False,
                 video_setting: bool = False,
                 last_frame: bool = False,
                 print_animal_names: bool = True,
                 slicing: Optional[Dict] = None,
                 core_cnt: int = -1,
                 roi: bool = False):

        log_event(logger_name=str(__class__.__name__), log_type=TagNames.CLASS_INIT.value, msg=self.create_log_msg_from_init_args(locals=locals()))
        if (not frame_setting) and (not video_setting) and (not last_frame):
            raise NoSpecifiedOutputError(msg="SIMBA ERROR: Please choice to create path frames and/or video path plots", source=self.__class__.__name__)
        check_valid_lst(data=data_paths, source=self.__class__.__name__, valid_dtypes=(str,), min_len=1)
        _ = [check_file_exist_and_readable(x) for x in data_paths]
        check_file_exist_and_readable(file_path=config_path)
        if style_attr is not None:
            check_if_keys_exist_in_dict(data=style_attr, key=STYLE_KEYS, name=f'{self.__class__.__name__} style_attr')
        check_valid_boolean(value=[frame_setting, video_setting, last_frame,print_animal_names, roi], source=self.__class__.__name__)
        if slicing is not None:
            check_if_keys_exist_in_dict(data=slicing, key=[START_TIME, END_TIME], name=f'{self.__class__.__name__} slicing')
            check_if_string_value_is_valid_video_timestamp(value=slicing[START_TIME], name="Video slicing START TIME")
            check_if_string_value_is_valid_video_timestamp(value=slicing[END_TIME], name="Video slicing END TIME")
            check_that_hhmmss_start_is_before_end(start_time=slicing[START_TIME], end_time=slicing[END_TIME], name="SLICE TIME STAMPS")
        ConfigReader.__init__(self, config_path=config_path)
        PlottingMixin.__init__(self)
        if roi:
            check_file_exist_and_readable(file_path=self.roi_coordinates_path)
            self.read_roi_data()
        if not video_setting and not frame_setting and not last_frame:
            raise NoSpecifiedOutputError(msg='Video, last frame, and frame are all False', source=self.__class__.__name__)
        self.clf_names = None
        if clf_attr is not None:
            check_instance(source=f'{self.__class__.__name__} clf_attr', instance=clf_attr, accepted_types=(dict,))
            for k, v in clf_attr.items():
                check_if_keys_exist_in_dict(data=v, key=[COLOR, SIZE], name=f'{self.__class__.__name__} clf_attr')
            self.clf_names = list(clf_attr.keys())
        self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        if not os.path.exists(self.path_plot_dir): os.makedirs(self.path_plot_dir)
        self.video_setting, self.frame_setting, self.style_attr, self.data_paths, self.animal_attr, self.clf_attr, self.last_frame, self.roi = video_setting, frame_setting, style_attr, data_paths, animal_attr, clf_attr, last_frame, roi
        self.print_animal_names, self.slicing = print_animal_names, slicing
        self.bp_data, self.data_cols, self.animal_names, self.colors = self.__get_bp_data()
        self.animal_names = None if not self.print_animal_names else self.animal_names
        self.core_cnt = find_core_cnt()[0] if core_cnt == -1 or core_cnt > find_core_cnt()[0] else core_cnt

    def __get_bp_data(self):
        bp_data, data_cols, animal_names, colors = {}, [], [], [],
        for cnt, (k, v) in enumerate(self.animal_attr.items()):
            check_if_keys_exist_in_dict(data=v, key=[COLOR, BODY_PART], name=f'{self.__class__.__name__} animal_attr')
            check_str(name=f'animal {k} body_part', value=v[BODY_PART], options=self.body_parts_lst)
            check_if_valid_rgb_tuple(data=v[COLOR], raise_error=True)
            name = self.find_animal_name_from_body_part_name(bp_name=v[BODY_PART], bp_dict=self.animal_bp_dict)
            bp_data[cnt] = {ANIMAL_NAME: name, 'x': f'{v[BODY_PART]}_x', 'y': f'{v[BODY_PART]}_y', 'p': f'{v[BODY_PART]}_p', COLOR: v[COLOR]}
            data_cols.extend([f'{v[BODY_PART]}_x', f'{v[BODY_PART]}_y'])
            animal_names.append(name)
            colors.append(v[COLOR])
        return bp_data, data_cols, animal_names, colors


    def __get_styles(self, style_attr):
        video_styles = {}
        optimal_font_size, _, _ = PlottingMixin().get_optimal_font_scales(text='MY LONG TEXT STRING', accepted_px_width=int(self.video_info["Resolution_width"] / 10), accepted_px_height=int(self.video_info["Resolution_width"] / 10))
        optimal_circle_size = PlottingMixin().get_optimal_circle_size(frame_size=(int(self.video_info["Resolution_width"]), int(self.video_info["Resolution_height"])), circle_frame_ratio=100)
        video_styles[STYLE_WIDTH] = int(self.video_info["Resolution_width"].values[0]) if style_attr[STYLE_WIDTH] == None else int(style_attr[STYLE_WIDTH])
        video_styles[STYLE_HEIGHT] = int(self.video_info["Resolution_height"].values[0]) if style_attr[STYLE_HEIGHT] == None else int(style_attr[STYLE_HEIGHT])
        video_styles[STYLE_CIRCLE_SIZE] = optimal_circle_size if style_attr[STYLE_CIRCLE_SIZE] == None else int(style_attr[STYLE_CIRCLE_SIZE])
        video_styles[STYLE_LINE_WIDTH] = 2 if style_attr[STYLE_LINE_WIDTH] == None else int(style_attr[STYLE_LINE_WIDTH])
        video_styles[STYLE_FONT_THICKNESS] = 2 if style_attr[STYLE_FONT_THICKNESS] == None else int(style_attr[STYLE_FONT_THICKNESS])
        video_styles[STYLE_FONT_SIZE] = optimal_font_size if style_attr[STYLE_FONT_SIZE] == None else int(style_attr[STYLE_FONT_SIZE])
        video_styles[STYLE_MAX_LINES] = len(self.data_df) if style_attr[STYLE_MAX_LINES] == None else int(self.fps * int(style_attr[STYLE_MAX_LINES]))
        video_styles[STYLE_BG_OPACITY] = None if style_attr[STYLE_BG_OPACITY] == None else float(style_attr[STYLE_BG_OPACITY])
        if isinstance(self.style_attr[STYLE_BG], tuple):
            video_styles[STYLE_BG] = self.style_attr[STYLE_BG]
        if isinstance(self.style_attr[STYLE_BG], (str, int)):
            video_path = find_video_of_file(video_dir=self.video_dir, filename=self.video_name, raise_error=False, warning=True)
            if video_path is None:
                raise InvalidVideoFileError(msg=f'Could not find video in expected location. If using frame or video as background, make sure the data file {self.video_name} is represented in the {self.video_dir} directory', source=self.__class__.__name__)
            if isinstance(self.style_attr[STYLE_BG], (int,)):
                video_styles[STYLE_BG] = read_frm_of_video(video_path=video_path, frame_index=self.style_attr[STYLE_BG], opacity=video_styles[STYLE_BG_OPACITY])
            else:
                if self.slicing is None:
                    check_video_and_data_frm_count_align(video=video_path, data=self.data_df, name=self.video_name, raise_error=False)
                video_styles[STYLE_BG] = video_path
        return video_styles


    def run(self):
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        print(f"Processing path plots for {len(self.data_paths)} video(s)...")
        for file_cnt, file_path in enumerate(self.data_paths):
            video_timer = SimbaTimer(start=True)
            _, self.video_name, _ = get_fn_ext(file_path)
            self.video_info, _, self.fps = self.read_video_info(video_name=self.video_name)
            self.in_df = read_df(file_path, self.file_type)
            check_valid_dataframe(df=self.in_df, source=file_path, valid_dtypes=Formats.NUMERIC_DTYPES.value, required_fields=self.data_cols)
            if self.clf_attr is not None:
                check_valid_dataframe(df=self.in_df, source=file_path, valid_dtypes=Formats.NUMERIC_DTYPES.value, required_fields=self.clf_names)
            self.save_frm_dir = None
            if self.slicing:
                frm_numbers = find_frame_numbers_from_time_stamp(start_time=self.slicing[START_TIME], end_time=self.slicing[END_TIME], fps=self.fps)
                if len(set(frm_numbers) - set(self.in_df.index)) > 0:
                    raise FrameRangeError(msg=f'The chosen time-period ({self.slicing[START_TIME]} - {self.slicing[END_TIME]}) does not exist in {self.video_name}.', source=self.__class__.__name__)
                self.in_df = self.in_df.loc[frm_numbers]
            else:
                frm_numbers = list(range(0, len(self.in_df)))
            self.data_df = self.in_df[self.data_cols]
            video_styles = self.__get_styles(self.style_attr)
            if self.video_setting:
                self.video_save_path = os.path.join(self.path_plot_dir, f"{self.video_name}.mp4")
                self.writer = cv2.VideoWriter(self.video_save_path, self.fourcc, self.fps, (video_styles[STYLE_WIDTH], video_styles[STYLE_HEIGHT]))
                self.video_temp_dir = os.path.join(self.path_plot_dir, self.video_name)
                create_directory(paths=self.video_temp_dir, overwrite=True)
            if self.frame_setting:
                self.save_frm_dir = os.path.join(self.path_plot_dir, self.video_name)
                create_directory(paths=self.save_frm_dir, overwrite=True)
            video_rois, video_roi_names = None, None
            if self.roi:
                video_rois, video_roi_names = slice_roi_dict_for_video(data=self.roi_dict, video_name=self.video_name)
                if len(video_roi_names) == 0:
                    ROIWarning(msg=f'NO ROI data found for video {self.video_name}. Skipping ROI plotting for this video.')
            if self.clf_attr is not None:
                clf_attr = {}
                for k, v in self.clf_attr.items():
                    check_that_column_exist(df=self.in_df, column_name=k, file_name=file_path)
                    clf_attr[k] = {COLOR: v[COLOR], SIZE: v[SIZE], 'clfs': self.in_df[k].values.astype(np.int8), 'positions': self.data_df[[self.bp_data[0]['x'], self.bp_data[0]['y']]].values}
            else:
                clf_attr = None

            line_data = []
            for k, v in self.bp_data.items():
                line_data.append(self.data_df[[v['x'], v['y']]].values)

            if self.last_frame:
                last_frame_save_path = os.path.join(self.path_plot_dir, f"{self.video_name}_final_frame.png")
                if isinstance(video_styles[STYLE_BG], str):
                    bg = read_frm_of_video(video_path=video_styles[STYLE_BG], opacity=video_styles[STYLE_BG_OPACITY], frame_index=len(self.data_df)-1)
                else:
                    bg = deepcopy(video_styles[STYLE_BG])
                last_frm = PlottingMixin.make_path_plot(data=line_data,
                                                        colors=self.colors,
                                                        width=video_styles[STYLE_WIDTH],
                                                        height=video_styles[STYLE_HEIGHT],
                                                        max_lines=video_styles[STYLE_MAX_LINES],
                                                        bg_clr=bg,
                                                        circle_size=video_styles[STYLE_CIRCLE_SIZE],
                                                        font_size=video_styles[STYLE_FONT_SIZE],
                                                        font_thickness=video_styles[STYLE_FONT_THICKNESS],
                                                        line_width=video_styles[STYLE_LINE_WIDTH],
                                                        animal_names=self.animal_names,
                                                        clf_attr=clf_attr,
                                                        save_path=None)
                if video_rois is not None:
                    last_frm = PlottingMixin.roi_dict_onto_img(img=last_frm, roi_dict=video_rois, show_tags=False, show_center=False)
                cv2.imwrite(filename=last_frame_save_path, img=last_frm)
                stdout_success(msg=f'Last path plot frame saved at {last_frame_save_path}')
            if self.video_setting or self.frame_setting:
                frm_cnt_range = np.arange(1, line_data[0].shape[0])
                frm_cnt_range = np.array_split(frm_cnt_range, self.core_cnt)
                frm_id_range = np.array_split(frm_numbers, self.core_cnt)
                frm_range = [(cnt, x, y) for cnt, (x, y) in enumerate(zip(frm_cnt_range, frm_id_range))]
                print(f"Creating path plots, multiprocessing (chunksize: {self.multiprocess_chunksize}, cores: {self.core_cnt})...")
                with multiprocessing.Pool(self.core_cnt, maxtasksperchild=self.maxtasksperchild) as pool:
                    constants = functools.partial(path_plot_mp,
                                                  line_data=line_data,
                                                  colors=self.colors,
                                                  video_setting=self.video_setting,
                                                  video_name=self.video_name,
                                                  frame_setting=self.frame_setting,
                                                  video_save_dir=self.video_temp_dir,
                                                  frame_folder_dir=self.save_frm_dir,
                                                  style_attr=video_styles,
                                                  clf_attr=clf_attr,
                                                  animal_names=self.animal_names,
                                                  fps=self.fps,
                                                  roi=video_rois)
                    for cnt, result in enumerate(pool.imap(constants, frm_range, chunksize=self.multiprocess_chunksize)):
                        print(f"Path batch {result+1}/{self.core_cnt} complete...")
                pool.terminate()
                pool.join()

                if self.video_setting:
                    print(f"Joining {self.video_name} multi-processed video...")
                    print(self.video_temp_dir, self.video_save_path)
                    concatenate_videos_in_folder(in_folder=self.video_temp_dir, save_path=self.video_save_path, remove_splits=False)
                video_timer.stop_timer()
                print(f"Path plot video {self.video_name} complete (elapsed time: {video_timer.elapsed_time_str}s) ...")

        self.timer.stop_timer()
        stdout_success(msg=f"Path plot visualizations for {len(self.data_paths)} video(s) created in {self.path_plot_dir} directory", elapsed_time=self.timer.elapsed_time_str, source=self.__class__.__name__)


#
# if __name__ == "__main__":
#     style_attr = {STYLE_WIDTH: None,
#                   STYLE_HEIGHT: None,
#                   STYLE_LINE_WIDTH: 5,
#                   STYLE_FONT_SIZE: 5,
#                   STYLE_FONT_THICKNESS: 2,
#                   STYLE_CIRCLE_SIZE: 5,
#                   STYLE_BG: 'video',
#                   STYLE_BG_OPACITY: 100,
#                   STYLE_MAX_LINES: None}
#
#     animal_attr = {0: {'body_part': 'Snout', 'color': (255, 0, 0)}, 1: {'body_part': 'Front Paw R', 'color': (0, 0, 255)}}  #['Ear_right_1', 'Red'], 1: ['Ear_right_2', 'Green']}
#     #clf_attr = {'Rearing': {'color': (155, 1, 10), 'size': 30}}
#     clf_attr = None
#     test = PathPlotterMulticore(config_path=r"C:\troubleshooting\open_field_below\project_folder\project_config.ini",
#                                  frame_setting=False,
#                                  video_setting=True,
#                                  last_frame=True,
#                                  slicing={'start_time': '00:00:00', 'end_time': '00:00:05'},#{'start_time': '00:00:00', 'end_time': '00:00:05'}, #{'start_time': '00:00:00', 'end_time': '00:00:50'}, #{'start_time': '00:00:00', 'end_time': '00:00:01'},, #{'start_time': '00:00:00', 'end_time': '00:00:01'}, #{'start_time': '00:00:00', 'end_time': '00:00:01'},
#                                  style_attr=style_attr,
#                                  animal_attr=animal_attr,
#                                  clf_attr=clf_attr,
#                                  print_animal_names=False,
#                                  core_cnt=1,
#                                  roi=True,
#                                  data_paths=[r"C:\troubleshooting\open_field_below\project_folder\csv\outlier_corrected_movement_location\raw_clip1.csv"])
#     test.run()
#



# if __name__ == "__main__":
#     style_attr = {STYLE_WIDTH: None,
#                   STYLE_HEIGHT: None,
#                   STYLE_LINE_WIDTH: 5,
#                   STYLE_FONT_SIZE: 5,
#                   STYLE_FONT_THICKNESS: 2,
#                   STYLE_CIRCLE_SIZE: 5,
#                   STYLE_BG: 'video',
#                   STYLE_BG_OPACITY: 100,
#                   STYLE_MAX_LINES: None}
#
#     animal_attr = {0: {'body_part': 'Ear_right', 'color': (255, 0, 0)}, 1: {'body_part': 'Center', 'color': (0, 0, 255)}}  #['Ear_right_1', 'Red'], 1: ['Ear_right_2', 'Green']}
#     clf_attr = {'Rearing': {'color': (155, 1, 10), 'size': 30}}
#     test = PathPlotterMulticore(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini",
#                                  frame_setting=False,
#                                  video_setting=True,
#                                  last_frame=False,
#                                  slicing={'start_time': '00:00:00', 'end_time': '00:00:05'},#{'start_time': '00:00:00', 'end_time': '00:00:05'}, #{'start_time': '00:00:00', 'end_time': '00:00:50'}, #{'start_time': '00:00:00', 'end_time': '00:00:01'},, #{'start_time': '00:00:00', 'end_time': '00:00:01'}, #{'start_time': '00:00:00', 'end_time': '00:00:01'},
#                                  style_attr=style_attr,
#                                  animal_attr=animal_attr,
#                                  clf_attr=clf_attr,
#                                  print_animal_names=False,
#                                  core_cnt=-1,
#                                  roi=True,
#                                  data_paths=[r"C:\troubleshooting\RAT_NOR\project_folder\csv\machine_results\03152021_NOB_IOT_8.csv"])
#     test.run()




# animal_attr = {0: {'bp': 'Ear_right', 'color': (255, 0, 0)}, 1: {'bp': 'Tail_base', 'color': (0, 0, 255)}}  #['Ear_right_1', 'Red'], 1: ['Ear_right_2', 'Green']}
# style_attr = {'width': 'As input',
#               'height': 'As input',
#               'line width': 2,
#               'font size': 0.9,
#               'font thickness': 2,
#               'circle size': 5,
#               'bg color': {'type': 'moving', 'opacity': 50, 'frame_index': 200}, #{'type': 'static', 'opacity': 100, 'frame_index': 200}
#               'max lines': 'entire video'}
# clf_attr = {'Nose to Nose': {'color': (155, 1, 10), 'size': 30}, 'Nose to Tailbase': {'color': (155, 90, 10), 'size': 30}}
# clf_attr=None
#
# path_plotter = PathPlotterMulticore(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/project_config.ini',
#                                     frame_setting=False,
#                                     video_setting=True,
#                                     last_frame=True,
#                                     clf_attr=clf_attr,
#                                     input_style_attr=None,
#                                     animal_attr=animal_attr,
#                                     roi=True,
#                                     files_found=['/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/csv/outlier_corrected_movement_location/2022-06-20_NOB_DOT_4.csv'],
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
