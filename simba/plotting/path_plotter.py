__author__ = "Simon Nilsson"

import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

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
from simba.utils.read_write import (find_video_of_file, get_fn_ext, read_df,
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

class PathPlotterSingleCore(ConfigReader, PlottingMixin):
    """
    Create "path plots" videos and/or images detailing the movement paths of individual animals in SimBA.

    .. note::
        For improved run-time, see :meth:`simba.path_plotter_mp.PathPlotterMulticore` for multiprocess class.

       `Visualization tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-11-visualizations>`__.

       `Notebook example <https://simba-uw-tf-dev.readthedocs.io/en/latest/nb/path_plots.html>`__.

    .. image:: _static/img/path_plot.png
       :width: 300
       :align: center

    .. image:: _static/img/path_plot_mp.gif
       :width: 500
       :align: center

    .. image:: _static/img/path_plot_1.png
       :width: 500
       :align: center

    :param str config_path: Path to SimBA project config file in Configparser format
    :param bool frame_setting: If True, individual frames will be created.
    :param bool video_setting: If True, compressed videos will be created.
    :param List[str] files_found: Data paths to create from which to create plots
    :param dict animal_attr: Animal body-parts and colors
    :param dict style_attr: Plot sttributes (line thickness, color, etc..)
    :param Optional[dict] slicing: If Dict, start time and end time of video slice to create path plot from. E.g., {'start_time': '00:00:01', 'end_time': '00:00:03'}. If None, creates path plot for entire video.
    :param Optional[bool] roi: If True, also plots the ROIs associated with the video. Default False.

    .. note::
       If style_attr['bg color'] is a dictionary, e.g., {'opacity': 100%}, then SimBA will use the video as background with set opacity.

    :examples:
    >>> style_attr = {'width': 'As input', 'height': 'As input', 'line width': 5, 'font size': 5, 'font thickness': 2, 'circle size': 5, 'bg color': 'White', 'max lines': 100, 'animal_names': True}
    >>> animal_attr = {0: ['Ear_right_1', 'Red']}
    >>> path_plotter = PathPlotterSingleCore(config_path=r'MyConfigPath', frame_setting=False, video_setting=True, style_attr=style_attr, animal_attr=animal_attr, files_found=['project_folder/csv/machine_results/MyVideo.csv'], print_animal_names=True).run()

    :references:
       .. [1] Boorman, Damien C., Simran K. Rehal, Maryam Fazili, and Loren J. Martin. “Sex and Strain Differences in Analgesic and Hyperlocomotor Effects of Morphine and μ‐Opioid Receptor Expression in Mice.” Journal of Neuroscience Research 103, no. 4 (April 2025): e70039. https://doi.org/10.1002/jnr.70039.



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

    #
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
            if self.frame_setting:
                self.save_video_folder = os.path.join(self.path_plot_dir, self.video_name)
                if os.path.exists(self.save_video_folder): remove_a_folder(folder_dir=self.save_video_folder, ignore_errors=True)
                os.makedirs(self.save_video_folder)
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
                for frm_id, frm_cnt in enumerate(range(frm_numbers[1], frm_numbers[-1])):
                    plot_arrs = [x[:frm_id+1, :] for x in line_data]
                    if isinstance(video_styles[STYLE_BG], str):
                        bg = read_frm_of_video(video_path=video_styles[STYLE_BG], opacity=video_styles[STYLE_BG_OPACITY], frame_index=frm_cnt)
                    else:
                        bg = deepcopy(video_styles[STYLE_BG])
                    img = PlottingMixin.make_path_plot(data=plot_arrs,
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
                        img = PlottingMixin.roi_dict_onto_img(img=img, roi_dict=video_rois, show_tags=False, show_center=False)
                    if self.video_setting:
                        self.writer.write(np.uint8(img))
                    if self.frame_setting:
                        frm_name = os.path.join(self.save_video_folder, f"{frm_cnt}.png")
                        cv2.imwrite(frm_name, np.uint8(img))
                    print(f"Path frame: {frm_cnt + 1} / {frm_numbers[-1]} created. Video: {self.video_name} ({str(file_cnt + 1)}/{len(self.data_paths)})")
                if self.video_setting:
                    self.writer.release()
                video_timer.stop_timer()
                print(f"Path visualization for video {self.video_name} saved (elapsed time {video_timer.elapsed_time_str}s)...")

        self.timer.stop_timer()
        stdout_success(msg=f"Path visualizations for {len(self.data_paths)} video(s) saved in {self.path_plot_dir} directory", elapsed_time=self.timer.elapsed_time_str, source=self.__class__.__name__)


# style_attr = {STYLE_WIDTH: None,
#               STYLE_HEIGHT: None,
#               STYLE_LINE_WIDTH: 5,
#               STYLE_FONT_SIZE: 5,
#               STYLE_FONT_THICKNESS: 2,
#               STYLE_CIRCLE_SIZE: 5,
#               STYLE_BG: 'video',
#               STYLE_BG_OPACITY: 100,
#               STYLE_MAX_LINES: None}
#
# animal_attr = {0: {'body_part': 'Ear_right', 'color': (255, 0, 0)}, 1: {'body_part': 'Center', 'color': (0, 0, 255)}}  #['Ear_right_1', 'Red'], 1: ['Ear_right_2', 'Green']}
# clf_attr = {'Rearing': {'color': (155, 1, 10), 'size': 30}}
# test = PathPlotterSingleCore(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini",
#                              frame_setting=False,
#                              video_setting=True,
#                              last_frame=True,
#                              slicing={'start_time': '00:00:25', 'end_time': '00:00:35'},#{'start_time': '00:00:00', 'end_time': '00:00:05'}, #{'start_time': '00:00:00', 'end_time': '00:00:50'}, #{'start_time': '00:00:00', 'end_time': '00:00:01'},, #{'start_time': '00:00:00', 'end_time': '00:00:01'}, #{'start_time': '00:00:00', 'end_time': '00:00:01'},
#                              style_attr=style_attr,
#                              animal_attr=animal_attr,
#                              clf_attr=clf_attr,
#                              print_animal_names=False,
#                              roi=True,
#                              data_paths=[r"C:\troubleshooting\RAT_NOR\project_folder\csv\machine_results\03152021_NOB_IOT_8.csv"])
# test.run()



#
#
# #
# style_attr = {'width': 'As input',
#               'height': 'As input',
#               'line width': 2,
#               'font size': 0.9,
#               'font thickness': 2,
#               'circle size': 5,
#               'bg color': {'type': 'moving', 'opacity': 50, 'frame_index': 200}, #{'type': 'static', 'opacity': 100, 'frame_index': 200}
#               'max lines': 'entire video'}
# #
# animal_attr = {0: {'bp': 'Ear_right', 'color': (255, 0, 0)}, 1: {'bp': 'Center', 'color': (0, 0, 255)}}  #['Ear_right_1', 'Red'], 1: ['Ear_right_2', 'Green']}
# # # clf_attr = {'Nose to Nose': {'color': (155, 1, 10), 'size': 30}, 'Nose to Tailbase': {'color': (155, 90, 10), 'size': 30}}
# style_attr = None
# test = PathPlotterSingleCore(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/project_config.ini',
#                              frame_setting=False,
#                              video_setting=True,
#                              last_frame=True,
#                              slicing=None,#{'start_time': '00:00:00', 'end_time': '00:00:05'}, #{'start_time': '00:00:00', 'end_time': '00:00:50'}, #{'start_time': '00:00:00', 'end_time': '00:00:01'},, #{'start_time': '00:00:00', 'end_time': '00:00:01'}, #{'start_time': '00:00:00', 'end_time': '00:00:01'},
#                              input_style_attr=style_attr,
#                              animal_attr=animal_attr,
#                              clf_attr=None,
#                              print_animal_names=False,
#                              roi=True,
#                              files_found=['/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/csv/outlier_corrected_movement_location/2022-06-20_NOB_DOT_4.csv'])
# test.run()

# test = PathPlotterSingleCore(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                              frame_setting=False,
#                              video_setting=True,
#                              last_frame=True,
#                              input_style_attr=style_attr,
#                              animal_attr=animal_attr,
#                              clf_attr=None,
#                              roi=True,
#                              slicing = {'start_time': '00:00:01', 'end_time': '00:00:08'},
#                              files_found=['/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'])
# #test.run()


# style_attr = {'width': 'As input',
#               'height': 'As input',
#               'line width': 5,
#               'font size': 5,
#               'font thickness': 2,
#               'circle size': 5,
#               'bg color': {'type': 'static', 'opacity': 70},
#               'max lines': 'entire video'}
# # #
# animal_attr = {0: ['LM_Ear_right_1', 'Red'], 1: ['UM_Ear_right_2', 'Green']}
# #clf_attr = {0: ['Attack', 'Black', 'Size: 30'], 1: ['Sniffing', 'Black', 'Size: 30']}
#
# test = PathPlotterSingleCore(config_path='/Users/simon/Desktop/envs/troubleshooting/dorian_2/project_folder/project_config.ini',
#                              frame_setting=False,
#                              video_setting=False,
#                              last_frame=True,
#                              input_style_attr=style_attr,
#                              animal_attr=animal_attr,
#                              input_clf_attr=None,
#                              files_found=['/Users/simon/Desktop/envs/troubleshooting/dorian_2/project_folder/csv/machine_results/HybCD1-B2-D6-Urine.csv'])
# test.run()
