__author__ = "Simon Nilsson"

import itertools
import os
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.roi_tools.roi_aggregate_statistics_analyzer import \
    ROIAggregateStatisticsAnalyzer
from simba.roi_tools.roi_utils import get_roi_dict_from_dfs
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists,
                                check_if_keys_exist_in_dict,
                                check_if_valid_rgb_tuple, check_int,
                                check_valid_boolean, check_valid_lst,
                                check_video_and_data_frm_count_align)
from simba.utils.data import (create_color_palettes, detect_bouts,
                              slice_roi_dict_for_video)
from simba.utils.enums import Formats, Keys, Paths, TagNames, TextOptions
from simba.utils.errors import (BodypartColumnNotFoundError, DuplicationError,
                                NoFilesFoundError, NoROIDataError,
                                ROICoordinatesNotFoundError)
from simba.utils.printing import SimbaTimer, log_event, stdout_success
from simba.utils.read_write import (get_video_meta_data, read_df,
                                    read_frm_of_video)
from simba.utils.warnings import FrameRangeWarning

SHOW_BODY_PARTS = 'show_body_part'
SHOW_ANIMAL_NAMES = 'show_animal_name'
STYLE_KEYS = [SHOW_BODY_PARTS, SHOW_ANIMAL_NAMES]
OUTSIDE_ROI = 'OUTSIDE REGIONS OF INTEREST'


class ROIPlotter(ConfigReader):
    """
    Visualize the ROI data (number of entries/exits, time-spent in ROIs etc).

    .. note::
       `ROI tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md>`__.

    .. seelalso::
       Use :func:`simba.plotting.ROI_plotter_mp.ROIPlotMultiprocess` for improved run-time.

    .. image:: _static/img/ROIPlot_1.png
       :width: 800
       :align: center

    .. image:: _static/img/ROIPlot_1.gif
       :width: 800
       :align: center

    .. video:: _static/img/outside_roi_example.mp4
       :width: 800
       :autoplay:
       :loop:

    ..  youtube:: Q2ByLfwJIaw
       :width: 640
       :height: 480
       :align: center

    :param Union[str, os.PathLike] config_path: Path to SimBA project config file in Configparser format
    :param Union[str, os.PathLike] video_path: Name of video to create ROI visualizations for
    :param Dict[str, bool] style_attr: User-defined visualization settings.
    :param List[str] body_parts: List of the body-parts to use as proxy for animal locations.
    :param Optional[float] threshold: Float between 0 and 1. Body-part locations detected below this confidence threshold are filtered. Default: 0.0.
    :param Optional[bool]: If True, SimBA will treat all areas NOT covered by a ROI drawing as a single additional ROI visualize the stats for this, single, ROI.
    :param Optional[Union[str, os.PathLike]] data_path: Optional path to the pose-estimation data. If None, then locates file in ``outlier_corrected_movement_location`` directory.
    :param Optional[Union[str, os.PathLike]] save_path: Optional path to where to save video If None, then saves it in ''frames/output/roi_analys`` directory of SimBA project.
    :param Optional[List[Tuple[int, int, int]] bp_colors: Optional list of tuples of same length as body_parts representing the colors of the body-parts. Defaults to None and colors are automatically chosen.
    :param Optional[List[int]] bp_sizes: Optional list of integers representing the sizes of the pose estimated body-part location. Defaults to None and size is automnatically inferred.


    :example:
    >>> test = ROIPlotter(config_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/project_config.ini',
    >>>                video_path="/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/videos/SI_DAY3_308_CD1_PRESENT.mp4",
    >>>                body_parts=['Nose'],
    >>>                style_attr={'show_body_part': True, 'show_animal_name': True})
    >>> test.run()


    :example II:
    >>> test = ROIPlotter(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini",
    >>>               video_path=r"C:\troubleshooting\mitra\project_folder\videos\501_MA142_Gi_Saline_0513.mp4",
    >>>               body_parts=['Nose'],
    >>>               style_attr={'show_body_part': True, 'show_animal_name': False})
    >>> test.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 video_path: Union[str, os.PathLike],
                 style_attr: Dict[str, bool],
                 body_parts: List[str],
                 outside_roi: bool = False,
                 threshold: float = 0.0,
                 verbose: Optional[bool] = True,
                 data_path: Optional[Union[str, os.PathLike]] = None,
                 save_path: Optional[Union[str, os.PathLike]] = None,
                 bp_colors: Optional[List[Tuple[int, int, int]]] = None,
                 bp_sizes: Optional[List[Union[int]]] = None):

        log_event(logger_name=str(__class__.__name__), log_type=TagNames.CLASS_INIT.value, msg=self.create_log_msg_from_init_args(locals=locals()))
        check_float(name=f'{self.__class__.__name__} threshold', value=threshold, min_value=0.0, max_value=1.0)
        check_if_keys_exist_in_dict(data=style_attr, key=STYLE_KEYS, name=f'{self.__class__.__name__} style_attr')
        check_valid_boolean(value=outside_roi, source=f'{self.__class__.__name__} outside_roi', raise_error=True)
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose', raise_error=True)
        self.video_meta = get_video_meta_data(video_path=video_path)
        self.video_path = video_path
        self.video_name = self.video_meta['video_name']
        ConfigReader.__init__(self, config_path=config_path)
        if not os.path.isfile(self.roi_coordinates_path):
            raise ROICoordinatesNotFoundError(expected_file_path=self.roi_coordinates_path)
        self.read_roi_data()
        self.sliced_roi_dict, self.shape_names = slice_roi_dict_for_video(data=self.roi_dict, video_name=self.video_name)
        if len(self.shape_names) == 0:
            raise NoROIDataError(msg=f"Cannot plot ROI data for video {self.video_name}. No ROIs defined for this video.")
        if data_path is None:
            data_path = os.path.join(self.outlier_corrected_dir, f'{self.video_name}.{self.file_type}')
        else:
            if not os.path.isfile(data_path):
                raise NoFilesFoundError(msg=f"SIMBA ERROR: Could not find the file at path {data_path}. Make sure the data file exist to create ROI visualizations",  source=self.__class__.__name__)
            check_file_exist_and_readable(file_path=data_path)
        if save_path is None:
            save_path = os.path.join(self.project_path, Paths.ROI_ANALYSIS.value, f'{self.video_name}.mp4')
            if not os.path.exists(os.path.dirname(save_path)): os.makedirs(os.path.dirname(save_path))
        else:
            check_if_dir_exists(os.path.dirname(save_path))
        self.save_path, self.data_path = save_path, data_path
        check_valid_lst(data=body_parts, source=f'{self.__class__.__name__} body-parts', valid_dtypes=(str,), min_len=1)
        if outside_roi: self.shape_names.append(OUTSIDE_ROI)
        if len(set(body_parts)) != len(body_parts):
            raise DuplicationError(msg=f'All body-part entries have to be unique. Got {body_parts}', source=self.__class__.__name__)
        for bp in body_parts:
            if bp not in self.body_parts_lst: raise BodypartColumnNotFoundError(msg=f'The body-part {bp} is not a valid body-part in the SimBA project. Options: {self.body_parts_lst}', source=self.__class__.__name__)
        self.roi_analyzer = ROIAggregateStatisticsAnalyzer(config_path=self.config_path, data_path=self.data_path,  detailed_bout_data=True, threshold=threshold, body_parts=body_parts, outside_rois=outside_roi, verbose=verbose)
        self.roi_analyzer.run()
        if bp_colors is not None:
            check_valid_lst(data=bp_colors, source=f'{self.__class__.__name__} bp_colors', valid_dtypes=(tuple,), exact_len=len(body_parts), raise_error=True)
            _ = [check_if_valid_rgb_tuple(x) for x in bp_colors]
            self.color_lst = bp_colors
        else:
            self.color_lst = create_color_palettes(self.roi_analyzer.animal_cnt, len(body_parts))[0]
        self.bp_sizes = bp_sizes
        try:
            self.detailed_roi_data = pd.concat(self.roi_analyzer.detailed_dfs, axis=0).reset_index(drop=True)
        except ValueError:
            self.detailed_roi_data = None

        self.detailed_roi_data = pd.concat(self.roi_analyzer.detailed_dfs, axis=0).reset_index(drop=True)
        self.bp_dict = self.roi_analyzer.bp_dict
        self.animal_names = [self.find_animal_name_from_body_part_name(bp_name=x, bp_dict=self.animal_bp_dict) for x in body_parts]
        self.data_df = read_df(file_path=self.data_path, file_type=self.file_type, usecols=self.roi_analyzer.roi_headers).fillna(0.0).reset_index(drop=True)
        self.shape_columns = []
        for x in itertools.product(self.animal_names, self.shape_names):
            self.data_df[f"{x[0]}_{x[1]}"] = 0; self.shape_columns.append(f"{x[0]}_{x[1]}")
        self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        check_video_and_data_frm_count_align(video=self.video_path, data=self.data_df, name=self.video_name, raise_error=False)
        self.cap = cv2.VideoCapture(self.video_path)
        self.threshold, self.body_parts, self.style_attr, self.outside_roi, self.verbose = threshold, body_parts, style_attr, outside_roi, verbose
        self.roi_dict_ = get_roi_dict_from_dfs(rectangle_df=self.sliced_roi_dict[Keys.ROI_RECTANGLES.value], circle_df=self.sliced_roi_dict[Keys.ROI_CIRCLES.value], polygon_df=self.sliced_roi_dict[Keys.ROI_POLYGONS.value])

    def __get_circle_sizes(self):
        optimal_circle_size = PlottingMixin().get_optimal_circle_size(frame_size=(int(self.video_meta["height"]), int(self.video_meta["height"])), circle_frame_ratio=70)
        if self.bp_sizes is None:
            self.circle_sizes = [optimal_circle_size] * len(self.animal_names)
        else:
            self.circle_sizes = []
            for circle_size in self.bp_sizes:
                if not check_int(name='circle_size', value=circle_size, min_value=1, raise_error=False)[0]:
                    self.circle_sizes.append(optimal_circle_size)
                else:
                    self.circle_sizes.append(int(circle_size))

    def __get_roi_columns(self):
        if self.detailed_roi_data is not None:
            roi_entries_dict = self.detailed_roi_data[["ANIMAL", "Event", "Start_frame", "End_frame"]].to_dict(orient="records")
            for entry_dict in roi_entries_dict:
                entry, exit = int(entry_dict["Start_frame"]), int(entry_dict["End_frame"])
                entry_dict["frame_range"] = list(range(entry, exit + 1))
                col_name =  f'{entry_dict["ANIMAL"]}_{entry_dict["Event"]}'
                self.data_df[col_name][self.data_df.index.isin(entry_dict["frame_range"])] = 1

    def __get_bordered_img_size(self) -> Tuple[int, int]:
        img = read_frm_of_video(video_path=self.video_path, frame_index=0)
        self.base_img = cv2.copyMakeBorder(img, 0, 0, 0, int(self.video_meta["width"]), borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return (self.base_img.shape[0], self.base_img.shape[1])

    def __get_text_locs(self) -> dict:
         loc_dict = {}
         txt_strs = []
         for animal_cnt, animal_name in enumerate(self.animal_names):
             for shape in self.shape_names:
                 txt_strs.append(f'{animal_name} {shape} entries')
         longest_text_str = max(txt_strs, key=len)
         self.font_size, x_spacer, y_spacer = PlottingMixin().get_optimal_font_scales(text=longest_text_str, accepted_px_width=int(self.video_meta["width"] / 1.5), accepted_px_height=int(self.video_meta["height"] / 10), text_thickness=TextOptions.TEXT_THICKNESS.value)
         row_counter = TextOptions.FIRST_LINE_SPACING.value
         for animal_cnt, animal_name in enumerate(self.animal_names):
             loc_dict[animal_name] = {}
             for shape in self.shape_names:
                 loc_dict[animal_name][shape] = {}
                 loc_dict[animal_name][shape]["timer_text"] = f"{shape} {animal_name} timer:"
                 loc_dict[animal_name][shape]["entries_text"] = f"{shape} {animal_name} entries:"
                 loc_dict[animal_name][shape]["timer_text_loc"] = ((self.video_meta["width"] + TextOptions.BORDER_BUFFER_X.value), (self.video_meta["height"] - (self.video_meta["height"] + TextOptions.BORDER_BUFFER_Y.value) + y_spacer * row_counter))
                 loc_dict[animal_name][shape]["timer_data_loc"] = (int(self.video_meta["width"] + x_spacer + TextOptions.BORDER_BUFFER_X.value), (self.video_meta["height"] - (self.video_meta["height"] + TextOptions.BORDER_BUFFER_Y.value) + y_spacer * row_counter))
                 row_counter += 1
                 loc_dict[animal_name][shape]["entries_text_loc"] = ((self.video_meta["width"] + TextOptions.BORDER_BUFFER_X.value), (self.video_meta["height"] - (self.video_meta["height"] + TextOptions.BORDER_BUFFER_Y.value) + y_spacer * row_counter))
                 loc_dict[animal_name][shape]["entries_data_loc"] = (int(self.video_meta["width"] + x_spacer + TextOptions.BORDER_BUFFER_X.value), (self.video_meta["height"]- (self.video_meta["height"] + TextOptions.BORDER_BUFFER_Y.value) + y_spacer * row_counter))
                 row_counter += 1
         return loc_dict

    def __get_counters(self) -> dict:
        cnt_dict = {}
        for animal_cnt, animal_name in enumerate(self.animal_names):
            cnt_dict[animal_name] = {}
            for shape in self.shape_names:
                cnt_dict[animal_name][shape] = {}
                cnt_dict[animal_name][shape]["timer"] = 0
                cnt_dict[animal_name][shape]["entries"] = 0
                cnt_dict[animal_name][shape]["entry_status"] = False
        return cnt_dict

    def __insert_texts(self, roi_dict, img):
        for animal_name in self.animal_names:
            for shape_name, shape_data in roi_dict.items():
                img = cv2.putText(img, self.loc_dict[animal_name][shape_name]["timer_text"], self.loc_dict[animal_name][shape_name]["timer_text_loc"], TextOptions.FONT.value, self.font_size, shape_data['Color BGR'], TextOptions.TEXT_THICKNESS.value)
                img = cv2.putText(img, self.loc_dict[animal_name][shape_name]["entries_text"], self.loc_dict[animal_name][shape_name]["entries_text_loc"], TextOptions.FONT.value, self.font_size, shape_data['Color BGR'], TextOptions.TEXT_THICKNESS.value)
            if self.outside_roi:
                img = cv2.putText(img, self.loc_dict[animal_name][OUTSIDE_ROI]["timer_text"], self.loc_dict[animal_name][OUTSIDE_ROI]["timer_text_loc"], TextOptions.FONT.value, self.font_size, TextOptions.WHITE.value, TextOptions.TEXT_THICKNESS.value)
                img = cv2.putText(img, self.loc_dict[animal_name][OUTSIDE_ROI]["entries_text"], self.loc_dict[animal_name][OUTSIDE_ROI]["entries_text_loc"], TextOptions.FONT.value, self.font_size, TextOptions.WHITE.value, TextOptions.TEXT_THICKNESS.value)
        return img

    def __get_cumulative_data(self):
        for animal_name in self.animal_names:
            for shape in self.shape_names:
                self.data_df[f"{animal_name}_{shape}_cum_sum_time"] = (self.data_df[f"{animal_name}_{shape}"].cumsum() / self.video_meta['fps'])
                roi_bouts = list(detect_bouts(data_df=self.data_df, target_lst=[f"{animal_name}_{shape}"], fps=self.video_meta['fps'])["Start_frame"])
                self.data_df[f"{animal_name}_{shape}_entry"] = 0
                self.data_df.loc[roi_bouts, f"{animal_name}_{shape}_entry"] = 1
                self.data_df[f"{animal_name}_{shape}_cum_sum_entries"] = (self.data_df[f"{animal_name}_{shape}_entry"].cumsum())

    def run(self):
        video_timer = SimbaTimer(start=True)
        self.__get_circle_sizes()
        self.__get_roi_columns()
        self.border_img_h, self.border_img_w = self.__get_bordered_img_size()
        writer = cv2.VideoWriter(self.save_path, self.fourcc, self.video_meta["fps"], (self.border_img_w, self.border_img_h))
        self.loc_dict = self.__get_text_locs()
        self.cnt_dict = self.__get_counters()
        self.__get_cumulative_data()
        frame_cnt = 0
        while self.cap.isOpened():
            ret, img = self.cap.read()
            if ret:
                img = cv2.copyMakeBorder(img, 0, 0, 0, int(self.video_meta["width"]),  borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
                img = self.__insert_texts(self.roi_dict_, img)
                img = PlottingMixin.roi_dict_onto_img(img=img, roi_dict=self.sliced_roi_dict)
                for animal_cnt, animal_name in enumerate(self.animal_names):
                    x, y, p = (self.data_df.loc[frame_cnt, self.bp_dict[animal_name]].fillna(0.0).values.astype(np.int32))
                    if (self.threshold <= p) and self.style_attr[SHOW_BODY_PARTS]:
                        img = cv2.circle(img, (x, y), self.circle_sizes[animal_cnt], self.color_lst[animal_cnt], -1)
                    if (self.threshold <= p) and self.style_attr[SHOW_ANIMAL_NAMES]:
                        img = cv2.putText(img, animal_name, (x, y), self.font, self.font_size, self.color_lst[animal_cnt], TextOptions.TEXT_THICKNESS.value)
                for animal_cnt, animal_name in enumerate(self.animal_names):
                    for shape_name, shape_data in self.roi_dict_.items():
                        time = str(round(self.data_df.loc[frame_cnt, f"{animal_name}_{shape_name}_cum_sum_time"], 2))
                        entries = str(int(self.data_df.loc[frame_cnt, f"{animal_name}_{shape_name}_cum_sum_entries"]))
                        img = cv2.putText(img, time, self.loc_dict[animal_name][shape_name]["timer_data_loc"], self.font, self.font_size, shape_data["Color BGR"], TextOptions.TEXT_THICKNESS.value)
                        img = cv2.putText(img, entries, self.loc_dict[animal_name][shape_name]["entries_data_loc"], self.font, self.font_size, shape_data["Color BGR"], TextOptions.TEXT_THICKNESS.value)
                    if self.outside_roi:
                        time = str(round(self.data_df.loc[frame_cnt, f"{animal_name}_{OUTSIDE_ROI}_cum_sum_time"], 2))
                        entries = str(int(self.data_df.loc[frame_cnt, f"{animal_name}_{OUTSIDE_ROI}_cum_sum_entries"]))
                        img = cv2.putText(img, time, self.loc_dict[animal_name][OUTSIDE_ROI]["timer_data_loc"], self.font, self.font_size, TextOptions.WHITE.value, TextOptions.TEXT_THICKNESS.value)
                        img = cv2.putText(img, entries, self.loc_dict[animal_name][OUTSIDE_ROI]["entries_data_loc"], self.font, self.font_size, TextOptions.WHITE.value, TextOptions.TEXT_THICKNESS.value)
                writer.write(img)
                if self.verbose: print(f"Frame: {frame_cnt+1} / {self.video_meta['frame_count']}, Video: {self.video_name}.")
                frame_cnt += 1
            else:
                FrameRangeWarning(msg=f'Could not read frame {frame_cnt} in video {self.video_name}', source=self.__class__.__name__)
                break
        writer.release()
        video_timer.stop_timer()
        if self.verbose: stdout_success(msg=f"Video {self.video_name} created. Video saved at {self.save_path}", elapsed_time=video_timer.elapsed_time_str, source=self.__class__.__name__)



# if __name__ == "__main__":
#     test = ROIPlotter(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini",
#                                video_path=r"C:\troubleshooting\mitra\project_folder\videos\502_MA141_Gi_Saline_0517.mp4",
#                                body_parts=['Nose'],
#                                style_attr={'show_body_part': True, 'show_animal_name': False})
#     test.run()


# test = ROIPlotter(config_path=r"C:\troubleshooting\roi_duplicates\project_folder\project_config.ini",
#                   video_path=r"C:\troubleshooting\roi_duplicates\project_folder\videos\2021-12-21_15-03-57_CO_Trimmed.mp4",
#                   body_parts=['Snout'],
#                   style_attr={'show_body_part': True, 'show_animal_name': False})
# test.run()



# if __name__ == "__main__":
#     test = ROIPlotMultiprocess(config_path=r"C:\troubleshooting\roi_duplicates\project_folder\project_config.ini",
#                                video_path=r"C:\troubleshooting\roi_duplicates\project_folder\videos\2021-12-21_15-03-57_CO_Trimmed.mp4",
#                                body_parts=['Snout'],
#                                style_attr={'show_body_part': True, 'show_animal_name': False},
#                                bp_sizes=[20],
#                                bp_colors=[(155, 255, 243)])
#     test.run()
#


# test = ROIPlotter(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini",
#                   video_path=r"C:\troubleshooting\mitra\project_folder\videos\501_MA142_Gi_Saline_0513.mp4",
#                   body_parts=['Nose'],
#                   style_attr={'show_body_part': True, 'show_animal_name': False},
#                   outside_roi=True)
# test.run()


# test = ROIPlot(config_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/project_config.ini',
#                video_path="/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/videos/SI_DAY3_308_CD1_PRESENT.mp4",
#                body_parts=['Nose'],
#                style_attr={'show_body_part': True, 'show_animal_name': True})
# test.run()

# test = ROIPlot(ini_path=r'/Users/simon/Desktop/envs/troubleshooting/Termites_5/project_folder/project_config.ini',
#                video_path="termite_test.mp4",
#                style_attr={'Show_body_part': True, 'Show_animal_name': True})
# test.insert_data()
# test.visualize_ROI_data()


# test = ROIPlot(ini_path=r'/Users/simon/Desktop/envs/troubleshooting/Termites_5/project_folder/project_config.ini', video_path="termite_test.mp4")
# test.insert_data()
# test.visualize_ROI_data()

# test = ROIPlot(ini_path=r'/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini', video_path=r"Together_1.avi")
# test.insert_data()
# test.visualize_ROI_data()


# test = ROIPlot(ini_path=r'/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                video_path="Together_1.avi",
#                style_attr={'Show_body_part': True, 'Show_animal_name': False},
#                body_parts={f'Simon': 'Ear_left_1'})
# test.insert_data()
# test.run()

# test = ROIPlot(ini_path=r'/Users/simon/Desktop/envs/troubleshooting/Termites_5/project_folder/project_config.ini',
#                video_path="termite_test.mp4",
#                style_attr={'Show_body_part': True, 'Show_animal_name': True},
#                body_parts={f'Simon': 'Termite_1_Head_1'})
# test.insert_data()
# test.run()
