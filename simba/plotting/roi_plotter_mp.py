__author__ = "Simon Nilsson"

import functools
import itertools
import multiprocessing
import os
import platform
import shutil
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
                                check_nvidea_gpu_available,
                                check_valid_boolean, check_valid_dict,
                                check_valid_lst,
                                check_video_and_data_frm_count_align)
from simba.utils.data import (create_color_palettes, detect_bouts,
                              slice_roi_dict_for_video)
from simba.utils.enums import ROI_SETTINGS, Formats, Keys, Paths, TextOptions
from simba.utils.errors import (BodypartColumnNotFoundError, DuplicationError,
                                NoFilesFoundError, NoROIDataError,
                                ROICoordinatesNotFoundError)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    find_core_cnt, get_video_meta_data,
                                    read_df)
from simba.utils.warnings import (DuplicateNamesWarning, FrameRangeWarning,
                                  GPUToolsWarning)

pd.options.mode.chained_assignment = None

SHOW_BODY_PARTS = 'show_body_part'
SHOW_ANIMAL_NAMES = 'show_animal_name'
STYLE_KEYS = [SHOW_BODY_PARTS, SHOW_ANIMAL_NAMES]


def _roi_plotter_mp(data: Tuple[int, np.ndarray],
                    loc_dict: dict,
                    font_size: float,
                    circle_sizes: list,
                    save_temp_directory: str,
                    video_shape_names: list,
                    input_video_path: str,
                    body_part_dict: dict,
                    roi_dfs_dict: Dict[str, pd.DataFrame],
                    roi_dict: dict,
                    bp_colors: list,
                    style_attr: dict,
                    animal_ids: list,
                    threshold: float,
                    outside_roi: bool,
                    verbose: bool):

    def __insert_texts(roi_dict, img):
        for animal_name in animal_ids:
            for shape_name, shape_data in roi_dict.items():
                img = cv2.putText(img, loc_dict[animal_name][shape_name]["timer_text"], loc_dict[animal_name][shape_name]["timer_text_loc"], TextOptions.FONT.value, font_size, shape_data['Color BGR'], TextOptions.TEXT_THICKNESS.value)
                img = cv2.putText(img, loc_dict[animal_name][shape_name]["entries_text"], loc_dict[animal_name][shape_name]["entries_text_loc"], TextOptions.FONT.value, font_size, shape_data['Color BGR'], TextOptions.TEXT_THICKNESS.value)
            if outside_roi:
                img = cv2.putText(img, loc_dict[animal_name][ROI_SETTINGS.OUTSIDE_ROI.value]["timer_text"], loc_dict[animal_name][ROI_SETTINGS.OUTSIDE_ROI.value]["timer_text_loc"], TextOptions.FONT.value, font_size, TextOptions.WHITE.value, TextOptions.TEXT_THICKNESS.value)
                img = cv2.putText(img, loc_dict[animal_name][ROI_SETTINGS.OUTSIDE_ROI.value]["entries_text"], loc_dict[animal_name][ROI_SETTINGS.OUTSIDE_ROI.value]["entries_text_loc"], TextOptions.FONT.value, font_size, TextOptions.WHITE.value, TextOptions.TEXT_THICKNESS.value)
            return img


        return img

    fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
    group_cnt, data_df = data[0], data[1]
    df_frm_range = data_df.index.tolist()
    start_frm, current_frm, end_frm = df_frm_range[0], df_frm_range[0], df_frm_range[-1]
    save_path = os.path.join(save_temp_directory, f"{group_cnt}.mp4")
    video_meta_data = get_video_meta_data(video_path=input_video_path)
    writer = cv2.VideoWriter(save_path, fourcc, video_meta_data["fps"], (video_meta_data["width"] * 2, video_meta_data["height"]))
    cap = cv2.VideoCapture(input_video_path)
    cap.set(1, start_frm)

    while current_frm <= end_frm:
        ret, img = cap.read()
        if ret:
            img = cv2.copyMakeBorder(img, 0, 0, 0, int(video_meta_data["width"]), borderType=cv2.BORDER_CONSTANT,value=[0, 0, 0])
            img = __insert_texts(roi_dict, img)
            img = PlottingMixin.roi_dict_onto_img(img=img, roi_dict=roi_dfs_dict)
            for animal_cnt, animal_name in enumerate(animal_ids):
                if style_attr[SHOW_BODY_PARTS] or style_attr[SHOW_ANIMAL_NAMES]:
                    x, y, p = (data_df.loc[current_frm, body_part_dict[animal_name]].fillna(0.0).values.astype(np.int32))
                    if threshold <= p:
                        if style_attr[SHOW_BODY_PARTS]:
                            img = cv2.circle(img, (x, y), circle_sizes[animal_cnt], bp_colors[animal_cnt], -1)
                        if style_attr[SHOW_ANIMAL_NAMES]:
                            img = cv2.putText(img, animal_name, (x, y), TextOptions.FONT.value, font_size, bp_colors[animal_cnt], TextOptions.TEXT_THICKNESS.value)
                for shape_name in video_shape_names:
                    shape_color = TextOptions.WHITE.value if shape_name == ROI_SETTINGS.OUTSIDE_ROI.value else roi_dict[shape_name]["Color BGR"]
                    timer = round(data_df.loc[current_frm, f"{animal_name}_{shape_name}_cum_sum_time"], 2)
                    entries = data_df.loc[current_frm, f"{animal_name}_{shape_name}_cum_sum_entries"]
                    img = cv2.putText(img, str(timer), loc_dict[animal_name][shape_name]["timer_data_loc"], TextOptions.FONT.value, font_size, shape_color, TextOptions.TEXT_THICKNESS.value)
                    img = cv2.putText(img, str(entries), loc_dict[animal_name][shape_name]["entries_data_loc"], TextOptions.FONT.value, font_size, shape_color, TextOptions.TEXT_THICKNESS.value)

            writer.write(img)
            current_frm += 1
            if verbose: print(f"Multi-processing video frame {current_frm} on core {group_cnt}...")
        else:
            FrameRangeWarning(msg=f'Could not read frame {current_frm} in video {video_meta_data["video_name"]}', source=_roi_plotter_mp.__name__)
            break

    cap.release()
    writer.release()
    return group_cnt

class ROIPlotMultiprocess(ConfigReader):
    """
    Visualize the ROI data (number of entries/exits, time-spent-in ROIs).

    .. note::
       `ROI tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md>`__.

    .. image:: _static/img/roi_visualize.png
        :width: 400
        :align: center

    .. image:: _static/img/ROIPlot_1.png
       :width: 1000
       :align: center

    .. image:: _static/img/ROIPlot_2.gif
       :width: 1000
       :align: center

    .. video:: _static/img/outside_roi_example.mp4
       :width: 800
       :autoplay:
       :loop:




    :param Union[str, os.PathLike] config_path: Path to SimBA project config file in Configparser format
    :param Union[str, os.PathLike] video_path: Name of video to create ROI visualizations for
    :param Dict[str, bool] style_attr: User-defined visualization settings.
    :param Optional[int] core_cnt: Number of cores to use. Default to -1 representing all available cores
    :param Optional[bool]: If True, SimBA will treat all areas NOT covered by a ROI drawing as a single additional ROI visualize the stats for this, single, ROI.
    :param List[str] body_parts: List of the body-parts to use as proxy for animal locations.
    :param Optional[float] threshold: Float between 0 and 1. Body-part locations detected below this confidence threshold are filtered. Default: 0.0.

    :example:
    >>> test = ROIPlotMultiprocess(config_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/project_config.ini',
    >>>                            video_path="/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/videos/SI_DAY3_308_CD1_PRESENT.mp4",
    >>>                            core_cnt=7,
    >>>                            style_attr={'show_body_part': True, 'show_animal_name': False},
    >>>                            body_parts=['Nose'])
    >>> test.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 video_path: Union[str, os.PathLike],
                 body_parts: List[str],
                 style_attr: Dict[str, bool],
                 threshold: Optional[float] = 0.0,
                 core_cnt: int = -1,
                 verbose: bool = True,
                 outside_roi: bool = False,
                 data_path: Optional[Union[str, os.PathLike]] = None,
                 save_path: Optional[Union[str, os.PathLike]] = None,
                 bp_colors: Optional[List[Tuple[int, int, int]]] = None,
                 bp_sizes: Optional[List[Union[int]]] = None,
                 gpu: bool = False):

        check_file_exist_and_readable(file_path=config_path)
        ConfigReader.__init__(self, config_path=config_path)
        self.video_meta_data = get_video_meta_data(video_path=video_path)
        self.video_name = self.video_meta_data['video_name']
        check_float(name=f'{self.__class__.__name__} threshold', value=threshold, min_value=0.0, max_value=1.0)
        check_int(name=f'{self.__class__.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0,])
        check_if_keys_exist_in_dict(data=style_attr, key=STYLE_KEYS, name=f'{self.__class__.__name__} style_attr')
        check_valid_dict(x=style_attr, required_keys=tuple(STYLE_KEYS,), valid_values_dtypes=(bool,))
        check_valid_boolean(value=[gpu], source=f'{self.__class__.__name__} gpu', raise_error=True)
        check_valid_boolean(value=[outside_roi], source=f'{self.__class__.__name__} outside_roi', raise_error=True)
        check_valid_boolean(value=[verbose], source=f'{self.__class__.__name__} verbose', raise_error=True)
        if gpu and not check_nvidea_gpu_available():
            GPUToolsWarning(msg='GPU not detected but GPU set to True - skipping GPU use.')
            gpu = False
        self.core_cnt = find_core_cnt()[0] if core_cnt == -1 or core_cnt > find_core_cnt()[0] else core_cnt
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
                raise NoFilesFoundError(msg=f"SIMBA ERROR: Could not find the file at path {data_path}. Make sure the data file exist to create ROI visualizations", source=self.__class__.__name__)
            check_file_exist_and_readable(file_path=data_path)
        if save_path is None:
            save_path = os.path.join(self.project_path, Paths.ROI_ANALYSIS.value, f'{self.video_name}.mp4')
            if not os.path.exists(os.path.dirname(save_path)): os.makedirs(os.path.dirname(save_path))
        else:
            check_if_dir_exists(os.path.dirname(save_path))
        self.save_path, self.data_path = save_path, data_path
        check_valid_lst(data=body_parts, source=f'{self.__class__.__name__} body-parts', valid_dtypes=(str,), min_len=1)
        if outside_roi: self.shape_names.append(ROI_SETTINGS.OUTSIDE_ROI.value)
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
        self.bp_dict = self.roi_analyzer.bp_dict
        self.animal_names = [self.find_animal_name_from_body_part_name(bp_name=x, bp_dict=self.animal_bp_dict) for x in body_parts]
        self.data_df = read_df(file_path=self.data_path, file_type=self.file_type, usecols=self.roi_analyzer.roi_headers).fillna(0.0).reset_index(drop=True)
        self.shape_columns = []
        for x in itertools.product(self.animal_names, self.shape_names):
            self.data_df[f"{x[0]}_{x[1]}"] = 0; self.shape_columns.append(f"{x[0]}_{x[1]}")
        self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        self.video_path = video_path
        check_video_and_data_frm_count_align(video=self.video_path, data=self.data_df, name=self.video_name, raise_error=False)
        self.cap = cv2.VideoCapture(self.video_path)
        self.threshold, self.body_parts, self.style_attr, self.gpu, self.outside_roi, self.verbose = threshold, body_parts, style_attr, gpu, outside_roi, verbose
        self.roi_dict_ = get_roi_dict_from_dfs(rectangle_df=self.sliced_roi_dict[Keys.ROI_RECTANGLES.value], circle_df=self.sliced_roi_dict[Keys.ROI_CIRCLES.value], polygon_df=self.sliced_roi_dict[Keys.ROI_POLYGONS.value])
        self.temp_folder = os.path.join(os.path.dirname(self.save_path), self.video_name, "temp")
        if os.path.exists(self.temp_folder): shutil.rmtree(self.temp_folder)
        os.makedirs(self.temp_folder)
        self.roi_dict_ = get_roi_dict_from_dfs(rectangle_df=self.sliced_roi_dict[Keys.ROI_RECTANGLES.value], circle_df=self.sliced_roi_dict[Keys.ROI_CIRCLES.value], polygon_df=self.sliced_roi_dict[Keys.ROI_POLYGONS.value])
        if platform.system() == "Darwin": multiprocessing.set_start_method("spawn", force=True)

    def __get_circle_sizes(self):
        optimal_circle_size = PlottingMixin().get_optimal_circle_size(frame_size=(int(self.video_meta_data["height"]), int(self.video_meta_data["height"])), circle_frame_ratio=100)
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

    def __get_text_locs(self) -> dict:
         loc_dict = {}
         txt_strs = []
         for animal_cnt, animal_name in enumerate(self.animal_names):
             for shape in self.shape_names:
                 txt_strs.append(animal_name + ' ' + shape + ' entries')
         longest_text_str = max(txt_strs, key=len)
         self.font_size, x_spacer, y_spacer = PlottingMixin().get_optimal_font_scales(text=longest_text_str, accepted_px_width=int(self.video_meta_data["width"] / 1.5), accepted_px_height=int(self.video_meta_data["height"] / 10), text_thickness=TextOptions.TEXT_THICKNESS.value)
         row_counter = TextOptions.FIRST_LINE_SPACING.value
         for animal_cnt, animal_name in enumerate(self.animal_names):
             loc_dict[animal_name] = {}
             for shape in self.shape_names:
                 loc_dict[animal_name][shape] = {}
                 loc_dict[animal_name][shape]["timer_text"] = f"{shape} {animal_name} timer:"
                 loc_dict[animal_name][shape]["entries_text"] = f"{shape} {animal_name} entries:"
                 loc_dict[animal_name][shape]["timer_text_loc"] = ((self.video_meta_data["width"] + TextOptions.BORDER_BUFFER_X.value), (self.video_meta_data["height"] - (self.video_meta_data["height"] + TextOptions.BORDER_BUFFER_Y.value) + y_spacer * row_counter))
                 loc_dict[animal_name][shape]["timer_data_loc"] = (int(self.video_meta_data["width"] + x_spacer + TextOptions.BORDER_BUFFER_X.value), (self.video_meta_data["height"] - (self.video_meta_data["height"] + TextOptions.BORDER_BUFFER_Y.value) + y_spacer * row_counter))
                 row_counter += 1
                 loc_dict[animal_name][shape]["entries_text_loc"] = ((self.video_meta_data["width"] + TextOptions.BORDER_BUFFER_X.value), (self.video_meta_data["height"] - (self.video_meta_data["height"] + TextOptions.BORDER_BUFFER_Y.value) + y_spacer * row_counter))
                 loc_dict[animal_name][shape]["entries_data_loc"] = (int(self.video_meta_data["width"] + x_spacer + TextOptions.BORDER_BUFFER_X.value), (self.video_meta_data["height"]- (self.video_meta_data["height"] + TextOptions.BORDER_BUFFER_Y.value) + y_spacer * row_counter))
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

    def __get_cumulative_data(self):
        for animal_name in self.animal_names:
            for shape in self.shape_names:
                self.data_df[f"{animal_name}_{shape}_cum_sum_time"] = (self.data_df[f"{animal_name}_{shape}"].cumsum() / self.video_meta_data['fps'])
                roi_bouts = list(detect_bouts(data_df=self.data_df, target_lst=[f"{animal_name}_{shape}"], fps=self.video_meta_data['fps'])["Start_frame"])
                self.data_df[f"{animal_name}_{shape}_entry"] = 0
                self.data_df.loc[roi_bouts, f"{animal_name}_{shape}_entry"] = 1
                self.data_df[f"{animal_name}_{shape}_cum_sum_entries"] = (self.data_df[f"{animal_name}_{shape}_entry"].cumsum())


    def __create_shape_dicts(self):
        shape_dicts = {}
        for shape, df in self.roi_dict.items():
            if not df["Name"].is_unique:
                df = df.drop_duplicates(subset=["Name"], keep="first")
                DuplicateNamesWarning(msg=f'Some of your ROIs with the same shape ({shape}) has the same names for video {self.video_name}. E.g., you have two rectangles named "My rectangle". SimBA prefers ROI shapes with unique names. SimBA will keep one of the unique shape names and drop the rest.', source=self.__class__.__name__)
            d = df.set_index("Name").to_dict(orient="index")
            shape_dicts = {**shape_dicts, **d}
        return shape_dicts

    def __get_bordered_img_size(self) -> Tuple[int, int]:
        cap = cv2.VideoCapture(self.video_path)
        cap.set(1, 1)
        _, img = self.cap.read()
        bordered_img = cv2.copyMakeBorder(img, 0, 0, 0, int(self.video_meta_data["width"]), borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        cap.release()
        return bordered_img.shape[0], bordered_img.shape[1]

    def run(self):
        check_video_and_data_frm_count_align(video=self.video_path, data=self.data_df, name=self.video_name, raise_error=False)
        video_timer = SimbaTimer(start=True)
        self.__get_circle_sizes()
        self.__get_roi_columns()
        self.border_img_h, self.border_img_w = self.__get_bordered_img_size()
        self.loc_dict = self.__get_text_locs()
        self.cnt_dict = self.__get_counters()
        self.__get_cumulative_data()
        data = np.array_split(self.data_df, self.core_cnt)
        data = [(i, j) for i, j in enumerate(data)]
        del self.data_df
        del self.roi_analyzer.logger
        if self.verbose: print(f"Creating ROI images, multiprocessing (chunksize: {self.multiprocess_chunksize}, cores: {self.core_cnt})...")
        with multiprocessing.Pool(self.core_cnt, maxtasksperchild=self.maxtasksperchild) as pool:
            constants = functools.partial(_roi_plotter_mp,
                                          loc_dict=self.loc_dict,
                                          font_size=self.font_size,
                                          circle_sizes=self.circle_sizes,
                                          save_temp_directory=self.temp_folder,
                                          body_part_dict=self.bp_dict,
                                          input_video_path=self.video_path,
                                          roi_dfs_dict=self.sliced_roi_dict,
                                          roi_dict = self.roi_dict_,
                                          video_shape_names=self.shape_names,
                                          bp_colors=self.color_lst,
                                          style_attr=self.style_attr,
                                          animal_ids=self.animal_names,
                                          threshold=self.threshold,
                                          outside_roi=self.outside_roi,
                                          verbose=self.verbose)

            for cnt, batch_cnt in enumerate(pool.imap(constants, data, chunksize=self.multiprocess_chunksize)):
                print(f'Image batch {batch_cnt+1} / {self.core_cnt} complete...')
            print(f"Joining {self.video_name} multi-processed ROI video...")
            concatenate_videos_in_folder(in_folder=self.temp_folder, save_path=self.save_path, video_format="mp4", remove_splits=True, gpu=self.gpu)
            pool.terminate()
            pool.join()
        video_timer.stop_timer()
        stdout_success(msg=f"Video {self.video_name} created. ROI video saved at {self.save_path}", elapsed_time=video_timer.elapsed_time_str, source=self.__class__.__name__, )



# if __name__ == "__main__":
#     test = ROIPlotMultiprocess(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini",
#                                video_path=r"C:\troubleshooting\mitra\project_folder\videos\501_MA142_Gi_Saline_0515.mp4",
#                                body_parts=['Nose'],
#                                style_attr={'show_body_part': True, 'show_animal_name': False},
#                                outside_roi=False,
#                                gpu=True)
#     test.run()

# if __name__ == "__main__":
#     test = ROIPlotMultiprocess(config_path=r"C:\troubleshooting\roi_duplicates\project_folder\project_config.ini",
#                                video_path=r"C:\troubleshooting\roi_duplicates\project_folder\videos\2021-12-21_15-03-57_CO_Trimmed.mp4",
#                                body_parts=['Snout'],
#                                style_attr={'show_body_part': True, 'show_animal_name': False},
#                                bp_sizes=[20],
#                                bp_colors=[(155, 255, 243)])
#     test.run()
#





# if __name__ == '__main__':
#     test = ROIPlotMultiprocess(config_path=r"C:\troubleshooting\platea\project_folder\project_config.ini",
#                               video_path=r"C:\troubleshooting\platea\project_folder\videos\Video_1.mp4",
#                               body_parts=['NOSE'],
#                               style_attr={'show_body_part': True, 'show_animal_name': False})
#     test.run()

# test = ROIPlotMultiprocess(config_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/open_field_below/project_folder/project_config.ini',
#                           video_path="/Users/simon/Desktop/envs/simba/troubleshooting/open_field_below/project_folder/videos/raw_clip1.mp4",
#                           body_parts=['Snout'],
#                           style_attr={'show_body_part': True, 'show_animal_name': False})
# test.run()


# test = ROIPlotMultiprocess(config_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/project_config.ini',
#                           video_path="/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/2022-06-20_NOB_DOT_4.mp4",
#                           body_parts=['Nose'],
#                           style_attr={'show_body_part': True, 'show_animal_name': False})
# test.run()



# test = ROIPlotMultiprocess(config_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/project_config.ini',
#                            video_path="/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/videos/SI_DAY3_308_CD1_PRESENT.mp4",
#                            core_cnt=-1,
#                            style_attr={'show_body_part': True, 'show_animal_name': False},
#                            body_parts=['Nose'])
# test.run()

# test = ROIPlotMultiprocess(config_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                            video_path="/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/videos/Together_1.avi",
#                            core_cnt=7,
#                            style_attr={'show_body_part': True, 'show_animal_name': True},
#                            body_parts=['Nose_1', 'Nose_2'])
# test.run()



#
# test = ROIPlotMultiprocess(ini_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/project_config.ini',
#                video_path="2022-06-20_NOB_DOT_4.mp4",
#                core_cnt=7,
#                style_attr={'Show_body_part': True, 'Show_animal_name': True}, body_parts={'Animal_1': 'Nose'})
# test.run()
#

#
# test = ROIPlotMultiprocess(ini_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/spontenous_alternation/project_folder/project_config.ini',
#                video_path="F1 HAB.mp4",
#                core_cnt=5,
#                style_attr={'Show_body_part': True, 'Show_animal_name': True})
# test.run()
#
# get_video_meta_data(video_path='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/frames/output/ROI_analysis/2022-06-20_NOB_DOT_4.mp4')
# get_video_meta_data(video_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/videos/Together_1.avi')

# test = ROIPlot(ini_path=r'/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini', video_path=r"Together_1.avi")
# test.insert_data()
# test.visualize_ROI_data()

# test = ROIPlot(ini_path=r"Z:\DeepLabCut\DLC_extract\Troubleshooting\ROI_2_animals\project_folder\project_config.ini", video_path=r"Z:\DeepLabCut\DLC_extract\Troubleshooting\ROI_2_animals\project_folder\videos\Video7.mp4")
# test.insert_data()
# test.visualize_ROI_data()
#
# test = ROIPlotMultiprocess(ini_path=r'/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                            video_path="Together_1.avi",
#                            style_attr={'Show_body_part': True, 'Show_animal_name': False},
#                            core_cnt=5)
# test.run()


# test = ROIPlotMultiprocess(ini_path=r'/Users/simon/Desktop/envs/troubleshooting/DLC_2_Black_animals/project_folder/project_config.ini',
#                            video_path="Together_1.avi",
#                            style_attr={'Show_body_part': True, 'Show_animal_name': False},
#                            core_cnt=5)
# test.run()
