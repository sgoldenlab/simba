__author__ = "Simon Nilsson"

import itertools
import os
from typing import Dict, List, Optional, Tuple, Union

import cv2

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.roi_tools.ROI_analyzer import ROIAnalyzer
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_keys_exist_in_dict, check_valid_lst,
                                check_video_and_data_frm_count_align)
from simba.utils.data import (create_color_palettes, detect_bouts,
                              slice_roi_dict_for_video)
from simba.utils.enums import Formats, Keys, Paths, TagNames, TextOptions
from simba.utils.errors import (BodypartColumnNotFoundError, CountError,
                                DuplicationError, NoFilesFoundError,
                                ROICoordinatesNotFoundError)
from simba.utils.printing import SimbaTimer, log_event, stdout_success
from simba.utils.read_write import get_fn_ext, get_video_meta_data
from simba.utils.warnings import DuplicateNamesWarning

SHOW_BODY_PARTS = 'show_body_part'
SHOW_ANIMAL_NAMES = 'show_animal_name'
STYLE_KEYS = [SHOW_BODY_PARTS, SHOW_ANIMAL_NAMES]


class ROIPlot(ConfigReader):
    """
    Visualize the ROI data (number of entries/exits, time-spent in ROIs etc).

    .. note::
       `ROI tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md>`__.

       Use :meth:`simba.plotting.ROI_plotter_mp.ROIPlotMultiprocess` for improved run-time.

    .. image:: _static/img/ROIPlot_1.png
       :width: 800
       :align: center

    .. image:: _static/img/ROIPlot_1.gif
       :width: 800
       :align: center

    :param Union[str, os.PathLike] config_path: Path to SimBA project config file in Configparser format
    :param Union[str, os.PathLike] video_path: Name of video to create ROI visualizations for
    :param Dict[str, bool] style_attr: User-defined visualization settings.
    :param List[str] body_parts: List of the body-parts to use as proxy for animal locations.
    :param Optional[float] threshold: Float between 0 and 1. Body-part locations detected below this confidence threshold are filtered. Default: 0.0.

    :example:
    >>> test = ROIPlot(config_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/project_config.ini',
    >>>                video_path="/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/videos/SI_DAY3_308_CD1_PRESENT.mp4",
    >>>                body_parts=['Nose'],
    >>>                style_attr={'show_body_part': True, 'show_animal_name': True})
    >>> test.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 video_path: Union[str, os.PathLike],
                 style_attr: Dict[str, bool],
                 body_parts: List[str],
                 threshold: Optional[float] = 0.0):

        check_float(name=f'{self.__class__.__name__} threshold', value=threshold, min_value=0.0, max_value=1.0)
        check_if_keys_exist_in_dict(data=style_attr, key=STYLE_KEYS, name=f'{self.__class__.__name__} style_attr')
        check_file_exist_and_readable(file_path=video_path)
        _, self.video_name, _ = get_fn_ext(video_path)
        ConfigReader.__init__(self, config_path=config_path)
        if not os.path.isfile(self.roi_coordinates_path):
            raise ROICoordinatesNotFoundError(expected_file_path=self.roi_coordinates_path)
        self.data_path = os.path.join(self.outlier_corrected_dir, f'{self.video_name}.{self.file_type}')
        if not os.path.isfile(self.data_path):
            raise NoFilesFoundError( msg=f"SIMBA ERROR: Could not find the file at path {self.data_path}. Make sure the data file exist to create ROI visualizations", source=self.__class__.__name__)
        log_event(logger_name=str(__class__.__name__), log_type=TagNames.CLASS_INIT.value, msg=self.create_log_msg_from_init_args(locals=locals()))
        check_valid_lst(data=body_parts, source=f'{self.__class__.__name__} body-parts', valid_dtypes=(str,), min_len=1)
        if len(set(body_parts)) != len(body_parts):
            raise DuplicationError(msg=f'All body-part entries have to be unique. Got {body_parts}', source=self.__class__.__name__)
        for bp in body_parts:
            if bp not in self.body_parts_lst:
                raise BodypartColumnNotFoundError(msg=f'The body-part {bp} is not a valid body-part in the SimBA project. Options: {self.body_parts_lst}', source=self.__class__.__name__)

        self.roi_analyzer = ROIAnalyzer(config_path=self.config_path, data_path=self.data_path,  detailed_bout_data=True, threshold=threshold, body_parts=body_parts)
        self.roi_analyzer.run()
        self.roi_entries_df = self.roi_analyzer.detailed_df
        self.data_df, self.style_attr = self.roi_analyzer.data_df, style_attr
        self.save_dir = os.path.join(self.project_path, Paths.ROI_ANALYSIS.value)
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        self.video_save_path = os.path.join(self.save_dir, f"{self.video_name}.mp4")
        self.read_roi_data()
        self.shape_columns = []
        self.roi_dict, self.shape_names = slice_roi_dict_for_video(data=self.roi_dict, video_name=self.video_name)
        if len(self.shape_names) == 0:
            raise CountError(msg=f'No drawn ROIs detected for video {self.video_name}, please draw ROIs on this video before visualizing ROIs', source=self.__class__.__name__)
        self.animal_names = [self.find_animal_name_from_body_part_name(bp_name=x, bp_dict=self.animal_bp_dict) for x in body_parts]
        for x in itertools.product(self.animal_names, self.shape_names):
            self.data_df[f"{x[0]}_{x[1]}"] = 0; self.shape_columns.append(f"{x[0]}_{x[1]}")
        self.bp_dict = self.roi_analyzer.bp_dict
        self.__insert_data()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.video_meta_data = get_video_meta_data(self.video_path)
        self.threshold, self.body_parts = threshold, body_parts

    def __insert_data(self):
        roi_entries_dict = self.roi_entries_df[["ANIMAL", "SHAPE NAME", "START FRAME", "END FRAME"]].to_dict(orient="records")
        for entry_dict in roi_entries_dict:
            entry, exit = int(entry_dict["START FRAME"]), int(entry_dict["END FRAME"])
            entry_dict["frame_range"] = list(range(entry, exit + 1))
            col_name =  f'{entry_dict["ANIMAL"]}_{entry_dict["SHAPE NAME"]}'
            self.data_df[col_name][self.data_df.index.isin(entry_dict["frame_range"])] = 1

    def __calc_text_locs(self) -> dict:
        loc_dict = {}
        line_spacer = TextOptions.FIRST_LINE_SPACING.value
        for animal_cnt, animal_name in enumerate(self.animal_names):
            loc_dict[animal_name] = {}
            for shape in self.shape_names:
                loc_dict[animal_name][shape] = {}
                loc_dict[animal_name][shape]["timer_text"] = f"{shape} {animal_name} timer:"
                loc_dict[animal_name][shape]["entries_text"] = f"{shape} {animal_name} entries:"
                loc_dict[animal_name][shape]["timer_text_loc"] = ((self.video_meta_data["width"] + TextOptions.BORDER_BUFFER_X.value), (self.video_meta_data["height"] - (self.video_meta_data["height"] + TextOptions.BORDER_BUFFER_Y.value) + self.scalers["space_size"] * line_spacer))
                loc_dict[animal_name][shape]["timer_data_loc"] = (int(self.border_img_w - (self.border_img_w / 8)), (self.video_meta_data["height"] - (self.video_meta_data["height"] + TextOptions.BORDER_BUFFER_Y.value) + self.scalers["space_size"] * line_spacer))
                line_spacer += TextOptions.LINE_SPACING.value
                loc_dict[animal_name][shape]["entries_text_loc"] = ((self.video_meta_data["width"] + TextOptions.BORDER_BUFFER_X.value), (self.video_meta_data["height"] - (self.video_meta_data["height"] + TextOptions.BORDER_BUFFER_Y.value) + self.scalers["space_size"] * line_spacer))
                loc_dict[animal_name][shape]["entries_data_loc"] = (int(self.border_img_w - (self.border_img_w / 8)), (self.video_meta_data["height"]- (self.video_meta_data["height"] + TextOptions.BORDER_BUFFER_Y.value) + self.scalers["space_size"] * line_spacer))
                line_spacer += TextOptions.LINE_SPACING.value
        return loc_dict

    def __insert_texts(self, shape_df):
        for animal_name in self.animal_names:
            for _, shape in shape_df.iterrows():
                shape_name, shape_color = shape["Name"], shape["Color BGR"]
                cv2.putText(self.border_img, self.loc_dict[animal_name][shape_name]["timer_text"], self.loc_dict[animal_name][shape_name]["timer_text_loc"], TextOptions.FONT.value, self.scalers["font_size"], shape_color, TextOptions.TEXT_THICKNESS.value)
                cv2.putText(self.border_img, self.loc_dict[animal_name][shape_name]["entries_text"], self.loc_dict[animal_name][shape_name]["entries_text_loc"], TextOptions.FONT.value, self.scalers["font_size"], shape_color, TextOptions.TEXT_THICKNESS.value)

    def __create_counters(self) -> dict:
        cnt_dict = {}
        for animal_cnt, animal_name in enumerate(self.animal_names):
            cnt_dict[animal_name] = {}
            for shape in self.shape_names:
                cnt_dict[animal_name][shape] = {}
                cnt_dict[animal_name][shape]["timer"] = 0
                cnt_dict[animal_name][shape]["entries"] = 0
                cnt_dict[animal_name][shape]["entry_status"] = False
        return cnt_dict

    def __calculate_cumulative(self):
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
                DuplicateNamesWarning(f'Some of your ROIs with the same shape ({shape}) has the same names for video {self.video_name}. E.g., you have two rectangles named "My rectangle". SimBA prefers ROI shapes with unique names. SimBA will keep one of the unique shape names and drop the rest.', source=self.__class__.__name__)
            d = df.set_index("Name").to_dict(orient="index")
            shape_dicts = {**shape_dicts, **d}
        return shape_dicts

    def __get_bordered_img_size(self) -> Tuple[int, int]:
        cap = cv2.VideoCapture(self.video_path)
        cap.set(1, 1)
        _, img = self.cap.read()
        self.base_img = cv2.copyMakeBorder(img, 0, 0, 0, int(self.video_meta_data["width"]), borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        self.base_img_h, self.base_img_w = self.base_img.shape[0], self.base_img.shape[1]
        cap.release()
        return self.base_img_h, self.base_img_w

    def run(self):
        video_timer = SimbaTimer(start=True)
        max_dim = max(self.video_meta_data["width"], self.video_meta_data["height"])
        self.scalers = {}
        self.scalers["circle_size"] = int(TextOptions.RADIUS_SCALER.value / (TextOptions.RESOLUTION_SCALER.value / max_dim))
        self.scalers["font_size"] = float(TextOptions.FONT_SCALER.value / (TextOptions.RESOLUTION_SCALER.value / max_dim))
        self.scalers["space_size"] = int(TextOptions.SPACE_SCALER.value / (TextOptions.RESOLUTION_SCALER.value / max_dim))
        color_lst = create_color_palettes(self.roi_analyzer.animal_cnt, len(self.body_parts))[0]
        self.border_img_h, self.border_img_w = self.__get_bordered_img_size()
        fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        writer = cv2.VideoWriter(self.video_save_path, fourcc, self.video_meta_data["fps"], (self.border_img_w, self.border_img_h))
        self.loc_dict = self.__calc_text_locs()
        self.cnt_dict = self.__create_counters()
        self.shape_dicts = self.__create_shape_dicts()
        self.__calculate_cumulative()
        check_video_and_data_frm_count_align(video=self.video_path, data=self.data_df, name=self.video_name, raise_error=False)
        frame_cnt = 0
        while self.cap.isOpened():
            ret, img = self.cap.read()
            if ret:
                self.border_img = cv2.copyMakeBorder(img, 0, 0, 0, int(self.video_meta_data["width"]),  borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
                self.__insert_texts(self.roi_dict[Keys.ROI_RECTANGLES.value])
                self.__insert_texts(self.roi_dict[Keys.ROI_CIRCLES.value])
                self.__insert_texts(self.roi_dict[Keys.ROI_POLYGONS.value])
                self.img_w_border = PlottingMixin.roi_dict_onto_img(img=self.border_img, roi_dict=self.roi_dict)
                for animal_cnt, animal_name in enumerate(self.animal_names):
                    bp_data = (self.data_df.loc[frame_cnt, self.bp_dict[animal_name]].fillna(0.0).values)
                    if self.threshold < bp_data[2]:
                        if self.style_attr[SHOW_BODY_PARTS]:
                            cv2.circle(self.border_img, (int(bp_data[0]), int(bp_data[1])), self.scalers["circle_size"], color_lst[animal_cnt], -1)
                        if self.style_attr[SHOW_ANIMAL_NAMES]:
                            cv2.putText(self.border_img, animal_name, (int(bp_data[0]), int(bp_data[1])), self.font, self.scalers["font_size"], color_lst[animal_cnt], TextOptions.TEXT_THICKNESS.value)
                for animal_cnt, animal_name in enumerate(self.animal_names):
                    for shape in self.shape_names:
                        time = str(round(self.data_df.loc[frame_cnt, f"{animal_name}_{shape}_cum_sum_time"], 2))
                        entries = str(int(self.data_df.loc[frame_cnt, f"{animal_name}_{shape}_cum_sum_entries"]))
                        cv2.putText(self.border_img, time, self.loc_dict[animal_name][shape]["timer_data_loc"], self.font, self.scalers["font_size"], self.shape_dicts[shape]["Color BGR"], TextOptions.TEXT_THICKNESS.value)
                        cv2.putText(self.border_img, entries, self.loc_dict[animal_name][shape]["entries_data_loc"], self.font, self.scalers["font_size"], self.shape_dicts[shape]["Color BGR"], TextOptions.TEXT_THICKNESS.value)
                writer.write(self.border_img)
                print(f"Frame: {frame_cnt+1} / {self.video_meta_data['frame_count']}, Video: {self.video_name}.")
                frame_cnt += 1
            else:
                break
        writer.release()
        video_timer.stop_timer()
        stdout_success(msg=f"Video {self.video_name} created. Video saved at {self.video_save_path}", elapsed_time=video_timer.elapsed_time_str, source=self.__class__.__name__)

# test = ROIPlot(config_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/project_config.ini',
#                video_path="/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/2022-06-20_NOB_DOT_4.mp4",
#                body_parts=['Nose'],
#                style_attr={'show_body_part': True, 'show_animal_name': False})
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
