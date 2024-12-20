__author__ = "Simon Nilsson"

import itertools
import os
from copy import deepcopy
from typing import Tuple, Optional, Union, Dict

import cv2
import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.roi_tools.ROI_analyzer import ROIAnalyzer
from simba.utils.data import create_color_palettes, slice_roi_dict_for_video, detect_bouts
from simba.utils.checks import check_float, check_if_keys_exist_in_dict, check_file_exist_and_readable, check_video_and_data_frm_count_align
from simba.utils.enums import Formats, Paths, TagNames, TextOptions
from simba.utils.errors import DuplicationError, NoFilesFoundError
from simba.utils.printing import SimbaTimer, log_event, stdout_success
from simba.utils.read_write import (get_fn_ext, get_video_meta_data)
from simba.utils.warnings import DuplicateNamesWarning

SHOW_BODY_PARTS = 'show_body_part'
SHOW_ANIMAL_NAMES = 'show_animal_name'
STYLE_KEYS = [SHOW_BODY_PARTS, SHOW_ANIMAL_NAMES]


class ROIPlot(ConfigReader, PlottingMixin):
    """
    Visualize the ROI data (number of entries/exits, time-spent in ROIs etc).

    .. note::
       `ROI tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md>`__.

       Use :meth:`simba.plotting.ROI_plotter_mp.ROIPlotMultiprocess` for improved run-time.

    :param str config_path: Path to SimBA project config file in Configparser format
    :param str video_path: Name of video to create ROI visualizations for
    :param dict style_attr: User-defined visualization settings.

    :example:
    >>> settings = {'show_body_parts': True, 'show_animal_name': True}
    >>> roi_visualizer = ROIPlot(ini_path=r'MyProjectConfig', video_path="MyVideo.mp4", settings=settings)
    >>> roi_visualizer.insert_data()
    >>> roi_visualizer.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 video_path: Union[str, os.PathLike],
                 style_attr: Dict[str, bool],
                 body_parts: Optional[Dict[str, str]] = None,
                 threshold: Optional[float] = 0.0):

        check_float(name=f'{self.__class__.__name__} threshold', value=threshold, min_value=0.0, max_value=1.0)
        check_if_keys_exist_in_dict(data=style_attr, key=STYLE_KEYS, name=f'{self.__class__.__name__} style_attr')
        check_file_exist_and_readable(file_path=video_path)
        ConfigReader.__init__(self, config_path=config_path)
        PlottingMixin.__init__(self)
        log_event(logger_name=str(__class__.__name__), log_type=TagNames.CLASS_INIT.value, msg=self.create_log_msg_from_init_args(locals=locals()))
        settings = None
        if body_parts: settings = {"body_parts": body_parts, 'threshold': threshold}
        self.roi_analyzer = ROIAnalyzer(ini_path=config_path, data_path="outlier_corrected_movement_location", settings=settings)
        if not body_parts: self.animal_id_lst = self.roi_analyzer.multi_animal_id_list
        else: self.animal_id_lst = list(body_parts.keys())
        _, self.video_name, _ = get_fn_ext(video_path)
        self.roi_analyzer.files_found = [os.path.join(self.roi_analyzer.input_folder, f"{self.video_name}.{self.roi_analyzer.file_type}")]
        if not os.path.isfile(self.roi_analyzer.files_found[0]):
            raise NoFilesFoundError( msg=f"SIMBA ERROR: Could not find the file at path {self.roi_analyzer.files_found[0]}. Please make sure you have corrected body-part outliers or indicated that you want to skip outlier correction", source=self.__class__.__name__, )
        self.roi_analyzer.run()
        self.roi_entries_df = self.roi_analyzer.detailed_df
        self.data_df, self.style_attr = self.roi_analyzer.data_df, style_attr
        self.save_dir = os.path.join(self.project_path, Paths.ROI_ANALYSIS.value)
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        self.video_save_path = os.path.join(self.save_dir, f"{self.video_name}.mp4")
        self.read_roi_data()
        self.shape_columns = []
        _, self.shape_names = slice_roi_dict_for_video(data=self.roi_dict, video_name=self.video_name)
        for animal in self.animal_id_lst:
            for shape_name in self.shape_names:
                self.data_df[f"{animal}_{shape_name}"] = 0; self.shape_columns.append(f"{animal}_{shape_name}")
        self.bp_dict = self.roi_analyzer.bp_dict
        self.__insert_data()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.video_meta_data = get_video_meta_data(self.video_path)
        self.threshold = threshold

    def __insert_data(self):
        roi_entries_dict = self.roi_entries_df[["ANIMAL", "SHAPE", "ENTRY FRAMES", "EXIT FRAMES"]].to_dict(orient="records")
        for entry_dict in roi_entries_dict:
            entry, exit = int(entry_dict["ENTRY FRAMES"]), int(entry_dict["EXIT FRAMES"])
            entry_dict["frame_range"] = list(range(entry, exit + 1))
            col_name = entry_dict["ANIMAL"] + "_" + entry_dict["SHAPE"]
            self.data_df[col_name][self.data_df.index.isin(entry_dict["frame_range"])] = 1

    def __calc_text_locs(self) -> dict:
        loc_dict = {}
        line_spacer = TextOptions.FIRST_LINE_SPACING.value
        for animal_cnt, animal_name in enumerate(self.animal_id_lst):
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
        for animal_name in self.animal_id_lst:
            for _, shape in shape_df.iterrows():
                shape_name, shape_color = shape["Name"], shape["Color BGR"]
                cv2.putText(self.border_img, self.loc_dict[animal_name][shape_name]["timer_text"], self.loc_dict[animal_name][shape_name]["timer_text_loc"], TextOptions.FONT.value, self.scalers["font_size"], shape_color, TextOptions.TEXT_THICKNESS.value)
                cv2.putText(self.border_img, self.loc_dict[animal_name][shape_name]["entries_text"], self.loc_dict[animal_name][shape_name]["entries_text_loc"], TextOptions.FONT.value, self.scalers["font_size"], shape_color, TextOptions.TEXT_THICKNESS.value)

    def __create_counters(self) -> dict:
        cnt_dict = {}
        for animal_cnt, animal_name in enumerate(self.animal_id_lst):
            cnt_dict[animal_name] = {}
            for shape in self.shape_names:
                cnt_dict[animal_name][shape] = {}
                cnt_dict[animal_name][shape]["timer"] = 0
                cnt_dict[animal_name][shape]["entries"] = 0
                cnt_dict[animal_name][shape]["entry_status"] = False
        return cnt_dict

    def __calculate_cumulative(self):
        for animal_name in self.animal_id_lst:
            for shape in self.shape_names:
                self.data_df[f"{animal_name}_{shape}_cum_sum_time"] = (self.data_df[f"{animal_name}_{shape}"].cumsum() / self.video_meta_data['fps'])
                roi_bouts = list(detect_bouts(data_df=self.data_df, target_lst=[f"{animal_name}_{shape}"], fps=self.video_meta_data['fps'])["Start_frame"])
                self.data_df[f"{animal_name}_{shape}_entry"] = 0
                self.data_df.loc[roi_bouts, f"{animal_name}_{shape}_entry"] = 1
                self.data_df[f"{animal_name}_{shape}_cum_sum_entries"] = (self.data_df[f"{animal_name}_{shape}_entry"].cumsum())

    def __create_shape_dicts(self):
        shape_dicts = {}
        for df in [self.roi_analyzer.video_recs, self.roi_analyzer.video_circs, self.roi_analyzer.video_polys]:
            if not df["Name"].is_unique:
                df = df.drop_duplicates(subset=["Name"], keep="first")
                DuplicateNamesWarning('Some of your ROIs with the same shape has the same names. E.g., you have two rectangles named "My rectangle". SimBA prefers ROI shapes with unique names. SimBA will keep one of the unique shape names and drop the rest.', source=self.__class__.__name__)
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
        color_lst = create_color_palettes(self.roi_analyzer.animal_cnt, int((len(self.roi_analyzer.bp_names) / 3)))[0]
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
                self.__insert_texts(self.roi_analyzer.video_recs)
                self.__insert_texts(self.roi_analyzer.video_circs)
                self.__insert_texts(self.roi_analyzer.video_polys)
                for _, row in self.roi_analyzer.video_recs.iterrows():
                    top_left_x, top_left_y, shape_name = (row["topLeftX"], row["topLeftY"], row["Name"])
                    bottom_right_x, bottom_right_y = (row["Bottom_right_X"], row["Bottom_right_Y"])
                    thickness, color = row["Thickness"], row["Color BGR"]
                    cv2.rectangle(self.border_img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), color,int(thickness))
                for _, row in self.roi_analyzer.video_circs.iterrows():
                    center_x, center_y, radius, shape_name = (row["centerX"],row["centerY"],row["radius"],row["Name"])
                    thickness, color = row["Thickness"], row["Color BGR"]
                    cv2.circle(self.border_img,(center_x, center_y),radius,color,int(thickness))
                for _, row in self.roi_analyzer.video_polys.iterrows():
                    vertices, shape_name = row["vertices"], row["Name"]
                    thickness, color = row["Thickness"], row["Color BGR"]
                    cv2.polylines(self.border_img,[vertices],True,color,thickness=int(thickness))
                for animal_cnt, animal_name in enumerate(self.animal_id_lst):
                    bp_data = (self.data_df.loc[frame_cnt, self.bp_dict[animal_name]].fillna(0.0).values)
                    if self.threshold < bp_data[2]:
                        if self.style_attr[SHOW_BODY_PARTS]:
                            cv2.circle(self.border_img, (int(bp_data[0]), int(bp_data[1])), self.scalers["circle_size"], color_lst[animal_cnt], -1)
                        if self.style_attr[SHOW_ANIMAL_NAMES]:
                            cv2.putText(self.border_img, animal_name, (int(bp_data[0]), int(bp_data[1])), self.font, self.scalers["font_size"], color_lst[animal_cnt], TextOptions.TEXT_THICKNESS.value)
                for animal_cnt, animal_name in enumerate(self.animal_id_lst):
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


# test = ROIPlot(config_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/project_config.ini',
#                video_path="/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/videos/SI_DAY3_308_CD1_PRESENT.mp4",
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
