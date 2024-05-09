import os
from typing import Union

import cv2
import numpy as np

from simba.feature_extractors.perimeter_jit import jitted_hull
from simba.mixins.circular_statistics import CircularStatisticsMixin
from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_int)
from simba.utils.enums import Formats, TextOptions
from simba.utils.lookups import integer_to_cardinality_lookup
from simba.utils.printing import stdout_success
from simba.utils.read_write import (find_video_of_file, get_fn_ext,
                                    get_video_meta_data, read_df)

FONT_SIZE = 'font_size'
SPACE_SIZE = 'space_size'
TEXT_THICKNESS = 'text_thickness'
CIRCLE_SIZE = 'circle_size'

class CircularFeaturePlotter(ConfigReader, PlottingMixin, FeatureExtractionMixin):
    """
    Create visualization of base angular features overlay on video. E.g., use to confirm
    accurate cardinality and angle degree computation.

    .. image:: _static/img/circular_visualiation.gif
       :width: 600
       :align: center

    :param Union[str, os.PathLike] config_path: path to SimBA project config file in Configparser format
    :param Union[str, os.PathLike] data_path: Path to file containing angular features.
    :param dict settings: Dictionary containing visualization attributes.

    :example:
    >>> settings = {'center': {'Animal_1': 'SwimBladder'}, 'text_settings': False, "palette": 'bwr'}
    >>> circular_feature_plotter = CircularFeaturePlotter(config_path='/Users/simon/Desktop/envs/troubleshooting/circular_features_zebrafish/project_folder/project_config.ini', data_path='/Users/simon/Desktop/envs/troubleshooting/circular_features_zebrafish/project_folder/csv/circular_features/20200730_AB_7dpf_850nm_0002.csv', settings=settings)
    >>> circular_feature_plotter.run()
    """

    def __init__(self,
                  config_path: Union[str, os.PathLike],
                  data_path: Union[str, os.PathLike],
                  settings: dict):

        PlottingMixin.__init__(self)
        ConfigReader.__init__(self, config_path=config_path, read_video_info=True)
        FeatureExtractionMixin.__init__(self)
        self.video_path = find_video_of_file( video_dir=self.video_dir, filename=get_fn_ext(filepath=data_path)[1])
        _, self.video_name, _ = get_fn_ext(filepath=self.video_path)
        check_file_exist_and_readable(file_path=data_path)
        self.data_path, self.config_path, self.settings, self.text_settings = (data_path, config_path, settings, settings["text_settings"])
        self.save_path = os.path.join(self.frames_output_dir,"circular_features", f"{self.video_name}.mp4",)
        if not os.path.isdir(os.path.dirname(self.save_path)):
            os.makedirs(os.path.dirname(self.save_path))
        self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)

    def _approximate_size_of_animal_in_video(self):
        self.animal_sizes = {}
        for animal_name, animal_bps in self.animal_bp_dict.items():
            self.animal_sizes[animal_name] = {}
            animal_bp_cols = [item for pair in zip(animal_bps["X_bps"], animal_bps["Y_bps"]) for item in pair]
            animal_bp_data = (self.df[animal_bp_cols].values.reshape(len(self.df), len(animal_bps["X_bps"]), 2).astype(np.float32))
            self.animal_sizes[animal_name]["area"] = np.nanmean(jitted_hull(points=animal_bp_data, target="area")).astype(np.int32)
            self.animal_sizes[animal_name]["diameter"] = (np.sqrt(self.animal_sizes[animal_name]["area"] / np.pi).astype(np.int32)* 3)

    def __get_print_settings(self):
        if self.text_settings is False:
            self.max_dim = max(self.video_meta_data["width"], self.video_meta_data["height"])
            self.circle_scale = int(TextOptions.RADIUS_SCALER.value / (TextOptions.RESOLUTION_SCALER.value / self.max_dim))
            self.font_size = float(TextOptions.FONT_SCALER.value / (TextOptions.RESOLUTION_SCALER.value / self.max_dim))
            self.spacing_scale = int(TextOptions.SPACE_SCALER.value / (TextOptions.RESOLUTION_SCALER.value / self.max_dim))
            self.text_thickness = 3
        else:
            check_float(name=f"{self.__class__.__name__} FONT_SIZE", value=self.text_settings[FONT_SIZE])
            check_int(name=f"{self.__class__.__name__} SPACE_SIZE", value=self.text_settings[SPACE_SIZE])
            check_int(name=f"{self.__class__.__name__} TEXT THICKNESS", value=self.text_settings[TEXT_THICKNESS])
            check_int(name=f"{self.__class__.__name__} CIRCLE SIZE", value=self.text_settings[CIRCLE_SIZE])
            self.font_size = float(self.text_settings[FONT_SIZE])
            self.spacing_scale = int(self.text_settings[SPACE_SIZE])
            self.text_thickness = int(self.text_settings[TEXT_THICKNESS])
            self.circle_scale = int(self.text_settings[CIRCLE_SIZE])

    def __calc_text_locs(self):
        add_spacer = 2
        self.loc_dict = {}
        for animal_cnt, animal_name in enumerate(self.multi_animal_id_list):
            self.loc_dict[animal_name] = {}
            self.loc_dict[animal_name]["degree_txt"] = f"{animal_name} angle:"
            self.loc_dict[animal_name]["cardinal_txt"] = f"{animal_name} cardinal:"
            self.loc_dict[animal_name]["degree_txt_loc"] = (5, (self.video_meta_data["height"] - (self.video_meta_data["height"] + 10) + self.spacing_scale * add_spacer))
            self.loc_dict[animal_name]["degree_data_loc"] = (200, (self.video_meta_data["height"] - (self.video_meta_data["height"] + 10) + self.spacing_scale * add_spacer))
            add_spacer += 1
            self.loc_dict[animal_name]["cardinal_txt_loc"] = ( 5, (self.video_meta_data["height"] - (self.video_meta_data["height"] + 10) + self.spacing_scale * add_spacer))
            self.loc_dict[animal_name]["cardinal_data_loc"] = (240, (self.video_meta_data["height"] - (self.video_meta_data["height"] + 10) + self.spacing_scale * add_spacer))
            add_spacer += 1

    def _create_palette(self):
        c = self.create_single_color_lst(pallete_name=self.settings["palette"], increments=360)
        self.colors = {}
        for color_cnt, color in enumerate(c):
            self.colors[color_cnt] = color

    def run(self):
        self.df = read_df(file_path=self.data_path, file_type=self.file_type)
        self.video_meta_data, frm_cnt = (get_video_meta_data(video_path=self.video_path), 0)
        if self.text_settings is False:
            self.__get_print_settings()
        self.cap = cv2.VideoCapture(self.video_path)
        self.writer = cv2.VideoWriter(self.save_path, self.fourcc, self.video_meta_data["fps"], (self.video_meta_data["width"], self.video_meta_data["height"]))
        self._approximate_size_of_animal_in_video()
        self._create_palette()
        self.__calc_text_locs()
        for animal_name, animal_bps in self.animal_bp_dict.items():
            self.df[f"Compass_cardinal_{animal_name}"] = CircularStatisticsMixin.degrees_to_cardinal(data=self.df[f"Fish_clockwise_angle_degrees"].values.astype(np.float32))
        while self.cap.isOpened():
            ret, self.frame = self.cap.read()
            if not ret:
                break
            frm_data = self.df.iloc[frm_cnt, :]
            for animal_name, animal_bps in self.animal_bp_dict.items():
                compass_center = tuple(frm_data[[f'{self.settings["center"][animal_name]}_x', f'{self.settings["center"][animal_name]}_y']].values.astype(int))
                animal_frm_angle, compass_cardinal = (int(frm_data[f"Fish_clockwise_angle_degrees"]), frm_data[f"Compass_cardinal_{animal_name}"])
                frm_compass_clr = self.colors[animal_frm_angle]
                cv2.circle(self.frame,compass_center,self.animal_sizes[animal_name]["diameter"],frm_compass_clr,self.text_thickness)

                cv2.putText(self.frame, self.loc_dict['Zebrafish']["degree_txt"], self.loc_dict['Zebrafish']["degree_txt_loc"], self.font, self.font_size, frm_compass_clr, 1)
                cv2.putText(self.frame,str(animal_frm_angle),self.loc_dict['Zebrafish']["degree_data_loc"],self.font,self.font_size,frm_compass_clr,1)
                cv2.putText(self.frame,self.loc_dict['Zebrafish']["cardinal_txt"],self.loc_dict['Zebrafish']["cardinal_txt_loc"],self.font,self.font_size,frm_compass_clr,1)
                cv2.putText(self.frame, compass_cardinal, self.loc_dict['Zebrafish']["cardinal_data_loc"], self.font, self.font_size, frm_compass_clr, 1)

            self.writer.write(self.frame)
            print(f'Image {frm_cnt+1}/{self.video_meta_data["frame_count"]} Video: {self.video_path}')
            frm_cnt += 1
        self.timer.stop_timer()
        self.writer.release()
        stdout_success(
            msg=f"Video {self.save_path} complete!",
            elapsed_time=self.timer.elapsed_time_str,
            source=self.__class__.__name__,
        )


# settings = {'center': {'Animal_1': 'Zebrafish_SwimBladder'},
#             'text_settings': False, "palette": 'bwr'}
#
# test = CircularFeaturePlotter(config_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/zebrafish/project_folder/project_config.ini',
#                               data_path='/Users/simon/Desktop/envs/simba/troubleshooting/zebrafish/project_folder/csv/features_extracted/test.csv',
#                               settings=settings)
# test.run()
