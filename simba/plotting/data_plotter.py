__author__ = "Simon Nilsson"

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from simba.data_processors.timebins_movement_calculator import \
    TimeBinsMovementCalculator
from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_file_exist_and_readable, check_if_valid_rgb_tuple, check_int,
    check_str, check_valid_boolean, check_valid_lst, check_valid_tuple)
from simba.utils.enums import Formats
from simba.utils.errors import NoSpecifiedOutputError
from simba.utils.printing import SimbaTimer, stdout_success

VELOCITY = 'Velocity (cm/s)'
MEASUREMENT = 'MEASUREMENT'
VALUE = 'VALUE'
MOVEMENT = 'Movement (cm)'
TIME_BIN = 'TIME BIN #'

class DataPlotter(ConfigReader):
    """
    Tabular data visualization of animal movement and distances in the current frame and their aggregate
    statistics.

    :param Union[str, os.PathLike] config_path:  Path to the SimBA project config file in ConfigParser format.
    :param List[Tuple[str, Tuple[int, int, int]]] body_parts:   A list of tuples, where each tuple consists of a body-part name (str) and an RGB tuple (int, int, int) indicating the text color used for that body-part in plots.
    :param List[Union[str, os.PathLike]] data_paths: List of paths to the CSV files containing time-binned animal movement data.
    :param Tuple[int, int, int] bg_clr: Background color of the output image(s) as an RGB tuple. Default is white (255, 255, 255).
    :param Tuple[int, int, int] header_clr:  Text color for the header labels (e.g., "ANIMAL", "TOTAL MOVEMENT") as an RGB tuple. Default is black (0, 0, 0).
    :param int font_thickness:  Thickness of the font used in output images. Must be >= 1. Default is 2.
    :param Tuple[int, int] img_size: Size of the output image as a tuple (width, height). Default is (640, 480).
    :param int decimals: Number of decimal places to round movement and velocity values. Must be >= 0. Default is 2.
    :param bool video_setting: Whether to generate a video output of the data plot. At least one of `video_setting` or `frame_setting` must be True.
    :param bool frame_setting: Whether to generate individual frame image outputs for each time bin.
    :param bool verbose: Whether to print progress information during execution. Default is True.

    .. note::
        `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#visualizing-data-tables`>_.

    .. image:: _static/img/data_plot.png
       :width: 300
       :align: center

    :examples:
    >>> _ = DataPlotter(config_path='MyConfigPath').run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 body_parts: List[Tuple[str, Tuple[int, int, int]]],
                 data_paths: List[Union[str, os.PathLike]],
                 bg_clr: Tuple[int, int, int] = (255, 255, 255),
                 header_clr: Tuple[int, int, int] = (0, 0, 0),
                 font_thickness: int = 2,
                 img_size: Tuple[int, int] = (640, 480),
                 decimals: int = 2,
                 video_setting: bool = True,
                 frame_setting: bool = False,
                 verbose: bool = True):

        if (not video_setting) and (not frame_setting):
            raise NoSpecifiedOutputError(msg="SIMBA ERROR: Please choose to create video and/or frames data plots. SimBA found that you ticked neither video and/or frames", source=self.__class__.__name__)
        check_valid_lst(data=data_paths, source=self.__class__.__name__, valid_dtypes=(str,), min_len=1)
        for data_path in data_paths: check_file_exist_and_readable(file_path=data_path)
        check_valid_lst(data=body_parts, source=self.__class__.__name__, valid_dtypes=(tuple,))
        ConfigReader.__init__(self, config_path=config_path)
        for bp_cnt, bp_attr in enumerate(body_parts):
            check_str(name=f'{self.__class__.__name__} body-part {bp_cnt}', value=bp_attr[0], options=self.body_parts_lst)
            check_if_valid_rgb_tuple(data=bp_attr[1], raise_error=True, source=f'{self.__class__.__name__} body-part {bp_cnt}')
        self.bp_names, self.txt_clrs = [i[0] for i in body_parts], [i[1] for i in body_parts]
        check_if_valid_rgb_tuple(data=bg_clr, raise_error=True, source=f'{self.__class__.__name__} bg_clr')
        check_if_valid_rgb_tuple(data=header_clr, raise_error=True, source=f'{self.__class__.__name__} header_clr')
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose', raise_error=True)
        check_int(name=f'{self.__class__.__name__} font_thickness', value=font_thickness, min_value=1)
        check_valid_tuple(x=img_size, source=f'{self.__class__.__name__} img_size', accepted_lengths=(2,), valid_dtypes=(int,), min_integer=1)
        check_int(name=f'{self.__class__.__name__} decimals', value=decimals, min_value=0)
        self.video_setting, self.frame_setting, self.body_parts = video_setting, frame_setting, body_parts
        self.data_paths, self.bg_clr = data_paths, bg_clr
        self.font_thickness, self.img_size, self.decimals, self.header_clr, self.verbose = font_thickness, img_size, decimals, header_clr, verbose
        self.font_size, self.col_shift, self.row_shift = PlottingMixin().get_optimal_font_scales(text='TOTAL MOVEMENT (CM) + EAT', accepted_px_width=int(img_size[0]/3), accepted_px_height=int(img_size[1]/3), text_thickness=font_thickness)
        if not os.path.exists(self.data_table_path):
            os.makedirs(self.data_table_path)
        print(f"Processing {len(self.data_paths)} video(s)...")
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        self.timebins_calculator = TimeBinsMovementCalculator(config_path=self.config_path, bin_length=1, file_paths=data_paths, plots=False, verbose=True, body_parts=self.bp_names)
        self.timebins_calculator.run()
        self.__compute_spacings()


    def __compute_spacings(self):
        """
        Private helper to compute appropriate spacing between printed text.
        """
        self.loc_dict = {}
        self.loc_dict["Animal"] = (10, self.row_shift)
        self.loc_dict["total_movement_header"] = (10+self.col_shift, self.row_shift)
        self.loc_dict["current_velocity_header"] = (int(10+(self.col_shift*2)), self.row_shift)
        self.loc_dict["animals"] = {}
        for animal_cnt, animal_name in enumerate(self.body_parts):
            self.loc_dict["animals"][animal_name[0]] = {"index_loc":           (10, int(self.row_shift * (animal_cnt +2))),
                                                        'total_movement_loc':  (10+self.col_shift, int(self.row_shift * (animal_cnt +2))),
                                                        'current_velocity_loc': (int(10+(self.col_shift*2)), int(self.row_shift * (animal_cnt +2)))}

    def run(self):
        def multiprocess_img_creation(data: list,
                                      location_dict: dict,
                                      video_data: pd.DataFrame,
                                      bg_clr: Tuple[int, int, int],
                                      text_clrs: List[Tuple[int, int, int]],
                                      font_size: int,
                                      header_clr: Tuple[int, int, int],
                                      font_thickness: int,
                                      img_size: tuple,
                                      decimals: int):

            time_bin_data = video_data[video_data['TIME BIN #'] == data[0]]
            img = np.zeros((img_size[1], img_size[0], 3))
            img[:] = bg_clr


            img = PlottingMixin().put_text(img=img, text='ANIMAL', pos=location_dict["Animal"], font_size=self.font_size, font_thickness=font_thickness, text_color=header_clr, text_bg_alpha=1, text_color_bg=bg_clr, font=cv2.FONT_HERSHEY_TRIPLEX)
            img = PlottingMixin().put_text(img=img, text='TOTAL MOVEMENT (CM)', pos=location_dict["total_movement_header"], font_size=self.font_size, font_thickness=font_thickness, text_color=header_clr, text_bg_alpha=1, text_color_bg=bg_clr, font=cv2.FONT_HERSHEY_TRIPLEX)
            img = PlottingMixin().put_text(img=img, text="VELOCITY (CM/S)", pos=location_dict["current_velocity_header"], font_size=font_size, font_thickness=font_thickness, text_color=header_clr, text_bg_alpha=1, text_color_bg=bg_clr, font=cv2.FONT_HERSHEY_TRIPLEX)

            for bp_cnt, bp_name in enumerate(time_bin_data['BODY-PART'].unique()):
                velocity_data = time_bin_data[time_bin_data['BODY-PART'] == bp_name]
                bp_data = video_data[video_data['BODY-PART'] == bp_name]
                velocity_data = round(velocity_data[velocity_data[MEASUREMENT] == VELOCITY]['VALUE'].values[0], decimals)
                movement_data = bp_data[bp_data[MEASUREMENT] == MOVEMENT]
                movement_data = round(movement_data[movement_data[TIME_BIN] <= data[0]][VALUE].sum(), decimals)
                img = PlottingMixin().put_text(img=img, text=bp_name, pos=location_dict["animals"][bp_name]["index_loc"], font_size=font_size, font_thickness=font_thickness, text_color=text_clrs[bp_cnt], text_bg_alpha=1, text_color_bg=bg_clr, font=cv2.FONT_HERSHEY_TRIPLEX)
                img = PlottingMixin().put_text(img=img, text=str(movement_data), pos=location_dict["animals"][bp_name]["total_movement_loc"], font_size=font_size, font_thickness=font_thickness, text_color=text_clrs[bp_cnt], text_bg_alpha=1,font=cv2.FONT_HERSHEY_TRIPLEX, text_color_bg=bg_clr)
                img = PlottingMixin().put_text(img=img, text=str(velocity_data), pos=location_dict["animals"][bp_name]["current_velocity_loc"], font_size=font_size, font_thickness=font_thickness, text_color=text_clrs[bp_cnt], text_bg_alpha=1,font=cv2.FONT_HERSHEY_TRIPLEX, text_color_bg=bg_clr)

            return img


        for video_cnt, video_data in enumerate(self.timebins_calculator.out_df_lst):
            video_timer = SimbaTimer(start=True)
            video_name = video_data['VIDEO'].loc[0]
            _, _, self.fps = self.read_video_info(video_name=video_name)
            if self.video_setting:
                self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
                self.video_save_path = os.path.join(self.data_table_path, f"{video_name}.mp4")
                self.writer = cv2.VideoWriter(self.video_save_path, self.fourcc, self.fps, self.img_size)
            if self.frame_setting:
                self.frame_save_path = os.path.join(self.data_table_path, video_name)
                if not os.path.exists(self.frame_save_path): os.makedirs(self.frame_save_path)


            time_bins = [[x] for x in list(set(video_data['TIME BIN #']))]
            print('Creating images...')
            self.imgs = Parallel(n_jobs=self.cpu_to_use, verbose=1, backend="threading")(delayed(multiprocess_img_creation)(x,
                                                                                                                            self.loc_dict,
                                                                                                                            video_data,
                                                                                                                            self.bg_clr,
                                                                                                                            self.txt_clrs,
                                                                                                                            self.font_size,
                                                                                                                            self.header_clr,
                                                                                                                            self.font_thickness,
                                                                                                                            self.img_size,
                                                                                                                            self.decimals) for x in time_bins)
            frm_cnt = 0
            for img_cnt, img in enumerate(self.imgs):
                for frame_cnt in range(int(self.fps)):
                    if self.video_setting:
                        self.writer.write(np.uint8(img))
                    if self.frame_setting:
                        frm_save_name = os.path.join(self.frame_save_path, f"{frame_cnt}.png")
                        cv2.imwrite(frm_save_name, np.uint8(img))
                frm_cnt += 1
                if self.verbose:
                    print(f"Frame: {int((img_cnt+1) * self.fps)} / {int(len(self.imgs) * self.fps)}. Video: {video_name} ({video_cnt+1}/{len(self.timebins_calculator.out_df_lst)})")

            print(f"Data tables created for video {video_name}...")
            if self.video_setting:
                self.writer.release()
                video_timer.stop_timer()
                if self.verbose:
                    stdout_success(msg=f'Video {self.video_save_path} complete', source=self.__class__.__name__, elapsed_time=video_timer.elapsed_time_str)

        self.timer.stop_timer()
        if self.verbose:
            stdout_success(msg=f"All data table videos created inside {self.data_table_path}", elapsed_time=self.timer.elapsed_time_str)

#
#
# # style_attr = {'bg_color': 'White', 'header_color': 'Black', 'font_thickness': 1, 'size': (640, 480), 'data_accuracy': 2}
# body_part_attr = [('Tail_base', (0, 255, 0)), ('Nose', (255, 255, 0))]
# data_paths = [r"C:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location\501_MA142_Gi_Saline_0513.csv"]
# #
# #
# test = DataPlotter(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini",
#                    body_parts=body_part_attr,
#                    data_paths=data_paths,
#                    video_setting=True,
#                    frame_setting=False)
# test.run()

# style_attr = {'bg_color': 'White', 'header_color': 'Black', 'font_thickness': 1, 'size': (640, 480), 'data_accuracy': 2}
# body_part_attr = [['Ear_left_1', 'Grey'], ['Ear_right_2', 'Red']]
# data_paths = ['/Users/simon/Desktop/envs/simba_dev/tests/test_data/two_C57_madlc/project_folder/csv/outlier_corrected_movement_location/Together_1.csv']
#
#
# test = DataPlotter(config_path='/Users/simon/Desktop/envs/simba_dev/tests/test_data/two_C57_madlc/project_folder/project_config.ini',
#                    style_attr=style_attr,
#                    body_part_attr=body_part_attr,
#                    data_paths=data_paths,
#                    video_setting=True,
#                    frame_setting=False)
# test.create_data_plots()
