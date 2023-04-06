__author__ = "Simon Nilsson"

import pandas as pd
from joblib import Parallel, delayed
import os
from simba.feature_extractors.unit_tests import read_video_info
from simba.movement_processor import MovementProcessor
from simba.misc_tools import (check_multi_animal_status,
                              get_fn_ext,
                              SimbaTimer,
                              get_color_dict)
import numpy as np
import cv2
from simba.enums import Formats
from simba.mixins.config_reader import ConfigReader
from simba.utils.errors import NoSpecifiedOutputError


class DataPlotter(ConfigReader):
    """
    Class for tabular data visualizations of animal movement and distances in the current frame and their aggregate
    statistics.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format

    Notes
    ----------
    `GitHub tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#visualizing-data-tables`__.

    Examples
    -----
    >>> data_plotter = DataPlotter(config_path='MyConfigPath')
    >>> data_plotter.process_movement()
    >>> data_plotter.create_data_plots()
    """

    def __init__(self,
                 config_path: str,
                 style_attr: dict,
                 body_part_attr: list,
                 data_paths: list,
                 video_setting: bool,
                 frame_setting: bool,
                 ):

        super().__init__(config_path=config_path)
        self.video_setting, self.frame_setting = video_setting, frame_setting
        if (not self.video_setting) and (not self.frame_setting):
            raise NoSpecifiedOutputError(msg='SIMBA ERROR: Please choose to create video and/or frames data plots. SimBA found that you ticked neither video and/or frames')
        self.files_found, self.style_attr, self.body_part_attr = data_paths, style_attr, body_part_attr
        if not os.path.exists(self.data_table_path):
            os.makedirs(self.data_table_path)
        self.multi_animal_status, self.multi_animal_id_list = check_multi_animal_status(self.config, len(self.body_part_attr))
        self.__compute_spacings()
        self.process_movement()
        print('Processing {} video(s)...'.format(str(len(self.files_found))))

    def __compute_spacings(self):
        """
        Private helper to compute appropriate spacing between printed text.
        """
        self.loc_dict = {}
        self.loc_dict['Animal'] = (50, 20)
        self.loc_dict['total_movement_header'] = (250, 20)
        self.loc_dict['current_velocity_header'] = (475, 20)
        self.loc_dict['animals'] = {}
        y_cord, x_cord = 75, 15
        for animal_cnt, animal_name in enumerate(self.multi_animal_id_list):
            self.loc_dict['animals'][animal_name] = {}
            self.loc_dict['animals'][animal_name]['index_loc'] = (50, y_cord)
            self.loc_dict['animals'][animal_name]['total_movement_loc'] = (250, y_cord)
            self.loc_dict['animals'][animal_name]['current_velocity_loc'] = (475, y_cord)
            y_cord += 50

    def process_movement(self):
        """
        Method to create movement data for visualization

        Returns
        -------
        Attribute: pd.Dataframe
            movement
        """
        self.config.set('process movements', 'no_of_animals', str(len(self.body_part_attr)))
        self.config.set('process movements', 'probability_threshold', str(0.00))
        for animal_cnt, animal in enumerate(self.body_part_attr):
            self.config.set('process movements', 'animal_{}_bp'.format(animal_cnt + 1), animal[0])
        with open(self.config_path, 'w') as file:
            self.config.write(file)
        movement_processor = MovementProcessor(config_path=self.config_path, visualization=True, files=self.files_found)
        movement_processor.process_movement()
        self.movement = movement_processor.movement_dict

    def create_data_plots(self):
        """
        Method to create and save visualizations on disk from data created in
        :meth:`~simba.DataPlotter.process_movement`. Results are stored in the `project_folder/frames/output/live_data_table`.

        Returns
        -------
        None
        """

        def multiprocess_img_creation(video_data_slice: list,
                                      location_dict: dict,
                                      animal_ids: list,
                                      video_data: pd.DataFrame,
                                      style_attr: dict,
                                      body_part_attr: dict):

            color_dict = get_color_dict()
            img = np.zeros((style_attr['size'][1], style_attr['size'][0], 3))
            img[:] = color_dict[style_attr['bg_color']]
            cv2.putText(img, 'Animal', location_dict['Animal'], cv2.FONT_HERSHEY_TRIPLEX, 0.5, color_dict[style_attr['header_color']], style_attr['font_thickness'])
            cv2.putText(img, 'Total movement (cm)', location_dict['total_movement_header'], cv2.FONT_HERSHEY_TRIPLEX, 0.5, color_dict[style_attr['header_color']], style_attr['font_thickness'])
            cv2.putText(img, 'Velocity (cm/s)',location_dict['current_velocity_header'], cv2.FONT_HERSHEY_TRIPLEX, 0.5, color_dict[style_attr['header_color']], style_attr['font_thickness'])
            for animal_cnt, animal_name in enumerate(animal_ids):
                clr = color_dict[body_part_attr[animal_cnt][1]]
                total_movement = str(round(video_data[animal_name].iloc[0:video_data_slice.index.max()].sum() / 10, style_attr['data_accuracy']))
                current_velocity = str(round(video_data_slice[animal_name].sum() / 10, style_attr['data_accuracy']))
                cv2.putText(img, animal_name, location_dict['animals'][animal_name]['index_loc'], cv2.FONT_HERSHEY_TRIPLEX, 0.5, clr, 1)
                cv2.putText(img, total_movement, location_dict['animals'][animal_name]['total_movement_loc'], cv2.FONT_HERSHEY_TRIPLEX, 0.5, clr, 1)
                cv2.putText(img, current_velocity, location_dict['animals'][animal_name]['current_velocity_loc'], cv2.FONT_HERSHEY_TRIPLEX, 0.5, clr, 1)
            return img

        for file_cnt, file_path in enumerate(self.files_found):
            video_timer = SimbaTimer()
            video_timer.start_timer()
            _, video_name, _ = get_fn_ext(file_path)
            video_data = pd.DataFrame(self.movement[video_name])
            _, _, self.fps = read_video_info(vid_info_df=self.video_info_df, video_name=video_name)
            if self.video_setting:
                self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
                self.video_save_path = os.path.join(self.data_table_path, video_name + '.mp4')
                self.writer = cv2.VideoWriter(self.video_save_path, self.fourcc, self.fps, self.style_attr['size'])
            if self.frame_setting:
                self.frame_save_path = os.path.join(self.data_table_path, video_name)
                if not os.path.exists(self.frame_save_path): os.makedirs(self.frame_save_path)
            video_data_lst = np.array_split(pd.DataFrame(video_data), int(len(video_data) / self.fps))
            self.imgs = Parallel(n_jobs=self.cpu_to_use, verbose=1, backend="threading")(delayed(multiprocess_img_creation)(x, self.loc_dict, self.multi_animal_id_list, video_data, self.style_attr, self.body_part_attr) for x in video_data_lst)
            frm_cnt = 0
            for img_cnt, img in enumerate(self.imgs):
                for frame_cnt in range(int(self.fps)):
                    if self.video_setting:
                        self.writer.write(np.uint8(img))
                    if self.frame_setting:
                        frm_save_name = os.path.join(self.frame_save_path, '{}.png'.format(str(frm_cnt)))
                        cv2.imwrite(frm_save_name, np.uint8(img))
                    frm_cnt += 1
                    print('Frame: {} / {}. Video: {} ({}/{})'.format(str(frm_cnt), str(len(video_data)),
                                                                   video_name, str(file_cnt + 1),
                                                                   len(self.files_found)))

            print('Data tables created for video {}...'.format(video_name))
            if self.video_setting:
                self.writer.release()
                video_timer.stop_timer()
                print('Video {} complete (elapsed time {}s)...'.format(video_name, video_timer.elapsed_time_str))

        self.timer.stop_timer()
        print('SIMBA COMPLETE: All data table videos created inside {} (elapsed time: {}s)'.format(self.data_table_path, self.timer.elapsed_time_str))

# style_attr = {'bg_color': 'White', 'header_color': 'Black', 'font_thickness': 1, 'size': (640, 480), 'data_accuracy': 2}
# body_part_attr = [['Ear_left_1', 'Grey'], ['Ear_right_2', 'Red']]
# data_paths = ['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv']
#
#
# test = DataPlotter(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                    style_attr=style_attr,
#                    body_part_attr=body_part_attr,
#                    data_paths=data_paths,
#                    video_setting=True,
#                    frame_setting=False)
# test.create_data_plots()

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

