__author__ = "Simon Nilsson", "JJ Choong"

import cv2

from simba.read_config_unit_tests import (read_config_entry,
                                          read_config_file,
                                          check_file_exist_and_readable)
from collections import deque
from simba.misc_tools import check_multi_animal_status, create_single_color_lst, get_fn_ext
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info
from simba.drop_bp_cords import getBpNames, createColorListofList
from simba.rw_dfs import read_df
import numpy as np
import pandas as pd
import os, glob


class PathPlotter(object):
    """
    Class for creating "path plots" videos and/or images detailing the movement paths of
    individual animals in SimBA.

    Parameters
    ----------
    config_path: str
        Path to SimBA project config file in Configparser format
    frame_setting: bool
        If True, individual frames will be created.
    video_setting: bool
        If True, compressed videos will be created.

    Notes
    ----------
    `Visualization tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-11-visualizations>`__.

    Examples
    ----------
    >>> path_plotter = PathPlotter(config_path=r'MyConfigPath', frame_setting=False, video_setting=True)
    >>> path_plotter.create_path_plots()
    """


    def __init__(self,
                 config_path: str=None,
                 frame_setting: bool=None,
                 video_setting: bool=None):

        if (not frame_setting) and (not video_setting):
            print('SIMBA ERROR: Please choice to create frames and/or video path plots')
            raise ValueError
        self.config = read_config_file(config_path)
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.no_animals_path_plot = read_config_entry(self.config, 'Path plot settings', 'no_animal_pathplot', 'int')
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.save_folder = os.path.join(self.project_path, 'frames', 'output', 'path_plots')
        if not os.path.exists(self.save_folder): os.makedirs(self.save_folder)
        self.video_setting, self.frame_setting = video_setting, frame_setting
        self.in_dir = os.path.join(self.project_path, 'csv', 'machine_results')
        self.max_lines = read_config_entry(self.config, 'Path plot settings', 'deque_points', 'int', 100)
        self.multi_animal_status, self.multi_animal_id_lst = check_multi_animal_status(self.config, self.no_animals_path_plot)
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, 'logs', 'video_info.csv'))
        self.x_cols, self.y_cols, self.pcols = getBpNames(config_path)
        self.color_lst_of_lst = createColorListofList(self.no_animals_path_plot, int(len(self.x_cols) + 1))
        self.font = cv2.FONT_HERSHEY_COMPLEX
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        self.files_found = glob.glob(self.in_dir + "/*." + self.file_type)
        self.plot_bps = {}
        self.deque_dict = {}
        for animal_cnt, animal in enumerate(self.multi_animal_id_lst):
            bp_name = read_config_entry(self.config, 'Path plot settings', 'animal_{}_bp'.format(str(animal_cnt+1)), 'str')
            self.plot_bps[animal] = bp_name
            self.deque_dict[animal] = deque(maxlen=self.max_lines)
        print('Processing {} videos...'.format(str(len(self.files_found))))

    def create_path_plots(self):
        """
        Method to create path plot videos and/or frames.Results are store in the
        'project_folder/frames/path_plots' directory of the SimBA project.



        """

        for file_cnt, file_path in enumerate(self.files_found):
            _, self.video_name, _ = get_fn_ext(file_path)
            self.video_info, _, self.fps = read_video_info(self.vid_info_df, self.video_name)
            self.width, self.height = int(self.video_info['Resolution_width'].values[0]), int(self.video_info['Resolution_height'].values[0])
            self.data_df = read_df(file_path, self.file_type)
            self.space_scaler, self.radius_scaler, self.res_scaler, self.font_scaler = 25, 10, 1500, 0.8
            self.max_res = max(self.width, self.height)
            self.circle_scale = int(self.radius_scaler / (self.res_scaler / self.max_res))
            self.font_scale = int(self.font_scaler / (self.res_scaler / self.max_res))
            if self.video_setting:
                self.video_save_path = os.path.join(self.save_folder, self.video_name + '.mp4')
                self.writer = cv2.VideoWriter(self.video_save_path, self.fourcc, self.fps, (self.width, self.height))
            if self.frame_setting:
                self.save_video_folder = os.path.join(self.save_folder, self.video_name)
                if not os.path.exists(self.save_video_folder): os.makedirs(self.save_video_folder)

            for frm_cnt in range(len(self.data_df)):
                img = np.ones((self.height, self.width, 3)) * 255
                for animal, deque_lst in self.deque_dict.items():
                    bp_x, bp_y = self.plot_bps[animal] + '_x', self.plot_bps[animal] + '_y'
                    self.deque_dict[animal].appendleft(tuple(self.data_df.loc[frm_cnt, [bp_x, bp_y]].values))
                for animal_cnt, (animal, animal_deque) in enumerate(self.deque_dict.items()):
                    clr = self.color_lst_of_lst[animal_cnt][0]
                    prior_position = None
                    for position_cnt, position in enumerate(animal_deque):
                        if position_cnt == 0:
                            cv2.circle(img, (int(position[0]), int(position[1])), 0, clr, self.circle_scale)
                            cv2.putText(img, animal, (int(position[0]), int(position[1])), self.font, self.font_scale, clr, 2)
                        if position_cnt > 0:
                            cv2.line(img, (int(position[0]), int(position[1])), (int(prior_position[0]), int(prior_position[1])), clr, 2)
                        prior_position = position
                if self.video_setting:
                    self.writer.write(np.uint8(img))
                if self.frame_setting:
                    frm_name = os.path.join(self.save_video_folder, str(frm_cnt) + '.png')
                    cv2.imwrite(frm_name, np.uint8(img))
                print('Frame: {} / {}. Video: {} ({}/{})'.format(str(frm_cnt), str(len(self.data_df)),
                                                                 self.video_name, str(file_cnt + 1),
                                                                 len(self.files_found)))
            if self.video_setting:
                self.writer.release()
                print('Video {} saved...'.format(self.video_name))

            print('Path plot video {} complete...'.format(self.video_name))
        print('SIMBA COMPLETE: All path plots saved in the project_folder/frames/output/path_plots directory')

#
# test = PathPlotter(config_path=r'/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini',
#                    frame_setting=False,
#                    video_setting=True)
# test.create_path_plots()