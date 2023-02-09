__author__ = "Simon Nilsson", "JJ Choong"

import cv2
import pandas as pd

from simba.read_config_unit_tests import (read_config_entry,
                                          read_config_file,
                                          check_if_filepath_list_is_empty,
                                          read_project_path_and_file_type)
from copy import deepcopy
from collections import deque
from simba.misc_tools import (check_multi_animal_status,
                              get_fn_ext,
                              SimbaTimer,
                              get_color_dict,
                              remove_a_folder,
                              concatenate_videos_in_folder)
from simba.enums import (Paths, ReadConfig, Formats, Defaults)
from numba import jit, prange
from simba.misc_visualizations import make_path_plot
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info
from simba.drop_bp_cords import getBpNames
from simba.rw_dfs import read_df
import numpy as np
import os
import functools
import multiprocessing
import platform

def _image_creator(data: np.array,
                   video_setting: bool,
                   frame_setting: bool,
                   video_save_dir: str,
                   video_name: str,
                   frame_folder_dir: str,
                   style_attr: dict,
                   animal_attr: dict,
                   fps: int,
                   video_info: pd.DataFrame,
                   clf_attr: dict):

    group = int(data[0][0][0])
    color_dict = get_color_dict()
    if video_setting:
        fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        video_save_path = os.path.join(video_save_dir, '{}.mp4'.format(str(group)))
        video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, (style_attr['width'], style_attr['height']))

    for i in range(data.shape[0]):
        frame_id = int(data[i, -1, 1] + 1)
        frame_data = data[i, :, 2:].astype(int)
        frame_data = np.split(frame_data, len(list(animal_attr.keys())), axis=1)

        img = np.zeros((int(video_info['Resolution_height'].values[0]), int(video_info['Resolution_width'].values[0]), 3))
        img[:] = style_attr['bg color']
        for animal_cnt, animal_data in enumerate(frame_data):
            animal_clr = style_attr['animal clrs'][animal_cnt]
            for j in range(animal_data.shape[0] - 1):
                cv2.line(img, tuple(animal_data[j]), tuple(animal_data[j+1]), animal_clr, style_attr['line width'])
            cv2.circle(img, tuple(animal_data[-1]), 0, animal_clr, style_attr['circle size'])
            cv2.putText(img, style_attr['animal names'][animal_cnt], tuple(animal_data[-1]), cv2.FONT_HERSHEY_COMPLEX, style_attr['font size'], animal_clr, style_attr['font thickness'])

        if clf_attr:
            for clf_cnt, clf_name in enumerate(clf_attr['data'].columns):
                clf_size = int(clf_attr[clf_cnt][-1].split(': ')[-1])
                clf_clr = color_dict[clf_attr[clf_cnt][-2]]
                clf_sliced = clf_attr['data'][clf_name].loc[0: frame_id]
                clf_sliced_idx = list(clf_sliced[clf_sliced == 1].index)
                locations = clf_attr['positions'][clf_sliced_idx, :]
                for i in range(locations.shape[0]):
                    cv2.circle(img, (locations[i][0], locations[i][1]), 0, clf_clr, clf_size)

        img = cv2.resize(img, (style_attr['width'], style_attr['height']))
        if video_setting:
            video_writer.write(np.uint8(img))
        if frame_setting:
            frm_name = os.path.join(frame_folder_dir, str(frame_id) + '.png')
            cv2.imwrite(frm_name, np.uint8(img))

        print('Path frame created: {}, Video: {}, Processing core: {}'.format(str(frame_id + 1), video_name, str(group + 1)))

    return group

class PathPlotterMulticore(object):
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
    last_frame: bool
        If True, creates a .png representing the final image of the path plot.
    files_found: list
        Data paths to create from which to create plots
    animal_attr: dict
        Animal body-parts and colors
    style_attr: dict
        Plot sttributes

    Notes
    ----------
    `Visualization tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-11-visualizations>`__.

    Examples
    ----------
    >>> style_attr = {'width': 'As input', 'height': 'As input', 'line width': 5, 'font size': 5, 'font thickness': 2, 'circle size': 5, 'bg color': 'White', 'max lines': 100}
    >>> animal_attr = {0: ['Ear_right_1', 'Red']}
    >>> clf_attr = {0: ['Attack', 'Black', 'Size: 30'], 1: ['Sniffing', 'Red', 'Size: 30']}
    >>> path_plotter = PathPlotterMulticore(config_path=r'MyConfigPath', frame_setting=False, video_setting=True, style_attr=style_attr, animal_attr=animal_attr, files_found=['project_folder/csv/machine_results/MyVideo.csv'], cores=5, clf_attr=clf_attr)
    >>> path_plotter.create_path_plots()
    """

    def __init__(self,
                 config_path: str,
                 frame_setting: bool,
                 video_setting: bool,
                 last_frame: bool,
                 files_found: list,
                 style_attr: dict or None,
                 animal_attr: dict,
                 clf_attr: dict,
                 cores: int):


        if platform.system() == "Darwin":
            multiprocessing.set_start_method('spawn', force=True)

        self.video_setting, self.frame_setting, self.style_attr, self.files_found, self.animal_attr, self.clf_attr, self.last_frame, self.cores = video_setting, frame_setting, style_attr, files_found, animal_attr, clf_attr, last_frame, cores
        if (not frame_setting) and (not video_setting) and (not last_frame):
            print('SIMBA ERROR: Please choice to create path frames and/or video path plots')
            raise ValueError('SIMBA ERROR: Please choice to create path frames and/or video path plots')
        self.timer = SimbaTimer()
        self.timer.start_timer()
        self.config = read_config_file(config_path)
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.no_animals_path_plot = len(animal_attr.keys())
        self.save_folder = os.path.join(self.project_path, Paths.PATH_PLOT_DIR.value)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        self.multi_animal_status, self.multi_animal_id_lst = check_multi_animal_status(self.config, self.no_animals_path_plot)
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, Paths.VIDEO_INFO.value))
        self.x_cols, self.y_cols, self.pcols = getBpNames(config_path)
        self.maxtasksperchild = Defaults.MAX_TASK_PER_CHILD.value
        self.chunksize = Defaults.CHUNK_SIZE.value
        check_if_filepath_list_is_empty(filepaths=self.files_found,
                                        error_msg='SIMBA ERROR: Zero files found in the project_folder/csv/machine_results directory. To plot paths without performing machine classifications, use path plotter functions in [ROI] tab.')
        print('Processing {} videos...'.format(str(len(self.files_found))))

    def create_path_plots(self):
        """
        Method to create path plot videos and/or frames.Results are store in the
        'project_folder/frames/path_plots' directory of the SimBA project.
        """

        for file_cnt, file_path in enumerate(self.files_found):
            video_timer = SimbaTimer()
            video_timer.start_timer()
            _, self.video_name, _ = get_fn_ext(file_path)
            self.video_info, _, self.fps = read_video_info(self.vid_info_df, self.video_name)
            self.__get_styles()
            self.data_df = read_df(file_path, self.file_type)
            self.temp_folder = os.path.join(self.save_folder, self.video_name, 'temp')
            self.save_frame_folder_dir = os.path.join(self.save_folder, self.video_name)
            if self.frame_setting:
                if os.path.exists(self.save_frame_folder_dir): remove_a_folder(self.save_frame_folder_dir)
                if not os.path.exists(self.save_frame_folder_dir): os.makedirs(self.save_frame_folder_dir)
            if self.video_setting:
                self.video_folder = os.path.join(self.save_folder, self.video_name)
                if os.path.exists(self.temp_folder):
                    remove_a_folder(self.temp_folder)
                    remove_a_folder(self.video_folder)
                os.makedirs(self.temp_folder)
                self.save_video_path = os.path.join(self.save_folder, self.video_name + '.mp4')

            if self.clf_attr:
                clf_names = []
                for v in self.clf_attr.values():
                    clf_names.append(v[0])

            if self.last_frame:
                self.__get_deque_lookups()
                _ = make_path_plot(data_df=self.data_df,
                                   video_info=self.video_info,
                                   style_attr=self.style_attr,
                                   deque_dict=self.deque_dict,
                                   clf_attr=self.clf_attr,
                                   save_path=os.path.join(self.save_folder, self.video_name + '_final_frame.png'))

            if self.video_setting or self.frame_setting:

                if self.clf_attr:
                    self.clf_attr['data'] = self.data_df[clf_names]

                data_arr = np.array(list(self.data_df.index)).reshape(-1, 1)
                for animal_cnt, animal_data in self.animal_attr.items():
                    bp_x_name = '{}_{}'.format(animal_data[0], 'x')
                    bp_y_name = '{}_{}'.format(animal_data[0], 'y')
                    data_arr = np.hstack((data_arr, self.data_df[[bp_x_name, bp_y_name]].astype(int).values))
                    if animal_cnt == 0:
                        self.clf_attr['positions'] = deepcopy(data_arr[:, 1:3])
                data_arr = self.__split_array_into_max_lines(data=data_arr, max_lines=self.style_attr['max lines'])
                data_arr = np.array_split(data_arr, self.cores)
                data = []
                for cnt, i in enumerate(data_arr):
                    data.append(self.__insert_group_idx_column(data=i, group=cnt))
                frm_per_core = data[0].shape[0]

                print('Creating gantt, multiprocessing (chunksize: {}, cores: {})...'.format(str(self.chunksize), str(self.cores)))
                with multiprocessing.Pool(self.cores, maxtasksperchild=self.maxtasksperchild) as pool:
                    constants = functools.partial(_image_creator,
                                                  video_setting=self.video_setting,
                                                  video_name=self.video_name,
                                                  frame_setting=self.frame_setting,
                                                  video_save_dir=self.temp_folder,
                                                  frame_folder_dir=self.save_frame_folder_dir,
                                                  style_attr=self.style_attr,
                                                  fps=self.fps,
                                                  animal_attr=self.animal_attr,
                                                  video_info=self.video_info,
                                                  clf_attr=self.clf_attr)
                    for cnt, result in enumerate(pool.imap(constants, data, chunksize=self.chunksize)):
                        print('Image {}/{}, Video {}/{}...'.format(str(int(frm_per_core * (result + 1))), str(len(self.data_df)), str(file_cnt + 1), str(len(self.files_found))))

                    pool.terminate()
                    pool.join()

                if self.video_setting:
                    print('Joining {} multiprocessed video...'.format(self.video_name))
                    concatenate_videos_in_folder(in_folder=self.temp_folder, save_path=self.save_video_path)

                video_timer.stop_timer()
                print('Path plot video {} complete (elapsed time: {}s) ...'.format(self.video_name,video_timer.elapsed_time_str))

        self.timer.stop_timer()
        print('SIMBA COMPLETE: Path plot visualizations for {} videos created in project_folder/frames/output/path_plots directory (elapsed time: {}s)'.format(str(len(self.files_found)), self.timer.elapsed_time_str))

    @staticmethod
    @jit(nopython=True)
    def __split_array_into_max_lines(data: np.array,
                                     max_lines: int):

        results = np.full((data.shape[0], max_lines, data.shape[1]), np.nan)
        for i in prange(data.shape[0]):
            start = i - max_lines
            if (i - max_lines) < 0:
                start = 0
            frm_data = data[start:i, :]
            missing_cnt = max_lines - frm_data.shape[0]
            if missing_cnt > 0:
                frm_data = np.vstack((np.full((missing_cnt, frm_data.shape[1]), -1), frm_data))
            results[i] = frm_data

        return results


    @staticmethod
    @jit(nopython=True)
    def __insert_group_idx_column(data: np.array,
                                  group: int):

        results = np.full((data.shape[0], data.shape[1], data.shape[2] + 1), np.nan)
        group_col = np.full((data.shape[1], 1), group)
        for frm_idx in prange(data.shape[0]):
            results[frm_idx] = np.hstack((group_col, data[frm_idx]))
        return results

    def __get_styles(self):
        self.color_dict = get_color_dict()
        if self.style_attr is not None:
            self.style_attr['bg color'] = self.color_dict[self.style_attr['bg color']]
            self.style_attr['max lines'] = int(self.style_attr['max lines'] * (int(self.video_info['fps'].values[0]) / 1000))
            if self.style_attr['width'] == 'As input':
                self.style_attr['width'], self.style_attr['height'] = int(self.video_info['Resolution_width'].values[0]), int(self.video_info['Resolution_height'].values[0])
            else:
                pass
        else:
            self.style_attr = {}
            space_scaler, radius_scaler, res_scaler, font_scaler = 25, 10, 1500, 0.8
            self.style_attr['width'] = int(self.video_info['Resolution_width'].values[0])
            self.style_attr['height'] = int(self.video_info['Resolution_height'].values[0])
            max_res = max(self.style_attr['width'], self.style_attr['height'])
            self.style_attr['circle size'] = int(radius_scaler / (res_scaler / max_res))
            self.style_attr['font size'] = int(font_scaler / (res_scaler / max_res))
            self.style_attr['bg color'] = self.color_dict['White']
            self.style_attr['max lines'] = int(self.video_info['fps'].values[0] * 2)
            self.style_attr['font thickness'] = 2
            self.style_attr['line width'] = 2

        self.style_attr['animal names'] = self.multi_animal_id_lst
        self.style_attr['animal clrs'] = []
        for animal_cnt, animal in enumerate(self.multi_animal_id_lst):
            self.style_attr['animal clrs'].append(self.color_dict[self.animal_attr[animal_cnt][1]])

    def __get_deque_lookups(self):
        self.deque_dict = {}
        for animal_cnt, animal in enumerate(self.multi_animal_id_lst):
            self.deque_dict[animal] = {}
            self.deque_dict[animal]['deque'] = deque(maxlen=self.style_attr['max lines'])
            self.deque_dict[animal]['bp'] = self.animal_attr[animal_cnt][0]
            self.deque_dict[animal]['clr'] = self.color_dict[self.animal_attr[animal_cnt][1]]



# style_attr = {'width': 'As input', 'height': 'As input', 'line width': 5, 'font size': 5, 'font thickness': 2, 'circle size': 5, 'bg color': 'Yellow', 'max lines': 100}
# animal_attr = {0: ['Ear_right_1', 'Red'], 1: ['Ear_right_2', 'Yellow']}
# clf_attr = {0: ['Attack', 'Black', 'Size: 30'], 1: ['Sniffing', 'Red', 'Size: 30']}
#
# style_attr = None
#
# path_plotter = PathPlotterMulticore(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                     frame_setting=False,
#                                     video_setting=False,
#                                     last_frame=True,
#                                     clf_attr=clf_attr,
#                                     style_attr=style_attr,
#                                     animal_attr=animal_attr,
#                                     files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'],
#                                     cores=5)
#
# path_plotter.create_path_plots()